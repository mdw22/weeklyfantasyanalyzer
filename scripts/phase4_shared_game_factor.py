"""
phase4_shared_game_factor.py

Phase 4's first cut, per the designer's explicit decision after Phase 3's
recentering-alone result showed no meaningful change (2026-10-04): "start
Phase 4: shared game-level factors first (per MODEL_ROADMAP.md's own Phase
4 scoping -- keep it simple, not a full covariance matrix), still offline,
validated against this exact same synthetic-matchup calibration framework."
OFFLINE/BACKTEST ONLY -- does not touch monteCarlo.js or any production
file.

Phase 3 deliberately isolated and ruled out "the center was wrong" as part
of the overconfidence story (see PHASE3_CALIBRATION_REPORT.md -- recentering
alone left the 90-100% bucket's ~11-point gap basically unchanged on both
RNG seeds). That leaves spread/cross-player independence as the remaining
candidates -- src/lib/monteCarlo.js samples every player's outcome fully
independently (documented, approved-as-a-known-v2-simplification), which
means a QB having a monster game and his WR1 having a monster game in the
SAME simulated trial are currently uncorrelated, even though in reality
they're driven by a shared context (that team's offense had a big day). This
is the exact mechanism CLAUDE.md's "v3+ ideas" and Step 4's own writeup both
already pointed to.

Design -- the simplest real correlation structure, not a full covariance
matrix. First attempt (kept here only in this comment as a record of the
iteration, not what shipped): correlate players by their own TEAM only.
Ran it; the measured overconfidence gap barely moved -- made sense on
reflection, since within one randomly-drawn 9-player synthetic roster, two
players sharing the same real team is the exception, not the rule, so a
same-team-only factor rarely has much to grab onto. "Shared GAME-level
factor" (the roadmap's own phrase) means both competing real teams in a
given NFL game, not just one team in isolation -- a shootout boosts BOTH
offenses and hurts BOTH defenses; a defensive struggle is the mirror image.
That's the version actually implemented: within each Monte Carlo trial,
every player (QB/RB/WR/TE/K/DEF) whose real team is playing in the SAME
real NFL game that week shares ONE random "game quantile" for that trial
(keyed by `(season, week, frozenset({team, opponent}))`, reusing the
team->opponent mapping already built for the Candidate 2 Vegas-line work --
zero new data). Offensive players sample their own recentered bootstrap
distribution AT that shared quantile (sorted ascending, indexed by
`floor(quantile * n)`); DEF samples at `1 - quantile` (inverted -- a big
shootout quantile is bad for both teams' defenses, a low one is good for
both). Two players on the SAME real team always share a game (the common
case the first attempt covered); two players on OPPOSING real teams in the
same real game now ALSO correctly share one, which is the piece that was
missing. Players whose real game isn't resolvable (shouldn't happen for an
already-played week, but don't assume) fall back to fully independent
sampling, same as before.

Builds directly on Phase 3's recentered distribution (same
`raw - raw.mean() + baseline_c_point` shift) -- this cut is additive on top
of that one, not a replacement, since Phase 3's recentering is still a
reasonable thing to do even though it alone didn't close the gap. Reuses
calibration_check.py's build_pool_by_position()/draw_synthetic_matchup()/
FIXED_SLOTS/FLEX_ELIGIBLE/PROB_BUCKETS and phase3_calibration_check.py's
recentered_distribution()/baseline-points-building, unchanged. Runs the
same two-seed (20261004, 777) robustness check.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_shared_game_factor.py
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import (  # noqa: E402
    assert_no_leakage,
    collect_actuals,
    compute_points,
    list_backtest_weeks,
    run_no_leakage_tests,
)
from calibration_check import (  # noqa: E402
    FIXED_SLOTS,
    FLEX_ELIGIBLE,
    PROB_BUCKETS,
    build_pool_by_position,
    draw_synthetic_matchup,
)
from generate_projections import (  # noqa: E402
    DEF_STAT_COLUMNS,
    K_STAT_COLUMNS,
    LOOKBACK_SEASONS,
    STAT_COLUMNS,
    add_def_stat_columns,
    add_k_stat_columns,
    build_history,
    build_projections,
    current_season,
    describe_defense,
    describe_player,
    get_current_season_and_week,
)
from phase2_game_environment import baseline_c_points, build_environment_factors  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402
from phase3_calibration_check import recentered_distribution  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_CORRELATION_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]


def simulate_matchup_correlated(my_entries, opp_entries, baseline_points, history, dist_cache, trials, rng) -> float:
    """entries are (entity_id, position, game_key) tuples -- game_key
    identifies the real NFL game (both competing teams), not just one team,
    so opposing real teams in the same real game correctly share one factor."""
    all_entries = my_entries + opp_entries
    games = {key for (_, _, key) in all_entries if key is not None}
    game_quantile = {key: rng.uniform(0.0, 1.0, size=trials) for key in games}

    def team_totals(entries):
        total = np.zeros(trials)
        for entity_id, pos, key in entries:
            dist = recentered_distribution(entity_id, baseline_points, history, dist_cache)
            if key in game_quantile:
                quantile = (1 - game_quantile[key]) if pos == "DEF" else game_quantile[key]
                sorted_dist = np.sort(dist)
                idx = np.clip((quantile * len(sorted_dist)).astype(int), 0, len(sorted_dist) - 1)
                sampled = sorted_dist[idx]
            else:
                sampled = rng.choice(dist, size=trials, replace=True)
            total += sampled
        return total

    my_totals = team_totals(my_entries)
    opp_totals = team_totals(opp_entries)
    return float(np.mean(my_totals > opp_totals))


def game_key(env_opponents, season, week, team):
    opp = env_opponents.get((season, week, team))
    return (season, week, frozenset({team, opp})) if opp else None


def run_backtest(targets, non_kickers, kickers, team_stats, opp, env_opponents, seed):
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    records, skipped_weeks, ties = [], [], 0

    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]

    for season, week in targets:
        actuals, projections, history = {}, {}, {}
        for df, id_col, stat_columns, describe_row in entity_groups:
            week_actuals = collect_actuals(df, season, week, id_col)
            if not week_actuals:
                continue
            relevant = df.filter(pl.col(id_col).is_in(list(week_actuals.keys())))
            if relevant.height == 0:
                continue
            assert_no_leakage(relevant, season, week)
            proj = build_projections(relevant, season, week, id_col, stat_columns, describe_row, n_games=8)
            hist = build_history(relevant, season, week, id_col, stat_columns)
            for entity_id, row in week_actuals.items():
                actuals[entity_id] = {c: row.get(c) for c in stat_columns}
            projections.update(proj)
            history.update(hist)

        relevant_opp = opp.filter(pl.col("player_id").is_in(list(actuals.keys())))
        opp_map = {}
        if relevant_opp.height > 0:
            assert_no_leakage(relevant_opp, season, week)
            opp_map = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)

        baseline_points = {eid: baseline_c_points(eid, proj, opp_map) for eid, proj in projections.items()}

        pool = build_pool_by_position(projections, actuals)
        dist_cache = {}
        matchups_built = 0
        for _ in range(MATCHUPS_PER_WEEK):
            drawn = draw_synthetic_matchup(pool, rng_py)
            if drawn is None:
                continue
            team_a, team_b = drawn
            entries_a = [(eid, projections[eid]["position"],
                          game_key(env_opponents, season, week, projections[eid].get("team"))) for eid in team_a]
            entries_b = [(eid, projections[eid]["position"],
                          game_key(env_opponents, season, week, projections[eid].get("team"))) for eid in team_b]
            win_prob_a = simulate_matchup_correlated(entries_a, entries_b, baseline_points, history, dist_cache, TRIALS, rng_np)

            actual_a = sum(compute_points(actuals[p]) for p in team_a)
            actual_b = sum(compute_points(actuals[p]) for p in team_b)
            if actual_a == actual_b:
                ties += 1
                continue

            favored_is_a = win_prob_a >= 0.5
            predicted_prob = win_prob_a if favored_is_a else (1 - win_prob_a)
            favored_won = (actual_a > actual_b) if favored_is_a else (actual_b > actual_a)
            records.append({"predicted_prob": predicted_prob, "favored_won": favored_won})
            matchups_built += 1

        if matchups_built == 0:
            skipped_weeks.append((season, week))
        print(f"  done: season {season} week {week} ({matchups_built} matchups)", end="\r")

    print()
    return records, skipped_weeks, ties


def bucket_table(records) -> str:
    lines = ["| Predicted probability bucket | n | Mean predicted prob | Actual win rate | 95% CI |", "|---|---|---|---|---|"]
    for lo, hi in PROB_BUCKETS:
        bucket = [r for r in records if lo <= r["predicted_prob"] < hi]
        n = len(bucket)
        if n == 0:
            lines.append(f"| {lo:.0%}-{hi:.0%} | 0 | - | - | - |")
            continue
        win_rate = sum(r["favored_won"] for r in bucket) / n
        mean_predicted = sum(r["predicted_prob"] for r in bucket) / n
        ci95 = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n) if n > 1 else 0.0
        lines.append(f"| {lo:.0%}-{hi:.0%} | {n} | {mean_predicted:.1%} | {win_rate:.1%} | ±{ci95:.1%} |")
    return "\n".join(lines)


def main() -> None:
    season = current_season()
    schedules_now = nfl.load_schedules(seasons=season)
    current_s, current_w = get_current_season_and_week(schedules_now)

    lookback_seasons = list(range(current_s - LOOKBACK_SEASONS, current_s + 1))
    print(f"Loading full historical data for seasons {lookback_seasons}...")

    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=lookback_seasons)
    run_no_leakage_tests(stats)

    is_k = (pl.col("position") == "K").fill_null(False)
    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    non_kickers = stats.filter(is_skill)
    kickers = add_k_stat_columns(stats.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )
    opp = load_opportunity_frame(lookback_seasons)
    _, env_opponents = build_environment_factors(schedules)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} historical weeks available; up to {MATCHUPS_PER_WEEK} "
          f"synthetic matchups/week, {TRIALS} MC trials each, {len(RNG_SEEDS)} seeds.")

    seed_tables = []
    for seed in RNG_SEEDS:
        print(f"\n=== Seed {seed} ===")
        records, skipped_weeks, ties = run_backtest(targets, non_kickers, kickers, team_stats, opp, env_opponents, seed)
        print(f"Total synthetic matchups: {len(records)} across {len(targets) - len(skipped_weeks)} weeks "
              f"({len(skipped_weeks)} skipped; {ties} ties excluded)")
        table = bucket_table(records)
        print(table)
        seed_tables.append((seed, table, len(records)))

    report_sections = "\n\n".join(
        f"### Seed {seed} ({n} synthetic matchups)\n\n{table}" for seed, table, n in seed_tables
    )
    report = f"""# Phase 4 Candidate: Shared Game-Level Factor Correlation Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE/backtest only. See MODEL_ROADMAP.md.

**What changed vs. the Phase 3 recentered result**: within each Monte Carlo trial, every player
(QB/RB/WR/TE/K/DEF) whose real team is playing in the SAME real NFL game that week now shares ONE random
"game quantile" for that trial (keyed by the real game, both competing teams together -- not just one team
in isolation) instead of each player picking an independent random index. Offensive players sample at that
quantile; DEF samples at `1 - quantile` (a shootout hurts both defenses, a defensive struggle helps both).
Builds on top of Phase 3's recentering (same distribution shift), not a replacement for it. (An earlier,
narrower same-TEAM-only version was tried first and showed essentially no effect -- see the script's own
docstring for why that was too narrow to matter much across randomly-drawn synthetic rosters.)

**For reference, Phase 3's recentered-only result, seed {RNG_SEEDS[0]}:**

| Predicted probability bucket | n | Mean predicted prob | Actual win rate | 95% CI |
|---|---|---|---|---|
| 50%-60% | 513 | 54.8% | 53.4% | ±4.3% |
| 60%-70% | 584 | 65.3% | 59.6% | ±4.0% |
| 70%-80% | 551 | 75.0% | 73.0% | ±3.7% |
| 80%-90% | 521 | 84.8% | 76.0% | ±3.7% |
| 90%-100% | 591 | 95.6% | 83.8% | ±3.0% |

**With shared game-level factors added:**

{report_sections}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
