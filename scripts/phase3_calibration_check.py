"""
phase3_calibration_check.py

Phase 3's first candidate, per the designer's explicit scoping (2026-10-04):
"decouple monteCarlo.js's simulated distribution from the raw
historical-games bootstrap, recenter/calibrate it around Baseline C's own
projection rather than the outcome-only mean it uses today, and validate the
result against Step 4's own synthetic-matchup calibration framework (same
buckets, same two-seed robustness check) so we can see directly whether the
90-100% overconfidence gap actually closes." OFFLINE/BACKTEST ONLY -- does
not touch monteCarlo.js or any production file; this is still not a
production change until reviewed.

Design, deliberately the SMALLEST change that matches the ask: keeps the
bootstrap's natural SHAPE (a player's own real game-to-game variability and
skew across his last 8 games is real signal about his boom/bust profile,
not something to discard) but SHIFTS its location so the sample's own mean
equals Baseline C's point estimate instead of whatever the raw historical
mean happens to be:
  shifted_sample = raw_bootstrap_sample - mean(raw_bootstrap_sample) + baseline_c_point_estimate
This does NOT touch the spread/variance itself, or the cross-player
independence assumption -- those are explicitly Phase 4's scope (shared
game-environment factors, QB/pass-catcher correlation), not this cut. This
backtest isolates exactly what the designer asked to see: how much of the
measured overconfidence gap closes from RECENTERING alone, leaving whatever
remains for Phase 4.

Reuses calibration_check.py's build_pool_by_position()/draw_synthetic_matchup()/
FIXED_SLOTS/FLEX_ELIGIBLE/PROB_BUCKETS unchanged -- a fair side-by-side
comparison against Step 4's original CALIBRATION_REPORT.md numbers needs the
harness identical apart from the one thing being tested (the distribution's
center). Reuses phase2_game_environment.py's baseline_c_points() (Baseline
C's QB/TE-blended point estimate) and phase2_opportunity_blend.py's
OPP_STAT_COLUMNS/load_opportunity_frame for the QB/TE opportunity blend
underneath Baseline C. Runs the SAME two-RNG-seed robustness check Step 4
itself used (default seed + seed 777), not a single run, per the designer's
explicit ask.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase3_calibration_check.py
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
from phase2_game_environment import baseline_c_points  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE3_CALIBRATION_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]  # same two seeds Step 4's own robustness check used


def recentered_distribution(entity_id, baseline_points, history, cache):
    """Same bootstrap-from-history-or-fallback shape as calibration_check.py's
    player_points_distribution(), but SHIFTED so the sample's own mean equals
    Baseline C's point estimate for this entity, instead of whatever the raw
    historical mean happens to be. No history -> a single deterministic value
    at the Baseline C estimate (same fallback behavior as before, just a
    different point estimate feeding it)."""
    if entity_id in cache:
        return cache[entity_id]
    target = baseline_points.get(entity_id, 0.0)
    games = history.get(entity_id)
    if games:
        raw = np.array([compute_points(g) for g in games])
        dist = raw - raw.mean() + target
    else:
        dist = np.array([target])
    cache[entity_id] = dist
    return dist


def simulate_matchup_recentered(my_ids, opp_ids, baseline_points, history, dist_cache, trials, rng) -> float:
    def team_totals(ids):
        total = np.zeros(trials)
        for entity_id in ids:
            dist = recentered_distribution(entity_id, baseline_points, history, dist_cache)
            total += rng.choice(dist, size=trials, replace=True)
        return total

    my_totals = team_totals(my_ids)
    opp_totals = team_totals(opp_ids)
    return float(np.mean(my_totals > opp_totals))


def run_backtest(targets, non_kickers, kickers, team_stats, opp, seed):
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
            win_prob_a = simulate_matchup_recentered(team_a, team_b, baseline_points, history, dist_cache, TRIALS, rng_np)

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

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} historical weeks available; up to {MATCHUPS_PER_WEEK} "
          f"synthetic matchups/week, {TRIALS} MC trials each, {len(RNG_SEEDS)} seeds.")

    seed_tables = []
    for seed in RNG_SEEDS:
        print(f"\n=== Seed {seed} ===")
        records, skipped_weeks, ties = run_backtest(targets, non_kickers, kickers, team_stats, opp, seed)
        print(f"Total synthetic matchups: {len(records)} across {len(targets) - len(skipped_weeks)} weeks "
              f"({len(skipped_weeks)} skipped; {ties} ties excluded)")
        table = bucket_table(records)
        print(table)
        seed_tables.append((seed, table, len(records)))

    report_sections = "\n\n".join(
        f"### Seed {seed} ({n} synthetic matchups)\n\n{table}" for seed, table, n in seed_tables
    )
    report = f"""# Phase 3 Candidate: Recentered Monte Carlo Calibration Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE/backtest only -- does not
touch monteCarlo.js or any production file. See MODEL_ROADMAP.md.

**What changed vs. Step 4's original CALIBRATION_REPORT.md**: each player's simulated point distribution is
still bootstrap-resampled from his own real last-8 games (same shape/skew as before), but the sample is
SHIFTED so its mean equals Baseline C's own (QB/TE opportunity-blended) point estimate instead of whatever
the raw historical mean happened to be. Spread/variance and the cross-player independence assumption are
untouched -- those are Phase 4's scope, not this cut. Same roster-drawing, same buckets, same two-seed
robustness check as Step 4's original run, for a fair side-by-side comparison.

**For reference, Step 4's original (pre-Phase-3) result, seed {RNG_SEEDS[0]}:**

| Predicted probability bucket | n | Mean predicted prob | Actual win rate | 95% CI |
|---|---|---|---|---|
| 50%-60% | 514 | 54.9% | 53.5% | ±4.3% |
| 60%-70% | 548 | 65.2% | 60.9% | ±4.1% |
| 70%-80% | 558 | 74.8% | 70.8% | ±3.8% |
| 80%-90% | 532 | 84.7% | 75.2% | ±3.7% |
| 90%-100% | 608 | 95.6% | 84.2% | ±2.9% |

**Recentered result:**

{report_sections}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
