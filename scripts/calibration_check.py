"""
calibration_check.py

Phase 1, Step 4 of PHASE1_EVALUATION_SPEC.md: synthetic matchup win-probability
calibration check. OFFLINE ONLY -- run manually/locally, does not touch
public/data/, the live site, or any production pipeline.

This app keeps no real historical matchup records (no league-matchup archive
exists), so there's no way to backtest "did our actual Week N win probability
come true." Per the spec, this instead builds many SYNTHETIC matchups: two
non-overlapping, roster-shaped sets of real players who actually played in a
given historical week (same 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX / 1 DEF / 1 K
starter shape as src/lib/rosterSlots.js's STARTER_SLOT_IDS -- bench/IR slots
don't affect a matchup's score, so they're excluded here). It runs a Python
port of src/lib/monteCarlo.js's bootstrap-resampling simulation using Step 1's
as-of-that-week projections/history (same no-leakage guard, re-demonstrated
below rather than just assumed from Step 1), records the model's claimed win
probability, then checks it against which side ACTUALLY scored more that
real week. Bucketed by predicted-probability range, this is a real
calibration check: if the model says "65% to win," real synthetic matchups
in that bucket should win roughly 65% of the time.

Deliberately reuses rather than reimplements: generate_projections.py's
build_projections()/build_history() (both already no-leakage-filtered, same
as Step 1), backtest_evaluation.py's FULL_PPR_VALUES/compute_points (same
scoring values as everywhere else in this project) and
list_backtest_weeks()/assert_no_leakage()/collect_actuals()/
run_no_leakage_tests() (same backtest-week selection and leakage guard as
Step 1/2). The only new logic is the synthetic-roster drawing and the Monte
Carlo port itself (vectorized with numpy instead of monteCarlo.js's
per-trial JS loop, but the same bootstrap-a-whole-past-game-line-then-score
sampling scheme).

Requires numpy (for vectorized bootstrap resampling across thousands of synthetic
matchups -- pure-Python per-trial loops at this volume were far too slow). NOT
added to scripts/requirements.txt, which drives the production daily pipeline
workflow that never runs this script; install it locally only
(`pip install numpy`, or `--break-system-packages` if the system Python is
externally managed, matching how nflreadpy itself is installed in this repo's
dev environment).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/calibration_check.py
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

OUT_PATH = Path(__file__).parent.parent / "CALIBRATION_REPORT.md"  # gitignored, same as BACKTEST_REPORT.md

TRIALS = 5000  # monteCarlo.js's production default is 10000; halved here purely
# for runtime across hundreds of synthetic matchups -- the calibration question
# (does the claimed probability match the real win rate) doesn't need
# production-grade per-matchup precision, just enough trials that each
# matchup's own probability estimate isn't noise-dominated.
MATCHUPS_PER_WEEK = 60
RNG_SEED = 20261004  # fixed so this report is reproducible; nothing adversarial here

# Starter shape, must match src/lib/rosterSlots.js's STARTER_SLOT_IDS exactly
# (QB1, RB1, RB2, TE1, WR1, WR2, FLEX, DEF1, K1 -- 9 starters). Bench/IR slots
# excluded: they never contribute to a real matchup's score.
FIXED_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DEF": 1, "K": 1}
FLEX_ELIGIBLE = ["RB", "WR", "TE"]
PROB_BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.001)]


def player_points_distribution(entity_id: str, projections: dict, history: dict, cache: dict) -> np.ndarray | None:
    """Python port of monteCarlo.js's buildSampler -- bootstrap from real
    past games, fall back to the projected mean line if there's no history
    yet (rookies/new starters). Returns the array of candidate FANTASY-POINT
    values up front (one per past game, or one projected value) rather than
    a per-trial closure: resampling a whole past stat line and then scoring
    it is equivalent to resampling its already-computed point value, since
    compute_points is linear and a bootstrap draw never mixes two games --
    this lets every trial for this player be one vectorized
    np.random.Generator.choice call instead of a Python-level loop.
    Cached per (week) call site since the same player recurs across many of
    that week's synthetic matchups with an identical distribution each time."""
    if entity_id in cache:
        return cache[entity_id]
    games = history.get(entity_id)
    if games:
        dist = np.array([compute_points(g) for g in games])
    else:
        proj = projections.get(entity_id)
        dist = np.array([compute_points(proj["projected_stats"])]) if proj else None
    cache[entity_id] = dist
    return dist


def simulate_matchup(my_ids, opp_ids, projections, history, dist_cache, trials, rng) -> float:
    def team_totals(ids):
        total = np.zeros(trials)
        for entity_id in ids:
            dist = player_points_distribution(entity_id, projections, history, dist_cache)
            total += rng.choice(dist, size=trials, replace=True)
        return total

    my_totals = team_totals(my_ids)
    opp_totals = team_totals(opp_ids)
    return float(np.mean(my_totals > opp_totals))


def build_pool_by_position(projections: dict, actuals: dict) -> dict:
    """Players eligible for a synthetic roster this week: must have BOTH an
    as-of-this-week projection (to simulate) AND a real actual stat line (to
    check the real outcome) -- same selection restriction Step 1/2 already
    made, for the same reason (see BACKTEST_REPORT.md's sample-selection
    caveat)."""
    pool = {pos: [] for pos in ["QB", "RB", "WR", "TE", "DEF", "K"]}
    for entity_id, proj in projections.items():
        if entity_id not in actuals:
            continue
        pos = proj["position"]
        if pos in pool:
            pool[pos].append(entity_id)
    return pool


def draw_synthetic_matchup(pool: dict, rng: random.Random):
    """Two non-overlapping 9-player starter-shaped rosters, drawn fresh from
    this week's pool. Returns None if the week's pool is too thin at some
    position to fill both sides (shouldn't happen in-season with real
    weekly data, but don't guess -- skip and count it instead)."""
    used = set()
    team_a, team_b = [], []

    for pos, count in FIXED_SLOTS.items():
        needed = count * 2
        candidates = [p for p in pool[pos] if p not in used]
        if len(candidates) < needed:
            return None
        picked = rng.sample(candidates, needed)
        used.update(picked)
        team_a.extend(picked[:count])
        team_b.extend(picked[count:])

    flex_candidates = [p for pos in FLEX_ELIGIBLE for p in pool[pos] if p not in used]
    if len(flex_candidates) < 2:
        return None
    flex_picked = rng.sample(flex_candidates, 2)
    team_a.append(flex_picked[0])
    team_b.append(flex_picked[1])

    return team_a, team_b


def main() -> None:
    season = current_season()
    schedules_now = nfl.load_schedules(seasons=season)
    current_s, current_w = get_current_season_and_week(schedules_now)

    lookback_seasons = list(range(current_s - LOOKBACK_SEASONS, current_s + 1))
    print(f"Loading full historical data for seasons {lookback_seasons}...")

    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=lookback_seasons)

    # Re-demonstrated here, not just assumed from Step 1 -- this script calls
    # build_projections()/build_history() independently, so the guard needs
    # its own proof against THIS script's actual usage, per the spec's
    # verification bar ("explicitly demonstrate the no-leakage test failing
    # when leakage is deliberately introduced before calling it passing").
    run_no_leakage_tests(stats)

    is_k = (pl.col("position") == "K").fill_null(False)
    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    non_kickers = stats.filter(is_skill)
    kickers = add_k_stat_columns(stats.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} historical weeks available; up to {MATCHUPS_PER_WEEK} "
          f"synthetic matchups/week, {TRIALS} MC trials each.")

    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]

    rng_py = random.Random(RNG_SEED)
    rng_np = np.random.default_rng(RNG_SEED)
    records = []
    skipped_weeks = []
    ties = 0

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
            # Restrict each actual row to ONLY its own group's stat_columns before
            # it's ever scored -- team_stats (DEF) rows also carry the team's own
            # OFFENSIVE passing_yards/rushing_yards/etc (unrelated to defense) under
            # the exact same column names FULL_PPR_VALUES uses for individual skill
            # players; passing a raw row straight to compute_points() would silently
            # add a defense's own team's passing yards to its fantasy score. This is
            # the same guard evaluate_week() in backtest_evaluation.py already
            # applies for Step 1/2 -- must be applied here too, not assumed.
            for entity_id, row in week_actuals.items():
                actuals[entity_id] = {c: row.get(c) for c in stat_columns}
            projections.update(proj)
            history.update(hist)

        pool = build_pool_by_position(projections, actuals)
        dist_cache = {}
        matchups_built = 0
        for _ in range(MATCHUPS_PER_WEEK):
            drawn = draw_synthetic_matchup(pool, rng_py)
            if drawn is None:
                continue
            team_a, team_b = drawn
            win_prob_a = simulate_matchup(team_a, team_b, projections, history, dist_cache, TRIALS, rng_np)

            actual_a = sum(compute_points(actuals[p]) for p in team_a)
            actual_b = sum(compute_points(actuals[p]) for p in team_b)
            if actual_a == actual_b:
                ties += 1
                continue  # no "favored side won" outcome to bucket for a true tie

            favored_is_a = win_prob_a >= 0.5
            predicted_prob = win_prob_a if favored_is_a else (1 - win_prob_a)
            favored_won = (actual_a > actual_b) if favored_is_a else (actual_b > actual_a)
            records.append({
                "season": season, "week": week,
                "predicted_prob": predicted_prob, "favored_won": favored_won,
            })
            matchups_built += 1

        if matchups_built == 0:
            skipped_weeks.append((season, week))
        print(f"  done: season {season} week {week} ({matchups_built} matchups)", end="\r")

    print()
    print(f"\nTotal synthetic matchups: {len(records)} across "
          f"{len(targets) - len(skipped_weeks)} weeks "
          f"({len(skipped_weeks)} weeks skipped: pool too thin; {ties} exact ties excluded)")

    bucket_lines = []
    for lo, hi in PROB_BUCKETS:
        bucket = [r for r in records if lo <= r["predicted_prob"] < hi]
        n = len(bucket)
        if n == 0:
            bucket_lines.append((lo, hi, 0, None, None, None))
            continue
        win_rate = sum(r["favored_won"] for r in bucket) / n
        mean_predicted = sum(r["predicted_prob"] for r in bucket) / n
        # Normal-approximation 95% CI on the actual win rate (binomial proportion),
        # same `1.96 * SE` convention used for MAE's CI in BACKTEST_REPORT.md.
        ci95 = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n) if n > 1 else 0.0
        bucket_lines.append((lo, hi, n, win_rate, ci95, mean_predicted))

    table_lines = [
        "| Predicted probability bucket | n | Mean predicted prob | Actual win rate | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for lo, hi, n, win_rate, ci95, mean_predicted in bucket_lines:
        if win_rate is None:
            table_lines.append(f"| {lo:.0%}-{hi:.0%} | 0 | - | - | - |")
        else:
            table_lines.append(
                f"| {lo:.0%}-{hi:.0%} | {n} | {mean_predicted:.1%} | {win_rate:.1%} | ±{ci95:.1%} |"
            )
    table = "\n".join(table_lines)
    print(table)

    report = f"""# Synthetic Matchup Calibration Report — Phase 1, Step 4

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only -- does not
affect the live site or production pipeline. See MODEL_ROADMAP.md / PHASE1_EVALUATION_SPEC.md.

**What this checks**: this app keeps no real historical matchup records, so this builds SYNTHETIC matchups
instead -- two non-overlapping, roster-shaped (1 QB/2 RB/2 WR/1 TE/1 FLEX/1 DEF/1 K) sets of real players who
actually played in a given historical week, runs the same bootstrap-resampling Monte Carlo logic
`src/lib/monteCarlo.js` uses in production (ported to Python, {TRIALS} trials/matchup) against Step 1's
as-of-that-week, no-leakage projections/history, and checks the model's claimed win probability against
which side actually scored more that real week.

**No-leakage test: PASSED** (re-demonstrated independently in this script, same guard as Step 1 -- see
`run_no_leakage_tests()`; real filter leaks zero rows, a deliberately-broken `<=` filter is shown to leak
exactly that week's own rows).

Window: {len(targets)} historical weeks, {targets[0]} through {targets[-1]}. Built {len(records)} synthetic
matchups across {len(targets) - len(skipped_weeks)} of those weeks ({len(skipped_weeks)} skipped — pool too
thin at some position to fill both sides; {ties} exact point-ties excluded, since there's no "favored side
won" outcome to bucket for those).

**Calibration table** (if the model is well-calibrated, each bucket's actual win rate should be close to its
own probability range):

{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
