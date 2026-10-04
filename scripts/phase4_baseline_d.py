"""
phase4_baseline_d.py

Phase 4, correction 2: Baseline D, per the designer's explicit instruction
(2026-10-04) to sequence shrinkage before any correlation/copula work --
"the mean problem inflates predicted margins no matter how the variance is
modeled, and it feeds Var(z) too."

Baseline D shrinks Baseline C's own projection toward a per-position prior:
  baseline_d(entity) = mean_actual_prior[pos] + beta[pos] * (baseline_c(entity) - mean_baseline_c_prior[pos])
`beta[pos]` is the OLS slope of actual points on Baseline C points for that
position -- i.e. exactly diagnostic 1's "slope of actual on predicted,"
estimated PER POSITION instead of pooled. Both `beta` and the two anchor
means are fit from an EXPANDING WINDOW of strictly-prior backtest weeks
only (never the week being scored), per the designer's explicit warning
not to fit the shrink factor on the data it's evaluated on. Early weeks
with too little prior data (< MIN_SAMPLE player-weeks for that position)
fall back to beta=1 / no shrink (Baseline D == Baseline C that week),
same "insufficient history -> unchanged fallback" pattern used everywhere
else in this project (`league_rate()` in phase2_nextgen_stats.py, etc.).

This script does three things, in order, exactly as asked:
1. Backtests Baseline D vs. Baseline C: MAE/RMSE/bias by position (same
   format as backtest_evaluation.py's Baseline A/B/C tables).
2. Reruns diagnostic 1 (slope of actual margin on predicted margin, and
   Var(z)) using Baseline D in place of Baseline C -- does the slope move
   toward 1?
3. Reruns the 90-100% win-probability calibration bucket, both seeds,
   using Baseline D's points as the Monte Carlo recentering target instead
   of Baseline C's (same harness as phase3_calibration_check.py --
   `recentered_distribution()` is generic over whatever points dict it's
   given, so Baseline D slots in with zero changes to that function).

Judge on all three (MAE/RMSE/bias, slope-toward-1, calibration) -- per the
designer's own note that shrinkage should mainly help RMSE and calibration,
not necessarily MAE.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_baseline_d.py
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
    POSITIONS,
    assert_no_leakage,
    collect_actuals,
    compute_points,
    format_table,
    list_backtest_weeks,
    run_no_leakage_tests,
    summarize,
)
from calibration_check import (  # noqa: E402
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
from phase3_calibration_check import recentered_distribution  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_BASELINE_D_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
MIN_SAMPLE = 50  # player-weeks of prior data required before trusting a fitted beta
SD_FLOOR = 5.0  # same floor as phase4_diagnostics.py's Var(z), same reason


def fit_shrink(prior_data: dict) -> dict:
    """{position: (beta, mean_actual, mean_baseline_c, is_fallback)} from an
    expanding window of strictly-prior (baseline_c, actual) pairs. Falls
    back (is_fallback=True) to no shrink when there's too little prior
    data -- an explicit flag, not a fragile "beta happens to equal 1.0"
    floating-point check."""
    params = {}
    for position, pairs in prior_data.items():
        if len(pairs) < MIN_SAMPLE:
            params[position] = (1.0, 0.0, 0.0, True)
            continue
        bc = np.array([p[0] for p in pairs])
        actual = np.array([p[1] for p in pairs])
        beta, _intercept = np.polyfit(bc, actual, 1)
        params[position] = (float(beta), float(actual.mean()), float(bc.mean()), False)
    return params


def main() -> None:
    season = current_season()
    schedules_now = nfl.load_schedules(seasons=season)
    current_s, current_w = get_current_season_and_week(schedules_now)

    lookback_seasons = list(range(current_s - LOOKBACK_SEASONS, current_s + 1))
    print(f"Loading full historical data for seasons {lookback_seasons}...")

    stats_df = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=lookback_seasons)
    run_no_leakage_tests(stats_df)

    is_k = (pl.col("position") == "K").fill_null(False)
    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    non_kickers = stats_df.filter(is_skill)
    kickers = add_k_stat_columns(stats_df.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )
    opp = load_opportunity_frame(lookback_seasons)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} historical weeks available.")

    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]

    # ---- Phase A: deterministic -- build Baseline D per week, backtest MAE/RMSE/bias, retain per-week state.
    print("\n=== Phase A: building Baseline D (out-of-sample shrink per position) ===")
    prior_data = {p: [] for p in POSITIONS}
    weekly_state = {}  # (season, week) -> {"projections":..., "actuals":..., "history":..., "baseline_d": {...}, "beta": {...}}
    results = {"Baseline D (shrinkage)": []}

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

        baseline_c = {eid: baseline_c_points(eid, proj, opp_map) for eid, proj in projections.items()}
        beta_params = fit_shrink(prior_data)  # fit on STRICTLY PRIOR weeks only, before this week's own data is added below

        baseline_d = {}
        for eid, proj in projections.items():
            position = proj["position"]
            beta, mean_actual, mean_bc, is_fallback = beta_params.get(position, (1.0, 0.0, 0.0, True))
            bc_val = baseline_c[eid]
            baseline_d[eid] = bc_val if is_fallback else (mean_actual + beta * (bc_val - mean_bc))

        for eid, proj in projections.items():
            actual_row = actuals.get(eid)
            if actual_row is None:
                continue
            actual_pts = compute_points(actual_row)
            projected_pts = baseline_d[eid]
            results["Baseline D (shrinkage)"].append({
                "position": proj["position"], "season": season, "week": week,
                "projected": projected_pts, "actual": actual_pts, "error": projected_pts - actual_pts,
            })
            # Feed this week's (baseline_c, actual) into the running prior for FUTURE weeks.
            prior_data.setdefault(proj["position"], []).append((baseline_c[eid], actual_pts))

        weekly_state[(season, week)] = {
            "projections": projections, "actuals": actuals, "history": history, "baseline_d": baseline_d,
        }
        print(f"  done: season {season} week {week}", end="\r")
    print()

    table = format_table(results)
    print(table)

    # ---- Phase B: seed-dependent -- synthetic matchups, diagnostic-1-style margin check + calibration buckets.
    print("\n=== Phase B: diagnostic 1 (margin slope/Var(z)) + calibration buckets using Baseline D ===")
    seed_outputs = []
    for seed in RNG_SEEDS:
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        margin_rows, bucket_rows = [], []

        for season, week in targets:
            state = weekly_state.get((season, week))
            if state is None:
                continue
            projections, actuals, history, baseline_d = (
                state["projections"], state["actuals"], state["history"], state["baseline_d"]
            )
            pool = build_pool_by_position(projections, actuals)
            dist_cache = {}
            for _ in range(MATCHUPS_PER_WEEK):
                drawn = draw_synthetic_matchup(pool, rng_py)
                if drawn is None:
                    continue
                team_a, team_b = drawn

                def team_totals(ids):
                    total = np.zeros(TRIALS)
                    for eid in ids:
                        dist = recentered_distribution(eid, baseline_d, history, dist_cache)
                        total += rng_np.choice(dist, size=TRIALS, replace=True)
                    return total

                my_totals = team_totals(team_a)
                opp_totals = team_totals(team_b)
                win_prob_a = float(np.mean(my_totals > opp_totals))
                margin_sd = float(np.std(my_totals - opp_totals))

                actual_a = sum(compute_points(actuals[p]) for p in team_a)
                actual_b = sum(compute_points(actuals[p]) for p in team_b)
                predicted_margin = (sum(baseline_d.get(p, 0.0) for p in team_a)
                                    - sum(baseline_d.get(p, 0.0) for p in team_b))
                margin_rows.append({"predicted_margin": predicted_margin, "actual_margin": actual_a - actual_b,
                                     "margin_sd": margin_sd})

                if actual_a == actual_b:
                    continue
                favored_is_a = win_prob_a >= 0.5
                predicted_prob = win_prob_a if favored_is_a else (1 - win_prob_a)
                favored_won = (actual_a > actual_b) if favored_is_a else (actual_b > actual_a)
                bucket_rows.append({"predicted_prob": predicted_prob, "favored_won": favored_won})
            print(f"  seed {seed}: done season {season} week {week}", end="\r")
        print()

        predicted = np.array([r["predicted_margin"] for r in margin_rows])
        actual = np.array([r["actual_margin"] for r in margin_rows])
        sd = np.array([r["margin_sd"] for r in margin_rows])
        slope, intercept = np.polyfit(predicted, actual, 1)
        corr = np.corrcoef(predicted, actual)[0, 1]
        informative = sd >= SD_FLOOR
        z = (actual[informative] - predicted[informative]) / sd[informative]
        var_z = float(np.var(z))

        bucket_lines = ["| Predicted probability bucket | n | Mean predicted prob | Actual win rate | 95% CI |", "|---|---|---|---|---|"]
        for lo, hi in PROB_BUCKETS:
            bucket = [r for r in bucket_rows if lo <= r["predicted_prob"] < hi]
            n = len(bucket)
            if n == 0:
                bucket_lines.append(f"| {lo:.0%}-{hi:.0%} | 0 | - | - | - |")
                continue
            win_rate = sum(r["favored_won"] for r in bucket) / n
            mean_pred = sum(r["predicted_prob"] for r in bucket) / n
            ci95 = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n) if n > 1 else 0.0
            bucket_lines.append(f"| {lo:.0%}-{hi:.0%} | {n} | {mean_pred:.1%} | {win_rate:.1%} | ±{ci95:.1%} |")
        bucket_table = "\n".join(bucket_lines)

        summary = (f"Seed {seed}: n={len(margin_rows)} matchups. Slope={slope:.3f} (intercept {intercept:+.2f}, "
                   f"corr {corr:.3f}). Var(z)={var_z:.3f} (n={len(z)} after SD floor).")
        print(summary)
        print(bucket_table)
        seed_outputs.append((seed, summary, bucket_table))

    report_sections = "\n\n".join(f"### Seed {seed}\n\n{summary}\n\n{bucket}" for seed, summary, bucket in seed_outputs)
    report = f"""# Phase 4 Baseline D (Shrinkage) Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
DESIGNER_RESPONSE.md's correction 2 (2026-10-04).

## Backtest vs. Baseline C: MAE/RMSE/bias by position

{table}

## Diagnostic 1 rerun + calibration buckets, using Baseline D

For reference, Baseline C's diagnostic 1 (phase4_diagnostics.py): slope 0.806, Var(z) 1.576 (n=5400 of 5520).

{report_sections}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
