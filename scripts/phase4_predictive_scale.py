"""
phase4_predictive_scale.py

Phase 4, the designer's "my own miss" correction (2026-10-04): diagnostic 3
tested a Student-t predictive interval, but the Monte Carlo doesn't draw
from that -- it resamples n historical games WITH REPLACEMENT, whose
variance is s^2*(n-1)/n (the empirical/population variance of the n-point
sample), not the correct predictive variance for a NEW observation,
s^2*(n+1)/n. For n=8 that's a 1.29x variance ratio (~13% SD) understated,
per player, before any correlation -- and since these are independent
per-player gaps, they add straight into the team total. This could be a
much simpler, more mechanical explanation for Var(z)>1 than correlation.

Two things, in the designer's order:

1. DIAGNOSTIC 3B -- test the distribution actually in use (full scale, by
   position), not a parametric t-interval: coverage of the bootstrap's own
   empirical 50%/80% quantile interval, plus mean squared prediction error
   / mean bootstrap variance (ratio well above 1 confirms the real-world
   gap directly, without needing a parametric assumption at all).
2. PREDICTIVE-SCALE CANDIDATE -- scale each player's deviations from his
   own mean by sqrt((n+1)/(n-1)) before feeding the Monte Carlo (guard
   n<3, where the ratio is unstable and there's barely a sample to begin
   with -- left unscaled). Rerun Baseline D's calibration buckets (both
   seeds) with this inflated distribution, and recompute
   z' = (actual - (a+b*predicted)) / SD with a freshly-fit (a,b) specific
   to this check (removing any remaining mean-miscalibration from the
   variance read, per the designer's own earlier nit).

Reuses Baseline D's own per-position shrink fit (same expanding-window,
same fallback) as the Monte Carlo's recentering target -- this isolates
the VARIANCE-SCALE fix from the MEAN-formulation question, which is a
separate, parallel question (phase4_baseline_d_ablation.py).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_predictive_scale.py
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
from calibration_check import PROB_BUCKETS, build_pool_by_position, draw_synthetic_matchup  # noqa: E402
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
from phase4_baseline_d import MIN_SAMPLE, fit_shrink  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_PREDICTIVE_SCALE_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
SD_FLOOR = 5.0
MIN_GAMES_FOR_INTERVAL = 4


def inflated_recentered_distribution(entity_id, baseline_points, history, cache):
    """Same shape as phase3's recentered_distribution, but deviations from
    the player's own mean are scaled by sqrt((n+1)/(n-1)) -- the correct
    predictive-variance correction for resampling n historical games,
    instead of taking the raw empirical (bootstrap) variance at face
    value. n<3 left unscaled (barely a sample, the ratio is unstable)."""
    if entity_id in cache:
        return cache[entity_id]
    target = baseline_points.get(entity_id, 0.0)
    games = history.get(entity_id)
    if games:
        raw = np.array([compute_points(g) for g in games])
        n = len(raw)
        if n >= 3:
            scale = math.sqrt((n + 1) / (n - 1))
            dist = target + (raw - raw.mean()) * scale
        else:
            dist = raw - raw.mean() + target
    else:
        dist = np.array([target])
    cache[entity_id] = dist
    return dist


def run_diagnostic_3b(targets, non_kickers, kickers, team_stats):
    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]
    by_position = {}
    for season, week in targets:
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
                games = hist.get(entity_id)
                if not games or len(games) < MIN_GAMES_FOR_INTERVAL:
                    continue
                points = np.array([compute_points(g) for g in games])
                mean = points.mean()
                bootstrap_var = points.var(ddof=0)  # the variance rng.choice(dist) actually produces
                actual_points = compute_points({c: row.get(c) for c in stat_columns})
                lo50, hi50 = np.percentile(points, [25, 75])
                lo80, hi80 = np.percentile(points, [10, 90])
                position = proj.get(entity_id, {}).get("position", "?")
                by_position.setdefault(position, []).append({
                    "sq_err": (actual_points - mean) ** 2, "bootstrap_var": bootstrap_var,
                    "cov50": lo50 <= actual_points <= hi50, "cov80": lo80 <= actual_points <= hi80,
                })
        print(f"  done: season {season} week {week}", end="\r")
    print()
    return by_position


def format_diagnostic_3b(by_position) -> str:
    lines = ["| Position | n | 50% empirical coverage | 80% empirical coverage | MSE / mean bootstrap var |",
             "|---|---|---|---|---|"]
    all_rows = []
    for position in sorted(by_position):
        rows = by_position[position]
        all_rows.extend(rows)
        n = len(rows)
        cov50 = sum(r["cov50"] for r in rows) / n
        cov80 = sum(r["cov80"] for r in rows) / n
        ratio = sum(r["sq_err"] for r in rows) / sum(r["bootstrap_var"] for r in rows)
        lines.append(f"| {position} | {n} | {cov50:.1%} | {cov80:.1%} | {ratio:.3f} |")
    n = len(all_rows)
    cov50 = sum(r["cov50"] for r in all_rows) / n
    cov80 = sum(r["cov80"] for r in all_rows) / n
    ratio = sum(r["sq_err"] for r in all_rows) / sum(r["bootstrap_var"] for r in all_rows)
    lines.append(f"| **Overall** | **{n}** | **{cov50:.1%}** | **{cov80:.1%}** | **{ratio:.3f}** |")
    lines.append("\n(nominal coverage targets: 50.0% and 80.0%; MSE/var ratio = 1.0 means well-calibrated)")
    return "\n".join(lines)


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

    print("\n=== Diagnostic 3b: coverage of the ACTUAL bootstrap distribution, by position ===")
    by_position = run_diagnostic_3b(targets, non_kickers, kickers, team_stats)
    d3b = format_diagnostic_3b(by_position)
    print(d3b)

    print("\n=== Predictive-scale candidate: rebuild Baseline D state, then rerun calibration with inflated SD ===")
    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]
    prior_data = {}
    weekly_state = {}
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
        beta_params = fit_shrink(prior_data)
        baseline_d = {}
        for eid, proj in projections.items():
            position = proj["position"]
            beta, mean_actual, mean_bc, is_fallback = beta_params.get(position, (1.0, 0.0, 0.0, True))
            baseline_d[eid] = baseline_c[eid] if is_fallback else (mean_actual + beta * (baseline_c[eid] - mean_bc))
        for eid, proj in projections.items():
            actual_row = actuals.get(eid)
            if actual_row is None:
                continue
            prior_data.setdefault(proj["position"], []).append((baseline_c[eid], compute_points(actual_row)))

        weekly_state[(season, week)] = {"projections": projections, "actuals": actuals, "history": history, "baseline_d": baseline_d}
        print(f"  (rebuild) done: season {season} week {week}", end="\r")
    print()

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
                        dist = inflated_recentered_distribution(eid, baseline_d, history, dist_cache)
                        total += rng_np.choice(dist, size=TRIALS, replace=True)
                    return total

                my_totals = team_totals(team_a)
                opp_totals = team_totals(team_b)
                win_prob_a = float(np.mean(my_totals > opp_totals))
                margin_sd = float(np.std(my_totals - opp_totals))

                actual_a = sum(compute_points(actuals[p]) for p in team_a)
                actual_b = sum(compute_points(actuals[p]) for p in team_b)
                predicted_margin = sum(baseline_d.get(p, 0.0) for p in team_a) - sum(baseline_d.get(p, 0.0) for p in team_b)
                margin_rows.append({"predicted_margin": predicted_margin, "actual_margin": actual_a - actual_b, "margin_sd": margin_sd})

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
        b_fit, a_fit = np.polyfit(predicted, actual, 1)  # polyfit returns [slope, intercept]
        informative = sd >= SD_FLOOR
        zprime = (actual[informative] - (a_fit + b_fit * predicted[informative])) / sd[informative]
        var_zprime = float(np.var(zprime))

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

        summary = f"Seed {seed}: n={len(margin_rows)}. b={b_fit:.3f}, a={a_fit:+.2f}. Var(z')={var_zprime:.3f} (n={len(zprime)})."
        print(summary)
        print(bucket_table)
        seed_outputs.append((seed, summary, bucket_table))

    report_sections = "\n\n".join(f"### Seed {seed}\n\n{summary}\n\n{bucket}" for seed, summary, bucket in seed_outputs)
    report = f"""# Phase 4 Predictive-Scale Candidate Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
DESIGNER_RESPONSE.md (2026-10-04).

## Diagnostic 3b -- coverage of the distribution actually used by the Monte Carlo

{d3b}

## Predictive-scale candidate (deviations scaled by sqrt((n+1)/(n-1))) -- calibration vs. Baseline D

For reference, Baseline D's (unscaled) diagnostic 1 Var(z): 1.512 (seed 1) / 1.532 (seed 2).

{report_sections}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
