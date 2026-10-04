"""
phase4_bucket_attribution.py

Phase 4, attributing the 90-100% calibration bucket's residual (2026-10-04):
after the predictive-scale fix closed most of the variance gap, that one
bucket still showed a real, CI-excluding overconfidence gap. Designer's
suspect: DEGENERATE histories (zero-variance or n<4) -- a player whose last
games were all 0.0 has a point-mass "distribution" even after predictive
scaling (scaling deviations-from-mean by a constant can't un-collapse a
distribution whose deviations are already all zero), but in reality even an
all-zeros history doesn't mean a true probability-zero of scoring (rule of
three: ~1/(n+1) chance of a nonzero outcome). A lopsided matchup stacked
with such players can produce a near-deterministic, ≥90%-probability
prediction that isn't actually that certain in reality.

This script, for every synthetic matchup landing in the 90-100% predicted-
probability bucket (both seeds, predictive-scale harness): records whether
either roster has >=1 player with a zero-variance or n<4 history, the
favored-win rate split by that flag, and the mean (actual - predicted) for
the favored side vs. the underdog side separately (a positive underdog
residual would point at a POOL-SELECTION effect too: the synthetic pool
only contains players who actually played, which can enrich the underdog/
scrub side with real role-expansion games a real roster decision couldn't
have foreseen in advance).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_bucket_attribution.py
"""

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
from calibration_check import build_pool_by_position, draw_synthetic_matchup  # noqa: E402
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
from phase4_baseline_d import fit_shrink  # noqa: E402
from phase4_predictive_scale import inflated_recentered_distribution  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_BUCKET_ATTRIBUTION_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
MIN_GAMES = 4


def is_degenerate(entity_id, history):
    games = history.get(entity_id)
    if not games or len(games) < MIN_GAMES:
        return True
    points = np.array([compute_points(g) for g in games])
    return points.var(ddof=0) == 0


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

    print("\n=== Rebuilding Baseline D weekly state ===")
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

    bucket_matchups = []  # across both seeds
    for seed in RNG_SEEDS:
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        print(f"\n=== Seed {seed} ===")
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

                predicted_prob = win_prob_a if win_prob_a >= 0.5 else (1 - win_prob_a)
                if not (0.90 <= predicted_prob <= 1.0):
                    continue

                favored_is_a = win_prob_a >= 0.5
                favored_ids = team_a if favored_is_a else team_b
                underdog_ids = team_b if favored_is_a else team_a

                actual_a = sum(compute_points(actuals[p]) for p in team_a)
                actual_b = sum(compute_points(actuals[p]) for p in team_b)
                predicted_a = sum(baseline_d.get(p, 0.0) for p in team_a)
                predicted_b = sum(baseline_d.get(p, 0.0) for p in team_b)
                actual_favored, predicted_favored = (actual_a, predicted_a) if favored_is_a else (actual_b, predicted_b)
                actual_underdog, predicted_underdog = (actual_b, predicted_b) if favored_is_a else (actual_a, predicted_a)
                favored_won = actual_favored > actual_underdog
                if actual_favored == actual_underdog:
                    continue

                has_degenerate = any(is_degenerate(eid, history) for eid in favored_ids + underdog_ids)

                bucket_matchups.append({
                    "favored_won": favored_won, "has_degenerate": has_degenerate,
                    "favored_residual": actual_favored - predicted_favored,
                    "underdog_residual": actual_underdog - predicted_underdog,
                })
            print(f"  done: season {season} week {week}", end="\r")
        print()

    n = len(bucket_matchups)
    with_deg = [m for m in bucket_matchups if m["has_degenerate"]]
    without_deg = [m for m in bucket_matchups if not m["has_degenerate"]]
    win_rate_with = sum(m["favored_won"] for m in with_deg) / len(with_deg) if with_deg else float("nan")
    win_rate_without = sum(m["favored_won"] for m in without_deg) / len(without_deg) if without_deg else float("nan")
    mean_favored_residual = sum(m["favored_residual"] for m in bucket_matchups) / n
    mean_underdog_residual = sum(m["underdog_residual"] for m in bucket_matchups) / n

    summary = f"""n = {n} matchups in the 90-100% predicted-probability bucket (both seeds combined)

Share with >=1 degenerate (zero-variance or n<{MIN_GAMES}) player on either roster: {len(with_deg)}/{n} ({len(with_deg)/n:.1%})

Favored-side win rate WITH >=1 degenerate player:    {win_rate_with:.1%}  (n={len(with_deg)})
Favored-side win rate WITHOUT any degenerate player:  {win_rate_without:.1%}  (n={len(without_deg)})

Mean (actual - predicted) for the FAVORED side:  {mean_favored_residual:+.2f}
Mean (actual - predicted) for the UNDERDOG side: {mean_underdog_residual:+.2f}
  (a positive underdog residual suggests the pool-selection effect: synthetic rosters only include
   players who actually played, which can enrich the underdog/scrub side with real role-expansion
   games a real roster decision couldn't have foreseen in advance)
"""
    print(summary)

    report = f"""# Phase 4 Bucket Attribution Report (90-100%)

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
DESIGNER_RESPONSE.md (2026-10-04).

{summary}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
