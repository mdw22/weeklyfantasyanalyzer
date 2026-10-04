"""
phase4_variance_pooling.py

Phase 4, the designer's "variance pooling" fix (2026-10-04), targeting the
exact cause confirmed by phase4_bucket_attribution.py: 85.6% of the
90-100% bucket's matchups involve a player with a zero-variance or n<4
history, and those matchups are the ones actually miscalibrated (80.2%
real win rate vs. ~95% predicted); matchups without one are much closer to
calibrated (91.1%, plausibly within noise at n=56). The predictive-scale
fix (sqrt((n+1)/(n-1))) can't help a genuinely zero-variance history --
scaling deviations-from-mean by a constant leaves an all-zero deviation
array at all-zero.

Fix: shrink each player's own variance toward his position's typical
(out-of-sample) variance before applying the predictive scale:
  var_i = w * s_i^2 + (1-w) * pos_var[position],  w = n / (n + k),  k = 3
`pos_var[position]` is an expanding-window, strictly-prior-weeks average of
s_i^2 across all same-position players with n>=2 that week (same no-leakage
discipline as everything else; falls back to no pooling, i.e. w=1, when
there's too little prior data). `k=3` is a fixed, "small" choice per the
designer's own "k tuned or fixed small (try 2-4)" -- picked the middle of
that range rather than sweeping, to keep this a single run.

Then the predictive scale is applied ON TOP: final_sd_i =
sqrt(var_i * (n+1)/(n-1)) for n>=2 (the bootstrap-correction multiplier
doesn't apply the same way with no bootstrap sample at all, so n<2 just
uses sqrt(var_i) directly).

Sampling construction:
- s_i > 0 (a real empirical shape exists): rescale the real last-8-games
  shape to the new SD -- `target + (raw - mean) * (final_sd_i / s_i)`.
- s_i == 0 or n<2 (no usable shape -- the exact degenerate case this fix
  targets): fall back to Normal(target, final_sd_i), since there's no
  empirical shape information to preserve. A large (10,000-draw) cached
  sample, consistent with the existing "cache holds a representative
  distribution, rng.choice resamples from it per matchup" contract used
  for every bootstrap-based entity elsewhere in this project.

Reruns both seeds: the 90-100% bucket specifically, and Var(z') (freshly
fit a/b, same as phase4_predictive_scale.py).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_variance_pooling.py
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
from phase4_baseline_d import fit_shrink  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_VARIANCE_POOLING_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
SD_FLOOR = 5.0
MIN_POS_VAR_SAMPLE = 30
K = 3  # fixed, "small" per the designer's own allowance
NORMAL_FALLBACK_POOL = 10000


def fit_pos_var(prior_var_data: dict) -> dict:
    """{position: pos_var or None} from an expanding window of strictly-
    prior per-player variances (n>=2 only -- a single-game "variance" is
    always 0 and would bias the pooled target down if included)."""
    out = {}
    for position, values in prior_var_data.items():
        out[position] = (sum(values) / len(values)) if len(values) >= MIN_POS_VAR_SAMPLE else None
    return out


def pooled_distribution(entity_id, baseline_points, history, pos_var_by_position, position_of, cache, rng):
    if entity_id in cache:
        return cache[entity_id]
    target = baseline_points.get(entity_id, 0.0)
    games = history.get(entity_id)
    n = len(games) if games else 0
    s2 = 0.0
    raw = None
    if n >= 1:
        raw = np.array([compute_points(g) for g in games])
        s2 = float(raw.var(ddof=0))

    pos_var = pos_var_by_position.get(position_of.get(entity_id))
    if pos_var is None:
        var_i = s2  # no prior reference yet -- fall back to the player's own (possibly zero) variance
    else:
        w = n / (n + K)
        var_i = w * s2 + (1 - w) * pos_var

    final_sd = math.sqrt(var_i * (n + 1) / (n - 1)) if n >= 2 else math.sqrt(var_i)

    if s2 > 0 and raw is not None:
        dist = target + (raw - raw.mean()) * (final_sd / math.sqrt(s2))
    else:
        dist = rng.normal(loc=target, scale=final_sd if final_sd > 0 else 1e-9, size=NORMAL_FALLBACK_POOL)
    cache[entity_id] = dist
    return dist


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

    print("\n=== Rebuilding Baseline D weekly state + out-of-sample position variances ===")
    prior_data, prior_var_data = {}, {}
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

        pos_var_this_week = fit_pos_var(prior_var_data)
        for eid, proj in projections.items():
            games = history.get(eid)
            if games and len(games) >= 2:
                s2 = float(np.array([compute_points(g) for g in games]).var(ddof=0))
                prior_var_data.setdefault(proj["position"], []).append(s2)

        position_of = {eid: proj["position"] for eid, proj in projections.items()}
        weekly_state[(season, week)] = {
            "projections": projections, "actuals": actuals, "history": history,
            "baseline_d": baseline_d, "pos_var": pos_var_this_week, "position_of": position_of,
        }
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
            projections, actuals, history, baseline_d, pos_var, position_of = (
                state["projections"], state["actuals"], state["history"], state["baseline_d"],
                state["pos_var"], state["position_of"],
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
                        dist = pooled_distribution(eid, baseline_d, history, pos_var, position_of, dist_cache, rng_np)
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
        b_fit, a_fit = np.polyfit(predicted, actual, 1)
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
    report = f"""# Phase 4 Variance Pooling Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
DESIGNER_RESPONSE.md (2026-10-04).

For reference: predictive-scale-only Var(z') was 1.156/1.178 (two seeds); its 90-100% bucket was
95.5%/82.7% (seed 1) and 95.6%/80.9% (seed 2).

{report_sections}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
