"""
phase4_combined_stack.py

Final combined-stack validation (designer decision, 2026-10-04), offline:
  Baseline C -> D-rolling (8-week rolling shrink) -> variance pooling
  (with a residual-shape fallback for degenerate histories) -> predictive
  scale.

Residual-shape fallback (replaces the Normal fallback, which is symmetric
and puts half of a near-zero scrub's draws below zero): for a player with
no usable empirical shape (s_i == 0 or n < 2), sample from his POSITION's
pooled residual shape -- residuals (actual - D-rolling projection) from
strictly-prior weeks at that position, standardized to mean 0 / sd 1,
rescaled to the player's pooled+scaled SD, shifted to his target. Keeps the
real right skew (busts are bounded by the projection, booms aren't). Until a
position has MIN_RESIDUALS prior residuals (only the earliest weeks), the
player stays a point mass at his target -- the pre-pooling behavior, not a
guessed shape.

Reports:
- MAE/RMSE/bias by position: Baseline A, Baseline C, combined stack (D-rolling).
- Var(z') and calibration buckets on TWO pools, both seeds:
  (a) all players who played (the synthetic pool used all along);
  (b) a realistic pool: each week, top-K by projection per position,
      approximating rostered players in a 14-team league
      (QB 20, RB 50, WR 60, TE 20, K 14, DEF 14).
- Negative-draw share for degenerate fallback distributions, residual
  shape vs. what the Normal fallback would have produced (sanity check on
  the floor the designer flagged).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_combined_stack.py
"""

import math
import random
import sys
from collections import deque
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
    format_table,
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
import model_core  # noqa: E402
from model_core import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402
from phase4_baseline_d_ablation import POSITIONS  # noqa: E402
from phase4_variance_pooling import K, fit_pos_var  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_COMBINED_STACK_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
SD_FLOOR = 5.0
MIN_RESIDUALS = 30
REALISTIC_TOP_K = {"QB": 20, "RB": 50, "WR": 60, "TE": 20, "K": 14, "DEF": 14}


def combined_distribution(entity_id, target, history, pos_var, resid_shape, position, cache, stats):
    """Pooled + predictive-scaled distribution centered on `target`.
    Real empirical shape when one exists; position residual shape otherwise."""
    if entity_id in cache:
        return cache[entity_id]
    games = history.get(entity_id)
    n = len(games) if games else 0
    raw = np.array([compute_points(g) for g in games]) if n else None
    s2 = float(raw.var(ddof=0)) if n else 0.0

    pv = pos_var.get(position)
    var_i = s2 if pv is None else (n / (n + K)) * s2 + (1 - n / (n + K)) * pv
    final_sd = math.sqrt(var_i * (n + 1) / (n - 1)) if n >= 2 else math.sqrt(var_i)

    if s2 > 0 and n >= 2:
        dist = target + (raw - raw.mean()) * (final_sd / math.sqrt(s2))
    else:
        shape = resid_shape.get(position)
        if shape is None or final_sd == 0:
            dist = np.array([target])
        else:
            dist = target + shape * final_sd
            stats["fallback_n"] += 1
            stats["resid_neg"] += float(np.mean(dist < 0))
            # What the old Normal fallback would have put below zero, for comparison.
            stats["normal_neg"] += 0.5 * math.erfc(target / (final_sd * math.sqrt(2)))
    cache[entity_id] = dist
    return dist


def realistic_pool(projections, actuals, baseline_d):
    full = build_pool_by_position(projections, actuals)
    return {pos: sorted(ids, key=lambda e: baseline_d[e], reverse=True)[:REALISTIC_TOP_K[pos]]
            for pos, ids in full.items()}


def resid_bin(target: float) -> int:
    return model_core.ratio_bin(target)


def build_state(extra_history_seasons: int = 0, return_pair_log: bool = False, bucket_stage=None):
    """Loads the backtest window and builds every week's deterministic state
    (projections, actuals, history, Baseline A/D points, pooled variances,
    residual shapes, projection-binned residual pools). Returns
    (targets, weekly_state, point_results).

    `extra_history_seasons` loads that many earlier seasons as HISTORY ONLY:
    they feed every fit (rolling shrink window, pos_var, residual pools,
    player histories) but are never scored or returned as targets -- the
    production-like setup where week 1 always has prior-season history.

    `return_pair_log=True` adds a 4th return value: every processed week's
    (season, week, scored, position, C points, actual points, D points, games used), oldest
    first, history-only weeks included -- for offline fit ablations.

    `bucket_stage` (None, "expanding", or a week count) adds the n-aware second
    stage for thin histories (model_core.fit_bucket_stage), fit only on
    (stage-1 D, actual) pairs from strictly-prior weeks in that window. The
    final projection then replaces D everywhere downstream (fallback pools,
    residuals, scoring); the pair log keeps stage-1 D as a 9th field."""
    season = current_season()
    schedules_now = nfl.load_schedules(seasons=season)
    current_s, current_w = get_current_season_and_week(schedules_now)
    scored_from = current_s - LOOKBACK_SEASONS
    lookback_seasons = list(range(scored_from - extra_history_seasons, current_s + 1))
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
    scored_targets = [t for t in targets if t[0] >= scored_from]
    print(f"\n{len(scored_targets)} scored historical weeks"
          f"{f' (+{len(targets) - len(scored_targets)} history-only)' if extra_history_seasons else ''}.")

    entity_groups = [
        (non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]

    # ---- Phase A: deterministic per-week state (no leakage: every fit uses strictly-prior weeks).
    # Per week, last ROLLING_WEEKS weeks: (position, C stats, actual stats) pairs for the shrink fit,
    # and (position, D points, actual points) entries for the fallback ratio pools.
    rolling_pair_weeks, rolling_entry_weeks = deque(), deque()
    stage_pair_weeks = []  # per week: (position, games_used, stage-1 D stats, actual stats)
    prior_var_data = {}
    prior_resid = {p: [] for p in POSITIONS}
    point_results = {"Baseline A -- last 8": [], "Baseline C -- opportunity blend": [], "Combined stack (D-rolling)": []}
    weekly_state = {}
    pair_log = []

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

        # Fits from strictly-prior data, taken BEFORE this week's data is added below.
        shrink = model_core.fit_window([pair for wk in rolling_pair_weeks for pair in wk])
        stage_window = (stage_pair_weeks if bucket_stage == "expanding"
                        else stage_pair_weeks[-bucket_stage:] if bucket_stage else [])
        stage = model_core.fit_bucket_stage([p for wk in stage_window for p in wk]) if bucket_stage else {}
        pos_var = fit_pos_var(prior_var_data)
        resid_shape = {}
        for p in POSITIONS:
            r = np.array(prior_resid[p])
            if len(r) >= MIN_RESIDUALS and r.std() > 0:
                resid_shape[p] = (r - r.mean()) / r.std()
        ratio_bins = {k: np.array(v) for k, v in
                      model_core.build_ratio_pools([e for wk in rolling_entry_weeks for e in wk]).items()}

        baseline_a, baseline_c, baseline_d, c_stats, d_stats, d1_points = {}, {}, {}, {}, {}, {}
        for eid, proj in projections.items():
            pos = proj["position"]
            opp_entry = opp_map.get(eid)
            c_stats[eid] = model_core.baseline_c_stats(proj["projected_stats"], pos,
                                                       opp_entry["projected_stats"] if opp_entry else None)
            baseline_a[eid] = compute_points(proj["projected_stats"])
            baseline_c[eid] = compute_points(c_stats[eid])
            d_stats[eid] = model_core.apply_d(c_stats[eid], pos, shrink)
            d1_points[eid] = compute_points(d_stats[eid])
            final = model_core.apply_bucket_stage(d_stats[eid], pos, proj["games_used"], stage) if stage else d_stats[eid]
            baseline_d[eid] = compute_points(final)

        scored = season >= scored_from
        week_pairs, week_entries, week_stage = [], [], []
        for eid, proj in projections.items():
            actual_row = actuals.get(eid)
            if actual_row is None:
                continue
            pos = proj["position"]
            actual_pts = compute_points(actual_row)
            if scored:
                for name, src in [("Baseline A -- last 8", baseline_a), ("Baseline C -- opportunity blend", baseline_c),
                                  ("Combined stack (D-rolling)", baseline_d)]:
                    point_results[name].append({"position": pos, "season": season, "week": week,
                                                "projected": src[eid], "actual": actual_pts,
                                                "error": src[eid] - actual_pts})
            week_pairs.append((pos, c_stats[eid], actual_row))
            week_stage.append((pos, proj["games_used"], d_stats[eid], actual_row))
            prior_resid[pos].append(actual_pts - baseline_d[eid])
            week_entries.append((pos, baseline_d[eid], actual_pts))
            pair_log.append((season, week, scored, pos, baseline_c[eid], actual_pts, baseline_d[eid],
                             proj["games_used"], d1_points[eid]))
            games = history.get(eid)
            if games and len(games) >= 2:
                prior_var_data.setdefault(pos, []).append(
                    float(np.array([compute_points(g) for g in games]).var(ddof=0)))

        rolling_pair_weeks.append(week_pairs)
        rolling_entry_weeks.append(week_entries)
        stage_pair_weeks.append(week_stage)
        while len(rolling_pair_weeks) > model_core.ROLLING_WEEKS:
            rolling_pair_weeks.popleft()
            rolling_entry_weeks.popleft()

        if scored:
            weekly_state[(season, week)] = {
                "projections": projections, "actuals": actuals, "history": history,
                "baseline_a": baseline_a, "baseline_d": baseline_d, "pos_var": pos_var,
                "resid_shape": resid_shape, "ratio_bins": ratio_bins,
            }
        print(f"  (state) done: season {season} week {week}", end="\r")
    print()
    if return_pair_log:
        return scored_targets, weekly_state, point_results, pair_log
    return scored_targets, weekly_state, point_results


def main() -> None:
    targets, weekly_state, point_results = build_state()
    point_table = format_table(point_results)
    print(point_table)

    # ---- Phase B: two pools x two seeds.
    pool_builders = {
        "(a) all players who played": lambda st: build_pool_by_position(st["projections"], st["actuals"]),
        "(b) realistic top-K pool": lambda st: realistic_pool(st["projections"], st["actuals"], st["baseline_d"]),
    }

    sections = []
    fallback_stats = {"fallback_n": 0, "resid_neg": 0.0, "normal_neg": 0.0}
    for pool_name, make_pool in pool_builders.items():
        for seed in RNG_SEEDS:
            rng_py = random.Random(seed)
            rng_np = np.random.default_rng(seed)
            margin_rows, bucket_rows = [], []
            for season, week in targets:
                st = weekly_state.get((season, week))
                if st is None:
                    continue
                pool = make_pool(st)
                position_of = {eid: p["position"] for eid, p in st["projections"].items()}
                dist_cache = {}
                for _ in range(MATCHUPS_PER_WEEK):
                    drawn = draw_synthetic_matchup(pool, rng_py)
                    if drawn is None:
                        continue
                    team_a, team_b = drawn

                    def team_totals(ids):
                        total = np.zeros(TRIALS)
                        for eid in ids:
                            dist = combined_distribution(eid, st["baseline_d"][eid], st["history"], st["pos_var"],
                                                         st["resid_shape"], position_of[eid], dist_cache, fallback_stats)
                            total += rng_np.choice(dist, size=TRIALS, replace=True)
                        return total

                    my_totals, opp_totals = team_totals(team_a), team_totals(team_b)
                    win_prob_a = float(np.mean(my_totals > opp_totals))
                    margin_sd = float(np.std(my_totals - opp_totals))
                    actual_a = sum(compute_points(st["actuals"][p]) for p in team_a)
                    actual_b = sum(compute_points(st["actuals"][p]) for p in team_b)
                    predicted_margin = (sum(st["baseline_d"][p] for p in team_a)
                                        - sum(st["baseline_d"][p] for p in team_b))
                    margin_rows.append((predicted_margin, actual_a - actual_b, margin_sd))
                    if actual_a == actual_b:
                        continue
                    favored_is_a = win_prob_a >= 0.5
                    bucket_rows.append({
                        "predicted_prob": win_prob_a if favored_is_a else 1 - win_prob_a,
                        "favored_won": (actual_a > actual_b) if favored_is_a else (actual_b > actual_a),
                    })
                print(f"  {pool_name[:3]} seed {seed}: done season {season} week {week}", end="\r")
            print()

            pred = np.array([r[0] for r in margin_rows])
            act = np.array([r[1] for r in margin_rows])
            sd = np.array([r[2] for r in margin_rows])
            b_fit, a_fit = np.polyfit(pred, act, 1)
            keep = sd >= SD_FLOOR
            var_zp = float(np.var((act[keep] - (a_fit + b_fit * pred[keep])) / sd[keep]))

            lines = ["| Bucket | n | Mean predicted | Actual win rate | 95% CI | Within CI? |", "|---|---|---|---|---|---|"]
            for lo, hi in PROB_BUCKETS:
                bucket = [r for r in bucket_rows if lo <= r["predicted_prob"] < hi]
                n = len(bucket)
                if n == 0:
                    lines.append(f"| {lo:.0%}-{hi:.0%} | 0 | - | - | - | - |")
                    continue
                wr = sum(r["favored_won"] for r in bucket) / n
                mp = sum(r["predicted_prob"] for r in bucket) / n
                ci = 1.96 * math.sqrt(wr * (1 - wr) / n) if n > 1 else 0.0
                within = "yes" if abs(wr - mp) <= ci else "NO"
                lines.append(f"| {lo:.0%}-{hi:.0%} | {n} | {mp:.1%} | {wr:.1%} | ±{ci:.1%} | {within} |")
            summary = (f"{pool_name}, seed {seed}: n={len(margin_rows)} matchups, slope b={b_fit:.3f}, "
                       f"Var(z')={var_zp:.3f} (n={int(keep.sum())} after SD floor)")
            print(summary)
            print("\n".join(lines))
            sections.append(f"### {summary}\n\n" + "\n".join(lines))

    fb = fallback_stats
    fb_line = (f"Degenerate-history fallback distributions built: {fb['fallback_n']}. Mean share of draws below 0: "
               f"residual shape {fb['resid_neg'] / max(fb['fallback_n'], 1):.1%} vs. "
               f"{fb['normal_neg'] / max(fb['fallback_n'], 1):.1%} the old Normal fallback would have had.")
    print("\n" + fb_line)

    report = f"""# Phase 4 Combined Stack Validation

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE only. Stack: Baseline C ->
D-rolling -> variance pooling (residual-shape fallback) -> predictive scale. See DESIGNER_RESPONSE.md.

## Point accuracy by position

{point_table}

## Calibration, two pools x two seeds

{chr(10).join(sections)}

## Degenerate-history fallback floor check

{fb_line}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
