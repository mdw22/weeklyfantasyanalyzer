"""
phase4_diagnostics.py

Diagnoses WHY the measured win-probability overconfidence exists, before
attempting another Monte Carlo correlation variant -- per the designer's
explicit correction (2026-10-04): calling cross-player correlation "the
mechanism" was stated more firmly than Step 4's own writeup supported
("strong candidate, not proven as the sole cause"), and
`phase4_shared_game_factor.py`'s shared-quantile coupling is PERFECT rank
correlation (rho=1), not the realistic ~0.15-0.3 real teammates show -- so
that negative result doesn't actually rule correlation in or out either
way. "Diagnose first, don't build a third variant yet." OFFLINE ONLY.

Three diagnostics, each isolating one specific question, all on the real
47-week backtest (not a small proxy sample):

1. DECOMPOSE THE MISS. For every synthetic matchup (Phase 3's recentered,
   fully INDEPENDENT sampling -- not Phase 4's correlated version), is the
   model wrong about the MEAN or the VARIANCE? Two different diseases, two
   different cures:
   (a) slope of actual margin on predicted margin (OLS) -- well below 1
       means the MEANS are overstated (a shrinkage/winner's-curse problem;
       the fix is shrinking projections toward a prior, not touching the
       simulated distribution at all).
   (b) Var(z), z = (actual_margin - predicted_margin) / simulated_margin_SD
       -- well above 1 means the simulated VARIANCE really is understated.
2. VERIFY THE CANCELLATION MECHANISM DIRECTLY. On the same matchups,
   compare margin SD under independent vs. game-level shared-quantile
   sampling, split by how many REAL games are shared across the two
   fantasy rosters (0 is a natural control group -- margin SD there must
   be identical between the two schemes by construction, since no shared
   quantile ever applies). If margin SD shrinks more as the shared-game
   count rises, the cancellation story holds up on its own terms; if not,
   it doesn't.
3. PER-PLAYER INTERVAL COVERAGE AT FULL SCALE, BY POSITION. The designer's
   own quick check (history.json, 266 players, one target week, 7-games-
   predict-8th proxy) suggested per-player spread looks roughly calibrated.
   Repeated here properly: the REAL as-of-week no-leakage bootstrap
   (build_history, same as everywhere else in this project), across all 47
   backtest weeks, by position, using a Student-t predictive interval (same
   method the designer used, for a direct comparison).

No new Monte Carlo variant is attempted here -- this is measurement.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_diagnostics.py
"""

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl
from scipy import stats

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
from phase2_game_environment import baseline_c_points, build_environment_factors  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402
from phase3_calibration_check import recentered_distribution  # noqa: E402
from phase4_shared_game_factor import game_key  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_DIAGNOSTICS_REPORT.md"  # gitignored
TRIALS = 5000
MATCHUPS_PER_WEEK = 60
RNG_SEEDS = [20261004, 777]
MIN_GAMES_FOR_INTERVAL = 4  # need a non-degenerate sample to fit a t-interval


def matchup_totals(ids, baseline_points, history, dist_cache, rng, trials=TRIALS):
    total = np.zeros(trials)
    for entity_id in ids:
        dist = recentered_distribution(entity_id, baseline_points, history, dist_cache)
        total += rng.choice(dist, size=trials, replace=True)
    return total


def matchup_totals_correlated(entries, baseline_points, history, dist_cache, game_quantile, rng, trials=TRIALS):
    total = np.zeros(trials)
    for entity_id, pos, key in entries:
        dist = recentered_distribution(entity_id, baseline_points, history, dist_cache)
        if key in game_quantile:
            quantile = (1 - game_quantile[key]) if pos == "DEF" else game_quantile[key]
            sorted_dist = np.sort(dist)
            idx = np.clip((quantile * len(sorted_dist)).astype(int), 0, len(sorted_dist) - 1)
            total += sorted_dist[idx]
        else:
            total += rng.choice(dist, size=trials, replace=True)
    return total


def run_matchup_diagnostics(targets, non_kickers, kickers, team_stats, opp, env_opponents, seed):
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    rows = []

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
        for _ in range(MATCHUPS_PER_WEEK):
            drawn = draw_synthetic_matchup(pool, rng_py)
            if drawn is None:
                continue
            team_a, team_b = drawn

            entries_a = [(eid, projections[eid]["position"],
                          game_key(env_opponents, season, week, projections[eid].get("team"))) for eid in team_a]
            entries_b = [(eid, projections[eid]["position"],
                          game_key(env_opponents, season, week, projections[eid].get("team"))) for eid in team_b]

            my_ind = matchup_totals(team_a, baseline_points, history, dist_cache, rng_np)
            opp_ind = matchup_totals(team_b, baseline_points, history, dist_cache, rng_np)
            margin_sd_independent = float(np.std(my_ind - opp_ind))

            games_a = {key for (_, _, key) in entries_a if key is not None}
            games_b = {key for (_, _, key) in entries_b if key is not None}
            shared_game_count = len(games_a & games_b)
            all_games = games_a | games_b
            game_quantile = {key: rng_np.uniform(0.0, 1.0, size=TRIALS) for key in all_games}
            my_corr = matchup_totals_correlated(entries_a, baseline_points, history, dist_cache, game_quantile, rng_np)
            opp_corr = matchup_totals_correlated(entries_b, baseline_points, history, dist_cache, game_quantile, rng_np)
            margin_sd_correlated = float(np.std(my_corr - opp_corr))

            predicted_margin = (sum(baseline_points.get(eid, 0.0) for eid in team_a)
                                - sum(baseline_points.get(eid, 0.0) for eid in team_b))
            actual_margin = (sum(compute_points(actuals[p]) for p in team_a)
                              - sum(compute_points(actuals[p]) for p in team_b))

            rows.append({
                "predicted_margin": predicted_margin, "actual_margin": actual_margin,
                "margin_sd_independent": margin_sd_independent, "margin_sd_correlated": margin_sd_correlated,
                "shared_game_count": shared_game_count,
            })
        print(f"  done: season {season} week {week}", end="\r")

    print()
    return rows


SD_FLOOR = 5.0  # points of margin SD -- see diagnostic_1()'s own comment


def diagnostic_1(rows) -> str:
    predicted = np.array([r["predicted_margin"] for r in rows])
    actual = np.array([r["actual_margin"] for r in rows])
    sd_ind = np.array([r["margin_sd_independent"] for r in rows])

    slope, intercept = np.polyfit(predicted, actual, 1)
    corr = np.corrcoef(predicted, actual)[0, 1]

    percentiles = np.percentile(sd_ind, [0, 1, 5, 25, 50])
    print(f"  margin_sd_independent percentiles (0/1/5/25/50): "
          f"{percentiles[0]:.3f} / {percentiles[1]:.3f} / {percentiles[2]:.3f} / {percentiles[3]:.3f} / {percentiles[4]:.3f}")

    # z = (actual-predicted)/sd_ind blows up for any matchup near-deterministic
    # enough that sd_ind is tiny (e.g. most of the 18 drafted players are
    # marginal bench-type guys who scored exactly 0.0 every one of their last
    # 8 games -- a real, legitimate zero-variance history, not a bug, but an
    # uninformative data point for judging whether SIMULATED variance matches
    # REAL variance, since there's essentially no simulated variance to judge
    # there). Floored at 5 points of margin SD -- comfortably below the ~21-23
    # typical value seen in diagnostic 2, well above the degenerate near-zero
    # cases -- and the excluded count is reported, not hidden.
    informative = sd_ind >= SD_FLOOR
    excluded = int((~informative).sum())
    z = (actual[informative] - predicted[informative]) / sd_ind[informative]
    var_z = float(np.var(z))

    return (
        f"n = {len(rows)} synthetic matchups ({excluded} excluded from Var(z): margin_sd_independent "
        f"< {SD_FLOOR} points -- near-deterministic matchups with essentially no simulated uncertainty "
        f"to judge, typically because most drafted players that week had a degenerate all-zero history)\n"
        f"Slope of actual margin on predicted margin: {slope:.3f} (intercept {intercept:+.2f}, corr {corr:.3f})\n"
        f"  -> 1.0 = means well-calibrated; well below 1.0 = means overstated (shrinkage needed)\n"
        f"Var(z), z = (actual - predicted) / simulated_margin_SD: {var_z:.3f}  (n={len(z)})\n"
        f"  -> 1.0 = simulated variance well-calibrated; well above 1.0 = variance understated"
    )


def diagnostic_2(rows) -> str:
    buckets = {0: [], 1: [], "2+": []}
    for r in rows:
        key = r["shared_game_count"] if r["shared_game_count"] < 2 else "2+"
        buckets[key].append(r)
    lines = ["| Shared real games | n | Mean independent SD | Mean correlated SD | Ratio (corr/indep) |", "|---|---|---|---|---|"]
    for key, bucket in buckets.items():
        if not bucket:
            lines.append(f"| {key} | 0 | - | - | - |")
            continue
        mean_ind = sum(r["margin_sd_independent"] for r in bucket) / len(bucket)
        mean_corr = sum(r["margin_sd_correlated"] for r in bucket) / len(bucket)
        ratio = mean_corr / mean_ind if mean_ind else float("nan")
        lines.append(f"| {key} | {len(bucket)} | {mean_ind:.2f} | {mean_corr:.2f} | {ratio:.3f} |")
    return "\n".join(lines)


def run_coverage_diagnostic(targets, non_kickers, kickers, team_stats):
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
                n = len(points)
                mean, sd = points.mean(), points.std(ddof=1)
                actual_points = compute_points({c: row.get(c) for c in stat_columns})
                position = proj.get(entity_id, {}).get("position", "?")
                by_position.setdefault(position, []).append((mean, sd, n, actual_points))
        print(f"  done: season {season} week {week}", end="\r")
    print()
    return by_position


def coverage_table(by_position) -> str:
    lines = ["| Position | n player-weeks | 50% interval coverage | 80% interval coverage |", "|---|---|---|---|"]
    all_rows = []
    for position in sorted(by_position):
        rows = by_position[position]
        all_rows.extend(rows)
        cov50 = cov80 = 0
        for mean, sd, n, actual in rows:
            if sd == 0:
                cov50 += 1 if actual == mean else 0
                cov80 += 1 if actual == mean else 0
                continue
            se = sd * (1 + 1 / n) ** 0.5
            t50 = stats.t.ppf(0.75, df=n - 1)
            t80 = stats.t.ppf(0.90, df=n - 1)
            if abs(actual - mean) <= t50 * se:
                cov50 += 1
            if abs(actual - mean) <= t80 * se:
                cov80 += 1
        lines.append(f"| {position} | {len(rows)} | {cov50/len(rows):.1%} | {cov80/len(rows):.1%} |")

    cov50 = cov80 = 0
    for mean, sd, n, actual in all_rows:
        if sd == 0:
            continue
        se = sd * (1 + 1 / n) ** 0.5
        t50 = stats.t.ppf(0.75, df=n - 1)
        t80 = stats.t.ppf(0.90, df=n - 1)
        if abs(actual - mean) <= t50 * se:
            cov50 += 1
        if abs(actual - mean) <= t80 * se:
            cov80 += 1
    lines.append(f"| **Overall** | **{len(all_rows)}** | **{cov50/len(all_rows):.1%}** | **{cov80/len(all_rows):.1%}** |")
    lines.append("\n(nominal targets: 50.0% and 80.0%)")
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
    _, env_opponents = build_environment_factors(schedules)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} historical weeks available.")

    print("\n=== Diagnostics 1+2: matchup margin decomposition ===")
    all_rows = []
    for seed in RNG_SEEDS:
        print(f"\n--- Seed {seed} ---")
        rows = run_matchup_diagnostics(targets, non_kickers, kickers, team_stats, opp, env_opponents, seed)
        all_rows.extend(rows)

    d1 = diagnostic_1(all_rows)
    d2 = diagnostic_2(all_rows)
    print("\nDiagnostic 1 (decompose the miss):")
    print(d1)
    print("\nDiagnostic 2 (cancellation mechanism by shared-game count):")
    print(d2)

    print("\n=== Diagnostic 3: per-player interval coverage, by position ===")
    by_position = run_coverage_diagnostic(targets, non_kickers, kickers, team_stats)
    d3 = coverage_table(by_position)
    print(d3)

    report = f"""# Phase 4 Diagnostics Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
MODEL_ROADMAP.md and DESIGNER_RESPONSE.md's "diagnose first" request (2026-10-04).

## Diagnostic 1 — decompose the miss (means vs. variance)

{d1}

## Diagnostic 2 — verify the cancellation mechanism directly

{d2}

(0 shared real games is a natural control group: margin SD there should be identical between independent
and correlated sampling by construction, since no shared quantile ever applies to those matchups.)

## Diagnostic 3 — per-player interval coverage, by position (full 47-week backtest)

{d3}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
