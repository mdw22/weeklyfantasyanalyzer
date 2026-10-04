"""
phase4_baseline_d_ablation.py

Phase 4, Baseline D bias ablation (2026-10-04), independent of the
predictive-scale/variance-pooling work (same no-leakage harness, no
Monte Carlo needed -- this is a backtest-level comparison only). The
original Baseline D (mean_actual_prior + beta*(bc - mean_bc_prior)) showed
a new +0.14..+0.29 positive bias (QB/RB/WR/TE) traced to the intercept
term (mean_actual_prior - mean_bc_prior) not carrying forward perfectly
out-of-sample. Three variants compared here, backtested by position AND
by season (to see whether the bias is concentrated in one season or
persistent across all three):

  D  (original):  mean_actual_prior[pos] + beta*(bc - mean_bc_prior[pos])
  D-slope-only:    mean_bc_prior[pos]     + beta*(bc - mean_bc_prior[pos])
                   (drops the actual-vs-bc intercept shift entirely --
                   pure regression-to-the-mean on bc's own scale, no
                   historical-miscalibration correction)
  D-rolling:       same formula as D, but beta/means fit on a ROLLING
                   window of the last 8 backtest weeks only (not the full
                   expanding window) -- may track any within-season drift
                   more responsively than an ever-growing window.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_baseline_d_ablation.py
"""

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
    list_backtest_weeks,
    run_no_leakage_tests,
    summarize,
)
from generate_projections import (  # noqa: E402
    DEF_STAT_COLUMNS,
    K_STAT_COLUMNS,
    LOOKBACK_SEASONS,
    STAT_COLUMNS,
    add_def_stat_columns,
    add_k_stat_columns,
    build_projections,
    current_season,
    describe_defense,
    describe_player,
    get_current_season_and_week,
)
from phase2_game_environment import baseline_c_points  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402
from phase4_baseline_d import MIN_SAMPLE  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_BASELINE_D_ABLATION_REPORT.md"  # gitignored
ROLLING_WEEKS = 8
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def fit_params(pairs):
    if len(pairs) < MIN_SAMPLE:
        return (1.0, 0.0, 0.0, True)
    bc = np.array([p[0] for p in pairs])
    actual = np.array([p[1] for p in pairs])
    beta, _intercept = np.polyfit(bc, actual, 1)
    return (float(beta), float(actual.mean()), float(bc.mean()), False)


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

    expanding_prior = {}
    rolling_prior = {}  # position -> deque of (bc, actual), maxlen = ROLLING_WEEKS worth of weeks' pairs
    rolling_week_sizes = {}  # position -> deque of per-week pair counts, to pop exactly ROLLING_WEEKS weeks back

    results = {"D (original)": [], "D-slope-only": [], "D-rolling": []}
    results_by_season = {"D (original)": {}, "D-slope-only": {}, "D-rolling": {}}

    for season, week in targets:
        actuals, projections = {}, {}
        for df, id_col, stat_columns, describe_row in entity_groups:
            week_actuals = collect_actuals(df, season, week, id_col)
            if not week_actuals:
                continue
            relevant = df.filter(pl.col(id_col).is_in(list(week_actuals.keys())))
            if relevant.height == 0:
                continue
            assert_no_leakage(relevant, season, week)
            proj = build_projections(relevant, season, week, id_col, stat_columns, describe_row, n_games=8)
            for entity_id, row in week_actuals.items():
                actuals[entity_id] = {c: row.get(c) for c in stat_columns}
            projections.update(proj)

        relevant_opp = opp.filter(pl.col("player_id").is_in(list(actuals.keys())))
        opp_map = {}
        if relevant_opp.height > 0:
            assert_no_leakage(relevant_opp, season, week)
            opp_map = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)

        baseline_c = {eid: baseline_c_points(eid, proj, opp_map) for eid, proj in projections.items()}

        expanding_params = {pos: fit_params(expanding_prior.get(pos, [])) for pos in POSITIONS}
        rolling_params = {pos: fit_params(list(rolling_prior.get(pos, []))) for pos in POSITIONS}

        this_week_pairs = {pos: [] for pos in POSITIONS}
        for eid, proj in projections.items():
            actual_row = actuals.get(eid)
            if actual_row is None:
                continue
            position = proj["position"]
            actual_pts = compute_points(actual_row)
            bc_val = baseline_c[eid]

            beta_e, mean_a_e, mean_bc_e, fb_e = expanding_params.get(position, (1.0, 0.0, 0.0, True))
            d_original = bc_val if fb_e else (mean_a_e + beta_e * (bc_val - mean_bc_e))
            d_slope_only = bc_val if fb_e else (mean_bc_e + beta_e * (bc_val - mean_bc_e))

            beta_r, mean_a_r, mean_bc_r, fb_r = rolling_params.get(position, (1.0, 0.0, 0.0, True))
            d_rolling = bc_val if fb_r else (mean_a_r + beta_r * (bc_val - mean_bc_r))

            for name, projected in [("D (original)", d_original), ("D-slope-only", d_slope_only), ("D-rolling", d_rolling)]:
                row = {"position": position, "season": season, "week": week,
                       "projected": projected, "actual": actual_pts, "error": projected - actual_pts}
                results[name].append(row)
                results_by_season[name].setdefault(season, []).append(row)

            this_week_pairs[position].append((bc_val, actual_pts))
            expanding_prior.setdefault(position, []).append((bc_val, actual_pts))

        for pos in POSITIONS:
            rolling_prior.setdefault(pos, deque())
            rolling_week_sizes.setdefault(pos, deque())
            rolling_prior[pos].extend(this_week_pairs[pos])
            rolling_week_sizes[pos].append(len(this_week_pairs[pos]))
            while len(rolling_week_sizes[pos]) > ROLLING_WEEKS:
                drop = rolling_week_sizes[pos].popleft()
                for _ in range(drop):
                    rolling_prior[pos].popleft()
        print(f"  done: season {season} week {week}", end="\r")
    print()

    lines = []
    for name, rows in results.items():
        lines.append(f"\n### {name}\n")
        lines.append("| Position | n | MAE | RMSE | Signed bias |")
        lines.append("|---|---|---|---|---|")
        overall = summarize(rows)
        for position in POSITIONS:
            pos_rows = [r for r in rows if r["position"] == position]
            s = summarize(pos_rows)
            if s["n"] == 0:
                lines.append(f"| {position} | 0 | - | - | - |")
                continue
            lines.append(f"| {position} | {s['n']} | {s['mae']:.2f} | {s['rmse']:.2f} | {s['bias']:+.2f} |")
        lines.append(f"| **Overall** | **{overall['n']}** | **{overall['mae']:.2f}** | **{overall['rmse']:.2f}** | **{overall['bias']:+.2f}** |")

        lines.append("\nBias by season:")
        lines.append("| Season | n | Signed bias |")
        lines.append("|---|---|---|")
        for s_year in sorted(results_by_season[name]):
            season_rows = results_by_season[name][s_year]
            s_sum = summarize(season_rows)
            lines.append(f"| {s_year} | {s_sum['n']} | {s_sum['bias']:+.2f} |")

    table = "\n".join(lines)
    print(table)

    report = f"""# Phase 4 Baseline D Bias Ablation Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only. See
DESIGNER_RESPONSE.md (2026-10-04).

Backtest-level comparison only (no Monte Carlo rerun) -- picks a candidate formulation before any further
calibration validation.
{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
