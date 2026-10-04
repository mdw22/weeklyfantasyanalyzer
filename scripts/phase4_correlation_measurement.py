"""
phase4_correlation_measurement.py

Phase 4, correction 1's "measurement first" step, per the designer's
explicit instruction (2026-10-04): do NOT scope correlation asymmetrically
to dodge the cross-side cancellation found in phase4_diagnostics.py --
"two players in the same real game really do share that game's outcome,
whichever fantasy rosters they sit on." Instead of guessing a correlation
strength (rho 0.15-0.3), MEASURE it from the real backtest: standardized
residuals, grouped into the real relationships that matter, each with n and
a 95% CI, as the input to a future Gaussian copula (not built here -- this
script is read-only measurement, per the designer's own ordering: "(1) the
empirical correlation measurement can run alongside [Baseline D]").

Standardized residual for one player-week: r = (actual_points - baseline_c_
points) / bootstrap_sd, where bootstrap_sd is the same real last-8-games
sample std used throughout Phase 3/4 (diagnostic 3's own coverage check).
Player-weeks with fewer than MIN_GAMES_FOR_INTERVAL games of history are
excluded (same threshold as phase4_diagnostics.py, for a stable SD).

Relationships measured (21 total), each as its own Pearson r with n and a
Fisher-z 95% CI:
- Same real team, by position pair: QB-RB, QB-WR, QB-TE, RB-RB, RB-WR,
  RB-TE, WR-WR, WR-TE, TE-TE (9). Multiple same-position entities on one
  team-week (e.g. 3 WRs) contribute all pairwise combinations, not just one
  "WR1" -- simpler, more data, at the cost of some non-independence across
  pairs sharing one teammate's residual (flagged, not hidden).
- Opposing offenses in the same real game, same position vs. same position:
  QB-vs-oppQB, RB-vs-oppRB, WR-vs-oppWR, TE-vs-oppTE (4) -- the classic
  "shootout" signal.
- Offense vs. own DEF: QB/RB/WR/TE vs. that SAME team's DEF (4).
- Offense vs. opposing DEF: QB/RB/WR/TE vs. the OPPONENT's DEF (4) -- the
  classic "a good defense suppresses the opposing offense" signal.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_correlation_measurement.py
"""

import sys
from datetime import datetime, timezone
from itertools import combinations, product
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
from phase2_game_environment import baseline_c_points, build_environment_factors  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_CORRELATION_MEASUREMENT.md"  # gitignored
MIN_GAMES_FOR_INTERVAL = 4

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
SAME_TEAM_PAIRS = list(combinations(SKILL_POSITIONS, 2)) + [(p, p) for p in SKILL_POSITIONS]


def fisher_ci(r: float, n: int):
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return (lo, hi)


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
    is_skill = pl.col("position").is_in(SKILL_POSITIONS)
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

    # Accumulated across the whole backtest: {relationship_name: [(r_a, r_b), ...]}
    pair_residuals = {f"{p1}-{p2}": [] for p1, p2 in SAME_TEAM_PAIRS}
    for p in SKILL_POSITIONS:
        pair_residuals[f"{p}-vs-opp{p}"] = []
        pair_residuals[f"{p}-vs-ownDEF"] = []
        pair_residuals[f"{p}-vs-oppDEF"] = []

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

        # Per-team-per-position lists of standardized residuals, this one week.
        by_team = {}  # team -> {position: [r, r, ...]}
        for entity_id, proj in projections.items():
            actual_row = actuals.get(entity_id)
            games = history.get(entity_id)
            if actual_row is None or not games or len(games) < MIN_GAMES_FOR_INTERVAL:
                continue
            points = np.array([compute_points(g) for g in games])
            sd = points.std(ddof=1)
            if not sd:
                continue
            baseline_pts = baseline_c_points(entity_id, proj, opp_map)
            actual_pts = compute_points(actual_row)
            r = (actual_pts - baseline_pts) / sd
            team = proj.get("team")
            position = proj["position"]
            if team:
                by_team.setdefault(team, {}).setdefault(position, []).append(r)

        # Same-team pairs.
        for team, by_pos in by_team.items():
            for p1, p2 in SAME_TEAM_PAIRS:
                list1, list2 = by_pos.get(p1, []), by_pos.get(p2, [])
                if not list1 or not list2:
                    continue
                key = f"{p1}-{p2}"
                if p1 == p2:
                    pair_residuals[key].extend(combinations(list1, 2))
                else:
                    pair_residuals[key].extend(product(list1, list2))

        # Opposing offenses / own-DEF / opposing-DEF.
        for team, by_pos in by_team.items():
            opponent = env_opponents.get((season, week, team))
            opp_by_pos = by_team.get(opponent, {}) if opponent else {}
            for p in SKILL_POSITIONS:
                mine = by_pos.get(p, [])
                if not mine:
                    continue
                theirs = opp_by_pos.get(p, [])
                if theirs:
                    pair_residuals[f"{p}-vs-opp{p}"].extend(product(mine, theirs))
                own_def = by_pos.get("DEF", [])
                if own_def:
                    pair_residuals[f"{p}-vs-ownDEF"].extend(product(mine, own_def))
                opp_def = opp_by_pos.get("DEF", [])
                if opp_def:
                    pair_residuals[f"{p}-vs-oppDEF"].extend(product(mine, opp_def))
        print(f"  done: season {season} week {week}", end="\r")

    print()
    lines = ["| Relationship | n pairs | Pearson r | 95% CI |", "|---|---|---|---|"]
    for key, pairs in pair_residuals.items():
        n = len(pairs)
        if n < 4:
            lines.append(f"| {key} | {n} | - | - |")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        r = float(np.corrcoef(a, b)[0, 1])
        lo, hi = fisher_ci(r, n)
        lines.append(f"| {key} | {n} | {r:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    table = "\n".join(lines)
    print(table)

    report = f"""# Phase 4 Correlation Measurement Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE, read-only measurement only --
builds no copula and changes no model. See DESIGNER_RESPONSE.md's correction 1 (2026-10-04).

**Standardized residual**: r = (actual_points - baseline_c_points) / bootstrap_sd, bootstrap_sd from the
same real last-8-games sample used throughout Phase 3/4. Player-weeks with fewer than {MIN_GAMES_FOR_INTERVAL}
games of history excluded (unstable SD otherwise).

**Caveat on same-position pairs (RB-RB, WR-WR, TE-TE) and any team-week with 3+ same-position entities**:
all pairwise combinations are included, not just one "starter" pair -- more data, but pairs sharing one
teammate's residual aren't independent of each other. Treat n as an upper bound on independent information,
not a literal sample size.

{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
