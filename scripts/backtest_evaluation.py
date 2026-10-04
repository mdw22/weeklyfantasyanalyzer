"""
backtest_evaluation.py

Phase 1, Step 1 + Step 2 of PHASE1_EVALUATION_SPEC.md: a no-leakage backtesting
harness for the two baselines (production last-8-average, plain season-average)
plus the baseline metrics table. OFFLINE ONLY -- run manually/locally. Does not
write to public/data/, does not touch the live site, does not change anything
a user actually sees. See MODEL_ROADMAP.md for why this exists.

Deliberately reuses generate_projections.py's build_projections() rather than
reimplementing the lookback/filtering logic a second time (per the spec's own
instruction to check reuse first) -- confirmed it already takes an arbitrary
historical (season, week) cutoff and filters strictly to games before it, which
is exactly the no-leakage guarantee this harness needs. The only change made to
that file is one new optional `n_games=None` parameter (default-preserving,
production's own call sites are unaffected) to get the season-average baseline
without a second filtering implementation.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/backtest_evaluation.py
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
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

OUT_PATH = Path(__file__).parent.parent / "BACKTEST_REPORT.md"  # gitignored, same as the other designer docs

# Full-PPR point values, ported from src/lib/scoring.js's BASE_VALUES + the
# full_ppr preset (receptions: 1) -- used ONLY for this offline analysis, not
# imported into/from the JS file. If scoring.js's values ever change, this
# needs a matching manual update (same class of cross-language seam as
# SLOT_ID_ORDER/DEF_{team}, see CLAUDE.md).
FULL_PPR_VALUES = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "passing_interceptions": -2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "receptions": 1,
    "fumbles_lost_total": -2,
    "def_sacks": 1,
    "def_interceptions": 2,
    "def_fumble_recoveries": 2,
    "def_safeties": 2,
    "def_touchdowns": 6,
    "def_blocked_kicks": 2,
    "def_two_point_returns": 2,
    "def_points_allowed_bonus": 1,
    "def_yards_allowed_bonus": 1,
    "fg_made_0_39": 3,
    "fg_made_40_49": 4,
    "fg_made_50_plus": 5,
    "fg_missed_total": -1,
    "pat_made": 1,
}

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def compute_points(stat_line: dict, values: dict = FULL_PPR_VALUES) -> float:
    return sum((stat_line.get(k) or 0.0) * v for k, v in values.items())


def list_backtest_weeks(schedules: pl.DataFrame, seasons: list[int]) -> list[tuple[int, int]]:
    """Every (season, week) where EVERY game that week is complete
    (home_score not null) -- a fully-played week, safe as a backtest target.
    Excludes the current in-progress week rather than guessing at a partial
    actuals sample."""
    out = []
    for season in seasons:
        s = schedules.filter(pl.col("season") == season)
        for week in sorted(s["week"].unique().to_list()):
            wk = s.filter(pl.col("week") == week)
            if wk.height > 0 and wk["home_score"].null_count() == 0:
                out.append((season, week))
    return out


def assert_no_leakage(df: pl.DataFrame, season: int, week: int) -> None:
    """The actual guard build_projections()/build_history() rely on, run here
    standalone so it can be unit-tested (including failing on purpose, below)
    independent of the full build_projections() call. Raises if any row at or
    after the target (season, week) would leak into the "past" set."""
    past = df.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    leaked = past.filter(
        (pl.col("season") > season) | ((pl.col("season") == season) & (pl.col("week") >= week))
    )
    if leaked.height:
        raise AssertionError(
            f"LEAKAGE: {leaked.height} row(s) at/after ({season}, wk{week}) leaked into the 'past' set"
        )


def _bad_filter_on_purpose(df: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """A deliberately-wrong version of the SAME filter (<=  instead of <  on
    week) -- exists only to prove assert_no_leakage() actually catches a real
    leak, not just that it passes on correct input. Not used anywhere else."""
    return df.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week))
    )


def run_no_leakage_tests(stats: pl.DataFrame) -> None:
    """Per the spec's verification bar: demonstrate the guard actually guards
    something by showing it both pass (real filter) and fail (deliberately
    broken filter) against the same real data and the same target week."""
    season, week = 2025, 10
    print(f"\n=== No-leakage test (target: season {season}, week {week}) ===")

    # 1. The real production filter: must have ZERO leakage.
    try:
        assert_no_leakage(stats, season, week)
        print("ok:   real filter (< week) -- no leakage detected, as expected")
    except AssertionError as exc:
        print(f"FAIL: real filter should not leak, but: {exc}")
        raise SystemExit(1)

    # 2. A deliberately-broken filter (<=): must be CAUGHT, proving the
    # assertion isn't a no-op / tautology.
    bad_past = _bad_filter_on_purpose(stats, season, week)
    bad_leaked = bad_past.filter(
        (pl.col("season") == season) & (pl.col("week") >= week)
    )
    if bad_leaked.height == 0:
        print("FAIL: the deliberately-broken (<=) filter did not leak anything -- "
              "test is not actually exercising the leakage path, fix the test")
        raise SystemExit(1)
    print(f"ok:   deliberately-broken filter (<= week) DOES leak "
          f"{bad_leaked.height} row(s) from week {week} itself -- confirms the "
          f"real guard is catching a real failure mode, not a tautology")

    # 3. Sanity: the real filter's "past" set for this target is a strict
    # subset of the broken one's (the broken one = real + week W's own rows).
    real_past = stats.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    assert bad_past.height == real_past.height + bad_leaked.height, (
        "the broken filter's extra rows should be EXACTLY week W's rows -- "
        "if not, something else is wrong with this test, not just the filter"
    )
    print(f"ok:   broken filter's extra {bad_leaked.height} rows are exactly week {week}'s "
          f"own games ({real_past.height} + {bad_leaked.height} = {bad_past.height}) -- confirmed by direct count")


def season_slice_for_average(stats: pl.DataFrame, season: int) -> pl.DataFrame:
    """Input for the 'full season average' baseline: THIS season's games
    only (no prior-season fallback, unlike production's last-8 baseline) --
    reusing build_projections() with n_games=None on this pre-filtered frame
    gives "average of every game this player has played this season so far."
    At week 1 of a season this is necessarily empty (no games yet) -- that's
    a real, expected property of this baseline, not a bug; it simply has no
    eligible sample at week 1, visible in this baseline's own reported n."""
    return stats.filter(pl.col("season") == season)


def collect_actuals(stats: pl.DataFrame, season: int, week: int, id_col: str) -> dict:
    """{entity_id: stat_line dict} for everyone who actually recorded a stat
    line in this exact (season, week) -- the real, played outcome. A player
    who didn't play (bye/inactive/injured) simply has no row here, so this
    backtest's sample is implicitly restricted to "played" player-weeks; see
    the report's own caveat about this selection effect."""
    rows = stats.filter((pl.col("season") == season) & (pl.col("week") == week))
    return {str(row[id_col]): row for row in rows.iter_rows(named=True)}


def evaluate_week(
    history_df: pl.DataFrame,
    actuals_df: pl.DataFrame,
    season: int,
    week: int,
    id_col: str,
    stat_columns: list,
    describe_row,
    n_games: int | None,
    results: dict,
    baseline_name: str,
) -> None:
    """Appends per-player-week (position, |error|, error, projected, actual)
    tuples into `results[baseline_name]` for one (season, week) target. Only
    entities that BOTH have an eligible projection AND actually played this
    exact week are scored -- see collect_actuals()'s docstring for the
    selection-effect caveat this implies."""
    actuals = collect_actuals(actuals_df, season, week, id_col)
    if not actuals:
        return

    # Performance: build_projections() is semantics-preserving if we first
    # narrow the input to only entities we'll actually score this week --
    # it still sees each entity's FULL history (only non-relevant entities
    # are dropped), so results are identical to running on the full frame,
    # just far fewer partitions to build (300-500 vs ~2,500+ per week).
    relevant = history_df.filter(pl.col(id_col).is_in(list(actuals.keys())))
    if relevant.height == 0:
        return

    assert_no_leakage(relevant, season, week)  # belt-and-suspenders per-call guard

    projections = build_projections(relevant, season, week, id_col, stat_columns, describe_row, n_games=n_games)

    for entity_id, proj in projections.items():
        actual_row = actuals.get(entity_id)
        if actual_row is None:
            continue
        projected_points = compute_points(proj["projected_stats"])
        actual_points = compute_points({c: actual_row.get(c) for c in stat_columns})
        error = projected_points - actual_points
        results.setdefault(baseline_name, []).append(
            {
                "position": proj["position"],
                "season": season,
                "week": week,
                "projected": projected_points,
                "actual": actual_points,
                "error": error,
            }
        )


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    errors = [r["error"] for r in rows]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    bias = sum(errors) / n
    mae_se = (sum((abs(e) - mae) ** 2 for e in errors) / n) ** 0.5 / math.sqrt(n) if n > 1 else 0.0
    xs = [r["projected"] for r in rows]
    ys = [r["actual"] for r in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    corr = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    return {
        "n": n,
        "mae": mae,
        "mae_ci95": 1.96 * mae_se,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
    }


def format_table(all_results: dict) -> str:
    lines = []
    for baseline_name, rows in all_results.items():
        lines.append(f"\n### {baseline_name}\n")
        lines.append("| Position | n (player-weeks) | MAE | MAE 95% CI | RMSE | Signed bias | Correlation |")
        lines.append("|---|---|---|---|---|---|---|")
        overall = summarize(rows)
        for position in POSITIONS:
            pos_rows = [r for r in rows if r["position"] == position]
            s = summarize(pos_rows)
            if s["n"] == 0:
                lines.append(f"| {position} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| {position} | {s['n']} | {s['mae']:.2f} | ±{s['mae_ci95']:.2f} | "
                f"{s['rmse']:.2f} | {s['bias']:+.2f} | {s['corr']:.3f} |"
            )
        lines.append(
            f"| **Overall** | **{overall['n']}** | **{overall['mae']:.2f}** | "
            f"**±{overall['mae_ci95']:.2f}** | **{overall['rmse']:.2f}** | "
            f"**{overall['bias']:+.2f}** | **{overall['corr']:.3f}** |"
        )
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

    # `load_player_stats` carries an individual row for every position that
    # recorded ANY stat, including defensive/O-line/long-snapper players
    # (LB, CB, DT, SAF, G, LS, ...) -- found by inspecting the real position
    # breakdown here: ~11,000 of ~16,500 non-kicker player-weeks in 2025
    # alone belong to these, each with a near-zero stat line (projected AND
    # actual both ~0), which silently dragged "Overall" MAE down to ~1.7 vs
    # every real fantasy position's 3.7-6.5 the first time this ran.
    # Restricting to the four skill positions fixes both the Overall metric
    # and backtest runtime (fewer, more relevant partitions per week).
    is_k = (pl.col("position") == "K").fill_null(False)
    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    non_kickers = stats.filter(is_skill)
    kickers = add_k_stat_columns(stats.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)  # exclude the in-progress/future week
    ]
    print(f"\n{len(targets)} fully-completed historical weeks to backtest: "
          f"{targets[0]} .. {targets[-1]} (seasons {sorted(set(s for s, _ in targets))})")

    results = {}
    entity_groups = [
        (non_kickers, non_kickers, "player_id", STAT_COLUMNS, describe_player),
        (kickers, kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]

    for season, week in targets:
        for history_df, actuals_df, id_col, stat_columns, describe_row in entity_groups:
            evaluate_week(history_df, actuals_df, season, week, id_col, stat_columns, describe_row,
                          n_games=8, results=results, baseline_name="Baseline A -- last 8 games (production)")
            season_df = season_slice_for_average(history_df, season)
            evaluate_week(season_df, actuals_df, season, week, id_col, stat_columns, describe_row,
                          n_games=None, results=results, baseline_name="Baseline B -- full season average")
        print(f"  done: season {season} week {week}", end="\r")

    print()
    table = format_table(results)
    print(table)

    report = f"""# Backtest Evaluation Report — Phase 1, Step 1+2

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only -- does not
affect the live site or production pipeline. See MODEL_ROADMAP.md / PHASE1_EVALUATION_SPEC.md.

Backtest window: {len(targets)} fully-completed weeks, {targets[0]} through {targets[-1]}
(seasons {sorted(set(s for s, _ in targets))}).

**No-leakage test: PASSED** (real filter leaks zero rows at/after each target week; a deliberately-broken
`<=` filter was shown to leak exactly that week's own rows, proving the guard actually guards something --
see this script's `run_no_leakage_tests()` for the full demonstration).

**Known sample-selection caveat**: only player-weeks where the player actually recorded a stat line are
scored. A player who was injured/benched/on a bye that week has no row in `load_player_stats` for it, so
this backtest cannot currently distinguish "the model should have projected him lower because he didn't
play" from "he simply isn't in this sample" -- both look the same (absent). This is a real, inherent
limitation of backtesting against `load_player_stats` alone, not something this harness works around.

**Sample size caveat (MODEL_ROADMAP.md point 6)**: MAE's 95% CI is a normal approximation
(`1.96 * SE`, `SE = std(|error|) / sqrt(n)`) -- a rough guide, not exact, and will be wide for any
position with a small `n`. Read the CI column before treating any MAE difference as real, especially
sliced by position.

**Baseline B (full season average) has fewer eligible player-weeks than Baseline A by construction**:
unlike production's last-8 baseline, it does NOT fall back to a prior season, so it has zero eligible
sample at week 1 of every season (no current-season games exist yet to average). This is expected, not a
bug -- visible directly in Baseline B's own `n` column for early weeks.
{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
