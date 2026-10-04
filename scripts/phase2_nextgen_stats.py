"""
phase2_nextgen_stats.py

Phase 2's third candidate (the designer's pick after Candidate 2's clean
negative: move to EPA/Next Gen Stats next). OFFLINE ONLY.

Scope, deliberately narrow for a first cheap test: RB rushing and WR/TE
receiving efficiency only, via real NFL Next Gen Stats tracking data
(`load_nextgen_stats`). QB is deferred -- CPOE (completion % over
expected) is a RATE on a percentage-point scale, not yardage, and turning
it into a projected-yards adjustment cleanly needs its own two-step design
(completion rate -> yardage) that isn't a "cheap" first test. K/DEF have
no analogous NGS model at all (same gap as the opportunity blend).

Real schema gotcha found and worth remembering: `load_nextgen_stats()`
mixes week=0 SEASON-AGGREGATE rows into the same table as real per-week
rows (51 of 648 rows in a real 2025 pull, with e.g. `rush_attempts=242` --
obviously a season total, not one game). Filtered out (`week > 0`) before
any per-game analysis, or a single bogus "season-total" row would badly
corrupt a last-8-games average the same way Step 1/2's position-filter bug
did to the very first backtest run.

Design -- NOT a naive "add the over-expected number on top of actual
yards" (that would double-count: a player's own real last-8
`rushing_yards` already includes however much he over/under-performed
per-play expectation that game, since that's literally how the NGS metric
is defined). Instead, an INDEPENDENT candidate projection, built the same
way Candidate 1's opportunity-based one was, and BLENDED (never added) with
Baseline C:
  league_rate = league-wide yards-per-touch, computed with the SAME strict
    no-leakage "past" filter used everywhere else in this project (not a
    whole-window average, which would be a mild look-ahead on a slowly-
    moving constant)
  expected_yards_per_touch = league_rate + player's own last-8-average
    over-expected-per-touch NGS metric
  expected_total_yards = expected_yards_per_touch * BASELINE's own
    projected touch volume (carries/receptions) -- only the per-touch RATE
    is NGS-adjusted, volume still comes from the existing last-8 average
  this candidate's points = compute_points() on Baseline C's own stat line
    with ONLY that one yardage field swapped for the NGS-derived estimate
Blended with Baseline C: `adjusted = (1-weight)*baseline + weight*ngs`,
swept 0/25/50/75/100%, identical structure to Candidate 1 (not Candidate
2's flat scalar, which was a clean negative -- see MODEL_ROADMAP.md).

Reuses generate_projections.py's build_projections() (for both the
baseline stat line AND the NGS metric's own last-8 average -- same
function, same no-leakage guard, two different inputs) and
backtest_evaluation.py's compute_points/list_backtest_weeks/
assert_no_leakage/collect_actuals/run_no_leakage_tests/summarize/POSITIONS/
format_table. phase2_baseline_c.py's PER_POSITION_WEIGHTS/opportunity blend
for QB/TE is reused unchanged underneath (this candidate adjusts the
rushing/receiving YARDS field of Baseline C's own stat line, not a
competing computation).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase2_nextgen_stats.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import (  # noqa: E402
    assert_no_leakage,
    collect_actuals,
    compute_points,
    format_table,
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
from phase2_baseline_c import PER_POSITION_WEIGHTS  # noqa: E402
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE2_NEXTGEN_REPORT.md"  # gitignored
NGS_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

RUSH_RENAME = {"player_gsis_id": "player_id", "player_position": "position", "team_abbr": "team"}
REC_RENAME = {"player_gsis_id": "player_id", "player_position": "position", "team_abbr": "team"}


def load_ngs_frame(stat_type: str, rename: dict, seasons: list[int]) -> pl.DataFrame:
    return (
        nfl.load_nextgen_stats(stat_type=stat_type, seasons=seasons)
        .filter(pl.col("week") > 0)  # drop the mixed-in season-aggregate rows
        .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
        .rename(rename)
    )


def league_rate(df: pl.DataFrame, season: int, week: int, numer_col: str, denom_col: str) -> float | None:
    """Same strict no-leakage filter used everywhere else -- a league-wide
    rate computed only from games strictly before the target. Returns None
    (not 0.0) when there's no prior data at all -- the very first target
    week in the whole lookback window (2024 week 1) has an empty "past" by
    construction, and a real league yards-per-touch rate is never actually
    0 -- a caller treating None as "0.0" would silently project near-zero
    yards for every RB/WR/TE that one week. Callers fall back to the
    unadjusted baseline for that week instead, the same way missing NGS
    row coverage is already handled."""
    past = df.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    denom = past[denom_col].sum() or 0.0
    if not denom:
        return None
    numer = past[numer_col].sum() or 0.0
    return numer / denom


def baseline_c_points(entity_id, proj, opp_proj) -> tuple[float, dict]:
    """Baseline C's own blended points AND the stat line it was computed
    from (callers need the stat line to build the NGS candidate on top)."""
    stat_line = dict(proj["projected_stats"])
    weight = PER_POSITION_WEIGHTS.get(proj["position"], 0.0)
    opp_entry = opp_proj.get(entity_id) if opp_proj else None
    if weight > 0 and opp_entry is not None:
        baseline_points = compute_points(stat_line)
        opp_points = compute_points(opp_entry["projected_stats"])
        return (1 - weight) * baseline_points + weight * opp_points, stat_line
    return compute_points(stat_line), stat_line


def main() -> None:
    season = current_season()
    schedules_now = nfl.load_schedules(seasons=season)
    current_s, current_w = get_current_season_and_week(schedules_now)

    lookback_seasons = list(range(current_s - LOOKBACK_SEASONS, current_s + 1))
    print(f"Loading full historical data for seasons {lookback_seasons}...")

    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=lookback_seasons)
    run_no_leakage_tests(stats)

    is_k = (pl.col("position") == "K").fill_null(False)
    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    is_rb = pl.col("position") == "RB"
    is_wrte = pl.col("position").is_in(["WR", "TE"])
    skill_stats = stats.filter(is_skill)
    rb_stats = stats.filter(is_rb)
    wrte_stats = stats.filter(is_wrte)
    kickers = add_k_stat_columns(stats.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )
    opp = load_opportunity_frame(lookback_seasons)
    ngs_rush = load_ngs_frame("rushing", RUSH_RENAME, lookback_seasons)
    ngs_rec = load_ngs_frame("receiving", REC_RENAME, lookback_seasons)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} fully-completed historical weeks to backtest: "
          f"{targets[0]} .. {targets[-1]}")

    results = {w: [] for w in NGS_WEIGHTS}
    coverage = {"rb_scored": 0, "rb_with_ngs": 0, "wrte_scored": 0, "wrte_with_ngs": 0}

    for season, week in targets:
        skill_actuals = collect_actuals(skill_stats, season, week, "player_id")
        if not skill_actuals:
            continue
        relevant_stats = skill_stats.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
        relevant_opp = opp.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
        if relevant_stats.height == 0:
            continue
        assert_no_leakage(relevant_stats, season, week)
        if relevant_opp.height > 0:
            assert_no_leakage(relevant_opp, season, week)

        proj_map = build_projections(relevant_stats, season, week, "player_id", STAT_COLUMNS, describe_player, n_games=8)
        opp_map = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)

        relevant_rush_ngs = ngs_rush.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
        relevant_rec_ngs = ngs_rec.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
        rush_ngs_map, rec_ngs_map = {}, {}
        if relevant_rush_ngs.height > 0:
            assert_no_leakage(relevant_rush_ngs, season, week)
            rush_ngs_map = build_projections(relevant_rush_ngs, season, week, "player_id",
                                              ["rush_yards_over_expected_per_att"], describe_player, n_games=8)
        if relevant_rec_ngs.height > 0:
            assert_no_leakage(relevant_rec_ngs, season, week)
            rec_ngs_map = build_projections(relevant_rec_ngs, season, week, "player_id",
                                             ["avg_yac_above_expectation"], describe_player, n_games=8)

        league_ypc = league_rate(rb_stats, season, week, "rushing_yards", "carries")
        league_ypr = league_rate(wrte_stats, season, week, "receiving_yards", "receptions")

        for entity_id, proj in proj_map.items():
            actual_row = skill_actuals.get(entity_id)
            if actual_row is None:
                continue
            base_points, stat_line = baseline_c_points(entity_id, proj, opp_map)
            actual_points = compute_points({c: actual_row.get(c) for c in STAT_COLUMNS})

            ngs_points = None
            if proj["position"] == "RB":
                coverage["rb_scored"] += 1
                ngs_entry = rush_ngs_map.get(entity_id)
                if ngs_entry is not None and stat_line.get("carries", 0) and league_ypc is not None:
                    coverage["rb_with_ngs"] += 1
                    expected_ypc = league_ypc + ngs_entry["projected_stats"]["rush_yards_over_expected_per_att"]
                    ngs_line = {**stat_line, "rushing_yards": expected_ypc * stat_line["carries"]}
                    ngs_points = compute_points(ngs_line)
            elif proj["position"] in ("WR", "TE"):
                coverage["wrte_scored"] += 1
                ngs_entry = rec_ngs_map.get(entity_id)
                if ngs_entry is not None and stat_line.get("receptions", 0) and league_ypr is not None:
                    coverage["wrte_with_ngs"] += 1
                    expected_ypr = league_ypr + ngs_entry["projected_stats"]["avg_yac_above_expectation"]
                    ngs_line = {**stat_line, "receiving_yards": expected_ypr * stat_line["receptions"]}
                    ngs_points = compute_points(ngs_line)

            for weight in NGS_WEIGHTS:
                adjusted = base_points if ngs_points is None else (1 - weight) * base_points + weight * ngs_points
                results[weight].append({
                    "position": proj["position"], "season": season, "week": week,
                    "projected": adjusted, "actual": actual_points, "error": adjusted - actual_points,
                })

        # K/DEF: no NGS model exists -- Baseline C unchanged, same as the opportunity blend's own gap.
        k_actuals = collect_actuals(kickers, season, week, "player_id")
        if k_actuals:
            relevant_k = kickers.filter(pl.col("player_id").is_in(list(k_actuals.keys())))
            if relevant_k.height > 0:
                assert_no_leakage(relevant_k, season, week)
                k_proj = build_projections(relevant_k, season, week, "player_id", K_STAT_COLUMNS, describe_player, n_games=8)
                for entity_id, proj in k_proj.items():
                    actual_row = k_actuals.get(entity_id)
                    if actual_row is None:
                        continue
                    pts = compute_points(proj["projected_stats"])
                    actual_points = compute_points({c: actual_row.get(c) for c in K_STAT_COLUMNS})
                    for weight in NGS_WEIGHTS:
                        results[weight].append({"position": "K", "season": season, "week": week,
                                                 "projected": pts, "actual": actual_points, "error": pts - actual_points})

        def_actuals = collect_actuals(team_stats, season, week, "def_id")
        if def_actuals:
            relevant_def = team_stats.filter(pl.col("def_id").is_in(list(def_actuals.keys())))
            if relevant_def.height > 0:
                assert_no_leakage(relevant_def, season, week)
                def_proj = build_projections(relevant_def, season, week, "def_id", DEF_STAT_COLUMNS, describe_defense, n_games=8)
                for entity_id, proj in def_proj.items():
                    actual_row = def_actuals.get(entity_id)
                    if actual_row is None:
                        continue
                    pts = compute_points(proj["projected_stats"])
                    actual_points = compute_points({c: actual_row.get(c) for c in DEF_STAT_COLUMNS})
                    for weight in NGS_WEIGHTS:
                        results[weight].append({"position": "DEF", "season": season, "week": week,
                                                 "projected": pts, "actual": actual_points, "error": pts - actual_points})
        print(f"  done: season {season} week {week}", end="\r")

    print()
    rb_cov = coverage["rb_with_ngs"] / coverage["rb_scored"] if coverage["rb_scored"] else 0
    wrte_cov = coverage["wrte_with_ngs"] / coverage["wrte_scored"] if coverage["wrte_scored"] else 0
    print(f"\nNGS coverage: RB {coverage['rb_with_ngs']}/{coverage['rb_scored']} ({rb_cov:.1%}), "
          f"WR/TE {coverage['wrte_with_ngs']}/{coverage['wrte_scored']} ({wrte_cov:.1%})")

    labeled = {f"NGS weight = {w:.0%}": rows for w, rows in results.items()}
    table = format_table(labeled)
    print(table)

    report = f"""# Phase 2 Candidate: Next Gen Stats (RB/WR/TE efficiency) Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only.
See MODEL_ROADMAP.md.

**Scope**: RB rushing (`rush_yards_over_expected_per_att`) and WR/TE receiving
(`avg_yac_above_expectation`) only. QB deferred (CPOE is a rate, not yardage -- needs its own design).
K/DEF untouched (no NGS model exists for them).

**Coverage**: RB {coverage['rb_with_ngs']}/{coverage['rb_scored']} ({rb_cov:.1%}), WR/TE
{coverage['wrte_with_ngs']}/{coverage['wrte_scored']} ({wrte_cov:.1%}) scored player-weeks had an NGS row to
build a candidate from; the rest fall back to Baseline C unchanged (same sample-parity principle as
Baseline C itself).

**Builds on Baseline C**, not Baseline A -- weight 0% should reproduce Baseline C's own numbers.
{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
