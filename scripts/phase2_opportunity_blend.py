"""
phase2_opportunity_blend.py

Phase 2's first candidate per MODEL_ROADMAP.md point 6: blend the current
last-8-average projection (Phase 1's Baseline A) with an opportunity-based
"expected points" signal, averaged the same no-leakage way, and backtest the
blend against real history using Phase 1's own harness. OFFLINE ONLY -- does
not touch public/data/, the live site, or any production pipeline.

Deliberate refinement vs. the roadmap's literal wording ("blend with
load_ff_opportunity's total_fantasy_points_exp"): checked first, before
writing any blend code, and `load_ff_opportunity()`'s own `total_fantasy_points`
(actual, not even _exp) is NOT on this project's scoring scale. Confirmed by
diffing it against this project's own compute_points() on the same real 2025
week 10 games: RB/WR/TE matched exactly (both already full-PPR), but QBs
differed by EXACTLY 2 points per passing TD (nflverse scores passing TDs at
6, this project scores them at 4 -- see FULL_PPR_VALUES) plus 2 points per
2-point conversion, which `src/lib/scoring.js` doesn't implement at all (a
separate, real, pre-existing gap -- confirmed by reading it, not something to
silently patch inside this blend). Blending `total_fantasy_points_exp`
directly would inject a QB-specific, 2pt-conversion-specific bias that has
nothing to do with whether "opportunity" is actually a good signal. Fixed by
using nflverse's own per-stat EXPECTED columns (`pass_yards_gained_exp`,
`pass_touchdown_exp`, etc. -- the underlying usage-based stat predictions,
not nflverse's own point total), renamed to this project's own stat-column
names and scored through this project's own compute_points(), so the
opportunity candidate and the existing baselines are judged on the exact
same points scale.

Also a real schema gotcha worth flagging for next time (not in CLAUDE.md's
existing nflreadpy-gotchas list yet): `load_ff_opportunity()`'s `season`
column is a STRING and `week` is a FLOAT -- different dtypes than
`load_player_stats`/`load_team_stats`. Must cast before any join/filter
against those, or a polars ComputeError (not a silent wrong answer, at
least) is the result.

Coverage caveat, confirmed by running this: `load_ff_opportunity()` only
covers meaningful-usage skill players (QB/RB/WR/TE) -- no K/DEF opportunity
model exists, so this candidate only touches those four positions; K/DEF
keep using Phase 1's Baseline A untouched. Not every skill player with real
stats has an opportunity-model row either (e.g. a position player who threw
one trick-play pass); those are excluded from the blended candidate's
sample rather than guessed, and the real coverage rate is reported.

Deliberately reuses rather than reimplements: generate_projections.py's
build_projections() (same no-leakage filter, used for BOTH the outcome-based
and opportunity-based last-8 averages here) and backtest_evaluation.py's
FULL_PPR_VALUES/compute_points/list_backtest_weeks/assert_no_leakage/
collect_actuals/run_no_leakage_tests/summarize/POSITIONS (same backtest-week
selection, leakage guard, and metrics as Phase 1).

Sweeps the blend weight (0% / 25% / 50% / 75% / 100% opportunity) rather than
committing to one arbitrary ratio -- at weight 0 this should reproduce Phase
1's own Baseline A numbers almost exactly, which doubles as a sanity check
that this script's harness usage is consistent with Step 1/2's.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase2_opportunity_blend.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import (  # noqa: E402
    POSITIONS,
    assert_no_leakage,
    collect_actuals,
    compute_points,
    list_backtest_weeks,
    run_no_leakage_tests,
    summarize,
)
from generate_projections import (  # noqa: E402
    LOOKBACK_SEASONS,
    STAT_COLUMNS,
    build_projections,
    current_season,
    describe_player,
    get_current_season_and_week,
)

OUT_PATH = Path(__file__).parent.parent / "PHASE2_OPPORTUNITY_REPORT.md"  # gitignored, same as the other reports

# nflverse's own per-stat EXPECTED columns -> this project's own stat-column
# names, so build_projections()/compute_points() can be reused as-is on this
# data exactly like on load_player_stats. No mapping exists for
# attempts/carries/targets (not scored by FULL_PPR_VALUES, so irrelevant) or
# fumbles_lost_total (nflverse has no fumble _exp column at all -- fumbles
# aren't usage-predictable, nflverse apparently agrees).
EXP_COLUMN_MAP = {
    "pass_completions_exp": "completions",
    "pass_yards_gained_exp": "passing_yards",
    "pass_touchdown_exp": "passing_tds",
    "pass_interception_exp": "passing_interceptions",
    "rush_yards_gained_exp": "rushing_yards",
    "rush_touchdown_exp": "rushing_tds",
    "receptions_exp": "receptions",
    "rec_yards_gained_exp": "receiving_yards",
    "rec_touchdown_exp": "receiving_tds",
    "full_name": "player_display_name",
    "posteam": "team",
}
OPP_STAT_COLUMNS = [v for k, v in EXP_COLUMN_MAP.items() if k.endswith("_exp")]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_opportunity_frame(seasons: list[int]) -> pl.DataFrame:
    """Selects only the id/description columns plus the `_exp` columns before
    renaming -- `load_ff_opportunity` also has RAW (non-`_exp`) columns with
    the same target names (e.g. a raw `receptions` alongside `receptions_exp`),
    so renaming in place without dropping the raw ones first collides."""
    exp_cols = [k for k in EXP_COLUMN_MAP if k.endswith("_exp")]
    keep = ["season", "week", "player_id", "full_name", "position", "posteam"] + exp_cols
    return (
        nfl.load_ff_opportunity(seasons=seasons)
        .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
        .select(keep)
        .rename(EXP_COLUMN_MAP)
    )


def format_blend_table(all_rows: dict) -> str:
    lines = []
    for weight, rows in all_rows.items():
        lines.append(f"\n### Opportunity weight = {weight:.0%}\n")
        lines.append("| Position | n | MAE | MAE 95% CI | RMSE | Signed bias | Correlation |")
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

    is_skill = pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    skill_stats = stats.filter(is_skill)
    opp = load_opportunity_frame(lookback_seasons)
    # Same guard, proven against THIS script's own data source specifically --
    # load_ff_opportunity's season/week columns needed casting (see docstring)
    # before the filter semantics even work; prove it on the casted frame
    # rather than assuming the cast was sufficient.
    assert_no_leakage(opp, 2025, 10)
    print("ok:   no-leakage guard also holds on load_ff_opportunity's (cast) season/week columns")

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} fully-completed historical weeks to backtest: "
          f"{targets[0]} .. {targets[-1]}")

    results = {w: [] for w in BLEND_WEIGHTS}
    total_scored, total_with_opportunity = 0, 0

    for season, week in targets:
        actuals = collect_actuals(skill_stats, season, week, "player_id")
        if not actuals:
            continue

        relevant_stats = skill_stats.filter(pl.col("player_id").is_in(list(actuals.keys())))
        relevant_opp = opp.filter(pl.col("player_id").is_in(list(actuals.keys())))
        if relevant_stats.height == 0:
            continue
        assert_no_leakage(relevant_stats, season, week)
        if relevant_opp.height > 0:
            assert_no_leakage(relevant_opp, season, week)

        baseline_proj = build_projections(relevant_stats, season, week, "player_id", STAT_COLUMNS, describe_player, n_games=8)
        opp_proj = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)

        for entity_id, proj in baseline_proj.items():
            actual_row = actuals.get(entity_id)
            if actual_row is None:
                continue
            total_scored += 1
            baseline_points = compute_points(proj["projected_stats"])
            actual_points = compute_points({c: actual_row.get(c) for c in STAT_COLUMNS})

            opp_entry = opp_proj.get(entity_id)
            if opp_entry is None:
                continue  # no opportunity-model coverage for this player -- excluded from the blend sample
            total_with_opportunity += 1
            opp_points = compute_points(opp_entry["projected_stats"])

            for weight in BLEND_WEIGHTS:
                blended = (1 - weight) * baseline_points + weight * opp_points
                results[weight].append({
                    "position": proj["position"], "season": season, "week": week,
                    "projected": blended, "actual": actual_points, "error": blended - actual_points,
                })
        print(f"  done: season {season} week {week}", end="\r")

    print()
    coverage = total_with_opportunity / total_scored if total_scored else 0.0
    print(f"\nOpportunity-model coverage: {total_with_opportunity}/{total_scored} "
          f"skill player-weeks ({coverage:.1%}) had both an actual outcome AND an opportunity projection.")

    table = format_blend_table(results)
    print(table)

    report = f"""# Phase 2 Candidate: Opportunity Blend Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only -- does not
affect the live site or production pipeline. See MODEL_ROADMAP.md point 6.

**What this tests**: blends Phase 1's existing last-8-average projection (outcome-based) with a last-8-average
of nflverse's own `load_ff_opportunity()` usage-based expected-stat model (opportunity-based), at several
blend weights, backtested against the same {len(targets)} historical weeks and no-leakage guard Phase 1 used.

**Scoring-scale fix, not a reinterpretation of scope**: nflverse's own `total_fantasy_points_exp` uses a
different scoring convention than this project (6pt passing TDs vs. this project's 4pt, plus 2-point
conversions this project's `scoring.js` doesn't implement at all) -- confirmed on real data, not assumed. Used
nflverse's per-stat `_exp` columns instead, scored through this project's own `compute_points()`, so every
candidate here is judged on the identical points scale.

**Coverage**: {total_with_opportunity}/{total_scored} ({coverage:.1%}) scored skill player-weeks had an
opportunity-model row to blend with; the rest (no opportunity coverage, e.g. extremely marginal usage) are
excluded from every weighted candidate's sample, not guessed. K/DEF are untouched by this candidate --
`load_ff_opportunity()` has no defense/kicker model, so Phase 1's Baseline A keeps projecting those positions.

**Weight 0% is Baseline A alone (pure outcome, no opportunity blended in) and should closely reproduce Phase
1's own Baseline A numbers from BACKTEST_REPORT.md** (same harness, same filter, same weeks) -- included
specifically as a cross-check that this script's reuse of `build_projections()` is behaving consistently with
Step 1/2's, not just as an endpoint of the sweep.
{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
