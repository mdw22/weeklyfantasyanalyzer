"""
phase2_baseline_c.py

Phase 2's first real candidate, per the designer's explicit decision after
reviewing Candidate 1's uniform-weight sweep (scripts/phase2_opportunity_blend.py):
a PER-POSITION opportunity blend -- QB and TE at 50% opportunity weight (both
showed a genuine, if small, monotonic MAE improvement through 50% in the
uniform sweep), RB/WR/K/DEF left at 0% (RB/WR showed no measured benefit in
the sweep; K/DEF have no opportunity-model coverage at all, so 0% there is
just "unchanged from Baseline A", not a separate judgment call). This is the
asymmetric combination ITSELF, backtested as its own named candidate across
all six positions -- not inferred from the uniform sweep's per-position rows,
per the designer's explicit instruction that those are not the same thing.

Named "Baseline C" to sit alongside Phase 1's Baseline A/B in
BACKTEST_REPORT.md: this script APPENDS a "## Baseline C" section to that
file (replacing any previous Baseline C section from a prior run of THIS
script, never accumulating duplicates). Note: BACKTEST_REPORT.md is also
unconditionally overwritten in full by backtest_evaluation.py's own run --
if that script runs after this one, the Baseline C section is wiped and
needs re-appending by re-running this script. Both files are gitignored
working docs, regenerated on demand, not committed artifacts -- an accepted,
documented limitation, not something worth solving with shared state.

Sample-parity choice: when a QB/TE entity has no opportunity-model coverage
for a given week (the ~3-5% gap measured in Candidate 1), this falls BACK to
the pure outcome-based projection for that one row (weight effectively 0),
rather than excluding the row from the sample the way the uniform-weight
sweep did. This keeps Baseline C's population identical to Baseline A's full
population (same n), which is what a real apples-to-apples Overall
MAE/RMSE comparison against Baseline A needs -- and it's also the
behaviorally correct choice if this ever shipped: a player lacking
opportunity coverage should keep getting the existing projection, not get
silently dropped.

Reuses backtest_evaluation.py's exact harness (entity groups, no-leakage
guard, backtest-week selection, evaluate_week() for K/DEF where weight=0
makes it literally identical to Baseline A, summarize/format_table for the
report) and phase2_opportunity_blend.py's opportunity-frame loading/column
mapping for the QB/TE blend. Zero new filtering or scoring logic -- only new
code is the per-position weighted blend itself.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase2_baseline_c.py
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
    evaluate_week,
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
from phase2_opportunity_blend import OPP_STAT_COLUMNS, load_opportunity_frame  # noqa: E402

REPORT_PATH = Path(__file__).parent.parent / "BACKTEST_REPORT.md"  # gitignored, shared with backtest_evaluation.py
BASELINE_C_MARKER = "## Baseline C"

PER_POSITION_WEIGHTS = {"QB": 0.5, "TE": 0.5}  # RB/WR/K/DEF default to 0.0 via .get()


def evaluate_week_blended(stats_df, opp_df, season, week, actuals, results, baseline_name):
    """Same shape as backtest_evaluation.py's evaluate_week(), but blends in
    the opportunity signal at each entity's OWN position's weight (0.0 for
    RB/WR -- identical to Baseline A) instead of a single global weight."""
    relevant_stats = stats_df.filter(pl.col("player_id").is_in(list(actuals.keys())))
    relevant_opp = opp_df.filter(pl.col("player_id").is_in(list(actuals.keys())))
    if relevant_stats.height == 0:
        return
    assert_no_leakage(relevant_stats, season, week)
    if relevant_opp.height > 0:
        assert_no_leakage(relevant_opp, season, week)

    baseline_proj = build_projections(relevant_stats, season, week, "player_id", STAT_COLUMNS, describe_player, n_games=8)
    opp_proj = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)

    for entity_id, proj in baseline_proj.items():
        actual_row = actuals.get(entity_id)
        if actual_row is None:
            continue
        baseline_points = compute_points(proj["projected_stats"])
        weight = PER_POSITION_WEIGHTS.get(proj["position"], 0.0)
        opp_entry = opp_proj.get(entity_id)
        if weight > 0 and opp_entry is not None:
            opp_points = compute_points(opp_entry["projected_stats"])
            blended = (1 - weight) * baseline_points + weight * opp_points
        else:
            blended = baseline_points  # weight 0, or no opportunity coverage this week -- falls back to Baseline A
        actual_points = compute_points({c: actual_row.get(c) for c in STAT_COLUMNS})
        results.setdefault(baseline_name, []).append({
            "position": proj["position"], "season": season, "week": week,
            "projected": blended, "actual": actual_points, "error": blended - actual_points,
        })


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
    skill_stats = stats.filter(is_skill)
    kickers = add_k_stat_columns(stats.filter(is_k))
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"), schedules
    )
    opp = load_opportunity_frame(lookback_seasons)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} fully-completed historical weeks to backtest: "
          f"{targets[0]} .. {targets[-1]}")

    results = {}
    baseline_name = "Baseline C -- per-position opportunity blend (QB 50%, TE 50%, RB/WR/K/DEF 0%)"

    for season, week in targets:
        skill_actuals = collect_actuals(skill_stats, season, week, "player_id")
        if skill_actuals:
            evaluate_week_blended(skill_stats, opp, season, week, skill_actuals, results, baseline_name)
        # K/DEF: weight is 0 for both, i.e. literally Baseline A unchanged --
        # reuse evaluate_week() directly rather than re-deriving the same thing.
        evaluate_week(kickers, kickers, season, week, "player_id", K_STAT_COLUMNS, describe_player,
                      n_games=8, results=results, baseline_name=baseline_name)
        evaluate_week(team_stats, team_stats, season, week, "def_id", DEF_STAT_COLUMNS, describe_defense,
                      n_games=8, results=results, baseline_name=baseline_name)
        print(f"  done: season {season} week {week}", end="\r")

    print()
    table = format_table(results)
    print(table)

    overall = summarize(results[baseline_name])
    section = f"""
{BASELINE_C_MARKER} — per-position opportunity blend (2026-10-04)

Phase 2, Candidate 1 reviewed: the uniform-weight sweep (`PHASE2_OPPORTUNITY_REPORT.md`) showed QB and TE
both improving (small, monotonic MAE gain through 50% opportunity weight) while RB and WR didn't move either
direction, and pure-opportunity (100%) had the worst Overall MAE of the whole sweep with a steadily growing
positive bias. Decision: a PER-POSITION blend, not a uniform one -- QB and TE at 50% opportunity weight,
RB/WR/K/DEF left at 0% (identical to Baseline A). Backtested here as its own named candidate across all six
positions, not inferred from the uniform sweep's rows.

Sample parity with Baseline A: when a QB/TE entity has no opportunity-model coverage for a given week, this
falls back to the pure outcome-based (Baseline A) projection for that one row rather than excluding it --
keeps this candidate's population identical to Baseline A's, so the Overall row below is a true
apples-to-apples comparison against Baseline A's own Overall MAE {overall['mae']:.2f} ±{overall['mae_ci95']:.2f} vs. Baseline A's reported 4.58 ±0.07 (see the Baseline A table above, same {len(targets)}-week window).
{table}
"""

    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    marker_pos = existing.find(BASELINE_C_MARKER)
    base = existing[:marker_pos].rstrip() + "\n" if marker_pos != -1 else existing.rstrip() + "\n"
    REPORT_PATH.write_text(base + section)
    print(f"\nBaseline C section written/updated in {REPORT_PATH} (generated {datetime.now(timezone.utc).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
