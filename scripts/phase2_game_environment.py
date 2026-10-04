"""
phase2_game_environment.py

Phase 2's second candidate, per MODEL_ROADMAP.md's phase list ("opponent
adjustment, game environment") and the designer's go-ahead to pick between
EPA/Next Gen Stats vs. game-environment signals for the next Phase 2
feature (designer's call left to me: picked this one -- zero new data
fetches, `load_schedules()` is already pulled every run, and it directly
fills a real gap Baseline A/B/C all share: none of them use any
opponent/environment signal at all). OFFLINE ONLY.

Signal: real Vegas lines (`spread_line`/`total_line` from `load_schedules`).
A team's own game's line is NOT leakage the way a future week's stat line
would be -- betting lines are set well before kickoff, so a team's own
game's line is available at projection time by definition. The no-leakage
constraint that matters everywhere else in this project (never let a later
week's STAT LINE influence an earlier projection) doesn't apply to a game's
own pre-game market data.

Formula, verified against real data before use (not assumed): `spread_line`
is the HOME team's spread -- negative means home favored (confirmed: e.g.
real 2025 wk10 MIA(home) -8.5 spread, won by 17; DEN(home) +9.5 spread,
i.e. a 9.5-point underdog, actually won outright -- a real upset, not a
data error). Standard implied-total split:
  home_implied = total_line/2 - spread_line/2
  away_implied = total_line/2 + spread_line/2
  (sums to total_line exactly, confirmed arithmetically)
Expressed as a fraction of an even 50/50 split (so it's self-contained --
no separate "league average" reference needed, no second leakage question
to answer about where that reference comes from):
  offense_factor = team_implied / (total_line/2) = 1 -/+ spread_line/total_line
A defense's own environment is the INVERSE of its opponent's offense_factor
(an opponent projected well below a neutral split means fewer points
allowed, which is good for the defense) -- symmetric around 1.0:
  defense_factor = 2 - opponent's offense_factor
Real coverage confirmed: 0 of 619 actually-played games in the lookback
window are missing spread_line/total_line (the only nulls are future,
not-yet-played 2026 games already excluded from the backtest targets).

Builds on Baseline C (QB/TE 50% opportunity blend, RB/WR/K/DEF unchanged --
see phase2_baseline_c.py), not Baseline A, since Baseline C is the current
adopted Phase 2 baseline. Sweeps the environment-adjustment weight
(0/25/50/75/100%) the same way Candidate 1 swept the opportunity weight,
applied via `adjusted = baseline * (1 + weight * (factor - 1))` -- at
weight 0 this reproduces Baseline C exactly (same population, same numbers,
a built-in cross-check), at 100% the full factor is applied.

Reuses generate_projections.py's build_projections() (QB/RB/WR/TE/K/DEF, same
as Baseline C) and backtest_evaluation.py's compute_points/list_backtest_weeks/
assert_no_leakage/collect_actuals/run_no_leakage_tests/summarize/POSITIONS/
format_table. The QB/TE opportunity-blend piece duplicates ~10 lines from
phase2_baseline_c.py's evaluate_week_blended() rather than importing it,
since that function appends straight into a results dict and doesn't expose
the per-entity point estimate this script needs as an intermediate value --
small, understood duplication, not a new formula.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase2_game_environment.py
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

OUT_PATH = Path(__file__).parent.parent / "PHASE2_GAME_ENVIRONMENT_REPORT.md"  # gitignored
ENV_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_environment_factors(schedules: pl.DataFrame):
    """{(season, week, team): offense_factor}, {(season, week, team): opponent}."""
    factors, opponents = {}, {}
    games = schedules.filter(
        pl.col("spread_line").is_not_null() & pl.col("total_line").is_not_null() & (pl.col("total_line") != 0)
    )
    for row in games.iter_rows(named=True):
        season, week = row["season"], row["week"]
        home, away = row["home_team"], row["away_team"]
        spread, total = row["spread_line"], row["total_line"]
        factors[(season, week, home)] = 1 - spread / total
        factors[(season, week, away)] = 1 + spread / total
        opponents[(season, week, home)] = away
        opponents[(season, week, away)] = home
    return factors, opponents


def offense_factor(factors, season, week, team):
    return factors.get((season, week, team), 1.0)


def defense_factor(factors, opponents, season, week, team):
    opp = opponents.get((season, week, team))
    if opp is None:
        return 1.0
    return 2 - offense_factor(factors, season, week, opp)


def baseline_c_points(entity_id, proj, opp_proj) -> float:
    """Same formula as phase2_baseline_c.py's evaluate_week_blended -- QB/TE
    at 50% opportunity weight, RB/WR/K/DEF unchanged (0%), falling back to
    the pure outcome-based projection when opportunity coverage is missing."""
    baseline_points = compute_points(proj["projected_stats"])
    weight = PER_POSITION_WEIGHTS.get(proj["position"], 0.0)
    opp_entry = opp_proj.get(entity_id) if opp_proj else None
    if weight > 0 and opp_entry is not None:
        opp_points = compute_points(opp_entry["projected_stats"])
        return (1 - weight) * baseline_points + weight * opp_points
    return baseline_points


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
    env_factors, env_opponents = build_environment_factors(schedules)

    targets = [
        (s, w) for s, w in list_backtest_weeks(schedules, lookback_seasons)
        if not (s == current_s and w >= current_w)
    ]
    print(f"\n{len(targets)} fully-completed historical weeks to backtest: "
          f"{targets[0]} .. {targets[-1]}")

    results = {w: [] for w in ENV_WEIGHTS}

    for season, week in targets:
        # Skill (QB/RB/WR/TE), same as Baseline C.
        skill_actuals = collect_actuals(skill_stats, season, week, "player_id")
        if skill_actuals:
            relevant_stats = skill_stats.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
            relevant_opp = opp.filter(pl.col("player_id").is_in(list(skill_actuals.keys())))
            if relevant_stats.height > 0:
                assert_no_leakage(relevant_stats, season, week)
                if relevant_opp.height > 0:
                    assert_no_leakage(relevant_opp, season, week)
                proj_map = build_projections(relevant_stats, season, week, "player_id", STAT_COLUMNS, describe_player, n_games=8)
                opp_map = build_projections(relevant_opp, season, week, "player_id", OPP_STAT_COLUMNS, describe_player, n_games=8)
                for entity_id, proj in proj_map.items():
                    actual_row = skill_actuals.get(entity_id)
                    if actual_row is None:
                        continue
                    base_points = baseline_c_points(entity_id, proj, opp_map)
                    actual_points = compute_points({c: actual_row.get(c) for c in STAT_COLUMNS})
                    factor = offense_factor(env_factors, season, week, proj.get("team"))
                    for weight in ENV_WEIGHTS:
                        adjusted = base_points * (1 + weight * (factor - 1))
                        results[weight].append({
                            "position": proj["position"], "season": season, "week": week,
                            "projected": adjusted, "actual": actual_points, "error": adjusted - actual_points,
                        })

        # Kickers -- offense_factor, same team-based logic, no opportunity blend (K isn't in PER_POSITION_WEIGHTS).
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
                    base_points = compute_points(proj["projected_stats"])
                    actual_points = compute_points({c: actual_row.get(c) for c in K_STAT_COLUMNS})
                    factor = offense_factor(env_factors, season, week, proj.get("team"))
                    for weight in ENV_WEIGHTS:
                        adjusted = base_points * (1 + weight * (factor - 1))
                        results[weight].append({
                            "position": "K", "season": season, "week": week,
                            "projected": adjusted, "actual": actual_points, "error": adjusted - actual_points,
                        })

        # DEF -- defense_factor (inverse of the OPPONENT's offense_factor).
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
                    base_points = compute_points(proj["projected_stats"])
                    actual_points = compute_points({c: actual_row.get(c) for c in DEF_STAT_COLUMNS})
                    factor = defense_factor(env_factors, env_opponents, season, week, proj.get("team"))
                    for weight in ENV_WEIGHTS:
                        adjusted = base_points * (1 + weight * (factor - 1))
                        results[weight].append({
                            "position": "DEF", "season": season, "week": week,
                            "projected": adjusted, "actual": actual_points, "error": adjusted - actual_points,
                        })
        print(f"  done: season {season} week {week}", end="\r")

    print()
    labeled = {f"Environment weight = {w:.0%}": rows for w, rows in results.items()}
    table = format_table(labeled)
    print(table)

    weight0 = summarize(results[0.0])
    print(f"\nWeight-0% cross-check vs. Baseline C's own Overall MAE 4.56: got {weight0['mae']:.2f}")

    report = f"""# Phase 2 Candidate: Game-Environment (Vegas Lines) Report

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE analysis only -- does not
affect the live site or production pipeline. See MODEL_ROADMAP.md's "opponent adjustment, game environment"
item.

**Signal**: real `spread_line`/`total_line` from `load_schedules` (100% coverage on actually-played games in
this window). A team's own game's line is available at projection time by definition, so using it is not
the leakage no-leakage testing elsewhere in this project guards against.

**Builds on Baseline C** (QB/TE 50% opportunity blend), not Baseline A -- weight 0% below should reproduce
Baseline C's own Overall MAE (4.56) as a cross-check.
{table}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
