"""
v3_parity_pipeline.py

PRODUCTION_MODEL_SPEC.md section 4, item 2 -- pipeline parity. For three
past weeks, the production path (generate_projections.build_v3, which
assembles its own rolling window from schedules) vs. the validated backtest
(phase4_combined_stack.build_state, 2023 loaded as history only). Both see
identical source data. Pass: max |D points diff| < 1e-6 per player under
default scoring. Also compares the fallback ratio pools (pipeline rounds
them to 4 dp for the JSON, so that comparison uses a 1e-4 tolerance).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/v3_parity_pipeline.py
"""

import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from generate_projections import (  # noqa: E402
    DEF_STAT_COLUMNS,
    K_STAT_COLUMNS,
    LOOKBACK_SEASONS,
    STAT_COLUMNS,
    add_def_stat_columns,
    add_k_stat_columns,
    build_v3,
    current_season,
    describe_defense,
    describe_player,
    get_current_season_and_week,
)
from phase4_combined_stack import build_state  # noqa: E402

PARITY_WEEKS = [(2024, 10), (2025, 1), (2026, 3)]
TOLERANCE = 1e-6


def main() -> None:
    _, weekly_state, _ = build_state(extra_history_seasons=1)

    current_s, _ = get_current_season_and_week(nfl.load_schedules(seasons=current_season()))
    seasons = list(range(current_s - LOOKBACK_SEASONS - 1, current_s + 1))  # same frames build_state loaded
    stats = nfl.load_player_stats(seasons=seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=seasons)
    kickers = add_k_stat_columns(stats.filter((pl.col("position") == "K").fill_null(False)))
    team_stats = add_def_stat_columns(nfl.load_team_stats(seasons=seasons, summary_level="week"), schedules)
    groups = [
        (stats.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])), "player_id", STAT_COLUMNS, describe_player),
        (kickers, "player_id", K_STAT_COLUMNS, describe_player),
        (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
    ]
    opp = model_core.load_opportunity_frame(seasons)

    all_pass = True
    for season, week in PARITY_WEEKS:
        st = weekly_state[(season, week)]
        d_stats, meta = build_v3(groups, opp, schedules, season, week, st["projections"])
        diffs = [abs(model_core.compute_points(d_stats[e]) - st["baseline_d"][e]) for e in st["projections"]]
        max_d = max(diffs)

        pool_diff, pool_mismatch = 0.0, 0
        for (pos, b), arr in st["ratio_bins"].items():
            shipped = meta["ratio_pools"].get(pos, {}).get(str(b))
            if shipped is None or len(shipped) != len(arr):
                pool_mismatch += 1
                continue
            pool_diff = max(pool_diff, max(abs(x - y) for x, y in zip(shipped, arr)))
        shipped_bins = sum(len(v) for v in meta["ratio_pools"].values())
        if shipped_bins != len(st["ratio_bins"]):
            pool_mismatch += abs(shipped_bins - len(st["ratio_bins"]))

        ok = max_d < TOLERANCE and pool_mismatch == 0 and pool_diff < 1e-4
        all_pass &= ok
        print(f"({season}, {week}): {len(diffs)} players, max |D diff| = {max_d:.2e}; "
              f"ratio pools: max |diff| = {pool_diff:.1e}, bin/length mismatches = {pool_mismatch} -> "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nPipeline parity: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
