"""
model_core.py

Single source of truth for the v3 projection model (PRODUCTION_MODEL_SPEC.md
sections 2-3). Imported by generate_projections.py (production) and by the
phase-4 backtest scripts, so the shipped model and the validated model are
the same code.

Pure functions over plain dicts -- no numpy, so the daily workflow needs no
new dependency. Data assembly (which weeks, which players, as-of-week
projections) stays with each caller; this module only does the math.

Pipeline, per player-week, fit under default (full-PPR) scoring:
  A: last-8 stat line (build_projections' projected_stats).
  C: QB/TE -> 0.5*A + 0.5*opportunity stat line (keys missing on the
     opportunity side count as 0); everyone else C = A.
  D: per position, from the last ROLLING_WEEKS completed weeks of
     (C stat line, actual stat line) pairs:
       D_stats = mean_actual_stats + beta * (C_stats - mean_C_stats)
     beta = OLS slope of actual points on C points, clamped to [0, 1]. Under default scoring
     points(D_stats) == mean_actual_pts + beta*(C_pts - mean_C_pts) exactly;
     in stat space it stays coherent under any linear custom scoring.
     Fewer than MIN_SAMPLE pairs for a position -> D = C.
  Fallback ratio pools (for players with no usable history shape):
     actual / D points for window players with D >= RATIO_MIN_D, binned by
     D (default points), up to RATIO_CAP most recent per bin.
"""

import nflreadpy as nfl
import polars as pl

# Full-PPR point values -- Python twin of src/lib/scoring.js's BASE_VALUES +
# the full_ppr preset. Hand-kept cross-language seam: if scoring.js changes,
# change this too.
FULL_PPR_VALUES = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "passing_interceptions": -2,
    "passing_2pt_conversions": 2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "rushing_2pt_conversions": 2,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "receiving_2pt_conversions": 2,
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

MODEL_VERSION = "v3"
SHRINK_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
OPPORTUNITY_POSITIONS = {"QB": 0.5, "TE": 0.5}
ROLLING_WEEKS = 8
MIN_SAMPLE = 50
POOLING_K = 3
RATIO_BIN_EDGES = [2.0, 5.0, 10.0]  # default points: <2, 2-5, 5-10, >=10
RATIO_CAP = 200
RATIO_MIN_D = 0.5
RATIO_MIN_POOL = 10  # bin thinner than this merges with neighbors, then the position pool

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


def compute_points(stat_line: dict, values: dict = FULL_PPR_VALUES) -> float:
    return sum((stat_line.get(k) or 0.0) * v for k, v in values.items())


def load_opportunity_frame(seasons: list[int]) -> pl.DataFrame:
    """nflverse's usage-based expected stats, renamed onto this project's stat
    names. Selects the `_exp` columns before renaming -- the raw columns share
    the target names (e.g. raw `receptions` next to `receptions_exp`).
    `season` arrives as a string and `week` as a float; both cast to Int64."""
    exp_cols = [k for k in EXP_COLUMN_MAP if k.endswith("_exp")]
    keep = ["season", "week", "player_id", "full_name", "position", "posteam"] + exp_cols
    return (
        nfl.load_ff_opportunity(seasons=seasons)
        .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
        .select(keep)
        .rename(EXP_COLUMN_MAP)
    )


def baseline_c_stats(a_stats: dict, position: str, opp_stats: dict | None) -> dict:
    weight = OPPORTUNITY_POSITIONS.get(position, 0.0)
    if weight == 0.0 or opp_stats is None:
        return dict(a_stats)
    return {k: (1 - weight) * (v or 0.0) + weight * (opp_stats.get(k) or 0.0) for k, v in a_stats.items()}


def fit_window(pairs: list) -> dict:
    """pairs: (position, C_stats, actual_stats) from the rolling window.
    Returns {position: params} for positions with >= MIN_SAMPLE pairs."""
    by_pos = {}
    for position, c_stats, actual_stats in pairs:
        by_pos.setdefault(position, []).append((c_stats, actual_stats))
    params = {}
    for position, rows in by_pos.items():
        n = len(rows)
        if position not in SHRINK_POSITIONS or n < MIN_SAMPLE:
            continue
        c_pts = [compute_points(c) for c, _ in rows]
        a_pts = [compute_points(a) for _, a in rows]
        m_c, m_act = sum(c_pts) / n, sum(a_pts) / n
        var_c = sum((x - m_c) ** 2 for x in c_pts)
        beta = sum((x - m_c) * (y - m_act) for x, y in zip(c_pts, a_pts)) / var_c if var_c else 1.0
        # Clamp to [0, 1]: a negative slope would invert the position's ranking and one above 1 would
        # amplify it -- neither is a real effect, just noise in an 8-week window (mostly K/DEF).
        beta = min(max(beta, 0.0), 1.0)
        keys = set()
        for c, _ in rows:
            keys.update(c)
        params[position] = {
            "beta": beta, "m_act": m_act, "m_c": m_c, "n_pairs": n,
            "mean_c_stats": {k: sum(c.get(k) or 0.0 for c, _ in rows) / n for k in keys},
            "mean_actual_stats": {k: sum(a.get(k) or 0.0 for _, a in rows) / n for k in keys},
        }
    return params


def apply_d(c_stats: dict, position: str, params: dict) -> dict:
    p = params.get(position)
    if p is None:
        return dict(c_stats)
    beta, m_act, m_c = p["beta"], p["mean_actual_stats"], p["mean_c_stats"]
    return {k: m_act.get(k, 0.0) + beta * ((v or 0.0) - m_c.get(k, 0.0)) for k, v in c_stats.items()}


def ratio_bin(d_points: float) -> int:
    return sum(d_points >= edge for edge in RATIO_BIN_EDGES)


def build_ratio_pools(entries: list) -> dict:
    """entries: (position, D points, actual points), oldest first.
    Returns {(position, bin): [ratio, ...]} capped at the RATIO_CAP most recent."""
    pools = {}
    for position, d_pts, actual_pts in entries:
        if d_pts >= RATIO_MIN_D:
            pools.setdefault((position, ratio_bin(d_pts)), []).append(actual_pts / d_pts)
    return {k: v[-RATIO_CAP:] for k, v in pools.items()}
