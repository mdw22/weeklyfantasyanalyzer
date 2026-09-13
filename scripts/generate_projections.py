"""
generate_projections.py

Run daily by .github/workflows/weekly-projections.yml.

Produces two files for the *current* NFL week:

  data/week_{NN}/projections.json
      Per-player RAW projected stat lines (not fantasy points -- scoring
      is applied client-side so any scoring settings, PPR or otherwise,
      can be recomputed instantly without touching this pipeline).

  data/week_{NN}/history.json
      Per-player list of their last N actual game stat-lines, for
      bootstrap-resampling in the Monte Carlo win-probability step
      the frontend runs later.

v1 projection method is deliberately dumb: a simple average of each
player's last N games. That's enough to get the whole pipeline (Actions
-> JSON -> frontend) working end-to-end. Swap in opponent adjustment,
injury filtering, and/or a blend with load_ff_rankings()'s FantasyPros
consensus once this is proven out -- see the TODOs below.

VERIFIED against nflreadpy 0.1.5 (2026-09-12) -- ran locally and checked
`schedules.columns` / `stats.columns` directly, per the spec's own
instructions:
  - `home_score` / `gameday` on load_schedules() are correct as assumed.
  - `interceptions` does NOT exist -- it's `passing_interceptions`
    (this exact gotcha was already known from the prior project, see
    CLAUDE.md). Fixed in STAT_COLUMNS below.
  - `fumbles_lost` does NOT exist either -- fumbles lost are split by
    play type (`sack_fumbles_lost`, `rushing_fumbles_lost`,
    `receiving_fumbles_lost`); nflreadpy also provides a pre-summed
    `fumbles_lost_total` across all of those (defense/ST included, but
    0 for offensive skill players), used here instead.
  - `player_display_name`, `position`, and `team` (not `recent_team`)
    all exist directly -- no fallback needed.
  - Real bug found and fixed: loading `seasons=[season]` (current
    season only) leaves ZERO usable history in/near week 1, since
    there's nothing before "week 1 of the current season" to average.
    Confirmed empirically against the live 2026 season (currently
    Week 1, 2 of 16 games played) -- the single-season query returned
    0 "past" rows. Fixed by loading a lookback window of
    `LOOKBACK_SEASONS` prior seasons alongside the current one, same
    pattern the prior project's model used for this reason.
"""

import json
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl

# How many past games to average for a projection / keep for Monte Carlo
# history. 8 is a reasonable starting point -- recent enough to reflect
# current role, long enough to smooth out one-off blowups.
N_GAMES = 8

# How many prior seasons (beyond the current one) to pull stats from, so
# there's still a usable history window early in a new season -- the
# current season alone has zero "past" games near week 1.
LOOKBACK_SEASONS = 2

# Verified against nflreadpy 0.1.5's actual `stats.columns` output.
STAT_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost_total",
]

DATA_DIR = Path("data")


def current_season() -> int:
    """NFL seasons are labeled by the year they START (e.g. games played
    in January 2027 are still part of the "2026 season"). Treat Jan/Feb
    as still belonging to the previous calendar year's season."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def get_current_season_and_week(schedules: pl.DataFrame) -> tuple[int, int]:
    """Pick the next unplayed game's (season, week) as "current."
    Falls back to the latest completed week if the season has ended
    (e.g. during the offseason).
    """
    upcoming = schedules.filter(pl.col("home_score").is_null()).sort(
        ["season", "week", "gameday"]
    )
    if upcoming.height == 0:
        last = schedules.sort(["season", "week"], descending=True).row(0, named=True)
        return last["season"], last["week"]
    first = upcoming.row(0, named=True)
    return first["season"], first["week"]


def _partition_by_player(df: pl.DataFrame) -> dict:
    """Wrapper around polars partition_by so a version difference in how
    the dict keys come back (bare value vs. 1-tuple) doesn't break both
    call sites below."""
    groups = df.partition_by("player_id", as_dict=True)
    return {(k[0] if isinstance(k, tuple) else k): v for k, v in groups.items()}


def build_projections(stats: pl.DataFrame, season: int, week: int) -> dict:
    """v1 baseline: each player's projection = mean of their last
    N_GAMES games across STAT_COLUMNS. No opponent adjustment yet."""
    past = stats.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )

    projections = {}
    for player_id, games in _partition_by_player(past).items():
        recent = games.sort("week", descending=True).head(N_GAMES)
        if recent.height == 0:
            continue
        row0 = recent.row(0, named=True)
        means = {
            stat: round(float(recent[stat].mean() or 0.0), 2)
            for stat in STAT_COLUMNS
            if stat in recent.columns
        }
        projections[str(player_id)] = {
            "player_name": row0["player_display_name"],
            "position": row0["position"],
            "team": row0["team"],
            "games_used": recent.height,
            "projected_stats": means,
        }
    return projections


def build_history(stats: pl.DataFrame, season: int, week: int) -> dict:
    """Per-player list of actual past game stat-lines (raw, un-scored)
    for the frontend's Monte Carlo bootstrap sampling."""
    past = stats.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )

    history = {}
    for player_id, games in _partition_by_player(past).items():
        recent = games.sort("week", descending=True).head(N_GAMES)
        game_rows = []
        for row in recent.iter_rows(named=True):
            line = {stat: row.get(stat, 0.0) for stat in STAT_COLUMNS if stat in recent.columns}
            line["season"] = row["season"]
            line["week"] = row["week"]
            game_rows.append(line)
        history[str(player_id)] = game_rows
    return history


def main() -> None:
    season = current_season()
    schedules = nfl.load_schedules(seasons=season)
    season, week = get_current_season_and_week(schedules)

    lookback_seasons = list(range(season - LOOKBACK_SEASONS, season + 1))
    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")

    projections = build_projections(stats, season, week)
    history = build_history(stats, season, week)

    out_dir = DATA_DIR / f"week_{week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "projections.json").write_text(json.dumps(projections, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Wrote projections + history for season {season}, week {week} "
          f"({len(projections)} players) to {out_dir}/")


if __name__ == "__main__":
    main()
