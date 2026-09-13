"""
generate_projections.py

Run daily by .github/workflows/weekly-projections.yml.

Produces two files for the *current* NFL week:

  data/week_{NN}/projections.json
      Per-player (and per-team-defense) RAW projected stat lines (not
      fantasy points -- scoring is applied client-side so any scoring
      settings, PPR or otherwise, can be recomputed instantly without
      touching this pipeline).

  data/week_{NN}/history.json
      Per-player (and per-team-defense) list of their last N actual game
      stat-lines, for bootstrap-resampling in the Monte Carlo
      win-probability step the frontend runs later.

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

TEAM DEFENSE (DEF) -- added after a real gap surfaced via ESPN sync:
`load_player_stats()` is per-INDIVIDUAL-player only; it has no team
defense/special-teams "player" at all, so DEF was previously never
fillable anywhere in the app (sync or manual). Added below from
`load_team_stats()`, keyed by a synthesized `DEF_{team}` id (e.g.
`DEF_KC`) -- this exact id scheme must match `sync_espn.py`'s
`f"DEF_{abbr}"` construction, another hand-maintained cross-file seam
(see CLAUDE.md). Only count-based defensive stats are included (sacks,
INTs, fumble recoveries, safeties, defensive/ST TDs, blocked kicks, 2pt
returns) -- points-allowed and yards-allowed tiers are NOT implemented
(tiered/bucketed scoring doesn't fit the linear `amount * point_value`
scoring engine `src/lib/scoring.js` uses everywhere else; a real,
deliberately deferred gap, not an oversight).
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

# Output field names for team defense, matching the categories
# src/lib/scoring.js's "Defense" STAT_FIELDS group scores on.
DEF_STAT_COLUMNS = [
    "def_sacks",
    "def_interceptions",
    "def_fumble_recoveries",
    "def_safeties",
    "def_touchdowns",
    "def_blocked_kicks",
    "def_two_point_returns",
]

# Lives under public/ (not repo-root data/) so the Vite dev server serves
# it and `vite build` bundles it into dist/ automatically -- no separate
# copy step needed anywhere in the deploy pipeline.
DATA_DIR = Path("public/data")


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


def _partition_by(df: pl.DataFrame, id_col: str) -> dict:
    """Wrapper around polars partition_by so a version difference in how
    the dict keys come back (bare value vs. 1-tuple) doesn't break both
    call sites below."""
    groups = df.partition_by(id_col, as_dict=True)
    return {(k[0] if isinstance(k, tuple) else k): v for k, v in groups.items()}


def build_projections(df: pl.DataFrame, season: int, week: int, id_col: str, stat_columns: list, describe_row) -> dict:
    """v1 baseline: each entity's (player OR team-defense) projection =
    mean of their last N_GAMES games across `stat_columns`. No opponent
    adjustment yet. `describe_row` turns one row into the entry's
    name/position/team fields."""
    past = df.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )

    projections = {}
    for entity_id, games in _partition_by(past, id_col).items():
        recent = games.sort("week", descending=True).head(N_GAMES)
        if recent.height == 0:
            continue
        row0 = recent.row(0, named=True)
        means = {
            stat: round(float(recent[stat].mean() or 0.0), 2)
            for stat in stat_columns
            if stat in recent.columns
        }
        entry = describe_row(row0)
        entry["games_used"] = recent.height
        entry["projected_stats"] = means
        projections[str(entity_id)] = entry
    return projections


def build_history(df: pl.DataFrame, season: int, week: int, id_col: str, stat_columns: list) -> dict:
    """Per-entity list of actual past game stat-lines (raw, un-scored)
    for the frontend's Monte Carlo bootstrap sampling."""
    past = df.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )

    history = {}
    for entity_id, games in _partition_by(past, id_col).items():
        recent = games.sort("week", descending=True).head(N_GAMES)
        game_rows = []
        for row in recent.iter_rows(named=True):
            line = {stat: row.get(stat, 0.0) for stat in stat_columns if stat in recent.columns}
            line["season"] = row["season"]
            line["week"] = row["week"]
            game_rows.append(line)
        history[str(entity_id)] = game_rows
    return history


def describe_player(row0: dict) -> dict:
    return {
        "player_name": row0["player_display_name"],
        "position": row0["position"],
        "team": row0["team"],
    }


def describe_defense(row0: dict) -> dict:
    team = row0["team"]
    return {
        "player_name": f"{team} D/ST",
        "position": "DEF",
        "team": team,
    }


def add_def_stat_columns(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Combines nflreadpy's granular team-defense columns into the
    handful of categories this app actually scores on, matching the
    prior project's Full-PPR defense scoring (see CLAUDE.md): sack, INT,
    fumble recovery, safety, defensive/ST TD, blocked kick, 2pt return.

    Two real judgment calls, made from inspecting real 2025-season data:
      - `fumble_recovery_opp` (recovering the OPPONENT's fumble) is the
        turnover stat, not `def_fumbles` -- a much smaller, differently
        defined column (49 vs 264 league-wide across a season; not
        documented, `fumble_recovery_opp`'s definition is unambiguous by
        name, `def_fumbles`'s is not).
      - `def_touchdowns` = def_tds (INT/fumble return TDs) +
        special_teams_tds (kick/punt return TDs) only. Does NOT also add
        `fumble_recovery_tds` -- that column is almost certainly already
        a subset of `def_tds` (a fumble recovered and returned for a
        touchdown IS a defensive touchdown), and adding it separately
        would double-count. Not verified with certainty (nflreadpy
        doesn't document the overlap), but double-counting was judged
        the worse failure mode than a small undercount here.
    """
    return team_stats.with_columns(
        (pl.col("fumble_recovery_opp")).alias("def_fumble_recoveries"),
        (pl.col("def_tds") + pl.col("special_teams_tds")).alias("def_touchdowns"),
        (pl.col("def_punt_blocks") + pl.col("def_pat_blocks") + pl.col("def_fg_blocks")).alias(
            "def_blocked_kicks"
        ),
        (pl.col("def_2pt_made")).alias("def_two_point_returns"),
        (pl.lit("DEF_") + pl.col("team")).alias("def_id"),
    )


def main() -> None:
    season = current_season()
    schedules = nfl.load_schedules(seasons=season)
    season, week = get_current_season_and_week(schedules)

    lookback_seasons = list(range(season - LOOKBACK_SEASONS, season + 1))

    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    projections = build_projections(stats, season, week, "player_id", STAT_COLUMNS, describe_player)
    history = build_history(stats, season, week, "player_id", STAT_COLUMNS)

    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week")
    )
    def_projections = build_projections(
        team_stats, season, week, "def_id", DEF_STAT_COLUMNS, describe_defense
    )
    def_history = build_history(team_stats, season, week, "def_id", DEF_STAT_COLUMNS)

    projections.update(def_projections)
    history.update(def_history)

    out_dir = DATA_DIR / f"week_{week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "projections.json").write_text(json.dumps(projections, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # The frontend is static and has no other way to know which week
    # folder is current -- it fetches this manifest first, then
    # data/week_{week}/*.json using the week number it names.
    #
    # espnSync reflects whether an espn-sync.json already exists for this
    # week (from an earlier successful run -- sync_espn.py runs after this
    # script and rewrites this flag to True the moment it succeeds today).
    # The frontend uses this to skip fetching espn-sync.json entirely when
    # it's False, instead of firing a request that's guaranteed to 404.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "latest.json").write_text(
        json.dumps(
            {
                "season": season,
                "week": week,
                "espnSync": (out_dir / "espn-sync.json").exists(),
            },
            indent=2,
        )
    )

    print(
        f"Wrote projections + history for season {season}, week {week}: "
        f"{len(projections) - len(def_projections)} players + "
        f"{len(def_projections)} team defenses -> {out_dir}/"
    )


if __name__ == "__main__":
    main()
