"""
sync_live_scores.py

Run every ~10 minutes during game windows by
.github/workflows/live-scores.yml. Writes public/data/week_{NN}/live.json:
each rostered player's real, currently-accumulated stat line plus their
game's status, for the frontend to show instead of the projection once a
game has started.

Two sources, deliberately different:
  - STATS come from ESPN's authenticated league box score (same
    ESPN_S2/ESPN_SWID/league as sync_espn.py, just `view=mBoxscore`).
    nflreadpy settles stats after the fact, so it can't do live. RAW
    per-stat amounts are used, never ESPN's baked-in fantasy points, so
    the frontend's computeFantasyPoints() keeps honoring any custom
    scoring settings.
  - GAME STATUS + CLOCK come from ESPN's public scoreboard (no login).
    It exposes state (pre/in/post), period and displayClock directly, which
    is more reliable than digging a status out of the fantasy response.

Soft-fails like sync_espn.py: any problem prints a warning, writes nothing,
exits 0. Only rewrites live.json when the players data actually changed, so
quiet ticks make no commit.

DEBUG: set LIVE_SCORES_DEBUG=1 (workflow_dispatch checkbox) to print, once,
the raw shape of a box-score entry and a per-player self-check: points
computed from our mapped stats under the app's default scoring vs ESPN's own
`appliedTotal`. Run it on a day with completed games. A MISMATCH means a
statId is mapped wrong, or league scoring differs from the app's defaults.

NOT VERIFIED against a real box-score response (needs the user's ESPN
cookies): the `mBoxscore` JSON paths, and the K and DEF statIds below. The
offense statIds are widely cited and match what the league's scoring
settings showed; K/DEF are from memory of community reverse-engineering. The
debug self-check exists to catch exactly this.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from generate_projections import POINTS_ALLOWED_TIERS, YARDS_ALLOWED_TIERS, tier_bonus_value
from sync_espn import ESPN_PRO_TEAM_ABBR, build_espn_to_gsis_map, resolve_def_team_id

DATA_DIR = Path("public/data")

# ESPN statId -> our STAT_FIELDS key (multiple ids may feed one field).
STAT_ID_TO_FIELD = {
    # Offense
    3: "passing_yards",
    4: "passing_tds",
    20: "passing_interceptions",
    24: "rushing_yards",
    25: "rushing_tds",
    42: "receiving_yards",
    43: "receiving_tds",
    53: "receptions",
    72: "fumbles_lost_total",
    # Kicker -- UNVERIFIED, see docstring
    80: "fg_made_0_39",
    77: "fg_made_40_49",
    74: "fg_made_50_plus",
    85: "fg_missed_total",
    86: "pat_made",
    # Defense -- UNVERIFIED, see docstring
    99: "def_sacks",
    95: "def_interceptions",
    96: "def_fumble_recoveries",
    98: "def_safeties",
    97: "def_blocked_kicks",
    93: "def_touchdowns",
    101: "def_touchdowns",
    102: "def_touchdowns",
    103: "def_touchdowns",
    104: "def_touchdowns",
}
# Raw running totals; converted to tier bonuses with the same league tables
# the daily pipeline uses. UNVERIFIED ids, see docstring.
RAW_POINTS_ALLOWED_ID = 120
RAW_YARDS_ALLOWED_ID = 127

# ESPN scoreboard abbreviations that differ from nflverse's (which the rest
# of this app uses).
SCOREBOARD_ABBR_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}

# Mirror of src/lib/scoring.js's default (Full PPR) values -- used ONLY by
# the debug self-check. Another hand-kept cross-language duplicate.
SELF_CHECK_VALUES = {
    "passing_yards": 0.04, "passing_tds": 4, "passing_interceptions": -2,
    "rushing_yards": 0.1, "rushing_tds": 6, "receiving_yards": 0.1,
    "receiving_tds": 6, "receptions": 1, "fumbles_lost_total": -2,
    "def_sacks": 1, "def_interceptions": 2, "def_fumble_recoveries": 2,
    "def_safeties": 2, "def_touchdowns": 6, "def_blocked_kicks": 2,
    "def_two_point_returns": 2, "def_points_allowed_bonus": 1,
    "def_yards_allowed_bonus": 1, "fg_made_0_39": 3, "fg_made_40_49": 4,
    "fg_made_50_plus": 5, "fg_missed_total": -1, "pat_made": 1,
}


# ---------------------------------------------------------------- fetching

def fetch_boxscore(season, league_id, espn_s2, swid, week):
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )
    resp = requests.get(
        url,
        params={"view": ["mBoxscore", "mMatchupScore"], "scoringPeriodId": week},
        cookies={"espn_s2": espn_s2, "SWID": swid},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_scoreboard(season, week):
    resp = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        params={"dates": season, "seasontype": 2, "week": week},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


# ----------------------------------------------------------------- parsing

def _clock_text(status: dict) -> str:
    stype = status.get("type", {})
    name = stype.get("name", "")
    period = status.get("period", 0)
    if name == "STATUS_HALFTIME":
        return "Half"
    if name == "STATUS_END_PERIOD":
        return f"End Q{period}" if period <= 4 else "End OT"
    if name == "STATUS_DELAYED" or "Delayed" in stype.get("shortDetail", ""):
        return "Delayed"
    label = f"Q{period}" if period <= 4 else "OT"
    return f"{label} {status.get('displayClock', '')}".strip()


def game_status_by_team(scoreboard: dict) -> dict:
    """{nflverse team abbr: {"status": ..., "clock"?: ...}} from the public
    scoreboard's per-game state."""
    out = {}
    for event in scoreboard.get("events", []):
        status = event.get("status", {})
        state = status.get("type", {}).get("state")
        if state == "post":
            info = {"status": "final"}
        elif state == "in":
            info = {"status": "in_progress", "clock": _clock_text(status)}
        else:
            info = {"status": "not_started"}
        for comp in event.get("competitions", [{}])[0].get("competitors", []):
            abbr = comp.get("team", {}).get("abbreviation")
            if abbr:
                out[SCOREBOARD_ABBR_FIX.get(abbr, abbr)] = info
    return out


def actual_stat_entry(player_stats, week):
    """ESPN's per-period ACTUAL stats: statSourceId 0 (1 = projected),
    statSplitTypeId 1 (single scoring period)."""
    for s in player_stats or []:
        if (
            s.get("statSourceId") == 0
            and s.get("statSplitTypeId") == 1
            and s.get("scoringPeriodId") == week
        ):
            return s
    return None


def map_stats(raw: dict, is_def: bool) -> tuple:
    """(our stat fields, {statId: value} of nonzero ids we didn't map)."""
    fields, unmapped = {}, {}
    for sid_str, value in (raw or {}).items():
        try:
            sid = int(sid_str)
        except (TypeError, ValueError):
            continue
        if sid in STAT_ID_TO_FIELD:
            field = STAT_ID_TO_FIELD[sid]
            fields[field] = fields.get(field, 0) + value
        elif sid not in (RAW_POINTS_ALLOWED_ID, RAW_YARDS_ALLOWED_ID) and value:
            unmapped[sid] = value
    if is_def:
        # Absent means "not reported yet", NOT zero -- 0 points allowed is
        # the best bracket, so never default a missing value to 0.
        if str(RAW_POINTS_ALLOWED_ID) in raw:
            fields["def_points_allowed_bonus"] = tier_bonus_value(
                raw[str(RAW_POINTS_ALLOWED_ID)], POINTS_ALLOWED_TIERS
            )
        if str(RAW_YARDS_ALLOWED_ID) in raw:
            fields["def_yards_allowed_bonus"] = tier_bonus_value(
                raw[str(RAW_YARDS_ALLOWED_ID)], YARDS_ALLOWED_TIERS
            )
    return fields, unmapped


def matchup_entries(league_json: dict, my_team_id: int, week: int) -> tuple:
    """(entries from both sides of my matchup, which roster key was used)."""
    for matchup in league_json.get("schedule", []):
        if matchup.get("matchupPeriodId") != week:
            continue
        sides = [matchup.get("home", {}), matchup.get("away", {})]
        if not any(side.get("teamId") == my_team_id for side in sides):
            continue
        entries, used = [], None
        for side in sides:
            for key in ("rosterForCurrentScoringPeriod", "rosterForMatchupPeriod"):
                found = side.get(key, {}).get("entries")
                if found:
                    entries.extend(found)
                    used = used or key
                    break
        return entries, used
    return [], None


def build_live_players(entries, week, scoreboard_status, espn_to_gsis, debug_rows=None):
    players = {}
    for entry in entries:
        pool = entry.get("playerPoolEntry", {})
        player = pool.get("player", {})
        name = player.get("fullName", "?")
        espn_player_id = str(entry.get("playerId", pool.get("id")))
        # Team defenses: defaultPositionId 16, and a negative espn_id.
        is_def = player.get("defaultPositionId") == 16 or espn_player_id.startswith("-")

        our_id = resolve_def_team_id(espn_player_id) if is_def else espn_to_gsis.get(espn_player_id)
        if our_id is None:
            print(f"WARNING: live scores: no id match for {name} (espn_id={espn_player_id}) -- skipping")
            continue

        team = our_id.removeprefix("DEF_") if is_def else ESPN_PRO_TEAM_ABBR.get(player.get("proTeamId"))
        info = scoreboard_status.get(team, {"status": "not_started"})

        stat_entry = actual_stat_entry(player.get("stats"), week)
        raw = (stat_entry or {}).get("stats", {}) or {}
        fields, unmapped = map_stats(raw, is_def)

        if info["status"] == "not_started":
            players[our_id] = {"status": "not_started"}
        else:
            players[our_id] = {**info, "stats": fields}

        if debug_rows is not None:
            debug_rows.append({
                "name": name, "id": our_id, "info": info, "raw": raw, "fields": fields,
                "unmapped": unmapped, "espn_total": (stat_entry or {}).get("appliedTotal"),
            })
    return players


# ------------------------------------------------------------------- debug

def print_debug(entries, rows, roster_key):
    print(f"[debug] {len(entries)} box-score entries; roster key used: {roster_key}")
    if entries:
        blob = json.dumps(entries[0], indent=1, default=str)
        print("[debug] first raw entry (truncated):")
        print(blob[:3500] + ("\n... (truncated)" if len(blob) > 3500 else ""))
    print("[debug] self-check: points from our mapped stats (app default scoring) vs ESPN appliedTotal")
    for r in rows:
        if r["info"]["status"] == "not_started":
            print(f"  {r['name']:<26} not_started")
            continue
        mine = sum(SELF_CHECK_VALUES.get(k, 0) * v for k, v in r["fields"].items())
        espn = r["espn_total"]
        verdict = "n/a (no ESPN total)" if espn is None else (
            "OK" if abs(mine - espn) <= 0.51 else f"MISMATCH (diff {mine - espn:+.2f})"
        )
        print(f"  {r['name']:<26} {r['info']['status']:<11} ours={mine:6.2f} espn={espn} -> {verdict}")
        if verdict.startswith("MISMATCH"):
            print(f"      raw statIds: {r['raw']}")
            print(f"      unmapped nonzero statIds: {r['unmapped']}")


# -------------------------------------------------------------------- main

def write_if_changed(week_dir: Path, players: dict) -> bool:
    path = week_dir / "live.json"
    if path.exists():
        try:
            if json.loads(path.read_text()).get("players") == players:
                return False
        except ValueError:
            pass
    week_dir.mkdir(parents=True, exist_ok=True)
    payload = {"updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "players": players}
    path.write_text(json.dumps(payload, indent=2))
    return True


def run(env, fetch_box=fetch_boxscore, fetch_board=fetch_scoreboard, crosswalk=build_espn_to_gsis_map) -> str:
    latest_path = DATA_DIR / "latest.json"
    latest = json.loads(latest_path.read_text())
    season, week = latest["season"], latest["week"]
    debug = env.get("LIVE_SCORES_DEBUG") == "1"

    league_json = fetch_box(season, env["ESPN_LEAGUE_ID"], env["ESPN_S2"], env["ESPN_SWID"], week)
    entries, roster_key = matchup_entries(league_json, int(env["ESPN_TEAM_ID"]), week)
    if not entries:
        return "no box-score entries found for this matchup -- nothing written"

    try:
        scoreboard_status = game_status_by_team(fetch_board(season, week))
    except Exception as exc:  # noqa: BLE001 -- status is best-effort
        print(f"WARNING: scoreboard unavailable ({exc!r}); treating every game as not started")
        scoreboard_status = {}

    rows = [] if debug else None
    players = build_live_players(entries, week, scoreboard_status, crosswalk(), rows)
    if debug:
        print_debug(entries, rows, roster_key)

    changed = write_if_changed(DATA_DIR / f"week_{week:02d}", players)
    if changed and not latest.get("liveScores"):
        latest["liveScores"] = True
        latest_path.write_text(json.dumps(latest, indent=2))
    started = sum(1 for p in players.values() if p["status"] != "not_started")
    return f"{len(players)} players ({started} started); live.json {'updated' if changed else 'unchanged'}"


def main():
    env = {k: os.environ.get(k) for k in
           ("ESPN_S2", "ESPN_SWID", "ESPN_LEAGUE_ID", "ESPN_TEAM_ID", "LIVE_SCORES_DEBUG")}
    if not all(env[k] for k in ("ESPN_S2", "ESPN_SWID", "ESPN_LEAGUE_ID", "ESPN_TEAM_ID")):
        print("ESPN not configured -- skipping live scores.")
        return
    try:
        print(f"Live scores: {run(env)}")
    except Exception as exc:  # noqa: BLE001 -- soft-fail, see docstring
        print(f"WARNING: live scores failed ({exc!r}) -- leaving any existing live.json untouched.")


if __name__ == "__main__":
    main()
