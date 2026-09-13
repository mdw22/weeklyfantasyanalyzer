"""
sync_espn.py

Run daily by .github/workflows/weekly-projections.yml, after
generate_projections.py (which must run first -- this script reads the
season/week it already wrote to public/data/latest.json rather than
recomputing it). Pulls the user's real ESPN league roster and this week's
opponent roster, translates ESPN's player IDs to the nflverse player_id
used everywhere else in this pipeline, and writes
public/data/week_{NN}/espn-sync.json for the frontend to read.

Deliberately soft-fails: a missing secret, auth failure, unexpected
response shape, or any other error prints a warning and exits 0 WITHOUT
writing anything (any previous espn-sync.json is left untouched) -- ESPN
sync is a display nicety layered on a pipeline that already works
standalone with manual roster entry, and it must never break the daily
projections commit.

ONE-TIME SETUP (done by the user, never through Claude, never pasted into
chat or committed):
  1. Log into fantasy.espn.com in a normal browser, with your league open.
  2. Open dev tools -> Application (Chrome) / Storage (Firefox) ->
     Cookies -> https://fantasy.espn.com.
  3. Copy the `espn_s2` and `SWID` cookie values.
  4. In this repo's GitHub Settings -> Secrets and variables -> Actions:
     add secrets ESPN_S2 and ESPN_SWID with those values.
  5. Add repo variables ESPN_LEAGUE_ID and ESPN_TEAM_ID (both visible,
     unauthenticated, in your league's own fantasy.espn.com URL -- not
     secret, just identifiers).
Sync silently stays off (falls back to manual entry) until all four are
set.

NOT VERIFIED AGAINST A REAL LEAGUE: this was written and reviewed against
publicly documented reverse-engineering of ESPN's fantasy API (the same
kind of unofficial-endpoint work as the leaguedefaults lookup used
elsewhere in this project -- see CLAUDE.md), but there is no way to test
it without a real ESPN account's own session cookies, which never reach
this environment. Before trusting the daily workflow, run this script
once locally with real secrets as env vars and check the printed roster
against your actual team by eye (see README's "ESPN Sync Setup" section).
The two things most likely to need adjusting after that first real run:
  - ESPN_SLOT_CATEGORY below (lineupSlotId -> our position category) --
    widely cited reverse-engineered values, but never confirmed against
    an actual API response in this environment.
  - The JSON path assumptions in extract_roster() / find_opponent_team_id()
    (e.g. `team["roster"]["entries"]`, `matchup["home"]["teamId"]`) --
    ESPN's response shape is undocumented and could differ by a level of
    nesting from what's assumed here.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

DATA_DIR = Path("public/data")

# ESPN's lineupSlotId -> position category. Reverse-engineered / widely
# cited in community ESPN API projects, not from official docs -- see the
# module docstring's "NOT VERIFIED" note.
ESPN_SLOT_CATEGORY = {
    0: "QB",
    2: "RB",
    4: "WR",
    6: "TE",
    23: "FLEX",
    16: "DEF",
    17: "K",
    20: "BENCH",
    21: "IR",
}

# Maps each position category to OUR roster slot ids, in the order
# ESPN's players for that category get assigned (first RB seen -> RB1,
# second -> RB2, etc). MUST stay in sync with src/lib/rosterSlots.js's
# ROSTER_SLOTS ordering -- there is no shared source of truth between
# this Python script and that JS file, the same class of duplication gap
# flagged for the K/DEF cross-position discount in the prior project (see
# CLAUDE.md). If the roster shape changes in rosterSlots.js, update here
# too.
SLOT_ID_ORDER = {
    "QB": ["QB"],
    "RB": ["RB1", "RB2"],
    "TE": ["TE"],
    "WR": ["WR1", "WR2"],
    "FLEX": ["FLEX"],
    "DEF": ["DEF"],
    "K": ["K"],
    "BENCH": [f"BN{i}" for i in range(1, 8)],
    "IR": ["IR"],
}


def fetch_espn_league(season, league_id, espn_s2, swid):
    # NOT fantasy.espn.com -- that host 302-redirects (to a marketing page,
    # which then 403s) rather than serving the API directly. Confirmed
    # working host for reads, from the prior project's own working
    # leaguedefaults lookup (see CLAUDE.md): lm-api-reads.fantasy.espn.com.
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )
    resp = requests.get(
        url,
        params={"view": ["mRoster", "mMatchup", "mTeam"]},
        cookies={"espn_s2": espn_s2, "SWID": swid},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def build_espn_to_gsis_map():
    """Cross-platform player ID lookup, ESPN included -- same source used
    for the FantasyPros consensus-ranking comparison work in the prior
    project (see CLAUDE.md)."""
    ids = nfl.load_ff_playerids().filter(
        pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null()
    )
    mapping = {}
    dupes = set()
    for row in ids.iter_rows(named=True):
        espn_id = str(row["espn_id"])
        if espn_id in mapping and mapping[espn_id] != row["gsis_id"]:
            dupes.add(espn_id)
        mapping[espn_id] = row["gsis_id"]
    if dupes:
        print(
            f"WARNING: {len(dupes)} espn_id value(s) map to more than one gsis_id "
            f"in nflreadpy's crosswalk -- using the last one seen: {sorted(dupes)}"
        )
    return mapping


def find_opponent_team_id(league_json, my_team_id, week):
    for matchup in league_json.get("schedule", []):
        if matchup.get("matchupPeriodId") != week:
            continue
        home = matchup.get("home", {})
        away = matchup.get("away", {})
        if home.get("teamId") == my_team_id:
            return away.get("teamId")
        if away.get("teamId") == my_team_id:
            return home.get("teamId")
    return None


def extract_roster(team_json, espn_to_gsis):
    """Translates one ESPN team's roster into our {slot, playerId} shape.
    Unmatched players are logged and skipped -- loud, not silent, per the
    build spec (a few percent mismatch on deep bench/D-ST is realistic)."""
    entries = team_json.get("roster", {}).get("entries", [])
    by_category = defaultdict(list)

    for entry in entries:
        raw_slot = entry.get("lineupSlotId")
        category = ESPN_SLOT_CATEGORY.get(raw_slot)
        player = entry.get("playerPoolEntry", {}).get("player", {})
        name = player.get("fullName", "?")
        espn_player_id = str(entry.get("playerId"))

        if category is None:
            print(f"WARNING: unknown ESPN lineupSlotId {raw_slot} for {name} -- skipping")
            continue

        gsis_id = espn_to_gsis.get(espn_player_id)
        if gsis_id is None:
            print(
                f"WARNING: no nflverse player_id match for ESPN player "
                f"{name} (espn_id={espn_player_id}) -- skipping"
            )
            continue

        by_category[category].append(gsis_id)

    roster_out = []
    for category, our_slot_ids in SLOT_ID_ORDER.items():
        players = by_category.get(category, [])
        if len(players) > len(our_slot_ids):
            print(
                f"WARNING: ESPN roster has {len(players)} {category} player(s) but "
                f"only {len(our_slot_ids)} matching slot(s) -- extra one(s) dropped"
            )
        for slot_id, gsis_id in zip(our_slot_ids, players):
            roster_out.append({"slot": slot_id, "playerId": gsis_id})

    return roster_out


def main():
    espn_s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("ESPN_SWID")
    league_id = os.environ.get("ESPN_LEAGUE_ID")
    my_team_id_raw = os.environ.get("ESPN_TEAM_ID")

    if not all([espn_s2, swid, league_id, my_team_id_raw]):
        print(
            "ESPN sync not configured (need ESPN_S2, ESPN_SWID, ESPN_LEAGUE_ID, "
            "ESPN_TEAM_ID) -- skipping, manual roster entry stays in effect."
        )
        return

    try:
        my_team_id = int(my_team_id_raw)

        latest_path = DATA_DIR / "latest.json"
        latest = json.loads(latest_path.read_text())
        season, week = latest["season"], latest["week"]

        league_json = fetch_espn_league(season, league_id, espn_s2, swid)

        teams = {t["id"]: t for t in league_json.get("teams", [])}
        if my_team_id not in teams:
            print(f"WARNING: ESPN_TEAM_ID {my_team_id} not found in league {league_id} -- skipping.")
            return

        opponent_team_id = find_opponent_team_id(league_json, my_team_id, week)
        if opponent_team_id is None or opponent_team_id not in teams:
            print(f"WARNING: could not find a week {week} opponent for team {my_team_id} -- skipping.")
            return

        espn_to_gsis = build_espn_to_gsis_map()

        my_roster = extract_roster(teams[my_team_id], espn_to_gsis)
        opponent_roster = extract_roster(teams[opponent_team_id], espn_to_gsis)

        out = {
            "syncedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "myTeam": {"espnTeamId": my_team_id, "roster": my_roster},
            "opponent": {"espnTeamId": opponent_team_id, "roster": opponent_roster},
        }

        out_dir = DATA_DIR / f"week_{week:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "espn-sync.json").write_text(json.dumps(out, indent=2))

        # generate_projections.py (which runs before this script) already
        # wrote latest.json's espnSync flag based on whether a file from a
        # PRIOR day existed -- flip it to True now so the frontend doesn't
        # wait until tomorrow's run to know today's sync succeeded.
        if not latest.get("espnSync"):
            latest["espnSync"] = True
            latest_path.write_text(json.dumps(latest, indent=2))

        print(
            f"Synced ESPN rosters for week {week}: {len(my_roster)} of my players, "
            f"{len(opponent_roster)} opponent players -> {out_dir}/espn-sync.json"
        )

    except Exception as exc:  # noqa: BLE001 -- soft-fail is the whole point, see docstring
        print(f"WARNING: ESPN sync failed ({exc!r}) -- leaving any previous espn-sync.json untouched.")


if __name__ == "__main__":
    main()
