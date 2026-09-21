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

UPDATE (first real run against the user's actual league, 2026 season):
the host in fetch_espn_league() was originally `fantasy.espn.com`, which
302-redirected to a marketing page and 403'd -- fixed to
`lm-api-reads.fantasy.espn.com` (the same host the prior project's own
leaguedefaults lookup already used, see CLAUDE.md). After that fix, auth
and the response shape both worked: `ESPN_SLOT_CATEGORY`, the
`team["roster"]["entries"]` / `matchup["home"]["teamId"]` JSON paths, and
15-of-17 real players matched correctly per team. The only real gap
found was team defenses (ESPN espn_id is negative and has no gsis_id at
all -- see `resolve_def_team_id()` and `ESPN_PRO_TEAM_ABBR` below, now
fixed and confirmed against two real espn_id values from this same
league). So this script IS now verified against a real league, not just
reviewed -- the risk profile here is much lower than when this docstring
was first written.
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

# ESPN's team defense "players" have no gsis_id at all (they're not
# individual players, so nflreadpy's load_ff_playerids() crosswalk can
# never match them) -- their espn_id instead encodes ESPN's own numeric
# proTeamId as a negative number. CONFIRMED against two real espn_id
# values pulled from an actual league (not just cited elsewhere):
#   -16030 -> 16030 - 16000 = 30 -> Jacksonville
#   -16033 -> 16033 - 16000 = 33 -> Baltimore
# both matched this exact `-(16000 + proTeamId)` formula with zero
# deviation. The proTeamId -> abbreviation table itself is the standard
# reverse-engineered ESPN mapping cited across community ESPN API
# projects (unchanged for years); abbreviations below are normalized to
# nflreadpy's own team codes (confirmed via load_team_stats()), which use
# "LA" not "LAR", "WAS" not "WSH", "LV" not "OAK", "LAC" not "SD".
ESPN_PRO_TEAM_ABBR = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}


def resolve_def_team_id(espn_player_id: str) -> str | None:
    """ESPN D/ST espn_id -> our `DEF_{team}` id (must match
    generate_projections.py's `DEF_{team}` scheme exactly)."""
    try:
        raw = int(espn_player_id)
    except (TypeError, ValueError):
        return None
    pro_team_id = -raw - 16000
    abbr = ESPN_PRO_TEAM_ABBR.get(pro_team_id)
    return f"DEF_{abbr}" if abbr else None


def log_scoring_settings(season, league_id, espn_s2, swid):
    """One-off diagnostic (opt-in via ESPN_LOG_SCORING=1): prints the
    league's real scoring rules so the DEF points/yards-allowed tier
    tables in generate_projections.py can be checked against them. Prints
    every statId with its points, plus any per-position overrides -- the
    D/ST points-allowed and yards-allowed brackets are a contiguous run of
    statIds in here, but the ids are undocumented, so DON'T map them from
    memory: cross-check the values against ESPN's UI (Settings -> Scoring
    -> Team Defense & Special Teams) to confirm which id is which."""
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )
    resp = requests.get(
        url,
        params={"view": "mSettings"},
        cookies={"espn_s2": espn_s2, "SWID": swid},
        timeout=20,
    )
    resp.raise_for_status()
    items = resp.json().get("settings", {}).get("scoringSettings", {}).get("scoringItems", [])
    print(f"ESPN scoringItems ({len(items)}):")
    for item in sorted(items, key=lambda i: i.get("statId", -1)):
        extra = f" overrides={item['pointsOverrides']}" if item.get("pointsOverrides") else ""
        print(f"  statId={item.get('statId')} points={item.get('points')}{extra}")


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


def resolve_entry_id(entry, espn_to_gsis):
    """(our id or None, display name, espn player id) for one roster entry.

    Team defenses are recognized from the PLAYER (defaultPositionId 16 /
    a negative espn id), not from the lineup slot: a D/ST on the bench or
    IR has slot category BENCH/IR, and going by slot would send it down the
    individual-player crosswalk, where it can never match."""
    player = entry.get("playerPoolEntry", {}).get("player", {})
    name = player.get("fullName", "?")
    espn_player_id = str(entry.get("playerId"))
    is_def = player.get("defaultPositionId") == 16 or espn_player_id.startswith("-")
    our_id = resolve_def_team_id(espn_player_id) if is_def else espn_to_gsis.get(espn_player_id)
    return our_id, name, espn_player_id


def extract_roster(team_json, espn_to_gsis):
    """Translates one ESPN team's roster into our {slot, playerId} shape.
    Unmatched players are logged and skipped -- loud, not silent, per the
    build spec (a few percent mismatch on deep bench/D-ST is realistic)."""
    entries = team_json.get("roster", {}).get("entries", [])
    by_category = defaultdict(list)

    for entry in entries:
        raw_slot = entry.get("lineupSlotId")
        category = ESPN_SLOT_CATEGORY.get(raw_slot)
        our_id, name, espn_player_id = resolve_entry_id(entry, espn_to_gsis)

        if category is None:
            print(f"WARNING: unknown ESPN lineupSlotId {raw_slot} for {name} -- skipping")
            continue
        if our_id is None:
            print(
                f"WARNING: couldn't match ESPN player {name} "
                f"(espn_id={espn_player_id}) to an nflverse/defense id -- skipping"
            )
            continue

        by_category[category].append(our_id)

    roster_out = []
    for category, our_slot_ids in SLOT_ID_ORDER.items():
        players = by_category.get(category, [])
        if len(players) > len(our_slot_ids):
            print(
                f"WARNING: ESPN roster has {len(players)} {category} player(s) but "
                f"only {len(our_slot_ids)} matching slot(s) -- extra one(s) dropped"
            )
        for slot_id, resolved_id in zip(our_slot_ids, players):
            roster_out.append({"slot": slot_id, "playerId": resolved_id})

    return roster_out


def extract_ownership(teams, espn_to_gsis):
    """{our id: {"owned": True, "espnTeamId": n}} for every player on ANY
    of the league's teams (bench and IR included) -- everyone absent from
    this map is a free agent, which is what the Lineup Advisor recommends
    from. The sync response already carries every team's roster, so this
    costs no extra ESPN request.

    A rostered player we fail to match would wrongly read as a free agent
    (and could be recommended), so unmatched ones are logged by name."""
    owned, unmatched = {}, []
    for team_id, team_json in teams.items():
        for entry in team_json.get("roster", {}).get("entries", []):
            our_id, name, _ = resolve_entry_id(entry, espn_to_gsis)
            if our_id is None:
                unmatched.append(f"{name} (team {team_id})")
            else:
                owned[our_id] = {"owned": True, "espnTeamId": team_id}
    if unmatched:
        shown = ", ".join(unmatched[:10]) + (" ..." if len(unmatched) > 10 else "")
        print(
            f"WARNING: {len(unmatched)} rostered player(s) unmatched for ownership "
            f"-- they'll wrongly look like free agents: {shown}"
        )
    return owned


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

        if os.environ.get("ESPN_LOG_SCORING") == "1":
            try:
                log_scoring_settings(season, league_id, espn_s2, swid)
            except Exception as exc:  # noqa: BLE001 -- diagnostic only, never block the sync
                print(f"WARNING: couldn't log ESPN scoring settings ({exc!r})")

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
            "ownership": extract_ownership(teams, espn_to_gsis),
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
            f"{len(opponent_roster)} opponent players, "
            f"{len(out['ownership'])} rostered league-wide -> {out_dir}/espn-sync.json"
        )

    except Exception as exc:  # noqa: BLE001 -- soft-fail is the whole point, see docstring
        print(f"WARNING: ESPN sync failed ({exc!r}) -- leaving any previous espn-sync.json untouched.")


if __name__ == "__main__":
    main()
