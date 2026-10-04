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
from datetime import date, datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

import model_core

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
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
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
    "def_points_allowed_bonus",
    "def_yards_allowed_bonus",
]

# Bonus-per-game tier tables for DEF points/yards allowed, as
# (lowest value in bracket, bonus points), ascending. CONFIRMED against
# the user's real league (read off ESPN's Settings -> Scoring -> Team
# Defense & Special Teams), not defaults. ESPN's UI splits points allowed
# 18-21 / 22-27 into two brackets that are both worth 0, merged here.
POINTS_ALLOWED_TIERS = [(0, 5), (1, 4), (7, 3), (14, 1), (18, 0), (28, -1), (35, -3), (46, -5)]
YARDS_ALLOWED_TIERS = [(0, 5), (100, 3), (200, 2), (300, 0), (350, -1), (400, -3), (450, -5), (500, -6), (550, -7)]

# Kicker output fields, matching src/lib/scoring.js's "Kicking" group.
# Kickers get their own build pass (like DEF) so the ~2,500 non-kickers
# don't each carry five always-zero fields.
K_STAT_COLUMNS = [
    "fg_made_0_39",
    "fg_made_40_49",
    "fg_made_50_plus",
    "fg_missed_total",
    "pat_made",
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


def build_projections(df: pl.DataFrame, season: int, week: int, id_col: str, stat_columns: list, describe_row,
                       n_games: int | None = N_GAMES) -> dict:
    """v1 baseline: each entity's (player OR team-defense) projection =
    mean of their last N_GAMES games across `stat_columns`. No opponent
    adjustment yet. `describe_row` turns one row into the entry's
    name/position/team fields.

    `n_games=None` lifts the games-back cap entirely -- used by
    `scripts/backtest_evaluation.py` for the "plain full-season average"
    baseline (source doc's other suggested baseline), reusing this exact
    function/filter rather than a second implementation. Production
    (`main()`, below) never passes this -- default is unchanged."""
    past = df.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    cap = n_games if n_games is not None else past.height + 1

    projections = {}
    for entity_id, games in _partition_by(past, id_col).items():
        recent = games.sort(["season", "week"], descending=True).head(cap)
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
        recent = games.sort(["season", "week"], descending=True).head(N_GAMES)
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


def tier_bonus_value(value: float, tiers: list) -> int:
    """Plain-Python twin of _tier_bonus, for one value (used by
    sync_live_scores.py so live DEF bonuses use the same tier tables)."""
    bonus = tiers[0][1]
    for lower, b in tiers[1:]:
        if value >= lower:
            bonus = b
    return bonus


def _tier_bonus(col: str, tiers: list) -> pl.Expr:
    """Maps a per-game value through a (lower_bound, bonus) bracket table."""
    expr = pl.lit(tiers[0][1])
    for lower, bonus in tiers[1:]:
        expr = pl.when(pl.col(col) >= lower).then(bonus).otherwise(expr)
    return expr


def add_def_stat_columns(team_stats: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
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

    Points/yards-allowed bonuses are computed PER GAME and then averaged
    like every other stat (rather than averaging points/yards allowed and
    mapping that average through the tier table): the tiers are
    non-linear, so mapping an average misstates the expected bonus, and
    per-game bonuses in history.json give the Monte Carlo real variance.
      - points allowed = the opponent's final score (load_schedules).
      - yards allowed = opponent's passing + rushing yards + sack_yards_lost
        (which is negative, so this is net yardage). Checked on 2025 data:
        327 yds and 23 pts per team-game league-wide, and team codes and
        game_ids join 100% between load_team_stats and load_schedules.
    """
    pts_allowed = pl.concat([
        schedules.select("game_id", pl.col("home_team").alias("team"), pl.col("away_score").alias("points_allowed")),
        schedules.select("game_id", pl.col("away_team").alias("team"), pl.col("home_score").alias("points_allowed")),
    ])
    opp_yards = team_stats.select(
        "game_id",
        pl.col("team").alias("opponent_team"),
        (pl.col("passing_yards") + pl.col("rushing_yards") + pl.col("sack_yards_lost")).alias("yards_allowed"),
    )
    team_stats = team_stats.join(pts_allowed, on=["game_id", "team"], how="left").join(
        opp_yards, on=["game_id", "opponent_team"], how="left"
    )
    return team_stats.with_columns(
        _tier_bonus("points_allowed", POINTS_ALLOWED_TIERS).alias("def_points_allowed_bonus"),
        _tier_bonus("yards_allowed", YARDS_ALLOWED_TIERS).alias("def_yards_allowed_bonus"),
        (pl.col("fumble_recovery_opp")).alias("def_fumble_recoveries"),
        (pl.col("def_tds") + pl.col("special_teams_tds")).alias("def_touchdowns"),
        (pl.col("def_punt_blocks") + pl.col("def_pat_blocks") + pl.col("def_fg_blocks")).alias(
            "def_blocked_kicks"
        ),
        (pl.col("def_2pt_made")).alias("def_two_point_returns"),
        (pl.lit("DEF_") + pl.col("team")).alias("def_id"),
    )


def add_k_stat_columns(kickers: pl.DataFrame) -> pl.DataFrame:
    """Collapses nflreadpy's per-distance field-goal columns into the
    buckets FG scoring actually uses (3 / 4 / 5 pts by distance), and
    combines misses with blocks for the -1 penalty. Verified against real
    2025 data: fg_att == fg_made + fg_missed + fg_blocked on every one of
    569 kicker-games (so blocks are NOT already inside fg_missed and must
    be added), and the six distance buckets sum exactly to fg_made. A
    missed/blocked PAT carries no penalty, so only pat_made is kept."""
    return kickers.with_columns(
        (pl.col("fg_made_0_19") + pl.col("fg_made_20_29") + pl.col("fg_made_30_39")).alias("fg_made_0_39"),
        (pl.col("fg_made_50_59") + pl.col("fg_made_60_")).alias("fg_made_50_plus"),
        (pl.col("fg_missed") + pl.col("fg_blocked")).alias("fg_missed_total"),
    )


def _injury_status(report_status) -> str | None:
    text = (report_status or "").strip().lower()
    for prefix, status in (("out", "OUT"), ("doubtful", "DOUBTFUL"), ("questionable", "QUESTIONABLE")):
        if text.startswith(prefix):
            return status
    return None


def load_availability(season: int, week: int) -> tuple[dict, set, set]:
    """(injury status by gsis_id, active-roster ids, IR/reserve ids) from
    nflreadpy -- public, keyed by the same gsis_id as everything else here.

    Verified against real 2026 data: load_injuries has one row per player
    per week, `report_status` is Questionable/Doubtful/Out/None (None =
    only practice participation, treated as healthy), and Puka Nacua
    (Questionable) / Zay Flowers (Doubtful) show up correctly.

    - Injuries: CURRENT week only. Last week's game designations must not
      carry over -- and before the new week's report is published there are
      simply no rows, meaning everyone reads as healthy, not stale.
    - IR: `status == "RES"` in weekly rosters (load_injuries stops emitting
      rows once someone is on IR -- see CLAUDE.md). Uses the latest weekly
      roster at or before the current week, since IR stays IR for weeks.
    - Active: `status == "ACT"` (the ~1,700 players on real 53-man
      rosters). Needed because projections.json holds anyone with stats in
      the last 3 seasons -- 343 QB/RB/WR/TE in it aren't on an active roster
      (retired, cut, practice squad, IR) and several project well, so
      without this they'd surface as the best "free agents".
    Any failure returns empty sets: the advisor then simply has less to
    say, and the rest of the pipeline is unaffected.
    """
    injuries, active, reserve = {}, set(), set()
    try:
        inj = nfl.load_injuries(seasons=[season]).filter(pl.col("week") == week)
        for row in inj.iter_rows(named=True):
            status = _injury_status(row.get("report_status"))
            if status and row.get("gsis_id"):
                injuries[row["gsis_id"]] = status
    except Exception as exc:  # noqa: BLE001 -- availability is best-effort
        print(f"WARNING: injury report unavailable ({exc!r}); everyone reads as healthy")
    try:
        rosters = nfl.load_rosters_weekly(seasons=[season]).filter(pl.col("week") <= week)
        if rosters.height:
            latest = rosters.filter(pl.col("week") == rosters["week"].max())
            for row in latest.select("gsis_id", "status").iter_rows(named=True):
                if not row["gsis_id"]:
                    continue
                if row["status"] == "ACT":
                    active.add(row["gsis_id"])
                elif row["status"] == "RES":
                    reserve.add(row["gsis_id"])
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: weekly rosters unavailable ({exc!r}); no active/IR flags written")
    return injuries, active, reserve


def apply_availability(projections: dict, schedules: pl.DataFrame, season: int, week: int,
                       injuries: dict, active: set, reserve: set) -> None:
    """Adds lineup-advisor fields IN PLACE, omitting defaults to keep the
    file small (a missing field means the default):
      injury_status: QUESTIONABLE | DOUBTFUL | OUT | IR   (default ACTIVE)
      on_bye:        true when the player's team has no game this week
      active:        true when on a real active NFL roster (defenses always)
    """
    all_teams = set(schedules["home_team"].to_list()) | set(schedules["away_team"].to_list())
    this_week = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    playing = set(this_week["home_team"].to_list()) | set(this_week["away_team"].to_list())

    for pid, entry in projections.items():
        if entry["position"] == "DEF" or pid in active:
            entry["active"] = True
        status = "IR" if pid in reserve else injuries.get(pid)
        if status:
            entry["injury_status"] = status
        # `playing` empty means the schedule had nothing for this week
        # (e.g. offseason) -- don't mark the whole league as on a bye.
        if playing and entry.get("team") in all_teams and entry["team"] not in playing:
            entry["on_bye"] = True


def completed_weeks_before(schedules: pl.DataFrame, season: int, week: int) -> list[tuple[int, int]]:
    """Every (season, week) strictly before the target whose games are all
    final, oldest first -- the same week set the backtest iterates."""
    out = []
    for s in sorted(schedules["season"].unique().to_list()):
        ss = schedules.filter(pl.col("season") == s)
        for w in sorted(ss["week"].unique().to_list()):
            if (s, w) >= (season, week):
                continue
            wk = ss.filter(pl.col("week") == w)
            if wk.height > 0 and wk["home_score"].null_count() == 0:
                out.append((s, w))
    return out


def _window_week_pairs(groups, opp: pl.DataFrame, season: int, week: int) -> list:
    """(position, C stats, actual stats) for everyone who played (season, week),
    with C built from games strictly before that week."""
    pairs = []
    played_ids = []
    projections, actuals = {}, {}
    for df, id_col, stat_columns, describe_row in groups:
        rows = df.filter((pl.col("season") == season) & (pl.col("week") == week))
        week_actuals = {str(r[id_col]): {c: r.get(c) for c in stat_columns} for r in rows.iter_rows(named=True)}
        if not week_actuals:
            continue
        relevant = df.filter(pl.col(id_col).is_in(list(week_actuals.keys())))
        projections.update(build_projections(relevant, season, week, id_col, stat_columns, describe_row, n_games=8))
        actuals.update(week_actuals)
        played_ids.extend(week_actuals.keys())
    opp_rel = opp.filter(pl.col("player_id").is_in(played_ids))
    opp_map = build_projections(opp_rel, season, week, "player_id", model_core.OPP_STAT_COLUMNS,
                                describe_player, n_games=8) if opp_rel.height else {}
    for eid, proj in projections.items():
        if eid not in actuals:
            continue
        o = opp_map.get(eid)
        c = model_core.baseline_c_stats(proj["projected_stats"], proj["position"], o["projected_stats"] if o else None)
        pairs.append((proj["position"], c, actuals[eid]))
    return pairs


def build_v3(groups, opp: pl.DataFrame, schedules: pl.DataFrame, season: int, week: int,
             projections: dict) -> tuple[dict, dict]:
    """v3 D stat lines for every entry in `projections` + model_meta. Mirrors
    the validated backtest: the shrink fit uses the last ROLLING_WEEKS
    completed weeks; each of those weeks' fallback ratios uses ITS OWN as-of
    D (fit on the 8 weeks before it), so 2x ROLLING_WEEKS weeks are built."""
    rw = model_core.ROLLING_WEEKS
    weeks = completed_weeks_before(schedules, season, week)[-2 * rw:]
    pairs_by_week = {wk: _window_week_pairs(groups, opp, *wk) for wk in weeks}
    recent = weeks[-rw:]

    entries = []
    for i, wk in enumerate(weeks):
        if wk not in recent:
            continue
        params_then = model_core.fit_window([p for prior in weeks[max(0, i - rw):i] for p in pairs_by_week[prior]])
        for pos, c, actual in pairs_by_week[wk]:
            entries.append((pos, model_core.compute_points(model_core.apply_d(c, pos, params_then)),
                            model_core.compute_points(actual)))
    params = model_core.fit_window([p for wk in recent for p in pairs_by_week[wk]])

    opp_cur = opp.filter(pl.col("player_id").is_in(list(projections.keys())))
    opp_map = build_projections(opp_cur, season, week, "player_id", model_core.OPP_STAT_COLUMNS,
                                describe_player, n_games=8) if opp_cur.height else {}
    d_stats = {}
    for eid, entry in projections.items():
        o = opp_map.get(eid)
        c = model_core.baseline_c_stats(entry["projected_stats"], entry["position"], o["projected_stats"] if o else None)
        d_stats[eid] = model_core.apply_d(c, entry["position"], params)

    pools = model_core.build_ratio_pools(entries)
    meta = {
        "model_version": model_core.MODEL_VERSION,
        "season": season,
        "week": week,
        "window_weeks": [list(wk) for wk in recent],
        "params": {pos: {k: p[k] for k in ("beta", "m_act", "m_c", "n_pairs")} for pos, p in params.items()},
        "ratio_bin_edges": model_core.RATIO_BIN_EDGES,
        "ratio_min_pool": model_core.RATIO_MIN_POOL,
        "pooling_k": model_core.POOLING_K,
        "ratio_pools": {},
    }
    for (pos, b), ratios in sorted(pools.items()):
        meta["ratio_pools"].setdefault(pos, {})[str(b)] = [round(r, 4) for r in ratios]
    return d_stats, meta


def main() -> None:
    season = current_season()
    schedules = nfl.load_schedules(seasons=season)
    season, week = get_current_season_and_week(schedules)

    lookback_seasons = list(range(season - LOOKBACK_SEASONS, season + 1))

    stats = nfl.load_player_stats(seasons=lookback_seasons, summary_level="week")
    is_k = (pl.col("position") == "K").fill_null(False)
    non_kickers = stats.filter(~is_k)
    kickers = add_k_stat_columns(stats.filter(is_k))

    projections = build_projections(non_kickers, season, week, "player_id", STAT_COLUMNS, describe_player)
    history = build_history(non_kickers, season, week, "player_id", STAT_COLUMNS)
    projections.update(build_projections(kickers, season, week, "player_id", K_STAT_COLUMNS, describe_player))
    history.update(build_history(kickers, season, week, "player_id", K_STAT_COLUMNS))

    lookback_schedules = nfl.load_schedules(seasons=lookback_seasons)
    team_stats = add_def_stat_columns(
        nfl.load_team_stats(seasons=lookback_seasons, summary_level="week"),
        lookback_schedules,
    )
    def_projections = build_projections(
        team_stats, season, week, "def_id", DEF_STAT_COLUMNS, describe_defense
    )
    def_history = build_history(team_stats, season, week, "def_id", DEF_STAT_COLUMNS)

    projections.update(def_projections)
    history.update(def_history)

    out_dir = DATA_DIR / f"week_{week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # v3 model (PRODUCTION_MODEL_SPEC.md). projected_stats becomes the v3 line;
    # the old last-8 line is kept as projected_stats_v2 for the in-app toggle
    # and rollback. If anything in the v3 build fails (e.g. the nflverse
    # opportunity feed is unavailable), today's v2 output ships unchanged and
    # no model_meta.json is written -- the frontend then runs v2 regardless
    # of the toggle.
    meta = None
    try:
        skill = stats.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        groups = [
            (skill, "player_id", STAT_COLUMNS, describe_player),
            (kickers, "player_id", K_STAT_COLUMNS, describe_player),
            (team_stats, "def_id", DEF_STAT_COLUMNS, describe_defense),
        ]
        opp = model_core.load_opportunity_frame(lookback_seasons)
        d_stats, meta = build_v3(groups, opp, lookback_schedules, season, week, projections)
        for eid, entry in projections.items():
            entry["projected_stats_v2"] = entry["projected_stats"]
            entry["projected_stats"] = {k: round(v, 4) for k, v in d_stats[eid].items()}
            entry["model_version"] = model_core.MODEL_VERSION
    except Exception as exc:  # noqa: BLE001 -- v3 is optional; never break the daily job over it
        print(f"WARNING: v3 model build failed ({exc!r}); shipping v2 projections only")
        meta = None

    injuries, active, reserve = load_availability(season, week)
    apply_availability(projections, schedules, season, week, injuries, active, reserve)

    (out_dir / "projections.json").write_text(json.dumps(projections, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    meta_path = out_dir / "model_meta.json"
    if meta is not None:
        meta_path.write_text(json.dumps(meta, indent=2))
    elif meta_path.exists():
        meta_path.unlink()  # a stale meta from an earlier run must not pair with v2-only projections

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
                "modelMeta": meta is not None,
            },
            indent=2,
        )
    )

    n_k = sum(1 for p in projections.values() if p["position"] == "K")
    # Every week gets a live.json from the start (empty until games begin),
    # so the frontend can poll it without ever hitting a 404 for the current
    # week. sync_live_scores.py fills it in; never clobber real data here.
    live_path = out_dir / "live.json"
    if not live_path.exists():
        live_path.write_text(json.dumps(
            {"updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "players": {}},
            indent=2,
        ))

    print(
        f"Wrote projections + history for season {season}, week {week}: "
        f"{len(projections) - len(def_projections) - n_k} players + {n_k} kickers + "
        f"{len(def_projections)} team defenses -> {out_dir}/"
    )


if __name__ == "__main__":
    main()
