"""
espn_comparison_log.py

Phase 1, Step 3 of PHASE1_EVALUATION_SPEC.md: a GO-FORWARD log of our
projection vs. ESPN's own, for every player where both exist. NOT a
backtest -- ESPN's public endpoint only exposes its CURRENT projection, not
historical ones, so there is no way to retroactively get what ESPN would
have projected for a past week. This starts accumulating from whenever it's
first run, and only has real signal after several weeks of entries exist.

OFFLINE ONLY, run manually -- not wired into any GitHub Actions workflow
(explicitly out of scope for Phase 1, per the spec: "whether it ever
becomes a scheduled job is a later call"). Reuses the exact fetch/parse
logic already built for the live-scores ESPN chip
(`sync_live_scores.py::fetch_player_pool`/`espn_week_projection`) and the
ESPN->gsis crosswalk already built for ESPN sync
(`sync_espn.py::build_espn_to_gsis_map`) -- no new ESPN integration code.

Durable storage: scripts/analysis_data/espn_comparison_log.jsonl, ONE JSON
object per line, upserted by (season, week, player_id) so running this
multiple times in the same week updates that week's entries rather than
duplicating them. Tracked in git (unlike the gitignored designer-working-
docs) because the whole point is accumulating real signal across weeks --
losing it between sessions would defeat the purpose.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/espn_comparison_log.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import FULL_PPR_VALUES, compute_points  # noqa: E402
from sync_espn import build_espn_to_gsis_map, resolve_def_team_id  # noqa: E402
from sync_live_scores import espn_week_projection, fetch_player_pool  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "public" / "data"
LOG_PATH = Path(__file__).parent / "analysis_data" / "espn_comparison_log.jsonl"


def resolve_pool_entry_id(entry: dict, espn_to_gsis: dict) -> str | None:
    """Same DEF-vs-individual-player distinction as sync_espn.py's
    resolve_entry_id, but the public pool's player dict isn't nested under
    `playerPoolEntry` the way a league roster entry is -- adapted for that
    shape rather than reused directly."""
    player = entry.get("player", {})
    espn_id = str(entry.get("id"))
    is_def = player.get("defaultPositionId") == 16 or espn_id.startswith("-")
    return resolve_def_team_id(espn_id) if is_def else espn_to_gsis.get(espn_id)


def load_existing_log() -> dict:
    """{(season, week, player_id): row}. Missing/empty file -> empty log,
    not an error -- this is the first run in that case."""
    if not LOG_PATH.exists():
        return {}
    existing = {}
    for line in LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        existing[(row["season"], row["week"], row["player_id"])] = row
    return existing


def write_log(rows_by_key: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows_by_key.values(), key=lambda r: (r["season"], r["week"], r["player_id"]))
    LOG_PATH.write_text("\n".join(json.dumps(r) for r in ordered) + "\n")


def main() -> None:
    latest = json.loads((DATA_DIR / "latest.json").read_text())
    season, week = latest["season"], latest["week"]
    week_str = str(week).zfill(2)
    projections = json.loads((DATA_DIR / f"week_{week_str}" / "projections.json").read_text())

    print(f"Fetching ESPN's public player pool for season {season}, week {week}...")
    pool = fetch_player_pool(season)
    espn_to_gsis = build_espn_to_gsis_map()

    existing = load_existing_log()
    logged_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    new_rows, skipped_no_match, skipped_no_espn_proj, skipped_no_our_proj = 0, 0, 0, 0

    for entry in pool:
        our_id = resolve_pool_entry_id(entry, espn_to_gsis)
        if our_id is None:
            skipped_no_match += 1
            continue
        espn_points = espn_week_projection(entry, season, week)
        if espn_points is None:
            skipped_no_espn_proj += 1
            continue
        proj = projections.get(our_id)
        if proj is None:
            skipped_no_our_proj += 1
            continue

        our_points = compute_points(proj["projected_stats"], FULL_PPR_VALUES)
        key = (season, week, our_id)
        existing[key] = {
            "season": season,
            "week": week,
            "player_id": our_id,
            "player_name": proj["player_name"],
            "position": proj["position"],
            "our_projection": round(our_points, 2),
            "espn_projection": round(espn_points, 2),
            "diff": round(our_points - espn_points, 2),
            "logged_at": logged_at,
        }
        new_rows += 1

    write_log(existing)

    this_week_rows = [r for (s, w, _), r in existing.items() if s == season and w == week]
    diffs = [r["diff"] for r in this_week_rows]
    n = len(diffs)
    mean_abs_diff = sum(abs(d) for d in diffs) / n if n else 0.0
    biggest = sorted(this_week_rows, key=lambda r: -abs(r["diff"]))[:5]

    all_weeks_logged = sorted({(s, w) for s, w, _ in existing})

    print(f"\nThis run: {new_rows} player comparisons logged/updated for season {season} week {week}")
    print(f"  skipped (no ESPN<->our-id match): {skipped_no_match}")
    print(f"  skipped (ESPN has no projection this week for them): {skipped_no_espn_proj}")
    print(f"  skipped (not in our projections.json): {skipped_no_our_proj}")
    print(f"\nWeek {week} summary: n={n}, mean |our - ESPN| = {mean_abs_diff:.2f} points")
    print("Biggest disagreements this week:")
    for r in biggest:
        print(f"  {r['player_name']:<24} {r['position']:<4} ours={r['our_projection']:6.2f}  "
              f"ESPN={r['espn_projection']:6.2f}  diff={r['diff']:+.2f}")

    print(f"\nLog now covers {len(all_weeks_logged)} week(s) total: {all_weeks_logged}")
    print(f"Total accumulated rows: {len(existing)}")
    print(f"Written to {LOG_PATH}")


if __name__ == "__main__":
    main()
