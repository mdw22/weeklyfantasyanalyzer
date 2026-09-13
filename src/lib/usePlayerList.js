import { useMemo } from "react";
import { computeFantasyPoints } from "./scoring.js";

/** Turns the projections.json map into a flat, sorted, filtered array of
 * players with their fantasy points already computed under the current
 * scoring settings -- shared by WeeklyProjectionsTable and TeamBuilder's
 * player picker so both filter/sort the same way. */
export function usePlayerList({
  projections,
  scoringValues,
  search = "",
  positions = null,
  sortBy = "points",
  sortDir = "desc",
}) {
  return useMemo(() => {
    if (!projections) return [];

    let list = Object.entries(projections).map(([id, p]) => ({
      id,
      name: p.player_name,
      position: p.position,
      team: p.team,
      points: computeFantasyPoints(p.projected_stats, scoringValues),
      stats: p.projected_stats,
    }));

    if (positions && positions.length > 0) {
      list = list.filter((p) => positions.includes(p.position));
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((p) => p.name?.toLowerCase().includes(q));
    }

    const dir = sortDir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      const av = sortBy === "name" ? a.name ?? "" : a[sortBy] ?? 0;
      const bv = sortBy === "name" ? b.name ?? "" : b[sortBy] ?? 0;
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });

    return list;
  }, [projections, scoringValues, search, positions, sortBy, sortDir]);
}
