import { computeFantasyPoints } from "./scoring.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "./rosterSlots.js";

const BENCH_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("BN"));

/** Why a player may not play, or null if there's no concern.
 *   severity "out"  -- OUT / IR: won't play            (red)
 *   severity "risk" -- QUESTIONABLE / DOUBTFUL: might   (amber)
 *   severity "bye"  -- team has no game: won't play, but routine (no color)
 * A bye wins over an injury: nothing else matters if there's no game. */
export function availabilityRisk(entry) {
  if (!entry) return null;
  if (entry.on_bye) return { reason: "BYE", severity: "bye" };
  const status = entry.injury_status;
  if (status === "OUT" || status === "IR") return { reason: status, severity: "out" };
  if (status === "QUESTIONABLE" || status === "DOUBTFUL") return { reason: status, severity: "risk" };
  return null;
}

/** Starters (never bench/IR -- those are already benched by choice) whose
 * player might not play. `roster` is slotId -> playerId. */
export function getAtRiskSlots(roster, projections) {
  const flagged = [];
  for (const slotId of STARTER_SLOT_IDS) {
    const playerId = roster[slotId];
    const risk = availabilityRisk(projections[playerId]);
    if (playerId && risk) {
      const slot = ROSTER_SLOTS.find((s) => s.id === slotId);
      flagged.push({ slotId, slot, playerId, ...risk });
    }
  }
  return flagged;
}

/** Best replacements for ONE at-risk slot (no whole-lineup re-optimizing):
 * my bench plus free agents, ranked together by projected points under the
 * user's scoring settings, tagged by source.
 *
 * Free agents = not on any team in the league (`ownership`, from the ESPN
 * sync) AND on a real active NFL roster (`active`, from the pipeline --
 * projections.json also holds retired/cut/IR players with stats from the
 * last 3 seasons, who'd otherwise rank as the "best free agents"). If
 * `ownership` is unknown (no ESPN sync), free agents are skipped entirely
 * rather than guessed at, and `freeAgentsAvailable` says so.
 *
 * Anyone OUT / IR / on a bye is excluded -- recommending another player
 * who can't play defeats the purpose. Questionable/doubtful candidates stay
 * (carrying `risk`, so the UI can tag them). */
export function getReplacementCandidates(slot, roster, projections, ownership, scoringValues, limit = 5) {
  const eligible = new Set(slot.eligible);
  const atRiskId = roster[slot.id];
  const seen = new Set([atRiskId]);
  const pool = [];

  function consider(playerId, source) {
    if (!playerId || seen.has(playerId)) return;
    const entry = projections[playerId];
    if (!entry || !eligible.has(entry.position)) return;
    const risk = availabilityRisk(entry);
    if (risk && risk.severity !== "risk") return;
    seen.add(playerId);
    pool.push({
      playerId,
      name: entry.player_name,
      position: entry.position,
      team: entry.team,
      points: computeFantasyPoints(entry.projected_stats, scoringValues),
      source,
      risk,
    });
  }

  for (const benchSlot of BENCH_SLOTS) consider(roster[benchSlot.id], "bench");

  const freeAgentsAvailable = !!ownership;
  if (freeAgentsAvailable) {
    for (const [playerId, entry] of Object.entries(projections)) {
      if (entry.active && !ownership[playerId]?.owned) consider(playerId, "free_agent");
    }
  }

  pool.sort((a, b) => b.points - a.points || a.name.localeCompare(b.name));
  return { candidates: pool.slice(0, limit), freeAgentsAvailable };
}
