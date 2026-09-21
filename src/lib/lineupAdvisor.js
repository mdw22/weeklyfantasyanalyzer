import { computeFantasyPoints } from "./scoring.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "./rosterSlots.js";

const BENCH_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("BN"));

/** Why a player may not play, or null if there's no concern.
 *   severity "out"  -- OUT / IR: won't play            (red)
 *   severity "risk" -- QUESTIONABLE / DOUBTFUL: might   (amber)
 *   severity "bye"  -- team has no game: won't play, but routine (no color)
 * A bye wins over an injury: nothing else matters if there's no game. */
export function availabilityRisk(entry, injuryOverride) {
  if (!entry) return null;
  if (entry.on_bye) return { reason: "BYE", severity: "bye" };
  // `injuryOverride` is ESPN's live status from the live-scores loop; it
  // beats the once-a-day nflverse `injury_status` (which is ~a day stale).
  // An override of "ACTIVE" clears the risk.
  const status = injuryOverride ?? entry.injury_status;
  if (status === "OUT" || status === "IR") return { reason: status, severity: "out" };
  if (status === "QUESTIONABLE" || status === "DOUBTFUL") return { reason: status, severity: "risk" };
  return null;
}

/** True when a player is effectively certain NOT to play: Out, IR, or on a
 * bye. Questionable/Doubtful are deliberately excluded -- those are
 * genuinely uncertain, so their full average projection stays the fair
 * expected value and the pill communicates the risk. */
export function isExpectedOut(entry, injuryOverride) {
  const risk = availabilityRisk(entry, injuryOverride);
  return !!risk && risk.severity !== "risk";
}

/** "not_started" | "in_progress" | "final" for one player's game. A rostered
 * player's own live entry wins; otherwise the team's game state (live.json's
 * `teams`), which also covers free agents. Defaults to not_started. */
export function gameStatusOf(playerId, entry, live) {
  const own = live?.players?.[playerId];
  if (own) return own.status;
  return live?.teams?.[entry?.team]?.status ?? "not_started";
}

/** The availability concern that's still ACTIONABLE, or null. Once a player's
 * game has started or finished, the pre-game injury tag is stale -- the real
 * result (Final / live clock) replaces it. Uses live ESPN injury overrides. */
export function currentRisk(playerId, entry, live) {
  if (gameStatusOf(playerId, entry, live) !== "not_started") return null;
  return availabilityRisk(entry, live?.injuries?.[playerId]);
}

/** Starters (never bench/IR -- those are already benched by choice) whose
 * player might not play. `roster` is slotId -> playerId. */
export function getAtRiskSlots(roster, projections, live = null) {
  const flagged = [];
  for (const slotId of STARTER_SLOT_IDS) {
    const playerId = roster[slotId];
    const risk = playerId ? currentRisk(playerId, projections[playerId], live) : null;
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
 * Candidates are ranked by min(our projection, ESPN's) when ESPN's exists (see
 * the sort below). Anyone whose game has already started or finished is excluded, as is
 * anyone OUT / IR / on a bye (using ESPN's live injury status when known) -- recommending another player
 * who can't play defeats the purpose. Questionable/doubtful candidates stay
 * (carrying `risk`, so the UI can tag them). */
export function getReplacementCandidates(slot, roster, projections, ownership, scoringValues, limit = 5, live = null) {
  const eligible = new Set(slot.eligible);
  const atRiskId = roster[slot.id];
  const seen = new Set([atRiskId]);
  const pool = [];

  function consider(playerId, source) {
    if (!playerId || seen.has(playerId)) return;
    const entry = projections[playerId];
    if (!entry || !eligible.has(entry.position)) return;
    // A game that's started or finished can't be used: the player is locked
    // (or done), so recommending him on his pre-game projection is wrong.
    if (gameStatusOf(playerId, entry, live) !== "not_started") return;
    const risk = availabilityRisk(entry, live?.injuries?.[playerId]);
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
      // ESPN's own projection, present only for Questionable/Doubtful players
      // whose game hasn't started (see sync_live_scores.py). Shown next to ours
      // so a large disagreement is visible rather than hidden.
      espn: live?.espnProjections?.[playerId] ?? null,
    });
  }

  for (const benchSlot of BENCH_SLOTS) consider(roster[benchSlot.id], "bench");

  const freeAgentsAvailable = !!ownership;
  if (freeAgentsAvailable) {
    for (const [playerId, entry] of Object.entries(projections)) {
      if (entry.active && !ownership[playerId]?.owned) consider(playerId, "free_agent");
    }
  }

  // Rank by the LOWER of our projection and ESPN's when ESPN has one. `points`
  // stays what's displayed; only the ORDER changes. This can only ever demote a
  // Questionable/Doubtful candidate (ESPN's number only exists for those), never
  // affect a healthy one, and an ESPN number above ours never lifts anyone.
  // Caveat, measured on the real pool: an empty ESPN projection among in-doubt
  // players occurs at about the base rate (2 of 20, vs 25 of 307 for healthy
  // players), so a 0.0 here is not proven to be injury-specific -- the rule
  // accepts that, because demoting a possibly-fine candidate is the cheap error.
  const rankKey = (c) => Math.min(c.points, c.espn ?? c.points);
  pool.sort((a, b) => rankKey(b) - rankKey(a) || b.points - a.points || a.name.localeCompare(b.name));
  return { candidates: pool.slice(0, limit), freeAgentsAvailable };
}

/** How far apart our projection and ESPN's must be (in points) before the
 * ESPN chip is emphasized as a real disagreement. */
export const ESPN_GAP_POINTS = 5;

/** Signed, one-decimal difference: "+8.1", "-2.3", "+0.0". */
export function formatDelta(candidatePoints, baselinePoints) {
  const delta = candidatePoints - baselinePoints;
  const rounded = Math.round(delta * 10) / 10;
  return `${rounded < 0 ? "-" : "+"}${Math.abs(rounded).toFixed(1)}`;
}
