import { computeFantasyPoints } from "./scoring.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "./rosterSlots.js";
import { currentRisk } from "./lineupAdvisor.js";

// Season-view logic for the Roster Moves tab. Everything here ranks players by
// their PLAIN projection (computeFantasyPoints of projected_stats) -- the same
// number Team Builder shows pre-kickoff. Deliberately NOT effectivePoints and
// not live game state: a player on this week's bye, or mid-game, is exactly as
// valuable a roster asset next week. (The status pills in `rosterFlags` are the
// one informational exception.)

export const POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"];

// How many of each position a team realistically STARTS (FLEX handled via the
// surplus test in getTradeTargets). Matches ROSTER_SLOTS: 1 QB, 2 RB, 2 WR, 1 TE,
// 1 K, 1 DEF plus one RB/WR/TE flex.
export const STARTERS_PER_POSITION = { QB: 1, RB: 2, WR: 2, TE: 1, K: 1, DEF: 1 };

// ---- tunable thresholds (chosen from real numbers; see DESIGNER_REPORT.md) ----
export const PICKUP_MARGIN = 3; // a free agent must beat the bench player by this many points
export const MAX_PICKUP_PAIRS = 5;
export const FREE_AGENTS_PER_POSITION = 5;
export const NEED_RANK_FROM_BOTTOM = 4; // a starter slot ranked in the league's bottom N = a need
export const STRENGTH_TOP_FRACTION = 0.25; // ...or in the top quarter, with someone beyond my starters

const BENCH_SLOT_IDS = ROSTER_SLOTS.filter((s) => s.id.startsWith("BN")).map((s) => s.id);
const slotLabel = (slotId) => ROSTER_SLOTS.find((s) => s.id === slotId)?.label ?? slotId;

const plainPoints = (entry, values) => computeFantasyPoints(entry.projected_stats, values);

function describe(playerId, entry, values, extra = {}) {
  return {
    playerId,
    name: entry.player_name,
    position: entry.position,
    team: entry.team,
    points: plainPoints(entry, values),
    ...extra,
  };
}

/** Everyone on my roster (all slots, IR included) as [slotId, playerId, entry]. */
function rosterEntries(roster, projections) {
  return Object.entries(roster)
    .filter(([, playerId]) => playerId && projections[playerId])
    .map(([slotId, playerId]) => [slotId, playerId, projections[playerId]]);
}

/** Excludes the IR slot only -- bench players stay in, they're what "surplus"
 * is supposed to catch. Used everywhere "my roster" feeds a startable-players
 * ranking (positionSummary's best option, getTradeTargets' Nth-best/surplus);
 * NOT used by rosterFlags, which exists specifically to surface IR status. */
const startable = ([slotId]) => slotId !== "IR";

// ------------------------------------------------------------ Section 1 --------

/** Per real position (not slot label): how many I roster and my best option. */
export function positionSummary(roster, projections, values) {
  return POSITIONS.map((position) => {
    const mine = rosterEntries(roster, projections)
      .filter(startable)
      .filter(([, , entry]) => entry.position === position)
      .map(([, playerId, entry]) => describe(playerId, entry, values))
      .sort((a, b) => b.points - a.points);
    return { position, count: mine.length, best: mine[0] ?? null };
  });
}

const SEVERITY_ORDER = { out: 0, risk: 1, bye: 2 };

/** Every rostered player (starters, bench, IR) currently Out/IR, Questionable/
 * Doubtful, or on a bye -- worst first. Informational only. Uses the same
 * currentRisk rule as the rest of the app, so these match Team Builder. */
export function rosterFlags(roster, projections, live, values) {
  return rosterEntries(roster, projections)
    .map(([slotId, playerId, entry]) => ({ slotId, playerId, entry, risk: currentRisk(playerId, entry, live) }))
    .filter((r) => r.risk)
    .map((r) => ({
      slotId: r.slotId,
      slotLabel: slotLabel(r.slotId),
      ...describe(r.playerId, r.entry, values),
      risk: r.risk,
    }))
    .sort((a, b) => SEVERITY_ORDER[a.risk.severity] - SEVERITY_ORDER[b.risk.severity] || b.points - a.points);
}

// ------------------------------------------------------------ Section 2 --------

/** Free agents = on a real active NFL roster AND not rostered by anyone in the
 * league. Returns null when ownership is unknown (never guess). */
function freeAgents(projections, ownership) {
  if (!ownership) return null;
  return Object.entries(projections).filter(([id, entry]) => entry.active && !ownership[id]?.owned);
}

/** Bench-for-free-agent swaps worth making, by plain projection. Same position
 * only; my BENCH slots only (starters and the IR slot are never drop candidates
 * -- there's no "designated for return" data to judge IR). Greedy unique
 * assignment: pairs sorted by gain, each bench player and each free agent used
 * at most once, so one stud free agent isn't suggested for several spots. */
export function getPickupsAndDrops(roster, projections, ownership, values, options = {}) {
  const margin = options.margin ?? PICKUP_MARGIN;
  const maxPairs = options.maxPairs ?? MAX_PICKUP_PAIRS;
  const pool = freeAgents(projections, ownership);
  if (!pool) return { pairs: [], freeAgentsAvailable: false, benchCount: 0 };

  const bench = BENCH_SLOT_IDS.map((slotId) => [slotId, roster[slotId]])
    .filter(([, playerId]) => playerId && projections[playerId])
    .map(([slotId, playerId]) => describe(playerId, projections[playerId], values, { slotId }));

  const topFreeAgents = {};
  for (const position of POSITIONS) {
    topFreeAgents[position] = pool
      .filter(([, entry]) => entry.position === position)
      .map(([id, entry]) => describe(id, entry, values))
      .sort((a, b) => b.points - a.points)
      .slice(0, options.faPerPosition ?? FREE_AGENTS_PER_POSITION);
  }

  const candidates = [];
  for (const drop of bench) {
    for (const add of topFreeAgents[drop.position] ?? []) {
      const gain = add.points - drop.points;
      if (gain >= margin) candidates.push({ position: drop.position, drop, add, gain });
    }
  }
  candidates.sort((a, b) => b.gain - a.gain);

  const usedDrops = new Set();
  const usedAdds = new Set();
  const pairs = [];
  for (const pair of candidates) {
    if (usedDrops.has(pair.drop.playerId) || usedAdds.has(pair.add.playerId)) continue;
    usedDrops.add(pair.drop.playerId);
    usedAdds.add(pair.add.playerId);
    pairs.push(pair);
    if (pairs.length >= maxPairs) break;
  }
  return { pairs, freeAgentsAvailable: true, benchCount: bench.length };
}

// ------------------------------------------------------------ Section 3 --------

/** Where my roster is deep (tradeable surplus) vs thin (a real need), per
 * position. No specific trade proposals -- just the shape of my roster against
 * the league's.
 *
 * Baseline: every currently ROSTERED, ACTIVE player leaguewide (ownership +
 * projections.active), grouped by team. Only rostered players count: a
 * bench-quality free agent shouldn't define "replacement level" if nobody would
 * start him. Non-active players (IR/retired/cut) are excluded for everyone --
 * their history-based projections say nothing about what they'll contribute.
 *
 * For a position that starts S players, each team's "S-th best" is its last
 * starter there. Comparing MY S-th best against every team's tells me how my
 * starting group stacks up:
 *   need     -- my S-th best is among the league's bottom NEED_RANK_FROM_BOTTOM
 *   strength -- EITHER I have a player beyond my S starters who'd beat the median
 *               team's S-th best (a starter for half the league): surplus;
 *               OR my S-th best ranks in the league's top STRENGTH_TOP_FRACTION
 *               and I roster someone beyond my starters (strong AND deep)
 *   adequate -- otherwise
 * For a "need", also reports the best free agent at the position when one beats
 * my S-th best by PICKUP_MARGIN: the fix is often a free-agent pickup, not a
 * trade, and the pickups section (bench swaps only) can't propose starter
 * upgrades like a better K or DEF. */
export function getTradeTargets(roster, projections, ownership, myTeamId, values, thresholds = {}) {
  if (!ownership) return { positions: [], leagueKnown: false, teamCount: 0 };
  const needFromBottom = thresholds.needFromBottom ?? NEED_RANK_FROM_BOTTOM;
  const topFraction = thresholds.strengthTopFraction ?? STRENGTH_TOP_FRACTION;
  const faPool = freeAgents(projections, ownership) ?? [];

  const byTeam = {};
  for (const [id, info] of Object.entries(ownership)) {
    const entry = projections[id];
    if (!entry || !entry.active || !info.owned) continue;
    (byTeam[info.espnTeamId] ??= []).push(describe(id, entry, values));
  }
  // Make sure my own team is present even if ownership predates a manual edit
  // and treat my roster (what the app shows) as the source for MY players.
  const mine = rosterEntries(roster, projections)
    .filter(startable)
    .filter(([, , entry]) => entry.active)
    .map(([slotId, id, entry]) => describe(id, entry, values, { slotId }));
  byTeam[myTeamId] = mine;

  const teamIds = Object.keys(byTeam);
  const nth = (players, position, n) => {
    const sorted = players.filter((p) => p.position === position).sort((a, b) => b.points - a.points);
    return sorted[n - 1]?.points ?? 0; // fewer than n players = 0 at that slot
  };

  const positions = POSITIONS.map((position) => {
    const S = STARTERS_PER_POSITION[position];
    const myList = mine.filter((p) => p.position === position).sort((a, b) => b.points - a.points);
    const leagueNth = teamIds.map((t) => nth(byTeam[t], position, S)).sort((a, b) => b - a);
    const myNth = myList[S - 1]?.points ?? 0;
    // 1 = best in the league; ties share the better rank.
    const rank = 1 + leagueNth.filter((v) => v > myNth).length;
    const teamCount = leagueNth.length;
    const median = leagueNth[Math.floor(teamCount / 2)] ?? 0;
    const surplus = myList[S] ?? null; // my first player beyond my S starters
    let verdict = "adequate";
    if (rank > teamCount - needFromBottom || myList.length < S) verdict = "need";
    else if (surplus && (surplus.points >= median || rank <= Math.ceil(teamCount * topFraction))) verdict = "strength";

    let bestFreeAgent = null;
    if (verdict === "need") {
      const top = faPool
        .filter(([, entry]) => entry.position === position)
        .map(([id, entry]) => describe(id, entry, values))
        .sort((a, b) => b.points - a.points)[0];
      if (top && top.points - myNth >= (thresholds.pickupMargin ?? PICKUP_MARGIN)) {
        bestFreeAgent = { ...top, gain: top.points - myNth };
      }
    }
    return {
      position,
      starters: S,
      rostered: myList.length,
      verdict,
      rank,
      teamCount,
      nthName: myList[S - 1]?.name ?? null,
      nthPoints: myNth,
      median,
      surplus,
      bestFreeAgent,
      top: myList.slice(0, S + 1),
    };
  });
  return { positions, leagueKnown: true, teamCount: teamIds.length };
}
