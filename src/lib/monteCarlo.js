import { computeFantasyPoints } from "./scoring.js";

const DEFAULT_TRIALS = 10000;

/** Builds a function that returns one bootstrap-sampled stat line for a
 * player: a real past game picked at random (with replacement) from
 * history.json. Falls back to the player's own projected mean line
 * (sampled deterministically every trial) when there's no history yet --
 * rookies / new starters, per the data pipeline's own fallback note. */
function buildSampler(playerId, projections, history) {
  const pastGames = history[playerId];
  if (pastGames && pastGames.length > 0) {
    return () => pastGames[Math.floor(Math.random() * pastGames.length)];
  }
  const projected = projections[playerId]?.projected_stats;
  return () => projected;
}

function percentile(sorted, p) {
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function summarize(totals) {
  const sorted = [...totals].sort((a, b) => a - b);
  const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
  return {
    mean,
    p10: percentile(sorted, 0.1),
    p25: percentile(sorted, 0.25),
    p75: percentile(sorted, 0.75),
    p90: percentile(sorted, 0.9),
  };
}

/** Player outcomes are treated as independent for v2 (no shared game-script
 * correlation between teammates) -- an approved simplification, see
 * CLAUDE.md. */
export function simulateMatchup({
  myPlayerIds,
  opponentPlayerIds,
  projections,
  history,
  scoringValues,
  trials = DEFAULT_TRIALS,
}) {
  const mySamplers = myPlayerIds.map((id) => buildSampler(id, projections, history));
  const oppSamplers = opponentPlayerIds.map((id) => buildSampler(id, projections, history));

  const myTotals = new Array(trials);
  const oppTotals = new Array(trials);
  let myWins = 0;

  for (let t = 0; t < trials; t++) {
    let myTotal = 0;
    for (const sample of mySamplers) myTotal += computeFantasyPoints(sample(), scoringValues);
    let oppTotal = 0;
    for (const sample of oppSamplers) oppTotal += computeFantasyPoints(sample(), scoringValues);

    myTotals[t] = myTotal;
    oppTotals[t] = oppTotal;
    if (myTotal > oppTotal) myWins += 1;
  }

  return {
    winProbability: myWins / trials,
    my: summarize(myTotals),
    opponent: summarize(oppTotals),
  };
}
