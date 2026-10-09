import { SCORING_PRESETS, computeFantasyPoints } from "./scoring.js";

const DEFAULT_TRIALS = 10000;
// Fallback bins are defined in default (full-PPR) points, as in the pipeline.
const DEFAULT_VALUES = SCORING_PRESETS.full_ppr.values;

function populationVariance(xs) {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length;
}

/** Per-position mean population variance of history points, under the user's
 * scoring, over active players with >= 2 games (PRODUCTION_MODEL_SPEC.md 3d).
 * Computed once per (data, scoring) by the caller. */
export function computePositionVariance(projections, history, scoringValues) {
  const sums = {};
  for (const [id, entry] of Object.entries(projections)) {
    const games = history[id];
    if (!games || games.length < 2) continue;
    if (!entry.active && entry.position !== "DEF") continue;
    const v = populationVariance(games.map((g) => computeFantasyPoints(g, scoringValues)));
    const s = (sums[entry.position] ??= { total: 0, n: 0 });
    s.total += v;
    s.n += 1;
  }
  const out = {};
  for (const [pos, { total, n }] of Object.entries(sums)) out[pos] = total / n;
  return out;
}

function ratioBin(points, edges) {
  return edges.reduce((b, edge) => b + (points >= edge ? 1 : 0), 0);
}

/** Same bin; else merge with adjacent bins; else the whole position pool. */
function ratioPool(pools, minPool, bin) {
  if (!pools) return null;
  const own = pools[String(bin)];
  if (own && own.length >= minPool) return own;
  const merged = [bin - 1, bin, bin + 1].flatMap((b) => pools[String(b)] ?? []);
  if (merged.length >= minPool) return merged;
  const all = Object.values(pools).flat();
  return all.length >= minPool ? all : null;
}

/** v3 per-player outcome set (spec section 2.4), as an array to resample from.
 * Own-history players (s > 0, n >= 2): projection + pooled predictive SD x the
 * position's pooled standardized residual shape (model_meta.resid_shapes,
 * mean 0 / SD 1 -- real big games included, so a draw can exceed the player's
 * own best game); an older meta without shapes rescales his own history
 * instead. Otherwise his projection times same-position, same-bin
 * actual/projected ratios (mean-normalized, so the center stays exactly the
 * projection). */
export function v3Outcomes(entry, games, scoringValues, posVar, modelMeta) {
  const target = computeFantasyPoints(entry.projected_stats, scoringValues);
  const points = (games ?? []).map((g) => computeFantasyPoints(g, scoringValues));
  const n = points.length;
  const s2 = n ? populationVariance(points) : 0;
  const pv = posVar[entry.position];
  const w = n / (n + modelMeta.pooling_k);
  const variance = pv === undefined ? s2 : w * s2 + (1 - w) * pv;
  const sd = n >= 2 ? Math.sqrt((variance * (n + 1)) / (n - 1)) : Math.sqrt(variance);

  if (s2 > 0 && n >= 2) {
    const shape = modelMeta.resid_shapes?.[entry.position];
    if (shape) return shape.map((z) => target + sd * z);
    const mean = points.reduce((a, b) => a + b, 0) / n;
    const scale = sd / Math.sqrt(s2);
    return points.map((p) => target + (p - mean) * scale);
  }
  const defaultTarget = computeFantasyPoints(entry.projected_stats, DEFAULT_VALUES);
  const pool = ratioPool(
    modelMeta.ratio_pools[entry.position],
    modelMeta.ratio_min_pool,
    ratioBin(defaultTarget, modelMeta.ratio_bin_edges)
  );
  if (!pool) return [target];
  const meanRatio = pool.reduce((a, b) => a + b, 0) / pool.length;
  return meanRatio ? pool.map((r) => (target * r) / meanRatio) : [target];
}

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

function buildV3Sampler(playerId, projections, history, scoringValues, posVar, modelMeta) {
  const entry = projections[playerId];
  if (!entry) return () => 0; // same as v2: a roster id missing from this week's data contributes 0
  const outcomes = v3Outcomes(entry, history[playerId], scoringValues, posVar, modelMeta);
  return () => outcomes[Math.floor(Math.random() * outcomes.length)];
}

/** Wraps one player's full-game points draw with what has already happened
 * (live.json's entry for him; PRODUCTION_MODEL_SPEC live-aware addendum):
 *   final        -> fixed at his actual points
 *   in progress  -> actual so far + the rest of the game: f*t + sqrt(f)*(x - t)
 *                   for a v3 draw x around center t (the remaining mean scales
 *                   with f, the spread with sqrt(f)); f*x on the v2 path
 *   not started / no live entry -> the full draw, unchanged.
 * f = fraction of the game remaining; an older live.json without it counts
 * an in-progress game as half played. */
export function liveAwareDraw(draw, liveEntry, { scoringValues, center = null }) {
  if (!liveEntry || liveEntry.status === "not_started") return draw;
  const actual = computeFantasyPoints(liveEntry.stats, scoringValues);
  if (liveEntry.status === "final") return () => actual;
  const f = Math.min(Math.max(liveEntry.fraction_remaining ?? 0.5, 0), 1);
  if (center === null) return () => actual + f * draw();
  const rootF = Math.sqrt(f);
  return () => actual + f * center + rootF * (draw() - center);
}

/** Player outcomes are treated as independent (no shared game-script
 * correlation between teammates) -- measured as second-order, see
 * MODEL_ROADMAP.md. model "v2" is today's raw bootstrap; "v3" needs
 * modelMeta + posVar (PRODUCTION_MODEL_SPEC.md). v3 samplers return points
 * directly; v2 samplers return stat lines scored per trial, as before. */
export function simulateMatchup({
  myPlayerIds,
  opponentPlayerIds,
  projections,
  history,
  scoringValues,
  model = "v2",
  modelMeta = null,
  posVar = null,
  live = null,
  trials = DEFAULT_TRIALS,
}) {
  const useV3 = model === "v3" && modelMeta && posVar;
  const score = useV3 ? (x) => x : (line) => computeFantasyPoints(line, scoringValues);
  const baseSampler = useV3
    ? (id) => buildV3Sampler(id, projections, history, scoringValues, posVar, modelMeta)
    : (id) => buildSampler(id, projections, history);
  // Every sampler returns points; live results (if any) fix or shorten the draw.
  const sampler = (id) => {
    const base = baseSampler(id);
    const draw = () => score(base());
    const entry = projections[id];
    const center = useV3 && entry ? computeFantasyPoints(entry.projected_stats, scoringValues) : null;
    return liveAwareDraw(draw, live?.players?.[id], { scoringValues, center });
  };
  const mySamplers = myPlayerIds.map(sampler);
  const oppSamplers = opponentPlayerIds.map(sampler);

  const myTotals = new Array(trials);
  const oppTotals = new Array(trials);
  let myWins = 0;

  for (let t = 0; t < trials; t++) {
    let myTotal = 0;
    for (const sample of mySamplers) myTotal += sample();
    let oppTotal = 0;
    for (const sample of oppSamplers) oppTotal += sample();

    myTotals[t] = myTotal;
    oppTotals[t] = oppTotal;
    if (myTotal > oppTotal) myWins += 1;
    else if (myTotal === oppTotal) myWins += 0.5;
  }

  return {
    winProbability: myWins / trials,
    my: summarize(myTotals),
    opponent: summarize(oppTotals),
  };
}
