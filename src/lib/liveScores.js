import { useEffect, useState } from "react";
import { computeFantasyPoints } from "./scoring.js";
import { REPO } from "./config.js";

/** One player's points for display: their real stat line once their game
 * has started (in progress or final), otherwise the projection. Live stats
 * go through the same computeFantasyPoints() as everything else, so custom
 * scoring settings apply to them too. */
export function resolvePlayerPoints(projection, liveEntry, scoringValues) {
  if (liveEntry && liveEntry.status !== "not_started") {
    return {
      points: computeFantasyPoints(liveEntry.stats, scoringValues),
      status: liveEntry.status,
      clock: liveEntry.clock ?? null,
    };
  }
  return {
    points: computeFantasyPoints(projection?.projected_stats, scoringValues),
    status: "not_started",
    clock: null,
  };
}

const EMPTY_LIVE = { players: {} };

const POLL_MS = 90_000;
const RATE_LIMITED_POLL_MS = 300_000;

/** Loads live.json for `week`, freshest source first.
 *
 * WHY THE GITHUB API, not the site's own /data/ files: GitHub Pages (and
 * raw.githubusercontent.com) serve everything through a CDN with a 10-minute
 * (5 for raw) cache that IGNORES query strings -- checked directly: brand-new
 * `?t=<unique>` URLs still came back `x-cache: HIT`. So a cache-buster can't
 * make those fresh; only a different URL or host can. The contents API reads
 * git directly (60s cache), and sends CORS headers, so a commit shows up in
 * about a minute instead of up to ~12.
 *
 * Limits: 60 requests/hour per IP unauthenticated (304s count), so polling
 * is every 90s and skipped while the tab is hidden. On rate-limit or any API
 * trouble, falls back to the (possibly older) copy served with the site, and
 * reports `rateLimited` so the caller can back off. A missing file (404)
 * means "no live data yet" -- not an error. */
export async function loadLiveScores(week, { fetchFn = fetch, baseUrl = import.meta.env?.BASE_URL ?? "/" } = {}) {
  const weekStr = String(week).padStart(2, "0");
  const apiUrl = `https://api.github.com/repos/${REPO}/contents/public/data/week_${weekStr}/live.json?ref=main`;
  let rateLimited = false;
  try {
    const res = await fetchFn(apiUrl, { headers: { Accept: "application/vnd.github.raw+json" } });
    if (res.ok) return { data: await res.json(), rateLimited: false };
    if (res.status === 404) return { data: EMPTY_LIVE, rateLimited: false };
    rateLimited = res.status === 403 || res.status === 429;
  } catch {
    // network trouble: fall through to the site copy
  }
  try {
    const res = await fetchFn(`${baseUrl}data/week_${weekStr}/live.json`);
    if (res.ok) return { data: await res.json(), rateLimited };
  } catch {
    // nothing left to try
  }
  return { data: null, rateLimited };
}

/** Polls live scores while the calling screen is mounted. Keeps the last
 * good data when a poll fails, so a hiccup never blanks the scores. */
export function useLiveScores(weekData) {
  const [live, setLive] = useState(EMPTY_LIVE);
  const ready = weekData.status === "ready";
  const week = ready ? weekData.week : null;

  useEffect(() => {
    if (!ready) return undefined;
    let cancelled = false;
    let timer;

    async function tick() {
      let delay = POLL_MS;
      if (!document.hidden) {
        const { data, rateLimited } = await loadLiveScores(week);
        if (data && !cancelled) setLive(data);
        if (rateLimited) delay = RATE_LIMITED_POLL_MS;
      }
      if (!cancelled) timer = setTimeout(tick, delay);
    }

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ready, week]);

  return live;
}
