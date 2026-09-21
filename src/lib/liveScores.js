import { useEffect, useState } from "react";
import { computeFantasyPoints } from "./scoring.js";

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

/** Polls live.json every `pollMs` while the calling screen is mounted.
 * Checks latest.json's `liveScores` flag first, so a week with no live.json
 * never triggers a request that would 404 (and log a console error no
 * matter how it's caught). Skips polling while the tab is hidden. */
export function useLiveScores(weekData, pollMs = 60000) {
  const [live, setLive] = useState(EMPTY_LIVE);
  const ready = weekData.status === "ready";
  const week = ready ? weekData.week : null;

  useEffect(() => {
    if (!ready) return undefined;
    let cancelled = false;
    let timer;
    const base = import.meta.env.BASE_URL;

    async function tick() {
      if (!document.hidden) {
        try {
          const res = await fetch(`${base}data/latest.json?t=${Date.now()}`);
          const latest = res.ok ? await res.json() : null;
          if (latest?.liveScores && latest.week === week) {
            const weekStr = String(week).padStart(2, "0");
            const liveRes = await fetch(`${base}data/week_${weekStr}/live.json?t=${Date.now()}`);
            if (liveRes.ok && !cancelled) setLive(await liveRes.json());
          }
        } catch {
          // Live scores are a bonus layer -- a failed poll just keeps the last data.
        }
      }
      if (!cancelled) timer = setTimeout(tick, pollMs);
    }

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ready, week, pollMs]);

  return live;
}
