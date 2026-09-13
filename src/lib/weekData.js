import { useEffect, useState } from "react";

async function fetchJSON(path) {
  const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

/** Same as fetchJSON, but a missing/failed fetch resolves to null instead
 * of throwing -- used for espn-sync.json, which is optional: it may not
 * exist yet (sync never configured) or may be stale from a failed daily
 * run, and neither case should break the rest of the app. */
async function fetchJSONOptional(path) {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Loads the current week's data/latest.json manifest, then that week's
 * projections.json + history.json (required) and espn-sync.json
 * (optional). Static data, generated daily by
 * .github/workflows/weekly-projections.yml. */
export function useWeekData() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const latest = await fetchJSON("data/latest.json");
        const weekStr = String(latest.week).padStart(2, "0");
        const [projections, history, espnSync] = await Promise.all([
          fetchJSON(`data/week_${weekStr}/projections.json`),
          fetchJSON(`data/week_${weekStr}/history.json`),
          // latest.json says whether espn-sync.json actually exists this
          // week -- skip the request entirely when it doesn't, instead of
          // firing a fetch that's guaranteed to 404 (and log as one, in
          // every browser, no matter how the rejection is caught).
          latest.espnSync
            ? fetchJSONOptional(`data/week_${weekStr}/espn-sync.json`)
            : Promise.resolve(null),
        ]);
        if (!cancelled) {
          setState({
            status: "ready",
            season: latest.season,
            week: latest.week,
            projections,
            history,
            espnSync,
          });
        }
      } catch (error) {
        if (!cancelled) setState({ status: "error", error });
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
