import { useEffect, useState } from "react";

async function fetchJSON(path) {
  const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

/** Loads the current week's data/latest.json manifest, then that week's
 * projections.json + history.json -- the only three files the whole
 * frontend needs to fetch. Static data, generated daily by
 * .github/workflows/weekly-projections.yml. */
export function useWeekData() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const latest = await fetchJSON("data/latest.json");
        const weekStr = String(latest.week).padStart(2, "0");
        const [projections, history] = await Promise.all([
          fetchJSON(`data/week_${weekStr}/projections.json`),
          fetchJSON(`data/week_${weekStr}/history.json`),
        ]);
        if (!cancelled) {
          setState({
            status: "ready",
            season: latest.season,
            week: latest.week,
            projections,
            history,
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
