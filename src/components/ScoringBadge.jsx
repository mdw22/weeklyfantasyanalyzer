import { useApp } from "../lib/AppContext.jsx";
import { SCORING_PRESETS, matchingPresetKey } from "../lib/scoring.js";

/** Small always-visible chip naming the active scoring preset. Every point
 * value on screen depends on it, and a stray non-default preset (left over
 * from testing) silently skews every total -- this makes that obvious. */
export function ScoringBadge() {
  const { scoringSettings } = useApp();
  const key = matchingPresetKey(scoringSettings.values);
  const label = key === "custom" ? "Custom scoring" : SCORING_PRESETS[key].label;
  return <span className="badge-chip" title="Active scoring settings (change via the gear icon)">{label}</span>;
}
