// Pure logic for merging a daily ESPN sync into a locally-held roster,
// without ever stomping a manual edit made during the current week.
// Kept separate from AppContext.jsx so it's testable without React.

export function emptyOverrides() {
  return { week: null, slots: {} };
}

/** True if `slotId` was manually touched (assigned/cleared) during
 * `week`. Overrides from a prior week don't count -- a new week starts
 * clean and sync applies fully again. */
export function isOverridden(overrides, week, slotId) {
  return overrides.week === week && !!overrides.slots[slotId];
}

/** Returns a new overrides object with `slotId` marked as touched during
 * `week`. If `overrides` was left over from a different (older) week, its
 * stale slots are dropped first. */
export function withOverride(overrides, week, slotId) {
  const base = overrides.week === week ? overrides.slots : {};
  return { week, slots: { ...base, [slotId]: true } };
}

/** Merges a synced ESPN roster (`syncEntries`: [{slot, playerId}, ...])
 * into `roster` (slotId -> playerId|null), skipping any slot manually
 * overridden this week. Returns { roster, changed } -- `changed` lets the
 * caller skip a state update (and localStorage write) when sync agrees
 * with what's already there. */
export function mergeSyncedRoster(roster, syncEntries, overrides, week) {
  const next = { ...roster };
  let changed = false;
  for (const entry of syncEntries) {
    if (isOverridden(overrides, week, entry.slot)) continue;
    if (next[entry.slot] !== entry.playerId) {
      next[entry.slot] = entry.playerId;
      changed = true;
    }
  }
  return { roster: next, changed };
}
