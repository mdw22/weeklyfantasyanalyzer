import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useWeekData } from "./weekData.js";
import { loadJSON, saveJSON } from "./storage.js";
import { DEFAULT_SCORING_SETTINGS } from "./scoring.js";
import { ROSTER_SLOTS } from "./rosterSlots.js";
import { emptyOverrides, mergeSyncedRoster, withOverride } from "./rosterSync.js";

const AppContext = createContext(null);

function emptyRoster() {
  const roster = {};
  for (const slot of ROSTER_SLOTS) roster[slot.id] = null;
  return roster;
}

export function AppProvider({ children }) {
  const weekData = useWeekData();

  const [scoringSettings, setScoringSettingsState] = useState(() =>
    loadJSON("scoringSettings", DEFAULT_SCORING_SETTINGS)
  );
  const [myRoster, setMyRosterRaw] = useState(() => loadJSON("myRoster", emptyRoster()));
  const [opponentRoster, setOpponentRosterRaw] = useState(() =>
    loadJSON("opponentRoster", emptyRoster())
  );
  const [myOverrides, setMyOverrides] = useState(() =>
    loadJSON("myRosterOverrides", emptyOverrides())
  );
  const [opponentOverrides, setOpponentOverrides] = useState(() =>
    loadJSON("opponentRosterOverrides", emptyOverrides())
  );

  function setScoringSettings(next) {
    setScoringSettingsState(next);
    saveJSON("scoringSettings", next);
  }

  function setMyRosterState(next) {
    setMyRosterRaw(next);
    saveJSON("myRoster", next);
  }

  function setOpponentRosterState(next) {
    setOpponentRosterRaw(next);
    saveJSON("opponentRoster", next);
  }

  /** The only way TeamBuilder should mutate a roster -- every manual
   * assign/clear both updates the roster and marks that slot as a
   * this-week override, so the daily ESPN sync won't silently revert it. */
  function assignPlayer(team, slotId, playerId) {
    const week = weekData.status === "ready" ? weekData.week : null;
    if (team === "mine") {
      setMyRosterState({ ...myRoster, [slotId]: playerId });
      const next = withOverride(myOverrides, week, slotId);
      setMyOverrides(next);
      saveJSON("myRosterOverrides", next);
    } else {
      setOpponentRosterState({ ...opponentRoster, [slotId]: playerId });
      const next = withOverride(opponentOverrides, week, slotId);
      setOpponentOverrides(next);
      saveJSON("opponentRosterOverrides", next);
    }
  }

  function clearSlot(team, slotId) {
    assignPlayer(team, slotId, null);
  }

  // Fold in the daily ESPN sync, if present for the current week: fills
  // every slot that hasn't been manually overridden this week, without
  // touching ones the user has already chosen differently. A missing or
  // failed sync (weekData.espnSync is null/undefined) just no-ops --
  // manual entry keeps working exactly as if sync were never configured.
  useEffect(() => {
    if (weekData.status !== "ready" || !weekData.espnSync) return;
    const week = weekData.week;

    if (weekData.espnSync.myTeam) {
      const { roster, changed } = mergeSyncedRoster(
        myRoster,
        weekData.espnSync.myTeam.roster,
        myOverrides,
        week
      );
      if (changed) setMyRosterState(roster);
    }
    if (weekData.espnSync.opponent) {
      const { roster, changed } = mergeSyncedRoster(
        opponentRoster,
        weekData.espnSync.opponent.roster,
        opponentOverrides,
        week
      );
      if (changed) setOpponentRosterState(roster);
    }

    // A new week starts with a clean override slate so next week's sync
    // applies in full again.
    if (myOverrides.week !== week) {
      const reset = { week, slots: {} };
      setMyOverrides(reset);
      saveJSON("myRosterOverrides", reset);
    }
    if (opponentOverrides.week !== week) {
      const reset = { week, slots: {} };
      setOpponentOverrides(reset);
      saveJSON("opponentRosterOverrides", reset);
    }
    // Only re-run when a new week's data (and its sync payload) arrives --
    // not on every roster/override change, or this would fight the
    // user's own edits on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weekData.status, weekData.week, weekData.espnSync]);

  const value = useMemo(
    () => ({
      weekData,
      scoringSettings,
      setScoringSettings,
      myRoster,
      opponentRoster,
      assignPlayer,
      clearSlot,
      isSynced: weekData.status === "ready" && !!weekData.espnSync,
      syncedAt: weekData.status === "ready" ? weekData.espnSync?.syncedAt ?? null : null,
    }),
    [weekData, scoringSettings, myRoster, opponentRoster, myOverrides, opponentOverrides]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
