import { createContext, useContext, useMemo, useState } from "react";
import { useWeekData } from "./weekData.js";
import { loadJSON, saveJSON } from "./storage.js";
import { DEFAULT_SCORING_SETTINGS } from "./scoring.js";
import { ROSTER_SLOTS } from "./rosterSlots.js";

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
  const [myRoster, setMyRosterState] = useState(() =>
    loadJSON("myRoster", emptyRoster())
  );
  const [opponentRoster, setOpponentRosterState] = useState(() =>
    loadJSON("opponentRoster", emptyRoster())
  );

  function setScoringSettings(next) {
    setScoringSettingsState(next);
    saveJSON("scoringSettings", next);
  }

  function setMyRoster(next) {
    setMyRosterState(next);
    saveJSON("myRoster", next);
  }

  function setOpponentRoster(next) {
    setOpponentRosterState(next);
    saveJSON("opponentRoster", next);
  }

  const value = useMemo(
    () => ({
      weekData,
      scoringSettings,
      setScoringSettings,
      myRoster,
      setMyRoster,
      opponentRoster,
      setOpponentRoster,
    }),
    [weekData, scoringSettings, myRoster, opponentRoster]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
