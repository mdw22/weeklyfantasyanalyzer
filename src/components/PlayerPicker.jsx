import { useState } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { usePlayerList } from "../lib/usePlayerList.js";
import { CloseIcon } from "./icons.jsx";

/** Modal for filling one roster slot. Scoped to the slot's eligible
 * positions, reuses the same filter/sort logic as WeeklyProjectionsTable. */
export function PlayerPicker({ slot, onPick, onClose }) {
  const { weekData, scoringSettings } = useApp();
  const [search, setSearch] = useState("");

  const players = usePlayerList({
    projections: weekData.status === "ready" ? weekData.projections : null,
    scoringValues: scoringSettings.values,
    search,
    positions: slot.eligible,
    sortBy: "points",
    sortDir: "desc",
  });

  return (
    <div className="picker-backdrop" onClick={onClose}>
      <div className="picker" onClick={(e) => e.stopPropagation()}>
        <div className="picker__header">
          <input
            className="search-input"
            autoFocus
            placeholder={`Search ${slot.eligible.join("/")}…`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button className="icon-btn" aria-label="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        <div className="picker__list">
          {players.slice(0, 100).map((p) => (
            <div key={p.id} className="picker__item" onClick={() => onPick(p.id)}>
              <span>
                {p.name} <span className="roster-row__meta">{p.position} · {p.team}</span>
              </span>
              <span className="mono">{p.points.toFixed(1)}</span>
            </div>
          ))}
          {players.length === 0 && <div className="state-message">No players match.</div>}
        </div>
      </div>
    </div>
  );
}
