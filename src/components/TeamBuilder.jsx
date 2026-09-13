import { useState } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { computeFantasyPoints } from "../lib/scoring.js";
import { ROSTER_SLOTS } from "../lib/rosterSlots.js";
import { PlayerPicker } from "./PlayerPicker.jsx";
import { PlusIcon, SwapIcon } from "./icons.jsx";

export function TeamBuilder({ initialTeam = "mine" }) {
  const { weekData, scoringSettings, myRoster, setMyRoster, opponentRoster, setOpponentRoster } =
    useApp();
  const [team, setTeam] = useState(initialTeam);
  const [pickingSlot, setPickingSlot] = useState(null);

  const roster = team === "mine" ? myRoster : opponentRoster;
  const setRoster = team === "mine" ? setMyRoster : setOpponentRoster;
  const projections = weekData.status === "ready" ? weekData.projections : {};

  function assign(slotId, playerId) {
    setRoster({ ...roster, [slotId]: playerId });
    setPickingSlot(null);
  }

  function clearSlot(slotId) {
    setRoster({ ...roster, [slotId]: null });
  }

  return (
    <div className="page page--narrow">
      <div className="team-builder__tabs">
        <button
          className={`team-builder__tab${team === "mine" ? " active--mine" : ""}`}
          onClick={() => setTeam("mine")}
        >
          My Team
        </button>
        <button
          className={`team-builder__tab${team === "opponent" ? " active--opponent" : ""}`}
          onClick={() => setTeam("opponent")}
        >
          Opponent
        </button>
      </div>

      <div className="manual-badge">MANUAL MODE — ESPN sync not connected yet</div>

      <div className="roster-grid">
        {ROSTER_SLOTS.map((slot) => {
          const playerId = roster[slot.id];
          const player = playerId ? projections[playerId] : null;
          const points = player ? computeFantasyPoints(player.projected_stats, scoringSettings.values) : 0;

          return (
            <div className={`slot-card${player ? "" : " slot-card--empty"}`} key={slot.id}>
              <span className="slot-card__slot">{slot.label}</span>
              <div className="slot-card__player">
                <div className="slot-card__name">{player ? player.player_name : "Empty"}</div>
                {player && (
                  <div className="slot-card__meta">
                    {player.position} · {player.team}
                  </div>
                )}
              </div>
              {player && <span className="slot-card__pts mono">{points.toFixed(1)}</span>}
              <button
                className="slot-btn"
                aria-label={player ? "Swap player" : "Add player"}
                onClick={() => setPickingSlot(slot)}
              >
                {player ? <SwapIcon /> : <PlusIcon />}
              </button>
              {player && (
                <button
                  className="slot-btn"
                  aria-label="Remove player"
                  style={{ marginLeft: 6 }}
                  onClick={() => clearSlot(slot.id)}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>

      {pickingSlot && (
        <PlayerPicker
          slot={pickingSlot}
          onPick={(playerId) => assign(pickingSlot.id, playerId)}
          onClose={() => setPickingSlot(null)}
        />
      )}
    </div>
  );
}
