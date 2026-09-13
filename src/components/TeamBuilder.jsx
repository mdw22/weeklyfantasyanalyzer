import { useMemo, useState } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { computeFantasyPoints } from "../lib/scoring.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "../lib/rosterSlots.js";
import { PlayerPicker } from "./PlayerPicker.jsx";
import { PlusIcon, SwapIcon } from "./icons.jsx";

const STARTER_SLOTS = ROSTER_SLOTS.filter((s) => STARTER_SLOT_IDS.includes(s.id));
const BENCH_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("BN"));
const IR_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("IR"));

function SlotRow({ slot, player, points, edited, onPick, onClear }) {
  if (!player) {
    return (
      <div className="slot empty">
        <span className="slot-tag">{slot.label}</span>
        <span className="slot-name">Empty</span>
        <span />
        <div className="slot-actions">
          <button className="slot-icon-btn slot-icon-btn--add" aria-label="Add player" onClick={onPick}>
            <PlusIcon />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="slot">
      <span className="slot-tag">{slot.label}</span>
      <span className="slot-name">
        {player.player_name}
        <span className="slot-name__meta">{player.team}</span>
        {edited && <span className="slot-name__meta slot-name__meta--edited"> &middot; edited</span>}
      </span>
      <span className="slot-pts mono">{points.toFixed(1)}</span>
      <div className="slot-actions">
        <button className="slot-icon-btn" aria-label="Swap player" onClick={onPick}>
          <SwapIcon />
        </button>
        <button className="slot-icon-btn" aria-label="Remove player" onClick={onClear}>
          ×
        </button>
      </div>
    </div>
  );
}

function RosterSection({ title, slots, roster, projections, scoringValues, isEdited, onPick, onClear }) {
  return (
    <div className="card roster-section">
      <div className="section-header">{title}</div>
      {slots.map((slot) => {
        const playerId = roster[slot.id];
        const player = playerId ? projections[playerId] : null;
        const points = player ? computeFantasyPoints(player.projected_stats, scoringValues) : 0;
        return (
          <SlotRow
            key={slot.id}
            slot={slot}
            player={player}
            points={points}
            edited={isEdited(slot.id)}
            onPick={() => onPick(slot)}
            onClear={() => onClear(slot.id)}
          />
        );
      })}
    </div>
  );
}

function formatSyncDate(isoString) {
  try {
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(
      new Date(isoString)
    );
  } catch {
    return null;
  }
}

export function TeamBuilder({ initialTeam = "mine" }) {
  const {
    weekData,
    scoringSettings,
    myRoster,
    opponentRoster,
    assignPlayer,
    clearSlot,
    isSlotOverridden,
    isSynced,
    syncedAt,
  } = useApp();
  const [team, setTeam] = useState(initialTeam);
  const [pickingSlot, setPickingSlot] = useState(null);

  const roster = team === "mine" ? myRoster : opponentRoster;
  const projections = weekData.status === "ready" ? weekData.projections : {};

  // A real player can only occupy one slot across both rosters at once.
  // Recomputed from current state every render, so a picker opened right
  // after a removal immediately sees that player as available again.
  const excludedIds = useMemo(() => {
    const ids = new Set();
    for (const [slotId, playerId] of Object.entries(myRoster)) {
      if (playerId && !(team === "mine" && pickingSlot?.id === slotId)) ids.add(playerId);
    }
    for (const [slotId, playerId] of Object.entries(opponentRoster)) {
      if (playerId && !(team === "opponent" && pickingSlot?.id === slotId)) ids.add(playerId);
    }
    return ids;
  }, [myRoster, opponentRoster, team, pickingSlot]);

  function handlePick(playerId) {
    assignPlayer(team, pickingSlot.id, playerId);
    setPickingSlot(null);
  }

  function handleClear(slotId) {
    clearSlot(team, slotId);
  }

  // Only worth flagging when sync is actually active -- when it's off,
  // every slot is manual by definition and the tag would just be noise.
  const isEdited = (slotId) => isSynced && isSlotOverridden(team, slotId);

  const sectionProps = {
    roster,
    projections,
    scoringValues: scoringSettings.values,
    isEdited,
    onPick: setPickingSlot,
    onClear: handleClear,
  };

  const syncDateLabel = isSynced ? formatSyncDate(syncedAt) : null;

  return (
    <div className="page page--narrow">
      <div className="team-builder__header">
        <span className="page-title">Edit Lineup</span>
        <span className="badge-chip">
          {isSynced ? `SYNCED FROM ESPN${syncDateLabel ? ` · ${syncDateLabel}` : ""}` : "MANUAL MODE"}
        </span>
      </div>

      <div className="team-builder__tabs">
        <button className={`tab${team === "mine" ? " active" : ""}`} onClick={() => setTeam("mine")}>
          My Team
        </button>
        <button className={`tab${team === "opponent" ? " active" : ""}`} onClick={() => setTeam("opponent")}>
          Opponent
        </button>
      </div>

      <RosterSection title="STARTERS" slots={STARTER_SLOTS} {...sectionProps} />
      <RosterSection title="BENCH (7)" slots={BENCH_SLOTS} {...sectionProps} />
      <RosterSection title="IR (1)" slots={IR_SLOTS} {...sectionProps} />

      {pickingSlot && (
        <PlayerPicker
          slot={pickingSlot}
          excludedIds={excludedIds}
          onPick={handlePick}
          onClose={() => setPickingSlot(null)}
        />
      )}
    </div>
  );
}
