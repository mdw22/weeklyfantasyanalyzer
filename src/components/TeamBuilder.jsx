import { Fragment, useMemo, useState } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { computeFantasyPoints } from "../lib/scoring.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "../lib/rosterSlots.js";
import { PlayerPicker } from "./PlayerPicker.jsx";
import { PlusIcon, SwapIcon } from "./icons.jsx";
import { availabilityRisk, getAtRiskSlots } from "../lib/lineupAdvisor.js";
import { AdvisorBanner, ReplacementPanel, RiskPill } from "./LineupAdvisor.jsx";

const STARTER_SLOTS = ROSTER_SLOTS.filter((s) => STARTER_SLOT_IDS.includes(s.id));
const BENCH_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("BN"));
const IR_SLOTS = ROSTER_SLOTS.filter((s) => s.id.startsWith("IR"));

function SlotRow({ slot, player, points, edited, risk, onPick, onClear }) {
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
    <div className="slot" id={`slot-${slot.id}`}>
      <span className="slot-tag">{slot.label}</span>
      <span className="slot-name">
        {player.player_name}
        <span className="slot-name__meta">{player.team}</span>
        <RiskPill risk={risk} />
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

function RosterSection({ title, slots, roster, projections, scoringValues, isEdited, riskFor, panelFor, onPick, onClear }) {
  return (
    <div className="card roster-section">
      <div className="section-header">{title}</div>
      {slots.map((slot) => {
        const playerId = roster[slot.id];
        const player = playerId ? projections[playerId] : null;
        const points = player ? computeFantasyPoints(player.projected_stats, scoringValues) : 0;
        return (
          <Fragment key={slot.id}>
            <SlotRow
              slot={slot}
              player={player}
              points={points}
              edited={isEdited(slot.id)}
              risk={riskFor(slot, player)}
              onPick={() => onPick(slot)}
              onClear={() => onClear(slot.id)}
            />
            {panelFor(slot)}
          </Fragment>
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
  const [expandedPanels, setExpandedPanels] = useState({});

  const roster = team === "mine" ? myRoster : opponentRoster;
  const projections = weekData.status === "ready" ? weekData.projections : {};
  // Who's rostered league-wide (from the ESPN sync) -- null when unknown.
  const ownership = weekData.status === "ready" ? weekData.espnSync?.ownership ?? null : null;

  // The advisor is about MY lineup only; the opponent tab just shows pills.
  const atRisk = useMemo(
    () => (team === "mine" && weekData.status === "ready" ? getAtRiskSlots(myRoster, projections) : []),
    [team, myRoster, projections, weekData.status]
  );
  const atRiskSlotIds = new Set(atRisk.map((a) => a.slotId));

  function showReplacements(slotId) {
    setExpandedPanels((prev) => ({ ...prev, [slotId]: true }));
    document.getElementById(`slot-${slotId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

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

  // Availability pills show on every row (informational). Only starters get
  // the banner / replacements panel -- bench and IR aren't playing anyway.
  const riskFor = (slot, player) => availabilityRisk(player);
  const panelFor = (slot) =>
    atRiskSlotIds.has(slot.id) ? (
      <ReplacementPanel
        slot={slot}
        roster={roster}
        projections={projections}
        ownership={ownership}
        scoringValues={scoringSettings.values}
        expanded={!!expandedPanels[slot.id]}
        onToggle={() => setExpandedPanels((prev) => ({ ...prev, [slot.id]: !prev[slot.id] }))}
      />
    ) : null;
  const noPanel = () => null;

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

      <AdvisorBanner atRisk={atRisk} projections={projections} onShow={showReplacements} />

      <RosterSection title="STARTERS" slots={STARTER_SLOTS} {...sectionProps} riskFor={riskFor} panelFor={panelFor} />
      <RosterSection title="BENCH (7)" slots={BENCH_SLOTS} {...sectionProps} riskFor={riskFor} panelFor={noPanel} />
      <RosterSection title="IR (1)" slots={IR_SLOTS} {...sectionProps} riskFor={riskFor} panelFor={noPanel} />

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
