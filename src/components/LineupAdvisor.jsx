import { formatDelta, getReplacementCandidates } from "../lib/lineupAdvisor.js";
import { effectivePoints } from "../lib/liveScores.js";
import { AlertIcon, ChevronDownIcon, ChevronUpIcon } from "./icons.jsx";

const LABEL = { QUESTIONABLE: "Questionable", DOUBTFUL: "Doubtful", OUT: "Out", IR: "IR" };

/** Availability tag for a player: amber pill (questionable/doubtful), red
 * pill (out/IR), or plain muted "BYE". Renders nothing when there's no risk.
 * Same shape and placement as the green live/final pill. */
export function RiskPill({ risk }) {
  if (!risk) return null;
  if (risk.severity === "bye") return <span className="risk-bye">BYE</span>;
  return <span className={`risk-pill risk-pill--${risk.severity}`}>{LABEL[risk.reason]}</span>;
}

/** Hard-to-miss banner above the roster: who might not play, and a link to
 * each one's suggested replacements. Tinted by the worst severity present. */
export function AdvisorBanner({ atRisk, projections, onShow }) {
  if (atRisk.length === 0) return null;
  const tone = atRisk.some((a) => a.severity === "out")
    ? "out"
    : atRisk.some((a) => a.severity === "risk")
      ? "risk"
      : "bye";
  const noun = atRisk.length === 1 ? "starter" : "starters";
  return (
    <div className={`advisor-banner advisor-banner--${tone}`} role="alert">
      <div className="advisor-banner__title">
        <AlertIcon />
        {atRisk.length} {noun} may not play this week
      </div>
      <ul className="advisor-banner__list">
        {atRisk.map((a) => (
          <li className="advisor-banner__item" key={a.slotId}>
            <strong>{projections[a.playerId]?.player_name}</strong>
            <span className="advisor-banner__slot">{a.slot.label}</span>
            <RiskPill risk={a} />
            <button className="advisor-link" onClick={() => onShow(a.slotId)}>
              See replacements
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Collapsible replacements list under a starter slot. Read-only: it
 * recommends, it never changes the roster or touches ESPN.
 *
 * `flagged` slots (the advisor thinks the starter may not play) read
 * "Suggested replacements". A healthy slot the user opened on demand reads
 * "Compare replacements" -- same ranking, just no implied warning.
 * `pastKickoff` means the starter's own game is underway or over: the slot is
 * locked, so this is a retrospective comparison, not advice to act on. */
export function ReplacementPanel({ slot, roster, projections, ownership, scoringValues, live, expanded, onToggle, flagged = true, pastKickoff = false }) {
  const result = expanded
    ? getReplacementCandidates(slot, roster, projections, ownership, scoringValues, 5, live)
    : null;

  // What the candidates' deltas are measured against: the starter's current
  // number by the same rule Matchup uses (real points once his game has
  // begun, 0 if he's not playing, otherwise his projection). None for an
  // empty slot -- a delta against nothing is just the candidate's own total.
  const starterId = roster[slot.id];
  const starterEntry = starterId ? projections[starterId] : null;
  const baseline = expanded && starterEntry ? effectivePoints(starterId, starterEntry, live, scoringValues) : null;
  const basis = !baseline
    ? null
    : baseline.status === "final"
      ? "actual"
      : baseline.status === "in_progress"
        ? "so far"
        : baseline.sittingOut
          ? "not playing, counts as 0"
          : "projected";
  return (
    <div className="advisor-panel">
      <button className="advisor-panel__toggle" aria-expanded={expanded} onClick={onToggle}>
        {flagged ? "Suggested replacements" : "Compare replacements"}
        {pastKickoff && <span className="advisor-panel__reference">· past kickoff, for reference only</span>}
        {expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
      </button>
      {expanded && (
        <>
          {baseline && (
            <div className="advisor-panel__baseline">
              Compared with {starterEntry.player_name}: {baseline.points.toFixed(1)} {basis}
            </div>
          )}
          {result.candidates.map((c) => (
            <div className="advisor-candidate" key={c.playerId}>
              <span>
                {c.name}
                <span className="advisor-candidate__meta">
                  {c.position} · {c.team}
                </span>
                <RiskPill risk={c.risk} />
              </span>
              <span className="advisor-source">{c.source === "bench" ? "BENCH" : "FREE AGENT"}</span>
              <span className="advisor-candidate__score">
                <span className="advisor-candidate__pts">{c.points.toFixed(1)}</span>
                {baseline && (
                  <span className="advisor-candidate__delta">({formatDelta(c.points, baseline.points)})</span>
                )}
              </span>
            </div>
          ))}
          {result.candidates.length === 0 && (
            <div className="advisor-panel__note">No eligible replacements found.</div>
          )}
          {!result.freeAgentsAvailable && (
            <div className="advisor-panel__note">
              Showing bench only — free-agent suggestions need ESPN sync (it tells us who's rostered).
            </div>
          )}
          <div className="advisor-panel__note">
            Projections are an average of recent games; a player who just became a starter may be
            undervalued here.
          </div>
        </>
      )}
    </div>
  );
}
