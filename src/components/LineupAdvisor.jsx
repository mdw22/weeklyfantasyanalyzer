import { getReplacementCandidates } from "../lib/lineupAdvisor.js";
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

/** Collapsible "Suggested replacements" list under one flagged slot. Read-only:
 * the advisor recommends, it never changes the roster or touches ESPN. */
export function ReplacementPanel({ slot, roster, projections, ownership, scoringValues, live, expanded, onToggle }) {
  const result = expanded
    ? getReplacementCandidates(slot, roster, projections, ownership, scoringValues, 5, live)
    : null;
  return (
    <div className="advisor-panel">
      <button className="advisor-panel__toggle" aria-expanded={expanded} onClick={onToggle}>
        Suggested replacements
        {expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
      </button>
      {expanded && (
        <>
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
              <span className="advisor-candidate__pts">{c.points.toFixed(1)}</span>
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
