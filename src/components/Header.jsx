import { useEffect, useRef } from "react";
import { GearIcon } from "./icons.jsx";

const TABS = [
  { id: "matchup", label: "Matchup" },
  { id: "players", label: "Players" },
  { id: "team", label: "Team Builder" },
  { id: "roster", label: "Roster Moves" },
];

export function Header({ view, onNavigate, onOpenSettings }) {
  const activeRef = useRef(null);

  // On a narrow screen the nav scrolls horizontally instead of wrapping (see
  // the CSS). Without this, loading straight into a tab other than the first
  // (e.g. Roster Moves, currently last) renders it clipped at the right edge
  // until the user manually scrolls -- scroll it into view on mount/switch.
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [view]);

  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">Weekly Fantasy Analyzer</div>
        <nav className="app-header__nav">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              ref={view === tab.id ? activeRef : undefined}
              className={`tab${view === tab.id ? " active" : ""}`}
              onClick={() => onNavigate(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <button className="icon-btn" aria-label="Scoring settings" onClick={onOpenSettings}>
          <GearIcon />
        </button>
      </div>
    </header>
  );
}
