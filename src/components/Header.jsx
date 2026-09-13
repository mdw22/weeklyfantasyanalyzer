import { GearIcon } from "./icons.jsx";

const TABS = [
  { id: "matchup", label: "Matchup" },
  { id: "players", label: "Players" },
  { id: "team", label: "Team Builder" },
];

export function Header({ view, onNavigate, onOpenSettings }) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">Weekly Fantasy Analyzer</div>
        <nav className="app-header__nav">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab${view === tab.id ? " active" : ""}`}
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
