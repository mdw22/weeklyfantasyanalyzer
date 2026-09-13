import { useState } from "react";
import { AppProvider } from "./lib/AppContext";
import { Header } from "./components/Header";
import { MatchupComparison } from "./components/MatchupComparison";
import { WeeklyProjectionsTable } from "./components/WeeklyProjectionsTable";
import { TeamBuilder } from "./components/TeamBuilder";
import { ScoringSettingsPanel } from "./components/ScoringSettingsPanel";

function AppShell() {
  const [view, setView] = useState("matchup");
  const [teamBuilderTeam, setTeamBuilderTeam] = useState("mine");
  const [settingsOpen, setSettingsOpen] = useState(false);

  function editTeam(team) {
    setTeamBuilderTeam(team);
    setView("team");
  }

  return (
    <>
      <Header view={view} onNavigate={setView} onOpenSettings={() => setSettingsOpen(true)} />

      {view === "matchup" && <MatchupComparison onEditTeam={editTeam} />}
      {view === "players" && (
        <div className="page">
          <WeeklyProjectionsTable />
        </div>
      )}
      {view === "team" && <TeamBuilder initialTeam={teamBuilderTeam} key={teamBuilderTeam} />}

      {settingsOpen && <ScoringSettingsPanel onClose={() => setSettingsOpen(false)} />}
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}
