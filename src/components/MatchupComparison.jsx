import { useMemo } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { resolvePlayerPoints, useLiveScores } from "../lib/liveScores.js";
import { ROSTER_SLOTS, STARTER_SLOT_IDS } from "../lib/rosterSlots.js";
import { simulateMatchup } from "../lib/monteCarlo.js";
import { ScoreRangeChart } from "./ScoreRangeChart.jsx";

function rosterRows(roster, projections, scoringValues, livePlayers) {
  return ROSTER_SLOTS.filter((s) => STARTER_SLOT_IDS.includes(s.id)).map((slot) => {
    const playerId = roster[slot.id];
    const player = playerId ? projections[playerId] : null;
    if (!player) return { slot, playerId, player, points: 0, status: "not_started", clock: null };
    const { points, status, clock } = resolvePlayerPoints(player, livePlayers[playerId], scoringValues);
    return { slot, playerId, player, points, status, clock };
  });
}

/** Muted status next to a player's points: "Final", or a pulsing dot plus
 * the game clock while live. Nothing pre-kickoff -- that number is still a
 * projection, same as before. */
function LiveTag({ status, clock }) {
  if (status === "final") return <span className="live-tag">Final</span>;
  if (status === "in_progress") {
    return (
      <span className="live-tag">
        <span className="live-dot" aria-hidden="true" />
        {clock}
      </span>
    );
  }
  return null;
}

function starterIds(roster) {
  return STARTER_SLOT_IDS.map((id) => roster[id]).filter(Boolean);
}

export function MatchupComparison({ onEditTeam }) {
  const { weekData, scoringSettings, myRoster, opponentRoster } = useApp();

  const ready = weekData.status === "ready";
  const projections = ready ? weekData.projections : {};
  const history = ready ? weekData.history : {};
  const live = useLiveScores(weekData);

  const myRows = useMemo(
    () => rosterRows(myRoster, projections, scoringSettings.values, live.players),
    [myRoster, projections, scoringSettings, live]
  );
  const oppRows = useMemo(
    () => rosterRows(opponentRoster, projections, scoringSettings.values, live.players),
    [opponentRoster, projections, scoringSettings, live]
  );

  const myTotal = myRows.reduce((sum, r) => sum + r.points, 0);
  const oppTotal = oppRows.reduce((sum, r) => sum + r.points, 0);

  const myPlayerIds = useMemo(() => starterIds(myRoster), [myRoster]);
  const oppPlayerIds = useMemo(() => starterIds(opponentRoster), [opponentRoster]);

  const bothTeamsFilled = myPlayerIds.length > 0 && oppPlayerIds.length > 0;

  const simulation = useMemo(() => {
    if (!ready || !bothTeamsFilled) return null;
    return simulateMatchup({
      myPlayerIds,
      opponentPlayerIds: oppPlayerIds,
      projections,
      history,
      scoringValues: scoringSettings.values,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, bothTeamsFilled, myRoster, opponentRoster, scoringSettings, weekData]);

  if (weekData.status === "loading") {
    return <div className="state-message">Loading matchup…</div>;
  }
  if (weekData.status === "error") {
    return <div className="state-message">Couldn't load this week's projections.</div>;
  }

  const winPct = simulation ? Math.round(simulation.winProbability * 100) : null;

  return (
    <div className="page">
      <div className="card matchup-summary">
        <div className="matchup-summary__team">
          <span className="team-label">
            <span className="dot dot--mine" /> My Team
          </span>
          <span className="team-total team-total--mine mono">{myTotal.toFixed(1)}</span>
        </div>

        <div className="win-hero">
          {simulation ? (
            <>
              <div className="win-hero__pct">{winPct}%</div>
              <div className="win-hero__label">Win Probability</div>
              <div className="win-bar">
                <div className="win-bar__segment--mine" style={{ width: `${winPct}%` }} />
                <div className="win-bar__segment--opponent" style={{ width: `${100 - winPct}%` }} />
              </div>
            </>
          ) : (
            <>
              <div className="win-hero__pct">{(myTotal - oppTotal >= 0 ? "+" : "") + (myTotal - oppTotal).toFixed(1)}</div>
              <div className="win-hero__label">
                {bothTeamsFilled ? "Projected Point Margin" : "Build both rosters to see a win probability"}
              </div>
            </>
          )}
        </div>

        <div className="matchup-summary__team matchup-summary__team--opponent">
          <span className="team-label">
            <span className="dot dot--opponent" /> Opponent
          </span>
          <span className="team-total team-total--opponent mono">{oppTotal.toFixed(1)}</span>
        </div>
      </div>

      <div className="roster-columns">
        <div className="card roster-col">
          <div className="roster-col__header">
            <span className="roster-col__title">
              <span className="dot dot--mine" /> My Team
            </span>
            <button className="edit-team-link" onClick={() => onEditTeam("mine")}>
              Edit Team
            </button>
          </div>
          {myRows.map((r) => (
            <div className="roster-row" key={r.slot.id}>
              <span className={r.player ? "roster-row__name" : "roster-row--empty"}>
                {r.player ? r.player.player_name : `Empty ${r.slot.label}`}
                {r.player && <span className="roster-row__meta">{r.player.position} · {r.player.team}</span>}
                {r.player && <LiveTag status={r.status} clock={r.clock} />}
              </span>
              <span className="roster-row__pts">{r.player ? r.points.toFixed(1) : "—"}</span>
            </div>
          ))}
        </div>

        <div className="card roster-col">
          <div className="roster-col__header">
            <button className="edit-team-link" onClick={() => onEditTeam("opponent")}>
              Edit Team
            </button>
            <span className="roster-col__title">
              Opponent <span className="dot dot--opponent" />
            </span>
          </div>
          {oppRows.map((r) => (
            <div className="roster-row" key={r.slot.id}>
              <span className={r.player ? "roster-row__name" : "roster-row--empty"}>
                {r.player ? r.player.player_name : `Empty ${r.slot.label}`}
                {r.player && <span className="roster-row__meta">{r.player.position} · {r.player.team}</span>}
                {r.player && <LiveTag status={r.status} clock={r.clock} />}
              </span>
              <span className="roster-row__pts">{r.player ? r.points.toFixed(1) : "—"}</span>
            </div>
          ))}
        </div>
      </div>

      {simulation && <ScoreRangeChart my={simulation.my} opponent={simulation.opponent} />}
    </div>
  );
}
