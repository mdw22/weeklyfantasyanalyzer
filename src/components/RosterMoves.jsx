import { useMemo } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { useLiveScores } from "../lib/liveScores.js";
import { currentRisk } from "../lib/lineupAdvisor.js";
import {
  MAX_PICKUP_PAIRS,
  PICKUP_MARGIN,
  getPickupsAndDrops,
  getTradeTargets,
  positionSummary,
  rosterFlags,
} from "../lib/rosterMoves.js";
import { RiskPill } from "./LineupAdvisor.jsx";

const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : n % 10 < 4 ? n % 10 : 0]}`;

const VERDICT_LABEL = { strength: "STRENGTH", adequate: "ADEQUATE", need: "NEED" };

export function GlanceSection({ summary, flags }) {
  return (
    <div className="card roster-section">
      <div className="section-header">MY TEAM AT A GLANCE</div>
      <div className="glance-grid">
        {summary.map((s) => (
          <div className="glance-tile" key={s.position}>
            <span className="slot-tag">{s.position}</span>
            <div className="glance-tile__body">
              <div className="glance-tile__count mono">x{s.count}</div>
              <div className="glance-tile__best">
                {s.best ? (
                  <>
                    {s.best.name} <span className="mono">{s.best.points.toFixed(1)}</span>
                  </>
                ) : (
                  "none rostered"
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="section-header roster-subheader">STATUS FLAGS</div>
      {flags.length === 0 && <div className="advisor-panel__note roster-note">No injury or bye flags on your roster.</div>}
      {flags.map((f) => (
        <div className="slot" key={f.playerId}>
          <span className="slot-tag">{f.slotLabel}</span>
          <span className="slot-name">
            {f.name}
            <span className="slot-name__meta">
              {f.position} · {f.team}
            </span>
            <RiskPill risk={f.risk} />
          </span>
          <span className="slot-pts mono">{f.points.toFixed(1)}</span>
          <span />
        </div>
      ))}
    </div>
  );
}

export function PickupsSection({ result, live, projections }) {
  const statusOf = (p) => currentRisk(p.playerId, projections[p.playerId], live);
  return (
    <div className="card roster-section">
      <div className="section-header">PICKUPS &amp; DROPS</div>
      {!result.freeAgentsAvailable && (
        <div className="advisor-panel__note roster-note">
          Needs ESPN sync — it tells us who's rostered, so we can tell who's actually available.
        </div>
      )}
      {result.freeAgentsAvailable && result.pairs.length === 0 && (
        <div className="advisor-panel__note roster-note">
          No clear upgrades — no available free agent beats a bench player by {PICKUP_MARGIN}+ points.
        </div>
      )}
      {result.pairs.map((pair) => (
        <div className="move-row" key={pair.drop.playerId}>
          <span className="slot-tag">{pair.position}</span>
          <div className="move-row__players">
            <div className="move-row__line">
              <span className="move-row__verb">DROP</span>
              <span>
                {pair.drop.name}
                <span className="slot-name__meta">
                  {pair.drop.team} · bench
                </span>
                <RiskPill risk={statusOf(pair.drop)} />
              </span>
              <span className="mono move-row__pts">{pair.drop.points.toFixed(1)}</span>
            </div>
            <div className="move-row__line">
              <span className="move-row__verb">ADD</span>
              <span>
                {pair.add.name}
                <span className="slot-name__meta">{pair.add.team} · free agent</span>
                <RiskPill risk={statusOf(pair.add)} />
              </span>
              <span className="mono move-row__pts">{pair.add.points.toFixed(1)}</span>
            </div>
          </div>
          <span className="move-row__gain mono">+{pair.gain.toFixed(1)}</span>
        </div>
      ))}
      <div className="advisor-panel__note roster-note">
        Based on projected points only — it doesn't know how recently you added someone. Bench players only
        (starters and the IR slot are never suggested as drops), same position only, up to {MAX_PICKUP_PAIRS} swaps.
      </div>
    </div>
  );
}

export function TradeSection({ result }) {
  return (
    <div className="card roster-section">
      <div className="section-header">TRADE TARGETS — WHERE YOU'RE DEEP AND THIN</div>
      {!result.leagueKnown && (
        <div className="advisor-panel__note roster-note">
          Needs ESPN sync — comparing against the league requires knowing every team's roster.
        </div>
      )}
      {result.positions.map((p) => (
        <div className="trade-row" key={p.position}>
          <span className="slot-tag">{p.position}</span>
          <div className="trade-row__body">
            <div>
              <span className={`verdict-pill verdict-pill--${p.verdict}`}>{VERDICT_LABEL[p.verdict]}</span>
              <span className="trade-row__headline">
                Your {p.starters === 1 ? "best" : `${ordinal(p.starters)}-best`} {p.position}
                {p.nthName ? ` (${p.nthName}, ${p.nthPoints.toFixed(1)})` : " — none rostered"} ranks{" "}
                <strong>{ordinal(p.rank)}</strong> of {p.teamCount} teams
              </span>
            </div>
            <div className="trade-row__detail">
              You roster {p.rostered} · league median {p.median.toFixed(1)}
              {p.verdict === "strength" && p.surplus &&
                ` · beyond your starters: ${p.surplus.name} ${p.surplus.points.toFixed(1)}`}
            </div>
            {p.bestFreeAgent && (
              <div className="trade-row__detail">
                A free agent may fix this without a trade: {p.bestFreeAgent.name} {p.bestFreeAgent.points.toFixed(1)}{" "}
                (+{p.bestFreeAgent.gain.toFixed(1)})
              </div>
            )}
          </div>
        </div>
      ))}
      <div className="advisor-panel__note roster-note">
        Each position is compared with every team's last starter there (their {"Nth"}-best, where N is how many
        start). Only rostered players on active NFL rosters count. Need = bottom 4 of the league; Strength = a
        starter-quality player beyond your starters, or a top-quarter group with depth. This shows where you're
        deep or thin — it doesn't propose specific trades.
      </div>
    </div>
  );
}

export function RosterMoves() {
  const { weekData, scoringSettings, myRoster } = useApp();
  const live = useLiveScores(weekData);
  const ready = weekData.status === "ready";
  const projections = ready ? weekData.projections : {};
  const ownership = ready ? weekData.espnSync?.ownership ?? null : null;
  const myTeamId = ready ? weekData.espnSync?.myTeam?.espnTeamId ?? null : null;
  const values = scoringSettings.values;

  const summary = useMemo(() => positionSummary(myRoster, projections, values), [myRoster, projections, values]);
  const flags = useMemo(() => rosterFlags(myRoster, projections, live, values), [myRoster, projections, live, values]);
  const pickups = useMemo(
    () => getPickupsAndDrops(myRoster, projections, ownership, values),
    [myRoster, projections, ownership, values]
  );
  const trades = useMemo(
    () => getTradeTargets(myRoster, projections, ownership, myTeamId, values),
    [myRoster, projections, ownership, myTeamId, values]
  );

  if (weekData.status === "loading") return <div className="state-message">Loading roster…</div>;
  if (weekData.status === "error") return <div className="state-message">Couldn't load this week's data.</div>;

  return (
    <div className="page page--narrow">
      <div className="team-builder__header">
        <span className="page-title">Roster Moves</span>
        <span className="badge-chip">SEASON VIEW</span>
      </div>
      <GlanceSection summary={summary} flags={flags} />
      <PickupsSection result={pickups} live={live} projections={projections} />
      <TradeSection result={trades} />
    </div>
  );
}
