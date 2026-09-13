import { useMemo, useState } from "react";
import { useApp } from "../lib/AppContext.jsx";
import { usePlayerList } from "../lib/usePlayerList.js";
import { ChevronDownIcon, ChevronUpIcon } from "./icons.jsx";

const POSITIONS = ["All", "QB", "RB", "WR", "TE", "K", "DEF"];
const PAGE_SIZE = 20;

const COLUMNS = [
  { key: "name", label: "Player", sortKey: "name" },
  { key: "position", label: "Pos" },
  { key: "team", label: "Team" },
  { key: "points", label: "Proj Pts", sortKey: "points", num: true },
];

export function WeeklyProjectionsTable() {
  const { weekData, scoringSettings, myRoster } = useApp();
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("All");
  const [sortBy, setSortBy] = useState("points");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);

  const myPlayerIds = useMemo(
    () => new Set(Object.values(myRoster).filter(Boolean)),
    [myRoster]
  );

  const players = usePlayerList({
    projections: weekData.status === "ready" ? weekData.projections : null,
    scoringValues: scoringSettings.values,
    search,
    positions: position === "All" ? null : [position],
    sortBy,
    sortDir,
  });

  const totalPages = Math.max(1, Math.ceil(players.length / PAGE_SIZE));
  const pageStart = page * PAGE_SIZE;
  const pageItems = players.slice(pageStart, pageStart + PAGE_SIZE);

  function toggleSort(key) {
    if (sortBy === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortBy(key);
      setSortDir("desc");
    }
    setPage(0);
  }

  if (weekData.status === "loading") {
    return <div className="state-message">Loading players…</div>;
  }
  if (weekData.status === "error") {
    return <div className="state-message">Couldn't load this week's projections.</div>;
  }

  return (
    <div>
      <div className="table-toolbar">
        <input
          className="search-input"
          placeholder="Search players…"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
        />
        <div className="filter-pills">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              className={`filter-pill${position === pos ? " active" : ""}`}
              onClick={() => {
                setPosition(pos);
                setPage(0);
              }}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      <div className="card table-wrap">
        <table className="players-table">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className={`${col.sortKey ? "sortable" : ""}${col.num ? " num" : ""}`}
                  onClick={col.sortKey ? () => toggleSort(col.sortKey) : undefined}
                >
                  {col.label}
                  {col.sortKey === sortBy &&
                    (sortDir === "desc" ? <ChevronDownIcon /> : <ChevronUpIcon />)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageItems.map((p) => (
              <tr key={p.id} className={myPlayerIds.has(p.id) ? "table-row--mine" : ""}>
                <td>
                  {myPlayerIds.has(p.id) && <span className="dot dot--mine" style={{ marginRight: 8, display: "inline-block" }} />}
                  {p.name}
                </td>
                <td>{p.position}</td>
                <td>{p.team}</td>
                <td className="num">{p.points.toFixed(1)}</td>
              </tr>
            ))}
            {pageItems.length === 0 && (
              <tr>
                <td colSpan={COLUMNS.length} className="state-message">
                  No players match.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <span>
          Showing {players.length === 0 ? 0 : pageStart + 1}–
          {Math.min(pageStart + PAGE_SIZE, players.length)} of {players.length}
        </span>
        <div className="pagination__controls">
          <button
            className="pagination__btn"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            Prev
          </button>
          <button
            className="pagination__btn"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
