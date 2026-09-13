# Weekly Fantasy Analyzer

An in-season weekly fantasy football matchup tool. Every week it:

- Shows projected stat lines for all active NFL players
- Lets you build "my team" vs. "opponent team" (auto-filled from ESPN, editable by hand)
- Simulates a win probability for the matchup via Monte Carlo simulation over each player's empirical scoring distribution
- Lets you toggle scoring between Standard / Half-PPR / Full PPR, or fully customize point values per stat category

## How it works

No paid hosting, no backend server — everything runs client-side in the browser or on a scheduled GitHub Actions job:

- A daily GitHub Actions workflow (`.github/workflows/weekly-projections.yml`) pulls the week's schedule and player stats via `nflreadpy`, generates raw projected stat lines and per-player game history for Monte Carlo sampling, and commits the results as JSON under `public/data/`.
- A second workflow (`.github/workflows/deploy.yml`) builds the React/Vite frontend and publishes it to GitHub Pages on every push to `main` — including that daily data commit, so the deployed site refreshes automatically.
- The frontend fetches that JSON at load time and does everything else — scoring, team building, and the Monte Carlo simulation — entirely in the browser, persisted to `localStorage`.
- ESPN roster sync is planned but not yet built; team rosters are entered manually for now.

## Development

```
npm install
npm run dev       # local dev server
python3 scripts/generate_projections.py   # regenerate public/data/ locally (requires nflreadpy + polars)
npm run build      # production build to dist/
```

## Status

Data pipeline and frontend (matchup comparison, players table, team builder, scoring settings) are built. ESPN sync is not yet implemented — see `CLAUDE.md` (untracked, local reference) for the full architecture and build plan.
