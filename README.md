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
- ESPN roster sync (optional) also runs in the daily job — see below. Manual roster entry in Team Builder always works standalone, whether or not sync is configured.

## Development

```
npm install
npm run dev       # local dev server
python3 scripts/generate_projections.py   # regenerate public/data/ locally (requires nflreadpy + polars)
python3 scripts/sync_espn.py              # optional -- see ESPN Sync Setup below; no-ops without credentials
npm run build      # production build to dist/
```

## ESPN Sync Setup (optional, one-time)

Without this, Team Builder works entirely with manual roster entry — nothing below is required. To have your real ESPN league roster and this week's opponent auto-fill each day instead:

1. Log into `fantasy.espn.com` in a normal browser, with your league open.
2. Open dev tools → Application (Chrome) / Storage (Firefox) → Cookies → `https://fantasy.espn.com`.
3. Copy the `espn_s2` and `SWID` cookie values.
4. In this repo's **Settings → Secrets and variables → Actions**:
   - Add **secrets** `ESPN_S2` and `ESPN_SWID` with those cookie values. Never paste these into chat, commit them, or put them anywhere but this one GitHub secrets page.
   - Add **variables** `ESPN_LEAGUE_ID` and `ESPN_TEAM_ID` — both are plain numbers visible, unauthenticated, in your league's own `fantasy.espn.com` URL, so they're not sensitive and don't need to be secrets.
5. The next daily run (or a manual `workflow_dispatch` run of "Update Weekly Projections") will sync automatically. Team Builder's badge switches from "MANUAL MODE" to "SYNCED FROM ESPN" once it's working.

A slot you edit manually in Team Builder during a given week is never overwritten by that week's sync — sync only fills gaps you haven't touched yourself. The override resets each new week.

**Verified against a real league.** Two real issues were found and fixed this way: the API host (`fantasy.espn.com` redirects and 403s; the correct host is `lm-api-reads.fantasy.espn.com`) and team defenses (see below). If sync ever 401s for you, double-check your `SWID` cookie value includes its surrounding curly braces (`{...}`) exactly as shown in your browser — that's the most common cause.

## Known limitation: DEF scoring

Team defense/special-teams (DEF) projections cover sacks, interceptions, fumble recoveries, safeties, defensive/special-teams touchdowns, blocked kicks, and 2-point returns — but **not points-allowed or yards-allowed tiers**, which many leagues also score. Tiered/bucketed scoring doesn't fit the simple per-stat point-value model every other stat in this app uses, so it was deliberately left out rather than bolted on. DEF projections and ESPN DEF sync both otherwise work normally.

## Status

Data pipeline (including team defenses), ESPN sync, and the full frontend (matchup comparison, players table, team builder, scoring settings) are built and verified against a real league. See `CLAUDE.md` (untracked, local reference) for the full architecture and build plan.
