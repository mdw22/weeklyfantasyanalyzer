# Weekly Fantasy Analyzer

An in-season weekly fantasy football matchup tool. Every week it:

- Shows projected stat lines for all active NFL players
- Lets you build "my team" vs. "opponent team" (auto-filled from ESPN, editable by hand)
- Simulates a win probability for the matchup via Monte Carlo simulation over each player's empirical scoring distribution
- Lets you toggle scoring between Standard / Half-PPR / Full PPR, or fully customize point values per stat category

## How it works

No paid hosting, no backend server — everything runs client-side in the browser or on a scheduled GitHub Actions job:

- A daily GitHub Actions workflow pulls the week's schedule, rosters, and injury data via `nflreadpy`, generates raw projected stat lines and empirical scoring distributions per player, and commits the results as JSON.
- The same daily job syncs ESPN league data (your roster, your opponent, and their roster) using stored credentials, avoiding any client-side CORS or credential-exposure issues.
- The React frontend fetches that JSON at load time and does everything else — scoring, team building, and the Monte Carlo simulation — entirely in the browser.

## Status

Early planning stage — no application code yet. See `CLAUDE.md` (untracked, local reference) for the full architecture and build plan.
