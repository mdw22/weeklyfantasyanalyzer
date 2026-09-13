// Matches the prior draft tool's roster shape (see CLAUDE.md): 1 QB, 2 RB,
// 1 TE, 2 WR, 1 FLEX, 1 DEF, 1 K, 7 bench, 1 IR. Kept as data, not
// hardcoded into components, so the league shape can change later.
const FLEX_ELIGIBLE = ["RB", "WR", "TE"];
const ANY_ELIGIBLE = ["QB", "RB", "WR", "TE", "K", "DEF"];

function starters(label, count, eligible) {
  return Array.from({ length: count }, (_, i) => ({
    id: `${label}${count > 1 ? i + 1 : ""}`,
    label,
    eligible,
  }));
}

export const ROSTER_SLOTS = [
  ...starters("QB", 1, ["QB"]),
  ...starters("RB", 2, ["RB"]),
  ...starters("TE", 1, ["TE"]),
  ...starters("WR", 2, ["WR"]),
  ...starters("FLEX", 1, FLEX_ELIGIBLE),
  ...starters("DEF", 1, ["DEF"]),
  ...starters("K", 1, ["K"]),
  ...starters("BN", 7, ANY_ELIGIBLE),
  ...starters("IR", 1, ANY_ELIGIBLE),
];

export const STARTER_SLOT_IDS = ROSTER_SLOTS.filter(
  (s) => !s.id.startsWith("BN") && !s.id.startsWith("IR")
).map((s) => s.id);
