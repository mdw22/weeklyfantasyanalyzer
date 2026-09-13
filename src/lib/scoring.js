// Stat categories present in data/week_{NN}/projections.json + history.json
// (see scripts/generate_projections.py STAT_COLUMNS). Grouped for the
// settings drawer UI.
export const STAT_FIELDS = [
  { key: "passing_yards", label: "Passing Yard", group: "Passing" },
  { key: "passing_tds", label: "Passing TD", group: "Passing" },
  { key: "passing_interceptions", label: "Interception", group: "Passing" },
  { key: "rushing_yards", label: "Rushing Yard", group: "Rushing" },
  { key: "rushing_tds", label: "Rushing TD", group: "Rushing" },
  { key: "receptions", label: "Reception", group: "Receiving" },
  { key: "receiving_yards", label: "Receiving Yard", group: "Receiving" },
  { key: "receiving_tds", label: "Receiving TD", group: "Receiving" },
  { key: "fumbles_lost_total", label: "Fumble Lost", group: "Misc" },
];

// Full-PPR values match the prior draft tool's league config (see
// CLAUDE.md) so both tools agree: 0.04/yd passing (1pt/25yd), 4pt passing
// TD, -2 INT; 0.1/yd rushing+receiving (1pt/10yd), 6pt rushing+receiving
// TD; -2 fumble lost. Presets differ only in the reception value.
const BASE_VALUES = {
  passing_yards: 0.04,
  passing_tds: 4,
  passing_interceptions: -2,
  rushing_yards: 0.1,
  rushing_tds: 6,
  receiving_yards: 0.1,
  receiving_tds: 6,
  fumbles_lost_total: -2,
};

export const SCORING_PRESETS = {
  standard: { label: "Standard", values: { ...BASE_VALUES, receptions: 0 } },
  half_ppr: { label: "Half PPR", values: { ...BASE_VALUES, receptions: 0.5 } },
  full_ppr: { label: "Full PPR", values: { ...BASE_VALUES, receptions: 1 } },
};

export const DEFAULT_SCORING_SETTINGS = {
  preset: "full_ppr",
  values: { ...SCORING_PRESETS.full_ppr.values },
};

/** Pure function: raw stat line -> fantasy points under the given settings. */
export function computeFantasyPoints(statLine, scoringValues) {
  if (!statLine) return 0;
  let total = 0;
  for (const field of STAT_FIELDS) {
    const amount = statLine[field.key];
    const value = scoringValues[field.key];
    if (amount && value) total += amount * value;
  }
  return total;
}

/** Detects whether a values object matches one of the named presets
 * exactly, or should be labeled "Custom" in the settings UI. */
export function matchingPresetKey(values) {
  for (const [key, preset] of Object.entries(SCORING_PRESETS)) {
    const matches = STAT_FIELDS.every(
      (f) => (values[f.key] ?? 0) === (preset.values[f.key] ?? 0)
    );
    if (matches) return key;
  }
  return "custom";
}
