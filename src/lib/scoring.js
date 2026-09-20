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
  { key: "def_sacks", label: "Sack", group: "Defense" },
  { key: "def_interceptions", label: "Interception", group: "Defense" },
  { key: "def_fumble_recoveries", label: "Fumble Recovery", group: "Defense" },
  { key: "def_safeties", label: "Safety", group: "Defense" },
  { key: "def_touchdowns", label: "Defensive/ST TD", group: "Defense" },
  { key: "def_blocked_kicks", label: "Blocked Kick", group: "Defense" },
  { key: "def_two_point_returns", label: "2pt Return", group: "Defense" },
  // Already expressed in points (tier tables applied in the pipeline), so
  // they score at a fixed 1x and aren't editable -- `fixed` hides them from
  // the settings drawer.
  { key: "def_points_allowed_bonus", label: "Points Allowed Bonus", group: "Defense", fixed: true },
  { key: "def_yards_allowed_bonus", label: "Yards Allowed Bonus", group: "Defense", fixed: true },
  { key: "fg_made_0_39", label: "FG Made (0-39 yd)", group: "Kicking" },
  { key: "fg_made_40_49", label: "FG Made (40-49 yd)", group: "Kicking" },
  { key: "fg_made_50_plus", label: "FG Made (50+ yd)", group: "Kicking" },
  { key: "fg_missed_total", label: "FG Missed/Blocked", group: "Kicking" },
  { key: "pat_made", label: "Extra Point Made", group: "Kicking" },
];

// Full-PPR values match the prior draft tool's league config (see
// CLAUDE.md) so both tools agree: 0.04/yd passing (1pt/25yd), 4pt passing
// TD, -2 INT; 0.1/yd rushing+receiving (1pt/10yd), 6pt rushing+receiving
// TD; -2 fumble lost; sack 1, INT 2, fumble recovery 2, safety 2,
// defensive/ST TD 6, blocked kick 2, 2pt return 2. Presets differ only in
// the reception value -- defense scoring is identical across all three.
// Kicker values (also identical across presets): FG 3/4/5 pts by distance,
// -1 miss or block, +1 PAT, no penalty for a missed PAT.
// Points-allowed/yards-allowed tiers don't fit the linear amount*value
// model, so the pipeline pre-converts them to points per game and they
// score at a fixed 1x here (not customizable; the league's real tiers live in POINTS_ALLOWED_TIERS /
// YARDS_ALLOWED_TIERS in scripts/generate_projections.py).
const BASE_VALUES = {
  passing_yards: 0.04,
  passing_tds: 4,
  passing_interceptions: -2,
  rushing_yards: 0.1,
  rushing_tds: 6,
  receiving_yards: 0.1,
  receiving_tds: 6,
  fumbles_lost_total: -2,
  def_sacks: 1,
  def_interceptions: 2,
  def_fumble_recoveries: 2,
  def_safeties: 2,
  def_touchdowns: 6,
  def_blocked_kicks: 2,
  def_two_point_returns: 2,
  def_points_allowed_bonus: 1,
  def_yards_allowed_bonus: 1,
  fg_made_0_39: 3,
  fg_made_40_49: 4,
  fg_made_50_plus: 5,
  fg_missed_total: -1,
  pat_made: 1,
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
