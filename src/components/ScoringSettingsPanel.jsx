import { useApp } from "../lib/AppContext.jsx";
import { SCORING_PRESETS, STAT_FIELDS, matchingPresetKey } from "../lib/scoring.js";
import { CloseIcon } from "./icons.jsx";

const PRESET_ORDER = ["standard", "half_ppr", "full_ppr"];
const SHRINK_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"];

function ModelNote({ meta }) {
  if (!meta) return <div className="stat-note">v3 isn't available this week; running v2.</div>;
  const [first, last] = [meta.window_weeks[0], meta.window_weeks[meta.window_weeks.length - 1]];
  const betas = SHRINK_POSITIONS.filter((p) => meta.params[p])
    .map((p) => `${p} ${meta.params[p].beta.toFixed(2)}`)
    .join(" · ");
  return (
    <div className="stat-note">
      v3 fit window: {first[0]} wk {first[1]} – {last[0]} wk {last[1]}. Shrink toward position average (β): {betas}.
    </div>
  );
}

export function ScoringSettingsPanel({ onClose }) {
  const { scoringSettings, setScoringSettings, weekData, effectiveModel, setProjectionModel } = useApp();
  const modelMeta = weekData.status === "ready" ? weekData.modelMeta : null;
  const activeKey = matchingPresetKey(scoringSettings.values);

  function applyPreset(key) {
    setScoringSettings({ preset: key, values: { ...SCORING_PRESETS[key].values } });
  }

  function updateStat(key, rawValue) {
    const parsed = parseFloat(rawValue);
    const nextValues = {
      ...scoringSettings.values,
      [key]: Number.isNaN(parsed) ? 0 : parsed,
    };
    setScoringSettings({ preset: "custom", values: nextValues });
  }

  const groups = [];
  for (const field of STAT_FIELDS.filter((f) => !f.fixed)) {
    let group = groups.find((g) => g.name === field.group);
    if (!group) {
      group = { name: field.group, fields: [] };
      groups.push(group);
    }
    group.fields.push(field);
  }

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer">
        <div className="drawer__header">
          <div className="drawer__title">Scoring Settings</div>
          <button className="icon-btn" aria-label="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        <div className="drawer__body">
          <div className="stat-group-title">Projection model</div>
          <div className="segmented">
            <button
              className={`segmented__option${effectiveModel === "v3" ? " active" : ""}`}
              disabled={!modelMeta}
              onClick={() => setProjectionModel("v3")}
            >
              v3
            </button>
            <button
              className={`segmented__option${effectiveModel === "v2" ? " active" : ""}`}
              onClick={() => setProjectionModel("v2")}
            >
              v2 (legacy)
            </button>
          </div>
          <ModelNote meta={modelMeta} />

          <div className="stat-group-title">Scoring</div>
          <div className="segmented">
            {PRESET_ORDER.map((key) => (
              <button
                key={key}
                className={`segmented__option${activeKey === key ? " active" : ""}`}
                onClick={() => applyPreset(key)}
              >
                {SCORING_PRESETS[key].label}
              </button>
            ))}
            <button className={`segmented__option${activeKey === "custom" ? " active" : ""}`} disabled>
              Custom
            </button>
          </div>

          {groups.map((group) => (
            <div key={group.name}>
              <div className="stat-group-title">{group.name}</div>
              {group.fields.map((field) => (
                <div className="stat-row" key={field.key}>
                  <span className="stat-row__label">{field.label}</span>
                  <input
                    className="stat-input"
                    type="number"
                    step="0.01"
                    value={scoringSettings.values[field.key] ?? 0}
                    onChange={(e) => updateStat(field.key, e.target.value)}
                  />
                </div>
              ))}
              {group.name === "Defense" && (
                <div className="stat-note">
                  Points/yards allowed bonus: this league's configured tiers, not customizable.
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
