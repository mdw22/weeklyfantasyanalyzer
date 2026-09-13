import { useApp } from "../lib/AppContext.jsx";
import { SCORING_PRESETS, STAT_FIELDS, matchingPresetKey } from "../lib/scoring.js";
import { CloseIcon } from "./icons.jsx";

const PRESET_ORDER = ["standard", "half_ppr", "full_ppr"];

export function ScoringSettingsPanel({ onClose }) {
  const { scoringSettings, setScoringSettings } = useApp();
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
  for (const field of STAT_FIELDS) {
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
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
