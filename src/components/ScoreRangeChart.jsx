const WIDTH = 600;
const ROW_HEIGHT = 44;
const CHART_HEIGHT = ROW_HEIGHT * 2 + 30;
const LEFT_PAD = 8;
const RIGHT_PAD = 8;

function niceTicks(min, max, count = 5) {
  const span = max - min;
  const rawStep = span / (count - 1);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const step = Math.ceil(rawStep / magnitude) * magnitude;
  const ticks = [];
  let t = Math.ceil(min / step) * step;
  while (t <= max) {
    ticks.push(Math.round(t));
    t += step;
  }
  return ticks;
}

function Row({ label, stats, color, wash, y, scale }) {
  const bandY = y + 8;
  const bandHeight = 14;
  return (
    <g>
      <text x={LEFT_PAD} y={y - 6} className="score-range__row-label-text" fill="var(--text-secondary)" fontSize="12">
        {label}
      </text>
      {/* 10th-90th percentile: light wash */}
      <rect
        x={scale(stats.p10)}
        y={bandY}
        width={Math.max(1, scale(stats.p90) - scale(stats.p10))}
        height={bandHeight}
        rx={bandHeight / 2}
        fill={wash}
      />
      {/* 25th-75th percentile: solid band */}
      <rect
        x={scale(stats.p25)}
        y={bandY}
        width={Math.max(1, scale(stats.p75) - scale(stats.p25))}
        height={bandHeight}
        rx={bandHeight / 2}
        fill={color}
        opacity="0.55"
      />
      {/* mean marker */}
      <line x1={scale(stats.mean)} x2={scale(stats.mean)} y1={bandY - 4} y2={bandY + bandHeight + 4} stroke={color} strokeWidth="2" />
      <text x={scale(stats.mean)} y={bandY - 8} textAnchor="middle" className="score-range__mean-label">
        {stats.mean.toFixed(0)}
      </text>
    </g>
  );
}

/** "Likely Score Range" chart: two rows sharing one axis, each showing a
 * light 10th-90th percentile band, a solid 25th-75th band, and a mean
 * marker. Domain is computed from the actual simulated range, not
 * hardcoded, with a little padding on each side. */
export function ScoreRangeChart({ my, opponent }) {
  const domainMin = Math.min(my.p10, opponent.p10);
  const domainMax = Math.max(my.p90, opponent.p90);
  const span = Math.max(1, domainMax - domainMin);
  const padding = span * 0.12;
  const lo = domainMin - padding;
  const hi = domainMax + padding;

  const usableWidth = WIDTH - LEFT_PAD - RIGHT_PAD;
  const scale = (v) => LEFT_PAD + ((v - lo) / (hi - lo)) * usableWidth;

  const ticks = niceTicks(lo, hi);
  const axisY = ROW_HEIGHT * 2 + 6;

  return (
    <div className="card score-range">
      <div className="section-title" style={{ marginBottom: 4 }}>Likely Score Range</div>
      <svg viewBox={`0 0 ${WIDTH} ${CHART_HEIGHT}`} width="100%" height={CHART_HEIGHT} className="score-range__rows">
        <Row label="My Team" stats={my} color="var(--blue)" wash="var(--blue-wash)" y={26} scale={scale} />
        <Row label="Opponent" stats={opponent} color="var(--orange)" wash="var(--orange-wash)" y={26 + ROW_HEIGHT} scale={scale} />
        <line x1={LEFT_PAD} x2={WIDTH - RIGHT_PAD} y1={axisY} y2={axisY} className="score-range__gridline" />
        <g className="score-range__axis">
          {ticks.map((t) => (
            <text key={t} x={scale(t)} y={axisY + 16} textAnchor="middle">
              {t}
            </text>
          ))}
        </g>
      </svg>
    </div>
  );
}
