"""
phase4_naware.py

n-aware mean shrinkage, point-level screen (designer request, 2026-10-05).
OFFLINE only; production-like setup (2023 history-only).

Premise, measured first: under shipped D-rolling the slope of actual on
projected is ~1.0 for players with 7-8 games of history but 0.51 for 1-3-game
QBs and 0.74-0.81 for thin-history RB/WR/TE -- thin-history projections still
spread too far from their center (bias is small; the spread is the problem).
Example: Case Keenum, one game of history, #3 QB under both v2 and v3.

History buckets by games_used: 1-3, 4-6, 7-8. Variants:
- D-rolling (shipped v3): one beta per position, rolling 8 weeks.
- V1, as specced: OLS of actual on C per position x bucket on the ROLLING
  8-week window, beta clamped to [0, 1]; a cell with < MIN_SAMPLE pairs falls
  back to the shipped position fit. Coverage is reported because thin cells
  rarely reach MIN_SAMPLE in 8 weeks.
- V2, expanding-window bucket correction: for the 1-3 and 4-6 buckets, a
  second stage on top of D-rolling, D2 = m_a + lambda * (D - m_d), fit per
  position x bucket on the EXPANDING window of (D, actual) pairs (lambda
  clamped to [0, 1]; < MIN_SAMPLE -> unchanged). Shrinks toward the bucket's
  own mean, not the position's. 7-8 is left as shipped (slope ~1.0 already).

Reports accuracy and calibration slope by position x bucket on scored weeks,
coverage, and Keenum's current-week projection under v2 / v3 / V1 / V2.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_naware.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from phase4_combined_stack import build_state  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "PHASE4_NAWARE_REPORT.md"  # gitignored
BUCKETS = ["1-3", "4-6", "7-8"]
POSITIONS = model_core.SHRINK_POSITIONS


def bucket(n: int) -> str:
    return "1-3" if n <= 3 else ("4-6" if n <= 6 else "7-8")


def ols_clamped(pairs):
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    b = sum((x - mx) * (y - my) for x, y in pairs) / vx if vx else 1.0
    return min(max(b, 0.0), 1.0), mx, my


def metrics(rows):
    """rows: (projected, actual). MAE, RMSE, bias, slope of actual on projected."""
    n = len(rows)
    errs = [p - a for p, a in rows]
    mp = sum(p for p, _ in rows) / n
    ma = sum(a for _, a in rows) / n
    vp = sum((p - mp) ** 2 for p, _ in rows)
    slope = sum((p - mp) * (a - ma) for p, a in rows) / vp if vp else float("nan")
    return (sum(abs(e) for e in errs) / n, math.sqrt(sum(e * e for e in errs) / n), sum(errs) / n, slope)


def main() -> None:
    _, _, _, log = build_state(extra_history_seasons=1, return_pair_log=True)
    by_week = defaultdict(list)
    for s, w, scored, pos, c, a, d, n, *_ in log:
        by_week[(s, w)].append((scored, pos, bucket(n), c, a, d))

    rolling = []                    # per week: list of (pos, bucket, c, a, d)
    expanding = defaultdict(list)   # (pos, bucket) -> [(d, a)]
    scored_rows = defaultdict(lambda: {"ship": [], "v1": [], "v2": []})  # (pos, bucket) -> [(proj, actual)]
    coverage = defaultdict(lambda: {"v1": 0, "v2": 0, "n": 0})

    def fits_for(week_rows_window, exp):
        cells = defaultdict(list)
        for wk_rows in week_rows_window:
            for pos, b, c, a, d in wk_rows:
                cells[(pos, b)].append((c, a))
        v1 = {k: ols_clamped(v) for k, v in cells.items() if len(v) >= model_core.MIN_SAMPLE}
        v2 = {k: ols_clamped(v) for k, v in exp.items() if k[1] != "7-8" and len(v) >= model_core.MIN_SAMPLE}
        return v1, v2

    for wk in sorted(by_week):
        v1_fit, v2_fit = fits_for(rolling[-model_core.ROLLING_WEEKS:], expanding)
        this_week = []
        for scored, pos, b, c, a, d in by_week[wk]:
            key = (pos, b)
            p1 = (v1_fit[key][2] + v1_fit[key][0] * (c - v1_fit[key][1])) if key in v1_fit else d
            p2 = (v2_fit[key][2] + v2_fit[key][0] * (d - v2_fit[key][1])) if key in v2_fit else d
            if scored:
                scored_rows[key]["ship"].append((d, a))
                scored_rows[key]["v1"].append((p1, a))
                scored_rows[key]["v2"].append((p2, a))
                coverage[key]["n"] += 1
                coverage[key]["v1"] += key in v1_fit
                coverage[key]["v2"] += key in v2_fit
            this_week.append((pos, b, c, a, d))
        for pos, b, c, a, d in this_week:
            expanding[(pos, b)].append((d, a))
        rolling.append(this_week)

    lines = ["| Pos | Bucket | n | Shipped MAE / RMSE / bias / slope | V1 (spec) | V1 coverage | V2 (expanding) | V2 coverage |",
             "|---|---|---|---|---|---|---|---|"]
    fmt = lambda m: f"{m[0]:.3f} / {m[1]:.3f} / {m[2]:+.3f} / {m[3]:.2f}"
    totals = {"ship": [], "v1": [], "v2": []}
    for pos in POSITIONS:
        for b in BUCKETS:
            r = scored_rows.get((pos, b))
            if not r or not r["ship"]:
                continue
            cov = coverage[(pos, b)]
            for k in totals:
                totals[k] += r[k]
            lines.append(f"| {pos} | {b} | {cov['n']} | {fmt(metrics(r['ship']))} | {fmt(metrics(r['v1']))} | "
                         f"{cov['v1'] / cov['n']:.0%} | {fmt(metrics(r['v2']))} | {cov['v2'] / cov['n']:.0%} |")
    lines.append(f"| **All** | | {len(totals['ship'])} | {fmt(metrics(totals['ship']))} | {fmt(metrics(totals['v1']))} | | "
                 f"{fmt(metrics(totals['v2']))} | |")
    thin = {k: [x for (pos, b), r in scored_rows.items() if b != "7-8" for x in r[k]] for k in totals}
    lines.append(f"| **Thin (1-6)** | | {len(thin['ship'])} | {fmt(metrics(thin['ship']))} | {fmt(metrics(thin['v1']))} | | "
                 f"{fmt(metrics(thin['v2']))} | |")
    table = "\n".join(lines)

    # Keenum, current week: fits through the latest completed week, applied to his live projection.
    latest = json.loads((ROOT / "public/data/latest.json").read_text())
    wkdir = ROOT / f"public/data/week_{latest['week']:02d}"
    proj = json.loads((wkdir / "projections.json").read_text())
    meta = json.loads((wkdir / "model_meta.json").read_text())
    v1_now, v2_now = fits_for(rolling[-model_core.ROLLING_WEEKS:], expanding)
    keenum_lines = []
    for eid, e in proj.items():
        if e["player_name"] != "Case Keenum":
            continue
        n = e["games_used"]
        key = (e["position"], bucket(n))
        v2pts = model_core.compute_points(e["projected_stats_v2"])
        d = model_core.compute_points(e["projected_stats"])
        qb = meta["params"]["QB"]
        c = qb["m_c"] + (d - qb["m_act"]) / qb["beta"]  # back out C from the live D
        p1 = v1_now[key][2] + v1_now[key][0] * (c - v1_now[key][1]) if key in v1_now else d
        p2 = v2_now[key][2] + v2_now[key][0] * (d - v2_now[key][1]) if key in v2_now else d
        qbs = sorted((model_core.compute_points(x["projected_stats"]) for x in proj.values()
                      if x["position"] == "QB" and x.get("active")), reverse=True)
        keenum_lines.append(f"Case Keenum ({e['team']}, {n} game(s) of history): v2 {v2pts:.1f}, v3 {d:.1f}, "
                            f"V1 {p1:.1f}{'' if key in v1_now else ' (fallback: cell too thin)'}, V2 {p2:.1f}"
                            f"{'' if key in v2_now else ' (fallback)'}. "
                            f"v3 rank among active QBs: {qbs.index(d) + 1 if d in qbs else '?'}; "
                            f"V2 would rank {sum(q > p2 for q in qbs) + 1}.")
    keenum = "\n".join(keenum_lines) or "Case Keenum not found in this week's projections."

    print(table, "\n\n", keenum, sep="")
    OUT_PATH.write_text(f"# n-aware Shrinkage Screen (point level)\n\n{table}\n\n{keenum}\n")
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
