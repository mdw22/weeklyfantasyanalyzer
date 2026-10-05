"""
phase4_naware_keenum.py

Current-week sanity check for the n-aware stage (designer request): Case
Keenum's projection and QB rank under v2 / v3 / n-aware (expanding) / n-aware
(16-week). Each variant's stage is fit through the latest completed week
(strictly before the current week), on (stage-1 D, actual) pairs, and applied
to every active QB's live v3 projection before ranking. Points-level under
default scoring, which equals the stat-space stage there (points are linear
in stats).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_naware_keenum.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from phase4_combined_stack import build_state  # noqa: E402

ROOT = Path(__file__).parent.parent


def fit(pairs):
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    b = sum((x - mx) * (y - my) for x, y in pairs) / vx if vx else 1.0
    return min(max(b, 0.0), 1.0), mx, my


def main() -> None:
    _, _, _, log = build_state(extra_history_seasons=1, return_pair_log=True)
    weeks = sorted({(r[0], r[1]) for r in log})
    latest = json.loads((ROOT / "public/data/latest.json").read_text())
    wk = ROOT / f"public/data/week_{latest['week']:02d}"
    proj = json.loads((wk / "projections.json").read_text())

    qbs = {eid: e for eid, e in proj.items() if e["position"] == "QB" and e.get("active")}
    results = {"v2": {e: model_core.compute_points(q["projected_stats_v2"]) for e, q in qbs.items()},
               "v3": {e: model_core.compute_points(q["projected_stats"]) for e, q in qbs.items()}}
    for name, window in [("n-aware expanding", weeks), ("n-aware 16-week", weeks[-16:])]:
        keep = set(window)
        cells = {}
        for s, w, scored, pos, c, a, d, n, d1 in log:
            b = model_core.history_bucket(n)
            if (s, w) in keep and pos == "QB" and b in model_core.BUCKET_STAGE_BUCKETS:
                cells.setdefault(b, []).append((d1, a))
        params = {b: fit(p) for b, p in cells.items() if len(p) >= model_core.MIN_SAMPLE}
        out = {}
        for eid, q in qbs.items():
            d = results["v3"][eid]
            p = params.get(model_core.history_bucket(q["games_used"]))
            out[eid] = d if p is None else p[2] + p[0] * (d - p[1])
        results[name] = out
        print(f"{name}: QB cells fit = { {b: (round(p[0], 3), len(cells[b])) for b, p in params.items()} }")

    for eid, q in qbs.items():
        if q["player_name"] != "Case Keenum":
            continue
        parts = []
        for name, vals in results.items():
            rank = sorted(vals.values(), reverse=True).index(vals[eid]) + 1
            parts.append(f"{name} {vals[eid]:.1f} (#{rank})")
        print(f"Case Keenum ({q['team']}, {q['games_used']} game of history): " + ", ".join(parts))


if __name__ == "__main__":
    main()
