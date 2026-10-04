"""
v3_parity_frontend.py

PRODUCTION_MODEL_SPEC.md section 4, items 3 and 4, on this week's real
pipeline output (public/data/week_NN: v3 projections, history, model_meta).

Item 3 -- JS parity: 200 matchups (100 full PPR, 100 standard, so the
custom-scoring path is exercised), win probability from the shipped
src/lib/monteCarlo.js (run in Node) vs. an independent Python port of spec
section 2.4, 10k trials each. Pass: |delta| < 0.02 for every matchup.

Item 4 -- pos_var parity: client-side computePositionVariance (default
scoring, current week, active players with n >= 2) vs. the backtest's latest
pos_var (phase4_combined_stack.build_state, 2023 history-only config).
Pass: within +/-15% per position; otherwise ship pos_var in model_meta.

Requires node on PATH. Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/v3_parity_frontend.py
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from calibration_check import draw_synthetic_matchup  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "public" / "data"
N_PER_SCORING = 100
TRIALS = 10000
SEED = 20261004
TOP_K = {"QB": 20, "RB": 50, "WR": 60, "TE": 20, "K": 14, "DEF": 14}
SCORINGS = {
    "full_ppr": dict(model_core.FULL_PPR_VALUES),
    "standard": {**model_core.FULL_PPR_VALUES, "receptions": 0},
}


def pts(line, values):
    return model_core.compute_points(line or {}, values)


def pos_var_client(projections, history, values):
    acc = {}
    for eid, e in projections.items():
        games = history.get(eid) or []
        if len(games) < 2 or (not e.get("active") and e["position"] != "DEF"):
            continue
        acc.setdefault(e["position"], []).append(float(np.var([pts(g, values) for g in games])))
    return {p: sum(v) / len(v) for p, v in acc.items()}


def ratio_pool(pools, min_pool, b):
    if not pools:
        return None
    own = pools.get(str(b))
    if own and len(own) >= min_pool:
        return own
    merged = [r for k in (b - 1, b, b + 1) for r in pools.get(str(k), [])]
    if len(merged) >= min_pool:
        return merged
    everything = [r for v in pools.values() for r in v]
    return everything if len(everything) >= min_pool else None


def outcomes(entry, games, values, pos_var, meta):
    """Independent Python port of spec section 2.4."""
    t = pts(entry["projected_stats"], values)
    p = np.array([pts(g, values) for g in (games or [])])
    n = len(p)
    s2 = float(p.var()) if n else 0.0
    pv = pos_var.get(entry["position"])
    w = n / (n + meta["pooling_k"])
    var = s2 if pv is None else w * s2 + (1 - w) * pv
    sd = (var * (n + 1) / (n - 1)) ** 0.5 if n >= 2 else var ** 0.5
    if s2 > 0 and n >= 2:
        return t + (p - p.mean()) * (sd / s2 ** 0.5)
    t_default = pts(entry["projected_stats"], model_core.FULL_PPR_VALUES)
    b = sum(t_default >= e for e in meta["ratio_bin_edges"])
    pool = ratio_pool(meta["ratio_pools"].get(entry["position"]), meta["ratio_min_pool"], b)
    if not pool or np.mean(pool) == 0:
        return np.array([t])
    r = np.array(pool)
    return t * r / r.mean()


def main() -> None:
    latest = json.loads((DATA / "latest.json").read_text())
    wk = DATA / f"week_{latest['week']:02d}"
    projections = json.loads((wk / "projections.json").read_text())
    history = json.loads((wk / "history.json").read_text())
    meta = json.loads((wk / "model_meta.json").read_text())
    assert meta["model_version"] == "v3", "pipeline output isn't v3"

    rng_py, rng_np = random.Random(SEED), np.random.default_rng(SEED)
    pool = {}
    for eid, e in projections.items():
        if e["position"] in TOP_K and (e.get("active") or e["position"] == "DEF") and not e.get("on_bye"):
            pool.setdefault(e["position"], []).append(eid)
    pool = {p: sorted(ids, key=lambda i: pts(projections[i]["projected_stats"], model_core.FULL_PPR_VALUES),
                      reverse=True)[:TOP_K[p]] for p, ids in pool.items()}

    matchups = []
    for scoring in SCORINGS:
        for _ in range(N_PER_SCORING):
            a, b = draw_synthetic_matchup(pool, rng_py)
            matchups.append({"scoring": scoring, "a": a, "b": b})

    py_probs, py_posvar = [], {}
    for scoring, values in SCORINGS.items():
        py_posvar[scoring] = pos_var_client(projections, history, values)
    for m in matchups:
        values, pv = SCORINGS[m["scoring"]], py_posvar[m["scoring"]]

        def total(ids):
            t = np.zeros(TRIALS)
            for eid in ids:
                t += rng_np.choice(outcomes(projections[eid], history.get(eid), values, pv, meta), size=TRIALS)
            return t
        py_probs.append(float(np.mean(total(m["a"]) > total(m["b"]))))

    js = run_js(wk, matchups, TRIALS)

    deltas = np.abs(np.array(js["probs"]) - np.array(py_probs))
    over = [i for i, d in enumerate(deltas) if d >= 0.02]
    print(f"Item 3 -- JS parity: {len(deltas)} matchups at {TRIALS} trials, max |delta win prob| = {deltas.max():.4f}, "
          f"mean = {deltas.mean():.4f}, signed mean = {np.mean(np.array(js['probs']) - np.array(py_probs)):+.4f}, "
          f">= 0.02: {len(over)} -> {'PASS' if not over else 'FAIL at the spec threshold'}")
    print(f"  Pure Monte Carlo noise alone: two independent {TRIALS}-trial estimates differ with SD ~0.007 at p=0.5, "
          f"so ~1 of 200 is expected past 0.02 by chance.")
    spread = np.array(py_probs)
    print(f"  (matchup win probs span {spread.min():.2f}-{spread.max():.2f})")
    if over:
        big = 200000
        recheck = [matchups[i] for i in over]
        js_big = run_js(wk, recheck, big)["probs"]
        for i, jp in zip(over, js_big):
            m = matchups[i]
            values, pv = SCORINGS[m["scoring"]], py_posvar[m["scoring"]]
            rng_big = np.random.default_rng(SEED + i)
            tot = lambda ids: sum(rng_big.choice(outcomes(projections[e], history.get(e), values, pv, meta), size=big)
                                  for e in ids)
            pp = float(np.mean(tot(m["a"]) > tot(m["b"])))
            print(f"  re-check matchup {i} at {big} trials: JS {jp:.4f} vs Python {pp:.4f} "
                  f"(|delta| {abs(jp - pp):.4f}; was {deltas[i]:.4f} at {TRIALS})")

    js_vs_py = max(abs(js["posVar"]["full_ppr"][p] - py_posvar["full_ppr"][p]) / py_posvar["full_ppr"][p]
                   for p in model_core.SHRINK_POSITIONS)
    print(f"  JS vs Python client pos_var (the 6 model positions), max relative diff: {js_vs_py:.2e}")

    from phase4_combined_stack import build_state  # heavy; only needed for item 4
    _, ws, _ = build_state(extra_history_seasons=1)
    backtest_pv = ws[max(ws)]["pos_var"]
    print(f"\nItem 4 -- pos_var parity (client, week {latest['week']} full PPR vs. backtest latest {max(ws)}):")
    ok_all = True
    for p in model_core.SHRINK_POSITIONS:
        c, b = js["posVar"]["full_ppr"].get(p), backtest_pv.get(p)
        rel = (c - b) / b if c is not None and b else float("nan")
        ok = abs(rel) <= 0.15
        ok_all &= ok
        print(f"  {p:3} client {c:7.2f}  backtest {b:7.2f}  diff {rel:+.1%}  {'ok' if ok else 'OUTSIDE ±15%'}")
    print(f"  -> {'PASS' if ok_all else 'FAIL: ship pos_var in model_meta.json instead'}")


def run_js(wk, matchups, trials):
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture.json"
        fixture.write_text(json.dumps({"matchups": matchups, "scorings": SCORINGS, "trials": trials}))
        runner = Path(tmp) / "run.mjs"
        runner.write_text(f"""
import {{ readFileSync }} from "node:fs";
import {{ simulateMatchup, computePositionVariance }} from "{(ROOT / 'src/lib/monteCarlo.js').as_uri()}";
const wk = "{wk.as_posix()}";
const projections = JSON.parse(readFileSync(wk + "/projections.json"));
const history = JSON.parse(readFileSync(wk + "/history.json"));
const modelMeta = JSON.parse(readFileSync(wk + "/model_meta.json"));
const fx = JSON.parse(readFileSync("{fixture.as_posix()}"));
const posVar = {{}};
for (const [k, values] of Object.entries(fx.scorings)) posVar[k] = computePositionVariance(projections, history, values);
const probs = fx.matchups.map((m) => simulateMatchup({{
  myPlayerIds: m.a, opponentPlayerIds: m.b, projections, history,
  scoringValues: fx.scorings[m.scoring], model: "v3", modelMeta, posVar: posVar[m.scoring], trials: fx.trials,
}}).winProbability);
console.log(JSON.stringify({{ probs, posVar }}));
""")
        out = subprocess.run(["node", str(runner)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


if __name__ == "__main__":
    main()
