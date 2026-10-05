"""
phase4_kicker_hybrid.py

Kicker hybrid-beta ablation (designer request, 2026-10-05). OFFLINE only.

Under v3 every kicker projects ~8.3-8.4: beta_K on the latest rolling
8-week window is 0.01, which erases all kicker ranking. Designer's
hypothesis: an 8-week window is a noisy place to fit a slope for a
low-signal position. Hybrid candidate:
  beta  -- OLS on the EXPANDING window (all strictly-prior weeks; stable)
  means -- mean actual / mean C on the ROLLING 8 weeks (tracks drift)
  D_hybrid = m_act_rolling + beta_expanding * (C - m_c_rolling)
Same fallback as D-rolling: fewer than MIN_SAMPLE pairs in either window
for a position -> D = C.

Production-like setup (2023 loaded as history only, so the expanding
window starts with a full prior season). Reports, scored weeks only:
- per-position MAE/RMSE/bias, D-rolling (shipped v3) vs. D-hybrid;
- beta trajectory per position (rolling vs. expanding), by season;
- kicker projection spread: mean within-week SD of projected points
  across kickers, D-rolling vs. hybrid vs. C (= today's v2 line for K).

Points-level formula; under default scoring it equals the stat-space
version production would use (points are linear in stats).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_kicker_hybrid.py
"""

import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from phase4_combined_stack import build_state  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_KICKER_HYBRID_REPORT.md"  # gitignored
POSITIONS = model_core.SHRINK_POSITIONS


def ols(pairs):
    n = len(pairs)
    mx = sum(c for c, _ in pairs) / n
    my = sum(a for _, a in pairs) / n
    vx = sum((c - mx) ** 2 for c, _ in pairs)
    beta = sum((c - mx) * (a - my) for c, a in pairs) / vx if vx else 1.0
    return beta, mx, my


def stats(errors):
    n = len(errors)
    return {"n": n, "mae": sum(abs(e) for e in errors) / n,
            "rmse": math.sqrt(sum(e * e for e in errors) / n), "bias": sum(errors) / n}


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main() -> None:
    _, _, _, log = build_state(extra_history_seasons=1, return_pair_log=True)
    weeks = sorted({(s, w) for s, w, *_ in log})
    by_week = defaultdict(list)
    for s, w, scored, pos, c, a, d, *_ in log:
        by_week[(s, w)].append((scored, pos, c, a, d))

    expanding = defaultdict(list)
    rolling_weeks = []  # list of per-week {pos: [(c, a)]}
    err_roll, err_hyb = defaultdict(list), defaultdict(list)
    beta_traj = defaultdict(list)  # pos -> (season, beta_rolling, beta_expanding)
    k_spread = {"C (v2 line)": [], "D-rolling (shipped v3)": [], "D-hybrid": []}

    for wk in weeks:
        rows = by_week[wk]
        rolling = defaultdict(list)
        for wk_pairs in rolling_weeks[-model_core.ROLLING_WEEKS:]:
            for pos, pairs in wk_pairs.items():
                rolling[pos].extend(pairs)
        params = {}
        for pos in POSITIONS:
            if len(rolling[pos]) >= model_core.MIN_SAMPLE and len(expanding[pos]) >= model_core.MIN_SAMPLE:
                b_roll, m_c, m_a = ols(rolling[pos])
                b_exp, _, _ = ols(expanding[pos])
                params[pos] = (b_exp, m_c, m_a)
                if rows[0][0]:
                    beta_traj[pos].append((wk[0], b_roll, b_exp))

        scored = rows[0][0]
        k_proj = {"C (v2 line)": [], "D-rolling (shipped v3)": [], "D-hybrid": []}
        this_week = defaultdict(list)
        for _, pos, c, a, d in rows:
            p = params.get(pos)
            hyb = c if p is None else p[2] + p[0] * (c - p[1])
            if scored:
                err_roll[pos].append(d - a)
                err_hyb[pos].append(hyb - a)
                if pos == "K":
                    k_proj["C (v2 line)"].append(c)
                    k_proj["D-rolling (shipped v3)"].append(d)
                    k_proj["D-hybrid"].append(hyb)
            this_week[pos].append((c, a))
        if scored and len(k_proj["C (v2 line)"]) >= 2:
            for name, xs in k_proj.items():
                k_spread[name].append(sd(xs))
        for pos, pairs in this_week.items():
            expanding[pos].extend(pairs)
        rolling_weeks.append(this_week)

    lines = ["| Position | n | D-rolling MAE / RMSE / bias | D-hybrid MAE / RMSE / bias |", "|---|---|---|---|"]
    tot_r, tot_h = [], []
    for pos in POSITIONS:
        r, h = stats(err_roll[pos]), stats(err_hyb[pos])
        tot_r += err_roll[pos]
        tot_h += err_hyb[pos]
        lines.append(f"| {pos} | {r['n']} | {r['mae']:.3f} / {r['rmse']:.3f} / {r['bias']:+.3f} | "
                     f"{h['mae']:.3f} / {h['rmse']:.3f} / {h['bias']:+.3f} |")
    r, h = stats(tot_r), stats(tot_h)
    lines.append(f"| **Overall** | {r['n']} | {r['mae']:.3f} / {r['rmse']:.3f} / {r['bias']:+.3f} | "
                 f"{h['mae']:.3f} / {h['rmse']:.3f} / {h['bias']:+.3f} |")
    accuracy = "\n".join(lines)

    blines = ["| Position | Season | Rolling beta: mean (min-max) | Expanding beta: mean (min-max) |", "|---|---|---|---|"]
    for pos in POSITIONS:
        for season in sorted({s for s, _, _ in beta_traj[pos]}):
            br = [b for s, b, _ in beta_traj[pos] if s == season]
            be = [b for s, _, b in beta_traj[pos] if s == season]
            blines.append(f"| {pos} | {season} | {sum(br)/len(br):.2f} ({min(br):.2f}-{max(br):.2f}) | "
                          f"{sum(be)/len(be):.2f} ({min(be):.2f}-{max(be):.2f}) |")
    betas = "\n".join(blines)

    spread = "\n".join(["| Kicker projection | Mean within-week SD across kickers (points) |", "|---|---|"] +
                       [f"| {name} | {sum(v)/len(v):.2f} |" for name, v in k_spread.items()])

    print(accuracy, "\n\n", betas, "\n\n", spread, sep="")
    OUT_PATH.write_text(f"""# Kicker Hybrid-Beta Ablation

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE only. 2023 loaded as history only;
scored weeks 2024 wk1 onward. D-hybrid = rolling-8-week means + expanding-window beta.

## Accuracy by position (scored weeks)

{accuracy}

## Beta trajectory (scored weeks)

{betas}

## Kicker projection spread

{spread}
""")
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
