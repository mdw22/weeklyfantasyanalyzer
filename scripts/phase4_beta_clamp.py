"""
phase4_beta_clamp.py

Impact of clamping the rolling shrink beta to [0, 1] (designer guardrail,
2026-10-05): a negative beta inverts a position's ranking, which is never a
real effect. OFFLINE only; production-like setup (2023 history-only).

Recomputes D-rolling per scored player-week from build_state's pair log,
unclamped and clamped. Self-check: the unclamped recompute must reproduce the
logged (shipped) D exactly. Reports how many position-weeks the clamp
touched and the MAE/RMSE/bias change by position.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_beta_clamp.py
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_core  # noqa: E402
from phase4_combined_stack import build_state  # noqa: E402


def ols(pairs):
    n = len(pairs)
    mx = sum(c for c, _ in pairs) / n
    my = sum(a for _, a in pairs) / n
    vx = sum((c - mx) ** 2 for c, _ in pairs)
    return (sum((c - mx) * (a - my) for c, a in pairs) / vx if vx else 1.0), mx, my


def stats(errors):
    n = len(errors)
    return (sum(abs(e) for e in errors) / n, math.sqrt(sum(e * e for e in errors) / n), sum(errors) / n)


def main() -> None:
    _, _, _, log = build_state(extra_history_seasons=1, return_pair_log=True)
    by_week = defaultdict(list)
    for s, w, scored, pos, c, a, d in log:
        by_week[(s, w)].append((scored, pos, c, a, d))

    rolling = []
    max_selfcheck = 0.0
    err_raw, err_clamp = defaultdict(list), defaultdict(list)
    touched = []
    for wk in sorted(by_week):
        rows = by_week[wk]
        window = defaultdict(list)
        for past in rolling[-model_core.ROLLING_WEEKS:]:
            for pos, pairs in past.items():
                window[pos].extend(pairs)
        fits = {p: ols(v) for p, v in window.items() if len(v) >= model_core.MIN_SAMPLE}
        scored = rows[0][0]
        if scored:
            for pos, (b, _, _) in fits.items():
                if not 0.0 <= b <= 1.0:
                    touched.append((wk, pos, b))
        this_week = defaultdict(list)
        for _, pos, c, a, d in rows:
            if pos in fits:
                b, mc, ma = fits[pos]
                raw = ma + b * (c - mc)
                clamped = ma + min(max(b, 0.0), 1.0) * (c - mc)
            else:
                raw = clamped = c
            max_selfcheck = max(max_selfcheck, abs(raw - d))
            if scored:
                err_raw[pos].append(raw - a)
                err_clamp[pos].append(clamped - a)
            this_week[pos].append((c, a))
        rolling.append(this_week)

    print(f"Self-check: unclamped recompute vs shipped D, max |diff| = {max_selfcheck:.2e}")
    n_pos_weeks = len({wk for wk, _, _ in touched}) and sum(1 for _ in touched)
    print(f"Clamp touched {len(touched)} scored position-weeks:")
    for (s, w), pos, b in touched:
        print(f"  {s} wk {w:>2} {pos:3} beta {b:+.3f} -> {min(max(b, 0.0), 1.0):.0f}")
    print("\n| Position | n | Unclamped MAE / RMSE / bias | Clamped MAE / RMSE / bias |\n|---|---|---|---|")
    all_r, all_c = [], []
    for pos in model_core.SHRINK_POSITIONS:
        r, c = stats(err_raw[pos]), stats(err_clamp[pos])
        all_r += err_raw[pos]
        all_c += err_clamp[pos]
        print(f"| {pos} | {len(err_raw[pos])} | {r[0]:.4f} / {r[1]:.4f} / {r[2]:+.4f} | "
              f"{c[0]:.4f} / {c[1]:.4f} / {c[2]:+.4f} |")
    r, c = stats(all_r), stats(all_c)
    print(f"| Overall | {len(all_r)} | {r[0]:.4f} / {r[1]:.4f} / {r[2]:+.4f} | {c[0]:.4f} / {c[1]:.4f} / {c[2]:+.4f} |")


if __name__ == "__main__":
    main()
