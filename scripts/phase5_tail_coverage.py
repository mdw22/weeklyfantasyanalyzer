"""
phase5_tail_coverage.py

Thin-tails check (designer, 2026-10-05). OFFLINE only. Shipped v3 resamples a
player's own (rescaled) games, so no draw can exceed his best resampled game:
"needs 26.3 from a TE whose 8-game max is lower" is impossible, not unlikely.

Per-player tail coverage on the full backtest (shipped configuration: 2023
history-only, n-aware stage on its expanding window): for every scored
player-week, the model's 90th / 95th / 99th percentile from DRAWS draws, and
how often the real score exceeds it (nominal 10% / 5% / 1%), by position,
with Wilson 95% CIs. Variants (phase4_ship_check.tail_draws; both keep each
player's mean and variance exactly):
- shipped v3 (no tail);
- smooth:C -- smoothed bootstrap, Gaussian jitter of C x the player's SD;
- mix:PI   -- with probability PI, a draw from the position's pooled
              standardized residuals rescaled to the player.
Production (Baseline A + raw bootstrap) is shown for reference. Rows are split
own-history vs. fallback players (the candidates only change the former).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase5_tail_coverage.py [mix:0.5 smooth:0.7 ...]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import compute_points  # noqa: E402
import model_core  # noqa: E402
from phase4_combined_stack import build_state  # noqa: E402
from phase4_ship_check import POSITIONS, tail_draws, v3_info, wilson_ci  # noqa: E402
from phase4_tail_attribution import production_distribution  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE5_TAIL_COVERAGE_REPORT.md"  # gitignored
DRAWS = 4000
SEED = 20261004
LEVELS = (0.90, 0.95, 0.99)
LOW_LEVELS = (0.10, 0.05, 0.01)  # lower tail: real score BELOW the model's percentile
VARIANTS = [("shipped v3", None), ("smooth:0.3", ("smooth", 0.3)), ("smooth:0.5", ("smooth", 0.5)),
            ("smooth:0.7", ("smooth", 0.7)), ("mix:0.1", ("mix", 0.1)), ("mix:0.2", ("mix", 0.2)),
            ("mix:0.3", ("mix", 0.3))]


def main() -> None:
    global OUT_PATH, VARIANTS
    if len(sys.argv) > 1:  # e.g. mix:0.5 mix:1.0 -- a follow-up sweep, written to its own report
        # "mixq:PI" = mix:PI with the shape quantized exactly as production ships it (model_core.residual_shape)
        VARIANTS = [("shipped v3", None)] + [(a, (a.split(":")[0], float(a.split(":")[1]))) for a in sys.argv[1:]]
        OUT_PATH = OUT_PATH.with_name("PHASE5_TAIL_COVERAGE_SWEEP2.md")
    targets, weekly_state, _ = build_state(extra_history_seasons=1, bucket_stage="expanding")
    names = ["production"] + [n for n, _ in VARIANTS]
    # (variant, position, own?) -> [count, exceed@90, @95, @99, below-zero draw share sum, mean shift sum,
    #                               below@10, @5, @1, real score < 0]
    acc = {}
    for wk in targets:
        st = weekly_state[wk]
        rng = np.random.default_rng(SEED + wk[0] * 100 + wk[1])
        info_cache, prod_cache, stats = {}, {}, {"neg": {}, "point_mass": 0}
        quantized = {p: np.array(model_core.residual_shape(list(v))) for p, v in st["resid_shape"].items()}
        for eid, row in st["actuals"].items():
            proj = st["projections"].get(eid)
            if proj is None or proj["position"] not in POSITIONS:
                continue
            pos = proj["position"]
            actual = compute_points(row)
            info = v3_info(eid, st["baseline_d"][eid], st["history"], st["pos_var"], st["ratio_bins"], pos,
                           info_cache, stats)
            samples = {"production": rng.choice(production_distribution(eid, st["history"], st["baseline_a"],
                                                                        prod_cache), size=DRAWS, replace=True)}
            for name, tail in VARIANTS:
                if tail and tail[0] == "mixq":
                    samples[name] = tail_draws(info, rng, DRAWS, ("mix", tail[1]), quantized.get(pos))
                else:
                    samples[name] = tail_draws(info, rng, DRAWS, tail, st["resid_shape"].get(pos))
            for name in names:
                x = samples[name]
                qs = np.quantile(x, LEVELS)
                a = acc.setdefault((name, pos, info[3]), [0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0])
                a[0] += 1
                for i, q in enumerate(qs):
                    a[1 + i] += actual > q
                for i, q in enumerate(np.quantile(x, LOW_LEVELS)):
                    a[6 + i] += actual < q
                a[9] += actual < 0
                a[4] += float(np.mean(x < 0))
                a[5] += float(x.mean() - info[1]) if name != "production" else 0.0
        print(f"  done {wk}", end="\r")
    print()

    def cell(k, n, nominal):
        lo, hi = wilson_ci(k, n)
        rate = nominal if nominal < 0.5 else 1 - nominal
        flag = "" if lo <= rate <= hi else (" ↑" if k / n > rate else " ↓")
        return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]{flag}"

    sections = []
    for own, label in ((True, "Own-history players (the candidates change these)"),
                       (False, "Fallback / point-mass players (unchanged by the candidates)")):
        lines = [f"### {label}", "",
                 "| Variant | Position | n | > p90 (nominal 10%) | > p95 (5%) | > p99 (1%) | Draws < 0 | Mean shift |",
                 "|---|---|---|---|---|---|---|---|"]
        low = ["", f"Lower tail, {label.split(' (')[0].lower()}:", "",
               "| Variant | Position | n | < p10 (nominal 10%) | < p5 (5%) | < p1 (1%) | Draws < 0 | Real < 0 |",
               "|---|---|---|---|---|---|---|---|"]
        for name in names:
            tot = [0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0]
            for pos in POSITIONS:
                a = acc.get((name, pos, own))
                if not a:
                    continue
                tot = [x + y for x, y in zip(tot, a)]
                lines.append(f"| {name} | {pos} | {a[0]} | {cell(a[1], a[0], .90)} | {cell(a[2], a[0], .95)} | "
                             f"{cell(a[3], a[0], .99)} | {a[4] / a[0]:.1%} | {a[5] / a[0]:+.3f} |")
                low.append(f"| {name} | {pos} | {a[0]} | {cell(a[6], a[0], .10)} | {cell(a[7], a[0], .05)} | "
                           f"{cell(a[8], a[0], .01)} | {a[4] / a[0]:.1%} | {a[9] / a[0]:.1%} |")
            if tot[0]:
                lines.append(f"| **{name}** | **all** | {tot[0]} | {cell(tot[1], tot[0], .90)} | "
                             f"{cell(tot[2], tot[0], .95)} | {cell(tot[3], tot[0], .99)} | {tot[4] / tot[0]:.1%} | "
                             f"{tot[5] / tot[0]:+.3f} |")
                low.append(f"| **{name}** | **all** | {tot[0]} | {cell(tot[6], tot[0], .10)} | "
                           f"{cell(tot[7], tot[0], .05)} | {cell(tot[8], tot[0], .01)} | {tot[4] / tot[0]:.1%} | "
                           f"{tot[9] / tot[0]:.1%} |")
        sections.append("\n".join(lines + low))

    body = "\n\n".join(sections)
    print(body)
    OUT_PATH.write_text(f"""# Phase 5 Tail Coverage

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE only. Shipped configuration (2023
history-only, n-aware expanding stage); scored weeks {targets[0]} .. {targets[-1]}. Each cell: share of player-weeks
whose real score exceeded the model's percentile ({DRAWS} draws), Wilson 95% CI; ↑/↓ = nominal outside the CI
(↑ = exceeded too often, i.e. the tail is too thin). Mean shift = draw mean minus projection (should be ~0).

{body}
""")
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
