"""
phase4_ship_check.py

PRODUCTION_MODEL_SPEC.md section 4, item 1 -- the final backtest the v3
ship decision rests on. OFFLINE only.

Setup (production-like history): 2023 is loaded as HISTORY ONLY. It feeds
every fit (rolling shrink window, pos_var, residual pools, player
histories) but is never scored, so every scored week (2024 wk1 onward) has
full history and no weeks are excluded.

Models, paired on identical roster draws (roster RNG separate from both
simulation RNGs), both seeds, realistic top-K and all-players pools:
- Production today: Baseline A center + raw bootstrap of the last 8 games.
- v3 stack (spec section 2): Baseline C -> D-rolling -> variance pooling
  (k=3) + predictive scale, with the spec section 2.4 fallback for s=0 or
  n<2 (revised 2026-10-04): multiplicative and mean-normalized,
  x = t * r_j / mean(r), r_j = actual_j / D_j from same-position window
  players in the same projection bin (bins <2, 2-5, 5-10, >=10 default
  points), D_j >= 0.5, up to 200 most recent. A bin with < MIN_BIN ratios
  merges with its adjacent bin(s), then the whole position pool; only if
  even that is thin does the player stay a point mass (counted).

Ship bar (spec section 4.1; multiplicity-corrected 2026-10-05): Var(z')
within 0.9-1.15 on every run; per-position bias within +/-0.2. A bucket
MISSES in one run if v3's actual win rate is outside v3's own 95% Wilson CI
AND v3's |actual - predicted| gap is larger than production's; a bucket
FAILS only if it misses on >= 2 of the 4 pool x seed runs in the SAME
direction (a perfectly calibrated model misses ~1 of 20 buckets per run by
chance). Plus a pooled check: per bucket, pooled across the four runs, the
predicted rate must be inside the pooled Wilson CI. Also reported: the
fallback's share of draws below zero vs. the real share for those players.

`--bucket-stage expanding|N` runs the same check with the n-aware second
stage for thin histories (model_core.fit_bucket_stage), and adds the two
thin-specific bars (designer, 2026-10-05): thin-history (1-6 games)
aggregate must improve on BOTH RMSE and |bias| vs. shipped v3, and no
position x bucket cell may have |bias| > 1.0. Shipped v3 is the pair log's
stage-1 D, so both sides come from the same run.

`--tail smooth:C | mix:PI` (Phase 5 thin-tails candidates, designer 2026-10-05)
gives own-history players a tail beyond their resampled games, variance- and
mean-preserving so the calibrated spread is untouched (see tail_draws).

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_ship_check.py [--bucket-stage expanding|16] [--tail smooth:0.5]
"""

import argparse
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import compute_points, summarize  # noqa: E402
from calibration_check import PROB_BUCKETS, build_pool_by_position, draw_synthetic_matchup  # noqa: E402
from phase4_combined_stack import (  # noqa: E402
    MATCHUPS_PER_WEEK,
    RNG_SEEDS,
    SD_FLOOR,
    TRIALS,
    build_state,
    realistic_pool,
    resid_bin,
)
from phase4_tail_attribution import production_distribution  # noqa: E402
from phase4_variance_pooling import K  # noqa: E402
import model_core  # noqa: E402

OUT_PATH = Path(__file__).parent.parent / "PHASE4_SHIP_CHECK_REPORT.md"  # gitignored
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
MIN_BIN = 10
VAR_Z_RANGE = (0.9, 1.15)
BIAS_LIMIT = 0.2
MAX_SAME_DIRECTION_MISSES = 1  # a bucket fails at 2+ same-direction misses across the 4 runs


def wilson_ci(successes: int, n: int, z: float = 1.96):
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def ratio_pool(ratio_bins, position, b):
    """Same bin; else merge with adjacent bins; else the whole position pool."""
    own = ratio_bins.get((position, b))
    if own is not None and len(own) >= MIN_BIN:
        return own
    merged = [ratio_bins[k] for k in [(position, b - 1), (position, b), (position, b + 1)] if k in ratio_bins]
    if merged and sum(len(m) for m in merged) >= MIN_BIN:
        return np.concatenate(merged)
    everything = [v for (p, _), v in ratio_bins.items() if p == position]
    if everything and sum(len(m) for m in everything) >= MIN_BIN:
        return np.concatenate(everything)
    return None


def parse_tail(spec):
    """None, ("smooth", c) or ("mix", pi)."""
    if not spec:
        return None
    kind, value = spec.split(":")
    assert kind in ("smooth", "mix"), spec
    return kind, float(value)


def v3_info(eid, target, history, pos_var, ratio_bins, position, cache, stats):
    """(outcome set, target, pooled predictive SD, own-history branch?) for one player."""
    if eid in cache:
        return cache[eid]
    games = history.get(eid)
    n = len(games) if games else 0
    raw = np.array([compute_points(g) for g in games]) if n else None
    s2 = float(raw.var(ddof=0)) if n else 0.0
    own = s2 > 0 and n >= 2
    cache[eid] = (v3_distribution(eid, target, history, pos_var, ratio_bins, position, {}, stats), target,
                  v3_sd(n, s2, pos_var.get(position)), own)
    return cache[eid]


def v3_sd(n, s2, pv):
    var_i = s2 if pv is None else (n / (n + K)) * s2 + (1 - n / (n + K)) * pv
    return math.sqrt(var_i * (n + 1) / (n - 1)) if n >= 2 else math.sqrt(var_i)


def tail_draws(info, rng, size, tail, resid_shape):
    """`size` draws for one player. Own-history players' outcome sets are centered on the target with
    population SD = sd, so both candidates keep the mean and variance exactly:
      smooth:C -- smoothed bootstrap: t + sqrt(1-C^2)*(x - t) + N(0, (C*sd)^2)
      mix:PI   -- with probability PI the draw is t + sd*z, z from the position's pooled standardized
                  residuals (real big games included; mean 0, variance 1)
    Fallback (ratio-pool) and point-mass players are unchanged."""
    dist, target, sd, own = info
    x = rng.choice(dist, size=size, replace=True)
    if tail is None or not own:
        return x
    kind, value = tail
    if kind == "smooth":
        return target + math.sqrt(1 - value * value) * (x - target) + rng.normal(0.0, value * sd, size)
    if resid_shape is None:
        return x
    swap = rng.random(size) < value
    x[swap] = target + sd * rng.choice(resid_shape, size=int(swap.sum()), replace=True)
    return x


def v3_distribution(eid, target, history, pos_var, ratio_bins, position, cache, stats):
    if eid in cache:
        return cache[eid]
    games = history.get(eid)
    n = len(games) if games else 0
    raw = np.array([compute_points(g) for g in games]) if n else None
    s2 = float(raw.var(ddof=0)) if n else 0.0
    pv = pos_var.get(position)
    var_i = s2 if pv is None else (n / (n + K)) * s2 + (1 - n / (n + K)) * pv
    sd = math.sqrt(var_i * (n + 1) / (n - 1)) if n >= 2 else math.sqrt(var_i)
    if s2 > 0 and n >= 2:
        dist = target + (raw - raw.mean()) * (sd / math.sqrt(s2))
    else:
        pool = ratio_pool(ratio_bins, position, resid_bin(target))
        if pool is not None and pool.mean() != 0:
            dist = target * pool / pool.mean()
            stats["neg"].setdefault(position, []).append(float(np.mean(dist < 0)))
        else:
            dist = np.array([target])
            stats["point_mass"] += 1
    cache[eid] = dist
    return dist


def simulate(ids_a, ids_b, sampler, rng, draw=None):
    """`draw(eid, rng, size)` overrides plain resampling of sampler(eid) (tail candidates)."""
    def totals(ids):
        t = np.zeros(TRIALS)
        for eid in ids:
            t += draw(eid, rng, TRIALS) if draw else rng.choice(sampler(eid), size=TRIALS, replace=True)
        return t
    a, b = totals(ids_a), totals(ids_b)
    return float(np.mean(a > b)), float(np.std(a - b))


def var_zprime(rows, p, sd_key):
    pred = np.array([r[p] for r in rows])
    act = np.array([r["actual"] for r in rows])
    sd = np.array([r[sd_key] for r in rows])
    b, a = np.polyfit(pred, act, 1)
    keep = sd >= SD_FLOOR
    return float(np.var((act[keep] - (a + b * pred[keep])) / sd[keep])), int((~keep).sum()), b


def buckets(rows, prob_key):
    out = []
    for lo, hi in PROB_BUCKETS:
        bk = [r for r in rows if r["actual"] != 0 and lo <= max(r[prob_key], 1 - r[prob_key]) < hi]
        n = len(bk)
        if n == 0:
            out.append((lo, hi, 0, None, None, None))
            continue
        mp = sum(max(r[prob_key], 1 - r[prob_key]) for r in bk) / n
        wins = sum((r["actual"] > 0) == (r[prob_key] >= 0.5) for r in bk)
        out.append((lo, hi, n, mp, wins / n, wilson_ci(wins, n)))
    return out


def thin_bars(log) -> tuple[bool, list]:
    """Designer's thin-history bars, from one pair log: final D vs. shipped v3 (stage-1 D)."""
    def m(rows):
        n = len(rows)
        e = [p - a for p, a in rows]
        return sum(abs(x) for x in e) / n, math.sqrt(sum(x * x for x in e) / n), sum(e) / n
    scored = [r for r in log if r[2]]
    thin_new = [(r[6], r[5]) for r in scored if r[7] <= 6]
    thin_old = [(r[8], r[5]) for r in scored if r[7] <= 6]
    (mae_n, rmse_n, bias_n), (mae_o, rmse_o, bias_o) = m(thin_new), m(thin_old)
    agg_ok = rmse_n < rmse_o and abs(bias_n) < abs(bias_o)
    lines = [f"Thin histories (1-6 games), n={len(thin_new)}: shipped v3 MAE {mae_o:.3f} / RMSE {rmse_o:.3f} / "
             f"bias {bias_o:+.3f} -> candidate {mae_n:.3f} / {rmse_n:.3f} / {bias_n:+.3f}: improves RMSE AND |bias| = "
             f"{'PASS' if agg_ok else 'FAIL'}", "",
             "| Position | Bucket | n | Shipped v3 bias | Candidate bias | |bias| <= 1.0? |", "|---|---|---|---|---|---|"]
    cells_ok = True
    for pos in POSITIONS:
        for b in ("1-3", "4-6", "7-8"):
            rows = [r for r in scored if r[3] == pos and model_core.history_bucket(r[7]) == b]
            if not rows:
                continue
            bn, bo = m([(r[6], r[5]) for r in rows])[2], m([(r[8], r[5]) for r in rows])[2]
            ok = abs(bn) <= 1.0
            cells_ok &= ok
            lines.append(f"| {pos} | {b} | {len(rows)} | {bo:+.3f} | {bn:+.3f} | {'yes' if ok else 'NO'} |")
    lines.append(f"\nNo position x bucket cell with |bias| > 1.0 = {'PASS' if cells_ok else 'FAIL'}")
    return agg_ok and cells_ok, lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-stage", default=None,
                        help="n-aware second stage window: 'expanding' or a number of weeks")
    parser.add_argument("--tail", default=None, help="thin-tails candidate: smooth:C or mix:PI")
    args = parser.parse_args()
    stage = args.bucket_stage if args.bucket_stage in (None, "expanding") else int(args.bucket_stage)
    tail = parse_tail(args.tail)
    out_path = OUT_PATH if stage is None else OUT_PATH.with_name(f"PHASE4_SHIP_CHECK_NAWARE_{stage}.md")
    if tail:
        out_path = OUT_PATH.with_name(f"PHASE5_SHIP_CHECK_TAIL_{tail[0]}_{tail[1]}.md")
    targets, weekly_state, point_results, pair_log = build_state(
        extra_history_seasons=1, return_pair_log=True, bucket_stage=stage)
    print(f"bucket_stage = {stage!r}, tail = {tail!r}")
    print(f"Scored window: {targets[0]} .. {targets[-1]} ({len(targets)} weeks)")

    # Per-position bias (ship bar), scored weeks only.
    bias_lines = ["| Position | n | MAE | RMSE | Bias | Within ±0.2? |", "|---|---|---|---|---|---|"]
    bias_ok = True
    v3_rows = point_results["Combined stack (D-rolling)"]
    for pos in POSITIONS:
        s = summarize([r for r in v3_rows if r["position"] == pos])
        ok = abs(s["bias"]) <= BIAS_LIMIT
        bias_ok &= ok
        bias_lines.append(f"| {pos} | {s['n']} | {s['mae']:.2f} | {s['rmse']:.2f} | {s['bias']:+.2f} | {'yes' if ok else 'NO'} |")
    s_all = summarize(v3_rows)
    a_all = summarize(point_results["Baseline A -- last 8"])
    bias_lines.append(f"| **Overall** | {s_all['n']} | {s_all['mae']:.2f} | {s_all['rmse']:.2f} | {s_all['bias']:+.2f} | |")
    bias_lines.append(f"\nBaseline A (production) overall on the same weeks: MAE {a_all['mae']:.2f}, "
                      f"RMSE {a_all['rmse']:.2f}, bias {a_all['bias']:+.2f}")

    # Real below-zero share for degenerate-history player-weeks (what the fallback should mimic).
    real_neg = {}
    for st in weekly_state.values():
        for eid, row in st["actuals"].items():
            proj = st["projections"].get(eid)
            if proj is None:
                continue
            games = st["history"].get(eid) or []
            if len(games) < 2 or np.var([compute_points(g) for g in games]) == 0:
                real_neg.setdefault(proj["position"], []).append(compute_points(row) < 0)

    pools = {
        "all players": lambda st: build_pool_by_position(st["projections"], st["actuals"]),
        "realistic top-K": lambda st: realistic_pool(st["projections"], st["actuals"], st["baseline_d"]),
    }
    stats = {"neg": {}, "point_mass": 0}
    sections, verdicts = [], []
    misses = {}  # (lo, hi) -> ["under" | "over", ...], one entry per run where the bucket missed
    pooled = {}  # (lo, hi) -> [n, wins, sum of predicted]
    for pool_name, make_pool in pools.items():
        for seed in RNG_SEEDS:
            rng_py = random.Random(seed)
            rng_v3, rng_prod = np.random.default_rng(seed), np.random.default_rng(seed + 1)
            rows = []
            for season, week in targets:
                st = weekly_state[(season, week)]
                pool = make_pool(st)
                pos_of = {e: p["position"] for e, p in st["projections"].items()}
                v3_cache, prod_cache = {}, {}

                def v3_sampler(e):
                    return v3_distribution(e, st["baseline_d"][e], st["history"], st["pos_var"],
                                           st["ratio_bins"], pos_of[e], v3_cache, stats)

                def prod_sampler(e):
                    return production_distribution(e, st["history"], st["baseline_a"], prod_cache)

                info_cache = {}

                def v3_draw(e, rng, size):
                    info = v3_info(e, st["baseline_d"][e], st["history"], st["pos_var"], st["ratio_bins"],
                                   pos_of[e], info_cache, stats)
                    return tail_draws(info, rng, size, tail, st["resid_shape"].get(pos_of[e]))

                for _ in range(MATCHUPS_PER_WEEK):
                    drawn = draw_synthetic_matchup(pool, rng_py)
                    if drawn is None:
                        continue
                    ta, tb = drawn
                    v3_p, v3_sd = simulate(ta, tb, v3_sampler, rng_v3, v3_draw if tail else None)
                    pr_p, pr_sd = simulate(ta, tb, prod_sampler, rng_prod)
                    rows.append({
                        "v3_p": v3_p, "v3_sd": v3_sd, "pr_p": pr_p, "pr_sd": pr_sd,
                        "v3_pred": sum(st["baseline_d"][e] for e in ta) - sum(st["baseline_d"][e] for e in tb),
                        "pr_pred": sum(st["baseline_a"][e] for e in ta) - sum(st["baseline_a"][e] for e in tb),
                        "actual": sum(compute_points(st["actuals"][e]) for e in ta)
                                  - sum(compute_points(st["actuals"][e]) for e in tb),
                    })
                print(f"  {pool_name} seed {seed}: done season {season} week {week}", end="\r")
            print()

            vz3, ex3, b3 = var_zprime(rows, "v3_pred", "v3_sd")
            vzp, exp_, bp = var_zprime(rows, "pr_pred", "pr_sd")
            bv3, bpr = buckets(rows, "v3_p"), buckets(rows, "pr_p")
            lines = ["| Bucket | Production: n, pred→actual (gap) | v3: n, pred→actual (gap) | v3 Wilson 95% CI | "
                     "Pred in CI? | v3 gap ≤ prod? | Bucket |", "|---|---|---|---|---|---|---|"]
            no_worse = True
            for (lo, hi, n3, mp3, wr3, ci3), (_, _, npr, mppr, wrpr, _) in zip(bv3, bpr):
                if n3 == 0:
                    lines.append(f"| {lo:.0%}-{hi:.0%} | {npr} | 0 | - | - | - | pass (empty) |")
                    continue
                g3, gp = abs(wr3 - mp3), (abs(wrpr - mppr) if npr else float("inf"))
                in_ci = ci3[0] <= mp3 <= ci3[1]
                ok = in_ci or g3 <= gp  # misses only if outside its own CI AND worse than production
                no_worse &= ok
                if not ok:
                    # actual above predicted = under-confident
                    misses.setdefault((lo, hi), []).append("under" if wr3 > mp3 else "over")
                acc = pooled.setdefault((lo, hi), [0, 0, 0.0])
                acc[0] += n3
                acc[1] += round(wr3 * n3)
                acc[2] += mp3 * n3
                prod_cell = f"{npr}, {mppr:.1%}→{wrpr:.1%} ({gp*100:.1f})" if npr else "0"
                lines.append(f"| {lo:.0%}-{hi:.0%} | {prod_cell} | {n3}, {mp3:.1%}→{wr3:.1%} ({g3*100:.1f}) | "
                             f"[{ci3[0]:.1%}, {ci3[1]:.1%}] | {'yes' if in_ci else 'no'} | "
                             f"{'yes' if g3 <= gp else 'no'} | {'pass' if ok else 'miss'} |")
            var_ok = VAR_Z_RANGE[0] <= vz3 <= VAR_Z_RANGE[1]
            verdicts.append((pool_name, seed, no_worse, var_ok))
            head = (f"### {pool_name}, seed {seed} ({len(rows)} matchups)\n\n"
                    f"v3: Var(z')={vz3:.3f} ({'within' if var_ok else 'OUTSIDE'} 0.9-1.15), slope {b3:.3f}, "
                    f"{ex3} below SD floor. Production: Var(z')={vzp:.3f}, slope {bp:.3f}, {exp_} below SD floor.")
            print(head)
            print("\n".join(lines))
            sections.append(head + "\n\n" + "\n".join(lines))

    neg_lines = ["| Position | Fallback dists | Fallback draws < 0 | Real outcomes < 0 (degenerate history) |",
                 "|---|---|---|---|"]
    for pos in POSITIONS:
        f = stats["neg"].get(pos, [])
        r = real_neg.get(pos, [])
        neg_lines.append(f"| {pos} | {len(f)} | {np.mean(f) if f else float('nan'):.1%} | "
                         f"{np.mean(r) if r else float('nan'):.1%} (n={len(r)}) |")
    neg_lines.append(f"\nPoint-mass fallbacks (bin too thin): {stats['point_mass']}")

    verdict_lines = [f"- {p}, seed {s}: Var(z') in range = {'PASS' if vo else 'FAIL'}; "
                     f"bucket misses this run (outside v3's Wilson CI AND gap > production's): "
                     f"{'none' if nw else 'some (see table)'}" for p, s, nw, vo in verdicts]
    bucket_fail = {k: d for k, v in misses.items() for d in ("under", "over")
                   if v.count(d) > MAX_SAME_DIRECTION_MISSES}
    miss_desc = ", ".join(f"{lo:.0%}-{hi:.0%}: {len(v)} ({'/'.join(v)})" for (lo, hi), v in sorted(misses.items()))
    verdict_lines.append(f"- Buckets missing on 2+ of {len(verdicts)} runs in the same direction = "
                         f"{'PASS (none)' if not bucket_fail else 'FAIL ' + str(sorted(bucket_fail))}"
                         f"{f'; misses by bucket: {miss_desc}' if misses else ''}")
    pooled_lines = ["| Bucket | Pooled n | Mean predicted | Actual | Pooled Wilson 95% CI | Pred in CI? |",
                    "|---|---|---|---|---|---|"]
    pooled_ok = True
    for (lo, hi), (n, wins, sum_pred) in sorted(pooled.items()):
        mp, (cl, ch) = sum_pred / n, wilson_ci(wins, n)
        ok = cl <= mp <= ch
        pooled_ok &= ok
        pooled_lines.append(f"| {lo:.0%}-{hi:.0%} | {n} | {mp:.1%} | {wins / n:.1%} | [{cl:.1%}, {ch:.1%}] | "
                            f"{'yes' if ok else 'NO'} |")
    verdict_lines.append(f"- Pooled across runs, every bucket's prediction inside its Wilson CI = "
                         f"{'PASS' if pooled_ok else 'FAIL'}")
    verdict_lines.append(f"- Per-position bias within ±0.2 = {'PASS' if bias_ok else 'FAIL'}")
    thin_lines = []
    thin_ok = True
    if stage is not None:
        thin_ok, thin_lines = thin_bars(pair_log)
        verdict_lines.append(f"- Thin-history bars (aggregate RMSE+|bias| vs shipped v3; every cell |bias| <= 1.0) = "
                             f"{'PASS' if thin_ok else 'FAIL'}")
    overall = bias_ok and thin_ok and pooled_ok and not bucket_fail and all(vo for _, _, _, vo in verdicts)
    verdict_lines.insert(0, f"**Overall: {'MEETS' if overall else 'MISSES'} the ship bar.**\n")

    print("\n" + "\n".join(bias_lines))
    print("\n" + "\n".join(neg_lines))
    print("\n" + "\n".join(pooled_lines))
    print("\n" + "\n".join(verdict_lines))

    report = f"""# Phase 4 Ship Check (PRODUCTION_MODEL_SPEC.md section 4, item 1){f" -- tail candidate {args.tail}" if tail else ""}

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE only. 2023 loaded as history only;
scored weeks {targets[0]} .. {targets[-1]}, no exclusions.

## Verdict

{chr(10).join(verdict_lines)}

## Thin-history bars

{chr(10).join(thin_lines) if thin_lines else 'n/a (no bucket stage)'}

## Per-position point accuracy (v3)

{chr(10).join(bias_lines)}

## Calibration pooled across the four runs

{chr(10).join(pooled_lines)}

## Calibration, paired vs. production

{chr(10).join(sections)}

## Binned fallback floor

{chr(10).join(neg_lines)}
"""
    out_path.write_text(report)
    print("\n" + "\n".join(thin_lines))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
