"""
phase4_tail_attribution.py

Realistic-pool tail diagnosis + ship lens (designer request, 2026-10-04).
OFFLINE only.

On the realistic top-K pool the combined stack has slope ~1 and Var(z') ~1
overall, yet its >=90% bucket is ~97.5% predicted vs ~73% actual -- a miss
local to the tail. Two candidate causes, separated here:
  (i)  means overstated specifically for the most lopsided matchups
       (level-dependent, invisible to one linear beta);
  (ii) simulated SD too small for specific matchups (heteroscedastic) --
       e.g. players whose last 8 games happened to be unusually consistent
       keep most of their own (understated) variance at k=3.

Diagnostics, realistic pool, both seeds pooled:
  1. Deciles of simulated margin SD: mean z' and Var(z') per decile.
     Var(z') well above 1 in low-SD deciles -> (ii).
  2. Deciles of |predicted margin| / SD: mean oriented actual margin /
     mean |predicted margin| per decile. Ratio well below 1 in the top
     decile -> (i).
  3. The >=90% bucket vs. all matchups: history length, degenerate-player
     count, per-position projected advantage, favored vs. underdog residual.

Ship lens: the CURRENT production model (Baseline A center + raw
bootstrap of the last 8 games, i.e. what's live) on the IDENTICAL matchups
(same roster draws -- roster RNG and simulation RNGs are separate), so the
comparison is paired, not two independent samples. The combined stack's
own numbers reproduce phase4_combined_stack.py's realistic-pool run
exactly (same seeds, same RNG streams) -- a cross-check.

Run: POLARS_SKIP_CPU_CHECK=1 python3 scripts/phase4_tail_attribution.py
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backtest_evaluation import compute_points  # noqa: E402
from calibration_check import PROB_BUCKETS, draw_synthetic_matchup  # noqa: E402
from phase4_combined_stack import (  # noqa: E402
    MATCHUPS_PER_WEEK,
    RNG_SEEDS,
    SD_FLOOR,
    TRIALS,
    build_state,
    combined_distribution,
    realistic_pool,
)

OUT_PATH = Path(__file__).parent.parent / "PHASE4_TAIL_ATTRIBUTION_REPORT.md"  # gitignored
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def production_distribution(entity_id, history, baseline_a, cache):
    """What monteCarlo.js does today: resample the raw last-8 game points;
    no history -> the projected mean as a point mass."""
    if entity_id in cache:
        return cache[entity_id]
    games = history.get(entity_id)
    dist = np.array([compute_points(g) for g in games]) if games else np.array([baseline_a[entity_id]])
    cache[entity_id] = dist
    return dist


def simulate(ids_a, ids_b, sampler, rng):
    def totals(ids):
        t = np.zeros(TRIALS)
        for eid in ids:
            t += rng.choice(sampler(eid), size=TRIALS, replace=True)
        return t
    a, b = totals(ids_a), totals(ids_b)
    return float(np.mean(a > b)), float(np.std(a - b))


def fit_zprime(rows, pred_key, sd_key):
    pred = np.array([r[pred_key] for r in rows])
    act = np.array([r["actual_margin"] for r in rows])
    sd = np.array([r[sd_key] for r in rows])
    b, a = np.polyfit(pred, act, 1)
    z = np.where(sd >= SD_FLOOR, (act - (a + b * pred)) / np.where(sd > 0, sd, 1), np.nan)
    return z, b


def bucket_table(rows, prob_key) -> str:
    lines = ["| Bucket | n | Mean predicted | Actual win rate | 95% CI | Within CI? |", "|---|---|---|---|---|---|"]
    for lo, hi in PROB_BUCKETS:
        b = [r for r in rows if r["actual_margin"] != 0 and lo <= max(r[prob_key], 1 - r[prob_key]) < hi]
        n = len(b)
        if n == 0:
            lines.append(f"| {lo:.0%}-{hi:.0%} | 0 | - | - | - | - |")
            continue
        mp = sum(max(r[prob_key], 1 - r[prob_key]) for r in b) / n
        wr = sum((r["actual_margin"] > 0) == (r[prob_key] >= 0.5) for r in b) / n
        ci = 1.96 * math.sqrt(wr * (1 - wr) / n) if n > 1 else 0.0
        lines.append(f"| {lo:.0%}-{hi:.0%} | {n} | {mp:.1%} | {wr:.1%} | ±{ci:.1%} | {'yes' if abs(wr - mp) <= ci else 'NO'} |")
    return "\n".join(lines)


def main() -> None:
    targets, weekly_state, _ = build_state()

    all_rows, ship_sections = [], []
    for seed in RNG_SEEDS:
        rng_py = random.Random(seed)
        rng_comb = np.random.default_rng(seed)      # same stream as phase4_combined_stack's run
        rng_prod = np.random.default_rng(seed + 1)  # separate, so it can't perturb the combined numbers
        rows = []
        for season, week in targets:
            st = weekly_state.get((season, week))
            if st is None:
                continue
            pool = realistic_pool(st["projections"], st["actuals"], st["baseline_d"])
            pos_of = {eid: p["position"] for eid, p in st["projections"].items()}
            comb_cache, prod_cache, stats = {}, {}, {"fallback_n": 0, "resid_neg": 0.0, "normal_neg": 0.0}

            def comb_sampler(eid):
                return combined_distribution(eid, st["baseline_d"][eid], st["history"], st["pos_var"],
                                             st["resid_shape"], pos_of[eid], comb_cache, stats)

            def prod_sampler(eid):
                return production_distribution(eid, st["history"], st["baseline_a"], prod_cache)

            for _ in range(MATCHUPS_PER_WEEK):
                drawn = draw_synthetic_matchup(pool, rng_py)
                if drawn is None:
                    continue
                team_a, team_b = drawn
                comb_p, comb_sd = simulate(team_a, team_b, comb_sampler, rng_comb)
                prod_p, prod_sd = simulate(team_a, team_b, prod_sampler, rng_prod)
                act_a = sum(compute_points(st["actuals"][p]) for p in team_a)
                act_b = sum(compute_points(st["actuals"][p]) for p in team_b)
                fav, dog = (team_a, team_b) if comb_p >= 0.5 else (team_b, team_a)
                hist_n = [len(st["history"].get(e) or []) for e in team_a + team_b]
                adv = {pos: sum(st["baseline_d"][e] for e in fav if pos_of[e] == pos)
                       - sum(st["baseline_d"][e] for e in dog if pos_of[e] == pos) for pos in POSITIONS}
                rows.append({
                    "seed": seed, "season": season, "week": week,
                    "pos_var_ready": all(st["pos_var"].get(p) is not None for p in POSITIONS),
                    "comb_p": comb_p, "comb_sd": comb_sd,
                    "comb_pred": sum(st["baseline_d"][e] for e in team_a) - sum(st["baseline_d"][e] for e in team_b),
                    "prod_p": prod_p, "prod_sd": prod_sd,
                    "prod_pred": sum(st["baseline_a"][e] for e in team_a) - sum(st["baseline_a"][e] for e in team_b),
                    "actual_margin": act_a - act_b,
                    "hist_mean": float(np.mean(hist_n)), "n_short": sum(n < 8 for n in hist_n),
                    "n_degenerate": sum(n < 2 or np.var([compute_points(g) for g in st["history"][e]]) == 0
                                        for e, n in zip(team_a + team_b, hist_n)),
                    "adv": adv,
                    "fav_resid": sum(compute_points(st["actuals"][e]) - st["baseline_d"][e] for e in fav),
                    "dog_resid": sum(compute_points(st["actuals"][e]) - st["baseline_d"][e] for e in dog),
                })
            print(f"  seed {seed}: done season {season} week {week}", end="\r")
        print()

        comb_z, comb_b = fit_zprime(rows, "comb_pred", "comb_sd")
        prod_z, prod_b = fit_zprime(rows, "prod_pred", "prod_sd")
        for r, z in zip(rows, comb_z):
            r["comb_z"] = z
        ship_sections.append(
            f"### Seed {seed}\n\n"
            f"**Production today** (Baseline A + raw bootstrap): slope b={prod_b:.3f}, "
            f"Var(z')={np.nanvar(prod_z):.3f}\n\n{bucket_table(rows, 'prod_p')}\n\n"
            f"**Combined stack**: slope b={comb_b:.3f}, Var(z')={np.nanvar(comb_z):.3f}\n\n{bucket_table(rows, 'comb_p')}"
        )
        all_rows.extend(rows)

    # ---- Diagnostic 1: deciles of simulated margin SD.
    valid = [r for r in all_rows if not np.isnan(r["comb_z"])]
    d1 = ["| SD decile | SD range | n | Mean z' | Var(z') |", "|---|---|---|---|---|"]
    for i, chunk in enumerate(np.array_split(sorted(valid, key=lambda r: r["comb_sd"]), 10), 1):
        z = np.array([r["comb_z"] for r in chunk])
        d1.append(f"| {i} | {chunk[0]['comb_sd']:.1f}-{chunk[-1]['comb_sd']:.1f} | {len(chunk)} | "
                  f"{z.mean():+.3f} | {z.var():.3f} |")

    # ---- Diagnostic 2: deciles of |predicted margin| / SD, oriented toward the predicted favorite.
    d2 = ["| |pred|/SD decile | Range | n | Mean |pred| | Mean oriented actual | Ratio | Mean fav prob | Fav win rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    keyed = sorted(valid, key=lambda r: abs(r["comb_pred"]) / r["comb_sd"])
    for i, chunk in enumerate(np.array_split(keyed, 10), 1):
        sign = np.array([1.0 if r["comb_pred"] >= 0 else -1.0 for r in chunk])
        pred_abs = np.array([abs(r["comb_pred"]) for r in chunk])
        oriented = np.array([r["actual_margin"] for r in chunk]) * sign
        fav_p = np.array([max(r["comb_p"], 1 - r["comb_p"]) for r in chunk])
        decided = [r for r in chunk if r["actual_margin"] != 0]
        win = sum((r["actual_margin"] > 0) == (r["comb_p"] >= 0.5) for r in decided) / len(decided)
        lo, hi = abs(chunk[0]["comb_pred"]) / chunk[0]["comb_sd"], abs(chunk[-1]["comb_pred"]) / chunk[-1]["comb_sd"]
        d2.append(f"| {i} | {lo:.2f}-{hi:.2f} | {len(chunk)} | {pred_abs.mean():.1f} | {oriented.mean():.1f} | "
                  f"{oriented.mean() / pred_abs.mean():.3f} | {fav_p.mean():.1%} | {win:.1%} |")

    # ---- Diagnostic 3: the >=90% bucket vs. all matchups.
    tail = [r for r in all_rows if max(r["comb_p"], 1 - r["comb_p"]) >= 0.90]

    def profile(rs):
        return {
            "n": len(rs),
            "hist_mean": np.mean([r["hist_mean"] for r in rs]),
            "n_short": np.mean([r["n_short"] for r in rs]),
            "n_degenerate": np.mean([r["n_degenerate"] for r in rs]),
            "sd": np.mean([r["comb_sd"] for r in rs]),
            "pred": np.mean([abs(r["comb_pred"]) for r in rs]),
            "fav_resid": np.mean([r["fav_resid"] for r in rs]),
            "dog_resid": np.mean([r["dog_resid"] for r in rs]),
            "adv": {p: np.mean([r["adv"][p] for r in rs]) for p in POSITIONS},
        }

    # ---- Diagnostic 4: is the tail concentrated in the backtest window's first weeks (no prior-season
    # data loaded, so everyone has 1-3 games and pooled variance isn't established yet)?
    from collections import Counter
    by_week = Counter((r["season"], r["week"]) for r in tail)
    d4 = [f"Tail matchups with simulated margin SD < {SD_FLOOR} (excluded from Var(z') and diagnostics 1-2): "
          f"{sum(r['comb_sd'] < SD_FLOOR for r in tail)}/{len(tail)}",
          f"Tail matchups in weeks where pooled variance isn't established for every position: "
          f"{sum(not r['pos_var_ready'] for r in tail)}/{len(tail)}",
          "", "| Season-week | Tail matchups |", "|---|---|"]
    d4 += [f"| {s}-{w} | {c} |" for (s, w), c in by_week.most_common(8)]
    for seed in RNG_SEEDS:
        ready = [r for r in all_rows if r["seed"] == seed and r["pos_var_ready"]]
        d4 += ["", f"**Seed {seed}, excluding not-ready weeks ({len(ready)} matchups) -- production today:**", "",
               bucket_table(ready, "prod_p"), "", f"**Seed {seed}, same matchups -- combined stack:**", "",
               bucket_table(ready, "comb_p")]

    pt, pa = profile(tail), profile(all_rows)
    d3 = ["| | >=90% bucket | All matchups |", "|---|---|---|",
          f"| n matchups | {pt['n']} | {pa['n']} |",
          f"| Mean history length (games, of 18 players) | {pt['hist_mean']:.2f} | {pa['hist_mean']:.2f} |",
          f"| Players with <8 games (of 18) | {pt['n_short']:.2f} | {pa['n_short']:.2f} |",
          f"| Degenerate players (of 18) | {pt['n_degenerate']:.2f} | {pa['n_degenerate']:.2f} |",
          f"| Mean simulated margin SD | {pt['sd']:.1f} | {pa['sd']:.1f} |",
          f"| Mean abs(predicted margin) | {pt['pred']:.1f} | {pa['pred']:.1f} |",
          f"| Favored side residual (actual - pred) | {pt['fav_resid']:+.2f} | {pa['fav_resid']:+.2f} |",
          f"| Underdog side residual | {pt['dog_resid']:+.2f} | {pa['dog_resid']:+.2f} |"]
    d3 += [f"| Projected advantage, {p} (fav - dog) | {pt['adv'][p]:+.1f} | {pa['adv'][p]:+.1f} |" for p in POSITIONS]

    for title, body in [("Diagnostic 4 -- tail by week, and buckets without the window-start weeks", d4),
                        ("Diagnostic 1 -- deciles of simulated margin SD", d1),
                        ("Diagnostic 2 -- deciles of |predicted margin| / SD", d2),
                        ("Diagnostic 3 -- >=90% bucket profile", d3)]:
        print(f"\n{title}\n" + "\n".join(body))
    print("\n\n".join(ship_sections))

    report = f"""# Phase 4 Tail Attribution + Ship Lens (realistic pool)

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}. OFFLINE only. Both seeds pooled for the
decile diagnostics; ship lens per seed on identical matchups.

## Diagnostic 4 -- tail by week, and buckets without the window-start weeks

{chr(10).join(d4)}

## Diagnostic 1 -- deciles of simulated margin SD

{chr(10).join(d1)}

## Diagnostic 2 -- deciles of |predicted margin| / SD

{chr(10).join(d2)}

## Diagnostic 3 -- >=90% bucket profile

{chr(10).join(d3)}

## Ship lens -- production today vs. combined stack, same matchups

{chr(10).join(ship_sections)}
"""
    OUT_PATH.write_text(report)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
