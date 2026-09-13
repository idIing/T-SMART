#!/usr/bin/env python3
"""Fit + freeze the learned "should-I-look" gate (#1).

Pre-registered in ``mmts_bench/PREREGISTRATION_learned_gate.md`` (committed BEFORE this
fit runs). This script is the *fit* step: it estimates, per branch, the realized net value
of looking and derives the per-branch fire/suppress partition, then writes the frozen
policy artifact consumed by ``tsqa/verifier/learned_gate.py``.

Estimator (frozen by the pre-reg, interpretable only):
    label   y = vision_on_correct - vision_off_correct  in {-1, 0, +1}
            (vision_on  = baseline arm final answer; vision_off = math-only answer)
    policy  fire branch b  iff  E[y | b] > tau
    tau     chosen by leave-one-subset-out CV (Base <-> InWild); Match/Align contribute
            zero fired rows (negative control, excluded).

The look policy is a SUPPRESSION FILTER over the additive gate: on fired branches it
defers byte-identically to the additive gate; on suppressed branches it drops the look.
It can only remove looks (the label exists only on additive-fired rows), so look-rate is
guaranteed <= additive — the pre-reg H1 mechanism.

Outputs ``tsqa/tsqa/verifier/gate_policy.json`` (hash-pinned before the held-out
TSExam evaluation). No held-out (TSExam) data is touched here.
"""

import csv
import json
import os
from collections import defaultdict
from datetime import date

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VT_DIR = os.path.join(
    _REPO, "mmts_bench", "outputs", "sweep_20260619", "visual_trust"
)
_POOLED = os.path.join(_VT_DIR, "discordance_pooled.csv")
_POLICY_OUT = os.path.join(_REPO, "tsqa", "tsqa", "verifier", "gate_policy.json")

ALL_BRANCHES = ["trend", "periodicity", "noise", "similarity", "anomaly", "causality"]


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _y(row):
    """Realized net value of looking on a paired fired row."""
    return int(row["final_pred"] == row["gold"]) - int(row["math_letter"] == row["gold"])


def _branch_means(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["branch_used"]].append(_y(r))
    return {b: (float(np.mean(v)), len(v)) for b, v in by.items()}


def _boot_ci(vals, n_boot=10000, seed=0):
    if not vals:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def loso_cv(rows, tau_grid):
    """Leave-one-subset-out CV over {Base, InWild}.

    For each held-out subset, fit per-branch E[y] on the OTHER subset, fire branches with
    train mean > tau, and score realized OA gain over math-only on the held-out subset.
    A branch unseen in the train fold defaults to KEEP (defer to additive) — we never
    suppress a branch we have not measured.
    """
    subsets = sorted({r["subset"] for r in rows})
    curve = {}
    for tau in tau_grid:
        fold_gains, fold_lookrates = [], []
        for test_sub in subsets:
            train = [r for r in rows if r["subset"] != test_sub]
            test = [r for r in rows if r["subset"] == test_sub]
            if not train or not test:
                continue
            train_means = _branch_means(train)
            fire = {b for b, (m, _) in train_means.items() if m > tau}
            # unseen-in-train branches default to keep (fire)
            seen = set(train_means)
            n = len(test)
            gain = sum(_y(r) for r in test if (r["branch_used"] in fire or r["branch_used"] not in seen))
            looks = sum(1 for r in test if (r["branch_used"] in fire or r["branch_used"] not in seen))
            fold_gains.append(gain / n)
            fold_lookrates.append(looks / n)
        if fold_gains:
            curve[round(tau, 4)] = (float(np.mean(fold_gains)), float(np.mean(fold_lookrates)))
    return curve


def main():
    if not os.path.exists(_POOLED):
        raise SystemExit(f"discordance fit set not found: {_POOLED}")
    rows = _load(_POOLED)
    n = len(rows)

    # ---- pooled per-branch realized net value (with bootstrap CI) ----
    by = defaultdict(list)
    for r in rows:
        by[r["branch_used"]].append(_y(r))
    pooled = {}
    for b in ALL_BRANCHES:
        vals = by.get(b, [])
        m = float(np.mean(vals)) if vals else None
        lo, hi = _boot_ci(vals) if vals else (None, None)
        pooled[b] = {"mean_y": m, "n": len(vals), "ci95": [lo, hi]}

    # ---- LOSO CV over tau ----
    tau_grid = [round(t, 2) for t in np.arange(-0.30, 0.301, 0.01)]
    curve = loso_cv(rows, tau_grid)
    best_tau = max(curve, key=lambda t: curve[t][0])
    # Report the contiguous basin of tau that yields the optimum partition.
    best_gain = curve[best_tau][0]
    basin = [t for t, (g, _) in curve.items() if abs(g - best_gain) < 1e-9]

    # tau=0 is the principled choice (fire iff E[y]>0 maximizes expected OA). Adopt it if
    # it lies in the optimal basin; otherwise adopt the CV argmax.
    tau_star = 0.0 if (min(basin) <= 0.0 <= max(basin)) else best_tau

    # ---- derive the partition from the pooled fit at tau_star ----
    fire = [b for b in ALL_BRANCHES if pooled[b]["mean_y"] is not None and pooled[b]["mean_y"] > tau_star]
    suppress = [b for b in ALL_BRANCHES if pooled[b]["mean_y"] is not None and pooled[b]["mean_y"] <= tau_star]

    # ---- additive (fire-all) reference on the discordance set ----
    additive_gain = float(np.mean([_y(r) for r in rows]))
    learned_gain = float(np.mean([_y(r) if r["branch_used"] in fire else 0 for r in rows]))
    add_lookrate = 1.0  # all discordance rows are additive-fired by construction
    learned_lookrate = sum(1 for r in rows if r["branch_used"] in fire) / n

    policy = {
        "version": f"learned_gate_v1_{date.today().isoformat()}",
        "policy_type": "per_branch_suppression_filter_over_additive",
        "fire_branches": fire,
        "suppress_branches": suppress,
        "threshold_tau": tau_star,
        "estimator": "per-branch base rate of realized y; fire iff E[y|branch] > tau; "
                     "tau selected by leave-one-subset-out CV (Base<->InWild)",
        "label_definition": "y = int(vision_on_final == gold) - int(math_only_final == gold)",
        "feature_set": "pre-look only: branch (dominant, LOSO sign-stable). "
                       "vision_confidence EXCLUDED (post-look). subtype/flag refinements "
                       "evaluated and REJECTED (see rejected_refinements).",
        "per_branch_realized_y": pooled,
        "loso_cv": {
            "tau_star": tau_star,
            "cv_argmax_tau": best_tau,
            "optimal_basin_tau": [min(basin), max(basin)],
            "discordance_set_gain_additive": additive_gain,
            "discordance_set_gain_learned": learned_gain,
            "discordance_set_lookrate_additive": add_lookrate,
            "discordance_set_lookrate_learned": learned_lookrate,
        },
        "rejected_refinements": (
            "Per-subtype splits within suppress branches: only noise.variance_level was "
            "marginally positive (+0.067, within-noise sampling noise) and contradicts the "
            "frozen nu_ad_fix (no_vision noise); within fire branches all subtypes were "
            ">= 0. Branch-only is the LOSO sign-stable, anti-overfit, transferable signal "
            "the pre-reg demands; subtype/flag conditioning rejected."
        ),
        "provenance": {
            "fit_source": os.path.relpath(_POOLED, _REPO),
            "n_fit_rows": n,
            "fit_date": date.today().isoformat(),
            "fit_subsets": sorted({r["subset"] for r in rows}),
            "preregistration": "mmts_bench/PREREGISTRATION_learned_gate.md (committed 646f9ac, pre-fit)",
        },
    }

    os.makedirs(os.path.dirname(_POLICY_OUT), exist_ok=True)
    with open(_POLICY_OUT, "w") as f:
        json.dump(policy, f, indent=2)

    # ---- human-readable report ----
    print("=" * 74)
    print("LEARNED SHOULD-I-LOOK GATE — FIT REPORT")
    print("=" * 74)
    print(f"fit set: {os.path.relpath(_POOLED, _REPO)}  (n={n} discordant fired rows)")
    print(f"subsets: {sorted({r['subset'] for r in rows})}")
    print("\nPooled per-branch realized net value of looking (y = vis_on - vis_off):")
    print(f"  {'branch':12s} {'n':>4} {'mean_y':>8} {'95% CI':>20}  decision")
    for b in ALL_BRANCHES:
        d = pooled[b]
        if d["mean_y"] is None:
            print(f"  {b:12s} {0:>4} {'  n/a':>8}")
            continue
        ci = d["ci95"]
        dec = "FIRE" if b in fire else "suppress"
        print(f"  {b:12s} {d['n']:>4} {d['mean_y']:>+8.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]   {dec}")
    print(f"\nLOSO CV: tau* = {tau_star:+.2f}  (CV argmax {best_tau:+.2f}, optimal basin "
          f"[{min(basin):+.2f},{max(basin):+.2f}])")
    print(f"  discordance-set realized OA gain over math-only:")
    print(f"     additive (fire all): {additive_gain:+.4f}   look-rate {add_lookrate:.2f}")
    print(f"     learned  (fire {','.join(fire)}): {learned_gain:+.4f}   look-rate {learned_lookrate:.2f}")
    print(f"\nFIRE     branches: {fire}")
    print(f"SUPPRESS branches: {suppress}")
    print(f"\nfrozen policy written -> {os.path.relpath(_POLICY_OUT, _REPO)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
