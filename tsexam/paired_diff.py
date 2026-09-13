"""
Paired diff for the TimeSeriesExam research harness (results.json format).
Mirrors mmts_bench/scripts/diff_configs.py: joins two run outputs per-row on id,
reports paired delta + bootstrap 95% CI + exact McNemar overall, per category,
and per branch_used (the mechanism view). Also reports how the vision gate moved.

Usage:
    python research/paired_diff.py outputs/baseline_full outputs/nu_ad_fix_full
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from tsqa.eval.cluster_diff import cluster_bootstrap_delta


def _load(tag_dir):
    rows = json.load(open(f"{tag_dir}/results.json"))
    return {str(r["id"]): r for r in rows if r and r.get("id") is not None}


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import erfc, sqrt
        return float(erfc(sqrt(((abs(b - c) - 1) ** 2 / n) / 2.0)))


def holm_adjust(pvals):
    """Holm–Bonferroni step-down adjusted p-values (FWER control) over a family
    of per-branch McNemar tests. Returned in input order; monotone in rank. Call
    per-branch significance on the multi-branch sweep by THESE, not the raw p —
    the Stage-1 promotion gate requires a multiplicity correction (finding #2)."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])  # step-down + monotone
        adj[i] = min(running, 1.0)
    return adj


def boot_ci(base, treat, n_boot=10000, seed=0):
    n = len(base)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = treat[idx].mean(1) - base[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def stats(base, treat):
    base, treat = np.asarray(base, int), np.asarray(treat, int)
    n = len(base)
    b = int(((base == 1) & (treat == 0)).sum())
    c = int(((base == 0) & (treat == 1)).sum())
    lo, hi = boot_ci(base, treat)
    return dict(n=n, ab=float(base.mean()) if n else float("nan"),
               at=float(treat.mean()) if n else float("nan"),
               d=(treat.mean() - base.mean()) if n else float("nan"),
               b=b, c=c, p=mcnemar_p(b, c), lo=lo, hi=hi)


def line(label, s):
    star = (" *" if s["d"] > 0 else " X") if s["p"] < 0.05 else ""
    bc = f"{s['b']}/{s['c']}"
    return (f"{label[:18]:<18}{s['n']:>5}{s['ab']:>8.3f}{s['at']:>8.3f}{s['d']:>+8.3f}"
            f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{bc:>9}{s['p']:>10.4f}{star}")


def grouped(ids, B, T, keyfn):
    groups = {}
    for i in ids:
        k = keyfn(B[i]) or "None"
        groups.setdefault(k, [[], []])
        groups[k][0].append(int(bool(B[i].get("correct"))))
        groups[k][1].append(int(bool(T[i].get("correct"))))
    out = [(k, stats(v[0], v[1])) for k, v in groups.items()]
    return sorted(out, key=lambda kv: kv[1]["n"], reverse=True)


def cluster_summary(ids, B, T, keyfn):
    base = [int(bool(B[i].get("correct"))) for i in ids]
    treat = [int(bool(T[i].get("correct"))) for i in ids]
    clusters = [str(keyfn(B[i]) or keyfn(T[i]) or "missing") for i in ids]
    if len(set(c for c in clusters if c.strip())) < 2:
        return None
    return cluster_bootstrap_delta(base, treat, clusters)


def print_migration(ids, T):
    """Initial->final branch migration in the TREATMENT arm — the Stage-1 reroute
    view. Empty when the loop is off (initial == final on every row)."""
    mig = defaultdict(Counter)
    moved = 0
    for i in ids:
        ini = str(T[i].get("initial_branch_used") or T[i].get("branch_used"))
        fin = str(T[i].get("branch_used"))
        mig[ini][fin] += 1
        if ini != fin:
            moved += 1
    if moved == 0:
        print("\n-- branch migration (treatment): none — initial == final on all rows")
        return
    print(f"\n-- branch migration initial->final (treatment), {moved}/{len(ids)} rows moved "
          + "-" * 18)
    for ini in sorted(mig):
        if all(f == ini for f in mig[ini]):
            continue  # this initial branch never rerouted
        dist = ", ".join(f"{fin}:{ct}" for fin, ct in sorted(mig[ini].items(),
                                                             key=lambda kv: -kv[1]))
        print(f"  {ini:<16} -> {dist}")


def print_cluster_block(ids, B, T):
    print("\n-- cluster bootstrap sensitivity " + "-" * 57)
    for label, keyfn in (
        ("category", lambda r: r.get("category")),
        # cluster on the PRE-reroute branch — clustering on a treatment-affected
        # variable would bias the cluster-robust CI (finding #1).
        ("initial_branch_used",
         lambda r: r.get("initial_branch_used") or r.get("branch_used")),
    ):
        res = cluster_summary(ids, B, T, keyfn)
        if res is None:
            print(f"{label:<18} skipped (fewer than two clusters)")
            continue
        finite = [v for v in res.sensitivity.values() if v is not None]
        lo = min(finite) if finite else float("nan")
        hi = max(finite) if finite else float("nan")
        print(
            f"{label:<18} n={res.n:<5} Δ={res.delta:+.3f} "
            f"CI=[{res.ci_lo:+.3f},{res.ci_hi:+.3f}] "
            f"leave-one Δ range=[{lo:+.3f},{hi:+.3f}]"
        )


def main():
    base_dir, treat_dir = sys.argv[1], sys.argv[2]
    B, T = _load(base_dir), _load(treat_dir)
    ids = sorted(set(B) & set(T))
    base = [int(bool(B[i].get("correct"))) for i in ids]
    treat = [int(bool(T[i].get("correct"))) for i in ids]
    overall = stats(base, treat)

    # Compare INITIAL (router) branches — deterministic at temp 0, so this stays
    # ~0% even with the Stage-1 loop on. Post-reroute branch_used legitimately
    # diverges between arms; that movement is shown in the migration block below.
    def _init_branch(r):
        return r.get("initial_branch_used") or r.get("branch_used")
    disagree = np.mean([str(_init_branch(B[i])) != str(_init_branch(T[i])) for i in ids])

    HDR = f"{'stratum':<18}{'n':>5}{'base':>8}{'treat':>8}{'Δ':>8}{'  95% CI':>18}{'b/c':>9}{'McNemar p':>10}"
    print("=" * 92)
    print(f"PAIRED TSExam DIFF  {base_dir}  vs  {treat_dir}   (treat - base)")
    print("=" * 92)
    print(f"paired rows: {len(ids)}   initial-router-branch disagreement: {disagree:.1%}   "
          "(* sig gain, X sig loss; p<0.05 exact McNemar)\n")
    print(HDR); print("-" * 92)
    print(line("OVERALL", overall))
    print("\n-- by category " + "-" * 77)
    print(HDR); print("-" * 92)
    for k, s in grouped(ids, B, T, lambda r: r.get("category")):
        print(line(k, s))
    print("\n-- by initial_branch_used (MECHANISM VIEW, Holm-corrected) " + "-" * 33)
    BHDR = HDR + f"{'Holm p':>10}"
    print(BHDR); print("-" * 102)
    branch_rows = grouped(ids, B, T, _init_branch)
    holm = holm_adjust([s["p"] for _, s in branch_rows])
    for (k, s), padj in zip(branch_rows, holm):
        star = (" *" if s["d"] > 0 else " X") if padj < 0.05 else ""
        bc = f"{s['b']}/{s['c']}"
        print(f"{k[:18]:<18}{s['n']:>5}{s['ab']:>8.3f}{s['at']:>8.3f}{s['d']:>+8.3f}"
              f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{bc:>9}{s['p']:>10.4f}{padj:>10.4f}{star}")
    print_migration(ids, T)
    print_cluster_block(ids, B, T)

    # Vision-gate movement (the whole point of nu_ad_fix)
    def vt(r):
        return bool(r.get("vision_triggered") or r.get("vision_used") or r.get("vision_letter"))
    bv = sum(vt(B[i]) for i in ids); tv = sum(vt(T[i]) for i in ids)
    print(f"\nvision fired: base {bv}/{len(ids)} ({bv/len(ids):.1%})  ->  "
          f"treat {tv}/{len(ids)} ({tv/len(ids):.1%})")
    print("=" * 92)


if __name__ == "__main__":
    main()
