#!/usr/bin/env python3
# scripts/diff_configs.py
"""
Paired config diff for MMTS-Bench — the generalization test's analysis stage.
=============================================================================
Loads two sets of `mmts_results_*` detail CSVs (one per config, e.g. baseline
vs nu_ad_fix), joins them PAIRWISE on (subset, sample_id), and reports — for
the whole set and stratified by subset, category, and `branch_used` — the
per-config OA, the paired delta (treatment - baseline), a bootstrap 95% CI on
that delta, and an exact McNemar test on the discordant pairs.

This is the mechanism view the pre-registration asks for: the PRIMARY
hypothesis is about rows where `branch_used == "noise"` (the only branch the
frozen `no_vision_branches=["noise"]` switch touches on MMTS), the GUARDRAIL is
neutrality on the vision-bypassing Match path and the other categories.

Usage
-----
    python scripts/diff_configs.py \
        --baseline  outputs/mmts_results_*_baseline_*.csv \
        --treatment outputs/mmts_results_*_nu_ad_fix_*.csv

(Shell globs expand to multiple files; all baseline CSVs are concatenated, all
treatment CSVs are concatenated, then joined on (subset, sample_id).)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC = _ROOT.parent / "tsqa"
sys.path.insert(0, str(_AGENTIC))

from tsqa.eval.cluster_diff import cluster_bootstrap_delta, paired_permutation_p  # type: ignore


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _as_bool(series: pd.Series) -> np.ndarray:
    """Map a 'correct' column (True/False/'True'/'False'/1/0) to a 0/1 array."""
    return (
        series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    ).to_numpy().astype(int)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant pair counts (b, c).

    b = baseline correct & treatment wrong ; c = baseline wrong & treatment correct.
    Under H0 the discordants split 50/50, so this is a two-sided binomial test
    of min(b, c) successes out of (b + c) trials at p=0.5.
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import erfc, sqrt
        chi2 = (abs(b - c) - 1) ** 2 / n          # continuity-corrected
        return float(erfc(sqrt(chi2 / 2.0)))


def bootstrap_delta_ci(base: np.ndarray, treat: np.ndarray,
                       n_boot: int = 10000, seed: int = 0):
    """Percentile bootstrap 95% CI on the PAIRED accuracy delta (treat - base).

    Resamples row indices (keeping each pair intact) so the CI reflects the
    paired structure, not two independent samples.
    """
    n = len(base)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = treat[idx].mean(axis=1) - base[idx].mean(axis=1)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def paired_stats(base: np.ndarray, treat: np.ndarray, seed: int = 0) -> dict:
    n = len(base)
    acc_b = float(base.mean()) if n else float("nan")
    acc_t = float(treat.mean()) if n else float("nan")
    b = int(((base == 1) & (treat == 0)).sum())   # baseline right, treat wrong
    c = int(((base == 0) & (treat == 1)).sum())   # baseline wrong, treat right
    lo, hi = bootstrap_delta_ci(base, treat, seed=seed)
    return {
        "n": n, "acc_base": acc_b, "acc_treat": acc_t,
        "delta": acc_t - acc_b, "b": b, "c": c,
        "mcnemar_p": mcnemar_p(b, c), "ci_lo": lo, "ci_hi": hi,
    }


def _holm(pairs):
    """Holm–Bonferroni step-down adjusted p-values (FWER control) for a family of
    per-branch McNemar tests. `pairs` is [(key, raw_p)]; returns {key: adj_p}. Use
    the adjusted p — not the raw — to call per-branch significance on the
    multi-branch sweep; the Stage-1 promotion gate requires it (finding #2)."""
    m = len(pairs)
    if m == 0:
        return {}
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = {}
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pairs[i][1])  # step-down + monotone
        adj[pairs[i][0]] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_many(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        if "checkpoint" in str(p):
            continue   # skip partial checkpoint CSVs; use only final detail files
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)
    if not frames:
        raise SystemExit("No input CSVs given/found (after skipping checkpoints).")
    out = pd.concat(frames, ignore_index=True)
    # Drop rows the harness marked as skipped/crashed — they carry no signal and
    # would otherwise count as a (config-independent) wrong answer on both sides.
    if "error" in out.columns:
        out = out[~out["error"].astype(str).str.strip().isin(["skipped"])]
    return out


def _join_paired(base_df: pd.DataFrame, treat_df: pd.DataFrame) -> pd.DataFrame:
    key = ["subset", "sample_id"] if "subset" in base_df.columns else ["sample_id"]
    for df in (base_df, treat_df):
        df["sample_id"] = df["sample_id"].astype(str)
        if "subset" in df.columns:
            df["subset"] = df["subset"].astype(str)
        # initial_branch_used is the pre-reroute stratifier (finding #1). Runs made
        # before it existed fall back to branch_used (== initial when the Stage-1
        # loop is off), so old CSVs still join and stratify correctly.
        if "initial_branch_used" not in df.columns:
            df["initial_branch_used"] = df.get("branch_used")
        else:
            df["initial_branch_used"] = df["initial_branch_used"].fillna(df["branch_used"])
    b = base_df.drop_duplicates(subset=key).copy()
    t = treat_df.drop_duplicates(subset=key).copy()
    b["ok_base"] = _as_bool(b["correct"])
    t["ok_treat"] = _as_bool(t["correct"])
    merged = b.merge(
        t[key + ["ok_treat", "branch_used", "initial_branch_used", "category"]],
        on=key, how="inner", suffixes=("", "_t"),
    )
    return merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HDR = f"{'stratum':<26}{'n':>6}{'base':>9}{'treat':>9}{'Δ':>9}" \
       f"{'95% CI':>18}{'b/c':>9}{'McNemar p':>12}"


def _row_line(label: str, s: dict) -> str:
    ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
    bc = f"{s['b']}/{s['c']}"
    star = ""
    if s["mcnemar_p"] < 0.05:
        star = " *" if s["delta"] > 0 else " X"
    return (f"{label[:26]:<26}{s['n']:>6}{s['acc_base']:>9.3f}{s['acc_treat']:>9.3f}"
            f"{s['delta']:>+9.3f}{ci:>18}{bc:>9}{s['mcnemar_p']:>12.4f}{star}")


def _grouped(merged: pd.DataFrame, by: str, seed: int) -> list:
    out = []
    for key, g in merged.groupby(by, dropna=False):
        s = paired_stats(g["ok_base"].to_numpy(), g["ok_treat"].to_numpy(), seed=seed)
        out.append((str(key), s))
    # sort by n desc for readability
    return sorted(out, key=lambda kv: kv[1]["n"], reverse=True)


def _print_migration(merged: pd.DataFrame) -> None:
    """Initial->final branch migration in the TREATMENT arm (Stage-1 reroute view).
    Empty when the loop is off (initial == final on every row)."""
    if "initial_branch_used_t" not in merged.columns or "branch_used_t" not in merged.columns:
        return
    ini = merged["initial_branch_used_t"].astype(str)
    fin = merged["branch_used_t"].astype(str)
    moved = int((ini != fin).sum())
    if moved == 0:
        print("\n-- branch migration (treatment): none — initial == final on all rows")
        return
    print(f"\n-- branch migration initial->final (treatment), {moved}/{len(merged)} rows moved "
          + "-" * 18)
    ct = pd.crosstab(ini, fin)
    for i in ct.index:
        moved_to = [(j, int(ct.loc[i, j])) for j in ct.columns if ct.loc[i, j] > 0 and j != i]
        if not moved_to:
            continue
        dist = ", ".join(f"{j}:{n}" for j, n in sorted(moved_to, key=lambda kv: -kv[1]))
        print(f"  {i:<16} -> {dist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="baseline detail CSV(s) (mmts_results_*_baseline_*.csv)")
    ap.add_argument("--treatment", nargs="+", required=True,
                    help="treatment detail CSV(s) (mmts_results_*_nu_ad_fix_*.csv)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary-branch", default="noise",
                    help="branch_used value the PRIMARY hypothesis is keyed on")
    ap.add_argument("--include-unscoreable", action="store_true",
                    help="Keep rows whose gold is not a valid A/B/C/D letter "
                         "(free-response numeric/stationarity items). Default: "
                         "drop them — an MCQ system cannot answer them and they "
                         "are wrong in every config (concordant, uninformative).")
    ap.add_argument("--cluster-by", default=None,
                    help="Optional column for cluster bootstrap/permutation sensitivity "
                         "(e.g. domain, category, source_id). Default: disabled.")
    args = ap.parse_args()

    base_df  = _load_many(args.baseline)
    treat_df = _load_many(args.treatment)
    merged   = _join_paired(base_df, treat_df)

    if merged.empty:
        raise SystemExit("No shared (subset, sample_id) rows between the two configs.")

    n_all = len(merged)
    if not args.include_unscoreable:
        valid = (merged["ground_truth"].astype(str).str.strip().str.upper()
                 .isin(list("ABCD")))
        dropped = int((~valid).sum())
        merged  = merged[valid].copy()
        if merged.empty:
            raise SystemExit("All paired rows are unscoreable (no valid gold letter).")
        print(f"\n[scoreable-only] dropped {dropped}/{n_all} free-response rows "
              f"(no A/B/C/D gold); keeping {len(merged)} MCQ rows. "
              "Use --include-unscoreable to keep them.")

    # Diagnostic: how often did the router land on a different branch across the
    # two runs (pure model nondeterminism — the config never changes routing).
    # Compare INITIAL (router) branches — deterministic at temp 0, so ~0% even with
    # the Stage-1 loop on. Post-reroute branch_used legitimately diverges; that
    # movement is shown in the migration block below the mechanism view.
    if "initial_branch_used" in merged.columns and "initial_branch_used_t" in merged.columns:
        disagree = (merged["initial_branch_used"].astype(str)
                    != merged["initial_branch_used_t"].astype(str)).mean()
    else:
        disagree = float("nan")

    overall = paired_stats(merged["ok_base"].to_numpy(),
                           merged["ok_treat"].to_numpy(), seed=args.seed)

    print("\n" + "=" * 96)
    print("PAIRED CONFIG DIFF  —  baseline  vs  treatment   (treat - base)")
    print("=" * 96)
    print(f"paired rows: {len(merged)}   router-branch disagreement across runs: "
          f"{disagree:.1%}   (* = sig. gain, X = sig. loss; p<0.05 exact McNemar)")
    print("\n" + _HDR)
    print("-" * 96)
    print(_row_line("OVERALL", overall))

    print("\n-- by subset " + "-" * 83)
    print(_HDR)
    print("-" * 96)
    for label, s in _grouped(merged, "subset", args.seed):
        print(_row_line(label, s))

    print("\n-- by initial_branch_used (MECHANISM VIEW, Holm-corrected) " + "-" * 38)
    print(_HDR + f"{'Holm p':>12}")
    print("-" * 108)
    branch_rows = _grouped(merged, "initial_branch_used", args.seed)
    holm = _holm([(lbl, s["mcnemar_p"]) for lbl, s in branch_rows])
    for label, s in branch_rows:
        padj = holm.get(label, 1.0)
        star = (" *" if s["delta"] > 0 else " X") if padj < 0.05 else ""
        ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
        bc = f"{s['b']}/{s['c']}"
        print(f"{label[:26]:<26}{s['n']:>6}{s['acc_base']:>9.3f}{s['acc_treat']:>9.3f}"
              f"{s['delta']:>+9.3f}{ci:>18}{bc:>9}{s['mcnemar_p']:>12.4f}{padj:>12.4f}{star}")
    _print_migration(merged)

    print("\n-- by category " + "-" * 81)
    print(_HDR)
    print("-" * 96)
    for label, s in _grouped(merged, "category", args.seed):
        print(_row_line(label, s))

    # ── Pre-registered verdict ───────────────────────────────────────────
    print("\n" + "=" * 96)
    print("PRE-REGISTERED VERDICT")
    print("=" * 96)

    pb = args.primary_branch
    # Key the primary on the pre-reroute branch (collider-free; finding #1). With
    # the Stage-1 loop off, initial_branch_used == branch_used, so the frozen
    # nu_ad_fix verdict is unchanged.
    pmask = merged["initial_branch_used"].astype(str) == pb
    if pmask.any():
        ps = paired_stats(merged.loc[pmask, "ok_base"].to_numpy(),
                          merged.loc[pmask, "ok_treat"].to_numpy(), seed=args.seed)
        regressed = (ps["delta"] < 0) and (ps["ci_hi"] < 0)     # sig. negative
        verdict = ("REGRESSED (falsifies)" if regressed
                   else "SUPPORTED — non-negative / within CI")
        print(f"PRIMARY  (initial_branch_used == '{pb}', n={ps['n']}): "
              f"Δ={ps['delta']:+.3f}  CI=[{ps['ci_lo']:+.3f},{ps['ci_hi']:+.3f}]  "
              f"McNemar p={ps['mcnemar_p']:.4f}  ->  {verdict}")
    else:
        print(f"PRIMARY  (initial_branch_used == '{pb}'): NO ROWS routed to '{pb}'. "
              "The lever acts on ~0 rows here — report as inapplicable, not refuted.")

    # GUARDRAIL: Match neutral + nothing significantly regressed
    regressions = []
    for label, s in _grouped(merged, "category", args.seed):
        if s["delta"] < 0 and s["ci_hi"] < 0 and s["mcnemar_p"] < 0.05:
            regressions.append((label, s))
    print(f"GUARDRAIL (neutrality): "
          + ("no category significantly regressed."
             if not regressions else
             "REGRESSIONS -> " + ", ".join(
                 f"{l} ({s['delta']:+.3f}, p={s['mcnemar_p']:.3f})" for l, s in regressions)))

    print(f"SECONDARY (headline OA): baseline {overall['acc_base']:.4f} -> "
          f"treatment {overall['acc_treat']:.4f}  (Δ {overall['delta']:+.4f}, "
          f"McNemar p={overall['mcnemar_p']:.4f})")
    if args.cluster_by:
        if args.cluster_by not in merged.columns:
            print(f"CLUSTER ({args.cluster_by}): column not present; skipped.")
        else:
            clusters = merged[args.cluster_by].fillna("missing").astype(str).to_numpy()
            cd = cluster_bootstrap_delta(
                merged["ok_base"].to_numpy(),
                merged["ok_treat"].to_numpy(),
                clusters,
                seed=args.seed,
            )
            pp = paired_permutation_p(
                merged["ok_base"].to_numpy(),
                merged["ok_treat"].to_numpy(),
                seed=args.seed,
            )
            worst = sorted(
                ((k, v) for k, v in cd.sensitivity.items() if v is not None),
                key=lambda kv: kv[1],
            )[:3]
            worst_txt = ", ".join(f"drop {k}: Δ={v:+.3f}" for k, v in worst) or "n/a"
            print(f"CLUSTER ({args.cluster_by}): Δ={cd.delta:+.4f} "
                  f"CI=[{cd.ci_lo:+.4f},{cd.ci_hi:+.4f}] "
                  f"paired permutation p={pp:.4f}; sensitivity: {worst_txt}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
