#!/usr/bin/env python3
# scripts/diff_numeric.py
"""
Paired NUMERIC diff — deterministic numeric HEAD vs the llm_numeric counterfactual.
====================================================================================
The free-response load-bearing proof's analysis stage. Joins a TOOL detail CSV
(numeric head; predicted_value = a tool-computed number) and a MODEL detail CSV
(`--config llm_numeric`; predicted_value = a number the LLM computed from the raw
series) on `sample_id` over the NUMERICAL rows (expected_schema=='numerical'),
scores BOTH arms by MMTS-Bench Accuracy@N% from predicted_value vs ground_truth,
and reports — overall, per `numeric_quantity`, and for the CLOSED-FORM (primary)
vs METHOD-SENSITIVE (bounds) strata — each arm's @10%, the paired Δ (tool − model),
a bootstrap 95% CI, an exact McNemar test, mean Relative Accuracy, the model's
abstention rate, and a {1,5,10,20}% sensitivity sweep.

A large positive Δ on the closed-form stratum is the pre-registered "tools are
load-bearing on a strong backbone" result (unlike MCQ, this should NOT tie).
Pre-registration: mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md.

Usage
-----
    python scripts/diff_numeric.py \
        --tool  outputs/mmts_results_Base_baseline_*.csv \
        --model outputs/mmts_results_Base_llm_numeric_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC = _ROOT.parent / "tsqa"
sys.path.insert(0, str(_AGENTIC))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for the diff_configs import

from diff_configs import mcnemar_p, bootstrap_delta_ci      # reuse the tested stats
from tsqa.eval.scoring import score_numeric_value           # the MMTS @N% scorer

# Quantity classes (pre-registered strata). Closed-form = exact computations the
# tool owns; method-sensitive = definition-dependent (reported as bounds, never
# folded into the headline).
CLOSED_FORM = {"std", "var", "mean", "median", "min", "max", "range",
               "percentile", "argmax", "argmin"}
METHOD_SENSITIVE = {"period", "count_local_max", "count_local_min", "slope"}

_ABSTAIN_NOTES = {"llm_unparseable", "llm_error", "no_quantity_match"}


def _load(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        if "checkpoint" in str(p):
            continue
        frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        raise SystemExit("No input CSVs given/found (after skipping checkpoints).")
    return pd.concat(frames, ignore_index=True)


def _score_col(preds, golds, tol):
    """Vectorized Accuracy@tol + Relative Accuracy from predicted_value vs gold."""
    ok = np.zeros(len(preds), dtype=int)
    rel = np.zeros(len(preds), dtype=float)
    for i, (p, g) in enumerate(zip(preds, golds)):
        hit, ra = score_numeric_value(p, g, rel_tol=tol)
        ok[i] = 1 if hit else 0
        rel[i] = ra if ra is not None else 0.0
    return ok, rel


def _pstats(tool_ok, model_ok, seed=0):
    n = len(tool_ok)
    b = int(((tool_ok == 1) & (model_ok == 0)).sum())   # tool right, model wrong
    c = int(((tool_ok == 0) & (model_ok == 1)).sum())   # tool wrong, model right
    lo, hi = bootstrap_delta_ci(model_ok, tool_ok, seed=seed)   # CI on (tool − model)
    return {
        "n": n,
        "acc_tool": float(tool_ok.mean()) if n else float("nan"),
        "acc_model": float(model_ok.mean()) if n else float("nan"),
        "delta": (float(tool_ok.mean()) - float(model_ok.mean())) if n else float("nan"),
        "b": b, "c": c, "mcnemar_p": mcnemar_p(b, c), "ci_lo": lo, "ci_hi": hi,
    }


_HDR = (f"{'stratum':<20}{'n':>5}{'tool@10':>9}{'model@10':>10}{'Δ(t-m)':>9}"
        f"{'95% CI':>18}{'b/c':>9}{'McNemar p':>12}")


def _line(label, s):
    ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
    bc = f"{s['b']}/{s['c']}"
    star = ""
    if s["mcnemar_p"] < 0.05:
        star = " *" if s["delta"] > 0 else " X"
    return (f"{label[:20]:<20}{s['n']:>5}{s['acc_tool']:>9.3f}{s['acc_model']:>10.3f}"
            f"{s['delta']:>+9.3f}{ci:>18}{bc:>9}{s['mcnemar_p']:>12.4f}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", nargs="+", required=True,
                    help="TOOL detail CSV(s) — numeric head (e.g. *_baseline_*.csv)")
    ap.add_argument("--model", nargs="+", required=True,
                    help="MODEL detail CSV(s) — *_llm_numeric_*.csv")
    ap.add_argument("--tol", type=float, default=0.10, help="headline rel-tol (default 0.10)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tdf, mdf = _load(args.tool), _load(args.model)
    for df in (tdf, mdf):
        df["sample_id"] = df["sample_id"].astype(str)

    # NUMERICAL rows only, using the TOOL arm's schema label as authoritative.
    tdf = tdf[tdf["expected_schema"].astype(str) == "numerical"].drop_duplicates("sample_id")
    mdf = mdf.drop_duplicates("sample_id")

    t = tdf[["sample_id", "ground_truth", "numeric_quantity",
             "predicted_value", "numeric_note"]].rename(
        columns={"predicted_value": "pred_tool", "numeric_note": "note_tool"})
    m = mdf[["sample_id", "predicted_value", "numeric_note"]].rename(
        columns={"predicted_value": "pred_model", "numeric_note": "note_model"})
    j = t.merge(m, on="sample_id", how="inner")
    if j.empty:
        raise SystemExit("No shared numerical sample_id rows between tool and model CSVs.")

    golds = j["ground_truth"].tolist()
    tool_ok, tool_rel = _score_col(j["pred_tool"].tolist(), golds, args.tol)
    model_ok, model_rel = _score_col(j["pred_model"].tolist(), golds, args.tol)
    j["tool_ok"], j["model_ok"] = tool_ok, model_ok
    j["qty"] = j["numeric_quantity"].fillna("(abstain)").astype(str).replace("", "(abstain)")
    j["klass"] = j["qty"].map(
        lambda q: "closed_form" if q in CLOSED_FORM
        else ("method_sensitive" if q in METHOD_SENSITIVE else "other"))

    overall = _pstats(tool_ok, model_ok, seed=args.seed)
    model_abstain = float(j["note_model"].astype(str).isin(_ABSTAIN_NOTES).mean())

    print("\n" + "=" * 100)
    print("PAIRED NUMERIC DIFF  —  numeric HEAD (tool)  vs  llm_numeric (model)   "
          "Δ = tool − model")
    print("=" * 100)
    print(f"paired numerical rows: {len(j)}   "
          f"mean RelAcc: tool {tool_rel.mean():.3f} / model {model_rel.mean():.3f}   "
          f"model abstention: {model_abstain:.1%}   (* = tool wins, X = model wins; p<0.05)")
    print("\n" + _HDR)
    print("-" * 100)
    print(_line("OVERALL", overall))

    # PRIMARY (closed-form) and BOUNDS (method-sensitive) strata.
    print("\n-- by quantity CLASS " + "-" * 78)
    print(_HDR)
    print("-" * 100)
    for klass in ("closed_form", "method_sensitive", "other"):
        g = j[j["klass"] == klass]
        if len(g):
            print(_line(klass, _pstats(g["tool_ok"].to_numpy(),
                                       g["model_ok"].to_numpy(), seed=args.seed)))

    print("\n-- by numeric_quantity " + "-" * 76)
    print(_HDR)
    print("-" * 100)
    rows = []
    for q, g in j.groupby("qty"):
        rows.append((q, _pstats(g["tool_ok"].to_numpy(), g["model_ok"].to_numpy(), seed=args.seed)))
    for q, s in sorted(rows, key=lambda kv: kv[1]["n"], reverse=True):
        print(_line(q, s))

    # Sensitivity sweep on the closed-form stratum (the headline).
    cf = j[j["klass"] == "closed_form"]
    print(f"\n-- Accuracy@N% sensitivity (closed-form stratum, n={len(cf)}) " + "-" * 30)
    print(f"{'N%':>5}{'tool':>9}{'model':>9}{'Δ(t-m)':>9}")
    for N in (0.01, 0.05, 0.10, 0.20):
        tok, _ = _score_col(cf["pred_tool"].tolist(), cf["ground_truth"].tolist(), N)
        mok, _ = _score_col(cf["pred_model"].tolist(), cf["ground_truth"].tolist(), N)
        d = (tok.mean() - mok.mean()) if len(cf) else float("nan")
        print(f"{int(N*100):>5}{tok.mean():>9.3f}{mok.mean():>9.3f}{d:>+9.3f}")

    # ── Pre-registered verdict ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT")
    print("=" * 100)
    cfs = _pstats(cf["tool_ok"].to_numpy(), cf["model_ok"].to_numpy(), seed=args.seed)
    supported = (cfs["delta"] > 0) and (cfs["ci_lo"] > 0) and (cfs["mcnemar_p"] < 0.05)
    verdict = ("SUPPORTED — tool is load-bearing (gap > 0, CI excludes 0, p<0.05)"
               if supported else
               "NOT supported — closed-form gap is not significantly > 0 "
               "(model computes these itself ⇒ thesis falsified on this surface)")
    print(f"PRIMARY (closed-form, n={cfs['n']}): tool {cfs['acc_tool']:.3f} vs "
          f"model {cfs['acc_model']:.3f}  Δ={cfs['delta']:+.3f}  "
          f"CI=[{cfs['ci_lo']:+.3f},{cfs['ci_hi']:+.3f}]  McNemar p={cfs['mcnemar_p']:.4f}")
    print("  ->  " + verdict)
    print("BOUNDS (method-sensitive) reported above as a separate stratum; the tool is itself")
    print("  weaker there (definition match), so the model may tie or win — not part of the headline.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
