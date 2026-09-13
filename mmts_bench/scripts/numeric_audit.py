#!/usr/bin/env python3
# scripts/numeric_audit.py
"""
Numeric-head audit (Workstream B verification).
===============================================
Two reports against `mmts_results_*` CSV(s) produced with the numeric head active
(a DEFAULT run, i.e. WITHOUT --mcq-only):

  1. SCHEMA DETECTION — confusion of the structural `expected_schema` against the
     dataset's independent `qa_type` label (the pre-registered "schema-detection
     accuracy on a labelled Base slice", reported BEFORE any OA number).

  2. NUMERIC ACCURACY — on `expected_schema == numerical` rows: Accuracy@10%
     (the pinned MMTS-Bench metric, the `correct` column), mean Relative Accuracy
     (`rel_acc`), and abstention rate, STRATIFIED by quantity class
     (closed_form / method_sensitive / regression) and per quantity. Quantity is
     read from the `numeric_quantity` column if present, else re-derived from the
     question via the same deterministic inference the runner uses (joined from
     the source data by (subset, sample_id)).

Usage
-----
    python scripts/numeric_audit.py \
        --results outputs/.../mmts_results_Base_baseline_*.csv \
        [--data-path ./MMTS-BENCH]
"""
import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC = _ROOT.parent / "tsqa"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_AGENTIC))

from tsqa.eval.numeric_head import (  # noqa: E402
    infer_numeric_quantity, _CLOSED_FORM, _METHOD_SENSITIVE, _REGRESSION,
)


def _as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"}).to_numpy()


def _qclass(q):
    if q in _CLOSED_FORM:
        return "closed_form"
    if q in _METHOD_SENSITIVE:
        return "method_sensitive"
    if q in _REGRESSION:
        return "regression"
    return "abstain/none"


def _load(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No CSVs matched: {paths}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    if "error" in df.columns:
        df = df[~df["error"].astype(str).str.strip().isin(["skipped"])]
    return df


def _query_map(data_path, subsets):
    """(subset, sample_id) -> question text, from the source data via the adapter."""
    from eval.dataloaders import MMTSBenchAdapter
    qm = {}
    for sub in subsets:
        try:
            ad = MMTSBenchAdapter(data_path, subset=sub)
        except Exception as e:
            print(f"  (warn: could not load subset {sub}: {e})")
            continue
        for s in ad:
            qm[(str(s["subset"]), str(s["sample_id"]))] = s["query"]
    return qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--data-path", default="./MMTS-BENCH")
    args = ap.parse_args()

    df = _load(args.results)
    df["sample_id"] = df["sample_id"].astype(str)
    if "subset" in df.columns:
        df["subset"] = df["subset"].astype(str)
    print(f"\nloaded {len(df)} non-skipped rows from {len(args.results)} pattern(s)")

    # ── 1. Schema detection vs qa_type ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("SCHEMA DETECTION  —  expected_schema (structural)  ×  qa_type (label)")
    print("=" * 78)
    if "qa_type" in df.columns and df["qa_type"].astype(str).str.strip().any():
        ct = pd.crosstab(df["expected_schema"].astype(str),
                         df["qa_type"].astype(str), dropna=False)
        print(ct.to_string())
        # accuracy: numerical<->numerical ; mcq<->{multiple/binary choice}
        def _true_schema(qa):
            qa = str(qa).lower()
            if "numerical" in qa:
                return "numerical"
            if "choice" in qa:
                return "mcq"
            return "other"
        df["_true"] = df["qa_type"].map(_true_schema)
        mask = df["_true"].isin(["numerical", "mcq"])
        agree = (df.loc[mask, "expected_schema"].astype(str) == df.loc[mask, "_true"]).mean()
        print(f"\nschema-detection accuracy (numerical/mcq labelled rows, n={int(mask.sum())}): "
              f"{agree:.4f}")
    else:
        print("  (no qa_type column — re-run with the updated logging to audit schema detection)")

    # ── 2. Numeric accuracy stratified by quantity class ────────────────────
    print("\n" + "=" * 78)
    print("NUMERIC HEAD ACCURACY  —  expected_schema == 'numerical'")
    print("=" * 78)
    num = df[df["expected_schema"].astype(str) == "numerical"].copy()
    if num.empty:
        print("  (no numerical rows in these results)")
        return

    # quantity: prefer logged column, else re-derive from the source question
    has_q = "numeric_quantity" in num.columns and num["numeric_quantity"].notna().any()
    if not has_q:
        subsets = sorted(num["subset"].unique()) if "subset" in num.columns else ["Base"]
        qm = _query_map(args.data_path, subsets)
        def _q(r):
            ques = qm.get((str(r.get("subset", "Base")), str(r["sample_id"])))
            spec = infer_numeric_quantity(ques) if ques else None
            return spec["quantity"] if spec else None
        num["numeric_quantity"] = num.apply(_q, axis=1)

    num["ok"] = _as_bool(num["correct"])
    num["abstained"] = num["predicted_value"].isna() | (num["predicted_value"].astype(str).str.strip() == "")
    num["rel_acc_f"] = pd.to_numeric(num["rel_acc"], errors="coerce")
    num["qclass"] = num["numeric_quantity"].map(_qclass)

    def _block(group_col, title):
        print(f"\n-- by {title} " + "-" * (60 - len(title)))
        print(f"  {'stratum':<22}{'n':>6}{'acc@10%':>9}{'rel_acc':>9}{'abstain':>9}{'acc|attempt':>13}")
        print("  " + "-" * 66)
        rows = []
        for key, g in num.groupby(group_col, dropna=False):
            n = len(g)
            acc = g["ok"].mean()
            rel = g["rel_acc_f"].mean()
            ab = g["abstained"].mean()
            att = g[~g["abstained"]]
            acc_att = att["ok"].mean() if len(att) else float("nan")
            rows.append((str(key), n, acc, rel, ab, acc_att))
        for key, n, acc, rel, ab, acc_att in sorted(rows, key=lambda x: -x[1]):
            print(f"  {key[:22]:<22}{n:>6}{acc:>9.3f}{rel:>9.3f}{ab:>9.2%}{acc_att:>13.3f}")

    n = len(num)
    print(f"\nOVERALL numerical rows: n={n}  acc@10%={num['ok'].mean():.3f}  "
          f"mean rel_acc={num['rel_acc_f'].mean():.3f}  abstain={num['abstained'].mean():.2%}")
    _block("qclass", "quantity class")
    _block("numeric_quantity", "quantity")
    print()


if __name__ == "__main__":
    main()
