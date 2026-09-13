#!/usr/bin/env python3
# scripts/run_numeric_stress.py
"""
Three-arm numeric-PARSER evaluation (A0 keyword / A1 deterministic / A2 llm_fallback).
=====================================================================================
Runs the numeric-head parser path directly (no router/branch) on an input CSV with
columns ``sample_id, layer, question, ground_truth, series`` and reports, per layer
and overall: Accuracy@10% for each arm, the paired A1−A0 and **A2−A1** deltas
(McNemar + bootstrap CI), and the A2 LLM-fire-rate. The gold was computed by the
SAME tools, so this measures only PARSING. A0/A1 make ZERO LLM calls; A2 calls the
gemini planner ONLY on rows where A1 abstains. log/012, pre-reg
PREREGISTRATION_numeric_parser.md.

Usage
-----
    python scripts/run_numeric_stress.py --input outputs/numeric_stress.csv   # E2
    python scripts/run_numeric_stress.py --input outputs/base_numerical.csv   # E1
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC = _ROOT.parent / "tsqa"
sys.path.insert(0, str(_AGENTIC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT.parent / ".env")
except Exception:
    pass

from tsqa.eval.runner import _select_numeric_plan          # the arm dispatcher
from tsqa.eval.numeric_head import evaluate_plan
from tsqa.eval.scoring import score_numeric_value
from tsqa.llm.factory import create_llm_client
from diff_configs import mcnemar_p, bootstrap_delta_ci      # reuse tested stats

ARMS = [("A0", "keyword"), ("A1", "deterministic"), ("A2", "llm_fallback")]


def _score_row(question, gold, ts, parser, client):
    try:
        plan, _, llm_fired = _select_numeric_plan(question, parser, client)
        if plan is None:
            return 0, llm_fired
        val, _ = evaluate_plan(plan, ts, None, None, {})
        if val is None:
            return 0, llm_fired
        ok, _ = score_numeric_value(val, gold, rel_tol=0.10)
        return (1 if ok else 0), llm_fired
    except Exception:
        return 0, False   # one bad row never nukes the batch (e.g. a 2-D series)


def _delta(a, b, seed=0):
    """Paired stats for (b − a): n, acc_a, acc_b, delta, McNemar, CI."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    bb = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    cc = int(((a == 0) & (b == 1)).sum())   # a wrong, b right
    lo, hi = bootstrap_delta_ci(a, b, seed=seed)
    return {"n": n, "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "delta": float(b.mean() - a.mean()), "b": bb, "c": cc,
            "mcnemar_p": mcnemar_p(bb, cc), "ci_lo": lo, "ci_hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    client = create_llm_client(provider="gemini")

    # results[arm_label] = {sample_id: (ok, llm_fired)}; layers kept per row
    results = {lbl: {} for lbl, _ in ARMS}
    layer_of = {}

    def work(r):
        sid = r["sample_id"]
        layer_of[sid] = r.get("layer", "all")
        ts = json.loads(r["series"])
        gold = float(r["ground_truth"])
        out = {}
        for lbl, parser in ARMS:
            out[lbl] = _score_row(r["question"], gold, ts, parser, client)
        return sid, out

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for sid, out in ex.map(work, rows):
            for lbl, _ in ARMS:
                results[lbl][sid] = out[lbl]

    layers = sorted(set(layer_of.values()))
    order = [L for L in ("clean", "base", "compositional", "typo", "paraphrase") if L in layers]
    order += [L for L in layers if L not in order]

    print("\n" + "=" * 92)
    print(f"THREE-ARM NUMERIC PARSER  —  {args.input}   (Accuracy@10%; gold = tools' own answer)")
    print("=" * 92)
    print(f"{'layer':14}{'n':>5}{'A0':>7}{'A1':>7}{'A2':>7}"
          f"{'A1-A0':>9}{'A2-A1':>9}{'A2 p':>9}{'A2 fire%':>10}")
    print("-" * 92)

    def slice_arm(lbl, sids):
        return [results[lbl][s][0] for s in sids]

    for layer in order + ["OVERALL"]:
        sids = ([s for s in layer_of if layer_of[s] == layer] if layer != "OVERALL"
                else list(layer_of))
        if not sids:
            continue
        a0, a1, a2 = slice_arm("A0", sids), slice_arm("A1", sids), slice_arm("A2", sids)
        d10 = _delta(a0, a1, args.seed)     # A1 − A0
        d21 = _delta(a1, a2, args.seed)     # A2 − A1 (the binding number)
        fire = float(np.mean([1 if results["A2"][s][1] else 0 for s in sids]))
        star = " *" if (d21["mcnemar_p"] < 0.05 and d21["delta"] > 0) else ""
        print(f"{layer:14}{len(sids):>5}{np.mean(a0):>7.2f}{np.mean(a1):>7.2f}{np.mean(a2):>7.2f}"
              f"{d10['delta']:>+9.2f}{d21['delta']:>+9.2f}{d21['mcnemar_p']:>9.4f}{fire:>9.0%}{star}")

    # ── Pre-registered verdict (binding: A2 − A1 on the paraphrase layer) ──
    para = [s for s in layer_of if layer_of[s] == "paraphrase"]
    print("\n" + "=" * 92)
    if para:
        d = _delta(slice_arm("A1", para), slice_arm("A2", para), args.seed)
        fire = float(np.mean([1 if results["A2"][s][1] else 0 for s in para]))
        supported = d["delta"] > 0 and d["mcnemar_p"] < 0.05 and fire > 0
        print("PRE-REGISTERED VERDICT (paraphrase layer):")
        print(f"  A2−A1 = {d['delta']:+.3f}  CI=[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]  "
              f"McNemar p={d['mcnemar_p']:.4f}  A2 fire-rate={fire:.0%}  (n={d['n']})")
        print("  -> " + ("SUPPORTED — the LLM-planner is load-bearing for parsing (open paraphrase)"
                         if supported else
                         "NOT supported — LLM redundant over a good deterministic parser even here"))
    else:
        print("(no paraphrase layer in this input — non-regression / E1 run)")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
