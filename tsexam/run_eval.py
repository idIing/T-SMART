"""
T-SMART research harness — one script, every experiment.
=========================================================

Replaces the category-broken validate_v2.py. Maps TimeSeriesExam1's full
category names to the paper's 2-letter codes, samples a *fixed paired* eval set
(reused across configs so deltas are meaningful), runs a named config, and
reports per-category accuracy with Wilson confidence intervals plus deltas vs
the baseline and vs TS-Agent.

Configs are pure architectural switches into run_dataset — NO per-failure prompt
nudging (that is overfitting). Available:

    baseline        single-branch, no vision forcing            (paper default)
    multi_branch    orchestrator: 3 candidates + quality gate + critic
    sa_vision       enable_sa_vision (sighted SA comparison image)
    mb_sa           multi_branch + sa_vision
    <free flags>    --multi-branch / --sa-vision / --sa-mode ... compose your own

Usage (from the repo root):
    python3 tsexam/run_eval.py --config baseline   --n-per-cat 30 --tag baseline
    python3 tsexam/run_eval.py --config multi_branch --n-per-cat 30 --tag mb
    python3 tsexam/run_eval.py --config baseline   --full --tag baseline_full
"""
import argparse
import json
import math
import os
import sys
import time
import datetime
from collections import defaultdict

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tsqa"))  # engine package dir (also pip-installable)
load_dotenv(os.path.join(REPO, ".env"))

from tsqa.eval.runner import run_dataset  # noqa: E402

OUT_ROOT = os.path.join(HERE, "outputs")

# Full TimeSeriesExam1 category name -> paper 2-letter code (note dataset typo).
CAT_MAP = {
    "Pattern Recognition": "PR",
    "Noise Understanding": "NU",
    "Anolmaly Detection": "AD",
    "Anomaly Detection": "AD",
    "Similarity Analysis": "SA",
    "Causality Analysis": "CA",
}
CATEGORIES = ["PR", "NU", "AD", "SA", "CA"]

# Table I reference rows (paper / leaderboard), as fractions. OA = macro mean.
REF = {
    "TS-Agent":          {"PR": 0.71, "NU": 0.61, "AD": 0.57, "SA": 0.57, "CA": 0.55, "OA": 0.602},
    "T-SMART(paper2.5)": {"PR": 0.63, "NU": 0.726, "AD": 0.472, "SA": 0.528, "CA": 0.667, "OA": 0.605},
}

# Named architectural configs -> run_dataset kwargs. No prompt nudging here.
CONFIGS = {
    "baseline":     dict(multi_branch=False, enable_sa_vision=False),
    # No-architecture control (Track-A de-confound): skip router/branches/vision
    # and ask the answer LLM the MCQ directly — one call, no tool evidence. The
    # raw-model baseline against which the deterministic architecture's lift is
    # measured at a fixed backbone. Not byte-identical to anything (deliberate
    # ablation arm); the answer-letter parser is shared with every other config.
    "llm_only":     dict(llm_only=True),
    # Free-response COUNTERFACTUAL (Track-A2 / load-bearing proof): on numerical
    # rows ask the answer LLM to COMPUTE the number directly from the raw series
    # (no tool, no options, no evidence); the deterministic numeric head is BYPASSED.
    # Paired against the tool-owned numeric head on the SAME rows, the @10% gap is the
    # "tools are load-bearing on a strong backbone" measurement
    # (mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md). MCQ rows fall back to
    # the llm_only letter path, so the arm is well-defined on every row.
    "llm_numeric":  dict(llm_numeric=True),
    # Numeric PARSER arms (three-arm parsing study, log/012). All three feed the
    # SAME tools via evaluate_plan; only the question→plan parser differs. A0=keyword
    # is the default head; A1=deterministic adds a composition grammar + fuzzy/synonym/
    # typo matching; A2=llm_fallback adds an LLM-planner (closed registry, emits a PLAN
    # never a number) fired only when A1 abstains.
    "numeric_det":  dict(numeric_parser="deterministic"),
    "numeric_llm":  dict(numeric_parser="llm_fallback"),
    # Vision suppressed on every branch → the "math position" counterfactual
    # (A2). Paired against baseline, the disagreements feed visual_trust.py.
    "vision_off":   dict(multi_branch=False, no_vision_branches=[
                         "trend", "periodicity", "anomaly", "noise",
                         "similarity", "causality"]),
    # raw_pixel_vision — the MEASUREMENT-ONLY arm of the Track-V structured-vs-pixel
    # study. Identical to `baseline` (same additive vision gate ⇒ vision fires on the
    # SAME rows) EXCEPT that on vision-fired rows the rendered PNG is attached DIRECTLY
    # to the answer LLM (bypassing analyze_image's JSON topology extraction). This
    # deliberately VIOLATES the Structured-Vision Invariant (SVI) to measure the raw
    # multimodal ceiling vs the structured sensor. It is NEVER promoted/shipped — SVI
    # stays the default everywhere; this config exists solely to produce the
    # structured-vs-pixel comparison (PREREGISTRATION_structured_vs_pixel.md). With the
    # switch off the pipeline is byte-identical to baseline (proven by
    # tests/test_runner_v2.py::TestRawPixelVision).
    "raw_pixel_vision": dict(multi_branch=False, enable_sa_vision=False,
                             raw_pixel_vision=True),
    "multi_branch": dict(multi_branch=True,  enable_sa_vision=False),
    "sa_vision":    dict(multi_branch=False, enable_sa_vision=True, sa_viz_mode="stacked"),
    "ad_vision":    dict(multi_branch=False, forced_vision_branches=["anomaly"]),
    # NU collapsed because vision over-fires on the (statistical) noise branch and
    # the LLM follows wrong visual suggestions. Suppress vision there.
    "nu_fix":       dict(multi_branch=False, no_vision_branches=["noise"]),
    # AD answered blind / with wrong viz. Force a line plot for anomaly.
    "ad_fix":       dict(multi_branch=False, forced_vision_branches=["anomaly"]),
    # Both fixes — disjoint branches (noise vs anomaly); PR/SA/CA untouched.
    "nu_ad_fix":    dict(multi_branch=False, no_vision_branches=["noise"],
                         forced_vision_branches=["anomaly"]),
    # Track-A2 surgical arm: vision_off on the five non-anomaly branches AND forced
    # on for anomaly — the single-variable contrast vs `vision_off` that isolates the
    # AD-vision lever on a weak backbone (PREREGISTRATION_c1_gpt4omini_vision.md).
    "ad_vision_only": dict(multi_branch=False,
                           no_vision_branches=["trend", "periodicity", "noise",
                                               "similarity", "causality"],
                           forced_vision_branches=["anomaly"]),
    "combined":     dict(multi_branch=True,  enable_sa_vision=True, sa_viz_mode="stacked",
                         forced_vision_branches=["anomaly"], no_vision_branches=["noise"]),
    # The learned "should-I-look" gate (#1): replaces the additive trigger with the
    # pre-registered per-branch suppression policy. Paired against `baseline`
    # (vision_gate="additive") on held-out TSExam — the surface it was NOT fit on.
    "learned_gate": dict(multi_branch=False, enable_sa_vision=False, vision_gate="learned"),
    "react_stage1_evidence_retry": dict(
        loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        max_loop_depth=1,
        loop_on_flags=["evidence_incomplete"],
        loop_mode="reroute_once",
    ),
    # Self-Consistency control arm (Stage-1 constraint #4; Wang et al. 2203.11171).
    # The cheaper lever the Stage-1 loop must beat: on the SAME trigger as the loop
    # (the evidence_incomplete flag), DON'T reroute — sample the answer LLM k=5
    # times at temp 0.7 and majority-vote the letter (deterministic alphabetic
    # tie-break). On non-flagged rows it is byte-identical to baseline. Mirrored
    # byte-for-byte into MMTS_CONFIGS and tsr_bench CONFIGS.
    "self_consistency": dict(
        self_consistency=5,
        self_consistency_temperature=0.7,
    ),
    # Stage-2 iterative tool refinement (research_journal/02_react_stages.md §Stage-2).
    # When the PRIMARY branch's deterministic evidence scores below the pre-existing
    # principled quality gate (QUALITY_THRESHOLD=0.55), ACCUMULATE a 2nd candidate
    # branch's deterministic evidence (keyword-selected, NO LLM) into the evidence
    # dict — depth bounded by max_loop_depth (=2 ⇒ ≤1 extra branch). Adds 0 LLM calls
    # (the critic is never invoked; the answer LLM is still called exactly once on the
    # richer evidence). On rows that already clear the gate it is a no-op ⇒ the
    # prediction is byte-identical to baseline (only the diagnostic quality_score —
    # None on every single-branch baseline row — is now populated). Mirrored
    # byte-for-byte into MMTS_CONFIGS and tsr_bench CONFIGS.
    "react_stage2_refine": dict(
        loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        max_loop_depth=2,
        loop_mode="refine",
        quality_threshold=0.55,
    ),
    # Stage-3 LLM-proposed actions (research_journal/02_react_stages.md §Stage-3) —
    # the LAST, highest-risk ReAct stage. On the SAME live gate as Stage 2
    # (quality_score < 0.55), the LLM PROPOSES the next action from a CLOSED typed
    # registry (the 6 branches + vision + numeric_head) where the Stage-2 keyword
    # candidate source found none (the binding constraint, log/005). One proposal-LLM
    # call per step (router client), <=2 executed actions/row (max_loop_depth=2).
    # Valid branch proposals ACCUMULATE deterministic evidence in the tool layer (SVI);
    # a vision proposal flags the existing vision gate; numeric_head is a structural
    # no-op on MCQ. Invalid / out-of-registry / over-budget / already-run proposals
    # fall back DETERMINISTICALLY (never retried). The proposal is a ROUTING decision —
    # the answer LLM is never re-prompted to re-answer; the proposal prompt OMITS the
    # options (no letter-nudging). Rows clearing the gate are a no-op ⇒ byte-identical
    # to baseline. Mirrored byte-for-byte into MMTS_CONFIGS and tsr_bench CONFIGS.
    "react_stage3_actions": dict(
        loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        max_loop_depth=2,
        loop_mode="propose",
        quality_threshold=0.55,
    ),
    # Phase-4 "Graceful Avalanche" DELIVERABLE alias (PREREGISTRATION_phase4.md).
    # The full agentic scaffold = the fullest agentic mode (react_stage3_actions);
    # it SUBSUMES the dormant Stage-1 (evidence_incomplete reroute, 0/746 on TSExam)
    # and Stage-2 (keyword-candidate refine, 0 candidates) triggers because the same
    # live gate (quality_score < 0.55) drives all three and Stage-3's LLM proposer is
    # the only one that ever finds an action where the keyword source found none
    # (logs 004-007). NOTHING was promoted per-branch on TSExam — the proposer
    # executes but is net-neutral (6 corrected / 8 broken). Shipped DARK / validated
    # for NON-REGRESSION cross-benchmark, NOT as an accuracy lever. Kwargs are
    # byte-identical to react_stage3_actions ⇒ rows clearing the gate are a no-op ⇒
    # byte-identical to baseline (tests/test_wave1_contracts.py::TestAgenticTsmartAlias).
    # Mirrored byte-for-byte into MMTS_CONFIGS and tsr_bench CONFIGS.
    "agentic_tsmart": dict(
        loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        max_loop_depth=2,
        loop_mode="propose",
        quality_threshold=0.55,
    ),
}


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_eval_rows(n_per_cat, seed, full):
    """Load TimeSeriesExam1, map categories, return a fixed paired sample."""
    from datasets import load_dataset
    import pandas as pd

    df = load_dataset("AutonLab/TimeSeriesExam1", split="test").to_pandas()
    df["cat_code"] = df["category"].map(CAT_MAP)
    df = df[df["cat_code"].notna()].copy()

    rows = []
    for code in CATEGORIES:
        sub = df[df["cat_code"] == code].sort_values("id")
        if not full:
            n = min(n_per_cat, len(sub))
            sub = sub.sample(n=n, random_state=seed)
        for _, r in sub.iterrows():
            rec = r.to_dict()
            rec["category"] = code            # overwrite with 2-letter code
            rec["subcategory_name"] = r["subcategory"]
            rows.append(rec)
    return rows


def compute_metrics(results):
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    errors = 0
    fallback = 0
    vision_used = 0
    multi_used = 0
    critic_used = 0
    for r in results:
        if r is None or "error" in r:
            errors += 1
            continue
        cat = r.get("category", "?")
        gold, pred = r.get("gold_letter"), r.get("predicted_letter")
        if gold:
            cat_total[cat] += 1
            if pred == gold:
                cat_correct[cat] += 1
        fallback += bool(r.get("used_fallback"))
        vision_used += bool(r.get("vision_used") or r.get("vision_letter"))
        multi_used += bool(r.get("multi_branch_used"))
        critic_used += bool(r.get("critic_called"))

    per_cat = {}
    accs = []
    for cat in CATEGORIES:
        n, k = cat_total[cat], cat_correct[cat]
        if n:
            lo, hi = wilson_ci(k, n)
            per_cat[cat] = {"acc": round(k / n, 4), "correct": k, "total": n,
                            "ci": [round(lo, 4), round(hi, 4)]}
            accs.append(k / n)
        else:
            per_cat[cat] = None

    micro_k = sum(cat_correct.values())
    micro_n = sum(cat_total.values())
    return {
        "macro_oa": round(sum(accs) / len(accs), 4) if accs else None,
        "micro_oa": round(micro_k / micro_n, 4) if micro_n else None,
        "n": micro_n,
        "per_category": per_cat,
        "errors": errors,
        "fallback_rate": round(fallback / len(results), 4) if results else None,
        "vision_rate": round(vision_used / len(results), 4) if results else None,
        "multi_branch_rate": round(multi_used / len(results), 4) if results else None,
        "critic_rate": round(critic_used / len(results), 4) if results else None,
    }


def slim_row(r):
    """Keep the fields needed to diagnose a failure; drop raw series."""
    if r is None:
        return {"error": "none"}
    ev = r.get("evidence") or {}
    keep_ev = {k: ev.get(k) for k in (
        "branch", "direction", "slope", "r2", "dominant_period_fft",
        "period_reconciled", "seasonal_strength", "waveform_hint",
        "is_stationary", "is_white_noise", "adf_pval", "kpss_pval",
        "outlier_count", "max_deviation", "best_lag", "direction",
        "best_corr", "pval_12", "pval_21", "flags",
        "vision_confidence", "vision_answer_parse_success",
    ) if k in ev}
    vtrig = ev.get("vision_trigger") if isinstance(ev.get("vision_trigger"), dict) else {}
    return {
        "id": r.get("id"),
        "category": r.get("category"),
        "subcategory": r.get("subcategory_name") or r.get("subcategory"),
        "question": r.get("question"),
        "viz_type": vtrig.get("viz_type"),
        "vision_triggered": vtrig.get("triggered"),
        "vision_forced": vtrig.get("forced"),
        "vision_suppressed": vtrig.get("suppressed"),
        "gold": r.get("gold_letter"),
        "pred": r.get("predicted_letter"),
        "correct": r.get("predicted_letter") == r.get("gold_letter"),
        "routing": r.get("routing"),
        "branch_used": r.get("branch_used"),
        "initial_branch_used": r.get("initial_branch_used"),
        "subtype_used": r.get("subtype_used"),
        "branch_candidates": r.get("branch_candidates"),
        "quality_score": r.get("quality_score"),
        "multi_branch_used": r.get("multi_branch_used"),
        "critic_called": r.get("critic_called"),
        "vision_used": r.get("vision_used"),
        "vision_letter": r.get("vision_letter"),
        "vision_confidence": r.get("vision_confidence"),
        # Track-V raw-pixel measurement arm: True on fired rows when the raw PNG was
        # attached to the answer LLM (raw_pixel_vision=True). Falsy/None otherwise.
        "raw_pixel_vision_used": r.get("raw_pixel_vision_used"),
        "expected_schema": r.get("expected_schema"),
        "predicted_value": r.get("predicted_value"),
        "used_fallback": r.get("used_fallback"),
        "flags": r.get("flags"),
        # Stage-1 loop + Self-Consistency control diagnostics (additive).
        "loop_used": r.get("loop_used"),
        "loop_depth": r.get("loop_depth"),
        "loop_final_branch": r.get("loop_final_branch"),
        "loop_stopped": r.get("loop_stopped"),
        "loop_trigger_flags": r.get("loop_trigger_flags"),
        "sc_sampled": r.get("sc_sampled"),
        "sc_k": r.get("sc_k"),
        # Stage-2 refine + Stage-3 propose diagnostics (additive; falsy/None when off).
        "refine_fired": r.get("refine_fired"),
        "refine_depth": r.get("refine_depth"),
        "refine_branches": r.get("refine_branches"),
        "propose_fired": r.get("propose_fired"),
        "propose_executed": r.get("propose_executed"),
        "propose_steps": r.get("propose_steps"),
        "propose_fallbacks": r.get("propose_fallbacks"),
        "evidence": keep_ev,
    }


def print_table(metrics, baseline=None):
    print(f"\n  {'cat':<5}{'acc':>8}{'95% CI':>18}{'n':>5}   Δbase   vs TS-Agent")
    print("  " + "-" * 64)
    for cat in CATEGORIES:
        pc = metrics["per_category"].get(cat)
        if not pc:
            continue
        ci = pc["ci"]
        d_base = ""
        if baseline and baseline["per_category"].get(cat):
            d_base = f"{pc['acc'] - baseline['per_category'][cat]['acc']:+.3f}"
        tsa = REF["TS-Agent"][cat]
        vs = f"{pc['acc'] - tsa:+.3f}"
        print(f"  {cat:<5}{pc['acc']:>8.3f}  [{ci[0]:.2f},{ci[1]:.2f}]{pc['total']:>5}{d_base:>8}{vs:>11}")
    print("  " + "-" * 64)
    dmo = ""
    if baseline:
        dmo = f"{metrics['macro_oa'] - baseline['macro_oa']:+.3f}"
    print(f"  {'OA':<5}{metrics['macro_oa']:>8.3f}{'(macro)':>18}{metrics['n']:>5}{dmo:>8}"
          f"{metrics['macro_oa'] - REF['TS-Agent']['OA']:>+11.3f}")
    print(f"  micro OA: {metrics['micro_oa']}  | vision {metrics['vision_rate']} "
          f"multi {metrics['multi_branch_rate']} critic {metrics['critic_rate']} "
          f"fallback {metrics['fallback_rate']} errors {metrics['errors']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default=None)
    ap.add_argument("--multi-branch", action="store_true")
    ap.add_argument("--sa-vision", action="store_true")
    ap.add_argument("--sa-mode", default="stacked")
    ap.add_argument("--force-vision", default="",
                    help="comma-separated branch names to force vision on, e.g. anomaly")
    ap.add_argument("--no-vision", default="",
                    help="comma-separated branch names to suppress vision on, e.g. noise")
    ap.add_argument("--n-per-cat", type=int, default=30)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baseline-tag", default="baseline",
                    help="tag whose metrics.json to diff against")
    ap.add_argument("--config-json", default=None,
                    help="JSON object of run_pipeline kwargs injected verbatim (the "
                         "autonomous-search freeform arm). Validated against the "
                         "run_pipeline signature; wins over --config/flags. Default "
                         "off ⇒ byte-identical to the named-config path.")
    # --provider/--model select the LLM backbone. Default provider=gemini ⇒
    # byte-identical to the original hardcoded GeminiClient path. --provider openai
    # routes the WHOLE pipeline (router+answer+vision) through gpt-4o-mini.
    from tsqa.llm.factory import add_provider_args  # noqa: E402
    add_provider_args(ap)
    args = ap.parse_args()

    if args.config_json:
        import inspect
        from tsqa.eval.runner import run_pipeline  # noqa: E402
        try:
            raw = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: --config-json is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(raw, dict):
            print("ERROR: --config-json must be a JSON object of run_pipeline kwargs")
            sys.exit(1)
        # Validate keys against the run_pipeline signature (minus the non-config
        # params), so a typo'd kwarg fails loudly instead of silently no-opping.
        allowed = set(inspect.signature(run_pipeline).parameters) - {
            "row", "llm_client", "router_client", "answer_client",
            "return_trace", "trace_metadata",
        }
        bad = set(raw) - allowed
        if bad:
            print(f"ERROR: --config-json has unknown run_pipeline kwargs: {sorted(bad)}")
            print(f"  allowed: {sorted(allowed)}")
            sys.exit(1)
        kw = dict(raw)
    elif args.config:
        kw = dict(CONFIGS[args.config])
    else:
        kw = dict(multi_branch=args.multi_branch,
                  enable_sa_vision=args.sa_vision, sa_viz_mode=args.sa_mode)
        if args.force_vision:
            kw["forced_vision_branches"] = [b.strip() for b in args.force_vision.split(",") if b.strip()]
        if args.no_vision:
            kw["no_vision_branches"] = [b.strip() for b in args.no_vision.split(",") if b.strip()]

    from tsqa.llm.factory import create_llm_client
    if args.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    client = create_llm_client(provider=args.provider, model=args.model)
    model_id = getattr(client, "model_name", args.provider)

    print(f"Config {args.tag}: {kw}  [provider={args.provider} model={model_id}]")
    rows = load_eval_rows(args.n_per_cat, args.seed, args.full)
    dist = defaultdict(int)
    for r in rows:
        dist[r["category"]] += 1
    print(f"Eval set: {len(rows)} rows {dict(dist)}")

    t0 = time.monotonic()
    results = run_dataset(rows, client, show_progress=True,
                          max_workers=args.max_workers, **kw)
    wall = time.monotonic() - t0
    metrics = compute_metrics(results)
    metrics["wall_sec"] = round(wall, 1)
    metrics["sec_per_row"] = round(wall / len(rows), 2)
    metrics["config"] = kw
    metrics["tag"] = args.tag
    metrics["provider"] = args.provider
    metrics["model"] = model_id
    # Real API token cost (the autonomous-search budget ledger reads this).
    metrics["tokens"] = client.usage_snapshot()
    metrics["timestamp"] = datetime.datetime.now().isoformat()
    from tsqa.eval.provenance import run_provenance
    metrics["provenance"] = run_provenance(extra={
        "dataset": {"benchmark": "TimeSeriesExam", "n_rows": len(rows),
                    "seed": args.seed, "full": bool(args.full),
                    "n_per_cat": args.n_per_cat},
    })

    baseline = None
    bpath = os.path.join(OUT_ROOT, args.baseline_tag, "metrics.json")
    if args.tag != args.baseline_tag and os.path.exists(bpath):
        with open(bpath) as f:
            baseline = json.load(f)

    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    # The runner result doesn't echo question/subcategory — join back from rows by id.
    by_id = {row["id"]: row for row in rows}
    for r in results:
        if r and "id" in r and r["id"] in by_id:
            src = by_id[r["id"]]
            r.setdefault("question", src.get("question"))
            r.setdefault("subcategory_name", src.get("subcategory_name"))
    slim = [slim_row(r) for r in results]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(slim, f, indent=2, default=str)
    with open(os.path.join(out_dir, "failures.json"), "w") as f:
        json.dump([s for s in slim if not s.get("correct")], f, indent=2, default=str)

    print_table(metrics, baseline)
    print(f"\n  wall {wall:.0f}s ({metrics['sec_per_row']}s/row) -> {out_dir}")


if __name__ == "__main__":
    main()
