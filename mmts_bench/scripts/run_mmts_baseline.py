#!/usr/bin/env python3
# scripts/run_mmts_baseline.py
"""
MMTS-Bench Baseline Runner
==========================
Evaluates the T-SMART pipeline against MMTS-Bench and outputs:
  • Per-category accuracy breakdown
  • Delta vs TS-Agent / ChatTS baselines (via DeltaMatrix)
  • CSV file in outputs/ for historical tracking

Usage examples
--------------
# Quick sanity test — 5 samples from Base subset
python scripts/run_mmts_baseline.py --subset Base --max-rows 5

# Full Base run
python scripts/run_mmts_baseline.py --subset Base

# Full run on all subsets
python scripts/run_mmts_baseline.py --subset All

# Hint-augmented mode
python scripts/run_mmts_baseline.py --subset Base --use-hint

Environment variables
---------------------
MMTS_BENCH_PATH   override default data path (./MMTS-BENCH)
GEMINI_API_KEY    required for Gemini pipeline calls
"""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parent.parent   # mmts_bench/  (contains eval/, tools/)
_AGENTIC = _ROOT.parent / "tsqa"           # ERSP-TS-Research/tsqa

# IMPORTANT: mmts_bench/ itself goes on sys.path directly (for `eval.*`,
# `tools.*`). tsqa is imported as the `tsqa` PACKAGE (not added to
# sys.path directly) to avoid an `eval`/`tools` namespace collision between
# mmts_bench/eval and tsqa/tsqa/eval.
sys.path.insert(0, str(_ROOT))      # -> eval.dataloaders, eval.baseline_matrix
sys.path.insert(0, str(_AGENTIC))   # -> tsqa.eval.runner, tsqa.llm (as package)

# GEMINI_API_KEY lives in the repo-root .env (this script, unlike run_eval.py,
# has no implicit dotenv load). Load it here so --api-key can default to it.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT.parent / ".env")
except Exception:
    pass

from eval.dataloaders     import MMTSBenchAdapter
from eval.baseline_matrix import DeltaMatrix

# Import the real pipeline from tsqa/tsqa (proper package import —
# required because runner.py uses relative imports like `from ..branches import`)
from tsqa.eval.runner import run_pipeline, _gold_letter   # type: ignore
from tsqa.router.parser import infer_expected_schema      # type: ignore
from tsqa.llm.factory import create_llm_client      # type: ignore
from tsqa.eval.scoring import (                     # type: ignore
    score_numeric_value as _central_score_numeric,
    score_categorical_value as _central_score_categorical,
)

# Extra building blocks needed for N-way ("choose it" / "smooth it" / etc.)
# multi-series comparison, which run_pipeline()/run_similarity() don't
# support natively (they only ever compare exactly 2 series).
from tsqa.branches import run_similarity            # type: ignore
from tsqa.verifier  import verify                   # type: ignore
from tsqa.llm       import build_answer_prompt, SYSTEM_PROMPT, parse_answer  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit protection constants
# ---------------------------------------------------------------------------
CALL_DELAY_SEC  = 0.0    # client now self-throttles (0.02s spacing); no extra pause
BACKOFF_INITIAL = 30.0   # first retry wait (seconds)
BACKOFF_MAX     = 300.0  # cap at 5 minutes
MAX_RETRIES     = 5
ROW_TIMEOUT_SEC = 120.0  # max time allowed for a single run_pipeline() call
CHECKPOINT_EVERY = 25    # save partial results to disk every N completed rows
DEFAULT_MAX_WORKERS = 16 # concurrent rows; the GeminiClient ceiling is ~3000 RPM

# Frozen architectural configs threaded into run_pipeline(). These are the SAME
# in-memory switches validated on TimeSeriesExam1 (config `nu_ad_fix`) — no
# prompt/letter tuning. The whole point of the MMTS experiment is to apply the
# UNCHANGED config to a benchmark it was never tuned on.
MMTS_CONFIGS = {
    "baseline":  {},
    "nu_ad_fix": {
        "no_vision_branches":     ["noise"],
        "forced_vision_branches": ["anomaly"],
    },
    # Vision fully suppressed on EVERY branch → the prediction is the pure "math
    # position". This is the counterfactual arm (A2) for the visual-trust study:
    # paired against the vision-permissive `baseline`, the rows where they
    # disagree are exactly the discordance set scripts/visual_trust.py analyses.
    "vision_off": {
        "no_vision_branches": ["trend", "periodicity", "anomaly",
                               "noise", "similarity", "causality"],
    },
    # Free-response COUNTERFACTUAL (Track-A2 / load-bearing proof): on numerical rows
    # the answer LLM COMPUTES the number directly from the raw series (no tool, no
    # options, no evidence); the deterministic numeric head is BYPASSED. Paired against
    # the tool-owned numeric head on the SAME Base rows, the @10% gap is the
    # "tools are load-bearing on a strong backbone" measurement. Byte-identical switch
    # to TSExam's CONFIGS["llm_numeric"] (mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md).
    "llm_numeric": {
        "llm_numeric": True,
    },
    # Numeric PARSER arms (three-arm parsing study, log/012) — byte-identical switches
    # to TSExam's CONFIGS["numeric_det"/"numeric_llm"]. A0=keyword default, A1=deterministic
    # (composition grammar + fuzzy/synonym/typo), A2=llm_fallback (LLM-planner over the
    # closed registry, fired only when A1 abstains). All feed the same tools.
    "numeric_det": {
        "numeric_parser": "deterministic",
    },
    "numeric_llm": {
        "numeric_parser": "llm_fallback",
    },
    # Track-A2 surgical arm: vision_off on the five non-anomaly branches AND forced on
    # for anomaly — the single-variable contrast vs `vision_off` isolating the AD-vision
    # lever on a weak backbone. Byte-identical switch to TSExam's CONFIGS["ad_vision_only"]
    # (tsqa/PREREGISTRATION_c1_gpt4omini_vision.md). Run on TSExam; mirrored here
    # to preserve the cross-harness config invariant.
    "ad_vision_only": {
        "no_vision_branches": ["trend", "periodicity", "noise",
                               "similarity", "causality"],
        "forced_vision_branches": ["anomaly"],
    },
    # The learned "should-I-look" gate (#1): the additive trigger is replaced by the
    # pre-registered per-branch suppression policy (fitted on MMTS, frozen). Paired
    # against `baseline` (vision_gate="additive"). Byte-identical switch to TSExam's
    # CONFIGS["learned_gate"] — the generalization claim depends on it being unchanged.
    "learned_gate": {
        "vision_gate": "learned",
    },
    "react_stage1_evidence_retry": {
        "loop_branches": ["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        "max_loop_depth": 1,
        "loop_on_flags": ["evidence_incomplete"],
        "loop_mode": "reroute_once",
    },
    # Self-Consistency control arm (Stage-1 constraint #4; 2203.11171). On the SAME
    # trigger as the loop (evidence_incomplete), sample the answer LLM k=5 times at
    # temp 0.7 and majority-vote (deterministic alphabetic tie-break). Non-flagged
    # rows stay byte-identical to baseline. Byte-identical to TSExam's CONFIGS.
    "self_consistency": {
        "self_consistency": 5,
        "self_consistency_temperature": 0.7,
    },
    # Stage-2 iterative tool refinement (research_journal/02_react_stages.md §Stage-2).
    # When the PRIMARY branch's deterministic evidence scores below the pre-existing
    # principled quality gate (QUALITY_THRESHOLD=0.55), ACCUMULATE a 2nd candidate
    # branch's deterministic evidence (keyword-selected, NO LLM) into the evidence
    # dict; depth bounded by max_loop_depth (=2 ⇒ ≤1 extra branch). Adds 0 LLM calls
    # (critic never invoked; answer LLM still called exactly once). Rows clearing the
    # gate are a no-op ⇒ prediction byte-identical to baseline. Byte-identical switch
    # to TSExam's CONFIGS["react_stage2_refine"].
    "react_stage2_refine": {
        "loop_branches": ["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        "max_loop_depth": 2,
        "loop_mode": "refine",
        "quality_threshold": 0.55,
    },
    # Stage-3 LLM-proposed actions (research_journal/02_react_stages.md §Stage-3).
    # On the SAME live gate as Stage 2 (quality_score < 0.55), the LLM PROPOSES the
    # next action from a CLOSED typed registry (6 branches + vision + numeric_head)
    # where the keyword candidate source found none (log/005). One proposal-LLM call
    # per step; <=2 executed actions/row. Valid branch proposals accumulate evidence
    # in the tool layer (SVI); invalid / out-of-registry / over-budget / already-run
    # proposals fall back deterministically. The proposal is a ROUTING decision (the
    # answer LLM is never re-prompted; the prompt omits the options). Rows clearing
    # the gate are a no-op ⇒ byte-identical to baseline. Byte-identical switch to
    # TSExam's CONFIGS["react_stage3_actions"].
    "react_stage3_actions": {
        "loop_branches": ["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        "max_loop_depth": 2,
        "loop_mode": "propose",
        "quality_threshold": 0.55,
    },
    # Phase-4 "Graceful Avalanche" DELIVERABLE alias. Full agentic scaffold =
    # react_stage3_actions (subsumes the dormant Stage-1/2 triggers; same live gate
    # quality_score < 0.55). Nothing per-branch-promoted on TSExam (logs 004-007);
    # shipped DARK / validated for NON-REGRESSION. Byte-identical switch to TSExam's
    # CONFIGS["agentic_tsmart"]; rows clearing the gate are a no-op ⇒ byte-identical
    # to baseline. The whole generalization claim depends on it being unchanged.
    "agentic_tsmart": {
        "loop_branches": ["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
        "max_loop_depth": 2,
        "loop_mode": "propose",
        "quality_threshold": 0.55,
    },
}

# MMTS-Bench's published numeric metric (paper §A.1, arXiv:2602.08588):
#   Accuracy@10% = 𝟙(|pred - gold| / |gold| <= 0.10)
#   Relative Accuracy = max(1 - |pred - gold| / |gold|, 0)
# Pinned to the paper's protocol, NOT a tuned knob (overridable via
# --numeric-rel-tol for a pre-registered sensitivity sweep). The paper leaves
# gold == 0 undefined; our declared rule scores it as exact-match within a tiny
# epsilon (those golds are integer indices/counts).
NUMERIC_REL_TOL = 0.10
_ZERO_GOLD_ATOL = 1e-9

# Canonical CSV schema (shared by checkpoint + final detail writers). Using a
# fixed superset with extrasaction="ignore" lets us resume from older CSVs and
# add the new `config` column without a schema clash.
RESULT_FIELDS = [
    "sample_id", "category", "branch_used", "is_dual", "ground_truth",
    "predicted", "correct", "used_fallback", "vision_used", "subset",
    "config", "error",
    # --- per-row diagnostics: parity with TSExam research/run_eval.py:slim_row.
    # Needed by scripts/visual_trust.py (the #2 visual-trust study) and the
    # schema-detection audit. extrasaction="ignore" + DictWriter restval keep
    # OLD checkpoints (which lack these columns) loadable — they write as empty.
    "subtype_used", "initial_branch_used", "vision_letter", "vision_confidence", "vision_triggered",
    "vision_forced", "vision_suppressed", "viz_type", "flags",
    "expected_schema", "predicted_value", "numeric_quantity", "numeric_note",
    "rel_acc", "evidence_json", "qa_type",
    # --- ReAct agency diagnostics (Phase-4 OOD agency test). Additive, NON-behavioral
    # (read straight off the runner result dict); falsy/None on every non-loop config so
    # old checkpoints + the frozen configs are byte-unchanged. Persist the gate fire
    # (quality_score < threshold), the action-executed set, and the loop trace so the
    # action-executed-row discordance (corrected vs broken) can be measured OOD — the
    # MMTS CSV previously dropped these (TSExam slim_row keeps them).
    "quality_score", "propose_fired", "propose_executed", "propose_steps",
    "propose_fallbacks", "loop_used", "loop_depth", "refine_fired", "refine_branches",
]


# ---------------------------------------------------------------------------
# Free-response scoring (Workstream B3) — pinned to MMTS-Bench's own protocol
# ---------------------------------------------------------------------------

def _to_float(x):
    """Parse a gold/predicted numeric value (string with stray spaces/commas, or
    already-numeric) to float; None if unparseable."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group()) if m else None


def score_numeric(pred, gold, rel_tol=NUMERIC_REL_TOL):
    """MMTS-Bench numeric scoring. Returns (acc_at_tol: bool, rel_acc: float|None).

    acc_at_tol is the pre-registered binary metric (Accuracy@rel_tol, default 10%)
    that composes into the paired McNemar with MCQ accuracy; rel_acc is the
    continuous Relative Accuracy reported alongside it.
    """
    return _central_score_numeric(pred, gold, rel_tol=rel_tol)


def _norm_label(s):
    return re.sub(r"[\s\-_]+", "", str(s).strip().lower()) if s is not None else ""


def score_categorical(pred, gold):
    """Exact (normalized) label match — MMTS-Bench scores categorical/true-false
    answers by exact match (paper §A.1)."""
    return _central_score_categorical(pred, gold)


# Evidence fields kept in the per-row diagnostic dump (mirrors slim_row's keep_ev).
_EV_SUMMARY_KEYS = (
    "branch", "direction", "slope", "r2", "dominant_period_fft",
    "period_reconciled", "seasonal_strength", "is_stationary", "is_white_noise",
    "adf_pval", "kpss_pval", "outlier_count", "max_deviation", "best_lag",
    "best_corr", "pval_12", "pval_21", "vision_confidence",
)


def _evidence_summary(ev) -> dict:
    if not isinstance(ev, dict):
        return {}
    out = {}
    for k in _EV_SUMMARY_KEYS:
        if k in ev:
            v = ev[k]
            if isinstance(v, float):
                out[k] = round(v, 6)
            elif isinstance(v, (int, str, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
    return out


def _diag_fields(result: dict, sample: dict) -> dict:
    """Per-row diagnostic columns (vision trigger meta, schema, evidence summary)
    extracted from the runner result — the data the visual-trust study keys on."""
    ev = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    vtrig = ev.get("vision_trigger") if isinstance(ev.get("vision_trigger"), dict) else {}
    flags = result.get("flags") or ev.get("flags") or []
    return {
        "subtype_used":      result.get("subtype_used"),
        "initial_branch_used": result.get("initial_branch_used"),
        "vision_letter":     result.get("vision_letter"),
        "vision_confidence": result.get("vision_confidence"),
        "vision_triggered":  vtrig.get("triggered"),
        "vision_forced":     vtrig.get("forced"),
        "vision_suppressed": vtrig.get("suppressed"),
        "viz_type":          vtrig.get("viz_type"),
        "flags":             flags if isinstance(flags, str) else json.dumps(flags),
        "expected_schema":   result.get("expected_schema", "mcq"),
        "predicted_value":   result.get("predicted_value"),
        "numeric_quantity":  result.get("numeric_quantity"),
        "numeric_note":      result.get("numeric_note"),
        "evidence_json":     json.dumps(_evidence_summary(ev), default=str),
        "qa_type":           sample.get("qa_type", ""),
        # ReAct agency diagnostics (Phase-4). None/falsy on non-loop configs.
        "quality_score":     result.get("quality_score"),
        "propose_fired":     result.get("propose_fired"),
        "propose_executed":  json.dumps(result.get("propose_executed"))
                             if result.get("propose_executed") is not None else None,
        "propose_steps":     json.dumps(result.get("propose_steps"), default=str)
                             if result.get("propose_steps") is not None else None,
        "propose_fallbacks": json.dumps(result.get("propose_fallbacks"), default=str)
                             if result.get("propose_fallbacks") is not None else None,
        "loop_used":         result.get("loop_used"),
        "loop_depth":        result.get("loop_depth"),
        "refine_fired":      result.get("refine_fired"),
        "refine_branches":   json.dumps(result.get("refine_branches"))
                             if result.get("refine_branches") is not None else None,
    }


# ---------------------------------------------------------------------------
# Row adapter — map MMTSBenchAdapter output → runner.py expected keys
# ---------------------------------------------------------------------------

_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _parse_options_block(raw_opts) -> list:
    # Parse the MMTS-Bench option column into a clean list of option texts
    # (prefixes like "A) " / "B. " stripped).
    #
    # Handles:
    #   - already a list             -> returned as-is (stripped)
    #   - Python-literal list string -> ast.literal_eval
    #   - block of "A) text", "B) text", "C) text" lines
    #     -> split on the A)/B)/C)/D) markers
    #   - plain comma-separated string -> split on commas (last resort)
    import ast, re as _re

    if isinstance(raw_opts, list):
        return [str(o).strip() for o in raw_opts]

    if not isinstance(raw_opts, str) or not raw_opts.strip():
        return []

    raw_opts = raw_opts.strip()

    # Python-literal list, e.g. "['foo', 'bar']"
    if raw_opts.startswith("["):
        try:
            parsed = ast.literal_eval(raw_opts)
            if isinstance(parsed, (list, tuple)):
                return [str(o).strip() for o in parsed]
        except Exception:
            pass

    # Block of "A) ...", "B) ...", "C) ..." lines -> split on letter markers.
    # Matches "A)", "A.", "A:", "(A)" etc. at the start of a line.
    # MMTS-Bench option blocks come in two layouts:
    #   multi-line:  "A) text\nB) text\nC) text"
    #   single-line: "A) text. B) text. C) text. D) text"   (Match subset)
    # Split on any whitespace-preceded letter marker (not just newline-anchored)
    # so both layouts parse into a clean list of option texts.
    parts = _re.split(r"(?:^|\s)([A-D])[\)\.\:]\s+", raw_opts)
    # parts looks like: ['', 'A', 'text1', 'B', 'text2', ...]
    if len(parts) > 1:
        options = []
        for i in range(1, len(parts), 2):
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if text:
                options.append(text)
        if options:
            return options

    # Last resort — comma split
    return [o.strip() for o in raw_opts.split(",") if o.strip()]


def _adapter_sample_to_row(sample: dict) -> dict:
    """
    Convert the standardised MMTSBenchAdapter dict into the row format
    expected by run_pipeline().
    """
    # runner.py expects:
    #   question  str
    #   options   list[str]    (option TEXT, no letter prefixes)
    #   answer    str          (TEXT of the correct option -- _gold_letter
    #                            in runner.py matches this against `options`
    #                            to derive the gold letter)
    #   category  str
    #   ts        np.ndarray   (single series)
    #   ts1       np.ndarray   (dual series -- first)
    #   ts2       np.ndarray   (dual series -- second)
    #   id        str | int
    #
    # MMTS-Bench quirks handled here:
    #   - `option` column is one block of "A) ...", "B) ...", "C) ..." lines
    #     -> parsed into a clean list with prefixes stripped.
    #   - `answer` column is just the letter ("A"/"B"/"C"/"D") -> converted
    #     to the matching option TEXT so runner.py's _gold_letter() works.
    options = _parse_options_block(sample.get("options", ""))

    raw_answer = str(sample.get("ground_truth", "")).strip()
    if raw_answer.upper() in _LETTER_TO_IDX:
        idx = _LETTER_TO_IDX[raw_answer.upper()]
        answer_text = options[idx] if idx < len(options) else raw_answer
    else:
        # Already full option text (e.g. some Align rows) — use as-is.
        answer_text = raw_answer

    row = {
        "id":       sample["sample_id"],
        "category": sample["category"],
        "question": sample["query"],
        "options":  options,
        "answer":   answer_text,
    }

    if sample.get("is_dual"):
        row["ts"]  = None
        row["ts1"] = sample["ts1"] if len(sample["ts1"]) > 0 else None
        row["ts2"] = sample["ts2"] if len(sample["ts2"]) > 0 else None
    else:
        ts = sample["ts_array"]
        row["ts"]  = ts if ts is not None and len(ts) > 0 else None
        row["ts1"] = None
        row["ts2"] = None

    return row


def _is_scoreable_mcq(sample: dict) -> bool:
    """True if this row is a multiple-choice question the pipeline can be scored
    on. MMTS-Bench Base mixes in free-response numeric items (e.g. "what is the
    standard deviation?" -> 260.79) and stationarity prompts with NO A/B/C/D
    options; an MCQ router cannot answer those and they are wrong by
    construction (gold letter is None). Multi-series rows derive their gold by
    text match and are always scoreable.
    """
    if sample.get("is_multi"):
        return True
    row = _adapter_sample_to_row(sample)
    return _gold_letter(row) is not None


# ---------------------------------------------------------------------------
# API call with exponential backoff
# ---------------------------------------------------------------------------

def _run_pipeline_with_timeout(row: dict, llm_client, use_hint: bool, timeout: float,
                               vision_kwargs: Optional[dict] = None):
    """
    Run run_pipeline() in a daemon thread with a hard timeout.

    Returns the result dict, or raises TimeoutError if it doesn't finish
    in `timeout` seconds. The hung thread is abandoned (daemon=True so it
    won't block process exit) -- Python cannot forcibly kill a thread, but
    this lets the main loop move on instead of hanging forever.

    vision_kwargs threads the frozen config switches (no_vision_branches /
    forced_vision_branches) into run_pipeline unchanged.
    """
    result_box: dict = {}
    error_box:  dict = {}

    def _target():
        try:
            result_box["value"] = run_pipeline(
                row, llm_client, use_hint=use_hint, **(vision_kwargs or {})
            )
        except Exception as e:
            error_box["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(
            f"run_pipeline() exceeded {timeout:.0f}s timeout for row {row.get('id')}"
        )
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def _build_multi_match_prompt(question: str, options: list, comparisons: list) -> str:
    """
    Build a prompt asking the LLM to pick the best-matching candidate series,
    given pre-computed similarity evidence for each (reference, candidate_i) pair.

    comparisons: list of dicts, one per option letter, each containing the
                 evidence returned by run_similarity() (verified).
    """
    letters = "ABCD"
    lines = [
        "You are comparing one reference time series against several candidate "
        "time series using pre-computed statistical evidence (trend, shape via "
        "DTW, seasonality, variance, autocorrelation). Pick the candidate that "
        "best matches the reference according to the question.",
        "",
        f"Question:\n{question}",
        "",
        "Answer choices:",
    ]
    for i, opt in enumerate(options[:4]):
        lines.append(f"{letters[i]}. {opt}")
    lines.append("")
    lines.append("Evidence per candidate (reference vs that candidate):")
    for i, ev in enumerate(comparisons):
        letter = letters[i] if i < len(letters) else "?"
        compact = {
            k: ev.get(k) for k in (
                "shape_similarity", "shape_similarity_label",
                "dtw_distance_normalized", "trend_similarity",
                "level_similarity", "range_similarity",
                "variance_similarity", "autocorrel_similarity",
                "scaled_version_detected", "flipped_version_detected",
                "overall_similarity_summary",
            ) if k in ev
        }
        lines.append(f"\n[{letter}] {compact}")
    lines.append(
        "\nReturn exactly this format:\n"
        "LETTER\n"
        "Confidence: high|medium|low\n"
        "One sentence explaining which candidate matches best and why."
    )
    return "\n".join(lines)


def _run_multi_series_pipeline(row: dict, sample: dict, llm_client, use_hint: bool) -> dict:
    """
    Handle MMTS-Bench "choose it" / "find it" / "smooth it" / "reverse it"
    style questions: one reference series + N candidate series, where the
    answer options map 1:1 to the candidates in order.

    run_pipeline()'s similarity branch only ever compares exactly 2 series,
    so for these rows we run run_similarity(reference, candidate_i) once per
    candidate ourselves, verify each result, then ask the LLM to pick the
    best match using all N comparison summaries at once.

    Returns a dict matching run_pipeline()'s output shape so it can be
    logged identically to single/dual-series rows.
    """
    reference  = sample["reference_series"]
    candidates = sample["candidate_series"]
    question   = row["question"]
    options    = row["options"]
    gold       = row.get("answer")

    comparisons = []
    for cand in candidates:
        try:
            n = min(len(reference), len(cand))
            ev = run_similarity(reference[:n], cand[:n], scope="global")
            ev = verify(ev)
        except Exception as e:
            ev = {"similarity_error": str(e)}
        comparisons.append(ev)

    result = {
        "id":               row.get("id"),
        "category":         row.get("category"),
        "gold_letter":      None,
        "predicted_letter": None,
        "correct":          False,
        "branch_used":      "similarity_multi",
        "used_fallback":    False,
        "vision_used":      False,
        "raw_answer":       None,
        "parse_success":    False,
    }

    # Gold letter: options[i] text == gold answer text -> letter i
    letters = "ABCD"
    for i, opt in enumerate(options[:4]):
        if str(opt).strip().lower() == str(gold).strip().lower():
            result["gold_letter"] = letters[i]
            break

    try:
        prompt = _build_multi_match_prompt(question, options, comparisons)
        raw_answer = llm_client.generate(SYSTEM_PROMPT, prompt)
        parsed = parse_answer(raw_answer)
        result["raw_answer"]    = raw_answer
        result["predicted_letter"] = parsed.get("letter")
        result["parse_success"]    = parsed.get("parse_success", False)
        result["correct"] = (
            result["predicted_letter"] == result["gold_letter"]
            if result["gold_letter"] else False
        )
    except Exception as e:
        result["raw_answer"] = str(e)

    return result


def _call_with_backoff(row: dict, llm_client, use_hint: bool,
                       vision_kwargs: Optional[dict] = None) -> Optional[dict]:
    """
    Call run_pipeline() with a per-call timeout and exponential backoff on
    rate-limit errors.
    Returns the result dict, or None if all retries are exhausted, the call
    times out, or a non-rate-limit error occurs.
    """
    backoff = BACKOFF_INITIAL
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_pipeline_with_timeout(
                row, llm_client, use_hint, ROW_TIMEOUT_SEC, vision_kwargs
            )
        except TimeoutError as exc:
            logger.error("Row %s timed out: %s -- skipping.", row.get("id"), exc)
            return None
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = (
                "429"               in msg
                or "quota"          in msg
                or "rate limit"     in msg
                or "resource exhausted" in msg
            )
            if is_rate_limit:
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — backing off %.0fs",
                    attempt, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            else:
                logger.error("Pipeline error on row %s: %s", row.get("id"), exc)
                return None
    logger.error("Max retries (%d) exhausted — skipping row %s.", MAX_RETRIES, row.get("id"))
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    data_path: str,
    subset: str,
    max_rows: Optional[int],
    output_dir: Path,
    use_hint: bool,
    gemini_api_key: str,
    provider: str = "gemini",
    model: str | None = None,
    shuffle: bool = False,
    seed: int = 42,
    resume_from: Optional[str] = None,
    config_name: str = "baseline",
    vision_kwargs: Optional[dict] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    mcq_only: bool = False,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> None:

    vision_kwargs = vision_kwargs or {}
    logger.info("Initialising %s client...", provider)
    llm_client = create_llm_client(
        provider=provider,
        model=model,
        api_key=gemini_api_key or None,
    )

    logger.info("Loading MMTS-Bench  subset=%s  data_path=%s", subset, data_path)
    adapter = MMTSBenchAdapter(data_path, subset=subset)
    total   = len(adapter)

    # Materialise rows so we can optionally shuffle (the Base CSV is grouped
    # by category in contiguous blocks -- a small --max-rows without shuffle
    # only ever sees the first 1-2 categories).
    all_samples = list(adapter)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        logger.info("Shuffled %d rows (seed=%d) before applying --max-rows", total, seed)

    limit = min(max_rows, total) if max_rows else total
    all_samples = all_samples[:limit]
    logger.info(
        "Dataset size: %d  |  Processing: %d rows  |  config=%s  workers=%d",
        total, limit, config_name, max_workers,
    )
    if vision_kwargs:
        logger.info("Frozen vision switches: %s", vision_kwargs)

    # ── Resume support ───────────────────────────────────────────────────
    # If --resume-from points at a checkpoint CSV, load it and skip any
    # sample_id already present so a crashed/killed run can continue.
    results_log: list = []
    done_ids:    set  = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            with open(resume_path, newline="") as f:
                for r in csv.DictReader(f):
                    results_log.append(r)
                    done_ids.add(str(r["sample_id"]))
            logger.info(
                "Resuming from %s -- %d rows already done, will skip those.",
                resume_path, len(done_ids),
            )
        else:
            logger.warning("--resume-from %s not found -- starting fresh.", resume_path)

    pending = [s for s in all_samples if str(s["sample_id"]) not in done_ids]

    if mcq_only:
        before  = len(pending)
        pending = [s for s in pending if _is_scoreable_mcq(s)]
        logger.info(
            "--mcq-only: kept %d/%d scoreable MCQ rows (dropped %d free-response "
            "numeric/stationarity items the MC pipeline cannot answer).",
            len(pending), before, before - len(pending),
        )

    # Checkpoint file (overwritten every CHECKPOINT_EVERY completions + at the end)
    run_timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag         = f"{subset}_{config_name}"
    checkpoint_path = output_dir / f"mmts_results_{run_tag}_{run_timestamp}_checkpoint.csv"

    def _save_checkpoint():
        if not results_log:
            return
        with open(checkpoint_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_log)

    def _process_sample(sample: dict) -> dict:
        """Run one sample (single/dual via run_pipeline with the frozen vision
        config; multi via N-way similarity) → a flat results_log dict."""
        row      = _adapter_sample_to_row(sample)
        category = sample["category"]

        if sample.get("is_multi"):
            # "choose it" / "find it" / "smooth it" / "reverse it" — N candidate
            # series via our own N-way similarity comparison (run_pipeline()
            # only supports 2 series). Vision-neutral by construction.
            try:
                result = _run_multi_series_pipeline(row, sample, llm_client, use_hint=use_hint)
            except Exception as exc:
                logger.error("Multi-series pipeline error on row %s: %s", row.get("id"), exc)
                result = None
        else:
            result = _call_with_backoff(row, llm_client, use_hint, vision_kwargs)

        if result is None:
            skipped = {
                "sample_id":     row["id"],        "category":     category,
                "branch_used":   None,             "is_dual":      sample["is_dual"],
                "ground_truth":  sample.get("ground_truth"),
                "predicted":     None,             "correct":      False,
                "used_fallback": None,             "vision_used":  None,
                "subset":        sample["subset"], "config":       config_name,
                "error":         "skipped",        "rel_acc":      None,
            }
            skipped.update(_diag_fields({}, sample))
            return skipped

        # Score by the answer SHAPE the row demands. MCQ rows keep the runner's
        # letter-match verdict (the vision experiment is untouched); free-response
        # rows are scored here by MMTS-Bench's own protocol (numeric Accuracy@10%
        # / categorical exact match).
        schema = result.get("expected_schema", "mcq")
        rel_acc = None
        if schema == "numerical":
            ground_truth = sample.get("ground_truth")
            predicted    = result.get("predicted_value")
            correct, rel_acc = score_numeric(predicted, ground_truth, numeric_rel_tol)
        elif schema == "categorical":
            ground_truth = sample.get("ground_truth")
            predicted    = result.get("predicted_value")
            correct      = score_categorical(predicted, ground_truth)
        else:  # mcq (incl. multi-series similarity rows, which carry no schema)
            ground_truth = result.get("gold_letter")
            predicted    = result.get("predicted_letter")
            correct      = bool(result.get("correct", False))

        log = {
            "sample_id":     row["id"],        "category":     category,
            "branch_used":   result.get("branch_used"),
            "is_dual":       sample["is_dual"],
            "ground_truth":  ground_truth,
            "predicted":     predicted,
            "correct":       correct,
            "used_fallback": result.get("used_fallback"),
            "vision_used":   result.get("vision_used"),
            "subset":        sample["subset"], "config":         config_name,
            "error":         "",               "rel_acc":        rel_acc,
        }
        log.update(_diag_fields(result, sample))
        return log

    logger.info("Dispatching %d rows across %d workers...", len(pending), max_workers)
    t0        = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_sample, s): s for s in pending}
        for fut in concurrent.futures.as_completed(futs):
            sample = futs[fut]
            try:
                log = fut.result()
            except Exception as exc:
                logger.error("Row %s crashed: %s", sample.get("sample_id"), exc)
                log = {
                    "sample_id": sample.get("sample_id"), "category": sample.get("category"),
                    "branch_used": None, "is_dual": sample.get("is_dual"),
                    "ground_truth": sample.get("ground_truth"), "predicted": None,
                    "correct": False, "used_fallback": None, "vision_used": None,
                    "subset": sample.get("subset"), "config": config_name, "error": str(exc),
                }
            results_log.append(log)
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                _save_checkpoint()
                rate = completed / max(time.monotonic() - t0, 1e-6)
                logger.info("  %d/%d done (%.1f rows/s) -> %s",
                            completed, len(pending), rate, checkpoint_path.name)

    # Final checkpoint save (covers any remainder not divisible by CHECKPOINT_EVERY)
    _save_checkpoint()
    wall = time.monotonic() - t0
    logger.info("Processed %d rows in %.0fs (%.2f rows/s)",
                len(pending), wall, len(pending) / max(wall, 1e-6))

    # ── Aggregate per-category accuracy from the full results_log ────────
    correct_by_cat: defaultdict = defaultdict(int)
    total_by_cat:   defaultdict = defaultdict(int)
    for r in results_log:
        cat = r.get("category", "unknown")
        total_by_cat[cat] += 1
        if str(r.get("correct")).strip().lower() == "true":
            correct_by_cat[cat] += 1

    logger.info("\n%s\n  Per-Category Accuracy\n%s", "="*60, "="*60)
    category_acc: dict = {}
    for cat in sorted(total_by_cat.keys()):
        n   = total_by_cat[cat]
        c   = correct_by_cat[cat]
        acc = round(100.0 * c / n, 2) if n > 0 else 0.0
        category_acc[cat] = acc
        logger.info("  %-32s  %d/%d  =  %.2f%%", cat[:32], c, n, acc)

    overall_correct = sum(correct_by_cat.values())
    overall_total   = sum(total_by_cat.values())
    overall_acc     = round(100.0 * overall_correct / overall_total, 2) if overall_total > 0 else 0.0
    logger.info("  %-32s  %d/%d  =  %.2f%%", "OVERALL", overall_correct, overall_total, overall_acc)

    dm = DeltaMatrix()
    dm.ingest_results(category_acc, method="ZS")
    logger.info("\n%s", dm.summary(method="ZS"))

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_path = output_dir / f"mmts_results_{run_tag}_{timestamp}.csv"
    with open(detail_path, "w", newline="") as f:
        if results_log:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_log)
    logger.info("Detailed results -> %s", detail_path)

    # Real API token cost sidecar (the autonomous-search budget ledger reads this;
    # run_mmts writes CSVs, not a metrics.json, so the token snapshot lives here).
    try:
        token_path = output_dir / f"mmts_tokens_{run_tag}_{timestamp}.json"
        with open(token_path, "w") as f:
            json.dump(llm_client.usage_snapshot(), f, indent=2)
        logger.info("Token usage -> %s", token_path)
    except Exception as e:
        logger.warning("could not write token sidecar: %s", e)

    matrix_path = output_dir / f"mmts_accuracy_{run_tag}_{timestamp}.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "correct", "total", "accuracy_pct"])
        for cat in sorted(total_by_cat.keys()):
            writer.writerow([cat, correct_by_cat[cat], total_by_cat[cat], category_acc[cat]])
        writer.writerow(["OVERALL", overall_correct, overall_total, overall_acc])
    logger.info("Accuracy matrix -> %s", matrix_path)

    latex_path = output_dir / f"mmts_latex_{run_tag}_{timestamp}.tex"
    latex_path.write_text(dm.export_latex_table(
        method  = "ZS",
        caption = f"MMTS-Bench Baseline Results -- {subset} subset",
        label   = f"tab:mmts_{subset.lower()}",
    ))
    logger.info("LaTeX table -> %s", latex_path)

    # JSON run-manifest: a self-describing, committable record of this run
    # (config, scoring params, per-category accuracy, and full provenance).
    from tsqa.eval.provenance import run_provenance
    model_id = getattr(llm_client, "model_name", provider)
    _usage_snapshot_fn = getattr(llm_client, "usage_snapshot", None)
    usage = _usage_snapshot_fn() if _usage_snapshot_fn is not None else None
    manifest = {
        "benchmark":     "MMTS-Bench",
        "subset":        subset,
        "config":        config_name,
        "vision_kwargs": vision_kwargs,
        "provider":      provider,
        "model":         model_id,
        "scoring":       {"numeric_rel_tol": numeric_rel_tol, "mcq_only": mcq_only},
        "seed":          seed,
        "n_rows":        overall_total,
        "overall":       {"correct": overall_correct, "total": overall_total,
                          "accuracy_pct": overall_acc},
        "per_category":  {cat: {"correct": correct_by_cat[cat],
                                "total":   total_by_cat[cat],
                                "accuracy_pct": category_acc[cat]}
                          for cat in sorted(total_by_cat)},
        "files":         {"detail_csv":   detail_path.name,
                          "accuracy_csv": matrix_path.name,
                          "latex":        latex_path.name},
        "provenance":    run_provenance(),
        "usage":         usage,
    }
    manifest_path = output_dir / f"mmts_run_{run_tag}_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Run manifest -> %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run MMTS-Bench baseline evaluation against T-SMART pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("MMTS_BENCH_PATH", "./MMTS-BENCH"),
        help="Path to MMTS-BENCH root, or 'huggingface'. Default: $MMTS_BENCH_PATH",
    )
    parser.add_argument(
        "--subset",
        default="Base",
        choices=["Base", "InWild", "Match", "Align", "All"],
        help="Which benchmark subset to evaluate (default: Base)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit rows processed -- useful for quick sanity tests (e.g. --max-rows 5)",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle rows before applying --max-rows, so a small sample "
             "covers all categories instead of the first N rows in file order "
             "(Base CSV is grouped by category in blocks)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --shuffle (default: 42, for reproducibility)",
    )
    parser.add_argument(
        "--output-dir", default="./outputs",
        help="Directory for result CSVs and LaTeX tables (default: ./outputs)",
    )
    parser.add_argument(
        "--use-hint", action="store_true",
        help="Enable hint-augmented prompting mode (requires question_hint column)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit provider API key. Default None ⇒ the factory resolves the "
             "right env var PER PROVIDER ($GEMINI_API_KEY / $OPENAI_API_KEY). Do NOT "
             "default this to the Gemini key — that was passed to the OpenAI client "
             "and 401'd every --provider openai call.",
    )
    parser.add_argument(
        "--provider", choices=["gemini", "openai"], default="gemini",
        help="LLM provider (default: gemini).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Provider model override. Defaults to the provider's frozen default.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a previous checkpoint CSV (mmts_results_*_checkpoint.csv). "
             "Rows with sample_ids already in that file are skipped.",
    )
    parser.add_argument(
        "--config", choices=list(MMTS_CONFIGS), default="baseline",
        help="Frozen architectural config threaded into run_pipeline(). "
             "'baseline' = no vision forcing; 'nu_ad_fix' = the TimeSeriesExam1 "
             "switches (no_vision noise + forced_vision anomaly), UNCHANGED.",
    )
    parser.add_argument(
        "--config-json", default=None,
        help="JSON object of run_pipeline kwargs injected verbatim (the "
             "autonomous-search freeform arm). Validated against the run_pipeline "
             "signature; wins over --config. Default off ⇒ byte-identical.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Explicit output label (config_name) so a search/baseline run's CSV "
             "is findable as mmts_results_<subset>_<tag>_*.csv. Defaults to the "
             "--config name (or 'configjson' with --config-json).",
    )
    parser.add_argument(
        "--force-vision", default="",
        help="Comma-separated branches to force vision on (overrides --config "
             "forced_vision). e.g. anomaly",
    )
    parser.add_argument(
        "--no-vision", default="",
        help="Comma-separated branches to suppress vision on (overrides --config "
             "no_vision). e.g. noise",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent rows (default {DEFAULT_MAX_WORKERS}). The GeminiClient "
             "self-throttles to ~3000 RPM.",
    )
    parser.add_argument(
        "--mcq-only", action="store_true",
        help="Skip free-response rows (no A/B/C/D options, e.g. Base 'basic "
             "analysis'/'stationarity'). With the numeric head a DEFAULT run now "
             "scores these via --numeric-rel-tol; keep --mcq-only to reproduce "
             "the honest MCQ-only OA and to isolate the vision experiment.",
    )
    parser.add_argument(
        "--numeric-rel-tol", type=float, default=NUMERIC_REL_TOL,
        help=f"Relative-error tolerance for numeric free-response scoring, pinned "
             f"to MMTS-Bench's Accuracy@N%% protocol (default {NUMERIC_REL_TOL} = "
             f"the paper's Accuracy@10%%). Override only for a pre-registered "
             f"sensitivity sweep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.api_key and args.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Gemini API key not set. Use --api-key or set $GEMINI_API_KEY")
        sys.exit(1)
    if not args.api_key and args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not set. Use --api-key or set $OPENAI_API_KEY")
        sys.exit(1)

    # Resolve the config: --config-json (freeform, validated) wins; else the named
    # frozen MMTS_CONFIGS; then --force-vision/--no-vision overrides. --tag sets the
    # output label (config_name) so the autonomous search can find this run's CSV.
    if args.config_json:
        import inspect
        try:
            raw = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: --config-json is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(raw, dict):
            print("ERROR: --config-json must be a JSON object of run_pipeline kwargs")
            sys.exit(1)
        allowed = set(inspect.signature(run_pipeline).parameters) - {
            "row", "llm_client", "router_client", "answer_client",
            "return_trace", "trace_metadata",
        }
        bad = set(raw) - allowed
        if bad:
            print(f"ERROR: --config-json has unknown run_pipeline kwargs: {sorted(bad)}")
            sys.exit(1)
        vision_kwargs = dict(raw)
        base_label = "configjson"
    else:
        vision_kwargs = dict(MMTS_CONFIGS[args.config])
        base_label = args.config
    overrides = []
    if args.force_vision:
        vision_kwargs["forced_vision_branches"] = [
            b.strip() for b in args.force_vision.split(",") if b.strip()
        ]
        overrides.append("fv-" + "-".join(vision_kwargs["forced_vision_branches"]))
    if args.no_vision:
        vision_kwargs["no_vision_branches"] = [
            b.strip() for b in args.no_vision.split(",") if b.strip()
        ]
        overrides.append("nv-" + "-".join(vision_kwargs["no_vision_branches"]))
    if args.tag:
        config_name = args.tag
    elif overrides:
        config_name = f"{base_label}_{'_'.join(overrides)}"
    else:
        config_name = base_label

    run_evaluation(
        data_path      = args.data_path,
        subset         = args.subset,
        max_rows       = args.max_rows,
        output_dir     = Path(args.output_dir),
        use_hint       = args.use_hint,
        gemini_api_key = args.api_key,
        provider       = args.provider,
        model          = args.model,
        shuffle        = args.shuffle,
        seed           = args.seed,
        resume_from    = args.resume_from,
        config_name    = config_name,
        vision_kwargs  = vision_kwargs,
        max_workers    = args.max_workers,
        mcq_only       = args.mcq_only,
        numeric_rel_tol = args.numeric_rel_tol,
    )
