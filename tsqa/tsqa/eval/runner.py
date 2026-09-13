import re
import concurrent.futures
from collections import Counter
import numpy as np
from tqdm.auto import tqdm

from ..branches import (
    run_trend,
    run_periodicity,
    run_anomaly,
    run_noise,
    run_similarity,
    run_causality,
)
from ..verifier import (
    verify,
    evaluate_visual_trigger,
    evaluate_visual_trigger_learned,
    validate_schema_compliance,
)
from ..tools import generate_ts_artifact
from ..tools.cwt import generate_cwt_artifact, generate_dual_cwt_artifact
from ..router import build_router_prompt, ROUTER_SYSTEM_PROMPT, parse_routing
from ..router.parser import infer_expected_schema
from .numeric_head import (
    infer_numeric_quantity,
    compute_numeric_answer,
    answer_categorical,
    parse_plan_deterministic,
    evaluate_plan,
)
from .sample import Channel, OptionType
from .tsr_heads import (
    HeadResult,
    event_head as _default_event_head,
    forecast_rank_head,
    scalar_mcq_head,
)
from ..llm import (
    build_answer_prompt,
    build_oneshot_prompt,
    build_llm_only_prompt,
    build_llm_numeric_prompt,
    SYSTEM_PROMPT,
    parse_answer,
    build_fallback_prompt,
    build_numeric_prompt,
    parse_numeric_answer,
    build_numeric_planner_prompt,
    parse_numeric_plan_json,
)
from ..llm.vision import analyze_image
from ..llm.prompt import _visual_guidance
from ..orchestrator.quality_gate import evaluate_quality, QUALITY_THRESHOLD
from ..orchestrator.candidates import get_keyword_candidates
from ..orchestrator.critic import CRITIC_SYSTEM_PROMPT, build_critic_prompt, parse_critic_selection
from ..orchestrator.action_proposer import (
    PROPOSER_SYSTEM_PROMPT,
    build_proposal_prompt,
    parse_proposal,
    is_branch_action,
)
from .hooks import ForecastRankResult, select_top2_nocritic
from .trace import attach_trace

# maps router branch name → branch function
# single-series branches take (ts, scope)
# dual-series branches take (ts1, ts2, scope)
_SINGLE_BRANCHES = {
    "trend": run_trend,
    "periodicity": run_periodicity,
    "anomaly": run_anomaly,
    "noise": run_noise,
}
_DUAL_BRANCHES = {
    "similarity": run_similarity,
    "causality": run_causality,
}
ALL_BRANCHES = set(_SINGLE_BRANCHES) | set(_DUAL_BRANCHES)


def _gold_letter(row: dict):
    """Return the option letter for the correct answer, or None if not found."""
    answer = str(row.get("answer", "")).strip()
    options = row.get("options")
    if options is None:
        options = []
    elif hasattr(options, "tolist"):
        options = options.tolist()

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(options[:len(letters)]):
        if re.sub(r"\s+", " ", str(opt).strip().lower()) == re.sub(
            r"\s+", " ", answer.lower()
        ):
            return letters[i]
    return None


def _extract_confidence(raw: str):
    text = raw or ""
    m = re.search(
        r"\bconfidence\s*[:=-]\s*(high|medium|low|0(?:\.\d+)?|1(?:\.0+)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def _build_vision_answer_prompt(
    question: str, options: list, subtype: str | None, evidence: dict
) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_block = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options[:len(letters)])
    )
    compact_evidence = {
        key: evidence.get(key)
        for key in (
            "branch",
            "scope",
            "best_fit_type",
            "direction",
            "r2",
            "exp_r2",
            "log_r2",
            "dominant_period_fft",
            "period_reconciled",
            "seasonal_strength",
            "spectral_entropy",
            "waveform_hint",
            "peak_regularity",
            "rise_fall_ratio",
            "decomp_type",
            "amplitude_change",
            "n_breakpoints",
            "breakpoint_positions",
            "largest_mean_shift",
            "largest_std_shift",
        )
        if key in evidence
    }

    # Include the vision sensor's topological summary if available — it often
    # contains the most discriminative description (e.g. "square-wave like pattern
    # with sharp transitions") and should be surfaced explicitly to the answer LLM.
    vision_sensor_summary = None
    vs = evidence.get("vision_struct")
    if isinstance(vs, dict) and vs.get("summary"):
        vision_sensor_summary = vs["summary"]

    vision_block = (
        f"\nVision sensor description (from topological analysis of the image):\n"
        f"\"{vision_sensor_summary}\"\n"
        "Treat this as a strong signal — prefer it over generic evidence fields "
        "when the question asks about visual shape or waveform type.\n"
        if vision_sensor_summary
        else ""
    )

    return (
        "You are providing an independent visual suggestion for a time-series "
        "multiple-choice question.\n"
        "Use the provided image as the primary signal. Use the compact numeric "
        "evidence only as light context, and do not overrule clear numeric evidence "
        "for exact value questions.\n\n"
        f"Subtype: {subtype}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer choices:\n{options_block}\n\n"
        f"Compact evidence:\n{compact_evidence}\n"
        f"{vision_block}\n"
        "Return exactly this format:\n"
        "LETTER\n"
        "Confidence: high|medium|low\n"
        "One sentence explaining the visual cue."
    )


def _run_vision_answer_suggestion(
    llm_client, question: str, options: list, subtype: str | None, evidence: dict
) -> dict:
    artifacts = evidence.get("artifacts") if isinstance(evidence, dict) else None
    if not artifacts:
        return {"parse_success": False, "raw": "no visual artifact"}

    prompt = _build_vision_answer_prompt(question, options, subtype, evidence)
    raw = llm_client.generate(
        "You are a careful visual time-series analyst.",
        prompt,
        artifacts=artifacts,
    )
    parsed = parse_answer(raw)
    return {
        "letter": parsed.get("letter"),
        "parse_success": parsed.get("parse_success", False),
        "confidence": _extract_confidence(raw),
        "raw": raw,
    }


def _extract_series(row: dict):
    """Return (ts, ts1, ts2) as numpy arrays or None."""

    def _arr(x):
        return np.array(x, dtype=float) if x is not None else None

    return _arr(row.get("ts")), _arr(row.get("ts1")), _arr(row.get("ts2"))


# ---------------------------------------------------------------------------
# Multi-branch helpers (used only when multi_branch=True)
# ---------------------------------------------------------------------------

def _run_branch_safe(branch_name, primary, ts1, ts2, scope):
    """Run one branch deterministically; return (branch_name, evidence_or_error_dict)."""
    try:
        if branch_name in _DUAL_BRANCHES:
            t1 = ts1 if ts1 is not None else primary
            t2 = ts2 if ts2 is not None else primary
            ev = _DUAL_BRANCHES[branch_name](t1, t2, scope=scope)
        else:
            if primary is None:
                return branch_name, {"branch": branch_name, "flags": ["no_series_data"]}
            ev = _SINGLE_BRANCHES[branch_name](primary, scope=scope)
        return branch_name, verify(ev)
    except Exception as exc:
        return branch_name, {"branch": branch_name, "flags": ["branch_error"], "error": str(exc)}


def _execute_branch_verified(branch_name, primary, ts1, ts2, active_series, scope):
    """Run the selected existing branch path and return verified evidence."""
    if branch_name in _DUAL_BRANCHES:
        t1 = ts1 if ts1 is not None else primary
        t2 = ts2 if ts2 is not None else primary
        raw_ev = _DUAL_BRANCHES[branch_name](t1, t2, scope=scope)
        return verify(raw_ev)

    if primary is None:
        raise ValueError("no_series_data")
    if len(active_series) > 1:
        series_results = {}
        for name, series in active_series:
            series_results[name] = _SINGLE_BRANCHES[branch_name](
                series, scope=scope
            )

        primary_name = active_series[0][0]
        raw_ev = dict(series_results[primary_name])
        raw_ev["series_mode"] = "multi"
        raw_ev["series_keys"] = [name for name, _ in active_series]
        raw_ev["series_lengths"] = {
            name: int(len(series)) for name, series in active_series
        }
        for name, ev in series_results.items():
            raw_ev[f"{name}_{branch_name}"] = dict(ev)
    else:
        raw_ev = _SINGLE_BRANCHES[branch_name](primary, scope=scope)
    return verify(raw_ev)


def _compact_evidence_for_loop(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        return {}
    blocked = {
        "artifacts",
        "vision_raw",
        "vision_answer_raw",
        "raw",
        "raw_answer",
        "raw_router",
    }
    compact = {}
    for key, value in evidence.items():
        if key in blocked:
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            compact[key] = value
    return compact


def _build_reroute_prompt(question, prior_routing, prior_branch, prior_flags, evidence):
    # Route on question semantics only. Like build_router_prompt (the initial
    # router, which takes only the question + series-presence flags), this
    # deliberately OMITS the A/B/C/D option block: showing the reroute router the
    # answer letters would add a letter-context surface the base router lacks
    # (no-letter-nudging norm) and make initial-vs-reroute routing non-comparable. (#9)
    return (
        "The previous deterministic evidence pass was incomplete. Re-route the "
        "same time-series question once using the existing branch taxonomy only.\n"
        "Do not invent tools or actions; output only the normal router JSON.\n\n"
        f"Question:\n{question}\n\n"
        f"Prior route:\n{prior_routing}\n\n"
        f"Prior branch: {prior_branch}\n"
        f"Prior verifier flags: {prior_flags}\n"
        f"Compact prior evidence:\n{_compact_evidence_for_loop(evidence)}\n\n"
        "Return the routing JSON now."
    )


def _maybe_reroute_evidence_once(
    *,
    question,
    routing,
    branch_name,
    subtype,
    scope,
    evidence,
    router_llm,
    primary,
    ts1,
    ts2,
    active_series,
    enabled,
    loop_branches,
    max_loop_depth,
    loop_on_flags,
    loop_mode,
):
    """Bounded Stage-1 evidence retry; inert unless all loop gates pass."""
    if loop_mode != "reroute_once":
        return routing, branch_name, subtype, scope, evidence, None
    if max_loop_depth <= 0 or not loop_branches:
        return routing, branch_name, subtype, scope, evidence, None
    allowed_loop_branches = set(loop_branches)
    trigger_names = set(loop_on_flags)

    current_flags = list((evidence or {}).get("flags") or [])
    trigger_flags = [f for f in current_flags if f in trigger_names]
    if branch_name not in allowed_loop_branches or not trigger_flags:
        return routing, branch_name, subtype, scope, evidence, None

    original_branch = branch_name
    meta = {
        "loop_used": True,
        "loop_depth": 0,
        "loop_trigger_flags": trigger_flags,
        "loop_original_branch": original_branch,
        "loop_final_branch": branch_name,
        "loop_error": None,
    }

    for depth in range(max_loop_depth):
        prompt = _build_reroute_prompt(
            question, routing, branch_name, trigger_flags, evidence
        )
        try:
            raw_loop_router = router_llm.generate(ROUTER_SYSTEM_PROMPT, prompt)
            reroute = parse_routing(raw_loop_router)
        except Exception as exc:
            meta["loop_error"] = f"reroute_exception: {exc}"
            break

        if not reroute.get("parse_success"):
            meta["loop_error"] = "reroute_parse_failed"
            meta["raw_loop_router"] = reroute.get("raw")
            break

        new_branch = reroute.get("branch")
        if new_branch == branch_name:
            # Router declined to move. Re-running the same branch re-executes
            # identical tools for identical evidence — a no-op that burns the depth
            # budget for zero gain (at temp 0 it could also spin). Stop before the
            # re-execution below.
            meta["loop_stopped"] = "same_branch_noop"
            meta["raw_loop_router"] = reroute.get("raw")
            break
        if new_branch not in enabled:
            meta["loop_error"] = f"branch_{new_branch}_disabled"
            meta["raw_loop_router"] = reroute.get("raw")
            break

        new_scope = reroute.get("scope", "global")
        try:
            new_evidence = _execute_branch_verified(
                new_branch, primary, ts1, ts2, active_series, new_scope
            )
        except Exception as exc:
            meta["loop_error"] = f"branch_error: {exc}"
            meta["raw_loop_router"] = reroute.get("raw")
            break

        routing = reroute
        branch_name = new_branch
        subtype = reroute.get("subtype")
        scope = new_scope
        evidence = new_evidence
        meta["loop_depth"] = depth + 1
        meta["loop_final_branch"] = branch_name
        meta["raw_loop_router"] = reroute.get("raw")

        current_flags = list((evidence or {}).get("flags") or [])
        trigger_flags = [f for f in current_flags if f in trigger_names]
        if not trigger_flags:
            break
        meta["loop_trigger_flags"] = trigger_flags
        if depth + 1 >= max_loop_depth:
            meta["loop_stopped"] = "max_depth"
            break
        if branch_name not in allowed_loop_branches:
            break

    if isinstance(evidence, dict):
        evidence = dict(evidence)
        evidence["loop_metadata"] = {
            k: v for k, v in meta.items() if k != "raw_loop_router"
        }
    return routing, branch_name, subtype, scope, evidence, meta


# ---------------------------------------------------------------------------
# Stage-2 iterative tool refinement (loop_mode="refine")
# ---------------------------------------------------------------------------
# When the PRIMARY branch's deterministic evidence scores below the quality gate
# (`evaluate_quality` < QUALITY_THRESHOLD = 0.55), the loop ACCUMULATES a second
# (and, up to `max_loop_depth`, third) candidate branch's evidence into the same
# evidence dict — it does NOT re-route/replace and it does NOT re-prompt any LLM.
# Candidate selection is the pure-keyword `get_keyword_candidates` (no LLM); the
# extra branches are deterministic numpy tools; the critic LLM is never invoked.
# This is the `multi_branch` orchestrator turned into a bounded, additive loop
# (research_journal/02_react_stages.md §Stage-2): correction lives in the
# tool/evidence layer (SVI / ADR-001). The richer accumulated evidence is then
# seen by the SINGLE existing answer-LLM call. On a row whose primary evidence
# already clears the gate this is a strict no-op ⇒ byte-identical to baseline.


def _maybe_refine_evidence(
    *,
    question,
    branch_name,
    subtype,
    scope,
    evidence,
    primary,
    ts1,
    ts2,
    active_series,
    enabled,
    loop_branches,
    max_loop_depth,
    loop_mode,
    quality_threshold,
):
    """Stage-2 evidence accumulation; inert unless `loop_mode='refine'` and the
    primary branch's quality score is below `quality_threshold`.

    Returns ``(evidence, meta)`` where ``meta`` is None when the refine path was
    not entered at all (i.e. byte-identical-to-baseline rows carry no extra keys),
    or a dict of diagnostics (``refine_fired``, ``refine_depth``,
    ``quality_score``, ``refine_branches``, ``refine_quality_trace``) otherwise.

    Adds **0 extra LLM calls** — candidate selection is keyword-only and the extra
    branches are deterministic tools. The depth is hard-bounded by both
    ``max_loop_depth`` and the exhaustion of the keyword candidate list.
    """
    if loop_mode != "refine":
        return evidence, None
    if max_loop_depth <= 1 or not loop_branches:
        # max_loop_depth<=1 means "no extra branch may be accumulated" ⇒ off.
        return evidence, None
    if not isinstance(evidence, dict):
        return evidence, None

    allowed = set(loop_branches)
    if branch_name not in allowed:
        return evidence, None

    # Score the PRIMARY branch's evidence with the pre-existing principled gate.
    primary_qr = evaluate_quality(evidence, None, None, branch_name, subtype)
    primary_score = primary_qr.score

    # Always record the (newly-computed) quality score for the primary branch —
    # this is the signal the Stage-1 lesson said was missing (None on all rows).
    meta = {
        "refine_fired": False,
        "refine_depth": 0,
        "quality_score": primary_score,
        "refine_branches": [],
        "refine_quality_trace": [primary_score],
    }

    if primary_score >= quality_threshold:
        # Gate cleared on the primary evidence ⇒ accumulate nothing. The pipeline
        # is byte-identical to baseline on this row EXCEPT for the diagnostic
        # quality_score (which baseline never computed and the answer prompt never
        # sees — math_evidence excludes the `refine_*` keys we did not add).
        return evidence, meta

    # Quality below threshold: pull the keyword candidate list (pure keyword, no
    # LLM) and accumulate up to (max_loop_depth - 1) ADDITIONAL branches' evidence.
    # The "-1" budgets the primary branch as depth 0; each accumulated branch is a
    # further depth step, capped at max_loop_depth.
    candidates = get_keyword_candidates(
        question, branch_name, max_candidates=max_loop_depth + 1
    )
    accumulated = dict(evidence)
    already_used = {branch_name}
    refine_branches = []
    quality_trace = [primary_score]
    current_score = primary_score

    for cand in candidates:
        if len(refine_branches) >= (max_loop_depth - 1):
            break
        if cand in already_used:
            continue
        if cand not in allowed or cand not in enabled:
            continue

        cand_name, cand_ev = _run_branch_safe(cand, primary, ts1, ts2, scope)
        already_used.add(cand_name)
        if not isinstance(cand_ev, dict):
            continue
        # ACCUMULATE (don't replace): the secondary branch's verified evidence is
        # merged under a namespaced sub-dict so it flows into the reasoner's
        # math_evidence bundle without clobbering the primary branch's fields.
        accumulated[f"refine_{cand_name}"] = cand_ev
        # Surface the secondary branch's flags at top level (de-duplicated) so the
        # verifier-flag view and the answer prompt see them too.
        cand_flags = list(cand_ev.get("flags") or [])
        merged_flags = list(accumulated.get("flags") or [])
        for f in cand_flags:
            if f not in merged_flags:
                merged_flags.append(f)
        accumulated["flags"] = merged_flags
        refine_branches.append(cand_name)

        # Re-score: the combined evidence's quality is the MAX over the primary
        # and each accumulated branch (accumulation can only add a stronger
        # signal; it must never lower the recorded quality). Stop early once the
        # gate clears.
        cand_score = evaluate_quality(cand_ev, None, None, cand_name, subtype).score
        current_score = max(current_score, cand_score)
        quality_trace.append(current_score)
        if current_score >= quality_threshold:
            break

    if not refine_branches:
        # No usable second branch was found (candidates exhausted / all disabled).
        # Evidence is unchanged apart from the recorded quality_score diagnostic.
        meta["quality_score"] = current_score
        meta["refine_quality_trace"] = quality_trace
        return evidence, meta

    accumulated["refine_metadata"] = {
        "refine_fired": True,
        "refine_depth": len(refine_branches),
        "refine_branches": refine_branches,
        "quality_score": current_score,
    }
    meta.update(
        refine_fired=True,
        refine_depth=len(refine_branches),
        quality_score=current_score,
        refine_branches=refine_branches,
        refine_quality_trace=quality_trace,
    )
    return accumulated, meta


# ---------------------------------------------------------------------------
# Stage-3 LLM-proposed actions (loop_mode="propose") — the LAST ReAct stage
# ---------------------------------------------------------------------------
# Gated on the SAME live signal as Stage 2 (`evaluate_quality` < quality_threshold).
# Where Stage 2's keyword `get_keyword_candidates` returned 0 second-branches on
# every sub-threshold row (log/005, the binding constraint), Stage 3 calls the LLM
# ONCE per step to PROPOSE the next action from the closed registry
# (orchestrator/action_proposer.ACTION_REGISTRY = 6 branches + vision + numeric_head).
# The proposal is parsed + validated against the typed schema; an invalid /
# out-of-registry / over-budget / already-run proposal falls back DETERMINISTICALLY
# (precondition 4). A valid branch action ACCUMULATES its deterministic evidence in
# the tool layer (SVI, namespaced key) exactly like refine; a valid `vision` action
# sets a flag the existing vision gate honors; `numeric_head` is recorded but is a
# structural no-op on MCQ rows (they can never enter the numeric head). The proposal
# is a ROUTING decision — the answer LLM is NEVER re-prompted to re-answer; the one
# existing answer call sees the enriched evidence. Inert unless loop_mode="propose"
# AND max_loop_depth>0 AND the branch is in loop_branches AND quality < threshold ⇒
# byte-identical to baseline everywhere else.


def _maybe_propose_actions(
    *,
    question,
    branch_name,
    subtype,
    scope,
    evidence,
    primary,
    ts1,
    ts2,
    active_series,
    enabled,
    proposer_llm,
    loop_branches,
    max_loop_depth,
    loop_mode,
    quality_threshold,
):
    """Stage-3 LLM-proposed action accumulation; inert unless ``loop_mode='propose'``
    and the primary branch's quality score is below ``quality_threshold``.

    Returns ``(evidence, meta)`` where ``meta`` is None when the propose path was
    not entered (byte-identical-to-baseline rows carry no extra keys), or a dict of
    auditable diagnostics otherwise (``propose_fired``, ``propose_quality_score``,
    ``propose_steps`` — the full per-step trace of proposed/validated/executed
    actions and fallbacks, ``propose_executed`` actions, ``propose_force_vision``,
    ``propose_fallbacks``).

    Budget (precondition 3): at most ``max_loop_depth`` executed actions and exactly
    one proposal-LLM call per step. Each step that produces an invalid / already-run
    proposal is a deterministic fallback (recorded, never retried) and STOPS the loop
    (a fresh evidence state would otherwise re-elicit the same proposal at temp 0).
    """
    if loop_mode != "propose":
        return evidence, None
    if max_loop_depth <= 0 or not loop_branches:
        return evidence, None
    if not isinstance(evidence, dict):
        return evidence, None

    allowed = set(loop_branches)
    if branch_name not in allowed:
        return evidence, None

    primary_qr = evaluate_quality(evidence, None, None, branch_name, subtype)
    primary_score = primary_qr.score

    meta = {
        "propose_fired": False,
        "propose_quality_score": primary_score,
        "propose_steps": [],          # full audit trace (one entry per LLM call)
        "propose_executed": [],       # actions whose tools actually ran
        "propose_force_vision": False,
        "propose_fallbacks": [],      # why a step fell back (parser-failure contract)
    }

    if primary_score >= quality_threshold:
        # Gate cleared on the primary evidence ⇒ propose nothing. Byte-identical to
        # baseline on this row save for the diagnostic quality_score (not in the
        # answer prompt's math_evidence, so the letter cannot move).
        return evidence, meta

    accumulated = dict(evidence)
    already_run = {branch_name}          # the primary branch already executed
    executed = []
    steps = []
    fallbacks = []
    force_vision = False

    # Budget: <= max_loop_depth EXECUTED actions; one proposal LLM call per step.
    for _ in range(max_loop_depth):
        prompt = build_proposal_prompt(
            question, branch_name, accumulated, already_run=already_run
        )
        try:
            raw = proposer_llm.generate(PROPOSER_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            fallbacks.append(f"proposer_exception: {exc}")
            steps.append({"proposed": None, "valid": False, "executed": False,
                          "fallback": "proposer_exception"})
            break

        proposal = parse_proposal(raw)
        if proposal is None:
            # Precondition 4: unparseable / invalid / out-of-registry ⇒ deterministic
            # fallback, NEVER retried.
            fallbacks.append("invalid_proposal")
            steps.append({"proposed": None, "valid": False, "executed": False,
                          "fallback": "invalid_proposal"})
            break

        action = proposal.action
        step = {"proposed": action, "reason": proposal.reason,
                "valid": True, "executed": False, "fallback": None}

        if action in already_run:
            # Re-proposing an already-executed action is a no-op (would re-run
            # identical tools for identical evidence and burn budget). Fall back.
            step["fallback"] = "already_run"
            fallbacks.append(f"already_run:{action}")
            steps.append(step)
            break

        if action == "vision":
            # Routing decision: flag the existing vision gate to fire (the actual
            # render + structured-vision sensor run in the runner's vision block,
            # AFTER this hook, preserving SVI). No evidence accumulated here.
            force_vision = True
            already_run.add(action)
            executed.append(action)
            step["executed"] = True
            steps.append(step)
            if len(executed) >= max_loop_depth:
                break
            continue

        if action == "numeric_head":
            # MCQ rows (the only rows that reach this gate on TSExam) can NEVER
            # enter the numeric head (expected_schema=='mcq' is authoritative), so
            # there is no exact-statistic tool to run here. Record the proposal as
            # a structural no-op rather than fabricating a value (honest coverage
            # gap). Counts against neither budget nor evidence; STOP (the same
            # proposal would re-elicit at temp 0).
            step["executed"] = False
            step["fallback"] = "numeric_head_unavailable_on_mcq"
            fallbacks.append("numeric_head_unavailable_on_mcq")
            steps.append(step)
            break

        # A branch action: run its deterministic tools and ACCUMULATE (don't
        # replace) under a namespaced key — identical SVI discipline to refine.
        if is_branch_action(action) and action in enabled:
            cand_name, cand_ev = _run_branch_safe(action, primary, ts1, ts2, scope)
            already_run.add(cand_name)
            if isinstance(cand_ev, dict):
                accumulated[f"propose_{cand_name}"] = cand_ev
                cand_flags = list(cand_ev.get("flags") or [])
                merged_flags = list(accumulated.get("flags") or [])
                for f in cand_flags:
                    if f not in merged_flags:
                        merged_flags.append(f)
                accumulated["flags"] = merged_flags
                executed.append(cand_name)
                step["executed"] = True
            else:
                step["fallback"] = "branch_returned_non_dict"
                fallbacks.append(f"branch_non_dict:{action}")
            steps.append(step)
            if len(executed) >= max_loop_depth:
                break
            continue

        # Branch disabled by enabled_branches ⇒ deterministic fallback.
        step["fallback"] = "action_disabled"
        fallbacks.append(f"action_disabled:{action}")
        steps.append(step)
        break

    meta["propose_steps"] = steps
    meta["propose_executed"] = executed
    meta["propose_force_vision"] = force_vision
    meta["propose_fallbacks"] = fallbacks

    if not executed:
        # Nothing ran (all steps fell back) ⇒ evidence unchanged apart from the
        # recorded quality_score diagnostic. propose_fired stays False.
        meta["propose_quality_score"] = primary_score
        return evidence, meta

    accumulated["propose_metadata"] = {
        "propose_fired": True,
        "propose_executed": executed,
        "propose_force_vision": force_vision,
        "quality_score": primary_score,
    }
    meta["propose_fired"] = True
    return accumulated, meta


# ---------------------------------------------------------------------------
# Self-Consistency control arm (Stage-1 constraint #4; Wang et al. 2203.11171)
# ---------------------------------------------------------------------------
# The cheaper lever the Stage-1 loop must beat. On rows where the SAME trigger as
# the loop fires (the `evidence_incomplete` verifier flag on the post-verify,
# pre-reroute branch evidence), instead of re-routing we sample the *answer* LLM
# k times at temp>0 and majority-vote the letter. On every other row the pipeline
# is byte-identical to baseline (this whole path is dark unless `self_consistency`
# k>0 AND the flag fires), so the frozen-config byte-identity guard is preserved.

SC_DEFAULT_K = 5
SC_DEFAULT_TEMPERATURE = 0.7


def _sc_majority_vote(letters: list) -> str | None:
    """Majority-vote a letter from k Self-Consistency samples.

    Tie-break (DETERMINISTIC, documented): among the letters sharing the top
    vote count, pick the alphabetically-smallest. Parse failures (None) are
    dropped before counting; if every sample failed to parse, return None so the
    caller records an abstention rather than a fabricated letter. The alphabetic
    rule is arbitrary-but-fixed — it exists only to keep the control reproducible
    under temp>0, and the matched paired diff is what actually judges the arm.
    """
    valid = [str(l).strip().upper() for l in letters if l]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    tied = sorted(l for l, c in counts.items() if c == top)
    return tied[0]


def _sc_sample_answer(answer_llm, system_prompt, user_prompt, k, temperature):
    """Sample the answer LLM k times at temp>0 and majority-vote the letter.

    Returns ``(letter, parse_success, raws)``. Each call passes the per-call
    ``temperature`` override (threaded through GeminiClient/OpenAIClient kwargs);
    no other call's temperature changes. A per-sample exception is swallowed (its
    vote is simply lost) so a single transient failure cannot nuke the row — the
    vote proceeds on whatever samples returned.
    """
    raws, letters = [], []
    for _ in range(max(1, int(k))):
        try:
            raw = answer_llm.generate(
                system_prompt, user_prompt, temperature=temperature
            )
        except Exception as exc:  # one bad sample must not kill the row
            raws.append(f"<sample_error: {exc}>")
            continue
        raws.append(raw)
        letters.append(parse_answer(raw).get("letter"))
    letter = _sc_majority_vote(letters)
    return letter, (letter is not None), raws


def _select_best_branch(
    candidates, primary, ts1, ts2, scope, subtype, question, options, llm_client,
    recovery: str | None = None,
):
    """
    Run candidate branches concurrently, score with evaluate_quality, and call
    the critic LLM at most once (on tie within 0.10 or all-fail).

    Returns (best_branch_name, best_evidence_dict, quality_score, critic_called).
    """
    if not candidates:
        return None, {}, 0.0, False

    # Parallel deterministic branch runs (I/O-bound on numpy, not API)
    candidate_evidences = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidates), 3)) as ex:
        futs = {
            ex.submit(_run_branch_safe, b, primary, ts1, ts2, scope): b
            for b in candidates
        }
        for fut in concurrent.futures.as_completed(futs):
            branch_name, ev = fut.result()
            candidate_evidences[branch_name] = ev

    # Score each candidate
    scores = {}
    for b, ev in candidate_evidences.items():
        qr = evaluate_quality(ev, None, None, b, subtype)
        scores[b] = qr

    if recovery == "top2_nocritic":
        arb = select_top2_nocritic(
            candidates,
            candidate_evidences,
            {b: qr.score for b, qr in scores.items()},
        )
        if arb.chosen_branch is None:
            return None, {}, 0.0, False
        best_qr = scores[arb.chosen_branch]
        return (
            arb.chosen_branch,
            candidate_evidences[arb.chosen_branch],
            best_qr.score,
            arb.critic_called,
        )

    sorted_cands = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)
    best_branch, best_qr = sorted_cands[0]

    critic_called = False
    if len(sorted_cands) >= 2:
        top_score = sorted_cands[0][1].score
        second_score = sorted_cands[1][1].score
        tie = (top_score - second_score) < 0.10
        fail = top_score < QUALITY_THRESHOLD
        if (tie or fail) and llm_client is not None:
            top2 = [(b, candidate_evidences[b], qr.score) for b, qr in sorted_cands[:2]]
            critic_prompt = build_critic_prompt(question, options, top2)
            try:
                raw_critic = llm_client.generate(CRITIC_SYSTEM_PROMPT, critic_prompt)
                picked = parse_critic_selection(raw_critic, [b for b, _ in sorted_cands[:2]])
                if picked:
                    best_branch = picked
                    best_qr = scores[best_branch]
                    critic_called = True
            except Exception:
                pass  # critic failed — keep quality-gate winner

    return best_branch, candidate_evidences[best_branch], best_qr.score, critic_called


# ---------------------------------------------------------------------------
# SA vision artifact dispatcher (used when enable_sa_vision=True)
# ---------------------------------------------------------------------------

def _generate_sa_artifact(ts_vis, sa_viz_mode, series_id):
    """Dispatch to the right renderer for the SA visual comparison."""
    mode = (sa_viz_mode or "stacked").lower()
    if mode == "dual_cwt":
        if ts_vis.ndim == 2 and ts_vis.shape[0] == 2:
            return generate_dual_cwt_artifact(ts_vis[0], ts_vis[1], series_id)
        elif ts_vis.ndim == 1:
            return generate_cwt_artifact(ts_vis, series_id)
        else:
            # Fall back to stacked for > 2 series
            return generate_ts_artifact(ts_vis, "stacked", series_id)
    elif mode == "overlay_line":
        return generate_ts_artifact(ts_vis, "line_plot", series_id)
    else:  # "stacked" (default)
        return generate_ts_artifact(ts_vis, "stacked", series_id)


def _normalize_row(row) -> dict:
    if hasattr(row, "_asdict"):
        row = row._asdict()
    elif not isinstance(row, dict):
        try:
            row = dict(row)
        except Exception:
            row = {}
            for key in (
                "id",
                "category",
                "question",
                "options",
                "answer",
                "ts",
                "ts1",
                "ts2",
            ):
                if hasattr(row, key):
                    row[key] = getattr(row, key)

    if "category" not in row:
        row["category"] = "unknown"

    return row


def _apply_row_control(row: dict, mode: str | None, seed: int = 0) -> dict:
    if not mode:
        return row
    out = dict(row)
    if mode == "option_only":
        out["question"] = "Select the best answer from the options."
        out["ts"] = out["ts1"] = out["ts2"] = None
        return out
    if mode == "metadata_only":
        out["question"] = "Answer using only generic benchmark metadata."
        out["options"] = []
        out["ts"] = out["ts1"] = out["ts2"] = None
        out.pop("category", None)
        return out
    if mode == "no_series":
        out["ts"] = out["ts1"] = out["ts2"] = None
        return out
    if mode == "shuffled_series":
        rng = np.random.default_rng(seed)
        for key in ("ts", "ts1", "ts2"):
            if out.get(key) is not None:
                values = np.asarray(out[key], dtype=float).copy()
                rng.shuffle(values)
                out[key] = values
        return out
    raise ValueError(f"unknown control mode: {mode}")


def run_pipeline(
    row: dict,
    llm_client,
    enabled_branches=None,
    use_hint: bool = False,
    router_client=None,
    answer_client=None,
    multi_branch: bool = False,
    enable_sa_vision: bool = False,
    sa_viz_mode: str = "stacked",
    forced_vision_branches=None,
    no_vision_branches=None,
    numeric_use_llm_format: bool = False,
    numeric_parser: str = "keyword",
    vision_gate: str = "additive",
    recovery: str | None = None,
    mcq_numeric_head: bool = False,
    forecast_ranker=None,
    event_head=None,
    answer_type: str | None = None,
    option_type: str | None = None,
    return_trace: bool = False,
    trace_metadata: dict | None = None,
    control: str | None = None,
    control_seed: int = 0,
    loop_branches: list[str] | None = None,
    max_loop_depth: int = 0,
    loop_on_flags: list[str] | None = None,
    loop_mode: str = "reroute_once",
    quality_threshold: float = QUALITY_THRESHOLD,
    self_consistency: int = 0,
    self_consistency_temperature: float = SC_DEFAULT_TEMPERATURE,
    llm_only: bool = False,
    llm_numeric: bool = False,
    raw_pixel_vision: bool = False,
) -> dict:
    """
    Run the full TS-QA pipeline on a single dataset row.

    Stages:
      1. Router  — LLM decides branch + scope
      2. Branch  — deterministic tool calls → evidence
      3. Verifier — rule checks → flags
      4. LLM interpreter — evidence + flags → A/B/C/D
      Fallback   — triggered if routing fails or evidence_incomplete is flagged

    Args:
        row:              dict-like dataset row with keys: question, options,
                          answer, category, ts, ts1, ts2
        llm_client:       GeminiClient instance (or any object with .generate(sys, usr))
        enabled_branches: optional set of branch names to allow; others fall back.
                          Pass None to enable all branches.
        use_hint:         if True, prepend the row's `question_hint` field to the
                          LLM interpreter prompt (hint-augmented prompting).
                          Requires the dataset rows to have a `question_hint` column.
        router_client:    optional client for router LLM (defaults to llm_client)
        answer_client:    optional client for answer LLM (defaults to llm_client)
        multi_branch:     if True, run up to 3 candidate branches in parallel, score
                          with the quality gate, and call the critic LLM on tie/failure.
                          Default False (preserves original single-branch behaviour).
        enable_sa_vision: if True, force the visual path on for similarity branch and
                          render a sighted comparison image for SA questions.
                          Default False.
        sa_viz_mode:      representation for SA sighted vision. One of:
                            "stacked"      — line-overlay + spectrogram panel (default)
                            "overlay_line" — line overlay only
                            "dual_cwt"     — Morlet CWT scalograms (2-series only)

    Returns dict with keys:
        id, category, predicted_letter, gold_letter, correct,
        routing, branch_used, evidence, flags,
        parse_success, used_fallback, raw_router, raw_answer,
        prompt_mode  ("zero_shot" | "hint_augmented"),
        multi_branch_used, critic_called, quality_score  (new, present when multi_branch=True)
    """
    row = _apply_row_control(_normalize_row(row), control, control_seed)
    enabled = enabled_branches or ALL_BRANCHES
    router_llm = router_client or llm_client
    answer_llm = answer_client or llm_client
    loop_on_flags = (
        ["evidence_incomplete"] if loop_on_flags is None else list(loop_on_flags)
    )

    ts, ts1, ts2 = _extract_series(row)
    primary = ts if ts is not None else ts1
    active_series = []
    if ts1 is not None:
        active_series.append(("ts1", ts1))
    if ts2 is not None:
        active_series.append(("ts2", ts2))
    if not active_series and primary is not None:
        active_series.append(("ts", primary))

    question = row.get("question")
    if not question:
        raise KeyError("question")
    _opts = row.get("options")
    options = list(_opts) if _opts is not None else []
    gold = _gold_letter(row)

    result = {
        "id": row.get("id"),
        "category": row.get("category"),
        "gold_letter": gold,
        "predicted_letter": None,
        "correct": False,
        "routing": None,
        "branch_used": None,
        # initial (pre-reroute) branch — stable stratifier for paired diffs (#1);
        # equals branch_used whenever the Stage-1 loop is off.
        "initial_branch_used": None,
        "subtype_used": None,
        "evidence": None,
        "flags": [],
        "parse_success": False,
        "used_fallback": False,
        "raw_router": None,
        "raw_answer": None,
        "prompt_mode": "hint_augmented" if use_hint else "zero_shot",
        "vision_used": False,
        "vision_letter": None,
        "vision_confidence": None,
        # raw_pixel_vision measurement arm (Track-V structured-vs-pixel study).
        # MEASUREMENT-ONLY — deliberately violates the SVI behind this switch to
        # bound the multimodal ceiling vs the structured-JSON sensor; NEVER
        # promoted/shipped. True only on vision-fired rows when raw_pixel_vision=True
        # (the raw PNG was attached directly to the ANSWER LLM call). On every other
        # row this stays False and the pipeline is byte-identical to baseline.
        "raw_pixel_vision_used": False,
        # numeric head metadata (#4a) — only populated for non-mcq rows; MCQ
        # rows keep expected_schema == "mcq" and predicted_value == None.
        "expected_schema": "mcq",
        "predicted_value": None,
        "numeric_quantity": None,
        "numeric_note": None,
        "used_head": False,
        "head_name": None,
        "head_note": None,
        # multi-branch metadata (populated below when multi_branch=True)
        "multi_branch_used": False,
        "critic_called": False,
        "quality_score": None,
        "branch_candidates": None,
        # Self-Consistency control-arm metadata (#4). sc_sampled is True only on
        # rows where the matched trigger (evidence_incomplete) fired AND k>0; on
        # every other row these stay falsy and the answer path is byte-identical.
        "sc_sampled": False,
        "sc_k": 0,
        # Stage-2 iterative tool refinement metadata (loop_mode="refine"). These
        # are populated ONLY when the refine path is entered (loop_mode="refine",
        # max_loop_depth>1, branch in loop_branches). refine_fired is True only on
        # rows whose PRIMARY quality_score < quality_threshold AND a 2nd branch was
        # accumulated; on every other row refine is a no-op and these stay falsy.
        "refine_fired": False,
        "refine_depth": 0,
        "refine_branches": None,
        # Stage-3 LLM-proposed-action metadata (loop_mode="propose"). Populated
        # ONLY when the propose path is entered (loop_mode="propose",
        # max_loop_depth>0, branch in loop_branches). propose_fired is True only on
        # rows whose PRIMARY quality_score < quality_threshold AND >=1 proposed
        # action actually executed; on every other row these stay falsy and the row
        # is byte-identical to baseline.
        "propose_fired": False,
        "propose_executed": None,
        "propose_steps": None,
        "propose_fallbacks": None,
    }

    def _finalize(res: dict, prompt_template_ids=None) -> dict:
        if not return_trace:
            return res
        client = answer_llm if res.get("raw_answer") is not None else router_llm
        model_id = getattr(client, "model_name", None)
        return attach_trace(
            res,
            routing=res.get("routing"),
            evidence=res.get("evidence"),
            model_id=model_id,
            prompt_template_ids=prompt_template_ids or [],
            metadata=trace_metadata,
        )

    # ------------------------------------------------------------------
    # llm_only: no-architecture control — ask the answer LLM the MCQ directly
    # (same question + options, NO router/branch/vision/evidence, ONE call).
    # Deliberate ablation arm for the fixed-backbone de-confound; not byte-
    # identical to anything. Short-circuits before the router so zero tool or
    # router calls are made. MCQ rows only (free-response rows keep their
    # numeric-head schema label but are answered the same single-call way here —
    # the Track-A experiment runs on the MCQ TSExam set).
    # ------------------------------------------------------------------
    if llm_only:
        return _finalize(
            _run_llm_only(result, question, options, answer_llm), ["llm_only"]
        )

    # ------------------------------------------------------------------
    # llm_numeric: free-response COUNTERFACTUAL — the model-computes arm. On a
    # numerical row, ask the answer LLM to COMPUTE the quantity directly from the
    # raw series (no router, no tools, no evidence, no options); the deterministic
    # numeric head is BYPASSED, so the paired @10% gap vs the head measures whether
    # the tool is load-bearing on a strong backbone. MCQ rows fall back to the
    # llm_only letter path so the arm is well-defined on every row. Short-circuits
    # before the router (zero router/tool calls), like llm_only.
    # ------------------------------------------------------------------
    if llm_numeric:
        schema = infer_expected_schema(
            options, question=question, category=row.get("category")
        )
        result["expected_schema"] = schema
        if schema == "numerical":
            return _finalize(
                _run_llm_numeric(result, question, ts, ts1, ts2, schema, answer_llm),
                ["llm_numeric"],
            )
        # categorical is dormant on MMTS Base (0 rows) and absent on MCQ TSExam;
        # answer those (and any mcq) via the llm_only letter path.
        return _finalize(
            _run_llm_only(result, question, options, answer_llm), ["llm_numeric"]
        )

    # ------------------------------------------------------------------
    # Stage 1: Router
    # ------------------------------------------------------------------
    try:
        router_prompt = build_router_prompt(
            question, has_ts1=ts1 is not None, has_ts2=ts2 is not None
        )
        raw_router = router_llm.generate(ROUTER_SYSTEM_PROMPT, router_prompt)
        routing = parse_routing(raw_router)
    except Exception as e:
        routing = {
            "parse_success": False,
            "branch": None,
            "scope": "global",
            "raw": str(e),
        }
        raw_router = str(e)

    result["routing"] = routing
    result["raw_router"] = routing.get("raw", "")

    branch_name = routing.get("branch")
    scope = routing.get("scope", "global")
    subtype = routing.get("subtype")
    result["subtype_used"] = subtype
    # Record the initial router branch on EVERY row. The Stage-1 reroute (#3) can
    # change branch_used downstream; stratifying a paired diff by branch_used would
    # then condition on a treatment-affected collider. initial_branch_used is the
    # stable pre-reroute stratifier (overridden just below for multi_branch).
    result["initial_branch_used"] = branch_name

    # Structural answer-shape detection (gates the numeric head, Workstream #4a).
    # Deterministic and authoritative — overrides any router-emitted value. Rows
    # with >=2 parseable options are guaranteed "mcq" and can NEVER enter the
    # numeric head, so the MCQ pipeline below stays byte-identical.
    expected_schema = infer_expected_schema(
        options, question=question, category=row.get("category"), subtype=subtype
    )
    result["expected_schema"] = expected_schema

    # fall back if routing failed or branch is disabled
    if not routing.get("parse_success") or branch_name not in enabled:
        return _finalize(_run_fallback(
            result,
            question,
            options,
            answer_llm,
            reason=(
                "router_failed"
                if not routing.get("parse_success")
                else f"branch_{branch_name}_disabled"
            ),
        ), ["fallback"])

    # ------------------------------------------------------------------
    # Multi-branch path (Stage 2 replacement when multi_branch=True)
    # ------------------------------------------------------------------
    if multi_branch and branch_name in enabled:
        candidates = get_keyword_candidates(question, branch_name, max_candidates=3)
        candidates = [b for b in candidates if b in enabled]
        if not candidates:
            candidates = [branch_name]

        best_b, best_ev, q_score, critic_fired = _select_best_branch(
            candidates, primary, ts1, ts2, scope, subtype, question, options, answer_llm,
            recovery=recovery,
        )

        result["multi_branch_used"] = True
        result["critic_called"] = critic_fired
        result["quality_score"] = q_score
        result["branch_candidates"] = candidates

        if best_b is None or not best_ev:
            return _finalize(
                _run_fallback(result, question, options, answer_llm, reason="multi_branch_failed"),
                ["fallback"],
            )

        branch_name = best_b
        result["branch_used"] = branch_name
        # multi_branch picks the executed branch deterministically (config-fixed),
        # so the orchestrator's choice is the correct pre-reroute stratifier.
        result["initial_branch_used"] = branch_name

        # Re-verify to ensure flags are populated (may already be from _run_branch_safe)
        evidence = best_ev if "flags" in best_ev else verify(best_ev)

        # Jump directly to visual trigger + LLM interpreter below
        # (wraps in the same try/except flow by setting raw_ev = evidence and skipping
        # the normal branch execution)
        raw_ev = evidence

    # ------------------------------------------------------------------
    # Stage 2: Branch + Stage 3: Verifier
    # ------------------------------------------------------------------
    try:
        if result.get("multi_branch_used"):
            pass  # evidence already set and verified in multi-branch block above
        elif branch_name in _DUAL_BRANCHES:
            t1 = ts1 if ts1 is not None else primary
            t2 = ts2 if ts2 is not None else primary
            raw_ev = _DUAL_BRANCHES[branch_name](t1, t2, scope=scope)
        else:
            if primary is None:
                return _finalize(_run_fallback(
                    result, question, options, answer_llm, reason="no_series_data"
                ), ["fallback"])
            if len(active_series) > 1:
                series_results = {}
                for name, series in active_series:
                    series_results[name] = _SINGLE_BRANCHES[branch_name](
                        series, scope=scope
                    )

                primary_name = active_series[0][0]
                raw_ev = dict(series_results[primary_name])
                raw_ev["series_mode"] = "multi"
                raw_ev["series_keys"] = [name for name, _ in active_series]
                raw_ev["series_lengths"] = {
                    name: int(len(series)) for name, series in active_series
                }
                for name, ev in series_results.items():
                    raw_ev[f"{name}_{branch_name}"] = dict(ev)
            else:
                raw_ev = _SINGLE_BRANCHES[branch_name](primary, scope=scope)

        if not result.get("multi_branch_used"):
            evidence = verify(raw_ev)

        # Self-Consistency control arm (#4): decide ONCE, on the post-verify /
        # pre-reroute evidence, using the SAME trigger as the Stage-1 loop
        # (`evidence_incomplete`). This makes the two arms a matched comparison —
        # they fire on the identical row set. When active, the final answer step
        # (answer LLM or fallback) is k-sampled at temp>0 and majority-voted
        # instead of a single temp-0 call. k<=0 ⇒ this stays False and the row is
        # byte-identical to baseline.
        sc_active = bool(
            self_consistency
            and self_consistency > 0
            and "evidence_incomplete" in list((evidence or {}).get("flags") or [])
        )
        if sc_active:
            result["sc_sampled"] = True
            result["sc_k"] = int(self_consistency)

        routing, branch_name, subtype, scope, evidence, loop_meta = _maybe_reroute_evidence_once(
            question=question,
            routing=routing,
            branch_name=branch_name,
            subtype=subtype,
            scope=scope,
            evidence=evidence,
            router_llm=router_llm,
            primary=primary,
            ts1=ts1,
            ts2=ts2,
            active_series=active_series,
            enabled=enabled,
            loop_branches=loop_branches,
            max_loop_depth=max_loop_depth,
            loop_on_flags=loop_on_flags,
            loop_mode=loop_mode,
        )
        if loop_meta is not None:
            # Copy the named loop fields explicitly rather than splatting the whole
            # meta dict (which also double-wrote raw_loop_router). Whitelisting the
            # known keys keeps the row's key-set explicit and stops any future meta
            # field from silently leaking into every result row. (#8)
            for _k in (
                "loop_used", "loop_depth", "loop_trigger_flags",
                "loop_original_branch", "loop_final_branch", "loop_error",
                "loop_stopped", "raw_loop_router",
            ):
                if _k in loop_meta:
                    result[_k] = loop_meta[_k]
            result["routing"] = routing

        # ── Stage-2 iterative tool refinement (loop_mode="refine") ───────────
        # Accumulate a 2nd (and up to max_loop_depth-th) candidate branch's
        # deterministic evidence when the PRIMARY branch's quality_score is below
        # quality_threshold. Adds 0 LLM calls; the richer evidence is then seen by
        # the single existing answer-LLM call below. On a row whose primary
        # evidence already clears the gate this is a no-op (byte-identical to
        # baseline) save for the recorded quality_score diagnostic.
        evidence, refine_meta = _maybe_refine_evidence(
            question=question,
            branch_name=branch_name,
            subtype=subtype,
            scope=scope,
            evidence=evidence,
            primary=primary,
            ts1=ts1,
            ts2=ts2,
            active_series=active_series,
            enabled=enabled,
            loop_branches=loop_branches,
            max_loop_depth=max_loop_depth,
            loop_mode=loop_mode,
            quality_threshold=quality_threshold,
        )
        if refine_meta is not None:
            result["refine_fired"] = refine_meta["refine_fired"]
            result["refine_depth"] = refine_meta["refine_depth"]
            # Keep refine_branches None (the default) when nothing was accumulated,
            # so a no-op refine row is field-identical to baseline on this key.
            result["refine_branches"] = refine_meta["refine_branches"] or None
            # The Stage-2 headline: the quality_score is now COMPUTED (it was None
            # on every single-branch row before — the Stage-1 lesson).
            result["quality_score"] = refine_meta["quality_score"]

        # ── Stage-3 LLM-proposed actions (loop_mode="propose") ───────────────
        # On the SAME live gate as Stage 2 (quality < quality_threshold), call the
        # LLM ONCE per step to PROPOSE the next action from the closed registry
        # where the keyword candidate source found none (the Stage-2 binding
        # constraint, log/005). Valid branch proposals accumulate their evidence in
        # the tool layer (SVI); a `vision` proposal flags the existing vision gate
        # below. Invalid / over-budget / already-run proposals fall back
        # deterministically. The answer LLM is NEVER re-prompted to re-answer.
        evidence, propose_meta = _maybe_propose_actions(
            question=question,
            branch_name=branch_name,
            subtype=subtype,
            scope=scope,
            evidence=evidence,
            primary=primary,
            ts1=ts1,
            ts2=ts2,
            active_series=active_series,
            enabled=enabled,
            proposer_llm=router_llm,
            loop_branches=loop_branches,
            max_loop_depth=max_loop_depth,
            loop_mode=loop_mode,
            quality_threshold=quality_threshold,
        )
        if propose_meta is not None:
            result["propose_fired"] = propose_meta["propose_fired"]
            # Keep these None (the default) when nothing executed, so a no-op
            # propose row stays field-identical to baseline on these keys.
            result["propose_executed"] = propose_meta["propose_executed"] or None
            result["propose_steps"] = propose_meta["propose_steps"] or None
            result["propose_fallbacks"] = propose_meta["propose_fallbacks"] or None
            # quality_score is now COMPUTED on every propose-config row (same as the
            # Stage-2 lesson) even when no action executed.
            result["quality_score"] = propose_meta["propose_quality_score"]
            # A proposed `vision` action forces the vision gate below to fire.
            if propose_meta.get("propose_force_vision"):
                evidence["_propose_force_vision"] = True

        # ── Numeric head (#4a) ───────────────────────────────────────────────
        # Free-response rows (numerical / categorical) leave the MCQ path HERE —
        # before the vision gate and the MCQ interpreter. The value is owned by
        # deterministic tools; the LLM only formats it, and only when
        # numeric_use_llm_format=True. MCQ rows (expected_schema == "mcq") never
        # enter this branch, so everything below is byte-identical for them.
        if expected_schema != "mcq":
            return _finalize(_run_numeric_head(
                result, question, options, expected_schema, subtype,
                ts, ts1, ts2, evidence, answer_llm, numeric_use_llm_format,
                numeric_parser,
            ), ["numeric_head"])

        head_result = _run_optional_heads(
            question=question,
            options=options,
            answer_type=answer_type or row.get("answer_type"),
            option_type=option_type or row.get("option_type"),
            ts=ts,
            ts1=ts1,
            ts2=ts2,
            evidence=evidence,
            mcq_numeric_head=mcq_numeric_head,
            forecast_ranker=forecast_ranker,
            event_head=event_head,
        )
        if head_result is not None:
            result["evidence"] = evidence
            result["flags"] = evidence.get("flags", [])
            result["branch_used"] = branch_name
            result["subtype_used"] = subtype
            result["predicted_letter"] = head_result.predicted_letter
            result["predicted_value"] = head_result.predicted_value
            result["parse_success"] = True
            result["used_head"] = True
            result["head_name"] = head_result.head_name
            result["head_note"] = head_result.head_note
            result["critic_called"] = False
            result["vision_used"] = False
            result["correct"] = (
                head_result.predicted_letter == gold
                if gold and head_result.predicted_letter
                else False
            )
            return _finalize(result, [head_result.head_name or "deterministic_head"])
        if isinstance(evidence, dict) and evidence.get("head_abstentions"):
            result["head_name"] = "forecast_ranker"
            result["head_note"] = ";".join(map(str, evidence.get("head_abstentions") or []))

        try:
            _query_semantics = {
                "question": question,
                "branch": branch_name,
                "scope": scope,
                "subtype": subtype,
                "routing": routing,
            }
            # vision_gate selects the LOOK policy. "additive" = the hand-tuned
            # heuristic (byte-identical baseline arm); "learned" = the
            # pre-registered per-branch suppression policy (#1). The forced/
            # no_vision overrides below still apply on top of either gate.
            if vision_gate == "learned":
                trigger, viz_type, trigger_meta = evaluate_visual_trigger_learned(
                    evidence, _query_semantics
                )
            elif vision_gate == "additive":
                trigger, viz_type, trigger_meta = evaluate_visual_trigger(
                    evidence, _query_semantics
                )
            else:
                raise ValueError(
                    f"unknown vision_gate={vision_gate!r}; expected 'additive' or 'learned'"
                )
            if trigger_meta is not None:
                evidence["vision_trigger"] = trigger_meta

            # Forced vision: some branches (e.g. anomaly) ask inherently visual
            # questions — speedup / cutoff / flip anomaly *types* are invisible to
            # z-score/changepoint stats but obvious on a plot. When the branch is
            # in forced_vision_branches we override BOTH the score gate and the
            # viz_type: spectral branches -> spectrogram, time-domain (anomaly,
            # trend, similarity) -> line_plot. A flat "cutoff" or a "flip" is
            # invisible on a spectrogram, so the verifier's spectrogram pick must
            # be corrected here even when vision already triggered.
            if forced_vision_branches and branch_name in forced_vision_branches:
                desired_viz = (
                    "spectrogram"
                    if branch_name in {"periodicity", "noise", "causality"}
                    else "line_plot"
                )
                if not trigger or viz_type != desired_viz:
                    trigger = True
                    viz_type = desired_viz
                    evidence["vision_trigger"] = {
                        **(trigger_meta or {}),
                        "triggered": True,
                        "viz_type": viz_type,
                        "forced": True,
                    }

            # Suppress vision for branches whose questions are statistical, not
            # visual (e.g. noise: ADF/KPSS/Ljung-Box/ARCH are definitive). The
            # verifier over-triggers vision on the noise branch via the ARCH gate,
            # and the answer LLM then follows a wrong visual suggestion off a cliff.
            # ARCH/heteroskedasticity is a deterministic signal that should inform
            # the answer directly, not escalate to a misleading image.
            if no_vision_branches and branch_name in no_vision_branches:
                trigger = False
                evidence["vision_trigger"] = {
                    **(trigger_meta or {}),
                    "triggered": False,
                    "suppressed": True,
                }

            # Stage-3: a proposed `vision` action escalates the LOOK decision. It
            # fires the gate (unless the branch is hard-suppressed by
            # no_vision_branches, which wins — hard-off is a frozen-config contract).
            # viz_type follows the same branch->representation rule as the forced
            # path: spectral branches -> spectrogram, time-domain -> line_plot.
            if evidence.get("_propose_force_vision") and not (
                no_vision_branches and branch_name in no_vision_branches
            ):
                if not trigger:
                    viz_type = (
                        "spectrogram"
                        if branch_name in {"periodicity", "noise", "causality"}
                        else "line_plot"
                    )
                    trigger = True
                evidence["vision_trigger"] = {
                    **(trigger_meta or {}),
                    "triggered": True,
                    "viz_type": viz_type,
                    "proposed": True,
                }

            if trigger:
                series_id = (
                    row.get("id")
                    or row.get("series_id")
                    or row.get("category")
                    or "series"
                )
                if branch_name in _DUAL_BRANCHES:
                    t1 = ts1 if ts1 is not None else primary
                    t2 = ts2 if ts2 is not None else primary
                    n = min(len(t1), len(t2))
                    ts_vis = np.vstack([t1[:n], t2[:n]])
                elif len(active_series) > 1:
                    min_len = min(len(series) for _, series in active_series)
                    ts_vis = np.vstack(
                        [series[:min_len] for _, series in active_series]
                    )
                    series_id = "combined_series"
                else:
                    ts_vis = primary

                artifact = generate_ts_artifact(ts_vis, viz_type, series_id)
                evidence["vision_tool"] = artifact.get("evidence", {})
                evidence["artifacts"] = artifact.get("artifacts", [])
                # Call the vision sensor to convert the image into structured
                # topological descriptions. Use the answer LLM (answer_llm)
                # as the vision-capable client.
                #
                # raw_pixel_vision MEASUREMENT ARM (Track-V structured-vs-pixel):
                # SKIP analyze_image's JSON extraction entirely so NO topology JSON
                # is produced — the raw PNG (already saved to tmp by
                # generate_ts_artifact) is instead attached directly to the ANSWER
                # LLM call below. This deliberately violates the SVI to measure the
                # multimodal ceiling vs the structured sensor; it is behind this
                # switch and is NEVER promoted/shipped (SVI stays the default).
                if not raw_pixel_vision:
                    try:
                        artifacts_list = artifact.get("artifacts", [])
                        if artifacts_list:
                            img_path = artifacts_list[0].get("path")
                        else:
                            img_path = None
                        if img_path:
                            guidance_text = _visual_guidance(viz_type)
                            vis_res = analyze_image(
                                answer_llm,
                                img_path,
                                viz_type,
                                layout=artifact.get("layout"),
                                series_map=artifact.get("series_map"),
                                color_map=artifact.get("color_map"),
                                guidance=guidance_text,
                            )
                            evidence["vision_struct"] = vis_res.get("vision_struct")
                            evidence["vision_text"] = vis_res.get("vision_text")
                            evidence["vision_raw"] = vis_res.get("raw")
                    except Exception as e:
                        evidence["vision_error"] = str(e)
                try:
                    # In raw_pixel mode vision_struct is absent, so this suggestion
                    # is a PURE raw-pixel look (its prompt's vision-summary block is
                    # empty without vision_struct, and it already attaches the raw
                    # image artifact). vision_letter therefore records what the raw
                    # plot suggested. In the structured arm it carries the JSON
                    # topology summary, exactly as before.
                    vision_suggestion = _run_vision_answer_suggestion(
                        answer_llm, question, options, subtype, evidence
                    )
                    evidence["vision_letter"] = vision_suggestion.get("letter")
                    evidence["vision_confidence"] = vision_suggestion.get("confidence")
                    evidence["vision_answer_raw"] = vision_suggestion.get("raw")
                    evidence["vision_answer_parse_success"] = vision_suggestion.get(
                        "parse_success", False
                    )
                except Exception as e:
                    evidence["vision_answer_error"] = str(e)
                # Flag the raw-pixel answer path for the final answer LLM call: the
                # rendered PNG is attached to the reasoner directly (SVI-violating,
                # measurement-only). Set only when vision actually fired AND the
                # artifact exists, so an off-switch row never carries this key.
                if raw_pixel_vision and evidence.get("artifacts"):
                    evidence["_raw_pixel_answer"] = True
            elif enable_sa_vision and branch_name == "similarity":
                # SA sighted comparison: forced visual path for similarity questions
                series_id = (
                    row.get("id") or row.get("series_id") or row.get("category") or "series"
                )
                t1 = ts1 if ts1 is not None else primary
                t2 = ts2 if ts2 is not None else primary
                if t1 is not None and t2 is not None:
                    n = min(len(t1), len(t2))
                    ts_vis = np.vstack([t1[:n], t2[:n]])
                else:
                    ts_vis = primary
                if ts_vis is not None:
                    artifact = _generate_sa_artifact(ts_vis, sa_viz_mode, series_id)
                    evidence["vision_tool"] = artifact.get("evidence", {})
                    evidence["artifacts"] = artifact.get("artifacts", [])
                    try:
                        artifacts_list = artifact.get("artifacts", [])
                        img_path = artifacts_list[0].get("path") if artifacts_list else None
                        if img_path:
                            guidance_text = _visual_guidance(sa_viz_mode)
                            vis_res = analyze_image(
                                answer_llm,
                                img_path,
                                sa_viz_mode,
                                layout=artifact.get("layout"),
                                series_map=artifact.get("series_map"),
                                color_map=artifact.get("color_map"),
                                guidance=guidance_text,
                            )
                            evidence["vision_struct"] = vis_res.get("vision_struct")
                            evidence["vision_text"] = vis_res.get("vision_text")
                            evidence["vision_raw"] = vis_res.get("raw")
                    except Exception as e:
                        evidence["vision_error"] = str(e)
                    try:
                        vision_suggestion = _run_vision_answer_suggestion(
                            answer_llm, question, options, subtype, evidence
                        )
                        evidence["vision_letter"] = vision_suggestion.get("letter")
                        evidence["vision_confidence"] = vision_suggestion.get("confidence")
                        evidence["vision_answer_raw"] = vision_suggestion.get("raw")
                        evidence["vision_answer_parse_success"] = vision_suggestion.get(
                            "parse_success", False
                        )
                    except Exception as e:
                        evidence["vision_answer_error"] = str(e)
        except Exception as e:
            evidence["vision_trigger_error"] = str(e)
    except Exception as e:
        return _finalize(_run_fallback(
            result, question, options, answer_llm, reason=f"branch_error: {e}"
        ), ["fallback"])

    # Consolidate evidence into separate math and vision streams for the reasoner
    result["evidence"] = evidence
    result["flags"] = evidence.get("flags", [])
    # Build an evidence bundle the orchestrator will pass to the reasoner.
    math_evidence = {
        k: v
        for k, v in evidence.items()
        if k
        not in (
            "vision_struct",
            "vision_text",
            "vision_raw",
            "artifacts",
            "vision_tool",
            # Stage-3 control-plane keys — NOT evidence; excluded so they never
            # enter the answer prompt (the accumulated `propose_<branch>` evidence
            # DOES flow in, exactly like refine — that is the enrichment payload).
            "_propose_force_vision",
            "propose_metadata",
            # raw_pixel_vision control-plane flag — routing signal, not evidence.
            "_raw_pixel_answer",
        )
    }
    evidence_bundle = {
        "math_evidence": math_evidence,
        "vision_evidence": evidence.get("vision_struct"),
        "vision_text": evidence.get("vision_text"),
        "vision_suggestion": {
            "letter": evidence.get("vision_letter"),
            "confidence": evidence.get("vision_confidence"),
            "raw": evidence.get("vision_answer_raw"),
            "parse_success": evidence.get("vision_answer_parse_success", False),
            "error": evidence.get("vision_answer_error"),
        },
        "flags": evidence.get("flags", []),
    }
    result["evidence_bundle"] = evidence_bundle
    result["branch_used"] = branch_name
    result["subtype_used"] = subtype
    result["vision_used"] = bool(evidence.get("vision_answer_parse_success"))
    result["vision_letter"] = evidence.get("vision_letter")
    result["vision_confidence"] = evidence.get("vision_confidence")
    # raw_pixel arm: record whether the raw PNG will be attached to the answer LLM
    # on this row (vision fired AND raw_pixel_vision=True). False everywhere else.
    result["raw_pixel_vision_used"] = bool(evidence.get("_raw_pixel_answer"))

    # ------------------------------------------------------------------
    # Stage 4a: evidence_incomplete → fallback
    # ------------------------------------------------------------------
    if "evidence_incomplete" in evidence.get("flags", []) and not evidence.get(
        "vision_trigger", {}
    ).get("triggered", False):
        return _finalize(_run_fallback(
            result, question, options, answer_llm, reason="evidence_incomplete",
            sc_active=sc_active, sc_k=self_consistency,
            sc_temperature=self_consistency_temperature,
        ), ["fallback"])

    # ------------------------------------------------------------------
    # Stage 4b: LLM interpreter
    # ------------------------------------------------------------------
    try:
        artifacts = evidence.get("artifacts") if isinstance(evidence, dict) else None
        if use_hint:
            # For hint-augmented mode also pass the consolidated evidence bundle
            answer_prompt = build_oneshot_prompt(
                question,
                options,
                evidence_bundle,
                question_hint=row.get("question_hint"),
                branch=branch_name,
                subtype=subtype,
            )
        else:
            answer_prompt = build_answer_prompt(
                question,
                options,
                evidence_bundle,
                branch=branch_name,
                subtype=subtype,
            )
        # Do NOT pass raw image artifacts to the final reasoning LLM by default.
        # The vision sensor above produces `vision_struct` and `vision_text`
        # which are merged into the textual prompt (the SVI default).
        #
        # raw_pixel_vision MEASUREMENT ARM: when set on a vision-fired row, attach
        # the rendered PNG directly to the answer LLM so the reasoner sees the raw
        # plot (NOT the JSON topology). This is the ONLY change vs baseline on fired
        # rows. SVI-violating, measurement-only, NEVER shipped — see _raw_pixel_answer.
        answer_artifacts = (
            evidence.get("artifacts")
            if (raw_pixel_vision and evidence.get("_raw_pixel_answer"))
            else None
        )
        if sc_active:
            # Self-Consistency control arm: k samples at temp>0, majority vote.
            sc_letter, sc_ok, sc_raws = _sc_sample_answer(
                answer_llm, SYSTEM_PROMPT, answer_prompt,
                self_consistency, self_consistency_temperature,
            )
            raw_answer = "\n---\n".join(str(r) for r in sc_raws)
            parsed = {"letter": sc_letter, "parse_success": sc_ok}
        elif answer_artifacts:
            raw_answer = answer_llm.generate(
                SYSTEM_PROMPT,
                answer_prompt,
                artifacts=answer_artifacts,
            )
            parsed = parse_answer(raw_answer)
        else:
            raw_answer = answer_llm.generate(
                SYSTEM_PROMPT,
                answer_prompt,
            )
            parsed = parse_answer(raw_answer)
    except Exception as e:
        return _finalize(_run_fallback(
            result, question, options, answer_llm, reason=f"llm_error: {e}",
            sc_active=sc_active, sc_k=self_consistency,
            sc_temperature=self_consistency_temperature,
        ), ["fallback"])

    result["raw_answer"] = raw_answer
    result["predicted_letter"] = parsed["letter"]
    result["parse_success"] = parsed["parse_success"]
    result["correct"] = (parsed["letter"] == gold) if gold else False
    return _finalize(result, ["answer_prompt"])


def _run_fallback(
    result: dict, question: str, options: list, llm_client, reason: str,
    sc_active: bool = False, sc_k: int = 0,
    sc_temperature: float = SC_DEFAULT_TEMPERATURE,
) -> dict:
    """Replace the answer step with the fallback prompt.

    When ``sc_active`` (the Self-Consistency control arm fired on this row), the
    single fallback call is replaced by k temp>0 samples + a majority vote — so
    the control still samples on the matched ``evidence_incomplete`` rows that
    reach the fallback path (vision-not-triggered). Default off ⇒ byte-identical.
    """
    result["used_fallback"] = True
    try:
        sys_p, usr_p = build_fallback_prompt(question, options, reason=reason)
        if sc_active:
            sc_letter, sc_ok, sc_raws = _sc_sample_answer(
                llm_client, sys_p, usr_p, sc_k, sc_temperature
            )
            raw_answer = "\n---\n".join(str(r) for r in sc_raws)
            parsed = {"letter": sc_letter, "parse_success": sc_ok}
        else:
            raw_answer = llm_client.generate(sys_p, usr_p)
            parsed = parse_answer(raw_answer)
    except Exception as e:
        result["raw_answer"] = str(e)
        return result

    result["raw_answer"] = raw_answer
    result["predicted_letter"] = parsed["letter"]
    result["parse_success"] = parsed["parse_success"]
    gold = result.get("gold_letter")
    result["correct"] = (parsed["letter"] == gold) if gold else False
    return result


def _run_llm_only(result: dict, question: str, options: list, answer_llm) -> dict:
    """No-architecture control: one neutral MCQ call, no tools/router/vision.

    Records the same answer fields the normal path does (predicted_letter,
    parse_success, correct) plus ``branch_used='llm_only'`` so the paired diff
    strata are well-defined. Any exception is captured (the row is recorded as an
    abstention, never crashes the batch).
    """
    result["branch_used"] = "llm_only"
    result["initial_branch_used"] = "llm_only"
    result["routing"] = {"branch": "llm_only", "parse_success": True, "scope": "global"}
    try:
        sys_p, usr_p = build_llm_only_prompt(question, options)
        raw_answer = answer_llm.generate(sys_p, usr_p)
        parsed = parse_answer(raw_answer)
    except Exception as e:
        result["raw_answer"] = str(e)
        return result
    result["raw_answer"] = raw_answer
    result["predicted_letter"] = parsed["letter"]
    result["parse_success"] = parsed["parse_success"]
    gold = result.get("gold_letter")
    result["correct"] = (parsed["letter"] == gold) if gold else False
    return result


def _run_llm_numeric(result, question, ts, ts1, ts2, expected_schema, answer_llm):
    """Free-response COUNTERFACTUAL arm: the answer LLM COMPUTES the number from the
    raw series directly — no router, no tools, no evidence, no options. The
    deterministic numeric head is deliberately BYPASSED so a paired Accuracy@10%
    comparison against the head measures whether the tool is load-bearing on a
    strong backbone (mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md).

    Emits ``predicted_value`` (the harness scores it @10%, exactly as it scores the
    head). Abstains (predicted_value=None, numeric_note='llm_unparseable') on a parse
    failure — never a fabricated number. Any exception is captured as an abstention
    so one bad row never nukes the batch.
    """
    result["branch_used"] = "llm_numeric"
    result["initial_branch_used"] = "llm_numeric"
    result["routing"] = {"branch": "llm_numeric", "parse_success": True, "scope": "global"}
    result["expected_schema"] = expected_schema
    result["vision_used"] = False
    try:
        sys_p, usr_p = build_llm_numeric_prompt(question, ts, ts1, ts2)
        raw = answer_llm.generate(sys_p, usr_p)
        result["raw_answer"] = raw
        parsed = parse_numeric_answer(raw)
        if parsed.get("parse_success") and parsed.get("value") is not None:
            result["predicted_value"] = parsed["value"]
            result["parse_success"] = True
        else:
            result["numeric_note"] = "llm_unparseable"
    except Exception as e:
        result["raw_answer"] = str(e)
        result["numeric_note"] = "llm_error"
    return result


def _format_numeric_with_llm(answer_llm, question, computed_value):
    """OPTIONAL, OFF by default. Ask the LLM to restate the precomputed value
    (the tool owns it; the prompt only formats). On a parse failure do at most
    ONE re-prompt, then fall back to the deterministic value. Never an unbounded
    loop — this protects temp-0 paired determinism and the latency win."""
    try:
        sys_p, usr_p = build_numeric_prompt(question, computed_value)
        for _ in range(2):  # initial attempt + at most one re-prompt
            raw = answer_llm.generate(sys_p, usr_p)
            parsed = parse_numeric_answer(raw)
            if parsed.get("parse_success") and parsed.get("value") is not None:
                return parsed["value"]
        return computed_value
    except Exception:
        return computed_value


def _run_optional_heads(
    *,
    question,
    options,
    answer_type,
    option_type,
    ts,
    ts1,
    ts2,
    evidence,
    mcq_numeric_head=False,
    forecast_ranker=None,
    event_head=None,
):
    """Dispatch optional deterministic heads.

    Defaults are all inert. TSRBench passes structural answer/option types from
    its adapter; TSExam/MMTS do not, so their frozen configs keep the old path.
    """
    try:
        ot = OptionType(option_type) if option_type else None
    except Exception:
        ot = None

    if mcq_numeric_head and ot == OptionType.SCALAR:
        res = scalar_mcq_head(question, options, ts, ts1, ts2, evidence)
        if res is not None:
            return res

    def _record_head_abstention(note: str):
        if isinstance(evidence, dict):
            evidence.setdefault("head_abstentions", []).append(note)

    if forecast_ranker is not None and ot not in {OptionType.TRAJECTORY, OptionType.RANGE}:
        _record_head_abstention(f"forecast_ranker:unsupported_option_type={ot.value if ot else 'unknown'}")

    if forecast_ranker is not None and ot in {OptionType.TRAJECTORY, OptionType.RANGE}:
        series = []
        if ts is not None:
            series.append(Channel("ts", ts))
        if ts1 is not None:
            series.append(Channel("ts1", ts1))
        if ts2 is not None:
            series.append(Channel("ts2", ts2))
        if len(series) != 1:
            _record_head_abstention(f"forecast_ranker:unsupported_series_count={len(series)}")
            return None
        ranker = forecast_rank_head if forecast_ranker is True else forecast_ranker
        res = _coerce_forecast_head_result(ranker(question, ot, options, series), options)
        if res is not None:
            return res
        _record_head_abstention("forecast_ranker:ranker_abstained")

    if event_head is not None and ot == OptionType.EVENT_LABEL:
        head = _default_event_head if event_head is True else event_head
        res = head(question, options, ts, ts1, ts2)
        if res is not None:
            return res

    return None


def _coerce_forecast_head_result(res, options) -> HeadResult | None:
    if res is None:
        return None
    if isinstance(res, HeadResult):
        return res
    if isinstance(res, ForecastRankResult) or hasattr(res, "predicted_index"):
        idx = int(getattr(res, "predicted_index"))
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if idx < 0 or idx >= min(len(options), len(letters)):
            return None
        method = str(getattr(res, "method", "") or "unknown")
        note = str(getattr(res, "note", "") or "").strip()
        return HeadResult(
            predicted_letter=letters[idx],
            head_name="forecast_ranker",
            head_note=f"{method}:{note}" if note else method,
        )
    return None


def _run_numeric_head(
    result, question, options, expected_schema, subtype,
    ts, ts1, ts2, evidence, answer_llm, numeric_use_llm_format=False,
    numeric_parser="keyword",
):
    """Answer a free-response row deterministically and emit ``predicted_value``.

    The harness scores ``predicted_value`` against gold (MMTS-Bench Accuracy@10%
    for numerical; exact label match for categorical). ``predicted_letter`` stays
    None and ``correct`` stays False here — numeric scoring is the harness's job
    because the tolerance is a pre-registered experiment parameter. On any
    coverage gap (unknown quantity / no series / no signal) we ABSTAIN
    (predicted_value=None) — an honest miss, never a fabricated number.
    """
    evidence = evidence if isinstance(evidence, dict) else {}
    result["evidence"] = evidence
    result["flags"] = evidence.get("flags", [])
    result["branch_used"] = result.get("branch_used") or evidence.get("branch")
    result["subtype_used"] = subtype
    result["expected_schema"] = expected_schema
    result["vision_used"] = False

    try:
        if expected_schema == "numerical":
            # Parser arm (A0 keyword / A1 deterministic / A2 llm_fallback) → a PLAN
            # over the closed registry; the tools (evaluate_plan) own the value.
            plan, parser_used, llm_fired = _select_numeric_plan(
                question, numeric_parser, answer_llm)
            result["numeric_parser"] = parser_used
            result["numeric_llm_fired"] = llm_fired
            if plan is None:
                result["numeric_note"] = "no_quantity_match"  # abstain, no guess
                return result
            value, note = evaluate_plan(plan, ts, ts1, ts2, evidence)
            result["numeric_quantity"] = (
                "composition:" + plan["op"] if "op" in plan else plan.get("quantity"))
            result["numeric_note"] = note
            if value is None:
                return result
            if numeric_use_llm_format:
                value = _format_numeric_with_llm(answer_llm, question, value)
            # Schema-compliance gate: emit only if the answer matches the shape.
            if validate_schema_compliance(value, "numerical"):
                result["predicted_value"] = value
                result["parse_success"] = True
            return result

        if expected_schema == "categorical":
            label, note = answer_categorical(question, options, evidence)
            result["numeric_note"] = note
            if label is not None and validate_schema_compliance(label, "categorical"):
                result["predicted_value"] = label
                result["parse_success"] = True
            return result
    except Exception as e:
        result["numeric_error"] = str(e)

    return result


def _select_numeric_plan(question, numeric_parser, answer_llm):
    """Pick the parser arm and return (plan, parser_label, llm_fired). The A2 LLM
    planner fires ONLY when the A1 deterministic parser abstains, so clean
    single-quantity rows never call it (non-regression + efficiency)."""
    if numeric_parser == "deterministic":
        return parse_plan_deterministic(question), "deterministic", False
    if numeric_parser == "llm_fallback":
        plan = parse_plan_deterministic(question)
        if plan is not None:
            return plan, "deterministic", False
        return _llm_numeric_plan(question, answer_llm), "llm_planner", True
    # default A0 keyword — byte-identical to the legacy head (a leaf plan)
    return infer_numeric_quantity(question), "keyword", False


def _llm_numeric_plan(question, answer_llm):
    """A2: ask the LLM for a PLAN over the closed numeric registry (never a
    number) and validate it. Returns a plan or None (abstain) on any failure."""
    try:
        sys_p, usr_p = build_numeric_planner_prompt(question)
        raw = answer_llm.generate(sys_p, usr_p)
        return parse_numeric_plan_json(raw)
    except Exception:
        return None


def run_dataset(
    rows,
    llm_client,
    enabled_branches=None,
    max_rows=None,
    show_progress: bool = True,
    use_hint: bool = False,
    router_client=None,
    answer_client=None,
    multi_branch: bool = False,
    enable_sa_vision: bool = False,
    sa_viz_mode: str = "stacked",
    forced_vision_branches=None,
    no_vision_branches=None,
    numeric_use_llm_format: bool = False,
    numeric_parser: str = "keyword",
    vision_gate: str = "additive",
    recovery: str | None = None,
    mcq_numeric_head: bool = False,
    forecast_ranker=None,
    event_head=None,
    return_trace: bool = False,
    max_workers: int = 1,
    row_timeout: float = None,
    control: str | None = None,
    control_seed: int = 0,
    loop_branches: list[str] | None = None,
    max_loop_depth: int = 0,
    loop_on_flags: list[str] | None = None,
    loop_mode: str = "reroute_once",
    quality_threshold: float = QUALITY_THRESHOLD,
    self_consistency: int = 0,
    self_consistency_temperature: float = SC_DEFAULT_TEMPERATURE,
    llm_only: bool = False,
    llm_numeric: bool = False,
    raw_pixel_vision: bool = False,
) -> list:
    """
    Run the pipeline over an iterable of dataset rows.

    Args:
        rows:             iterable of row dicts (e.g. ds["test"].to_pandas().itertuples())
        llm_client:       GeminiClient instance
        enabled_branches: optional set of branch names; None = all
        max_rows:         stop early after this many rows (useful for debug runs)
        show_progress:    show tqdm progress bar
        use_hint:         if True, pass question_hint to the LLM interpreter prompt
        router_client:    optional client for router LLM (defaults to llm_client)
        answer_client:    optional client for answer LLM (defaults to llm_client)
        multi_branch:     if True, run up to 3 candidate branches and quality-gate them
        enable_sa_vision: if True, force visual path on for similarity branch
        sa_viz_mode:      representation for SA sighted vision ("stacked", "overlay_line", "dual_cwt")
        max_workers:      number of rows to process concurrently (default 1 = serial).
                          API calls are still serialized by GeminiClient's rate limiter.
        row_timeout:      per-row wall-clock timeout in seconds for the concurrent path
                          (default None = no timeout). Rows that exceed this limit are
                          marked with {"error": "timeout", "id": ...} and the batch
                          continues. Has no effect when max_workers <= 1.

    Returns:
        List of result dicts, one per row.
    """
    row_list = []
    for i, row in enumerate(rows):
        if max_rows is not None and i >= max_rows:
            break
        row_list.append(row._asdict() if hasattr(row, "_asdict") else dict(row))

    def _process(row_dict):
        return run_pipeline(
            row_dict,
            llm_client,
            enabled_branches,
            use_hint=use_hint,
            router_client=router_client,
            answer_client=answer_client,
            multi_branch=multi_branch,
            enable_sa_vision=enable_sa_vision,
            sa_viz_mode=sa_viz_mode,
            forced_vision_branches=forced_vision_branches,
            no_vision_branches=no_vision_branches,
            numeric_use_llm_format=numeric_use_llm_format,
            numeric_parser=numeric_parser,
            vision_gate=vision_gate,
            recovery=recovery,
            mcq_numeric_head=mcq_numeric_head,
            forecast_ranker=forecast_ranker,
            event_head=event_head,
            answer_type=row_dict.get("answer_type"),
            option_type=row_dict.get("option_type"),
            return_trace=return_trace,
            control=control,
            control_seed=control_seed,
            loop_branches=loop_branches,
            max_loop_depth=max_loop_depth,
            loop_on_flags=loop_on_flags,
            loop_mode=loop_mode,
            quality_threshold=quality_threshold,
            self_consistency=self_consistency,
            self_consistency_temperature=self_consistency_temperature,
            llm_only=llm_only,
            llm_numeric=llm_numeric,
            raw_pixel_vision=raw_pixel_vision,
        )

    if max_workers <= 1:
        iterable = tqdm(row_list, desc="TS-QA eval") if show_progress else row_list
        out = []
        for r in iterable:
            try:
                out.append(_process(r))
            except Exception as exc:
                # Never let one bad row nuke a multi-hour run; record and continue.
                out.append({"error": str(exc), "id": r.get("id")})
        return out

    results = [None] * len(row_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process, r): i for i, r in enumerate(row_list)}
        completed = concurrent.futures.as_completed(futs)
        if show_progress:
            completed = tqdm(completed, total=len(row_list), desc="TS-QA eval")
        for fut in completed:
            idx = futs[fut]
            try:
                results[idx] = fut.result(timeout=row_timeout)
            except concurrent.futures.TimeoutError:
                results[idx] = {"error": "timeout", "id": row_list[idx].get("id")}
            except Exception as exc:
                results[idx] = {"error": str(exc), "id": row_list[idx].get("id")}
    return results


def audit_routing(
    rows,
    llm_client,
    router_client=None,
    max_rows=None,
    show_progress: bool = True,
) -> dict:
    """
    Router-only audit pass to compare routing distributions across models.

    Returns a dict with keys:
        routes            — list of routing dicts
        by_branch         — counts per branch
        by_branch_subtype — counts per (branch, subtype)
    """
    router_llm = router_client or llm_client
    results = []
    iterable = tqdm(rows, desc="Routing audit") if show_progress else rows
    for i, row in enumerate(iterable):
        if max_rows is not None and i >= max_rows:
            break
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)
        ts, ts1, ts2 = _extract_series(row_dict)
        question = row_dict["question"]
        expected_schema = infer_expected_schema(
            row_dict.get("options"), question=question,
            category=row_dict.get("category"),
        )
        try:
            router_prompt = build_router_prompt(
                question,
                has_ts1=ts1 is not None,
                has_ts2=ts2 is not None,
            )
            raw_router = router_llm.generate(ROUTER_SYSTEM_PROMPT, router_prompt)
            routing = parse_routing(raw_router)
        except Exception as e:
            routing = {
                "parse_success": False,
                "branch": None,
                "scope": "global",
                "subtype": None,
                "raw": str(e),
            }
            raw_router = str(e)

        results.append(
            {
                "id": row_dict.get("id"),
                "category": row_dict.get("category"),
                "branch": routing.get("branch"),
                "scope": routing.get("scope", "global"),
                "subtype": routing.get("subtype"),
                "expected_schema": expected_schema,
                "parse_success": routing.get("parse_success", False),
                "raw_router": routing.get("raw", raw_router),
            }
        )

    branch_counts = Counter(r["branch"] for r in results if r["branch"])
    branch_subtype_counts = Counter(
        (r["branch"], r["subtype"]) for r in results if r["branch"] and r["subtype"]
    )
    schema_counts = Counter(r["expected_schema"] for r in results)
    return {
        "routes": results,
        "by_branch": dict(branch_counts),
        "by_branch_subtype": {
            f"{branch}:{subtype}": count
            for (branch, subtype), count in branch_subtype_counts.items()
        },
        "by_expected_schema": dict(schema_counts),
    }
