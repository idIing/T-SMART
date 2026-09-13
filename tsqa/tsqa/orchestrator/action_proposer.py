"""
Stage-3 LLM action proposer (the `loop_mode="propose"` path).
================================================================

The LAST and highest-risk ReAct stage (research_journal/02_react_stages.md
§Stage 3): the LLM *proposes* the next tool. Stages 1 (`evidence_incomplete`
reroute) and 2 (keyword `get_keyword_candidates` accumulation) both reached a
clean H0 — Stage 1 because its trigger fired 0/746, Stage 2 because the **keyword
candidate generator returned 0 second-branches** on every one of the 123 rows the
quality gate flagged (log/004, log/005). Stage 3 attacks *that exact bottleneck*:
it replaces the keyword matcher with the LLM as a constrained proposer, gated on
the SAME live `quality_score < QUALITY_THRESHOLD` signal. It is the cleanest test
of "does an LLM proposer find useful actions where keywords found none?" — a real
effect OR a clean "even the LLM proposer can't help" null, both valuable.

THE FOUR HARD PRECONDITIONS (non-negotiable — §Stage 3; tested in
tests/test_wave1_contracts.py::TestStage3Propose):
  1. **Typed action schema.** Every proposal validates against ``ProposedAction``
     (a dataclass + validator): ``{"action": <enum>, "reason": str}``. Anything
     that does not parse to this shape is invalid.
  2. **Finite CLOSED tool registry.** ``action`` must be one of the 8 members of
     ``ACTION_REGISTRY`` — the 6 branches + ``vision`` + ``numeric_head``. An
     action outside the registry is rejected. There are NO open-ended actions.
  3. **Hard per-row call budget.** ``max_executed`` executed actions/row (the
     runner passes ``max_loop_depth=2``) and **exactly one** proposal-LLM call per
     step. Over budget ⇒ stop.
  4. **Parser-failure contract.** Any unparseable / invalid / out-of-registry /
     over-budget / already-run proposal falls back **deterministically** to the
     current answer path; it is NEVER silently retried.

SVI / model-freeze discipline (research_journal/03_related_work.md §A,B): the
proposal is a **routing decision**, never a re-answer. Executed branch/numeric
actions accumulate their deterministic evidence under namespaced keys in the
tool/evidence layer; the single existing answer-LLM call then sees the richer
evidence. The proposal prompt **omits the answer options** (no letter-nudging,
exactly like the Stage-1 reroute prompt). The proposer NEVER re-prompts the answer
LLM to re-answer on the same evidence (avoids the self-correction blind-spot /
SCoRe collapse / sycophancy failure modes, CRITIC 2305.11738 / Kamoi 2406.01297).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# --- Precondition 2: the finite, CLOSED tool registry -----------------------
# The 6 deterministic branches + the two on-demand tools. A proposal mapping to
# anything outside this frozen set is invalid. No open-ended actions exist.
_BRANCH_ACTIONS = frozenset(
    {"trend", "periodicity", "anomaly", "noise", "similarity", "causality"}
)
_TOOL_ACTIONS = frozenset({"vision", "numeric_head"})
ACTION_REGISTRY: frozenset[str] = _BRANCH_ACTIONS | _TOOL_ACTIONS


# --- Precondition 1: the typed action schema --------------------------------
@dataclass(frozen=True)
class ProposedAction:
    """A validated LLM proposal. Construction goes through :func:`parse_proposal`,
    which is the ONLY sanctioned constructor — it enforces the closed registry and
    the ``reason: str`` field. A bare ``ProposedAction(...)`` bypasses validation
    and must not be used on the proposal path."""

    action: str
    reason: str

    def __post_init__(self) -> None:
        # Defense in depth: even direct construction cannot smuggle an
        # out-of-registry action or a non-string reason past the type.
        if self.action not in ACTION_REGISTRY:
            raise ValueError(f"action {self.action!r} not in closed registry")
        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


# Evidence keys safe to surface to the proposer. Mirrors the critic's compact
# view (orchestrator/critic.py) — math fields + flags only, NEVER raw pixels /
# base64 / vision_raw (SVI). The proposer reasons over the SAME structured
# evidence the rest of the pipeline does.
_PROPOSE_KEYS = {
    "branch", "scope", "subtype", "direction", "slope", "r2", "exp_r2", "log_r2",
    "best_fit_type", "dominant_period_fft", "period_reconciled",
    "seasonal_strength", "spectral_entropy", "waveform_hint", "is_white_noise",
    "is_stationary", "adf_pval", "kpss_pval", "outlier_count", "max_deviation",
    "n_breakpoints", "best_lag", "best_corr", "pval_12", "pval_21",
    "quality_score", "flags",
}


def _compact_evidence(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        return {}
    return {k: v for k, v in evidence.items() if k in _PROPOSE_KEYS}


PROPOSER_SYSTEM_PROMPT = """\
You are a constrained action proposer for a time-series question-answering
pipeline. The current deterministic evidence scored BELOW the quality gate, so a
second analysis action may help. Choose the SINGLE next action most likely to add
decisive evidence, from the closed registry only.

Allowed actions (choose EXACTLY one, no others exist):
  trend         - linear/exponential/log fit, direction, change points
  periodicity   - FFT / seasonality / waveform shape
  anomaly       - outliers, point anomalies, structural breaks
  noise         - stationarity (ADF/KPSS), white-noise, volatility, AR/MA
  similarity    - compare two series (statistical / shape / lag)
  causality     - Granger lead-lag between two series
  vision        - render and visually inspect the series (use only when the
                  question is about visual SHAPE that statistics miss)
  numeric_head  - compute an exact numeric statistic deterministically

Output ONLY a JSON object with two fields and nothing else:
  "action": one of the allowed action names above, exactly
  "reason": one short sentence (<= 20 words) on why it adds decisive evidence

No prose, no markdown, no answer letters.
"""


def build_proposal_prompt(
    question: str,
    primary_branch: Optional[str],
    evidence: dict,
    already_run: Optional[set] = None,
) -> str:
    """Build the user-turn proposal prompt.

    DELIBERATELY OMITS the answer options (no-letter-nudging — identical stance to
    the Stage-1 reroute prompt ``_build_reroute_prompt``). The proposer sees the
    question, the primary route, the compact structured evidence, and the set of
    actions already executed (so it is steered away from re-proposing a no-op),
    but never the A/B/C/D choices.
    """
    already = sorted(already_run or set())
    already_block = (
        f"\nActions already executed this row (do NOT repeat — they would be a "
        f"no-op): {already}\n"
        if already
        else ""
    )
    return (
        "The deterministic evidence below scored under the quality gate. Propose "
        "ONE next action from the closed registry to add decisive evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Primary branch already run: {primary_branch}\n"
        f"Compact evidence:\n{json.dumps(_compact_evidence(evidence), default=str, indent=2)}\n"
        f"{already_block}\n"
        'Return the JSON now: {"action": "...", "reason": "..."}'
    )


def parse_proposal(raw: str) -> Optional[ProposedAction]:
    """Parse + validate one raw proposal into a :class:`ProposedAction`.

    Returns ``None`` on ANY failure (precondition 4): unparseable JSON, missing
    fields, non-string reason, or an ``action`` outside the closed
    ``ACTION_REGISTRY``. The caller treats ``None`` as a deterministic fallback —
    it is NEVER retried. Validation is strict and closed: we do NOT fuzzy-scan the
    raw text for a stray branch name the way the critic parser does, because an
    out-of-schema response on the action path must fail closed, not be coerced.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group()

    try:
        obj = json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    action = obj.get("action")
    reason = obj.get("reason")
    if not isinstance(action, str):
        return None
    action = action.strip().lower()
    if action not in ACTION_REGISTRY:  # closed registry — fail closed
        return None
    # The typed schema mandates BOTH fields: a proposal with no (or an empty)
    # `reason` is malformed and fails closed (precondition 1 — the LLM must justify
    # the routing decision, and an absent justification signals a degenerate output).
    if not isinstance(reason, str) or not reason.strip():
        return None

    try:
        return ProposedAction(action=action, reason=reason.strip())
    except ValueError:
        return None


def is_branch_action(action: str) -> bool:
    return action in _BRANCH_ACTIONS


def is_tool_action(action: str) -> bool:
    return action in _TOOL_ACTIONS


__all__ = [
    "ACTION_REGISTRY",
    "ProposedAction",
    "PROPOSER_SYSTEM_PROMPT",
    "build_proposal_prompt",
    "parse_proposal",
    "is_branch_action",
    "is_tool_action",
]
