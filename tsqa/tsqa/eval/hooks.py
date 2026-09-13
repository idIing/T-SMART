"""``run_pipeline`` extension points — **frozen Contract B** (gate-zero).

The Wave-1 mechanisms (trace hashing, top-2 recovery, forecast ranking) plug into the
pipeline at fixed points. Freezing their signatures here means each worker owns ONE module
that satisfies a contract, and ``runner.py`` gains a single orchestrator-owned dispatch
line per hook — instead of four workers editing the runner's internals and colliding.

Three contracts:
  * :class:`TraceFields` + :func:`compute_trace_hash` — Worker C. Semantic, run-stable.
  * :class:`RecoveryResult` + :func:`select_top2_nocritic` — Worker E. Model-call-free,
    fully implemented here (it is pure arbitration); the worker only wires it in.
  * :class:`HeadResult` + :class:`ForecastRanker` — Worker F. Default abstains.

None of these import ``runner`` (no cycles); they depend only on stdlib + numpy + the
:mod:`tsqa.eval.sample` enums.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .sample import Channel, OptionType, FORECASTABLE_OPTION_TYPES
from .tsr_heads import HeadResult

# ---------------------------------------------------------------------------
# Trace hashing (Worker C) — semantic, NOT byte-literal.
# ---------------------------------------------------------------------------

#: Keys whose values are run-volatile and MUST be excluded from the trace hash:
#: tempfile paths, raw blobs, wall-clock timing. A naive hash would "drift" on these
#: even when the decision is identical. (Add an artifact *content* hash explicitly if a
#: vision artifact's pixels are ever load-bearing for a claim.)
VOLATILE_KEYS = frozenset({
    "path", "artifact_path", "artifacts", "tmp", "tmpdir",
    "timestamp", "time", "duration", "duration_ms", "raw", "raw_answer", "raw_router",
})


def _canonicalize(obj, *, ndigits: int = 6):
    """Recursively normalize for hashing: sort dict keys, drop volatile keys, round
    floats, coerce numpy scalars/arrays to plain python. Order-independent for dicts,
    order-preserving for lists (sequence order is semantic)."""
    if isinstance(obj, dict):
        return {
            k: _canonicalize(v, ndigits=ndigits)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if k not in VOLATILE_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v, ndigits=ndigits) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_canonicalize(v, ndigits=ndigits) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@dataclass
class TraceFields:
    """The semantic fingerprint of one pipeline execution. Populated by ``run_pipeline``;
    a confirmatory claim requires duplicate-run *trace* agreement, not just final-answer
    agreement (hosted models are not bitwise-deterministic at temp 0)."""
    route: Optional[str] = None
    branch: Optional[str] = None
    tool_sequence: list = field(default_factory=list)
    tool_args: dict = field(default_factory=dict)
    tool_outputs: dict = field(default_factory=dict)
    prompt_template_ids: list = field(default_factory=list)
    model_id: Optional[str] = None
    answer: Optional[str] = None
    verifier_state: Optional[str] = None


def compute_trace_hash(fields: TraceFields, *, ndigits: int = 6) -> str:
    """Stable SHA-256 over the canonicalized trace. Invariant under irrelevant changes
    (dict ordering, tempfile paths, float noise below ``ndigits``); sensitive to route,
    tool sequence, rounded tool outputs, prompt-*template* id, model id, answer, verifier."""
    payload = _canonicalize(
        {
            "route": fields.route,
            "branch": fields.branch,
            "tool_sequence": list(fields.tool_sequence),
            "tool_args": fields.tool_args,
            "tool_outputs": fields.tool_outputs,
            "prompt_template_ids": list(fields.prompt_template_ids),
            "model_id": fields.model_id,
            "answer": fields.answer,
            "verifier_state": fields.verifier_state,
        },
        ndigits=ndigits,
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Top-2 recovery (Worker E) — deterministic, ZERO model calls.
# ---------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    chosen_branch: Optional[str]
    ranked: list            # list[tuple[branch_name, score]] sorted desc, name tie-break
    critic_called: bool = False


def select_top2_nocritic(
    candidates: Sequence[str],
    evidences: dict,
    scores: dict,
) -> RecoveryResult:
    """Deterministic arbitration for Phase-1 recovery (``recovery="top2_nocritic"``).

    Ranks by ``(-score, name)`` so ties break by branch NAME — eliminating dict-iteration-
    order nondeterminism — and **never** invokes the critic LLM. The contract guarantee,
    asserted by the Phase-1 test, is ``critic_called is False`` and that this introduces no
    model call beyond the baseline router+answer. ``evidences`` is accepted for parity with
    the caller and future tie-break refinements; it is not consulted here.
    """
    ranked = sorted(
        ((b, float(scores.get(b, 0.0))) for b in candidates),
        key=lambda kv: (-kv[1], kv[0]),
    )
    chosen = ranked[0][0] if ranked else None
    return RecoveryResult(chosen_branch=chosen, ranked=ranked, critic_called=False)


@runtime_checkable
class RecoveryPolicy(Protocol):
    def __call__(self, candidates: Sequence[str], evidences: dict, scores: dict) -> RecoveryResult: ...


# ---------------------------------------------------------------------------
# Forecast ranking (Worker F) — default ABSTAINS; option_type guard is the contract.
# ---------------------------------------------------------------------------

@dataclass
class ForecastRankResult:
    """Legacy forecast result shape retained for compatibility.

    New rankers should return :class:`HeadResult` directly. The runner converts
    this object to ``HeadResult`` at a single boundary if an older ranker is
    supplied.
    """
    predicted_index: int    # index into options of the chosen trajectory/range
    scores: list            # per-option consistency score (higher = better)
    method: str             # which baseline won: persistence | drift | seasonal_naive | ...
    note: str = ""          # provenance / distractor-artifact flags


@runtime_checkable
class ForecastRanker(Protocol):
    def __call__(
        self,
        question: str,
        option_type: OptionType,
        options: Sequence[str],
        series: Sequence[Channel],
    ) -> Optional[HeadResult]: ...


def null_forecast_ranker(
    question: str,
    option_type: OptionType,
    options: Sequence[str],
    series: Sequence[Channel],
) -> Optional[HeadResult]:
    """Default ranker: ABSTAIN (returns ``None``). Worker F replaces this. The contract any
    real ranker MUST honor: return ``None`` unless ``option_type in
    FORECASTABLE_OPTION_TYPES`` — ranking-by-forecast is the wrong solver on categorical /
    event-label / text options."""
    return None


def forecast_eligible(option_type: OptionType) -> bool:
    """Single source of truth for the Phase-2 option-type guard (shared by runtime + census)."""
    return option_type in FORECASTABLE_OPTION_TYPES


__all__ = [
    "VOLATILE_KEYS",
    "TraceFields",
    "compute_trace_hash",
    "RecoveryResult",
    "select_top2_nocritic",
    "RecoveryPolicy",
    "ForecastRankResult",
    "ForecastRanker",
    "null_forecast_ranker",
    "forecast_eligible",
]
