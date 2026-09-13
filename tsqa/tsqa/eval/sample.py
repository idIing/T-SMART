"""Canonical experimental object — **frozen Contract A** (gate-zero).

One typed record that serves TSExam, MMTS-Bench, TSRBench, and future benchmarks so
that adapters and the per-schema scorer code against a single shape instead of
benchmark-specific dicts. This file is the contract the Wave-0 workers build against;
freeze it before the fan-out.

Two hard rules are encoded **structurally** (not by convention), because both have
already bitten this project:

1. **Reporting-only labels never ride on the runtime object.** ``dimension``, task,
   ``domain``, template/source ids — anything an answer must not be able to see — live
   in a SEPARATE :class:`ReportingLabels` sidecar keyed by ``id`` and handed only to the
   eval / diff layer. The pipeline consumes :class:`BenchmarkSample`, which has *no*
   task/dimension field, so the "route on the benchmark's own dimension label" oracle
   leak is *unrepresentable* rather than merely discouraged.
2. **``to_runner_row()`` is the only bridge to the current ``run_pipeline`` row dict**
   (``{ts, ts1, ts2}``). It is intentionally lossy above two channels (documented), since
   today's pipeline consumes <=2 series; the canonical object still retains all N for the
   future N-series branches.

The ``AnswerType`` / ``OptionType`` enums ARE the taxonomy the TSRBench census (0a) must
populate; the scorer (0b) dispatches on ``answer_type`` and the forecast head (Phase 2)
fires only on ``option_type in`` :data:`FORECASTABLE_OPTION_TYPES`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AnswerType(str, Enum):
    """Shape of the gold answer — decides which scorer branch runs."""
    MCQ = "mcq"                 # exact letter / option-text match
    NUMERICAL = "numerical"     # relative-tolerance (numeric-head protocol)
    CATEGORICAL = "categorical" # exact label match
    TEXTUAL = "textual"         # free text — excluded from paired-McNemar by default
    OPEN = "open"               # open-ended / generative — excluded, reported separately


class OptionType(str, Enum):
    """Shape of the MCQ options — decides forecast-head eligibility and scoring."""
    TRAJECTORY = "trajectory"           # each option is a candidate future series
    SCALAR = "scalar"                   # each option is a single number
    RANGE = "range"                     # each option is a numeric interval
    MATRIX = "matrix"                   # adjacency/correlation/transition matrix answer
    ORDERING = "ordering"               # chronological order / permutation answer
    CATEGORICAL_TEXT = "categorical_text"
    EVENT_LABEL = "event_label"         # e.g. "Rain" / "No Rain"
    NONE = "none"                       # no options (open-ended row)
    UNKNOWN = "unknown"                 # census could not classify — never assume eligible


#: Option types on which the Phase-2 forecast-ranking head may fire. Frozen here so the
#: runtime guard and the census share ONE definition (ranking-by-forecast is the wrong
#: solver for categorical/event/text options — see tsr_bench/PREREGISTRATION_forecast_head.md).
FORECASTABLE_OPTION_TYPES = frozenset({OptionType.TRAJECTORY, OptionType.RANGE})


@dataclass(frozen=True, slots=True)
class Channel:
    """A single named time-series channel. ``values`` is coerced to a 1-D float array."""
    name: str
    values: np.ndarray
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))
        if self.values.ndim != 1:
            raise ValueError(f"Channel '{self.name}' values must be 1-D, got {self.values.ndim}-D")
        self.values.setflags(write=False)
        if self.timestamps is not None:
            object.__setattr__(self, "timestamps", np.asarray(self.timestamps, dtype=float))
            if self.timestamps.shape != self.values.shape:
                raise ValueError(
                    f"Channel '{self.name}' timestamps {self.timestamps.shape} "
                    f"!= values {self.values.shape}"
                )
            self.timestamps.setflags(write=False)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """A benchmark row the runtime pipeline is allowed to see.

    Deliberately carries NO ``dimension`` / ``task`` / ``domain`` — those are reporting-only
    (see :class:`ReportingLabels`). ``answer_type`` and ``option_type`` are derived from the
    *question-visible* answer/option structure (like the existing ``infer_expected_schema``),
    so gating on them is not a label leak.
    """
    id: str
    question: str
    options: list[str]
    answer: str
    answer_type: AnswerType
    option_type: OptionType = OptionType.NONE
    series: list[Channel] = field(default_factory=list)
    category: str = "unknown"

    def to_runner_row(self) -> dict:
        """Project onto the current ``run_pipeline`` row dict (``{ts, ts1, ts2}``).

        1 channel -> ``ts``; 2+ channels -> ``ts1``/``ts2``. **Channels beyond the first
        two are dropped** (the current pipeline consumes <=2 series); the canonical object
        keeps them for future N-series branches.
        """
        row = {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "options": list(self.options),
            "answer": self.answer,
            "ts": None,
            "ts1": None,
            "ts2": None,
        }
        n = len(self.series)
        if n == 1:
            row["ts"] = self.series[0].values
        elif n >= 2:
            row["ts1"] = self.series[0].values
            row["ts2"] = self.series[1].values
        return row


@dataclass(frozen=True)
class ReportingLabels:
    """Reporting-only sidecar — **never handed to the runtime pipeline**.

    Held by the adapter and exposed only to the eval / diff / stratification layer. Frozen
    (all hashable) so it can key cluster maps. Keeping these OFF :class:`BenchmarkSample` is
    what makes dimension/task leakage structurally impossible.
    """
    id: str
    dimension: Optional[str] = None
    task: Optional[str] = None
    domain: Optional[str] = None
    series_name: Optional[str] = None
    template_id: Optional[str] = None
    source_id: Optional[str] = None


def cluster_keys(labels: ReportingLabels) -> dict:
    """Cluster keys for cluster-aware bootstrap / paired permutation (0f).

    Returns only the *grouping* fields (never ``dimension``/``task``, which are outcome
    labels, not clusters). Missing keys are ``None`` and should be skipped by the bootstrap.
    """
    return {
        "domain": labels.domain,
        "series_name": labels.series_name,
        "template_id": labels.template_id,
        "source_id": labels.source_id,
    }


__all__ = [
    "AnswerType",
    "OptionType",
    "FORECASTABLE_OPTION_TYPES",
    "Channel",
    "BenchmarkSample",
    "ReportingLabels",
    "cluster_keys",
]
