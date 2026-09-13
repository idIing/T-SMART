"""Central scoring utilities for benchmark adapters.

All benchmark harnesses should delegate answer correctness here so denominator
accounting stays consistent across TSExam, MMTS-Bench, TSRBench, and future
adapters.
"""
from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .sample import AnswerType


DEFAULT_NUMERIC_REL_TOL = 0.10
ZERO_GOLD_ATOL = 1e-9
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ScoreResult:
    sample_id: str
    answer_type: AnswerType
    eligible: bool
    correct: bool = False
    predicted: Any = None
    gold: Any = None
    rel_acc: Optional[float] = None
    reason: str = ""
    abstained: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class ScoreSummary:
    total: int
    eligible: int
    correct: int
    incorrect: int
    numeric: int
    categorical: int
    mcq: int
    open_excluded: int
    abstention: int
    error: int

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.eligible if self.eligible else None

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "eligible": self.eligible,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "mcq": self.mcq,
            "open_excluded": self.open_excluded,
            "abstention": self.abstention,
            "error": self.error,
            "accuracy": self.accuracy,
        }


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
        return out if math.isfinite(out) else None
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        out = float(match.group())
        return out if math.isfinite(out) else None


def score_numeric_value(
    pred: Any,
    gold: Any,
    *,
    rel_tol: float = DEFAULT_NUMERIC_REL_TOL,
) -> tuple[bool, Optional[float]]:
    p_val = to_float(pred)
    g_val = to_float(gold)
    if p_val is None or g_val is None:
        return False, None
    err = abs(p_val - g_val)
    if abs(g_val) < ZERO_GOLD_ATOL:
        ok = err <= ZERO_GOLD_ATOL
        return ok, 1.0 if ok else 0.0
    rel = err / abs(g_val)
    return rel <= rel_tol, max(1.0 - rel, 0.0)


def normalize_label(value: Any) -> str:
    return re.sub(r"[\s\-_]+", "", str(value).strip().lower()) if value is not None else ""


def score_categorical_value(pred: Any, gold: Any) -> bool:
    if pred is None:
        return False
    return normalize_label(pred) == normalize_label(gold)


def _score_mcq(pred: Any, gold: Any, options: Optional[list[str]] = None) -> bool:
    if pred is None or gold is None:
        return False
    pred_s = str(pred).strip()
    gold_s = str(gold).strip()
    if len(pred_s) == 1 and pred_s.upper() in LETTERS:
        if len(gold_s) == 1 and gold_s.upper() in LETTERS:
            return pred_s.upper() == gold_s.upper()
        if options:
            idx = LETTERS.index(pred_s.upper())
            return idx < len(options) and normalize_label(options[idx]) == normalize_label(gold_s)
    if len(gold_s) == 1 and gold_s.upper() in LETTERS and options:
        idx = LETTERS.index(gold_s.upper())
        if idx < len(options):
            return normalize_label(pred_s) == normalize_label(options[idx])
    return normalize_label(pred_s) == normalize_label(gold_s)


def score_prediction(
    *,
    sample_id: str,
    answer_type: AnswerType | str,
    predicted: Any,
    gold: Any,
    options: Optional[list[str]] = None,
    rel_tol: float = DEFAULT_NUMERIC_REL_TOL,
    error: Optional[str] = None,
) -> ScoreResult:
    answer_type = AnswerType(answer_type)
    if error:
        return ScoreResult(
            sample_id=str(sample_id),
            answer_type=answer_type,
            eligible=answer_type not in {AnswerType.TEXTUAL, AnswerType.OPEN},
            predicted=predicted,
            gold=gold,
            reason="error",
            error=error,
        )
    if answer_type in {AnswerType.TEXTUAL, AnswerType.OPEN}:
        return ScoreResult(
            sample_id=str(sample_id),
            answer_type=answer_type,
            eligible=False,
            predicted=predicted,
            gold=gold,
            reason="open_excluded",
            abstained=predicted is None,
        )
    if predicted is None:
        return ScoreResult(
            sample_id=str(sample_id),
            answer_type=answer_type,
            eligible=True,
            predicted=predicted,
            gold=gold,
            reason="abstained",
            abstained=True,
        )
    if answer_type == AnswerType.NUMERICAL:
        correct, rel_acc = score_numeric_value(predicted, gold, rel_tol=rel_tol)
    elif answer_type == AnswerType.CATEGORICAL:
        correct, rel_acc = score_categorical_value(predicted, gold), None
    else:
        correct, rel_acc = _score_mcq(predicted, gold, options), None
    return ScoreResult(
        sample_id=str(sample_id),
        answer_type=answer_type,
        eligible=True,
        correct=bool(correct),
        predicted=predicted,
        gold=gold,
        rel_acc=rel_acc,
        reason="scored",
    )


def summarize_scores(results: Iterable[ScoreResult]) -> ScoreSummary:
    rows = list(results)
    counts = Counter(r.answer_type for r in rows)
    return ScoreSummary(
        total=len(rows),
        eligible=sum(1 for r in rows if r.eligible),
        correct=sum(1 for r in rows if r.eligible and r.correct),
        incorrect=sum(1 for r in rows if r.eligible and not r.correct and not r.abstained and not r.error),
        numeric=counts[AnswerType.NUMERICAL],
        categorical=counts[AnswerType.CATEGORICAL],
        mcq=counts[AnswerType.MCQ],
        open_excluded=sum(1 for r in rows if not r.eligible),
        abstention=sum(1 for r in rows if r.eligible and r.abstained),
        error=sum(1 for r in rows if r.eligible and r.error),
    )


__all__ = [
    "DEFAULT_NUMERIC_REL_TOL",
    "ZERO_GOLD_ATOL",
    "ScoreResult",
    "ScoreSummary",
    "to_float",
    "score_numeric_value",
    "normalize_label",
    "score_categorical_value",
    "score_prediction",
    "summarize_scores",
]
