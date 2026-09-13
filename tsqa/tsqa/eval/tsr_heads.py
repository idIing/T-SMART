"""Deterministic TSRBench heads.

These helpers are inert unless a harness explicitly passes the corresponding
runner switches. They are intentionally conservative: every unsupported parse
returns ``None`` so the normal pipeline can continue or the harness can record
an abstention.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .numeric_head import compute_numeric_answer, infer_numeric_quantity
from .sample import OptionType
from .scoring import to_float


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class HeadResult:
    predicted_letter: Optional[str] = None
    predicted_value: object = None
    head_name: str = ""
    head_note: str = ""


def _primary_series(ts, ts1, ts2):
    for arr in (ts, ts1, ts2):
        if arr is not None:
            a = np.asarray(arr, dtype=float)
            if a.size:
                return a
    return None


def _numeric_options(options: Sequence[str]) -> Optional[list[float]]:
    values = [to_float(opt) for opt in options]
    if not values or any(v is None for v in values):
        return None
    return [float(v) for v in values]


def scalar_mcq_head(question, options, ts, ts1, ts2, evidence) -> Optional[HeadResult]:
    """Choose the closest scalar option for closed-form numeric MCQs."""
    values = _numeric_options(options)
    if values is None or len(values) < 2:
        return None
    spec = infer_numeric_quantity(question)
    if spec is None:
        return None
    value, note = compute_numeric_answer(spec, ts, ts1, ts2, evidence)
    if value is None:
        return None
    idx = int(np.argmin([abs(float(value) - opt) for opt in values]))
    if idx >= len(LETTERS):
        return None
    return HeadResult(
        predicted_letter=LETTERS[idx],
        predicted_value=value,
        head_name="scalar_mcq",
        head_note=f"{spec['quantity']}:{note}",
    )


def _parse_number_list(text: str) -> Optional[np.ndarray]:
    s = str(text).strip()
    if not s:
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            arr = np.asarray(parsed, dtype=float)
            return arr.ravel() if arr.size else None
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 2:
        return np.asarray([float(n) for n in nums], dtype=float)
    return None


def _parse_range(text: str) -> Optional[tuple[float, float]]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if len(nums) < 2:
        return None
    lo, hi = float(nums[0]), float(nums[1])
    return (min(lo, hi), max(lo, hi))


def _seasonal_period(series: np.ndarray) -> Optional[int]:
    n = series.size
    if n < 8:
        return None
    centered = series - np.nanmean(series)
    spec = np.abs(np.fft.rfft(centered))
    if spec.size <= 1 or not np.isfinite(spec[1:]).any():
        return None
    k = int(np.argmax(spec[1:]) + 1)
    if k <= 0:
        return None
    p = int(round(n / k))
    return p if 2 <= p <= n // 2 else None


def _forecast_candidates(series: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    last = float(series[-1])
    persistence = np.full(horizon, last, dtype=float)
    if series.size >= 2:
        slope = (float(series[-1]) - float(series[0])) / max(series.size - 1, 1)
        drift = last + slope * np.arange(1, horizon + 1, dtype=float)
    else:
        drift = persistence.copy()
    period = _seasonal_period(series)
    if period and series.size >= period:
        tail = series[-period:]
        seasonal = np.resize(tail, horizon).astype(float)
    else:
        seasonal = persistence.copy()
    return {"persistence": persistence, "drift": drift, "seasonal_naive": seasonal}


def forecast_rank_head(question, option_type, options, series) -> Optional[HeadResult]:
    """Rank trajectory/range options against simple forecast baselines."""
    try:
        option_type = OptionType(option_type)
    except Exception:
        return None
    if option_type not in {OptionType.TRAJECTORY, OptionType.RANGE}:
        return None
    base = None
    for ch in series or []:
        values = getattr(ch, "values", ch)
        arr = np.asarray(values, dtype=float)
        if arr.size:
            base = arr
            break
    if base is None or base.size < 2 or not options:
        return None

    scores = []
    notes = []
    if option_type == OptionType.TRAJECTORY:
        parsed = [_parse_number_list(opt) for opt in options]
        if any(p is None for p in parsed):
            return None
        horizon = max(int(len(p)) for p in parsed if p is not None)
        forecasts = _forecast_candidates(base, horizon)
        for opt_arr in parsed:
            arr = np.asarray(opt_arr, dtype=float)
            best_method = None
            best_err = math.inf
            for method, fc in forecasts.items():
                n = min(len(arr), len(fc))
                err = float(np.mean((arr[:n] - fc[:n]) ** 2))
                if err < best_err:
                    best_err, best_method = err, method
            scores.append(-best_err)
            notes.append(best_method or "unknown")
    else:
        ranges = [_parse_range(opt) for opt in options]
        if any(r is None for r in ranges):
            return None
        forecasts = _forecast_candidates(base, 1)
        point_by_method = {k: float(v[0]) for k, v in forecasts.items()}
        for lo, hi in ranges:
            best_method = None
            best_score = -math.inf
            mid = (lo + hi) / 2.0
            width = max(hi - lo, 1e-12)
            for method, point in point_by_method.items():
                score = 1.0 if lo <= point <= hi else -abs(point - mid) / width
                if score > best_score:
                    best_score, best_method = score, method
            scores.append(best_score)
            notes.append(best_method or "unknown")

    idx = int(np.argmax(scores))
    if idx >= len(LETTERS):
        return None
    return HeadResult(
        predicted_letter=LETTERS[idx],
        head_name="forecast_ranker",
        head_note=f"{option_type.value}:{notes[idx]}",
    )


_COMPARISON_RE = re.compile(
    r"(?:greater than|more than|above|exceeds?|>=|at least|less than|below|under|<=|at most)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def event_head(question, options, ts, ts1, ts2) -> Optional[HeadResult]:
    """Answer only explicit threshold/comparison event-label rows."""
    q = question or ""
    m = _COMPARISON_RE.search(q)
    if not m or len(options) < 2:
        return None
    threshold = float(m.group(1))
    s = _primary_series(ts, ts1, ts2)
    if s is None:
        return None
    ql = q.lower()
    if any(tok in ql for tok in ("less than", "below", "under", "<=", "at most")):
        event = bool(np.any(s <= threshold))
    else:
        event = bool(np.any(s >= threshold))

    positive = {"yes", "true", "occur", "occurs", "event", "above", "exceed", "greater"}
    negative = {"no", "false", "not", "none", "below"}
    target_idx = None
    for i, opt in enumerate(options):
        text = str(opt).strip().lower()
        if event and any(tok in text for tok in positive):
            target_idx = i
            break
        if not event and any(tok in text for tok in negative):
            target_idx = i
            break
    if target_idx is None or target_idx >= len(LETTERS):
        return None
    return HeadResult(
        predicted_letter=LETTERS[target_idx],
        predicted_value=event,
        head_name="event_threshold",
        head_note=f"threshold={threshold:g}",
    )


__all__ = [
    "HeadResult",
    "scalar_mcq_head",
    "forecast_rank_head",
    "event_head",
]
