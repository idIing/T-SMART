"""Adapter helpers built around :class:`BenchmarkSample`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .sample import AnswerType, BenchmarkSample, Channel, OptionType, ReportingLabels


@dataclass(frozen=True)
class AdaptedSample:
    sample: BenchmarkSample
    labels: ReportingLabels


class BenchmarkAdapter(Protocol):
    def __iter__(self) -> Iterable[AdaptedSample]: ...


def infer_answer_type(*, options: list[str], answer: str, qa_type: str = "") -> AnswerType:
    qa = (qa_type or "").strip().lower()
    if len(options) >= 2 or "choice" in qa or "mcq" in qa:
        return AnswerType.MCQ
    if "numeric" in qa or "numerical" in qa:
        return AnswerType.NUMERICAL
    text = str(answer).strip()
    try:
        float(text.replace(",", ""))
        return AnswerType.NUMERICAL
    except Exception:
        pass
    if text:
        return AnswerType.CATEGORICAL
    return AnswerType.OPEN


def infer_option_type(options: list[str]) -> OptionType:
    if not options:
        return OptionType.NONE
    numeric = 0
    ranged = 0
    for opt in options:
        s = str(opt).strip().lower()
        if any(tok in s for tok in ("[", "]", "trajectory", "series", "sequence")):
            return OptionType.TRAJECTORY
        if any(tok in s for tok in (" to ", "-", "range")):
            ranged += 1
        try:
            float(s.replace(",", ""))
            numeric += 1
        except Exception:
            pass
    if numeric == len(options):
        return OptionType.SCALAR
    if ranged == len(options):
        return OptionType.RANGE
    return OptionType.CATEGORICAL_TEXT


def tsexam_row_to_sample(row: dict) -> AdaptedSample:
    options = list(row.get("options") or [])
    channels = []
    if row.get("ts") is not None:
        channels.append(Channel("ts", np.asarray(row["ts"], dtype=float)))
    if row.get("ts1") is not None:
        channels.append(Channel("ts1", np.asarray(row["ts1"], dtype=float)))
    if row.get("ts2") is not None:
        channels.append(Channel("ts2", np.asarray(row["ts2"], dtype=float)))
    sample = BenchmarkSample(
        id=str(row.get("id", "")),
        question=str(row.get("question", "")),
        options=options,
        answer=str(row.get("answer", "")),
        answer_type=infer_answer_type(options=options, answer=row.get("answer", "")),
        option_type=infer_option_type(options),
        series=channels,
        category=str(row.get("category", "unknown")),
    )
    labels = ReportingLabels(
        id=sample.id,
        dimension=row.get("dimension"),
        task=row.get("task"),
        domain=row.get("domain"),
        series_name=row.get("series_name"),
        template_id=row.get("template_id"),
        source_id=row.get("source_id"),
    )
    return AdaptedSample(sample=sample, labels=labels)


def mmts_dict_to_sample(sample_dict: dict, *, options: list[str] | None = None) -> AdaptedSample:
    options = list(options if options is not None else sample_dict.get("options") or [])
    channels = []
    if sample_dict.get("is_dual"):
        if len(sample_dict.get("ts1", [])) > 0:
            channels.append(Channel("ts1", sample_dict["ts1"]))
        if len(sample_dict.get("ts2", [])) > 0:
            channels.append(Channel("ts2", sample_dict["ts2"]))
    elif sample_dict.get("ts_array") is not None and len(sample_dict.get("ts_array", [])) > 0:
        arr = np.asarray(sample_dict["ts_array"], dtype=float)
        if arr.ndim == 1:
            channels.append(Channel("ts", arr))
    answer = str(sample_dict.get("ground_truth", ""))
    sample = BenchmarkSample(
        id=str(sample_dict.get("sample_id", "")),
        question=str(sample_dict.get("query", "")),
        options=options,
        answer=answer,
        answer_type=infer_answer_type(
            options=options,
            answer=answer,
            qa_type=str(sample_dict.get("qa_type", "")),
        ),
        option_type=infer_option_type(options),
        series=channels,
        category=str(sample_dict.get("category", "unknown")),
    )
    labels = ReportingLabels(
        id=sample.id,
        dimension=sample_dict.get("category"),
        task=sample_dict.get("qa_type"),
        domain=sample_dict.get("domain"),
        source_id=sample_dict.get("subset"),
    )
    return AdaptedSample(sample=sample, labels=labels)


__all__ = [
    "AdaptedSample",
    "BenchmarkAdapter",
    "infer_answer_type",
    "infer_option_type",
    "tsexam_row_to_sample",
    "mmts_dict_to_sample",
]
