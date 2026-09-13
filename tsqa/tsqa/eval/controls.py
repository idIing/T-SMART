"""Input controls for exact-ablation experiments."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from .sample import BenchmarkSample, Channel

ControlMode = Literal["option_only", "metadata_only", "shuffled_series", "no_series"]


def apply_control(sample: BenchmarkSample, mode: ControlMode, *, seed: int = 0) -> BenchmarkSample:
    """Return a runtime-safe controlled sample.

    ``metadata_only`` intentionally uses only runtime fields on
    :class:`BenchmarkSample`; reporting-only labels such as task/dimension are not
    accepted by this function and therefore cannot leak into the prompt.
    """
    if mode == "option_only":
        return replace(sample, question="Select the best answer from the options.", series=[])
    if mode == "metadata_only":
        return replace(
            sample,
            question="Answer using only generic benchmark metadata.",
            options=[],
            series=[],
        )
    if mode == "no_series":
        return replace(sample, series=[])
    if mode == "shuffled_series":
        rng = np.random.default_rng(seed)
        shuffled = []
        for channel in sample.series:
            values = np.array(channel.values, dtype=float, copy=True)
            rng.shuffle(values)
            shuffled.append(Channel(channel.name, values, None))
        return replace(sample, series=shuffled)
    raise ValueError(f"unknown control mode: {mode}")


__all__ = ["ControlMode", "apply_control"]
