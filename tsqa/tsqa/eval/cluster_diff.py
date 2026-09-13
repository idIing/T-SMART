"""Cluster-aware paired-diff scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ClusterDiffResult:
    n: int
    delta: float
    ci_lo: float
    ci_hi: float
    sensitivity: dict[str, Optional[float]]


def cluster_bootstrap_delta(
    base: Iterable[int],
    treatment: Iterable[int],
    clusters: Iterable[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> ClusterDiffResult:
    base_arr = np.asarray(list(base), dtype=float)
    treat_arr = np.asarray(list(treatment), dtype=float)
    cluster_arr = np.asarray(list(clusters), dtype=object)
    if not (len(base_arr) == len(treat_arr) == len(cluster_arr)):
        raise ValueError("base, treatment, and clusters must have equal length")
    n = len(base_arr)
    if n == 0:
        return ClusterDiffResult(0, float("nan"), float("nan"), float("nan"), {})
    delta = float(treat_arr.mean() - base_arr.mean())
    unique = np.asarray(sorted(set(map(str, cluster_arr))), dtype=object)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(cluster_arr.astype(str) == c) for c in picked])
        samples.append(float(treat_arr[idx].mean() - base_arr[idx].mean()))
    leave_one = {}
    for c in unique:
        mask = cluster_arr.astype(str) != c
        leave_one[str(c)] = float(treat_arr[mask].mean() - base_arr[mask].mean()) if mask.any() else None
    return ClusterDiffResult(
        n=n,
        delta=delta,
        ci_lo=float(np.percentile(samples, 2.5)),
        ci_hi=float(np.percentile(samples, 97.5)),
        sensitivity=leave_one,
    )


def paired_permutation_p(
    base: Iterable[int],
    treatment: Iterable[int],
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    base_arr = np.asarray(list(base), dtype=float)
    treat_arr = np.asarray(list(treatment), dtype=float)
    if len(base_arr) != len(treat_arr):
        raise ValueError("base and treatment must have equal length")
    if len(base_arr) == 0:
        return 1.0
    diffs = treat_arr - base_arr
    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if abs(float((diffs * signs).mean())) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


__all__ = ["ClusterDiffResult", "cluster_bootstrap_delta", "paired_permutation_p"]
