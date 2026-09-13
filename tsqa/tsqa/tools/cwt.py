"""
CWT scalogram artifact generator.

Ported from tsmart-v2-pipeline/tools/cwt.py and adapted for DDI compliance:
  - saves PNG to disk under /tmp/ts_artifacts/ (same as generate_ts_artifact)
  - returns {"artifacts": [{"kind": "image", "path": "..."}], ...}
  - never returns Base64 or raw tensors into the orchestration loop

generate_cwt_artifact(ts_array, series_id)      — single series
generate_dual_cwt_artifact(ts1, ts2, series_id) — two series, stacked vertically
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def _morlet(t: np.ndarray, w0: float = 5.0) -> np.ndarray:
    return np.pi ** (-0.25) * np.exp(1j * w0 * t) * np.exp(-0.5 * t**2)


def _cwt(series: np.ndarray, widths: np.ndarray) -> np.ndarray:
    n = len(series)
    cwt_matrix = np.zeros((len(widths), n), dtype=complex)
    for i, w in enumerate(widths):
        t_lim = 5.0 * w
        t = np.arange(-t_lim, t_lim + 1)
        wavelet = np.conj(_morlet(t / w)) / np.sqrt(w)
        cwt_matrix[i] = signal.convolve(series, wavelet, mode="same")
    return np.abs(cwt_matrix)


def _prep_series(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        finite_mask = np.isfinite(a)
        if not finite_mask.any():
            a = np.zeros_like(a)
        else:
            idx = np.arange(len(a))
            a[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], a[finite_mask])
    std = float(np.std(a))
    if std > 0:
        a = (a - float(np.mean(a))) / std
    return a


def _base_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "ts_artifacts")
    os.makedirs(d, exist_ok=True)
    return d


def _sanitize_id(series_id) -> str:
    safe = str(series_id) if series_id is not None else "series"
    for sep in (os.sep, os.altsep):
        if sep:
            safe = safe.replace(sep, "_")
    return safe.replace(" ", "_") or "series"


def generate_cwt_artifact(ts_array, series_id) -> dict:
    """Single-series CWT scalogram saved to disk. Returns DDI-compliant dict."""
    fig = None
    try:
        series = _prep_series(ts_array)
        if len(series) < 4:
            raise ValueError("series too short for CWT (min 4 points)")

        widths = np.arange(1, min(64, len(series) // 2))
        scalogram = _cwt(series, widths)

        safe_id = _sanitize_id(series_id)
        path = os.path.join(_base_dir(), f"{safe_id}_cwt.png")

        plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(scalogram, cmap="viridis", aspect="auto", origin="lower")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=100)
        fig.clf()
        plt.close("all")

        return {
            "layout": "single_panel",
            "series_map": {"panel_0": "ts"},
            "color_map": {},
            "evidence": {
                "tool_executed": "generate_cwt_artifact",
                "status": "success",
                "metadata": {"viz_type": "cwt", "sequence_length": int(len(series))},
            },
            "artifacts": [
                {
                    "kind": "image",
                    "path": path,
                    "mime_type": "image/png",
                    "caption": f"CWT scalogram of {safe_id}.",
                    "usage_instructions": (
                        "Pass this image path to the vision sensor to extract "
                        "topological heuristics, regime shifts, or spectral bands."
                    ),
                }
            ],
        }
    except Exception as e:
        if fig is not None:
            fig.clf()
        plt.close("all")
        return {
            "evidence": {
                "tool_executed": "generate_cwt_artifact",
                "status": "failed",
                "error_message": str(e),
            },
            "artifacts": [],
        }


def generate_dual_cwt_artifact(ts1, ts2, series_id) -> dict:
    """Two-series CWT scalograms stacked vertically, saved to disk."""
    fig = None
    try:
        s1 = _prep_series(ts1)
        s2 = _prep_series(ts2)
        if len(s1) < 4 or len(s2) < 4:
            raise ValueError("series too short for CWT (min 4 points)")

        w1 = np.arange(1, min(64, len(s1) // 2))
        w2 = np.arange(1, min(64, len(s2) // 2))
        scal1 = _cwt(s1, w1)
        scal2 = _cwt(s2, w2)

        safe_id = _sanitize_id(series_id)
        path = os.path.join(_base_dir(), f"{safe_id}_dual_cwt.png")

        plt.ioff()
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=False)
        axes[0].imshow(scal1, cmap="viridis", aspect="auto", origin="lower")
        axes[0].axis("off")
        axes[1].imshow(scal2, cmap="viridis", aspect="auto", origin="lower")
        axes[1].axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05)
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=100)
        fig.clf()
        plt.close("all")

        return {
            "layout": "vertical_stack",
            "series_map": {"panel_0": "ts1", "panel_1": "ts2"},
            "color_map": {},
            "evidence": {
                "tool_executed": "generate_dual_cwt_artifact",
                "status": "success",
                "metadata": {
                    "viz_type": "dual_cwt",
                    "sequence_lengths": [int(len(s1)), int(len(s2))],
                },
            },
            "artifacts": [
                {
                    "kind": "image",
                    "path": path,
                    "mime_type": "image/png",
                    "caption": f"Dual CWT scalogram of {safe_id} (ts1 top, ts2 bottom).",
                    "usage_instructions": (
                        "Pass this image path to the vision sensor to extract "
                        "phase relationships, lag structure, or spectral similarity."
                    ),
                }
            ],
        }
    except Exception as e:
        if fig is not None:
            fig.clf()
        plt.close("all")
        return {
            "evidence": {
                "tool_executed": "generate_dual_cwt_artifact",
                "status": "failed",
                "error_message": str(e),
            },
            "artifacts": [],
        }
