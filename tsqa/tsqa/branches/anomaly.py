import numpy as np
from ..tools import chunker, anomaly_score, amp_stats, change_point


def run(ts: np.ndarray, scope: str = "global") -> dict:
    """
    Anomaly branch: detects point anomalies (z-score/IQR) AND structural
    breaks (change point detection).

    Running both detectors in every case lets the LLM distinguish between:
      - Point anomaly  — a spike/outlier at isolated indices
      - Pattern flip   — a structural break where the whole series changes

    Args:
        ts: 1-D time series array
        scope: "global" — anomaly_score + change_point over full series
               "local"  — chunker(n=4) then anomaly_score per chunk
                          (change_point still runs globally for structural breaks)

    Returns evidence dict with keys:
        branch, scope, flags, series_length
        Global:
          outlier_indices, outlier_count, max_deviation,
          outlier_positions (normalised 0-1), amp_context
          n_breakpoints, breakpoint_positions, largest_mean_shift,
          largest_std_shift, segment_means, segment_stds
        Local (adds):
          chunk_edges, chunk_outlier_counts, chunk_max_deviations,
          chunk_outlier_indices

    n_breakpoints > 0 with a large largest_mean_shift strongly suggests a
    pattern flip / structural break rather than a point anomaly.

    Flags populated by verifier (checks.py), not here.
    """
    evidence = {
        "branch": "anomaly",
        "scope": scope,
        "flags": [],
        "series_length": len(ts),
    }

    # --- change point: always run globally regardless of scope ---
    try:
        cp = change_point(ts)
        evidence.update({
            "n_breakpoints":       cp["n_breakpoints"],
            "breakpoint_positions": cp["breakpoint_positions"],
            "largest_mean_shift":  cp["largest_mean_shift"],
            "largest_std_shift":   cp["largest_std_shift"],
            "segment_means":       cp["segment_means"],
            "segment_stds":        cp["segment_stds"],
            "cp_method":           cp["method_used"],
        })
    except Exception:
        evidence.update({
            "n_breakpoints":       None,
            "breakpoint_positions": [],
            "largest_mean_shift":  None,
            "largest_std_shift":   None,
            "segment_means":       [],
            "segment_stds":        [],
            "cp_method":           None,
        })

    if scope == "global":
        scores = anomaly_score(ts)
        stats  = amp_stats(ts)

        idx = scores["outlier_indices"]
        evidence.update({
            "outlier_indices":   idx.tolist(),
            "outlier_count":     int(len(idx)),
            "max_deviation":     scores["max_deviation"],
            "outlier_positions": (idx / len(ts)).tolist() if len(idx) > 0 else [],
            "amp_context":       stats,
        })

    else:  # local
        chunks_result = chunker(ts, n_chunks=4)
        chunk_scores  = [anomaly_score(c) for c in chunks_result["chunks"]]
        evidence.update({
            "chunk_edges":           chunks_result["chunk_edges"],
            "chunk_outlier_counts":  [int(len(s["outlier_indices"])) for s in chunk_scores],
            "chunk_max_deviations":  [s["max_deviation"]             for s in chunk_scores],
            "chunk_outlier_indices": [s["outlier_indices"].tolist()   for s in chunk_scores],
        })

    return evidence
