import numpy as np
from ..tools import (
    chunker,
    trend_est,
    change_point,
    shape_features,
    calculate_ols_trend,
)


def run(ts: np.ndarray, scope: str = "global") -> dict:
    """
    Trend branch: estimates trend globally or per chunk, including functional
    form classification (linear / exponential / log).

    Args:
        ts: 1-D time series array
        scope: "global" — single trend over full series
               "local"  — chunker(n=4) then trend_est per chunk

    Returns evidence dict with keys:
        branch, scope, flags, series_length
        Global:
          slope, direction, r2, exp_r2, log_r2, best_fit_type
          slope_acceleration: slope of last chunk minus slope of first chunk
                              (positive = accelerating, negative = decelerating).
                              Key signal for exponential vs linear.
          n_breakpoints, breakpoint_positions, largest_mean_shift:
                              from change_point — non-zero n_breakpoints
                              suggests the trend changes at a specific cutoff.
        Local:
          chunk_edges, slopes, directions, r2s, best_fit_types
          slope_acceleration: same scalar derived from first/last chunk slopes
          n_breakpoints, breakpoint_positions, largest_mean_shift: same as global

    Flags populated by verifier (checks.py), not here.
    """
    evidence = {
        "branch": "trend",
        "scope": scope,
        "flags": [],
        "series_length": len(ts),
    }

    # always run chunked trend_est for slope_acceleration (n=4 gives good resolution)
    chunks_result = chunker(ts, n_chunks=4)
    chunk_trends = [trend_est(c) for c in chunks_result["chunks"]]
    chunk_slopes = [t["slope"] for t in chunk_trends]
    slope_acceleration = float(chunk_slopes[-1] - chunk_slopes[0])

    # shape / pattern features for PR
    try:
        sf = shape_features(ts)
        shape_fields = {
            "peak_count": sf["peak_count"],
            "trough_count": sf["trough_count"],
            "peak_regularity": sf["peak_regularity"],
            "rise_fall_ratio": sf["rise_fall_ratio"],
            "waveform_hint": sf["waveform_hint"],
        }

        # Derived pattern labels for easier LLM interpretation
        if sf["peak_count"] >= 2 and sf["peak_regularity"] >= 0.8:
            pattern_regularity = "high"
        elif sf["peak_count"] >= 2 and sf["peak_regularity"] >= 0.5:
            pattern_regularity = "medium"
        else:
            pattern_regularity = "low"

        if sf["peak_count"] == 0 and sf["trough_count"] == 0:
            pattern_type = "monotonic_or_flat"
        elif pattern_regularity == "high":
            pattern_type = "repeating_pattern"
        elif pattern_regularity == "medium":
            pattern_type = "somewhat_repeating"
        else:
            pattern_type = "irregular_or_noisy"

        shape_fields.update(
            {
                "pattern_regularity": pattern_regularity,
                "pattern_type": pattern_type,
            }
        )

    except Exception:
        shape_fields = {
            "peak_count": None,
            "trough_count": None,
            "peak_regularity": None,
            "rise_fall_ratio": None,
            "waveform_hint": None,
            "pattern_regularity": None,
            "pattern_type": None,
        }

    # change point: always run to detect trend-shift cutoff questions
    # Use a higher penalty than the default (3*log n) to avoid over-segmenting.
    # Default detects 4-6 breakpoints on synthetic series; 10*log(n) gives 1-3 which
    # matches what change_point questions actually ask about.
    try:
        n = len(ts)
        high_pen = 10.0 * float(np.log(n))
        cp = change_point(ts, penalty=high_pen)
        cp_fields = {
            "n_breakpoints": cp["n_breakpoints"],
            "n_segments": cp["n_breakpoints"] + 1,          # explicit so LLM never miscounts
            "breakpoint_positions": cp["breakpoint_positions"],
            "largest_mean_shift": cp["largest_mean_shift"],
        }
        # Per-segment trend analysis using actual breakpoint locations (integer indices)
        # NOTE: use "breakpoint_locations" (raw indices), NOT "breakpoint_positions" (0-1 fractions)
        bps = cp.get("breakpoint_locations") or []
        boundaries = [0] + [int(b) for b in bps] + [n]
        seg_trends = []
        for i in range(len(boundaries) - 1):
            seg = ts[boundaries[i] : boundaries[i + 1]]
            if len(seg) >= 3:
                seg_trends.append(trend_est(seg))
            else:
                seg_trends.append({"slope": None, "direction": "flat", "r2": None, "best_fit_type": "linear"})
        cp_fields["segment_best_fit_types"] = [t["best_fit_type"] for t in seg_trends]
        cp_fields["segment_slopes"] = [t["slope"] for t in seg_trends]
        cp_fields["segment_directions"] = [t["direction"] for t in seg_trends]
        cp_fields["segment_r2s"] = [t["r2"] for t in seg_trends]
    except Exception:
        cp_fields = {
            "n_breakpoints": None,
            "n_segments": None,
            "breakpoint_positions": [],
            "largest_mean_shift": None,
            "segment_best_fit_types": [],
            "segment_slopes": [],
            "segment_directions": [],
            "segment_r2s": [],
        }

    ols_summary = calculate_ols_trend(ts)

    if scope == "global":
        result = trend_est(ts)
        evidence.update(
            {
                "slope": result["slope"],
                "direction": result["direction"],
                "r2": result["r2"],
                "exp_r2": result["exp_r2"],
                "log_r2": result["log_r2"],
                "best_fit_type": result["best_fit_type"],
                "slope_acceleration": slope_acceleration,
            }
        )
        evidence.update(cp_fields)
        evidence.update(shape_fields)
        evidence["ols_trend_summary"] = ols_summary

    else:  # local
        evidence.update(
            {
                "chunk_edges": chunks_result["chunk_edges"],
                "slopes": chunk_slopes,
                "directions": [t["direction"] for t in chunk_trends],
                "r2s": [t["r2"] for t in chunk_trends],
                "best_fit_types": [t["best_fit_type"] for t in chunk_trends],
                "slope_acceleration": slope_acceleration,
            }
        )
        evidence.update(cp_fields)
        evidence.update(shape_fields)
        evidence["ols_trend_summary"] = ols_summary

    return evidence
