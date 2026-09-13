import numpy as np
from ..tools import amp_stats, trend_est, fft_period, chunker, seasonal_decompose, dtw_distance, shape_features


def run(ts1: np.ndarray, ts2: np.ndarray, scope: str = "global") -> dict:
    """
    Similarity branch: compares two series across amplitude, trend, frequency,
    seasonal structure, and shape (DTW).

    Tool chain: amp_stats × 2 → trend_est × 2 → fft_period × 2
                → seasonal_decompose × 2 → dtw_distance
    Local scope also chunks each series and compares per-segment amp_stats.

    Args:
        ts1, ts2: 1-D time series arrays (need not be the same length)
        scope:    "global" — compare summary stats over full series
                  "local"  — chunker(n=2) on each, compare chunk-level amp_stats

    Returns evidence dict with keys:
        branch, scope, flags, series_length_1, series_length_2
        Global:
          amp_stats_1, amp_stats_2         — amplitude summaries
          trend_1, trend_2                 — {slope, direction, r2, best_fit_type}
          fft_1, fft_2                     — {dominant_period, dominant_freq, spectral_entropy}
          seasonal_strength_1/2            — STL seasonal strength per series [0,1]
          both_strongly_periodic           — bool, True if both seasonal_strength > 0.4
          mean_diff, range_ratio           — scalar level comparisons
          shape_similarity                 — float [0,1] from DTW; 1 = identical shape
          dtw_distance_normalized          — raw normalised DTW cost (lower = more similar)
        Local (adds):
          chunk_amp_1, chunk_amp_2         — list of amp_stats dicts per chunk

    shape_similarity: key signal for "do the two series have the same waveform
        shape" questions. Uses z-score normalised DTW so amplitude differences
        don't dominate — only shape matters.

    Flags populated by verifier (checks.py), not here.
    """
    evidence = {
        "branch": "similarity",
        "scope": scope,
        "flags": [],
        "series_length_1": len(ts1),
        "series_length_2": len(ts2),
    }

    amp1 = amp_stats(ts1)
    amp2 = amp_stats(ts2)
    tr1  = trend_est(ts1)
    tr2  = trend_est(ts2)
    fft1 = fft_period(ts1)
    fft2 = fft_period(ts2)

    # STL seasonal strength for each series
    def _seasonal_strength(ts, fft_result):
        try:
            p = int(round(fft_result["dominant_period"])) if fft_result["dominant_period"] else None
            return seasonal_decompose(ts, period=p)["seasonal_strength"]
        except Exception:
            return None

    ss1 = _seasonal_strength(ts1, fft1)
    ss2 = _seasonal_strength(ts2, fft2)
    both_periodic = (ss1 is not None and ss2 is not None and
                     ss1 > 0.4 and ss2 > 0.4)

    range1, range2 = amp1["range"], amp2["range"]
    range_ratio = (
        min(range1, range2) / max(range1, range2)
        if max(range1, range2) > 0 else 1.0
    )

    # DTW shape similarity (z-score normalised, so amplitude doesn't matter)
    try:
        dtw_result    = dtw_distance(ts1, ts2)
        shape_sim     = dtw_result["shape_similarity"]
        dtw_norm      = dtw_result["dtw_distance_normalized"]
    except Exception:
        shape_sim = None
        dtw_norm  = None

    evidence.update({
        "amp_stats_1":  amp1,
        "amp_stats_2":  amp2,
        "trend_1":      {k: tr1[k] for k in ("slope", "direction", "r2", "best_fit_type")},
        "trend_2":      {k: tr2[k] for k in ("slope", "direction", "r2", "best_fit_type")},
        "fft_1":        {k: fft1[k] for k in ("dominant_period", "dominant_freq", "spectral_entropy")},
        "fft_2":        {k: fft2[k] for k in ("dominant_period", "dominant_freq", "spectral_entropy")},
        "seasonal_strength_1":    ss1,
        "seasonal_strength_2":    ss2,
        "both_strongly_periodic": both_periodic,
        "mean_diff":              float(abs(amp1["mean"] - amp2["mean"])),
        "range_ratio":            float(range_ratio),
        "shape_similarity":        shape_sim,
        "dtw_distance_normalized": dtw_norm,
    })

    # Derived comparison labels for easier LLM interpretation
    try:
        # Trend similarity
        slope_diff = abs(tr1["slope"] - tr2["slope"])
        if tr1["direction"] == tr2["direction"] and slope_diff < 0.1:
            trend_similarity = "high"
        elif tr1["direction"] == tr2["direction"] or slope_diff < 0.5:
            trend_similarity = "medium"
        else:
            trend_similarity = "low"

        # Level similarity
        mean_diff = evidence["mean_diff"]
        if mean_diff < 0.1:
            level_similarity = "high"
        elif mean_diff < 1.0:
            level_similarity = "medium"
        else:
            level_similarity = "low"

        # Range / amplitude similarity
        if range_ratio > 0.8:
            range_similarity = "high"
        elif range_ratio > 0.5:
            range_similarity = "medium"
        else:
            range_similarity = "low"

        # Variance / noise-level comparison
        std1 = amp1["std"]
        std2 = amp2["std"]

        if max(std1, std2) > 0:
            std_ratio = min(std1, std2) / max(std1, std2)
        else:
            std_ratio = 1.0

        if std_ratio > 0.85:
            variance_similarity = "high"
            higher_variance_series = "same"
        elif std1 > std2:
            variance_similarity = "low"
            higher_variance_series = "series_1"
        else:
            variance_similarity = "low"
            higher_variance_series = "series_2"

        # Scaled-version detection
        try:
            a = np.asarray(ts1, dtype=float)
            b = np.asarray(ts2, dtype=float)

            if a.size == b.size and a.size > 1:
                denom_12 = float(np.dot(b, b)) + 1e-12
                scale_12 = float(np.dot(a, b) / denom_12)  # ts1 ≈ scale_12 * ts2

                pred_a = scale_12 * b
                scale_error_12 = float(np.linalg.norm(a - pred_a) / (np.linalg.norm(a) + 1e-12))

                denom_21 = float(np.dot(a, a)) + 1e-12
                scale_21 = float(np.dot(b, a) / denom_21)  # ts2 ≈ scale_21 * ts1

                pred_b = scale_21 * a
                scale_error_21 = float(np.linalg.norm(b - pred_b) / (np.linalg.norm(b) + 1e-12))

                best_scale_error = min(scale_error_12, scale_error_21)

                if best_scale_error < 0.15 and abs(scale_12 - 1.0) > 0.15:
                    scaled_version_detected = True
                    if scale_error_12 <= scale_error_21:
                        scaled_version_direction = "series_1_scaled_from_series_2"
                        scale_factor = scale_12
                    else:
                        scaled_version_direction = "series_2_scaled_from_series_1"
                        scale_factor = scale_21
                else:
                    scaled_version_detected = False
                    scaled_version_direction = "none"
                    scale_factor = None
            else:
                scaled_version_detected = False
                scaled_version_direction = "unknown"
                scale_factor = None
                best_scale_error = None

        except Exception:
            scaled_version_detected = False
            scaled_version_direction = "unknown"
            scale_factor = None
            best_scale_error = None

        # Flipped-version detection
        try:
            if a.size == b.size and a.size > 1:
                corr = float(np.corrcoef(a, b)[0, 1])
                flipped_corr = float(np.corrcoef(a, -b)[0, 1])

                flipped_version_detected = flipped_corr > 0.85
                direct_correlation = corr
                flipped_correlation = flipped_corr

                # Lag correlation detection
                try:
                    from ..tools import lag_corr
                    lc = lag_corr(a, b)
                    best_lag = lc["best_lag"]
                    best_lag_corr = lc["best_corr"]
                except Exception:
                    best_lag = 0
                    best_lag_corr = corr
            else:
                flipped_version_detected = False
                direct_correlation = None
                flipped_correlation = None
                best_lag = None
                best_lag_corr = None

        except Exception:
            flipped_version_detected = False
            direct_correlation = None
            flipped_correlation = None
            best_lag = None
            best_lag_corr = None

        # Autocorrelation similarity
        try:
            def _acf_summary(x, max_lag=20):
                x = np.asarray(x, dtype=float)
                x = x - np.mean(x)
                denom = np.dot(x, x) + 1e-12

                vals = []
                for lag in range(1, min(max_lag, len(x) - 1) + 1):
                    vals.append(float(np.dot(x[:-lag], x[lag:]) / denom))

                if not vals:
                    return {
                        "acf_values": [],
                        "acf_peak_lag": None,
                        "acf_peak_strength": None,
                        "acf_mean_abs": None,
                    }

                vals_arr = np.asarray(vals, dtype=float)
                peak_idx = int(np.argmax(np.abs(vals_arr)))

                return {
                    "acf_values": vals,
                    "acf_peak_lag": peak_idx + 1,
                    "acf_peak_strength": float(vals_arr[peak_idx]),
                    "acf_mean_abs": float(np.mean(np.abs(vals_arr))),
                }

            acf1 = _acf_summary(ts1)
            acf2 = _acf_summary(ts2)

            if acf1["acf_values"] and acf2["acf_values"]:
                acf_corr = float(np.corrcoef(acf1["acf_values"], acf2["acf_values"])[0, 1])

                if acf_corr > 0.8:
                    autocorr_similarity = "high"
                elif acf_corr > 0.5:
                    autocorr_similarity = "medium"
                else:
                    autocorr_similarity = "low"
            else:
                acf_corr = None
                autocorr_similarity = "unknown"

        except Exception:
            acf1 = None
            acf2 = None
            acf_corr = None
            autocorr_similarity = "unknown"

        # Shape similarity label from DTW
        if shape_sim is None:
            shape_similarity_label = "unknown"
        elif shape_sim > 0.8:
            shape_similarity_label = "high"
        elif shape_sim > 0.5:
            shape_similarity_label = "medium"
        else:
            shape_similarity_label = "low"


        # Natural-language overall similarity summary
        periodic_text = (
            "with matching periodic behavior"
            if both_periodic
            else "with differing periodic behavior"
        )

        if (
            trend_similarity == "high"
            and shape_similarity_label == "high"
        ):
            overall_similarity_summary = (
                f"The two series exhibit highly similar waveform shapes and trends "
                f"{periodic_text}, with {range_similarity} amplitude similarity."
            )

        elif (
            trend_similarity == "low"
            or shape_similarity_label == "low"
        ):
            overall_similarity_summary = (
                f"The series differ substantially in trend direction or waveform structure "
                f"{periodic_text}, despite having {range_similarity} amplitude similarity."
            )

        else:
            overall_similarity_summary = (
                f"The two series share moderately similar trends and waveform characteristics "
                f"{periodic_text}."
            )

        evidence.update({
            "trend_similarity": trend_similarity,
            "level_similarity": level_similarity,
            "range_similarity": range_similarity,
            "shape_similarity_label": shape_similarity_label,
            "overall_similarity_summary": overall_similarity_summary,
            "std_ratio": float(std_ratio),
            "variance_similarity": variance_similarity,
            "higher_variance_series": higher_variance_series,
            "scaled_version_detected": scaled_version_detected,
            "scaled_version_direction": scaled_version_direction,
            "scale_factor": scale_factor,
            "best_scale_error": best_scale_error,
            "flipped_version_detected": flipped_version_detected,
            "direct_correlation": direct_correlation,
            "flipped_correlation": flipped_correlation,
            "best_lag": best_lag,
            "best_lag_corr": best_lag_corr,
            "acf_1": acf1,
            "acf_2": acf2,
            "acf_similarity_corr": acf_corr,
            "autocorr_similarity": autocorr_similarity,
        })

    except Exception as e:
        evidence["similarity_label_error"] = str(e)

    # shape features for each series (peak/trough regularity, waveform hint)
    try:
        sf1 = shape_features(ts1)
        sf2 = shape_features(ts2)
        evidence.update({
            "shape_features_1": {k: sf1[k] for k in (
                "peak_count", "peak_regularity", "rise_fall_ratio", "waveform_hint"
            )},
            "shape_features_2": {k: sf2[k] for k in (
                "peak_count", "peak_regularity", "rise_fall_ratio", "waveform_hint"
            )},
            "waveform_hints_match": sf1["waveform_hint"] == sf2["waveform_hint"],
        })
    except Exception:
        evidence.update({
            "shape_features_1":    None,
            "shape_features_2":    None,
            "waveform_hints_match": None,
        })

    if scope == "local":
        c1 = chunker(ts1, n_chunks=2)
        c2 = chunker(ts2, n_chunks=2)
        evidence.update({
            "chunk_amp_1": [amp_stats(c) for c in c1["chunks"]],
            "chunk_amp_2": [amp_stats(c) for c in c2["chunks"]],
        })

    return evidence
