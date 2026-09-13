import numpy as np


def shape_features(ts: np.ndarray, prominence_frac: float = 0.05) -> dict:
    """
    Extract morphological shape features from a time series: peaks, troughs,
    regularity, and waveform symmetry.

    Useful for similarity questions asking whether two series have the same
    "shape" (e.g. both sinusoidal, both sawtooth) and for periodicity questions
    asking about waveform type.

    Args:
        ts:               1-D time series array
        prominence_frac:  minimum peak prominence as a fraction of the series
                          range. Default 0.05 (5%). Lower = more peaks found.

    Returns:
        peak_count:           int   — number of significant peaks detected
        trough_count:         int   — number of significant troughs detected
        mean_peak_value:      float — mean value at peak locations
        mean_trough_value:    float — mean value at trough locations
        peak_to_trough_amp:   float — mean_peak_value - mean_trough_value;
                                      typical oscillation amplitude
        inter_peak_cv:        float — coefficient of variation of inter-peak
                                      intervals; low (< 0.2) = regular/periodic,
                                      high = irregular
        peak_regularity:      float in [0,1] — 1/(1+inter_peak_cv);
                                      1 = perfectly regular, 0 = chaotic
        rise_fall_ratio:      float — mean rise duration / mean fall duration;
                                      ~1 = symmetric (sine-like),
                                      >2 = slow-rise fast-fall (sawtooth),
                                      <0.5 = fast-rise slow-fall
        waveform_hint:        str   — "symmetric" | "slow_rise_fast_fall" |
                                      "fast_rise_slow_fall" | "irregular" |
                                      "insufficient_peaks"

    Used by: similarity branch (shape subtype), periodicity branch
             (waveform_type subtype)
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return _empty_result()

    ts_f  = ts.astype(float)
    n     = len(ts_f)
    rng   = float(np.max(ts_f) - np.min(ts_f))
    prom  = prominence_frac * rng if rng > 0 else 0.0

    # --- peaks ---
    peak_idx, _   = find_peaks(ts_f,  prominence=prom)
    trough_idx, _ = find_peaks(-ts_f, prominence=prom)

    peak_count   = int(len(peak_idx))
    trough_count = int(len(trough_idx))

    mean_peak_val   = float(np.mean(ts_f[peak_idx]))   if peak_count   > 0 else float(np.max(ts_f))
    mean_trough_val = float(np.mean(ts_f[trough_idx])) if trough_count > 0 else float(np.min(ts_f))
    peak_to_trough  = mean_peak_val - mean_trough_val

    # --- inter-peak regularity ---
    if peak_count >= 2:
        gaps      = np.diff(peak_idx).astype(float)
        mean_gap  = float(np.mean(gaps))
        cv        = float(np.std(gaps) / mean_gap) if mean_gap > 0 else 0.0
        regularity = float(1.0 / (1.0 + cv))
    else:
        cv         = 0.0
        regularity = 0.0

    # --- rise/fall symmetry (requires interleaved peaks and troughs) ---
    rise_fall_ratio, waveform_hint = _rise_fall(ts_f, peak_idx, trough_idx,
                                                 cv, peak_count)

    return {
        "peak_count":         peak_count,
        "trough_count":       trough_count,
        "mean_peak_value":    mean_peak_val,
        "mean_trough_value":  mean_trough_val,
        "peak_to_trough_amp": float(peak_to_trough),
        "inter_peak_cv":      float(cv),
        "peak_regularity":    float(regularity),
        "rise_fall_ratio":    rise_fall_ratio,
        "waveform_hint":      waveform_hint,
    }


def _rise_fall(ts, peak_idx, trough_idx, cv, peak_count):
    """Compute mean rise/fall durations from interleaved trough→peak→trough."""
    if peak_count < 2:
        return 1.0, "insufficient_peaks"

    # find interleaved (trough, peak, next_trough) triplets
    rise_durs = []
    fall_durs = []

    for pk in peak_idx:
        # closest trough before peak
        before = trough_idx[trough_idx < pk]
        after  = trough_idx[trough_idx > pk]
        if len(before) == 0 or len(after) == 0:
            continue
        tr_before = before[-1]
        tr_after  = after[0]
        rise_durs.append(pk - tr_before)
        fall_durs.append(tr_after - pk)

    if not rise_durs or not fall_durs:
        return 1.0, "insufficient_peaks"

    mean_rise = float(np.mean(rise_durs))
    mean_fall = float(np.mean(fall_durs))
    ratio     = float(mean_rise / mean_fall) if mean_fall > 0 else 1.0

    if cv > 0.4:
        hint = "irregular"
    elif ratio > 2.0:
        hint = "slow_rise_fast_fall"
    elif ratio < 0.5:
        hint = "fast_rise_slow_fall"
    else:
        hint = "symmetric"

    return ratio, hint


def _empty_result():
    return {
        "peak_count":         0,
        "trough_count":       0,
        "mean_peak_value":    None,
        "mean_trough_value":  None,
        "peak_to_trough_amp": None,
        "inter_peak_cv":      None,
        "peak_regularity":    0.0,
        "rise_fall_ratio":    1.0,
        "waveform_hint":      "insufficient_peaks",
    }
