import numpy as np


def hilbert_env(ts: np.ndarray) -> dict:
    """
    Compute the instantaneous amplitude envelope of a time series using the
    Hilbert transform, then characterise whether the amplitude is increasing,
    decreasing, or stable over time.

    This directly answers questions like:
      "Does the amplitude of the cycle increase/decrease/remain the same?"

    Unlike rolling_std (which captures overall variance including trend and
    noise), the Hilbert envelope tracks the oscillation amplitude specifically,
    even when a trend is present.

    Args:
        ts: 1-D time series array. Works best on oscillatory series; for
            strongly trending series the series is linearly detrended first
            so the envelope reflects cycle amplitude rather than level.

    Returns:
        envelope_mean:      float — mean amplitude over the full series
        envelope_std:       float — std of the envelope; high = variable amplitude
        envelope_trend:     str   — "increasing" | "decreasing" | "stable"
                                    based on linear regression of the envelope
        envelope_slope:     float — slope of linear fit to the envelope
        envelope_r2:        float — R² of the linear fit; high = consistent trend
        first_half_mean:    float — mean envelope in the first half
        second_half_mean:   float — mean envelope in the second half
        amplitude_ratio:    float — second_half_mean / first_half_mean;
                                    > 1.2 = growing amplitude,
                                    < 0.8 = shrinking amplitude,
                                    ~1.0 = stable
        amplitude_change:   str   — "increase" | "decrease" | "remain the same"
                                    (matches dataset answer phrasing exactly)

    Used by: periodicity branch (waveform_type subtype), noise branch
             (variance_level subtype)
    """
    from scipy.signal import hilbert
    from scipy.signal import detrend as sp_detrend
    from scipy import stats

    ts_f = ts.astype(float)
    n    = len(ts_f)

    # detrend so the envelope reflects oscillation amplitude, not level
    try:
        ts_dt = sp_detrend(ts_f, type="linear")
    except Exception:
        ts_dt = ts_f - np.mean(ts_f)

    # Hilbert envelope
    try:
        analytic = hilbert(ts_dt)
        envelope = np.abs(analytic)
    except Exception:
        # fallback: rolling absolute deviation as crude envelope
        win      = max(4, n // 20)
        envelope = np.array([
            float(np.mean(np.abs(ts_dt[max(0, i-win):i+win])))
            for i in range(n)
        ])

    # smooth envelope slightly to reduce HF noise
    win_smooth = max(3, n // 50)
    kernel     = np.ones(win_smooth) / win_smooth
    envelope_s = np.convolve(envelope, kernel, mode="same")

    env_mean = float(np.mean(envelope_s))
    env_std  = float(np.std(envelope_s))

    # linear trend of envelope
    x         = np.arange(n, dtype=float)
    lin       = stats.linregress(x, envelope_s)
    env_slope = float(lin.slope)
    env_r2    = float(lin.rvalue ** 2)

    # threshold: slope is meaningful if change over series > 10% of mean
    total_change_frac = abs(env_slope * n) / env_mean if env_mean > 0 else 0.0
    if total_change_frac < 0.10 or env_r2 < 0.05:
        env_trend = "stable"
    elif env_slope > 0:
        env_trend = "increasing"
    else:
        env_trend = "decreasing"

    # half-comparison
    mid              = n // 2
    first_h_mean     = float(np.mean(envelope_s[:mid]))
    second_h_mean    = float(np.mean(envelope_s[mid:]))
    amp_ratio        = float(second_h_mean / first_h_mean) if first_h_mean > 0 else 1.0

    if amp_ratio > 1.20:
        amp_change = "increase"
    elif amp_ratio < 0.80:
        amp_change = "decrease"
    else:
        amp_change = "remain the same"

    return {
        "envelope_mean":     env_mean,
        "envelope_std":      env_std,
        "envelope_trend":    env_trend,
        "envelope_slope":    env_slope,
        "envelope_r2":       env_r2,
        "first_half_mean":   first_h_mean,
        "second_half_mean":  second_h_mean,
        "amplitude_ratio":   amp_ratio,
        "amplitude_change":  amp_change,
    }
