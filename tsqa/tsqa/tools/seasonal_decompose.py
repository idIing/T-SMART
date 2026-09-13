import numpy as np
from statsmodels.tsa.seasonal import STL


def seasonal_decompose(ts: np.ndarray, period: int = None) -> dict:
    """
    Decompose ts into trend, seasonal, and residual components using STL
    (Seasonal-Trend decomposition using LOESS).

    STL is robust to outliers and works on any periodicity, unlike classical
    additive decomposition which requires period to divide series length evenly.

    Args:
        ts:     1-D time series array (length >= 2 * period)
        period: dominant period in samples. If None, estimated from the ACF
                first peak lag; falls back to len(ts) // 10 if ACF gives nothing.
                Must be >= 2.

    Returns:
        trend:             np.ndarray — smoothed trend component, same length as ts
        seasonal:          np.ndarray — repeating seasonal component
        residual:          np.ndarray — remainder after trend + seasonal removed
        period_used:       int        — the period value actually used for decomposition
        seasonal_strength: float in [0,1] — how much variance is explained by
                           the seasonal component (0 = no seasonality, 1 = pure seasonal).
                           Formula: max(0, 1 - Var(residual) / Var(seasonal + residual))
        trend_strength:    float in [0,1] — same idea for trend component.
                           Formula: max(0, 1 - Var(residual) / Var(trend + residual))
        residual_std:      float — std of residual; large values suggest anomalies
                           or model misfit

    Used by: periodicity branch, similarity branch, trend branch
    """
    ts_f = ts.astype(float)
    n    = len(ts_f)

    # --- period estimation if not provided ---
    if period is None:
        period = _estimate_period(ts_f)
    period = max(2, int(period))

    # STL needs at least 2 full cycles
    if n < 2 * period:
        period = max(2, n // 4)

    stl    = STL(ts_f, period=period, robust=True)
    result = stl.fit()

    trend    = result.trend
    seasonal = result.seasonal
    residual = result.resid

    seasonal_strength = _component_strength(residual, seasonal)
    trend_strength    = _component_strength(residual, trend)

    return {
        "trend":             trend,
        "seasonal":          seasonal,
        "residual":          residual,
        "period_used":       period,
        "seasonal_strength": seasonal_strength,
        "trend_strength":    trend_strength,
        "residual_std":      float(np.std(residual)),
    }


def _estimate_period(ts: np.ndarray) -> int:
    """Estimate dominant period via ACF first significant peak."""
    try:
        from statsmodels.tsa.stattools import acf
        from scipy.signal import argrelmax

        # detrend before ACF so a linear trend doesn't distort lag detection
        from scipy.signal import detrend as sp_detrend
        ts_dt = sp_detrend(ts, type="linear")

        max_lag   = min(len(ts_dt) // 2 - 1, 200)
        acf_vals  = acf(ts_dt, nlags=max_lag, fft=True)
        bound     = 2.0 / np.sqrt(len(ts_dt))
        peaks     = argrelmax(acf_vals[1:], order=1)[0]
        sig_peaks = [p for p in peaks if acf_vals[p + 1] > bound]
        if sig_peaks:
            return int(sig_peaks[0]) + 1
    except Exception:
        pass
    return max(2, len(ts) // 10)


def _component_strength(residual: np.ndarray, component: np.ndarray) -> float:
    """Hyndman-Athanasopoulos strength metric for trend or seasonal."""
    var_resid = np.var(residual)
    var_denom = np.var(component + residual)
    if var_denom == 0:
        return 0.0
    return float(max(0.0, 1.0 - var_resid / var_denom))
