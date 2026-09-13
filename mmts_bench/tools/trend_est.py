# tsqa/tools/trend_est.py
"""
Trend estimation utilities for time-series evaluation.

All public functions return a result dict or a descriptive error string —
they never raise exceptions to the caller.
"""

from __future__ import annotations

import numpy as np
from typing import Union


def linear_trend(ts: Union[list, np.ndarray]) -> Union[dict, str]:
    """
    Fit a simple OLS linear trend to a 1-D time series.

    Returns
    -------
    dict  with keys 'slope', 'intercept', 'r_squared', 'direction'
    str   error message on failure
    """
    try:
        arr = np.asarray(ts, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Error: Cannot convert input to numeric array — {exc}"

    if arr.ndim != 1:
        return "Error: linear_trend requires a 1-D time series"
    if len(arr) == 0:
        return "Error: Time series is empty"
    if len(arr) < 2:
        return "Error: Time series too short for trend estimation (min length 2)"

    clean_mask = ~np.isnan(arr)
    n_clean    = int(clean_mask.sum())
    if n_clean < 2:
        return "Error: Too few non-NaN values for trend estimation (min 2)"

    x = np.where(clean_mask)[0].astype(float)
    y = arr[clean_mask]

    # OLS via numpy lstsq
    A = np.column_stack([x, np.ones_like(x)])
    try:
        result, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        return f"Error: Linear algebra failure during trend estimation — {exc}"

    slope, intercept = float(result[0]), float(result[1])

    # R²
    y_pred  = slope * x + intercept
    ss_res  = float(np.sum((y - y_pred) ** 2))
    ss_tot  = float(np.sum((y - np.mean(y)) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if abs(slope) < 1e-10:
        direction = "stationary"
    elif slope > 0:
        direction = "upward"
    else:
        direction = "downward"

    return {
        "slope":     slope,
        "intercept": intercept,
        "r_squared": r2,
        "direction": direction,
    }


def hurst_exponent(ts: Union[list, np.ndarray]) -> Union[dict, str]:
    """
    Estimate the Hurst Exponent using the R/S (rescaled range) method.

    Returns
    -------
    dict  with keys 'hurst', 'interpretation'
    str   error message on failure

    Interpretation
    --------------
    H > 0.5  → persistent (trending)
    H ≈ 0.5  → random walk
    H < 0.5  → anti-persistent (mean-reverting)

    ⚠️  Bias note: The R/S estimator is positively biased on finite series.
    Empirical values of 0.6–1.0+ are common even for pure random walks when
    n < 1000.  Treat the result as a directional signal, not an exact value.
    For publication use a corrected estimator (e.g. Whittle MLE via hurst
    or nolds packages).
    """
    try:
        arr = np.asarray(ts, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Error: Cannot convert input to numeric array — {exc}"

    if arr.ndim != 1:
        return "Error: hurst_exponent requires a 1-D time series"

    arr = arr[~np.isnan(arr)]  # drop NaNs

    if len(arr) < 20:
        return (
            "Error: Time series too short for Hurst Exponent estimation "
            "(min 20 non-NaN values recommended)"
        )

    n = len(arr)
    lags = range(2, min(n // 2, 100))
    rs_values = []

    for lag in lags:
        # Split into non-overlapping sub-series of length `lag`
        rs_list = []
        for start in range(0, n - lag + 1, lag):
            sub = arr[start : start + lag]
            mean_sub = np.mean(sub)
            devs  = np.cumsum(sub - mean_sub)
            r     = float(np.max(devs) - np.min(devs))
            s     = float(np.std(sub, ddof=1))
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 2:
        return "Error: Not enough sub-series to estimate Hurst Exponent"

    log_lags = np.log([v[0] for v in rs_values])
    log_rs   = np.log([v[1] for v in rs_values])

    try:
        A = np.column_stack([log_lags, np.ones_like(log_lags)])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_rs, rcond=None)
    except np.linalg.LinAlgError as exc:
        return f"Error: Hurst estimation regression failed — {exc}"

    h = float(coeffs[0])

    if h > 0.55:
        interpretation = "persistent (trending)"
    elif h < 0.45:
        interpretation = "anti-persistent (mean-reverting)"
    else:
        interpretation = "random walk"

    return {"hurst": h, "interpretation": interpretation}
