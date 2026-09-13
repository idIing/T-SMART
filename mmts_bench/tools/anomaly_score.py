# tsqa/tools/anomaly_score.py
"""
Anomaly scoring utilities for time-series evaluation.

All public functions return a float or dict on success, and a descriptive
error *string* on failure — they never raise exceptions to the caller.
"""

from __future__ import annotations

import numpy as np
from typing import Union


def zscore_anomaly_score(
    ts: Union[list, np.ndarray],
    threshold: float = 3.0,
) -> Union[dict, str]:
    """
    Detect anomalies via Z-score thresholding.

    Parameters
    ----------
    ts        : 1-D array-like of floats
    threshold : Z-score magnitude above which a point is anomalous

    Returns
    -------
    dict  with keys:
        'anomaly_indices' (list[int]),
        'anomaly_scores'  (list[float]),
        'n_anomalies'     (int)
    str   error message on failure
    """
    try:
        arr = np.asarray(ts, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Error: Cannot convert input to numeric array — {exc}"

    if arr.ndim != 1:
        return "Error: zscore_anomaly_score requires a 1-D time series"
    if len(arr) == 0:
        return "Error: Time series is empty"
    if len(arr) < 2:
        return "Error: Time series too short for Z-score anomaly detection (min length 2)"

    n_nan = int(np.isnan(arr).sum())
    if n_nan == len(arr):
        return "Error: Time series contains only NaN values"

    # Work on non-NaN values for statistics but preserve original indexing
    clean = arr[~np.isnan(arr)]
    if len(clean) < 2:
        return "Error: Too few non-NaN values for Z-score computation (min 2)"

    mu  = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))

    if std == 0.0:
        return "Error: Standard deviation is zero — Z-score undefined for constant series"

    zscores = np.abs((arr - mu) / std)
    anomaly_mask = zscores > threshold

    indices = [int(i) for i in np.where(anomaly_mask)[0]]
    scores  = [float(zscores[i]) for i in indices]

    return {
        "anomaly_indices": indices,
        "anomaly_scores":  scores,
        "n_anomalies":     len(indices),
    }


def iqr_anomaly_score(
    ts: Union[list, np.ndarray],
    k: float = 1.5,
) -> Union[dict, str]:
    """
    Detect anomalies using the IQR (box-plot) method.

    Parameters
    ----------
    ts : 1-D array-like
    k  : IQR multiplier (default 1.5 = standard box-plot rule)

    Returns
    -------
    dict  with 'anomaly_indices', 'lower_fence', 'upper_fence', 'n_anomalies'
    str   error message on failure
    """
    try:
        arr = np.asarray(ts, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Error: Cannot convert input to numeric array — {exc}"

    if arr.ndim != 1:
        return "Error: iqr_anomaly_score requires a 1-D time series"
    if len(arr) == 0:
        return "Error: Time series is empty"
    if len(arr) < 4:
        return "Error: Time series too short for IQR anomaly detection (min length 4)"

    clean = arr[~np.isnan(arr)]
    if len(clean) < 4:
        return "Error: Too few non-NaN values for IQR computation (min 4)"

    q1, q3 = float(np.percentile(clean, 25)), float(np.percentile(clean, 75))
    iqr    = q3 - q1

    if iqr == 0.0:
        return "Error: IQR is zero — anomaly fences undefined for near-constant series"

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    indices = [int(i) for i in np.where((arr < lower) | (arr > upper))[0]]

    return {
        "anomaly_indices": indices,
        "lower_fence":     lower,
        "upper_fence":     upper,
        "n_anomalies":     len(indices),
    }
