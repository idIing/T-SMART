import numpy as np


def anomaly_score(ts: np.ndarray) -> dict:
    """
    Score each point for anomalousness using z-score and Tukey IQR fence.

    Args:
        ts: 1-D time series array

    Returns:
        zscore: np.ndarray of standardised scores, shape (n,)
        iqr_outliers: np.ndarray of bool, True where point falls outside
                      [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        outlier_indices: np.ndarray of int indices where iqr_outliers is True
        max_deviation: float, max(abs(zscore))

    Used by: anomaly branch (global + local)
    """
    ts_f = ts.astype(float)
    std = np.std(ts_f)
    zscore = (ts_f - np.mean(ts_f)) / std if std > 0 else np.zeros_like(ts_f)

    q25, q75 = np.percentile(ts_f, [25, 75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    iqr_outliers = (ts_f < lower) | (ts_f > upper)
    outlier_indices = np.where(iqr_outliers)[0]
    max_deviation = float(np.max(np.abs(zscore)))

    return {
        "zscore": zscore,
        "iqr_outliers": iqr_outliers,
        "outlier_indices": outlier_indices,
        "max_deviation": max_deviation,
    }
