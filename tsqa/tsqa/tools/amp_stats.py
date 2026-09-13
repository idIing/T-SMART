import numpy as np


def amp_stats(ts: np.ndarray) -> dict:
    """
    Compute amplitude and distributional summary statistics.

    Args:
        ts: 1-D time series array

    Returns:
        mean: float
        std: float
        min: float
        max: float
        range: float, max - min
        q25: float, 25th percentile
        q75: float, 75th percentile

    Used by: anomaly branch (context), shape/level branch, similarity branch
    """
    ts_f = ts.astype(float)
    q25, q75 = np.percentile(ts_f, [25, 75])
    return {
        "mean": float(np.mean(ts_f)),
        "std": float(np.std(ts_f)),
        "min": float(np.min(ts_f)),
        "max": float(np.max(ts_f)),
        "range": float(np.max(ts_f) - np.min(ts_f)),
        "q25": float(q25),
        "q75": float(q75),
    }
