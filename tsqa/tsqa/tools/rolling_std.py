import numpy as np


def rolling_std(ts: np.ndarray, window: int = None) -> dict:
    """
    Compute rolling standard deviation to characterise how variance evolves
    over time. Useful for distinguishing heteroscedastic from homoscedastic
    noise, and for detecting variance changes in noise/stationarity questions.

    Args:
        ts:     1-D time series array
        window: rolling window size in samples. Defaults to len(ts) // 10,
                clamped to [4, 200].

    Returns:
        window_used:        int    — window size actually used
        rolling_std_values: list   — rolling std at each position (NaN-filled
                                     at the start where the window isn't full)
        mean_rolling_std:   float  — mean of valid (non-NaN) rolling std values
        std_of_rolling_std: float  — std of rolling std values; high value
                                     means variance is changing over time
        variance_trend:     str    — "stable" | "increasing" | "decreasing"
                                     based on whether rolling std in the second
                                     half is notably higher/lower than the first
        variance_ratio:     float  — mean rolling std (second half) /
                                     mean rolling std (first half). Values
                                     notably above 1 = heteroscedastic (growing
                                     variance); near 1 = homoscedastic.

    Used by: noise branch (variance level + stationarity subtype)
    """
    ts_f = ts.astype(float)
    n    = len(ts_f)

    if window is None:
        window = int(np.clip(n // 10, 4, 200))
    window = max(2, min(window, n))

    # compute rolling std using a sliding window (pandas-free)
    rstd = np.full(n, np.nan)
    for i in range(window - 1, n):
        rstd[i] = float(np.std(ts_f[i - window + 1 : i + 1]))

    valid = rstd[~np.isnan(rstd)]
    mean_rstd = float(np.mean(valid)) if len(valid) > 0 else 0.0
    std_rstd  = float(np.std(valid))  if len(valid) > 0 else 0.0

    # compare first half vs second half of valid values to determine trend
    if len(valid) >= 4:
        mid      = len(valid) // 2
        first_h  = float(np.mean(valid[:mid]))
        second_h = float(np.mean(valid[mid:]))
        ratio    = second_h / first_h if first_h > 0 else 1.0

        if ratio > 1.20:
            variance_trend = "increasing"
        elif ratio < 0.80:
            variance_trend = "decreasing"
        else:
            variance_trend = "stable"
    else:
        ratio          = 1.0
        variance_trend = "stable"

    return {
        "window_used":        window,
        "rolling_std_values": rstd.tolist(),
        "mean_rolling_std":   mean_rstd,
        "std_of_rolling_std": std_rstd,
        "variance_trend":     variance_trend,
        "variance_ratio":     float(ratio),
    }
