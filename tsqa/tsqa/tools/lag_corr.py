import numpy as np
from scipy.stats import pearsonr


def lag_corr(ts1: np.ndarray, ts2: np.ndarray, max_lag: int = 20) -> dict:
    """
    Cross-correlation between ts1 and ts2 swept across lags [-max_lag, +max_lag].

    A positive lag means ts2 is shifted forward (ts1 leads ts2).
    A negative lag means ts2 is shifted backward (ts2 leads ts1).

    Args:
        ts1: 1-D array
        ts2: 1-D array, same length as ts1
        max_lag: maximum absolute lag (default 20)

    Returns:
        best_lag: int lag at which |correlation| is maximised
        best_corr: float Pearson r at best_lag (signed)
        corr_at_zero: float Pearson r at lag 0

    Used by: causality branch
    """
    ts1_f = ts1.astype(float)
    ts2_f = ts2.astype(float)
    n = len(ts1_f)

    corr_at_zero = float(pearsonr(ts1_f, ts2_f)[0])

    best_lag = 0
    best_corr = corr_at_zero
    best_abs = abs(corr_at_zero)

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            # ts1 leads: compare ts1[:-lag] with ts2[lag:]
            a, b = ts1_f[:-lag], ts2_f[lag:]
        else:
            # ts2 leads: compare ts1[-lag:] with ts2[:lag]  (lag is negative)
            a, b = ts1_f[-lag:], ts2_f[:lag]

        if len(a) < 3:
            continue
        r = float(pearsonr(a, b)[0])
        if abs(r) > best_abs:
            best_abs = abs(r)
            best_corr = r
            best_lag = lag

    return {
        "best_lag": best_lag,
        "best_corr": best_corr,
        "corr_at_zero": corr_at_zero,
    }
