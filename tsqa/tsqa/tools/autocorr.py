import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelmax


def autocorr(ts: np.ndarray, max_lag: int = 50) -> dict:
    """
    Compute sample ACF and locate the first significant peak beyond lag 0.

    Args:
        ts: 1-D time series array
        max_lag: maximum lag to compute (default 50; capped at len(ts)//2 - 1)

    Returns:
        acf_values: np.ndarray of ACF values at lags [0 .. max_lag]
        first_peak_lag: int lag of first local ACF maximum exceeding the 95% CI bound;
                        None if no such peak exists
        peak_strength: float ACF value at first_peak_lag; None if no peak found

    Used by: periodicity branch, noise branch
    """
    max_lag = min(max_lag, len(ts) // 2 - 1)
    acf_vals = acf(ts, nlags=max_lag, fft=True)

    confidence_bound = 2.0 / np.sqrt(len(ts))

    # search lags 1.. (index 0 is always 1.0)
    search = acf_vals[1:]
    local_max_offsets = argrelmax(search, order=1)[0]  # indices into search array

    significant = [i for i in local_max_offsets if search[i] > confidence_bound]

    if significant:
        first_peak_lag = int(significant[0]) + 1  # +1 to convert back to lag index
        peak_strength = float(acf_vals[first_peak_lag])
    else:
        first_peak_lag = None
        peak_strength = None

    return {
        "acf_values": acf_vals,
        "first_peak_lag": first_peak_lag,
        "peak_strength": peak_strength,
    }
