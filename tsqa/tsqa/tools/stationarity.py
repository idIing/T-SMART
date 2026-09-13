import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(ts: np.ndarray) -> dict:
    """
    Test stationarity via ADF (H0: unit root) and KPSS (H0: stationary).

    Args:
        ts: 1-D time series array

    Returns:
        adf_stat: float
        adf_pval: float  — p < 0.05 → reject unit root → evidence of stationarity
        kpss_stat: float
        kpss_pval: float — p < 0.05 → reject stationarity → evidence of non-stationarity
                   Note: statsmodels clips KPSS p-values to [0.01, 0.10]; treat
                   boundary values as inequalities, not exact probabilities.
        is_stationary: bool — True only when adf_pval < 0.05 AND kpss_pval >= 0.05

    Used by: noise branch
    """
    ts_f = ts.astype(float)

    adf_stat, adf_pval, *_ = adfuller(ts_f, autolag="AIC")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_pval, *_ = kpss(ts_f, regression="c", nlags="auto")

    is_stationary = bool((adf_pval < 0.05) and (kpss_pval >= 0.05))

    return {
        "adf_stat": float(adf_stat),
        "adf_pval": float(adf_pval),
        "kpss_stat": float(kpss_stat),
        "kpss_pval": float(kpss_pval),
        "is_stationary": is_stationary,
    }
