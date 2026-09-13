import numpy as np
from statsmodels.stats.diagnostic import het_arch


def check_arch_effects(ts: np.ndarray, max_lag: int = 12) -> dict:
    """
    Engle's ARCH LM test for heteroskedasticity.

    Args:
        ts: 1-D time series array
        max_lag: number of lags for the ARCH test

    Returns:
        arch_effect_detected: bool
        pvalue: float | None
        lm_stat: float | None
        error: str | None
    """
    result = {
        "arch_effect_detected": False,
        "pvalue": None,
        "lm_stat": None,
        "error": None,
    }

    try:
        y = np.asarray(ts, dtype=float)
        y = y[np.isfinite(y)]

        if y.size < max(20, max_lag + 5):
            raise ValueError("insufficient data for ARCH test")
        if float(np.std(y)) == 0.0:
            raise ValueError("zero variance series")

        resid = y - np.mean(y)
        lm_stat, pvalue, _, _ = het_arch(resid, nlags=max_lag)
        result["lm_stat"] = float(lm_stat)
        result["pvalue"] = float(pvalue)
        result["arch_effect_detected"] = bool(pvalue < 0.05)
    except Exception as exc:
        result["error"] = str(exc)

    return result
