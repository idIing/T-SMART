import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR


def granger(ts1: np.ndarray, ts2: np.ndarray, max_lag: int = 5) -> dict:
    """
    Granger causality test between ts1 and ts2.

    Tests whether ts1 Granger-causes ts2 (pval_12) and whether ts2
    Granger-causes ts1 (pval_21) at the lag order minimising VAR AIC.

    Args:
        ts1: 1-D array
        ts2: 1-D array, same length as ts1
        max_lag: maximum lag order to consider (default 5)

    Returns:
        best_lag: int lag order selected by minimum VAR AIC over 1..max_lag
        pval_12: float p-value for ts1 → ts2 (ssr F-test)
        pval_21: float p-value for ts2 → ts1 (ssr F-test)
        direction: "1->2" | "2->1" | "both" | "none"
                   based on which p-values fall below 0.05

    Used by: causality branch
    """
    ts1_f = ts1.astype(float)
    ts2_f = ts2.astype(float)

    # select best lag by VAR AIC
    data = np.column_stack([ts1_f, ts2_f])
    best_lag = 1
    best_aic = np.inf
    for lag in range(1, max_lag + 1):
        try:
            aic = VAR(data).fit(lag).aic
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        except Exception:
            continue

    # ts1 → ts2: grangercausalitytests expects [target, predictor] columns
    def _pval(target, predictor, lag):
        import warnings
        d = np.column_stack([target, predictor])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(d, maxlag=[lag], verbose=False)
            return float(res[lag][0]["ssr_ftest"][1])
        except Exception:
            return 1.0

    pval_12 = _pval(ts2_f, ts1_f, best_lag)
    pval_21 = _pval(ts1_f, ts2_f, best_lag)

    sig12 = pval_12 < 0.05
    sig21 = pval_21 < 0.05
    if sig12 and sig21:
        direction = "both"
    elif sig12:
        direction = "1->2"
    elif sig21:
        direction = "2->1"
    else:
        direction = "none"

    return {
        "best_lag": best_lag,
        "pval_12": pval_12,
        "pval_21": pval_21,
        "direction": direction,
    }
