import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box(ts: np.ndarray, lags=10) -> dict:
    """
    Ljung-Box portmanteau test for autocorrelation (H0: white noise).

    Args:
        ts: 1-D time series array
        lags: int → test lags 1..lags; or list of specific lag values (default 10)

    Returns:
        lb_stat: list of float test statistics, one per tested lag
        lb_pval: list of float p-values, one per tested lag
        is_white_noise: bool — True if ALL tested p-values >= 0.05

    Used by: noise branch
    """
    result = acorr_ljungbox(ts.astype(float), lags=lags, return_df=True)
    lb_stat = result["lb_stat"].tolist()
    lb_pval = result["lb_pvalue"].tolist()
    is_white_noise = bool(all(p >= 0.05 for p in lb_pval))

    return {
        "lb_stat": lb_stat,
        "lb_pval": lb_pval,
        "is_white_noise": is_white_noise,
    }
