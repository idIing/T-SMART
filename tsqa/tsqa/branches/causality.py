import numpy as np
from ..tools import lag_corr, granger


def run(ts1: np.ndarray, ts2: np.ndarray, scope: str = "global") -> dict:
    """
    Causality branch: tests directed influence between two series.

    Tool chain: lag_corr → granger

    lag_corr is run first as a cheap pre-screen. If |best_corr| < 0.05 the
    series are essentially unrelated and granger is still run (the verifier
    will flag "weak_correlation"), but the result is unlikely to be significant.

    Args:
        ts1, ts2: 1-D arrays of equal length
        scope:    always "global" for causality questions (scope param accepted
                  for API consistency but local causality is not meaningful)

    Returns evidence dict with keys:
        branch, scope, flags, series_length
        best_lag, best_corr, corr_at_zero   — from lag_corr
        granger_best_lag                    — VAR lag order selected by AIC
        pval_12, pval_21                    — Granger p-values (ts1→ts2, ts2→ts1)
        direction                           — "1->2" | "2->1" | "both" | "none"

    pval_12: probability that ts1 does NOT Granger-cause ts2 under the null.
             Low p-value (< 0.05) means ts1 adds predictive power for ts2.

    Flags populated by verifier (checks.py), not here.
    """
    n = min(len(ts1), len(ts2))
    ts1, ts2 = ts1[:n].astype(float), ts2[:n].astype(float)

    evidence = {
        "branch": "causality",
        "scope": scope,
        "flags": [],
        "series_length": n,
    }

    lc = lag_corr(ts1, ts2, max_lag=min(60, n // 5))
    gr = granger(ts1, ts2, max_lag=min(5, n // 20))

    evidence.update({
        # lag_corr
        "best_lag":       lc["best_lag"],
        "best_corr":      lc["best_corr"],
        "corr_at_zero":   lc["corr_at_zero"],
        # granger
        "granger_best_lag": gr["best_lag"],
        "pval_12":          gr["pval_12"],
        "pval_21":          gr["pval_21"],
        "direction":        gr["direction"],
    })

    return evidence
