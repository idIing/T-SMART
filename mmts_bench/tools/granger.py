# tsqa/tools/granger.py
"""
Granger causality test for time-series evaluation.

Returns a result dict or a descriptive error string — never raises.
"""

from __future__ import annotations

import numpy as np
from typing import Union


def _lag_matrix(arr: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for OLS regression with `max_lag` lags.
    Returns (design_matrix, target_vector).
    """
    n = len(arr)
    y = arr[max_lag:]
    cols = [arr[max_lag - lag : n - lag] for lag in range(1, max_lag + 1)]
    X = np.column_stack(cols)
    return X, y


def _ols_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS fit; returns residuals."""
    A = np.column_stack([X, np.ones(len(y))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coeffs


def granger_causality(
    ts_cause: Union[list, np.ndarray],
    ts_effect: Union[list, np.ndarray],
    max_lag: int = 3,
) -> Union[dict, str]:
    """
    Test whether ``ts_cause`` Granger-causes ``ts_effect``.

    Uses an F-test comparing the restricted model (effect ~ lags of effect)
    against the unrestricted model (effect ~ lags of effect + lags of cause).

    Parameters
    ----------
    ts_cause  : 1-D array-like — the potential causal series
    ts_effect : 1-D array-like — the series to be explained
    max_lag   : number of lags to include (default 3)

    Returns
    -------
    dict  with keys 'f_statistic', 'p_value', 'max_lag',
                    'granger_causes' (bool), 'interpretation'
    str   error message on failure
    """
    # ---- Input validation ----
    try:
        cause  = np.asarray(ts_cause,  dtype=float)
        effect = np.asarray(ts_effect, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Error: Cannot convert inputs to numeric arrays — {exc}"

    if cause.ndim != 1 or effect.ndim != 1:
        return "Error: granger_causality requires two 1-D time series"

    if len(cause) != len(effect):
        return (
            f"Error: Time series must have equal length "
            f"(got {len(cause)} and {len(effect)})"
        )

    # Drop rows where either series has NaN
    valid_mask = ~(np.isnan(cause) | np.isnan(effect))
    cause  = cause[valid_mask]
    effect = effect[valid_mask]
    n      = len(cause)

    if n == 0:
        return "Error: No valid (non-NaN) observations remain after alignment"

    min_required = max_lag * 3 + 1   # rough rule: at least 3× lags + 1
    if n < min_required:
        return (
            f"Error: Time series too short for Granger Causality "
            f"(need ≥ {min_required} observations for max_lag={max_lag}, got {n})"
        )

    if max_lag < 1:
        return "Error: max_lag must be ≥ 1"

    # ---- Restricted model: effect ~ lags(effect) ----
    X_r, y = _lag_matrix(effect, max_lag)
    try:
        resid_r = _ols_residuals(X_r, y)
    except np.linalg.LinAlgError as exc:
        return f"Error: OLS fit failed for restricted model — {exc}"

    ssr_r = float(np.sum(resid_r ** 2))

    # ---- Unrestricted model: effect ~ lags(effect) + lags(cause) ----
    X_cause, _ = _lag_matrix(cause, max_lag)
    X_u = np.column_stack([X_r, X_cause])
    try:
        resid_u = _ols_residuals(X_u, y)
    except np.linalg.LinAlgError as exc:
        return f"Error: OLS fit failed for unrestricted model — {exc}"

    ssr_u = float(np.sum(resid_u ** 2))

    # ---- F-statistic ----
    T    = len(y)
    k_r  = X_r.shape[1] + 1   # +1 for intercept
    k_u  = X_u.shape[1] + 1

    denom_df = T - k_u
    if denom_df <= 0:
        return (
            f"Error: Not enough degrees of freedom for Granger F-test "
            f"(T={T}, k_u={k_u})"
        )

    if ssr_u == 0.0:
        return "Error: Residual sum of squares is zero — perfect fit, test undefined"

    # ---- F-statistic ----
    # If ssr_u >= ssr_r the unrestricted model didn't help — clamp to 0
    # and immediately return a non-significant result (p=1.0).
    if ssr_u >= ssr_r:
        return {
            "f_statistic":    0.0,
            "p_value":        1.0,
            "max_lag":        max_lag,
            "granger_causes": False,
            "interpretation": (
                "No Granger causality — unrestricted model did not reduce "
                "residual variance (F=0.0, p=1.0)"
            ),
        }

    f_stat = ((ssr_r - ssr_u) / max_lag) / (ssr_u / denom_df)

    # ---- p-value (scipy is in requirements.txt — raise if missing) ----
    from scipy.stats import f as f_dist  # type: ignore
    p_val = float(1.0 - f_dist.cdf(f_stat, dfn=max_lag, dfd=denom_df))

    alpha = 0.05
    causes = bool(p_val < alpha)

    interp = (
        f"ts_cause Granger-causes ts_effect at the {alpha} significance level "
        f"(F={f_stat:.3f}, p={p_val:.4f})"
        if causes
        else
        f"No significant Granger causality detected "
        f"(F={f_stat:.3f}, p={p_val:.4f})"
    )

    return {
        "f_statistic":     f_stat,
        "p_value":         p_val,
        "max_lag":         max_lag,
        "granger_causes":  causes,
        "interpretation":  interp,
    }
