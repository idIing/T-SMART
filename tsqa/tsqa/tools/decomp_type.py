import numpy as np


def decomp_type(ts: np.ndarray, period: int = None) -> dict:
    """
    Classify whether a time series is better described by additive or
    multiplicative decomposition.

    Additive:       Y(t) = Trend + Seasonal + Residual
                    Seasonal amplitude stays constant regardless of trend level.
    Multiplicative: Y(t) = Trend × Seasonal × Residual
                    Seasonal amplitude grows/shrinks proportionally with trend.

    Uses two complementary methods and combines them:

    Method 1 — Residual variance comparison:
        Fit STL on raw series (additive assumption) → residual variance.
        Fit STL on log(ts) (equivalent to multiplicative) → residual variance.
        Normalise each by the variance of the (log-)series for a fair comparison.
        Whichever gives a lower normalised residual variance wins.
        Only applicable when all ts values are strictly positive.

    Method 2 — Seasonal amplitude vs trend correlation:
        From the additive STL, compute the rolling amplitude of the seasonal
        component in windows. If this amplitude correlates strongly with the
        trend level, the relationship is multiplicative (amplitude scales with
        level). Low correlation → additive.

    Args:
        ts:     1-D time series array
        period: dominant period in samples. If None, estimated internally.
                Must be >= 2.

    Returns:
        decomp_type:          str   — "additive" | "multiplicative" | "unknown"
        confidence:           str   — "high" | "medium" | "low"
        additive_resid_var:   float — normalised residual variance under additive
                                      assumption (lower = better additive fit)
        mult_resid_var:       float — normalised residual variance under
                                      multiplicative assumption; None if series
                                      has non-positive values
        amplitude_trend_corr: float — Pearson correlation between seasonal
                                      amplitude and trend level [−1, 1].
                                      Values > 0.5 suggest multiplicative.
        all_positive:         bool  — whether the series is strictly positive
                                      (required for multiplicative to be valid)
        method_used:          str   — "both" | "residual_only" | "amplitude_only"
                                      | "fallback"

    Used by: periodicity branch (decomposition_type subtype), trend branch
    """
    from .seasonal_decompose import seasonal_decompose as _stl

    ts_f = ts.astype(float)
    n    = len(ts_f)
    all_positive = bool(np.all(ts_f > 0))

    if period is None:
        period = _estimate_period(ts_f)
    period = max(2, int(period))
    if n < 2 * period:
        period = max(2, n // 4)

    # ------------------------------------------------------------------
    # Method 1: residual variance comparison
    # ------------------------------------------------------------------
    add_resid_var  = None
    mult_resid_var = None

    try:
        stl_add   = _stl(ts_f, period=period)
        add_resid = stl_add["residual"]
        ts_var    = np.var(ts_f)
        add_resid_var = float(np.var(add_resid) / ts_var) if ts_var > 0 else None
    except Exception:
        pass

    if all_positive:
        try:
            log_ts    = np.log(ts_f)
            stl_log   = _stl(log_ts, period=period)
            log_resid = stl_log["residual"]
            log_var   = np.var(log_ts)
            mult_resid_var = float(np.var(log_resid) / log_var) if log_var > 0 else None
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Method 2: seasonal amplitude vs trend correlation
    # ------------------------------------------------------------------
    amp_trend_corr = None

    try:
        stl_add     = _stl(ts_f, period=period)
        seasonal    = stl_add["seasonal"]
        trend_comp  = stl_add["trend"]

        # rolling amplitude of seasonal in windows of 2*period
        win = min(2 * period, n // 4)
        win = max(win, 4)
        amps   = []
        levels = []
        for i in range(0, n - win, win // 2):
            seg = seasonal[i: i + win]
            amps.append(float(np.max(seg) - np.min(seg)))
            levels.append(float(np.mean(np.abs(trend_comp[i: i + win]))))

        if len(amps) >= 4:
            amps_arr   = np.array(amps)
            levels_arr = np.array(levels)
            # pearson correlation
            if np.std(amps_arr) > 0 and np.std(levels_arr) > 0:
                amp_trend_corr = float(np.corrcoef(amps_arr, levels_arr)[0, 1])
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Combine signals → final classification
    # ------------------------------------------------------------------
    decomp, confidence, method = _classify(
        add_resid_var, mult_resid_var, amp_trend_corr, all_positive
    )

    return {
        "decomp_type":          decomp,
        "confidence":           confidence,
        "additive_resid_var":   add_resid_var,
        "mult_resid_var":       mult_resid_var,
        "amplitude_trend_corr": amp_trend_corr,
        "all_positive":         all_positive,
        "method_used":          method,
    }


def _classify(add_rv, mult_rv, amp_corr, all_positive):
    """Combine method signals into a final classification."""
    votes_add  = 0
    votes_mult = 0
    method     = "fallback"

    # residual variance vote
    if add_rv is not None and mult_rv is not None:
        method = "both"
        margin = abs(add_rv - mult_rv)
        if add_rv < mult_rv:
            votes_add  += 2 if margin > 0.05 else 1
        else:
            votes_mult += 2 if margin > 0.05 else 1
    elif add_rv is not None:
        method = "amplitude_only" if amp_corr is not None else "fallback"

    # amplitude-trend correlation vote
    if amp_corr is not None:
        if method == "fallback":
            method = "amplitude_only"
        if amp_corr > 0.5:
            votes_mult += 2 if amp_corr > 0.75 else 1
        elif amp_corr < 0.2:
            votes_add  += 1

    # non-positive series → can only be additive
    if not all_positive:
        if votes_mult > 0:
            votes_mult = 0
        votes_add = max(votes_add, 1)

    total = votes_add + votes_mult
    if total == 0:
        return "unknown", "low", method

    if votes_add > votes_mult:
        decomp     = "additive"
        confidence = "high" if votes_add >= 3 else ("medium" if votes_add >= 2 else "low")
    elif votes_mult > votes_add:
        decomp     = "multiplicative"
        confidence = "high" if votes_mult >= 3 else ("medium" if votes_mult >= 2 else "low")
    else:
        decomp     = "unknown"
        confidence = "low"

    return decomp, confidence, method


def _estimate_period(ts: np.ndarray) -> int:
    """Quick period estimate via ACF — mirrors seasonal_decompose._estimate_period."""
    try:
        from statsmodels.tsa.stattools import acf
        from scipy.signal import argrelmax, detrend as sp_detrend

        ts_dt   = sp_detrend(ts, type="linear")
        max_lag = min(len(ts_dt) // 2 - 1, 200)
        acf_v   = acf(ts_dt, nlags=max_lag, fft=True)
        bound   = 2.0 / np.sqrt(len(ts_dt))
        peaks   = argrelmax(acf_v[1:], order=1)[0]
        sig     = [p for p in peaks if acf_v[p + 1] > bound]
        if sig:
            return int(sig[0]) + 1
    except Exception:
        pass
    return max(2, len(ts) // 10)
