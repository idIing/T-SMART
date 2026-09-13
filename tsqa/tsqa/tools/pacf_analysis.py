import numpy as np


def pacf_analysis(ts: np.ndarray, max_lag: int = 20) -> dict:
    """
    Compute the Partial Autocorrelation Function (PACF) and classify the
    likely ARMA process type based on ACF/PACF cutoff patterns.

    ACF and PACF together identify process type:
      AR(p): ACF decays gradually, PACF cuts off sharply after lag p
      MA(q): ACF cuts off sharply after lag q, PACF decays gradually
      ARMA:  Both ACF and PACF decay gradually
      White noise: Neither has significant values beyond lag 0

    Args:
        ts:      1-D time series array
        max_lag: maximum lag to compute (capped at len(ts)//2 - 1, max 40)

    Returns:
        pacf_values:       list  — PACF values at lags [0 .. max_lag]
        pacf_at_lag1:      float — PACF value at lag 1; key discriminator
        pacf_at_lag2:      float — PACF value at lag 2
        pacf_significant_lags: list — lags where |PACF| exceeds 95% CI bound
        acf_at_lag1:       float — ACF at lag 1 (repeated here for convenience)
        acf_significant_lags: list — lags where ACF exceeds 95% CI bound
        acf_cuts_off_at:   int | None — lag after which ACF becomes
                                        consistently insignificant; None if
                                        it never cuts off cleanly
        pacf_cuts_off_at:  int | None — same for PACF
        process_type_hint: str  — "AR(1)" | "AR(p)" | "MA(1)" | "MA(q)" |
                                   "ARMA" | "white_noise" | "unknown"
        confidence:        str  — "high" | "medium" | "low"

    Used by: noise branch (ar_ma_process subtype)
    """
    from statsmodels.tsa.stattools import acf, pacf

    ts_f    = ts.astype(float)
    n       = len(ts_f)
    max_lag = min(max_lag, n // 2 - 1, 40)
    bound   = 2.0 / np.sqrt(n)   # 95% CI for both ACF and PACF

    # --- compute ACF and PACF ---
    try:
        acf_vals  = acf(ts_f,  nlags=max_lag, fft=True)
    except Exception:
        acf_vals  = np.zeros(max_lag + 1)

    try:
        pacf_vals = pacf(ts_f, nlags=max_lag, method="ywm")
    except Exception:
        pacf_vals = np.zeros(max_lag + 1)

    # significant lags (exclude lag 0 which is always 1)
    acf_sig  = [int(i) for i in range(1, len(acf_vals))
                if abs(acf_vals[i])  > bound]
    pacf_sig = [int(i) for i in range(1, len(pacf_vals))
                if abs(pacf_vals[i]) > bound]

    acf_at_lag1  = float(acf_vals[1])  if len(acf_vals)  > 1 else None
    acf_at_lag2  = float(acf_vals[2])  if len(acf_vals)  > 2 else None
    pacf_at_lag1 = float(pacf_vals[1]) if len(pacf_vals) > 1 else None
    pacf_at_lag2 = float(pacf_vals[2]) if len(pacf_vals) > 2 else None

    # cutoff detection: first lag where the function drops below bound
    # and stays below for the next 2 lags (to avoid noise spikes)
    acf_cutoff  = _cutoff_lag(acf_vals,  bound)
    pacf_cutoff = _cutoff_lag(pacf_vals, bound)

    # --- process classification ---
    process_hint, confidence = _classify(
        acf_sig, pacf_sig, acf_cutoff, pacf_cutoff, bound,
        acf_at_lag1, pacf_at_lag1
    )

    return {
        "pacf_values":            pacf_vals.tolist(),
        "pacf_at_lag1":           pacf_at_lag1,
        "pacf_at_lag2":           pacf_at_lag2,
        "pacf_significant_lags":  pacf_sig,
        "acf_at_lag1":            acf_at_lag1,
        "acf_at_lag2":            acf_at_lag2,
        "acf_significant_lags":   acf_sig,
        "acf_cuts_off_at":        acf_cutoff,
        "pacf_cuts_off_at":       pacf_cutoff,
        "process_type_hint":      process_hint,
        "confidence":             confidence,
    }


def _cutoff_lag(vals: np.ndarray, bound: float):
    """
    Return the first lag i (>= 1) where |vals[i]| < bound AND
    |vals[i+1]| < bound (two consecutive insignificant lags).
    Returns None if no clean cutoff found.
    """
    for i in range(1, len(vals) - 1):
        if abs(vals[i]) < bound and abs(vals[i + 1]) < bound:
            return int(i)
    return None


def _classify(acf_sig, pacf_sig, acf_cut, pacf_cut, bound,
              acf_lag1, pacf_lag1):
    """Map ACF/PACF patterns to ARMA process type."""
    no_acf  = len(acf_sig)  == 0
    no_pacf = len(pacf_sig) == 0

    # white noise: neither has significant values
    if no_acf and no_pacf:
        return "white_noise", "high"

    acf_cuts_early  = acf_cut  is not None and acf_cut  <= 3
    pacf_cuts_early = pacf_cut is not None and pacf_cut <= 3

    # AR(1): PACF significant only at lag 1, ACF decays
    if (pacf_sig == [1] or (len(pacf_sig) == 1 and pacf_sig[0] == 1)):
        if not acf_cuts_early:
            return "AR(1)", "high"

    # AR(p): PACF cuts off at lag p > 1, ACF decays
    if pacf_cuts_early and not acf_cuts_early and pacf_cut and pacf_cut > 1:
        return f"AR({pacf_cut})", "medium"

    # MA(1): ACF significant only at lag 1, PACF decays
    if (acf_sig == [1] or (len(acf_sig) == 1 and acf_sig[0] == 1)):
        if not pacf_cuts_early:
            return "MA(1)", "high"

    # MA(q): ACF cuts off at lag q > 1, PACF decays
    if acf_cuts_early and not pacf_cuts_early and acf_cut and acf_cut > 1:
        return f"MA({acf_cut})", "medium"

    # both cut off early → ARMA
    if acf_cuts_early and pacf_cuts_early:
        return "ARMA", "medium"

    # both decay without clear cutoff
    if len(acf_sig) > 3 and len(pacf_sig) > 3:
        return "ARMA", "low"

    return "unknown", "low"
