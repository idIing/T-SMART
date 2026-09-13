import numpy as np
from scipy import stats


def trend_est(ts: np.ndarray) -> dict:
    """
    Estimate trend over ts by fitting and comparing linear, exponential, and
    log curve models. Returns the best fit type alongside all R² values.

    Args:
        ts: 1-D time series array

    Returns:
        slope:         float  linear regression coefficient (units/sample)
        direction:     str    "increasing" | "decreasing" | "flat"
        r2:            float  R² of linear fit
        exp_r2:        float  R² of exponential fit (None if ts has non-positive values)
        log_r2:        float  R² of log fit
        best_fit_type: str    "linear" | "exponential" | "log"

    Used by: trend branch (global + local), similarity branch
    """
    x = np.arange(len(ts), dtype=float)
    y = ts.astype(float)

    # --- linear ---
    lin   = stats.linregress(x, y)
    slope = float(lin.slope)
    r2    = float(lin.rvalue ** 2)

    threshold = 1e-4 * float(np.std(y)) if np.std(y) > 0 else 1e-10
    if abs(slope) < threshold:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    # --- exponential: fit log(y) ~ x via linregress (requires y > 0) ---
    exp_r2 = None
    if np.all(y > 0):
        try:
            exp_fit = stats.linregress(x, np.log(y))
            exp_r2  = float(exp_fit.rvalue ** 2)
        except Exception:
            pass

    # --- log: y = a + b*log(x+1) ---
    log_r2 = None
    try:
        log_fit = stats.linregress(np.log(x + 1), y)
        log_r2  = float(log_fit.rvalue ** 2)
    except Exception:
        pass

    candidates = {"linear": r2}
    if exp_r2 is not None:
        candidates["exponential"] = exp_r2
    if log_r2 is not None:
        candidates["log"] = log_r2
    best_fit_type = max(candidates, key=candidates.get)

    return {
        "slope":         slope,
        "direction":     direction,
        "r2":            r2,
        "exp_r2":        exp_r2,
        "log_r2":        log_r2,
        "best_fit_type": best_fit_type,
    }
