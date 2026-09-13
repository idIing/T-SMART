import numpy as np
from scipy import stats


def calculate_ols_trend(ts: np.ndarray) -> str:
    """
    Compute a global OLS slope summary string for a time series.

    Args:
        ts: 1-D time series array

    Returns:
        Formatted trend summary string.
    """
    try:
        y = np.asarray(ts, dtype=float)
        y = y[np.isfinite(y)]

        if y.size < 2:
            return (
                "Global OLS Slope = 0.00, indicating an undefined/flat trend "
                "(insufficient data)."
            )
        if float(np.std(y)) == 0.0:
            return "Global OLS Slope = 0.00, indicating a flat trend (zero variance)."

        x = np.arange(y.size, dtype=float)
        slope, _, _, _, _ = stats.linregress(x, y)

        if slope > 0:
            direction = "upward"
        elif slope < 0:
            direction = "downward"
        else:
            direction = "flat"

        return f"Global OLS Slope = {slope:.2f}, indicating a {direction} trend."
    except Exception as exc:
        return (
            "Global OLS Slope = 0.00, indicating an undefined trend " f"(error: {exc})."
        )
