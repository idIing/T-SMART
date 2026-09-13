import numpy as np
import scipy.stats as stats

_VALID_OPS = (">", ">=", "<", "<=", "==", "!=")


def _clean_1d(ts) -> np.ndarray:
    """
    Coerce ``ts`` to a clean 1-D float array, dropping NaNs.

    Hardened bad-input contract (raises ``ValueError`` on):
      * None or input that cannot be coerced to a float array
      * empty array
      * non-1-D array
      * input that is entirely NaN (no finite values remain after dropping NaNs)

    NaN handling matches the project's hardened sibling tools (e.g.
    ``trend_est.linear_trend``, ``anomaly_score.zscore_anomaly_score``):
    NaNs are dropped, not propagated. Note that this re-indexes the series,
    so ``value_at_index`` operates on the NaN-free view.
    """
    if ts is None:
        raise ValueError("Input time series is None")

    try:
        arr = np.asarray(ts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert input to a numeric array — {exc}")

    if arr.ndim != 1:
        raise ValueError(f"Time series must be 1-D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("Time series is empty")

    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        raise ValueError("Time series contains only NaN values")

    return clean


def basic_stats(ts) -> dict:
    """
    Common closed-form summary statistics of a 1-D series.

    Deterministic, pure numpy — the value comes entirely from the data, there
    is nothing to overfit. NaNs are dropped before computation.

    Args:
        ts: 1-D array-like time series.

    Returns:
        dict with keys (all python floats/ints):
            n       : int, number of (non-NaN) elements
            mean    : float
            median  : float
            std     : float, POPULATION standard deviation (ddof=0)
            var     : float, POPULATION variance (ddof=0)
            min     : float
            max     : float
            range   : float, max - min
            sum     : float
            abs_max : float, max(abs(value))
            q1      : float, 25th percentile (linear interpolation)
            q3      : float, 75th percentile (linear interpolation)
            iqr     : float, q3 - q1
            first   : float, first (non-NaN) element
            last    : float, last (non-NaN) element

    Note:
        ``std``/``var`` are POPULATION statistics (ddof=0) to match the numpy
        defaults; pass through ``np.std``/``np.var`` if a sample estimate is
        needed.

    Used by: numeric/summary-stat questions across branches.
    """
    arr = _clean_1d(ts)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "var": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "sum": float(np.sum(arr)),
        "abs_max": float(np.max(np.abs(arr))),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "skewness": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
    }


def percentile(ts, q) -> float:
    """
    Return the ``q``-th percentile of the series (linear interpolation).

    Args:
        ts: 1-D array-like time series (NaNs dropped).
        q:  percentile in [0, 100].

    Returns:
        float percentile value (numpy default linear interpolation).

    Raises:
        ValueError: on bad input (see ``_clean_1d``) or if ``q`` not in [0, 100].
    """
    arr = _clean_1d(ts)
    try:
        q_f = float(q)
    except (TypeError, ValueError):
        raise ValueError(f"Percentile q must be a number in [0, 100], got {q!r}")
    if np.isnan(q_f) or q_f < 0.0 or q_f > 100.0:
        raise ValueError(f"Percentile q must be in [0, 100], got {q_f}")
    return float(np.percentile(arr, q_f))


def value_at_index(ts, i) -> float:
    """
    Return ``ts[i]`` (supports negative indices).

    Operates on the NaN-free view of the series (NaNs are dropped first), so
    indices refer to positions among the finite values.

    Args:
        ts: 1-D array-like time series (NaNs dropped).
        i:  integer index; negative indices count from the end.

    Returns:
        float value at position ``i``.

    Raises:
        ValueError: on bad input (see ``_clean_1d``) or if ``i`` is not an integer.
        IndexError: if ``i`` is out of range.
    """
    arr = _clean_1d(ts)
    try:
        idx = int(i)
    except (TypeError, ValueError):
        raise ValueError(f"Index must be an integer, got {i!r}")
    if idx != float(i):  # reject non-integer floats like 1.5
        raise ValueError(f"Index must be an integer, got {i!r}")
    n = arr.size
    if idx < -n or idx >= n:
        raise IndexError(f"Index {idx} out of range for series of length {n}")
    return float(arr[idx])


def count_threshold(ts, op, threshold) -> int:
    """
    Count elements satisfying ``(element op threshold)``.

    Args:
        ts:        1-D array-like time series (NaNs dropped).
        op:        comparison operator, one of {'>', '>=', '<', '<=', '==', '!='}.
        threshold: numeric threshold to compare against.

    Returns:
        int count of elements satisfying the comparison.

    Raises:
        ValueError: on bad input (see ``_clean_1d``), unknown ``op``, or a
                    non-numeric ``threshold``.
    """
    arr = _clean_1d(ts)
    if op not in _VALID_OPS:
        raise ValueError(
            f"Unknown operator {op!r}; expected one of {_VALID_OPS}"
        )
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        raise ValueError(f"Threshold must be a number, got {threshold!r}")

    if op == ">":
        mask = arr > thr
    elif op == ">=":
        mask = arr >= thr
    elif op == "<":
        mask = arr < thr
    elif op == "<=":
        mask = arr <= thr
    elif op == "==":
        mask = arr == thr
    else:  # "!="
        mask = arr != thr

    return int(np.count_nonzero(mask))
