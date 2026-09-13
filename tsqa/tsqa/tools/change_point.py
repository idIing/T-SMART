import numpy as np


def change_point(ts: np.ndarray, n_bkps: int = None,
                 penalty: float = None) -> dict:
    """
    Detect structural breakpoints in a time series — i.e., locations where the
    statistical properties (mean, variance, trend) shift abruptly.

    Tries the `ruptures` library first (PELT algorithm with RBF cost, O(n)).
    Falls back to a pure-numpy binary segmentation if ruptures is not installed.

    Args:
        ts:      1-D time series array
        n_bkps:  expected number of breakpoints. If None, PELT determines it
                 automatically via a penalty. Ignored in fallback mode (always
                 finds the single largest mean-shift breakpoint).
        penalty: PELT penalty parameter. If None, defaults to
                 3 * log(len(ts)) — a standard heuristic. Smaller penalty =
                 more breakpoints detected. Only used when n_bkps is None.

    Returns:
        n_breakpoints:        int   — number of breakpoints found
        breakpoint_locations: list  — sample indices of breakpoints (exclusive
                                      right edge of each segment, so the last
                                      value is always len(ts))
        breakpoint_positions: list  — breakpoint_locations normalised to [0,1]
                                      by dividing by len(ts)
        segment_means:        list  — mean of each segment between breakpoints
        segment_stds:         list  — std  of each segment between breakpoints
        largest_mean_shift:   float — max absolute difference between
                                      consecutive segment means; large value =
                                      strong structural break
        largest_std_shift:    float — max absolute difference between
                                      consecutive segment stds; large value =
                                      variance change point
        method_used:          str   — "ruptures" | "numpy_fallback"

    Used by: anomaly branch (pattern_flip subtype), trend branch (change_point
             subtype)
    """
    ts_f = ts.astype(float).reshape(-1, 1)
    n    = len(ts_f)

    bkps       = None
    method_used = "numpy_fallback"

    # --- try ruptures ---
    try:
        import ruptures as rpt  # noqa: F401

        if penalty is None:
            penalty = 3.0 * np.log(n)

        algo = rpt.Pelt(model="rbf", min_size=max(2, n // 20)).fit(ts_f)

        if n_bkps is not None:
            # fixed number of breakpoints
            bkps_result = rpt.Binseg(model="rbf").fit(ts_f).predict(n_bkps=n_bkps)
        else:
            bkps_result = algo.predict(pen=penalty)

        # ruptures always appends n as the final breakpoint
        bkps        = bkps_result
        method_used = "ruptures"

    except Exception:
        pass

    # --- numpy fallback: find single largest mean-shift breakpoint ---
    if bkps is None:
        bkps = _numpy_binseg(ts_f.ravel(), n_bkps=n_bkps or 1)

    # ensure the final breakpoint is exactly n (ruptures convention)
    if not bkps or bkps[-1] != n:
        bkps = [b for b in bkps if b < n] + [n]

    # build segment stats
    starts      = [0] + bkps[:-1]
    ends        = bkps
    flat        = ts_f.ravel()
    seg_means   = [float(np.mean(flat[s:e])) for s, e in zip(starts, ends)]
    seg_stds    = [float(np.std(flat[s:e]))  for s, e in zip(starts, ends)]

    mean_shifts = [abs(seg_means[i+1] - seg_means[i]) for i in range(len(seg_means)-1)]
    std_shifts  = [abs(seg_stds[i+1]  - seg_stds[i])  for i in range(len(seg_stds)-1)]

    n_bkps_found = len(bkps) - 1  # exclude the trailing n

    return {
        "n_breakpoints":        n_bkps_found,
        "breakpoint_locations": bkps[:-1],        # exclude trailing n
        "breakpoint_positions": [b / n for b in bkps[:-1]],
        "segment_means":        seg_means,
        "segment_stds":         seg_stds,
        "largest_mean_shift":   float(max(mean_shifts)) if mean_shifts else 0.0,
        "largest_std_shift":    float(max(std_shifts))  if std_shifts  else 0.0,
        "method_used":          method_used,
    }


def _numpy_binseg(ts: np.ndarray, n_bkps: int = 1) -> list:
    """
    Greedy binary segmentation: repeatedly find the single split point that
    minimises total within-segment sum of squared deviations.

    Returns list of breakpoints (exclusive right edges) including trailing n.
    """
    n      = len(ts)
    bkps   = []
    segments = [(0, n)]

    for _ in range(n_bkps):
        best_gain = -1.0
        best_bkp  = None
        best_seg  = None

        for (start, end) in segments:
            if end - start < 4:
                continue
            gain, bkp = _best_split(ts, start, end)
            if gain > best_gain:
                best_gain = gain
                best_bkp  = bkp
                best_seg  = (start, end)

        if best_bkp is None:
            break

        segments.remove(best_seg)
        segments.append((best_seg[0], best_bkp))
        segments.append((best_bkp,    best_seg[1]))
        bkps.append(best_bkp)

    bkps.sort()
    if not bkps or bkps[-1] != n:
        bkps.append(n)
    return bkps


def _best_split(ts: np.ndarray, start: int, end: int):
    """
    Find the split index in [start, end) that maximises the reduction in
    total within-segment sum of squared deviations.
    Returns (gain, split_index).
    """
    seg  = ts[start:end]
    n    = len(seg)
    best_gain = -1.0
    best_idx  = start + n // 2

    total_var = np.var(seg) * n

    # vectorised: compute left/right variance for every possible split
    left_means  = np.cumsum(seg)[:-1] / np.arange(1, n)
    right_means = (np.cumsum(seg[::-1])[:-1] / np.arange(1, n))[::-1]

    left_sq  = np.cumsum(seg**2)[:-1]
    right_sq = (np.cumsum((seg[::-1])**2)[:-1])[::-1]

    ns_l = np.arange(1, n)
    ns_r = np.arange(n-1, 0, -1)

    var_l = left_sq  / ns_l - left_means**2
    var_r = right_sq / ns_r - right_means**2

    cost   = var_l * ns_l + var_r * ns_r
    gains  = total_var - cost
    best_i = int(np.argmax(gains))
    best_gain = float(gains[best_i])

    return best_gain, start + best_i + 1
