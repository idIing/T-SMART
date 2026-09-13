import numpy as np


def dtw_distance(ts1: np.ndarray, ts2: np.ndarray,
                 radius: int = None) -> dict:
    """
    Compute Dynamic Time Warping (DTW) distance between two time series.

    DTW allows elastic alignment — it stretches or compresses the time axis to
    find the best match between two series. This makes it much better than
    Euclidean distance for shape similarity questions where the series may have
    the same waveform but different phases, speeds, or lengths.

    Uses a pure-numpy Sakoe-Chiba band constrained DTW so no extra library is
    required. Falls back to an unconstrained DTW if the band is too tight.

    Args:
        ts1, ts2: 1-D time series arrays (need not be the same length)
        radius:   Sakoe-Chiba band radius in samples. Defaults to
                  max(10, min(len) // 10). Larger = more flexible alignment.

    Returns:
        dtw_distance:            float — raw DTW path cost
        dtw_distance_normalized: float — DTW cost divided by path length,
                                         comparable across different series lengths
        euclidean_distance:      float — L2 distance after z-score normalisation,
                                         for comparison
        shape_similarity:        float in [0, 1] — 1 - tanh(dtw_distance_normalized)
                                         0 = very different shapes, 1 = identical
        shorter_length:          int
        longer_length:           int
        length_ratio:            float — shorter / longer, 1.0 = same length

    Used by: similarity branch (shape comparison subtype)
    """
    a = _zscore(ts1.astype(float))
    b = _zscore(ts2.astype(float))

    n, m   = len(a), len(b)
    shorter = min(n, m)
    longer  = max(n, m)

    if radius is None:
        radius = max(10, shorter // 10)

    dtw_cost, path_len = _dtw_sakoe_chiba(a, b, radius)

    dtw_norm = dtw_cost / path_len if path_len > 0 else 0.0

    # z-scored euclidean (trim to shorter length for fair comparison)
    min_len = shorter
    euc = float(np.sqrt(np.sum((a[:min_len] - b[:min_len]) ** 2)) / min_len)

    shape_sim = float(1.0 - np.tanh(dtw_norm))

    return {
        "dtw_distance":            float(dtw_cost),
        "dtw_distance_normalized": float(dtw_norm),
        "euclidean_distance":      euc,
        "shape_similarity":        shape_sim,
        "shorter_length":          shorter,
        "longer_length":           longer,
        "length_ratio":            float(shorter / longer) if longer > 0 else 1.0,
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    return (x - np.mean(x)) / std if std > 0 else x - np.mean(x)


def _dtw_sakoe_chiba(a: np.ndarray, b: np.ndarray,
                     radius: int) -> tuple:
    """
    DTW with Sakoe-Chiba band constraint.
    Returns (total_cost, path_length).
    """
    n, m = len(a), len(b)
    INF  = float("inf")

    # initialise cost matrix with infinity
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end   = min(m, i + radius) + 1
        for j in range(j_start, j_end):
            cost    = float(abs(a[i - 1] - b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j],      # insertion
                                 D[i, j - 1],       # deletion
                                 D[i - 1, j - 1])   # match

    # if band was too tight, fall back to unconstrained
    if D[n, m] == INF:
        D = np.full((n + 1, m + 1), INF)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost    = float(abs(a[i - 1] - b[j - 1]))
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # traceback to find path length
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1

    return float(D[n, m]), path_len
