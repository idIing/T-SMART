import numpy as np


def chunker(ts: np.ndarray, n_chunks: int = 4) -> dict:
    """
    Split ts into n_chunks equal-length segments.

    Args:
        ts: 1-D time series array
        n_chunks: number of segments (default 4)

    Returns:
        chunks: list of np.ndarray, one per segment
        chunk_edges: list of (start_idx, end_idx) tuples (end is exclusive)

    Used by: trend (local), periodicity (local), anomaly (local), shape/level (local)
    """
    n = len(ts)
    chunk_size = n // n_chunks
    edges = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else n
        edges.append((start, end))
    chunks = [ts[s:e] for s, e in edges]
    return {"chunks": chunks, "chunk_edges": edges}
