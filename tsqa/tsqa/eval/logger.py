import json
import os
from datetime import datetime
import numpy as np


def save_run(results: list, metrics: dict, config: dict, output_dir: str = "runs") -> str:
    """
    Save a full evaluation run to a timestamped JSON file.

    Args:
        results:    list of result dicts from run_dataset
        metrics:    dict from compute_metrics
        config:     arbitrary config dict to record (model name, branch set, etc.)
        output_dir: directory to write into (created if it doesn't exist)

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename  = os.path.join(output_dir, f"run_{timestamp}.json")

    payload = {
        "timestamp": timestamp,
        "config":    config,
        "metrics":   metrics,
        "results":   [_serialisable(r) for r in results],
    }

    with open(filename, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"Run saved → {filename}")
    return filename


def _serialisable(result: dict) -> dict:
    """Strip non-serialisable values (large arrays) from a result dict."""
    _SKIP = {"acf_values", "zscore", "iqr_outliers", "outlier_indices"}
    out = {}
    for k, v in result.items():
        if k == "evidence" and isinstance(v, dict):
            out[k] = {ek: ev for ek, ev in v.items() if ek not in _SKIP}
        else:
            out[k] = v
    return out


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
