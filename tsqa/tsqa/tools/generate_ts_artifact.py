import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def generate_ts_artifact(ts_array, viz_type, series_id):
    """
    Render a time series as an image artifact and return a JSON-serializable dict.
    """
    plt.ioff()
    fig = None
    try:
        arr = _to_2d_array(ts_array)
        n_channels, seq_len = arr.shape
        if seq_len < 2:
            raise ValueError("ts_array must have length >= 2")

        cleaned = np.zeros_like(arr, dtype=float)
        for i in range(n_channels):
            cleaned[i] = _clean_channel(arr[i])

        viz = str(viz_type).strip().lower()
        if viz not in ("line_plot", "spectrogram", "stacked"):
            raise ValueError("viz_type must be 'line_plot', 'spectrogram', or 'stacked'")

        import tempfile

        base_dir = os.path.join(tempfile.gettempdir(), "ts_artifacts")
        os.makedirs(base_dir, exist_ok=True)
        safe_id = _sanitize_series_id(series_id)
        file_name = f"{safe_id}_{viz}.png"
        path = os.path.join(base_dir, file_name)

        layout = "single"
        series_map = {}
        color_map = {}

        if viz == "line_plot":
            layout = "single_panel_overlay"
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2 * n_channels))
            x = np.arange(seq_len)
            palette = ["#FF0000", "#00FF00", "#0000FF"]
            for i in range(n_channels):
                ax.plot(x, cleaned[i], alpha=0.75, color=palette[i % len(palette)])
                c_hex = palette[i % len(palette)]
                color_map[c_hex] = f"series_{i}"
            ax.axis("off")
        elif viz == "stacked":
            # Top panel: line-plot overlay; bottom panels: per-channel spectrograms.
            # Designed for SA similarity comparisons — fuses spatial morphology and
            # spectral structure in one image.
            n_rows = 1 + n_channels  # 1 line-overlay + 1 spectrogram per channel
            layout = "stacked_line_spectrogram"
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=1,
                figsize=(6, 2 * n_rows),
                gridspec_kw={"height_ratios": [2] + [1] * n_channels},
            )
            if n_rows == 1:
                axes = [axes]
            x = np.arange(seq_len)
            palette = ["#FF0000", "#00FF00", "#0000FF"]
            for i in range(n_channels):
                axes[0].plot(x, cleaned[i], alpha=0.75, color=palette[i % len(palette)])
                c_hex = palette[i % len(palette)]
                color_map[c_hex] = f"series_{i}"
            axes[0].axis("off")
            for i in range(n_channels):
                series_map[f"panel_{i + 1}"] = f"series_{i}"
                z = _standardize(cleaned[i])
                _, _, sxx = spectrogram(z)
                log_sxx = 10.0 * np.log10(sxx + 1e-10)
                axes[i + 1].imshow(log_sxx, aspect="auto", origin="lower", cmap="magma")
                axes[i + 1].axis("off")
        else:
            layout = "vertical_stack" if n_channels > 1 else "single_panel"
            for i in range(n_channels):
                series_map[f"panel_{i}"] = f"series_{i}"
            fig, axes = plt.subplots(
                nrows=n_channels,
                ncols=1,
                figsize=(6, 2 * n_channels),
                sharex=True,
            )
            if n_channels == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                z = _standardize(cleaned[i])
                _, _, sxx = spectrogram(z)
                log_sxx = 10.0 * np.log10(sxx + 1e-10)
                ax.imshow(log_sxx, aspect="auto", origin="lower", cmap="magma")
                ax.axis("off")

        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=100)
        fig.clf()
        plt.close("all")

        return {
            "layout": layout,
            "series_map": series_map,
            "color_map": color_map,
            "evidence": {
                "tool_executed": "generate_ts_artifact",
                "status": "success",
                "metadata": {
                    "viz_type": viz,
                    "sequence_length": int(seq_len),
                    "channels": int(n_channels),
                },
            },
            "artifacts": [
                {
                    "kind": "image",
                    "path": path,
                    "mime_type": "image/png",
                    "caption": f"Topological rendering of {safe_id} using {viz}.",
                    "usage_instructions": (
                        "Pass this image path to the Verifier LLM to extract global "
                        "topological heuristics, structural breaks, or phase shifts."
                    ),
                }
            ],
        }
    except Exception as e:
        if fig is not None:
            fig.clf()
        plt.close("all")
        return {
            "evidence": {
                "tool_executed": "generate_ts_artifact",
                "status": "failed",
                "error_message": str(e),
            },
            "artifacts": [],
        }


def _to_2d_array(ts_array):
    try:
        arr = np.asarray(ts_array, dtype=float)
    except Exception as e:
        raise ValueError("ts_array must be array-like") from e

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("ts_array must be 1D or 2D")

    if arr.shape[1] == 0:
        raise ValueError("ts_array must be non-empty")

    return arr


def _clean_channel(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("channel is empty")

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("channel has no finite values")

    if finite.all():
        return values

    idx = np.arange(values.size)
    if finite.sum() == 1:
        return np.full_like(values, values[finite][0], dtype=float)

    filled = values.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    return filled


def _standardize(values):
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std > 0:
        return (values - mean) / std
    return values - mean


def _sanitize_series_id(series_id):
    safe = str(series_id) if series_id is not None else "series"
    for sep in (os.sep, os.altsep):
        if sep:
            safe = safe.replace(sep, "_")
    safe = safe.replace(" ", "_")
    return safe or "series"
