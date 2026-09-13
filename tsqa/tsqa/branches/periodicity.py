import numpy as np
from ..tools import chunker, fft_period, autocorr, seasonal_decompose, decomp_type, hilbert_env, shape_features


def run(ts: np.ndarray, scope: str = "global") -> dict:
    """
    Periodicity branch: estimates dominant period using FFT + ACF + STL.

    Args:
        ts: 1-D time series array
        scope: "global" — FFT + ACF + STL over full series, then reconcile
               "local"  — chunker(n=2) then FFT per chunk, compare periods

    Returns evidence dict with keys:
        branch, scope, flags, series_length
        Global:  dominant_period_fft, dominant_freq, spectral_entropy,
                 acf_peak_lag, acf_peak_strength, period_reconciled,
                 seasonal_strength, trend_strength, residual_std
        Local:   chunk_edges, chunk_periods, chunk_freqs, chunk_seasonal_strengths

    period_reconciled (global): best single period estimate.
        Prefers ACF peak lag when FFT and ACF agree within 20%; else uses FFT.
    seasonal_strength: from STL — 0 means no periodicity, 1 means pure seasonal.

    Flags populated by verifier (checks.py), not here.
    """
    evidence = {
        "branch": "periodicity",
        "scope": scope,
        "flags": [],
        "series_length": len(ts),
    }

    if scope == "global":
        fft_result = fft_period(ts)
        acf_result = autocorr(ts, max_lag=min(50, len(ts) // 2 - 1))

        fft_p = fft_result["dominant_period"]
        acf_p = float(acf_result["first_peak_lag"]) if acf_result["first_peak_lag"] is not None else None

        if fft_p is not None and acf_p is not None:
            agree = abs(fft_p - acf_p) / max(fft_p, acf_p) < 0.20
            period_reconciled = acf_p if agree else fft_p
        else:
            period_reconciled = acf_p if acf_p is not None else fft_p

        # STL decomposition using reconciled period
        stl_period = int(round(period_reconciled)) if period_reconciled else None
        try:
            stl = seasonal_decompose(ts, period=stl_period)
            seasonal_strength = stl["seasonal_strength"]
            trend_strength    = stl["trend_strength"]
            residual_std      = stl["residual_std"]
            stl_period_used   = stl["period_used"]
        except Exception:
            seasonal_strength = None
            trend_strength    = None
            residual_std      = None
            stl_period_used   = None

        # decomposition type (additive vs multiplicative)
        try:
            dt = decomp_type(ts, period=stl_period_used or stl_period)
            decomp_fields = {
                "decomp_type":          dt["decomp_type"],
                "decomp_confidence":    dt["confidence"],
                "amplitude_trend_corr": dt["amplitude_trend_corr"],
            }
        except Exception:
            decomp_fields = {
                "decomp_type":          None,
                "decomp_confidence":    None,
                "amplitude_trend_corr": None,
            }

        evidence.update({
            "dominant_period_fft":  fft_p,
            "dominant_freq":        fft_result["dominant_freq"],
            "spectral_entropy":     fft_result["spectral_entropy"],
            "spectrum_top3":        fft_result["spectrum_top3"],
            "acf_peak_lag":         acf_result["first_peak_lag"],
            "acf_peak_strength":    acf_result["peak_strength"],
            "period_reconciled":    period_reconciled,
            "stl_period_used":      stl_period_used,
            "seasonal_strength":    seasonal_strength,
            "trend_strength":       trend_strength,
            "residual_std":         residual_std,
        })
        evidence.update(decomp_fields)

        # amplitude envelope — answers "does amplitude increase/decrease/remain same?"
        try:
            henv = hilbert_env(ts)
            evidence.update({
                "amplitude_change": henv["amplitude_change"],
                "amplitude_ratio":  henv["amplitude_ratio"],
            })
        except Exception:
            evidence.update({"amplitude_change": None, "amplitude_ratio": None})

        # shape features — waveform hint for waveform_type subtype questions
        try:
            sf = shape_features(ts)
            evidence.update({
                "waveform_hint":   sf["waveform_hint"],
                "peak_regularity": sf["peak_regularity"],
                "rise_fall_ratio": sf["rise_fall_ratio"],
            })
        except Exception:
            evidence.update({
                "waveform_hint":   None,
                "peak_regularity": None,
                "rise_fall_ratio": None,
            })

    else:  # local
        chunks_result = chunker(ts, n_chunks=2)
        chunk_ffts    = [fft_period(c) for c in chunks_result["chunks"]]

        chunk_seasonal_strengths = []
        for chunk, cf in zip(chunks_result["chunks"], chunk_ffts):
            try:
                p = int(round(cf["dominant_period"])) if cf["dominant_period"] else None
                stl = seasonal_decompose(chunk, period=p)
                chunk_seasonal_strengths.append(stl["seasonal_strength"])
            except Exception:
                chunk_seasonal_strengths.append(None)

        evidence.update({
            "chunk_edges":              chunks_result["chunk_edges"],
            "chunk_periods":            [cf["dominant_period"] for cf in chunk_ffts],
            "chunk_freqs":              [cf["dominant_freq"]   for cf in chunk_ffts],
            "chunk_seasonal_strengths": chunk_seasonal_strengths,
        })

    return evidence
