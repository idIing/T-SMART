import numpy as np
from ..tools import autocorr, ljung_box, stationarity, rolling_std, chunker, pacf_analysis, hilbert_env, check_arch_effects


def run(ts: np.ndarray, scope: str = "global") -> dict:
    """
    Noise branch: characterises randomness and stationarity of a single series.

    Tool chain: autocorr → pacf_analysis → ljung_box → stationarity
                → rolling_std → hilbert_env
                + stationarity on differenced/detrended series
                + per-chunk stationarity when scope == "local"

    Args:
        ts:    1-D time series array
        scope: "global" — full-series analysis (default)
               "local"  — adds per-chunk stationarity for "is any part
                          stationary?" and "which segment is stationary?"
                          questions

    Returns evidence dict with keys:
        branch, scope, flags, series_length
        acf_at_lag1               — ACF value at lag 1
        acf_peak_lag, acf_peak_strength
        pacf_at_lag1              — PACF at lag 1; key AR vs MA discriminator
        process_type_hint         — "AR(1)"|"MA(1)"|"ARMA"|"white_noise"|...
        is_white_noise, lb_pval_min
        variance                  — np.var(ts)
        adf_pval, kpss_pval, is_stationary
        is_stationary_after_diff
        is_stationary_after_detrend
        amplitude_change          — "increase"|"decrease"|"remain the same"
        amplitude_ratio           — second_half / first_half envelope mean
        variance_trend, variance_ratio
        Local only:
        chunk_edges               — segment boundaries
        chunk_is_stationary       — bool per chunk
        chunk_variances           — variance per chunk
        any_chunk_stationary      — bool: True if any chunk is stationary

    Flags populated by verifier (checks.py), not here.
    """
    evidence = {
        "branch": "noise",
        "scope": scope,
        "flags": [],
        "series_length": len(ts),
    }

    ac   = autocorr(ts, max_lag=min(40, len(ts) // 2 - 1))
    pac  = pacf_analysis(ts, max_lag=min(20, len(ts) // 2 - 1))
    lb   = ljung_box(ts, lags=min(20, len(ts) // 4))
    sta  = stationarity(ts)
    rstd = rolling_std(ts)
    henv = hilbert_env(ts)
    arch = check_arch_effects(ts)

    # lag-1 ACF
    acf_vals    = ac["acf_values"]
    acf_at_lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else None

    # explicit variance
    variance = float(np.var(ts.astype(float)))

    # stationarity after first differencing
    try:
        ts_diff              = np.diff(ts.astype(float))
        sta_diff             = stationarity(ts_diff)
        is_stationary_after_diff = sta_diff["is_stationary"]
    except Exception:
        is_stationary_after_diff = None

    # stationarity after linear detrend
    try:
        from scipy.signal import detrend as sp_detrend
        ts_detrended             = sp_detrend(ts.astype(float), type="linear")
        sta_detrend              = stationarity(ts_detrended)
        is_stationary_after_detrend = sta_detrend["is_stationary"]
    except Exception:
        is_stationary_after_detrend = None


    evidence.update({
        # autocorr
        "acf_at_lag1":        acf_at_lag1,
        "acf_peak_lag":       ac["first_peak_lag"],
        "acf_peak_strength":  ac["peak_strength"],
        # pacf + process type
        "pacf_at_lag1":       pac["pacf_at_lag1"],
        "pacf_at_lag2":       pac["pacf_at_lag2"],
        "process_type_hint":  pac["process_type_hint"],
        "pacf_confidence":    pac["confidence"],
        # ljung-box
        "is_white_noise":     lb["is_white_noise"],
        "lb_pval_min":        float(min(lb["lb_pval"])),
        # variance
        "variance":           variance,
        # stationarity
        "adf_pval":           sta["adf_pval"],
        "kpss_pval":          sta["kpss_pval"],
        "is_stationary":      sta["is_stationary"],
        "is_stationary_after_diff":    is_stationary_after_diff,
        "is_stationary_after_detrend": is_stationary_after_detrend,
        # amplitude envelope
        "amplitude_change":   henv["amplitude_change"],
        "amplitude_ratio":    henv["amplitude_ratio"],
        # rolling variance
        "variance_trend":     rstd["variance_trend"],
        "variance_ratio":     rstd["variance_ratio"],
        "arch_effect_detected": arch["arch_effect_detected"],
        "arch_pvalue": arch["pvalue"],
        "arch_lm_stat": arch["lm_stat"],
    })

    if arch.get("error"):
        evidence["arch_error"] = arch["error"]


    # --- local scope: per-chunk stationarity ---
    if scope == "local":
        chunks_result = chunker(ts, n_chunks=3)
        chunk_stats   = []
        chunk_vars    = []
        for c in chunks_result["chunks"]:
            try:
                s = stationarity(c)
                chunk_stats.append(s["is_stationary"])
            except Exception:
                chunk_stats.append(None)
            chunk_vars.append(float(np.var(c.astype(float))))

        evidence.update({
            "chunk_edges":          chunks_result["chunk_edges"],
            "chunk_is_stationary":  chunk_stats,
            "chunk_variances":      chunk_vars,
            "any_chunk_stationary": any(s is True for s in chunk_stats),
        })

    return evidence
