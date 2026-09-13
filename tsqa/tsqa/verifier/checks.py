import numpy as np


def verify(evidence: dict) -> dict:
    """
    Apply rule-based sanity checks to an evidence dict and return an updated
    copy with the flags list populated.

    Rules applied:
        "low_r2"                — trend branch, r2 < 0.65 (global) or any chunk r2 < 0.65 (local)
        "fft_unreliable"        — dominant_period_fft > series_length / 2
        "adf_kpss_disagree"     — ADF says stationary but KPSS says non-stationary (or vice versa)
        "granger_not_significant" — causality branch with direction == "none"
        "weak_correlation"      — causality branch with |best_corr| < 0.2
        "evidence_incomplete"   — any required field for the branch is None
        "arch_effects"          — ARCH LM test indicates heteroskedasticity

    Args:
        evidence: dict returned by a branch's run()

    Returns:
        Updated evidence dict with flags list. Does NOT mutate the input.
    """
    flags = list(evidence.get("flags", []))
    branch = evidence.get("branch")
    scope = evidence.get("scope", "global")
    n = evidence.get("series_length")

    def _add(flag):
        if flag not in flags:
            flags.append(flag)

    # --- low_r2 (trend branch) ---
    if branch == "trend":
        if scope == "global":
            r2 = evidence.get("r2")
            if r2 is not None and r2 < 0.65:
                _add("low_r2")
        else:
            r2s = evidence.get("r2s", [])
            if any(r < 0.65 for r in r2s if r is not None):
                _add("low_r2")

    # --- fft_unreliable ---
    dominant_period = evidence.get("dominant_period_fft")
    if dominant_period is not None and n is not None and dominant_period > n / 2:
        _add("fft_unreliable")

    # local periodicity: check per-chunk periods
    if branch == "periodicity" and scope == "local":
        chunk_edges = evidence.get("chunk_edges", [])
        chunk_periods = evidence.get("chunk_periods", [])
        for (start, end), period in zip(chunk_edges, chunk_periods):
            chunk_len = end - start
            if period is not None and period > chunk_len / 2:
                _add("fft_unreliable")
                break

    # --- adf_kpss_disagree (noise branch) ---
    adf_pval = evidence.get("adf_pval")
    kpss_pval = evidence.get("kpss_pval")
    if adf_pval is not None and kpss_pval is not None:
        adf_says_stationary = adf_pval < 0.10
        kpss_says_stationary = kpss_pval >= 0.05
        if adf_says_stationary != kpss_says_stationary:
            _add("adf_kpss_disagree")

    # --- granger_not_significant (causality branch) ---
    if branch == "causality":
        if evidence.get("direction") == "none":
            _add("granger_not_significant")

    # --- weak_correlation (causality branch) ---
    best_corr = evidence.get("best_corr")
    if best_corr is not None and abs(best_corr) < 0.2:
        _add("weak_correlation")

    # --- arch_effects ---
    if evidence.get("arch_effect_detected") is True:
        _add("arch_effects")

    # --- high_volatility ---
    cv = _max_coefficient_of_variation(evidence)
    if cv is not None and cv > 1.5:
        _add("high_volatility")

    # --- evidence_incomplete ---
    required_by_branch = {
        "trend": (
            ["slope", "direction", "r2"]
            if scope == "global"
            else ["chunk_edges", "slopes", "directions", "r2s"]
        ),
        "periodicity": (
            ["dominant_period_fft", "dominant_freq", "period_reconciled"]
            if scope == "global"
            else ["chunk_edges", "chunk_periods"]
        ),
        "anomaly": (
            ["outlier_indices", "outlier_count", "max_deviation"]
            if scope == "global"
            else ["chunk_edges", "chunk_outlier_counts"]
        ),
        "noise": ["adf_pval", "kpss_pval", "is_stationary", "is_white_noise"],
        "similarity": ["amp_stats_1", "amp_stats_2", "trend_1", "trend_2"],
        "causality": ["best_lag", "pval_12", "pval_21", "direction", "best_corr"],
    }
    required = required_by_branch.get(branch, [])
    if any(evidence.get(f) is None for f in required):
        _add("evidence_incomplete")

    updated = dict(evidence)
    updated["flags"] = flags
    return updated


def validate_schema_compliance(answer, expected_schema: str) -> bool:
    """Does the emitted answer match the SHAPE the row demands?

    Used by the numeric head's bounded schema-compliance gate: a numeric row
    must emit a finite real number, a categorical row a non-empty label, an MCQ
    row an A/B/C/D letter. On a mismatch the caller does at most ONE format-only
    re-prompt and then accepts the deterministic value — never an unbounded
    "regenerate until right" loop (that would break temp-0 paired determinism
    and the Phase-0 latency win).
    """
    if expected_schema == "numerical":
        if isinstance(answer, bool):
            return False  # bools are ints in Python; a yes/no is not a number
        if isinstance(answer, (int, float)):
            return bool(np.isfinite(answer))
        return False
    if expected_schema == "categorical":
        return isinstance(answer, str) and bool(answer.strip())
    if expected_schema == "mcq":
        return isinstance(answer, str) and answer.strip().upper() in {"A", "B", "C", "D"}
    return False


_EXPLICIT_FLAGS = {
    "low_r2",
    "fft_unreliable",
    "adf_kpss_disagree",
    "granger_not_significant",
    "weak_correlation",
    "evidence_incomplete",
    "high_volatility",
    "arch_effects",
}

_VISUAL_KEYWORDS = {
    "shape",
    "visual",
    "visualize",
    "plot",
    "spectrogram",
    "regime shift",
    "topology",
    "pattern",
    "waveform",
    "morphology",
    "spectrum",
    "frequency",
    # amplitude envelope questions are inherently visual
    "amplitude",
    "envelope",
}

_SPECTROGRAM_KEYWORDS = {
    "spectrogram",
    "spectrum",
    "frequency",
    "time-frequency",
    "stft",
    "cwt",
    "period",
    "seasonal",
    "heteroskedastic",
    "regime",
}

_SHAPE_KEYWORDS = {
    "shape",
    "trend",
    "morphology",
    "waveform",
    "line",
    "plot",
}

_SEASONAL_KEYWORDS = {"seasonal", "period", "cycle", "frequency"}
_ANOMALY_KEYWORDS = {"anomaly", "outlier", "spike", "jump", "break"}


def evaluate_visual_trigger(evidence_dict: dict, query_semantics) -> tuple:
    """
    Decide whether to invoke a vision artifact and select the viz_type.

    Returns:
        (triggered: bool, viz_type: str | None, meta: dict)
    """
    evidence = evidence_dict or {}
    flags = set(evidence.get("flags", []))
    flags_present = flags & _EXPLICIT_FLAGS
    flags_count = len(flags_present)

    query_text = _extract_query_text(query_semantics)
    branch = (
        query_semantics.get("branch") if isinstance(query_semantics, dict) else None
    )
    subtype = (
        query_semantics.get("subtype") if isinstance(query_semantics, dict) else None
    )
    explicit_visual = _contains_any(query_text, _VISUAL_KEYWORDS)
    spectro_hint = _contains_any(query_text, _SPECTROGRAM_KEYWORDS)
    shape_hint = _contains_any(query_text, _SHAPE_KEYWORDS)
    season_hint = _contains_any(query_text, _SEASONAL_KEYWORDS)
    anomaly_hint = _contains_any(query_text, _ANOMALY_KEYWORDS)

    reasons = []
    score = 1.0 - 0.15 * flags_count
    if "evidence_incomplete" in flags_present:
        score -= 0.25
    for flag in sorted(flags_present):
        reasons.append(flag)

    r2_values = _collect_r2_values(evidence)
    best_r2, second_r2 = _best_and_second(r2_values)
    if best_r2 is not None and best_r2 < 0.65:
        score -= 0.2
        reasons.append("low_fit")
    if best_r2 is not None and second_r2 is not None and (best_r2 - second_r2) < 0.05:
        score -= 0.15
        reasons.append("fit_ambiguous")

    pval_12 = _safe_float(evidence.get("pval_12"), default=1.0)
    pval_21 = _safe_float(evidence.get("pval_21"), default=1.0)
    if _is_marginal_pval(pval_12) or _is_marginal_pval(pval_21):
        score -= 0.15
        reasons.append("pval_marginal")

    if evidence.get("arch_effect_detected") is True:
        score -= 0.2
        reasons.append("arch_effects")

    adf_pval = _safe_float(evidence.get("adf_pval"), default=None)
    if adf_pval is not None and adf_pval > 0.10:
        score -= 0.15
        reasons.append("adf_high_pval")

    best_corr = _safe_float(evidence.get("best_corr"), default=None)
    if best_corr is not None and abs(best_corr) >= 0.3 and max(pval_12, pval_21) >= 0.1:
        score -= 0.15
        reasons.append("corr_granger_conflict")

    if _has_chunk_variance(evidence):
        score -= 0.15
        reasons.append("chunk_variance")

    seasonal_strength = _safe_float(evidence.get("seasonal_strength"), default=None)
    trend_strength = _safe_float(evidence.get("trend_strength"), default=None)
    max_deviation = _safe_float(evidence.get("max_deviation"), default=None)

    if season_hint and seasonal_strength is not None and seasonal_strength < 0.2:
        score -= 0.1
        reasons.append("weak_seasonal_signal")
    if season_hint and trend_strength is not None and trend_strength < 0.2:
        score -= 0.05
        reasons.append("weak_trend_signal")
    if anomaly_hint and max_deviation is not None and max_deviation < 2.0:
        score -= 0.1
        reasons.append("weak_anomaly_signal")

    score = float(max(0.0, min(1.0, score)))

    branch_failure = evidence.get("branch") in (None, "", "unknown")
    direct_trigger = flags_count >= 2 or (
        "evidence_incomplete" in flags_present and branch_failure
    )
    arch_trigger = evidence.get("arch_effect_detected") is True and (
        branch == "noise" or subtype == "variance_level"
    )
    if arch_trigger:
        direct_trigger = True

    high_confidence = _is_high_confidence(
        evidence,
        flags_count,
        best_r2,
        second_r2,
        pval_12,
        pval_21,
        best_corr,
    )

    if high_confidence and not explicit_visual:
        triggered = False
    else:
        triggered = direct_trigger or score <= 0.85 or explicit_visual

    if not triggered:
        return (
            False,
            None,
            {
                "triggered": False,
                "viz_type": None,
                "confidence": score,
                "flags_count": flags_count,
                "reasons": reasons,
                "explicit_visual": explicit_visual,
            },
        )

    viz_type = _select_viz_type(
        evidence, reasons, flags_present, spectro_hint, shape_hint, subtype=subtype
    )
    return (
        True,
        viz_type,
        {
            "triggered": True,
            "viz_type": viz_type,
            "confidence": score,
            "flags_count": flags_count,
            "reasons": reasons,
            "explicit_visual": explicit_visual,
        },
    )


def _extract_query_text(query_semantics) -> str:
    if query_semantics is None:
        return ""
    if isinstance(query_semantics, dict):
        parts = []
        for key in ("question", "text", "query", "prompt"):
            val = query_semantics.get(key)
            if val:
                parts.append(str(val))
        return " ".join(parts).lower()
    return str(query_semantics).lower()


def _contains_any(text: str, keywords: set) -> bool:
    if not text:
        return False
    return any(k in text for k in keywords)


def _collect_r2_values(evidence: dict) -> list:
    r2_values = []
    for key in ("r2", "exp_r2", "log_r2"):
        val = _safe_float(evidence.get(key), default=None)
        if val is not None:
            r2_values.append(val)

    r2s = evidence.get("r2s")
    if isinstance(r2s, list):
        for val in r2s:
            v = _safe_float(val, default=None)
            if v is not None:
                r2_values.append(v)

    for trend_key in ("trend_1", "trend_2"):
        trend_obj = evidence.get(trend_key)
        if isinstance(trend_obj, dict):
            v = _safe_float(trend_obj.get("r2"), default=None)
            if v is not None:
                r2_values.append(v)

    return r2_values


def _best_and_second(values: list) -> tuple:
    if not values:
        return None, None
    sorted_vals = sorted(values, reverse=True)
    best = sorted_vals[0]
    second = sorted_vals[1] if len(sorted_vals) > 1 else None
    return best, second


def _is_marginal_pval(pval) -> bool:
    if pval is None:
        return False
    return 0.04 <= pval <= 0.10


def _has_chunk_variance(evidence: dict) -> bool:
    candidates = []
    for key in (
        "chunk_outlier_counts",
        "chunk_max_deviations",
        "chunk_periods",
        "chunk_freqs",
        "chunk_seasonal_strengths",
        "segment_means",
        "segment_stds",
    ):
        vals = evidence.get(key)
        if isinstance(vals, list) and _high_variance(vals):
            candidates.append(key)

    for key in ("chunk_amp_1", "chunk_amp_2"):
        chunk_stats = evidence.get(key)
        if isinstance(chunk_stats, list):
            means = [cs.get("mean") for cs in chunk_stats if isinstance(cs, dict)]
            stds = [cs.get("std") for cs in chunk_stats if isinstance(cs, dict)]
            if _high_variance(means) or _high_variance(stds):
                candidates.append(key)

    variance_ratio = _safe_float(evidence.get("variance_ratio"), default=None)
    if variance_ratio is not None and (variance_ratio >= 1.5 or variance_ratio <= 0.67):
        candidates.append("variance_ratio")

    return len(candidates) > 0


def _high_variance(values: list) -> bool:
    vals = [v for v in values if v is not None]
    if len(vals) < 3:
        return False
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean == 0:
        return std > 0
    return abs(std / mean) > 0.5


def _select_viz_type(
    evidence: dict, reasons: list, flags: set, spectro_hint: bool, shape_hint: bool,
    subtype: str | None = None,
) -> str:
    # Subtypes where the line plot is strictly better than a spectrogram:
    # - waveform_type: need to see wave shape (sine/square/sawtooth)
    # - amplitude_change: envelope change is obvious on line plot, invisible on spectrogram
    # - period_change: cycle compression/expansion is visible on line plot
    _LINE_PLOT_SUBTYPES = {"waveform_type", "amplitude_change", "period_change"}
    if subtype in _LINE_PLOT_SUBTYPES:
        return "line_plot"

    spectro_reasons = {
        "fft_unreliable",
        "adf_kpss_disagree",
        "adf_high_pval",
        "pval_marginal",
        "corr_granger_conflict",
        "chunk_variance",
        "weak_seasonal_signal",
        "high_volatility",
        "arch_effects",
    }
    line_reasons = {
        "low_r2",
        "weak_correlation",
        "fit_ambiguous",
        "low_fit",
    }

    if spectro_hint:
        return "spectrogram"
    if any(r in spectro_reasons for r in reasons) or any(
        f in spectro_reasons for f in flags
    ):
        return "spectrogram"
    if (
        shape_hint
        or any(r in line_reasons for r in reasons)
        or any(f in line_reasons for f in flags)
    ):
        return "line_plot"

    branch = evidence.get("branch")
    if branch in {"periodicity", "noise", "causality"}:
        return "spectrogram"
    return "line_plot"


def _is_high_confidence(
    evidence: dict, flags_count: int, best_r2, second_r2, pval_12, pval_21, best_corr
) -> bool:
    if flags_count > 0:
        return False

    if best_r2 is not None:
        margin_ok = second_r2 is None or (best_r2 - second_r2) >= 0.1
        if best_r2 >= 0.75 and margin_ok:
            return True

    if best_corr is not None and abs(best_corr) >= 0.3:
        if min(pval_12, pval_21) <= 0.01:
            return True

    seasonal_strength = _safe_float(evidence.get("seasonal_strength"), default=None)
    trend_strength = _safe_float(evidence.get("trend_strength"), default=None)
    if seasonal_strength is not None and seasonal_strength >= 0.6:
        return True
    if trend_strength is not None and trend_strength >= 0.6:
        return True

    return False


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_coefficient_of_variation(evidence: dict):
    pairs = []
    for key in ("amp_context", "amp_stats", "amp_stats_1", "amp_stats_2"):
        data = evidence.get(key)
        if isinstance(data, dict):
            pairs.append((data.get("mean"), data.get("std")))

    for value in evidence.values():
        if isinstance(value, dict) and "mean" in value and "std" in value:
            pairs.append((value.get("mean"), value.get("std")))

    if not pairs:
        return None

    max_cv = None
    for mean, std in pairs:
        m = _safe_float(mean, default=None)
        s = _safe_float(std, default=None)
        if m is None or s is None:
            continue
        denom = abs(m)
        if denom < 1e-8:
            cv = float("inf") if s > 0 else 0.0
        else:
            cv = abs(s / denom)
        if max_cv is None or cv > max_cv:
            max_cv = cv

    return max_cv
