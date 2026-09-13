import json

# ---------------------------------------------------------------------------
# SYSTEM prompt — fixed, sent once per session if the API supports it
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert time series analyst. You will be given a multiple-choice \
question about one or two time series, along with a structured evidence object \
produced by a deterministic analysis pipeline. Your job is to select the single \
best answer from the provided options using the evidence.

Rules:
- Base your answer ONLY on the provided evidence. Do not guess or hallucinate values.
- Output your answer as a single capital letter matching one provided option on the first line.
- Follow it with exactly one sentence of justification referencing specific evidence fields.
- If the evidence flags indicate uncertainty, reason carefully and acknowledge it.
"""

# ---------------------------------------------------------------------------
# llm_only baseline — the no-architecture control arm
# ---------------------------------------------------------------------------
# Asks the answer LLM the multiple-choice question DIRECTLY: same question +
# options, NO router, NO branch tools, NO vision, NO evidence — a single call.
# This is the deliberate "raw model" ablation used to isolate the architecture's
# contribution at a fixed backbone (Track-A de-confound). It is NOT required to be
# byte-identical to anything. The prompt is letter-neutral: it does not coach,
# rank, or hint at any option (no prompt-letter nudging norm).

LLM_ONLY_SYSTEM_PROMPT = """\
You are an expert time series analyst answering a multiple-choice question \
about one or two time series. Reason carefully from first principles and pick \
the single best answer from the provided options.\
"""

_LLM_ONLY_TEMPLATE = """\
## Question
{question}

## Answer choices
{options_block}

## Task
Choose the single best answer. Reply with the option letter on the first line, \
then one sentence of justification.\
"""


def build_llm_only_prompt(question: str, options: list) -> tuple:
    """Build the no-architecture (raw-model) MCQ prompt.

    Returns ``(system_prompt, user_prompt)``. The output format matches the rest
    of the pipeline (letter on line 1) so the shared ``parse_answer`` extracts it.
    No evidence, no series, no per-option steering.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_block = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options[: len(letters)])
    )
    user_prompt = _LLM_ONLY_TEMPLATE.format(
        question=question, options_block=options_block
    )
    return LLM_ONLY_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# llm_numeric COUNTERFACTUAL — the no-architecture control for FREE-RESPONSE rows
# ---------------------------------------------------------------------------
# Sibling of the llm_only arm above, for numerical (free-response) questions. It
# hands the answer LLM the RAW SERIES VALUES and the question and asks it to
# COMPUTE the quantity directly — NO router, NO tools, NO evidence, NO options,
# NO precomputed value. Paired against the deterministic numeric head on the SAME
# rows, the Accuracy@10% gap measures whether the tool is load-bearing on a strong
# backbone (mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md). This
# deliberately asks the model to compute (unlike the SHIPPED numeric-head prompts
# in prompt_numeric.py, which only RESTATE a tool-owned value); there are no
# options, so the no-letter-nudging norm is not engaged.

LLM_NUMERIC_SYSTEM_PROMPT = """\
You are a precise time-series calculator. You are given the raw values of one or \
more time series and a question asking for a single numeric quantity (for example \
a standard deviation, a percentile value, an index, or a count). Compute the \
answer directly and exactly from the values provided. Reply with the numeric \
answer ALONE on the first line — digits only, no units, no words, no option letters.\
"""

_LLM_NUMERIC_TEMPLATE = """\
## Question
{question}

## Time series data
{series_block}

## Task
Compute the numeric answer to the question from the values above. Put the number \
alone on the first line (no units or words). Optionally add one short sentence on \
a second line.\
"""


def build_llm_numeric_prompt(question, ts=None, ts1=None, ts2=None,
                             max_points: int = 4096) -> tuple:
    """Build the free-response COUNTERFACTUAL prompt: ask the model to COMPUTE the
    numeric answer directly from the raw series (no tool / options / evidence).

    The deterministic numeric head is bypassed for this arm; a paired Accuracy@10%
    comparison against the head measures whether the tool is load-bearing on a
    strong backbone. Output format (number on line 1) matches ``parse_numeric_answer``.
    Series are rendered as comma-separated values; a series longer than
    ``max_points`` is uniformly subsampled (noted in-prompt) to bound prompt size —
    in practice no MMTS-Base row triggers this, so the model sees the full data the
    tool sees (a FAIR counterfactual).

    Returns ``(LLM_NUMERIC_SYSTEM_PROMPT, user_prompt)``.
    """
    if ts1 is not None or ts2 is not None:
        items = []
        if ts1 is not None:
            items.append(("Time Series 1", ts1))
        if ts2 is not None:
            items.append(("Time Series 2", ts2))
    else:
        items = [("Time Series", ts)] if ts is not None else []

    blocks = [_render_series_block(name, s, max_points) for name, s in items]
    series_block = "\n\n".join(blocks) if blocks else "(no series provided)"
    user_prompt = _LLM_NUMERIC_TEMPLATE.format(question=question, series_block=series_block)
    return LLM_NUMERIC_SYSTEM_PROMPT, user_prompt


def _render_series_block(name: str, series, max_points: int) -> str:
    """Render one series as 'name (N points): v1, v2, ...', uniformly subsampling
    (endpoints preserved) only when longer than max_points, with a note."""
    try:
        vals = list(series)
    except TypeError:
        return f"{name}: (unavailable)"
    n = len(vals)
    note = ""
    if n > max_points > 1:
        idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
        vals = [vals[j] for j in idx]
        note = f", uniformly subsampled to {len(vals)} of {n} points"
    body = ", ".join(_fmt_series_value(v) for v in vals)
    return f"{name} ({n} points{note}):\n{body}"


def _fmt_series_value(v) -> str:
    """Compact 6-significant-figure rendering; integer-valued floats stay clean."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


_COGNITIVE_GUIDELINES = {
    ("similarity", "statistical"): (
        "- Autocorrelation & Noise Similarity: To determine if two series share the same noise type, inspect autocorr_similarity and acf_similarity_corr. If autocorr_similarity is \"high\" or acf_similarity_corr > 0.8, they likely share the same noise type. If they differ significantly (e.g. one has very low ACF values representing white noise, while the other has high ACF values or strong peaks indicating red/autoregressive noise), they have different noise types."
    ),
    ("periodicity", "period_value"): (
        "- Cycle Detection: If the seasonal_strength in the mathematical evidence is extremely low (e.g., < 0.25) or spectral_entropy is very high (e.g., > 0.85), there is no cyclical/periodic pattern at all, regardless of the waveform_hint value."
    ),
    ("similarity", "lag_offset"): (
        "- Lag Direction Mapping:\n"
        "  * If best_lag > 0 (meaning ts1 leads ts2 in time): the correct answer is 'No, time series 1 is a lagged version of time series 2'.\n"
        "  * If best_lag < 0 (meaning ts2 leads ts1 in time): the correct answer is 'Yes' (meaning time series 2 is a lagged version of time series 1).\n"
        "  * If best_lag is 0 or the correlation is extremely weak, they do not share a lagged relationship."
    ),
    ("anomaly", "pattern_flip"): (
        "- Abrupt Frequency Change: An abrupt frequency change is best identified via the spectrogram (from the vision sensor) showing different horizontal frequency ridges or distinct spectral shifts across the timeline, even if the time-domain statistics show a standard deviation or variance change."
    ),
    ("trend", "functional_form"): (
        "- Distortion by Anomalies & VLM Priority: If the mathematical fit has low R^2 (indicated by a low_r2 flag or model fit alerts), the mathematical branch fits (like best_fit_type or waveform_hint) are highly distorted and unreliable due to structural breaks. In this case, you MUST ignore the mathematical fits and heavily weight the structured Visual/Topological Evidence (e.g. 'upward-curving' indicating an exponential trend, or 'vertical bars' indicating a square wave) to make your decision."
    ),
    ("trend", "change_point"): (
        "- Segment counting: use n_segments directly (= n_breakpoints + 1, pre-computed). "
        "Do NOT add 1 yourself if n_segments is present.\n"
        "- Trend ordering: PREFER segment_best_fit_types (one entry per actual segment, aligned to ruptures breakpoints) "
        "over best_fit_types (fixed 4-chunk split that may not align). "
        "Example: segment_best_fit_types=['exponential','linear','log'] → 'Exponential → Linear → Log'.\n"
        "- First-half vs second-half: compare segment_directions[0] vs segment_directions[-1]. "
        "If they match, the answer is 'Same direction in both halves'.\n"
        "- Fall back to best_fit_types / slopes / directions (equal chunks) ONLY if segment_best_fit_types is empty or None."
    ),
    ("periodicity", "waveform_type"): (
        "- Vision priority: when the mathematical evidence is ambiguous (e.g. waveform_hint='unknown', "
        "amplitude_change='remain the same', or best_fit_types is None), STRONGLY prefer the vision sensor's "
        "answer (vision_letter) over the math evidence. The vision sensor directly observes waveform shape.\n"
        "- Amplitude vs shape: amplitude_change describes whether the wave HEIGHT changes over time "
        "(increase/decrease/remain the same). The waveform TYPE (sine/square/sawtooth/triangle) is a "
        "separate property — do not confuse them.\n"
        "- If amplitude_change='remain the same' and vision_letter is available, use vision_letter as your answer."
    )
}

# ---------------------------------------------------------------------------
# Answer-mapping prompt template
# Placeholders:  {question}  {options_block}  {evidence_json}  {flags_block}
# ---------------------------------------------------------------------------

_ANSWER_TEMPLATE = """\
## Question
{question}

## Answer choices
{options_block}

## Evidence (from deterministic analysis pipeline)
```json
{evidence_json}
```
{trend_block}
{artifacts_block}
{vision_block}
{flags_block}
## Task
{task_block}
"""

_FLAGS_SECTION = """\
## Analysis flags  ⚠️
The pipeline flagged the following issues with the evidence — reason carefully:
{flag_list}

"""

_NO_FLAGS_SECTION = ""


_CORE_KEYS = {
    "branch",
    "scope",
    "flags",
    "series_length",
    "series_length_1",
    "series_length_2",
    "vision_trigger",
    "vision_error",
    "vision_trigger_error",
    "best_fit_type",
    "ols_trend_summary",
    "dominant_period_fft",
    "dominant_freq",
    "seasonal_strength",
    "spectral_entropy",
    "amp_stats_1",
    "amp_stats_2",
    "amp_context",
    "variance",
    "is_stationary",
    "is_white_noise",
}

_SUBTYPE_KEYS = {
    ("trend", "direction"): {
        "slope",
        "direction",
        "r2",
        "best_fit_type",
        "ols_trend_summary",
    },
    ("trend", "functional_form"): {
        "slope",
        "direction",
        "r2",
        "exp_r2",
        "log_r2",
        "best_fit_type",
        "ols_trend_summary",
        "waveform_hint",
        "rise_fall_ratio",
        "pattern_type",
    },
    ("trend", "change_point"): {
        # breakpoint summary
        "n_breakpoints",
        "n_segments",                   # = n_breakpoints + 1, pre-computed
        "breakpoint_positions",
        "largest_mean_shift",
        "largest_std_shift",
        # per-ACTUAL-segment trend detail (aligned to ruptures breakpoints)
        "segment_best_fit_types",
        "segment_slopes",
        "segment_directions",
        "segment_r2s",
        # per-equal-chunk detail (fallback / slope_acceleration)
        "chunk_edges",
        "slopes",
        "directions",
        "best_fit_types",
        "r2s",
        "slope_acceleration",
        # global fit as fallback context
        "best_fit_type",
        "direction",
        "r2",
    },
    ("periodicity", "period_value"): {
        "dominant_period_fft",
        "dominant_freq",
        "period_reconciled",
        "stl_period_used",
        "seasonal_strength",
        "waveform_hint",
        "rise_fall_ratio",
        "spectral_entropy",
    },
    ("periodicity", "period_change"): {
        "chunk_edges",
        "chunk_periods",
        "chunk_freqs",
        "chunk_seasonal_strengths",
        "period_reconciled",
        "dominant_period_fft",
        "amplitude_change",
        "amplitude_ratio",
    },
    ("periodicity", "amplitude_change"): {
        "amplitude_change",
        "amplitude_ratio",
        "envelope_mean",
        "envelope_trend",
        "envelope_slope",
        "envelope_r2",
        "first_half_mean",
        "second_half_mean",
        "chunk_edges",
        "chunk_seasonal_strengths",
    },
    ("periodicity", "waveform_type"): {
        "waveform_hint",
        "peak_regularity",
        "rise_fall_ratio",
        "seasonal_strength",
        "decomp_type",
        # amplitude trend fields — many "cycle pattern" questions have amplitude answers
        "amplitude_change",
        "amplitude_ratio",
        # period trend fields — some "cycle pattern" questions ask about period change
        "chunk_periods",
        "chunk_freqs",
        "period_reconciled",
    },
    ("anomaly", "point_anomaly"): {
        "outlier_indices",
        "outlier_count",
        "max_deviation",
        "outlier_positions",
    },
    ("anomaly", "pattern_flip"): {
        "n_breakpoints",
        "breakpoint_positions",
        "largest_mean_shift",
        "largest_std_shift",
        "segment_means",
        "segment_stds",
        "chunk_edges",
        "chunk_outlier_counts",
    },
    ("noise", "white_noise"): {
        "is_white_noise",
        "lb_pval_min",
        "variance",
        "acf_at_lag1",
    },
    ("noise", "stationarity"): {
        "adf_pval",
        "kpss_pval",
        "is_stationary",
        "is_stationary_after_diff",
        "is_stationary_after_detrend",
    },
    ("noise", "variance_level"): {
        "variance",
        "variance_trend",
        "variance_ratio",
        "arch_effect_detected",
        "arch_pvalue",
        "arch_lm_stat",
    },
    ("noise", "autocorr_value"): {"acf_at_lag1", "acf_peak_lag", "acf_peak_strength"},
    ("noise", "ar_ma_process"): {
        "acf_at_lag1",
        "pacf_at_lag1",
        "pacf_at_lag2",
        "process_type_hint",
        "pacf_confidence",
        "arch_effect_detected",
    },
    ("similarity", "statistical"): {
        "amp_stats_1",
        "amp_stats_2",
        "mean_diff",
        "range_ratio",
        "std_ratio",
        "variance_similarity",
        "higher_variance_series",
        "level_similarity",
        "range_similarity",
        "autocorr_similarity",
        "acf_similarity_corr",
        "acf_1",
        "acf_2",
    },
    ("similarity", "shape"): {
        "shape_similarity",
        "dtw_distance_normalized",
        "shape_similarity_label",
        "overall_similarity_summary",
        "waveform_hints_match",
        "shape_features_1",
        "shape_features_2",
    },
    ("similarity", "lag_offset"): {
        "direct_correlation",
        "flipped_correlation",
        "best_lag",
        "best_lag_corr",
        "shape_similarity",
        "dtw_distance_normalized",
    },
    ("causality", "lag_direction"): {
        "best_lag",
        "best_corr",
        "corr_at_zero",
        "granger_best_lag",
        "pval_12",
        "pval_21",
        "direction",
    },
    ("causality", "lag_value"): {
        "best_lag",
        "best_corr",
        "corr_at_zero",
        "granger_best_lag",
        "pval_12",
        "pval_21",
        "direction",
    },
}


def build_answer_prompt(
    question: str,
    options: list,
    evidence: dict,
    flags=None,
    branch: str = None,
    subtype: str = None,
) -> str:
    """
    Build the answer-mapping prompt for the LLM interpreter.

    Args:
        question: raw question string from the dataset
        options:  list of answer-choice strings, e.g. ["yes", "no", "maybe", ...]
        evidence: dict returned by a branch's run() after verify()
        flags:    list of flag strings from evidence["flags"]; if None, read from evidence

    Returns:
        Formatted prompt string ready to send to the LLM.
        SYSTEM_PROMPT should be sent separately as the system message.

    The options are labelled A, B, C... automatically. TimeSeriesExam1 uses
    2-4 options; TSRBench can use more.
    """
    if flags is None:
        flags = evidence.get("flags", [])

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_block = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options[:len(letters)])
    )

    # If the orchestrator passed a bundled evidence object, split math/vision
    if isinstance(evidence, dict) and "math_evidence" in evidence:
        math_evidence = evidence.get("math_evidence", {}) or {}
        vision_evidence = evidence.get("vision_evidence")
        vision_text = evidence.get("vision_text")
        vision_suggestion = evidence.get("vision_suggestion")
        # strip large arrays from math evidence before serialising
        _SKIP_KEYS = {"acf_values", "zscore", "iqr_outliers"}
        clean_math = {k: v for k, v in math_evidence.items() if k not in _SKIP_KEYS}

        # Apply Tasteful dynamic filtering to focus the reasoner context
        if branch and subtype and (branch, subtype) in _SUBTYPE_KEYS:
            allowed = _CORE_KEYS | _SUBTYPE_KEYS[(branch, subtype)]
            clean_math = {k: v for k, v in clean_math.items() if k in allowed}

        evidence_json = json.dumps(clean_math, indent=2, default=_json_default)
        # avoid exposing raw artifact paths to the reasoner
        artifacts_block = ""
        # build a minimal vision block using the vision evidence provided
        vision_block = _build_vision_block(
            {
                "vision_struct": vision_evidence,
                "vision_text": vision_text,
                "vision_suggestion": vision_suggestion,
            }
        )
    else:
        # strip large arrays from evidence before serialising to keep prompt concise
        _SKIP_KEYS = {"acf_values", "zscore", "iqr_outliers"}
        clean_evidence = {k: v for k, v in evidence.items() if k not in _SKIP_KEYS}

        if branch and subtype and (branch, subtype) in _SUBTYPE_KEYS:
            allowed = _CORE_KEYS | _SUBTYPE_KEYS[(branch, subtype)]
            clean_evidence = {k: v for k, v in clean_evidence.items() if k in allowed}

        # convert numpy types so json.dumps doesn't choke
        evidence_json = json.dumps(clean_evidence, indent=2, default=_json_default)

    if flags:
        flag_list = "\n".join(f"  - `{f}`" for f in flags)
        flags_block = _FLAGS_SECTION.format(flag_list=flag_list)
    else:
        flags_block = _NO_FLAGS_SECTION

    if isinstance(evidence, dict) and "math_evidence" in evidence:
        trend_block = _build_trend_block(evidence["math_evidence"])
    else:
        trend_block = _build_trend_block(evidence)
    # For the non-bundle path, build artifacts/vision blocks from raw evidence.
    # The bundle path already built vision_block above — do NOT overwrite it.
    if "math_evidence" not in (evidence or {}):
        artifacts_block = _build_artifacts_block(evidence)
        vision_block = _build_vision_block(evidence)

    task_guidelines = ""
    if branch and subtype and (branch, subtype) in _COGNITIVE_GUIDELINES:
        task_guidelines = _COGNITIVE_GUIDELINES[(branch, subtype)] + "\n\n"

    task_block = (
        f"When available, pay special attention to derived interpretation fields such as "
        f"pattern_type, pattern_regularity, overall_similarity_summary, trend_similarity, "
        f"level_similarity, range_similarity, variance_similarity, higher_variance_series, "
        f"std_ratio, scaled_version_detected, scaled_version_direction, "
        f"flipped_version_detected, direct_correlation, flipped_correlation, "
        f"autocorr_similarity, acf_similarity_corr, period_similarity, "
        f"higher_period_series, trend_family_match, trend_family_summary, "
        f"amplitude_comparison, distribution_similarity, and shape_similarity_label.\n\n"
        f"If a visual answer suggestion is provided, treat it as an independent "
        f"visual vote rather than as ground truth. For exact numeric period, lag, "
        f"or value questions, prioritize reliable statistical evidence. For shape, "
        f"waveform, morphology, regime-change, or visually ambiguous low-fit cases, "
        f"give a high-confidence visual suggestion substantial weight. If math and "
        f"vision disagree, choose the answer supported by the evidence source that "
        f"is most relevant to the question type.\n\n"
        f"{task_guidelines}"
        f"Select the best answer. Reply with the option letter on the first line, then one sentence of justification."
    )

    return _ANSWER_TEMPLATE.format(
        question=question,
        options_block=options_block,
        evidence_json=evidence_json,
        trend_block=trend_block,
        artifacts_block=artifacts_block,
        vision_block=vision_block,
        flags_block=flags_block,
        task_block=task_block,
    )


def _build_vision_block(evidence: dict) -> str:
    if not isinstance(evidence, dict):
        return ""

    vis_struct = evidence.get("vision_struct")
    vis_text = evidence.get("vision_text")
    vision_suggestion = evidence.get("vision_suggestion")
    lines = []
    if vis_struct:
        # include structured vision fields as JSON for the LLM to consume
        try:
            vis_json = json.dumps(vis_struct, indent=2, default=str)
            lines.append("## Visual / Topological Evidence")
            lines.append("```json")
            lines.append(vis_json)
            lines.append("```")
        except Exception:
            pass

    if vis_text and isinstance(vis_text, str):
        lines.append("Summary: " + vis_text)

    if isinstance(vision_suggestion, dict) and vision_suggestion.get("parse_success"):
        letter = vision_suggestion.get("letter")
        confidence = vision_suggestion.get("confidence")
        suggestion = {
            "letter": letter,
            "confidence": confidence,
            "raw": vision_suggestion.get("raw"),
        }
        lines.append("## Visual Answer Suggestion")
        lines.append("```json")
        lines.append(json.dumps(suggestion, indent=2, default=str))
        lines.append("```")
        if confidence == "high" and letter:
            lines.append(
                f"⚠️ HIGH CONFIDENCE VISUAL SIGNAL: The visual analysis strongly "
                f"suggests answer {letter}. You MUST select {letter} unless a specific "
                f"numeric value in the math evidence directly contradicts it. "
                f"Do NOT let generic fields like amplitude_change or waveform_hint "
                f"override a high-confidence visual answer."
            )

    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _json_default(obj):
    """Fallback serialiser for numpy scalars and arrays."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _build_artifacts_block(evidence: dict) -> str:
    if not isinstance(evidence, dict):
        return ""

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return ""

    image_artifacts = [
        a for a in artifacts if isinstance(a, dict) and a.get("kind") == "image"
    ]
    if not image_artifacts:
        return ""

    vision_meta = (
        evidence.get("vision_trigger", {})
        if isinstance(evidence.get("vision_trigger"), dict)
        else {}
    )
    viz_type = vision_meta.get("viz_type") or "unknown"

    alert = _build_vision_alert(vision_meta, viz_type)
    # Avoid passing detailed visual guidance to the reasoning LLM to reduce
    # visual sycophancy. Only expose the representation type.
    guidance = (
        f"Representation type: {viz_type}"
        if viz_type
        else "Representation type: unknown"
    )
    lines = []
    if alert:
        lines.extend(["## Vision alert", alert, ""])
    lines.extend(["## Visual artifacts", guidance, ""])

    for idx, item in enumerate(image_artifacts, start=1):
        caption = item.get("caption", "")
        usage = item.get("usage_instructions", "")
        path = item.get("path", "")
        entry = f"Image {idx}: {caption}".strip()
        if path:
            entry = f"{entry} (path: {path})"
        lines.append(entry)
        if usage:
            lines.append(f"Usage: {usage}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_trend_block(evidence: dict) -> str:
    if not isinstance(evidence, dict):
        return ""

    summary = evidence.get("ols_trend_summary")
    if not summary:
        return ""

    return f"## Trend summary\n{summary}\n"


def _visual_guidance(viz_type: str) -> str:
    if viz_type == "line_plot":
        return (
            "Guidance: Series are overlaid on one axis; color order is fixed to "
            "RGB: red (#FF0000) for series 1, green (#00FF00) for series 2, "
            "blue (#0000FF) for series 3, then repeats. Use the line plot to judge "
            "global morphology, trend direction, structural breaks, and phase alignment."
        )
    if viz_type == "spectrogram":
        return (
            "Guidance: Spectrogram panels are stacked vertically (top to bottom) in "
            "input order: series 1 at the top, series 2 below, and so on. Use the "
            "spectrogram to detect time-frequency changes, hidden periodicities, and "
            "regime shifts in variance or frequency bands."
        )
    if viz_type == "stacked":
        return (
            "Guidance: Top panel is a color-coded line overlay (red = series 1, "
            "green = series 2) for direct morphological comparison. Bottom panels "
            "are per-series spectrograms. Use both to judge shape similarity and "
            "any spectral differences between the two series."
        )
    if viz_type == "overlay_line":
        return (
            "Guidance: Series are overlaid on one axis; color order is fixed to "
            "RGB: red (#FF0000) for series 1, green (#00FF00) for series 2. "
            "Use the line overlay to assess shape similarity, trend alignment, "
            "amplitude differences, and phase offsets."
        )
    if viz_type == "dual_cwt":
        return (
            "Guidance: Both panels are Morlet CWT scalograms — series 1 on top, "
            "series 2 below. Shared horizontal frequency bands indicate similar "
            "spectral structure; vertical alignment of power ridges reveals "
            "phase or lag relationships between the two series."
        )
    return (
        "Guidance: Use any provided image to assess topological structure that may "
        "not be visible in scalar evidence."
    )


def _build_vision_alert(vision_meta: dict, viz_type: str) -> str:
    if not isinstance(vision_meta, dict):
        return ""
    if not vision_meta.get("triggered", False):
        return ""

    reasons = vision_meta.get("reasons") or []
    return _select_alert_message(reasons, viz_type)


def _select_alert_message(reasons: list, viz_type: str) -> str:
    viz_label = _visual_label(viz_type)
    if not reasons:
        return (
            "SYSTEM ALERT: Vision tool triggered due to low confidence in the "
            "mathematical evidence. You MUST analyze the provided "
            f"{viz_label} to resolve the question."
        )

    priority = [
        "low_r2",
        "low_fit",
        "fit_ambiguous",
        "fft_unreliable",
        "adf_kpss_disagree",
        "adf_high_pval",
        "arch_effects",
        "pval_marginal",
        "corr_granger_conflict",
        "chunk_variance",
        "high_volatility",
        "weak_seasonal_signal",
        "weak_trend_signal",
        "weak_anomaly_signal",
    ]
    reason = next((r for r in priority if r in reasons), reasons[0])

    templates = {
        "low_r2": (
            "SYSTEM ALERT: Standard regressions failed to capture a clear trend "
            "(R^2 < 0.65). Do not rely on linear assumptions. You MUST analyze the "
            f"provided {viz_label} to identify non-linear structure or regime shifts."
        ),
        "low_fit": (
            "SYSTEM ALERT: Model fit quality is weak (R^2 < 0.65). You MUST analyze "
            f"the provided {viz_label} to determine the dominant pattern."
        ),
        "fit_ambiguous": (
            "SYSTEM ALERT: Competing fits are too close to separate numerically. You "
            f"MUST analyze the provided {viz_label} to resolve the pattern visually."
        ),
        "fft_unreliable": (
            "SYSTEM ALERT: Text-based spectral analysis is unreliable. You MUST look "
            f"at the provided {viz_label} and locate dominant spectral ridges."
        ),
        "adf_kpss_disagree": (
            "SYSTEM ALERT: Stationarity tests disagree. You MUST inspect the provided "
            f"{viz_label} for regime shifts or variance changes."
        ),
        "adf_high_pval": (
            "SYSTEM ALERT: ADF indicates likely non-stationarity (p > 0.10). You MUST "
            f"inspect the provided {viz_label} for structural changes."
        ),
        "arch_effects": (
            "SYSTEM ALERT: ARCH effects detected (heteroskedasticity). You MUST "
            f"inspect the provided {viz_label} for volatility clustering or regime shifts."
        ),
        "pval_marginal": (
            "SYSTEM ALERT: Causality p-values are marginal. You MUST analyze the "
            f"provided {viz_label} to assess lagged relationships."
        ),
        "corr_granger_conflict": (
            "SYSTEM ALERT: Lag correlation contradicts Granger tests. You MUST analyze "
            f"the provided {viz_label} to resolve hidden structure."
        ),
        "chunk_variance": (
            "SYSTEM ALERT: Local metrics vary sharply across chunks. You MUST analyze "
            f"the provided {viz_label} for regime shifts or phase changes."
        ),
        "high_volatility": (
            "SYSTEM ALERT: High volatility detected (std/mean > 1.5). You MUST analyze "
            f"the provided {viz_label} to separate noise from signal."
        ),
        "weak_seasonal_signal": (
            "SYSTEM ALERT: Seasonal signal is weak numerically. You MUST analyze the "
            f"provided {viz_label} to confirm or reject periodic structure."
        ),
        "weak_trend_signal": (
            "SYSTEM ALERT: Trend signal is weak numerically. You MUST analyze the "
            f"provided {viz_label} to determine the true direction."
        ),
        "weak_anomaly_signal": (
            "SYSTEM ALERT: Anomaly signal is weak numerically. You MUST analyze the "
            f"provided {viz_label} to detect rare events or breaks."
        ),
    }

    return templates.get(
        reason,
        (
            "SYSTEM ALERT: Vision tool triggered due to low confidence in the "
            "mathematical evidence. You MUST analyze the provided "
            f"{viz_label} to resolve the question."
        ),
    )


def _visual_label(viz_type: str) -> str:
    if viz_type == "spectrogram":
        return "spectrogram image"
    if viz_type == "line_plot":
        return "line plot image"
    return "image"
