# ---------------------------------------------------------------------------
# Router prompt
# The router is the ONLY place that reads the raw question text and decides
# which analysis branch to invoke. It outputs JSON — nothing else.
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are a routing classifier for a time series question-answering pipeline.
Given a multiple-choice question about time series data, output a JSON object
that selects the correct analysis branch, scope, and subtype.

You must output ONLY valid JSON — no prose, no markdown fences, no explanation.

Branch taxonomy:
  "trend"        — questions about direction, increase/decrease, slope, drift,
                   long-term movement, or whether a series is growing/falling
  "periodicity"  — questions about cycles, repeating patterns, frequency,
                   period length, seasonality, or sine/cosine shape
  "anomaly"      — questions about spikes, outliers, unusual points,
                   sudden jumps, pattern flips, or structural breaks
  "noise"        — questions about randomness, white noise, stationarity,
                   autocorrelation, random walk, or predictability
  "similarity"   — questions comparing two series by shape, level, amplitude,
                   correlation, or whether they look alike
  "causality"    — questions about whether one series drives or predicts
                   another, lag relationships, Granger causality, or when
                   one series is described as a lagged/shifted version of another

Scope:
  "global" — the question asks about the whole series (default)
  "local"  — the question asks about a specific region, segment, or whether
             behaviour changes across different parts of the series.
             Use "local" when the question contains phrases like:
             "latter half", "second half", "first half", "at some point",
             "changes over time", "transition", "concatenated", "any part",
             "does the trend change", or asks about trend/pattern evolution
             across segments.

Series type:
  "single"  — question involves one time series
  "dual"    — question involves two time series

Subtype — pick the ONE that best matches the question's core ask:
  trend branch:
    "direction"       — is the series increasing, decreasing, or flat?
    "functional_form" — is the trend linear, exponential, or logarithmic?
    "change_point"    — where does the trend change, how many segments/pieces/regimes,
                        what is the ordering of trend components, or how does the first
                        half compare to the second half? Always use scope="local".

  periodicity branch:
    "period_value"      — what is the numeric period or frequency?
    "period_change"     — does the period/frequency change over time (increases, decreases, varies)?
                          Use scope="local". Trigger on: "period increase", "period decrease",
                          "frequency change", "period vary", "cycle length change".
    "amplitude_change"  — does the amplitude (height of cycles) change over time?
                          Use scope="local". Trigger on: "amplitude increase", "amplitude decrease",
                          "amplitude vary", "amplitude change", "how does the amplitude",
                          "cycle height", "envelope increase/decrease".
                          Do NOT use "waveform_type" for these questions.
    "waveform_type"     — what is the SHAPE of the waveform itself (sine / square / sawtooth /
                          triangular)? Use ONLY when the question asks about the geometric form
                          or type of wave, NOT about amplitude or period trends.

  anomaly branch:
    "point_anomaly"   — are there isolated spikes or outliers?
    "pattern_flip"    — is there a structural break or pattern reversal?

  noise branch:
    "white_noise"      — is the series white noise?
    "stationarity"     — is the series stationary?
    "variance_level"   — what is the noise variance / how noisy is it?
    "autocorr_value"   — what is the autocorrelation at a specific lag?
    "ar_ma_process"    — is the series more likely AR(1), MA(1), or similar?
    **Note**: Use "similarity" instead of "noise" if the question explicitly asks to COMPARE two different time series models against each other.

  similarity branch (Use whenever there are TWO series being compared):
    "statistical"     — are the statistical properties (mean, range, variance) similar?
    "shape"           — do the two series have the same waveform shape?
    "lag_offset"      — is one series a lagged/shifted version of the other?

  causality branch:
    "lag_direction"   — which series leads or causes the other?
    "lag_value"       — what is the exact lag value?

Output schema (all fields required):
{
  "branch":      "<one of the six branch names above>",
  "scope":       "global" | "local",
  "series_type": "single" | "dual",
  "subtype":     "<one of the subtype values for the chosen branch>"
}
"""

_ROUTER_TEMPLATE = """\
## Question
{question}

## Series info
{series_info}

Output the routing JSON now.\
"""


def build_router_prompt(
    question: str, has_ts1: bool = False, has_ts2: bool = False
) -> str:
    """
    Build the routing prompt for the LLM router call.

    Args:
        question: raw question string from the dataset
        has_ts1:  True if the dataset row has a non-null ts1 field
        has_ts2:  True if the dataset row has a non-null ts2 field

    Returns:
        User-turn prompt string. Send ROUTER_SYSTEM_PROMPT as the system message.

    The series_info line gives the router a factual signal about how many
    series are present so it does not need to infer it purely from question text.
    """
    if has_ts1 and has_ts2:
        series_info = "Two time series (ts1 and ts2) are provided."
    elif has_ts1 or has_ts2:
        series_info = "One time series is provided."
    else:
        series_info = "Series data is embedded in the question context."

    return _ROUTER_TEMPLATE.format(question=question, series_info=series_info)
