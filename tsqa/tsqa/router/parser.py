import json
import re

VALID_BRANCHES = {"trend", "periodicity", "anomaly", "noise", "similarity", "causality"}
VALID_SCOPES = {"global", "local"}
VALID_SERIES = {"single", "dual"}
# expected_schema describes the SHAPE of the answer the row demands, independent
# of the analysis branch. It is the switch that gates the numeric head: only
# rows with expected_schema != "mcq" leave the (byte-identical) MCQ path.
VALID_SCHEMAS = {"mcq", "numerical", "categorical"}

# Free-response questions that ask for a categorical determination rather than a
# number. Kept deliberately narrow + literal (no model judgement). On MMTS-Bench
# Base this matches ~0 rows (its "stationarity" task_type asks for numeric
# min/max/mean, not a stationary/non-stationary label) — it exists for
# generality to other benchmarks, and that dormancy is reported, not hidden.
_CATEGORICAL_Q_PATTERNS = (
    "is the series stationary",
    "is this series stationary",
    "stationary or non-stationary",
    "stationary or not",
    "stationary or nonstationary",
)

# valid subtypes per branch
VALID_SUBTYPES = {
    "trend": {"direction", "functional_form", "change_point"},
    "periodicity": {"period_value", "period_change", "amplitude_change", "waveform_type"},
    "anomaly": {"point_anomaly", "pattern_flip"},
    "noise": {
        "white_noise",
        "stationarity",
        "variance_level",
        "autocorr_value",
        "ar_ma_process",
    },
    "similarity": {"statistical", "shape", "lag_offset"},
    "causality": {"lag_direction", "lag_value"},
}

# strip markdown code fences if the model wraps its JSON in ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_routing(raw: str) -> dict:
    """
    Parse the router LLM output into a validated routing decision.

    Args:
        raw: raw string returned by the router LLM call

    Returns:
        branch:        str  one of the six valid branch names, or None
        scope:         str  "global" | "local", defaults to "global" on ambiguity
        series_type:   str  "single" | "dual", defaults to "single" on ambiguity
        subtype:       str  branch-specific subtype, or None if missing/invalid
        parse_success: bool False if JSON could not be parsed or branch is invalid
        raw:           str  original raw output, for logging

    On parse failure the runner should fall back to a keyword-based branch guess
    or route to the most general branch ("noise") rather than crashing.
    """
    text = raw.strip()

    # strip markdown fences if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # try to find the outermost braces
        brace_m = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_m:
            try:
                obj = json.loads(brace_m.group())
            except json.JSONDecodeError:
                return _failed(raw)
        else:
            return _failed(raw)

    branch = str(obj.get("branch", "")).lower().strip()
    scope = str(obj.get("scope", "global")).lower().strip()
    series_type = str(obj.get("series_type", "single")).lower().strip()
    subtype_raw = str(obj.get("subtype", "")).lower().strip()
    # expected_schema defaults to "mcq" — the router prompt is unchanged and does
    # not emit it, so MCQ behaviour is untouched. The authoritative value is set
    # structurally downstream via infer_expected_schema(); this only carries a
    # value through if a future router prompt ever emits one.
    schema_raw = str(obj.get("expected_schema", "mcq")).lower().strip()
    expected_schema = schema_raw if schema_raw in VALID_SCHEMAS else "mcq"

    if branch not in VALID_BRANCHES:
        return _failed(raw)

    # coerce slightly malformed values to valid ones rather than hard-failing
    if scope not in VALID_SCOPES:
        scope = "global"
    if series_type not in VALID_SERIES:
        series_type = "single"

    # validate subtype against the branch's allowed set; None if unrecognised
    allowed_subtypes = VALID_SUBTYPES.get(branch, set())
    subtype = subtype_raw if subtype_raw in allowed_subtypes else None

    return {
        "branch": branch,
        "scope": scope,
        "series_type": series_type,
        "subtype": subtype,
        "expected_schema": expected_schema,
        "parse_success": True,
        "raw": raw,
    }


def _failed(raw: str) -> dict:
    return {
        "branch": None,
        "scope": "global",
        "series_type": "single",
        "subtype": None,
        "expected_schema": "mcq",
        "parse_success": False,
        "raw": raw,
    }


def infer_expected_schema(options, question: str = None,
                          category: str = None, subtype: str = None) -> str:
    """Deterministically decide the answer SHAPE a row demands.

    This is the structural switch that gates the numeric head. It is intentionally
    NOT an LLM decision — it keys on whether parseable answer options exist:

        >= 2 non-empty options present   -> "mcq"        (unchanged pipeline path)
        no options + categorical wording -> "categorical"
        no options otherwise             -> "numerical"

    On MMTS-Bench Base this separates the 308 MCQ/binary rows (all have options)
    from the 392 free-response rows (all have none) with 100% structural fidelity,
    so an MCQ row can NEVER be misrouted into the numeric head — the isolation
    guarantee the experiment depends on. `category`/`subtype` are accepted for
    forward compatibility but only the option structure + literal categorical
    wording are consulted (no fuzzy heuristics that could leak into MCQ rows).
    """
    n_opts = 0
    if options is not None:
        try:
            n_opts = sum(1 for o in options if str(o).strip())
        except TypeError:
            n_opts = 0
    if n_opts >= 2:
        return "mcq"

    q = (question or "").lower()
    if any(p in q for p in _CATEGORICAL_Q_PATTERNS):
        return "categorical"
    return "numerical"
