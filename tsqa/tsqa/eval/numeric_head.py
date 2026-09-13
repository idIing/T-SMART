"""
Numeric head (Workstream #4a) — deterministic closed-form / categorical answers.
================================================================================
For free-response rows (``expected_schema != "mcq"``) the *value comes from a
deterministic tool*; the LLM, if used at all, only formats it. There is therefore
nothing to overfit — this is the project's overfitting-proof lever.

Public API (consumed by ``tsqa.eval.runner``):
  - ``infer_numeric_quantity(question)`` : question text -> a quantity descriptor
    (pure, no LLM). Returns ``None`` when the question matches no known
    closed-form template — the caller then ABSTAINS (records ``None``) rather
    than guessing. An honest coverage gap is never a fabricated number.
  - ``compute_numeric_answer(spec, ts, ts1, ts2, evidence)`` -> ``(value, note)``.
  - ``answer_categorical(question, options, evidence)`` -> ``(label, note)``.

Declared coverage (MMTS-Bench Base free-response templates):
  closed-form (exact):  std, var, mean, median, min, max, range, Nth percentile,
                        argmax/argmin index.
  method-sensitive:     local-extrema count, seasonal period length — the value
                        depends on the extremum/period DEFINITION; we implement
                        the standard one and report these as method-sensitive,
                        not as strong claims.
  regression:           trend slope (OLS).
Open-ended free-text (#4b) is explicitly OUT OF SCOPE and recorded as a gap.

The ``note`` field travels with every answer so the analysis can stratify by
``closed_form`` vs ``method_sensitive`` vs ``regression`` and never blur a
brittle period estimate into the exact-stat headline.
"""
import difflib
import re

import numpy as np

from ..tools.basic_stats import basic_stats, percentile as _percentile

# quantity -> provenance class (for honest stratified reporting)
_CLOSED_FORM = {"std", "var", "mean", "median", "min", "max", "range",
                "percentile", "argmax", "argmin"}
_METHOD_SENSITIVE = {"count_local_max", "count_local_min", "period"}
_REGRESSION = {"slope"}


# Canonical quantity keywords used for difflib typo-correction (A1 fuzzy path).
# Restricted to discriminative tokens so we never "correct" a function word.
_QUANTITY_VOCAB = ["standard", "deviation", "variance", "median", "mean", "average",
                   "minimum", "maximum", "range", "percentile", "slope", "period",
                   "skewness", "kurtosis"]

# Paraphrase phrase -> a phrase the ladder already recognises (A1 fuzzy path).
# Longer patterns are applied first (see _apply_synonyms) so "how spread out"
# wins over "spread". This is a DELIBERATELY finite table — open paraphrases it
# does not contain are exactly what the A2 LLM-planner fallback is measured on.
_SYNONYMS = [
    ("how spread out", "standard deviation"),
    ("how variable", "standard deviation"),
    ("dispersion", "standard deviation"),
    ("variability", "standard deviation"),
    ("spread", "standard deviation"),
    ("typical value", "the mean of"),
    ("central tendency", "the mean of"),
    ("central value", "the mean of"),
    ("middle value", "median"),
    ("smallest value", "minimum value"),
    ("lowest value", "minimum value"),
    ("largest value", "maximum value"),
    ("highest value", "maximum value"),
    ("peak value", "maximum value"),
]


def _select_series(question: str) -> str:
    """Which series the question ASKS about. Dual rows name BOTH in the preamble,
    so key on the LAST "time series N" mention; "2" => ts2, else => primary."""
    q = (question or "").lower()
    _ts_mentions = re.findall(r"time series\s*(\d+)", q)
    return "ts2" if _ts_mentions and _ts_mentions[-1] == "2" else "primary"


def _match_ladder(q: str):
    """Pure quantity if-ladder on lower-cased text. Returns ``{"quantity","param"}``
    (no series) or ``None``. Order is specific-first: index/count templates are
    matched BEFORE the bare min/max-value templates so "maximum value" inside an
    index question is never mistaken for a plain max."""
    # --- index questions (argmax / argmin) — most specific first ---
    if "where does the global maximum" in q or ("index" in q and "maximum" in q and "how many" not in q):
        return {"quantity": "argmax", "param": None}
    if "where does the global minimum" in q or ("index" in q and "minimum" in q and "how many" not in q):
        return {"quantity": "argmin", "param": None}

    # --- count questions (local extrema) ---
    if "how many local maximum" in q or "how many local maxima" in q or "number of local maxim" in q:
        return {"quantity": "count_local_max", "param": None}
    if "how many local minimum" in q or "how many local minima" in q or "number of local minim" in q:
        return {"quantity": "count_local_min", "param": None}

    # --- percentile (incl. the 50th == median template) ---
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", q)
    if m:
        return {"quantity": "percentile", "param": float(m.group(1))}

    # --- plain summary statistics ---
    if "standard deviation" in q:
        return {"quantity": "std", "param": None}
    if "variance" in q:
        return {"quantity": "var", "param": None}
    if "median" in q:
        return {"quantity": "median", "param": None}
    if "mean value" in q or "average value" in q or "the mean of" in q or "the average of" in q:
        return {"quantity": "mean", "param": None}
    if "range of" in q or "what is the range" in q:
        return {"quantity": "range", "param": None}
    if "minimum value" in q or "the minimum of" in q:
        return {"quantity": "min", "param": None}
    if "maximum value" in q or "the maximum of" in q:
        return {"quantity": "max", "param": None}
    if "skewness" in q or "skew" in q:
        return {"quantity": "skewness", "param": None}
    if "kurtosis" in q:
        return {"quantity": "kurtosis", "param": None}

    # --- seasonal period / trend slope (method-sensitive / regression) ---
    if "period length" in q or "period of the seasonal" in q or "seasonal period" in q:
        return {"quantity": "period", "param": None}
    if "slope" in q:
        return {"quantity": "slope", "param": None}

    return None


# Bare quantity nouns — the A1 fuzzy fallback for composition operands like
# "the range" that the context-specific ladder (which wants "range of") does not
# catch on a stripped span. RESTRICTED to UNAMBIGUOUS technical terms only:
# "mean"/"average"/"minimum"/"maximum" are deliberately excluded because they
# appear incidentally when describing OTHER quantities (e.g. "average distance
# from the mean" = std), where a bare match would confidently MIS-parse and — since
# A2 only fires on an A1 abstain — block the LLM-planner gate. Their full forms
# ("mean value", "minimum value") are already caught by the ladder.
_BARE_NOUNS = [
    ("standard deviation", "std"), ("variance", "var"), ("median", "median"),
    ("range", "range"), ("skewness", "skewness"), ("kurtosis", "kurtosis"),
    ("slope", "slope"), ("period", "period"),
]


def _match_bare(q: str):
    """Last-resort A1 fuzzy matcher for a bare quantity noun in a span."""
    for noun, quant in _BARE_NOUNS:
        if re.search(r"\b" + re.escape(noun) + r"\b", q):
            return {"quantity": quant, "param": None}
    return None


def _apply_synonyms(q: str) -> str:
    """Substitute known paraphrase phrases with ladder-recognised phrases."""
    for pat, repl in sorted(_SYNONYMS, key=lambda kv: -len(kv[0])):
        if pat in q:
            q = q.replace(pat, repl)
    return q


def _typo_correct(q: str) -> str:
    """Edit-distance correct quantity tokens against ``_QUANTITY_VOCAB`` (cutoff
    0.8, only tokens >=4 chars), so "varaince"->"variance", "standar"->"standard".
    Conservative: never touches a token that has no close vocab match."""
    out = []
    for tok in re.split(r"(\W+)", q):  # keep delimiters
        if tok.isalpha() and len(tok) >= 4 and tok not in _QUANTITY_VOCAB:
            cand = difflib.get_close_matches(tok, _QUANTITY_VOCAB, n=1, cutoff=0.8)
            out.append(cand[0] if cand else tok)
        else:
            out.append(tok)
    return "".join(out)


def match_single_quantity(span: str, fuzzy: bool = False):
    """Map a text SPAN to ``{"quantity","param"}`` or ``None``.

    ``fuzzy=False`` is the exact A0 ladder. ``fuzzy=True`` (used by the A1
    deterministic parser) first applies the synonym table, and on a miss retries
    once after edit-distance typo correction.
    """
    q = (span or "").lower()
    if fuzzy:
        q = _apply_synonyms(q)
    m = _match_ladder(q)
    if m is None and fuzzy:
        qc = _typo_correct(q)
        if qc != q:
            m = _match_ladder(qc)
    if m is None and fuzzy:
        m = _match_bare(q)   # bare operand nouns ("the range", "the mean")
    return m


def infer_numeric_quantity(question: str):
    """A0 (unchanged behaviour): map a free-response question to a deterministic
    quantity descriptor ``{"quantity","param","series"}`` or ``None``. Exact-match
    only (no synonyms/typos) — the brittle baseline the three-arm study controls
    against. ``parse_plan_deterministic`` (A1) is the upgraded parser.
    """
    m = _match_ladder((question or "").lower())
    if m is None:
        return None  # honest coverage gap — caller abstains
    m["series"] = _select_series(question)
    return m


def _pick_series(series_sel, ts, ts1, ts2):
    """Select the series the question refers to (defensive about None/empty)."""
    def _ok(a):
        return a is not None and getattr(a, "size", len(a) if a is not None else 0) > 0

    if series_sel == "ts2" and _ok(ts2):
        return np.asarray(ts2, dtype=float)
    if _ok(ts):
        return np.asarray(ts, dtype=float)
    if _ok(ts1):
        return np.asarray(ts1, dtype=float)
    return None


def _count_local_extrema(s, kind):
    """Standard strict interior local-extrema count.

    A point i (0 < i < n-1) is a local max iff s[i-1] < s[i] > s[i+1] (min
    analogously). Endpoints are excluded and plateaus are NOT counted. This is a
    *definition choice*; the value is method-sensitive and reported as such.
    """
    s = np.asarray(s, dtype=float)
    if s.size < 3:
        return 0
    left = s[1:-1] - s[:-2]
    right = s[1:-1] - s[2:]
    if kind == "max":
        return int(np.sum((left > 0) & (right > 0)))
    return int(np.sum((left < 0) & (right < 0)))


def _dominant_period(s):
    """Fallback dominant period via the real FFT of the mean-removed series.

    Used only when the routed branch did not supply a period estimate. Returns
    the period (series_length / dominant_non_DC_frequency_bin) as a float, or
    ``None`` if no non-trivial peak exists. Method-sensitive by nature.
    """
    s = np.asarray(s, dtype=float)
    n = s.size
    if n < 4:
        return None
    sd = s - s.mean()
    spectrum = np.abs(np.fft.rfft(sd))
    if spectrum.size <= 1:
        return None
    k = int(np.argmax(spectrum[1:]) + 1)  # skip DC
    if k == 0:
        return None
    return float(n) / float(k)


def compute_numeric_answer(spec, ts, ts1, ts2, evidence):
    """Compute the requested quantity. Returns ``(value, note)``.

    ``value`` is a python float/int (or ``None`` if uncomputable). ``note`` is the
    provenance class: ``closed_form`` | ``method_sensitive`` | ``regression``,
    suffixed (e.g. ``method_sensitive:branch`` vs ``method_sensitive:fft``) when
    relevant so the source is auditable.
    """
    quantity = spec["quantity"]
    s = _pick_series(spec.get("series", "primary"), ts, ts1, ts2)
    evidence = evidence if isinstance(evidence, dict) else {}

    # slope / period prefer the routed branch's already-computed value, falling
    # back to a direct computation so the answer never depends on routing luck.
    if quantity == "slope":
        slope = evidence.get("slope")
        if slope is not None:
            return float(slope), "regression:branch"
        if s is not None and s.size >= 2:
            x = np.arange(s.size, dtype=float)
            return float(np.polyfit(x, s, 1)[0]), "regression:ols"
        return None, "regression:unavailable"

    if quantity == "period":
        for k in ("period_reconciled", "dominant_period_fft"):
            v = evidence.get(k)
            if v is not None:
                try:
                    return float(v), f"method_sensitive:branch:{k}"
                except (TypeError, ValueError):
                    pass
        if s is not None:
            p = _dominant_period(s)
            if p is not None:
                return p, "method_sensitive:fft"
        return None, "method_sensitive:unavailable"

    if s is None:
        return None, "no_series"

    if quantity == "percentile":
        return float(_percentile(s, spec["param"])), "closed_form"
    if quantity == "argmax":
        return int(np.argmax(s)), "closed_form"
    if quantity == "argmin":
        return int(np.argmin(s)), "closed_form"
    if quantity == "count_local_max":
        return _count_local_extrema(s, "max"), "method_sensitive:strict_interior"
    if quantity == "count_local_min":
        return _count_local_extrema(s, "min"), "method_sensitive:strict_interior"

    stats = basic_stats(s)
    if quantity in stats:
        return float(stats[quantity]), "closed_form"
    return None, "unknown_quantity"


# ---------------------------------------------------------------------------
# A1 deterministic COMPOSITION parser + the unified plan evaluator.
# A "plan" is either a leaf {"quantity","param","series"} or a composition
# {"op": <binop>, "args": [plan, plan], "series": sel}. Closed op registry only.
# ---------------------------------------------------------------------------

_COMPOSE_OPS = {
    "subtract": lambda a, b: a - b,
    "add":      lambda a, b: a + b,
    "product":  lambda a, b: a * b,
    "ratio":    lambda a, b: (a / b) if b != 0 else None,
    "abs_diff": lambda a, b: abs(a - b),
}

# The CLOSED registry the A2 LLM-planner may emit (and parse_numeric_plan_json
# validates against). Quantities are exactly those compute_numeric_answer owns;
# ops are the _COMPOSE_OPS keys. The LLM proposes a PLAN over this registry —
# never a number — so the arm is overfitting-safe (cf. action_proposer).
NUMERIC_REGISTRY_QUANTITIES = ["std", "var", "mean", "median", "min", "max", "range",
                               "percentile", "argmax", "argmin", "count_local_max",
                               "count_local_min", "period", "slope"]
NUMERIC_REGISTRY_OPS = list(_COMPOSE_OPS)


def validate_plan(plan) -> bool:
    """True iff ``plan`` is a well-formed plan over the closed registry: a leaf
    with a registry quantity, or a binary composition with a registry op and two
    valid sub-plans. Used to reject anything the LLM hallucinates outside the
    registry (overfitting-safety guard)."""
    if not isinstance(plan, dict):
        return False
    if "op" in plan:
        args = plan.get("args")
        return (plan["op"] in NUMERIC_REGISTRY_OPS
                and isinstance(args, list) and len(args) == 2
                and all(validate_plan(a) for a in args))
    return plan.get("quantity") in NUMERIC_REGISTRY_QUANTITIES


# Prefix forms: "(sum|difference|ratio|product) of/between A {and|to} B".
_PREFIX_COMPOSITION = [
    (re.compile(r"difference between (.+?) and (.+)"), "subtract"),
    (re.compile(r"ratio of (.+?) to (.+)"),            "ratio"),
    (re.compile(r"ratio between (.+?) and (.+)"),      "ratio"),
    (re.compile(r"sum of (.+?) and (.+)"),             "add"),
    (re.compile(r"product of (.+?) and (.+)"),         "product"),
]
# Infix operator words (kept conservative — ambiguous tokens like "over"/"times"
# are excluded so a single-quantity question is never mis-split into an abstain).
_INFIX_COMPOSITION = [
    (" divided by ", "ratio"),
    (" minus ", "subtract"),
    (" plus ", "add"),
]


def _detect_composition(q: str):
    """Detect a binary composition. Returns ``(op, span1, span2)`` or
    ``(None, None, None)``. Prefix forms (more specific) are tried before infix."""
    for rx, op in _PREFIX_COMPOSITION:
        m = rx.search(q)
        if m:
            return op, m.group(1), m.group(2)
    for token, op in _INFIX_COMPOSITION:
        if token in q:
            i = q.index(token)
            return op, q[:i], q[i + len(token):]
    return None, None, None


def parse_plan_deterministic(question: str):
    """A1: deterministic parser with a composition grammar + fuzzy/synonym/typo
    matching. Returns a plan (leaf or composition) or ``None`` (abstain — the A2
    LLM-planner then gets its turn). NO LLM is involved."""
    q = (question or "").lower()
    series = _select_series(question)
    op, s1, s2 = _detect_composition(q)
    if op:
        l1 = match_single_quantity(s1, fuzzy=True)
        l2 = match_single_quantity(s2, fuzzy=True)
        if l1 and l2:
            l1["series"] = series
            l2["series"] = series
            return {"op": op, "args": [l1, l2], "series": series}
        return None  # one operand unresolved → abstain (A2's turn)
    spec = match_single_quantity(q, fuzzy=True)
    if spec:
        spec["series"] = series
        return spec
    return None


def evaluate_plan(plan, ts, ts1, ts2, evidence):
    """Execute a plan (leaf or composition) with the deterministic tools →
    ``(value, note)``. The closed op registry is enforced — an unknown op or an
    unavailable operand ABSTAINS (None), never fabricates. Single evaluator for
    all three parser arms; leaves delegate to ``compute_numeric_answer`` so the
    tool computation is identical across A0/A1/A2."""
    if not isinstance(plan, dict):
        return None, "invalid_plan"
    if "op" in plan:
        op = plan.get("op")
        if op not in _COMPOSE_OPS:
            return None, f"invalid_op:{op}"
        args = plan.get("args") or []
        if len(args) != 2:
            return None, "composition:arity"
        vals = []
        for arg in args:
            v, _ = evaluate_plan(arg, ts, ts1, ts2, evidence)
            if v is None:
                return None, "composition:operand_unavailable"
            vals.append(v)
        try:
            r = _COMPOSE_OPS[op](float(vals[0]), float(vals[1]))
        except Exception:
            return None, "composition:error"
        if r is None:
            return None, "composition:divzero"
        return float(r), f"composition:{op}"
    # leaf → existing tool computation (identical across arms)
    if plan.get("quantity") is None:
        return None, "invalid_leaf"
    spec = {"quantity": plan.get("quantity"), "param": plan.get("param"),
            "series": plan.get("series", "primary")}
    return compute_numeric_answer(spec, ts, ts1, ts2, evidence)


# ---------------------------------------------------------------------------
# Categorical path (dormant on MMTS-Bench Base — no such rows there — but kept
# real for generality to benchmarks that DO ask "is this series stationary?").
# ---------------------------------------------------------------------------

def answer_categorical(question, options, evidence):
    """Map a deterministic categorical determination to a label.

    Currently covers the stationarity determination: the noise branch's
    ADF/KPSS-derived ``is_stationary`` flag -> "stationary" / "non-stationary".
    Returns ``(label, note)``; ``label`` is ``None`` if no deterministic signal
    is available (honest abstain).
    """
    evidence = evidence if isinstance(evidence, dict) else {}
    is_stat = evidence.get("is_stationary")
    if is_stat is None:
        return None, "categorical:no_signal"
    label = "stationary" if bool(is_stat) else "non-stationary"
    # If explicit options exist, map the determination onto the closest option
    # text (case/space-insensitive substring) so the emitted label matches the
    # benchmark's expected surface form.
    if options:
        target = label.replace("-", "").replace(" ", "")
        for opt in options:
            o = str(opt).strip().lower().replace("-", "").replace(" ", "")
            if target in o or o in target:
                return str(opt).strip(), "categorical:stationarity:mapped"
    return label, "categorical:stationarity"
