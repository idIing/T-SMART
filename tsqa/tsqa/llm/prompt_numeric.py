"""Format-only prompts for the numeric / categorical head.

This module serves the free-response branch of the pipeline: questions whose
answer was ALREADY computed by a deterministic tool (e.g. "what is the standard
deviation?" → 260.79, or "is this series stationary?" → "stationary"). The tool
owns the value; the LLM's ONLY job is to RESTATE that value in the answer form
the question asks for.

No-nudging contract (load-bearing — see CLAUDE.md "Research norms"):
  - The prompts NEVER ask the model to reason about, recompute, verify, or
    second-guess the value. They hand the model the answer and ask it to format
    it.
  - There is no "pick the best", no options-pressure, no hint toward a different
    number/label. Rounding to match a requested precision is the only permitted
    transformation of a numeric value.
  - If the prompt cannot make the model change the answer, the head cannot
    introduce answer-nudging — by construction.

Public API (imported by the runner):
  build_numeric_prompt(question, computed_value, quantity_label=None) -> (system, user)
  build_categorical_prompt(question, options, computed_label)         -> (system, user)
  parse_numeric_answer(raw)                                           -> dict
  parse_categorical_answer(raw, allowed=None)                         -> dict

The two builders mirror the (system_prompt, user_prompt) tuple convention used
by fallback.py; the two parsers mirror the lenient, dict-returning style of
parser.py.
"""

import re

# ---------------------------------------------------------------------------
# SYSTEM prompts — neutral, format-only. They explicitly forbid recomputation.
# ---------------------------------------------------------------------------

NUMERIC_SYSTEM_PROMPT = """\
You are a careful assistant that reports a precomputed numeric result exactly.
A deterministic analysis tool has already computed the answer. Your only job is
to restate that value as the final answer — do NOT recompute, verify, or change
it. You may round it to match the precision the question asks for, but the
underlying number must be the value you were given. Report nothing else.
"""

CATEGORICAL_SYSTEM_PROMPT = """\
You are a careful assistant that reports a precomputed categorical result
exactly. A deterministic analysis tool has already determined the result. Your
only job is to state that result (mapped to the matching answer option when
options are provided) — do NOT argue for a different label or reconsider the
determination. Report nothing else.
"""

# ---------------------------------------------------------------------------
# User-prompt templates
# ---------------------------------------------------------------------------

_NUMERIC_TEMPLATE = """\
## Question
{question}

## Precomputed result
The {label} has already been computed by a deterministic tool:

    {value}

## Task
Report exactly this value as the answer. Do not recompute or change it; you may
only round it to match the precision the question requests. Put the number alone
on the first line. Optionally add one short sentence on a second line.
"""

_CATEGORICAL_TEMPLATE = """\
## Question
{question}
{options_block}
## Precomputed result
A deterministic tool has already determined the result:

    {label}

## Task
Report this result as the answer{options_hint}. Do not argue for a different
label. Put the chosen label alone on the first line. Optionally add one short
sentence on a second line.
"""

_OPTIONS_SECTION = """\

## Answer choices
{options_list}
"""


def build_numeric_prompt(
    question: str, computed_value, quantity_label: str | None = None
) -> tuple[str, str]:
    """Build a (system_prompt, user_prompt) pair that asks the model to RESTATE
    an already-computed numeric value.

    Args:
        question:       raw free-response question string.
        computed_value: the value the deterministic tool produced (number-like).
        quantity_label: optional name of the quantity (e.g. "standard
                        deviation"); used only to phrase the prompt neutrally.

    Returns:
        (NUMERIC_SYSTEM_PROMPT, user_prompt) — the user prompt explicitly
        contains the computed value and instructs the model to report exactly
        that value (rounding to the requested precision is allowed; recomputing
        or changing it is not). Output format: number alone on the first line,
        optional one-sentence note on the second.
    """
    label = (quantity_label or "value").strip() or "value"
    user_prompt = _NUMERIC_TEMPLATE.format(
        question=question,
        label=label,
        value=_format_value(computed_value),
    )
    return NUMERIC_SYSTEM_PROMPT, user_prompt


def build_categorical_prompt(
    question: str, options: list, computed_label: str
) -> tuple[str, str]:
    """Build a (system_prompt, user_prompt) pair that asks the model to map an
    already-determined categorical result to the matching answer/option.

    Args:
        question:       raw question string.
        options:        list of answer-choice strings (may be empty/None).
        computed_label: the categorical result the deterministic tool produced
                        (e.g. "stationary").

    Returns:
        (CATEGORICAL_SYSTEM_PROMPT, user_prompt) — the user prompt includes the
        computed label and the options (if any) and instructs the model never to
        argue for a different label. Output: chosen label/option alone on the
        first line, optional one-sentence reason on the second.
    """
    opts = list(options) if options else []
    if opts:
        options_list = "\n".join(f"- {opt}" for opt in opts)
        options_block = _OPTIONS_SECTION.format(options_list=options_list)
        options_hint = ", choosing the matching option above"
    else:
        options_block = ""
        options_hint = ""

    user_prompt = _CATEGORICAL_TEMPLATE.format(
        question=question,
        options_block=options_block,
        label=str(computed_label).strip(),
        options_hint=options_hint,
    )
    return CATEGORICAL_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Parsers — lenient, dict-returning (mirrors parser.parse_answer style)
# ---------------------------------------------------------------------------

# A well-formed number: optional sign, digits with optional thousands-separator
# commas, optional decimal part, optional scientific-notation exponent.
# Also matches a leading ".5"-style decimal.
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
# Stricter "has at least one digit" guard used to reject empty matches.
_HAS_DIGIT_RE = re.compile(r"\d")

# Leading boilerplate to strip before number extraction (case-insensitive).
_LEADING_BOILERPLATE_RE = re.compile(r"^\s*(?:answer|result|value)\s*[:=]?\s*", re.IGNORECASE)


def parse_numeric_answer(raw: str) -> dict:
    """Extract a single number from raw LLM text.

    Lenient by design: tolerates leading text/labels ("Answer:", a bare "="),
    commas as thousands separators, scientific notation, trailing units, and a
    trailing percent sign. Prefers the FIRST well-formed number on the first
    non-empty line; if that line has no number, falls back to the first number
    anywhere in the text. Negative numbers and decimals are supported.

    Returns:
        {"value": float | None, "parse_success": bool, "raw": raw}
    """
    if not raw or not str(raw).strip():
        return {"value": None, "parse_success": False, "raw": raw}

    text = str(raw)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]

    # 1) First well-formed number on the first non-empty line.
    if lines:
        val = _first_number(lines[0])
        if val is not None:
            return {"value": val, "parse_success": True, "raw": raw}

    # 2) First well-formed number anywhere in the text.
    val = _first_number(text)
    if val is not None:
        return {"value": val, "parse_success": True, "raw": raw}

    return {"value": None, "parse_success": False, "raw": raw}


def parse_categorical_answer(raw: str, allowed: list | None = None) -> dict:
    """Extract a categorical label from raw LLM text.

    If ``allowed`` is given, the raw text is matched (case-insensitive,
    whitespace-normalized, substring-tolerant) against each allowed label and
    the canonical allowed value is returned. Without an allowed list, the first
    non-empty line is returned stripped.

    Returns:
        {"label": str | None, "parse_success": bool, "raw": raw}
    """
    if not raw or not str(raw).strip():
        return {"label": None, "parse_success": False, "raw": raw}

    text = str(raw)
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    # Strip a leading "Answer:"/"Result:" label if present.
    first_line_clean = _LEADING_BOILERPLATE_RE.sub("", first_line).strip()

    if allowed:
        canonical = _match_allowed(text, first_line_clean, allowed)
        if canonical is not None:
            return {"label": canonical, "parse_success": True, "raw": raw}
        return {"label": None, "parse_success": False, "raw": raw}

    # No allowed list: return the first non-empty line (label-stripped).
    label = first_line_clean or first_line
    if label:
        return {"label": label, "parse_success": True, "raw": raw}
    return {"label": None, "parse_success": False, "raw": raw}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
    """Render the computed value for inclusion in the prompt without altering it.

    Floats are passed through ``repr`` to preserve full precision; everything
    else is stringified as-is. The model is told it may round, so we hand it the
    most precise form available.
    """
    if isinstance(value, bool):  # avoid True/False being treated as 1/0
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _first_number(text: str):
    """Return the first well-formed number in ``text`` as a float, or None.

    Strips a leading boilerplate label ("Answer:", "= ", etc.) before scanning
    so the number itself is matched. Handles thousands-separator commas,
    scientific notation, a trailing percent sign, and surrounding units/text.
    """
    if not text:
        return None
    cleaned = _LEADING_BOILERPLATE_RE.sub("", text)
    # Also tolerate a bare leading "=" (e.g. "= 1,234.5").
    cleaned = cleaned.lstrip()
    if cleaned.startswith("="):
        cleaned = cleaned[1:].lstrip()
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = re.sub(r"\bminus\s+(?=\d|\.)", "-", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([+-])\s+(?=\d|\.)", r"\1", cleaned)

    for m in _NUMBER_RE.finditer(cleaned):
        token = m.group(0)
        if not token or not _HAS_DIGIT_RE.search(token):
            continue
        num = token.replace(",", "")
        try:
            value = float(num)
        except ValueError:
            continue
        # Honor a trailing percent sign immediately after the number.
        rest = cleaned[m.end():].lstrip()
        if rest.startswith("%"):
            value = value / 100.0
        return value
    return None


def _normalize(s: str) -> str:
    """Lower-case and collapse internal whitespace for tolerant comparison."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _match_allowed(full_text: str, first_line: str, allowed: list):
    """Map text to one of ``allowed`` (case-insensitive, substring-tolerant).

    Matching order, returning the canonical allowed value on first hit:
      1. Exact (normalized) equality with the first line.
      2. Exact (normalized) equality with any non-empty line.
      3. Substring: an allowed label appears within the first line, or the first
         line appears within an allowed label.
      4. Substring anywhere in the full text (longest allowed label first, so
         e.g. "non-stationary" wins over "stationary").
    """
    norm_first = _normalize(first_line)
    norm_full = _normalize(full_text)
    norm_allowed = [(_normalize(opt), opt) for opt in allowed if str(opt).strip()]

    # 1) exact match on first line
    for norm_opt, opt in norm_allowed:
        if norm_opt and norm_opt == norm_first:
            return opt

    # 2) exact match on any non-empty line
    norm_lines = [_normalize(ln) for ln in full_text.splitlines() if ln.strip()]
    for norm_opt, opt in norm_allowed:
        if norm_opt and norm_opt in norm_lines:
            return opt

    # 3) substring within the first line (either direction)
    for norm_opt, opt in sorted(norm_allowed, key=lambda x: len(x[0]), reverse=True):
        if not norm_opt:
            continue
        if norm_opt in norm_first or (norm_first and norm_first in norm_opt):
            return opt

    # 4) substring anywhere in the full text (longest label first)
    for norm_opt, opt in sorted(norm_allowed, key=lambda x: len(x[0]), reverse=True):
        if norm_opt and norm_opt in norm_full:
            return opt

    return None


# ---------------------------------------------------------------------------
# A2 LLM-PLANNER (the fallback parser arm). The LLM maps a question to a PLAN over
# the closed numeric registry — NEVER a number; deterministic tools then execute
# it. This is the numeric analogue of the Stage-3 action proposer: the LLM makes a
# routing/parsing decision over a closed registry, so there is nothing to overfit.
# ---------------------------------------------------------------------------

import json as _json

NUMERIC_PLANNER_SYSTEM_PROMPT = """\
You translate a numeric time-series question into a PLAN that deterministic tools
will execute. You NEVER compute or output a number yourself — only a JSON plan that
says WHICH quantity (or composition of quantities) the question asks for, using ONLY
the allowed quantities and operations.
"""

_PLANNER_TEMPLATE = """\
## Allowed quantities (a leaf plan)
{quantities}
(For "percentile" include "param" = the N in "Nth percentile", e.g. 25.)

## Allowed operations (a binary composition plan)
{ops}

## Plan JSON
A leaf:        {{"quantity": "<name>", "param": <number or null>}}
A composition: {{"op": "<operation>", "args": [<plan>, <plan>]}}

## Question
{question}

## Task
Output ONLY the JSON plan capturing what the question asks — a single leaf, or a
composition for "A minus B", "the ratio of A to B", "difference between A and B",
etc. Use ONLY the names listed above. If the asked quantity is NOT in the list,
output {{"quantity": null}}. No prose, no numbers — only the JSON plan.
"""


def build_numeric_planner_prompt(question: str) -> tuple:
    """Build the (system, user) prompt asking the LLM for a PLAN over the closed
    numeric registry (it never emits a number). Registry is pulled live from
    numeric_head so the prompt can never drift from what the evaluator accepts."""
    from ..eval.numeric_head import (NUMERIC_REGISTRY_QUANTITIES,  # lazy: avoid cycle
                                     NUMERIC_REGISTRY_OPS)
    return NUMERIC_PLANNER_SYSTEM_PROMPT, _PLANNER_TEMPLATE.format(
        quantities=", ".join(NUMERIC_REGISTRY_QUANTITIES),
        ops=", ".join(NUMERIC_REGISTRY_OPS),
        question=question,
    )


def _first_json_object(text: str):
    """Return the first balanced {...} object parsed from text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def parse_numeric_plan_json(raw: str):
    """Extract a plan from the LLM reply and VALIDATE it against the closed
    registry (`numeric_head.validate_plan`). Returns the plan dict or None — an
    out-of-registry / malformed / `quantity:null` plan is rejected (abstain),
    never fabricated."""
    from ..eval.numeric_head import validate_plan  # lazy: avoid import cycle
    plan = _first_json_object(str(raw) if raw is not None else "")
    if plan is None:
        return None
    return plan if validate_plan(plan) else None
