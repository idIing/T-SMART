import json
from .prompt import (
    SYSTEM_PROMPT,
    _json_default,
    _FLAGS_SECTION,
    _NO_FLAGS_SECTION,
    _build_artifacts_block,
    _build_trend_block,
    _build_vision_block,
    _CORE_KEYS,
    _SUBTYPE_KEYS,
    _COGNITIVE_GUIDELINES,
)

# ---------------------------------------------------------------------------
# One-shot answer-mapping prompt template
# Adds a "Hint" section populated from the dataset's question_hint field.
# Placeholders: {hint_block} {question} {options_block} {evidence_json} {flags_block}
# ---------------------------------------------------------------------------

_ONESHOT_TEMPLATE = """\
{hint_block}\
## Question
{question}

## Answer choices
{options_block}

## Evidence (from deterministic analysis pipeline)
```json
{evidence_json}
```
{trend_block}\
{artifacts_block}\
{vision_block}\
{flags_block}\
## Task
{task_block}
"""

_HINT_SECTION = """\
## Hint (from dataset)
{hint}

"""


def build_oneshot_prompt(
    question: str,
    options: list,
    evidence: dict,
    question_hint: str = None,
    flags=None,
    branch: str = None,
    subtype: str = None,
) -> str:
    """
    Build a one-shot answer-mapping prompt that incorporates the dataset's
    question_hint field as an additional context section.

    Args:
        question:      raw question string from the dataset
        options:       list of answer-choice strings
        evidence:      dict returned by a branch's run() after verify()
        question_hint: value of the dataset's `question_hint` column; if None
                       or empty, the hint section is omitted (falls back to
                       the standard prompt format)
        flags:         list of flag strings; if None, read from evidence

    Returns:
        Formatted prompt string ready to send to the LLM.
        SYSTEM_PROMPT should be sent separately as the system message.

    The hint provides the LLM with a high-level clue about what the question
    is testing (e.g. "trend", "periodicity", "anomaly") without revealing the
    answer, acting as a one-shot guide to focus the reasoning on the right
    evidence fields.
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
            {"vision_struct": vision_evidence, "vision_text": vision_text}
        )
    else:
        # strip large arrays from evidence before serialising to keep prompt concise
        _SKIP_KEYS = {"acf_values", "zscore", "iqr_outliers"}
        clean_evidence = {k: v for k, v in evidence.items() if k not in _SKIP_KEYS}

        if branch and subtype and (branch, subtype) in _SUBTYPE_KEYS:
            allowed = _CORE_KEYS | _SUBTYPE_KEYS[(branch, subtype)]
            clean_evidence = {k: v for k, v in clean_evidence.items() if k in allowed}

        evidence_json = json.dumps(clean_evidence, indent=2, default=_json_default)
        artifacts_block = _build_artifacts_block(evidence)
        vision_block = _build_vision_block(evidence)

    if flags:
        flag_list = "\n".join(f"  - `{f}`" for f in flags)
        flags_block = _FLAGS_SECTION.format(flag_list=flag_list)
    else:
        flags_block = _NO_FLAGS_SECTION

    if question_hint and str(question_hint).strip():
        hint_block = _HINT_SECTION.format(hint=str(question_hint).strip())
    else:
        hint_block = ""

    if isinstance(evidence, dict) and "math_evidence" in evidence:
        trend_block = _build_trend_block(evidence["math_evidence"])
    else:
        trend_block = _build_trend_block(evidence)

    task_guidelines = ""
    if branch and subtype and (branch, subtype) in _COGNITIVE_GUIDELINES:
        task_guidelines = _COGNITIVE_GUIDELINES[(branch, subtype)] + "\n\n"

    task_block = (
        f"{task_guidelines}"
        f"Select the best answer. Reply with the option letter on the first line, then one sentence of justification."
    )

    return _ONESHOT_TEMPLATE.format(
        hint_block=hint_block,
        question=question,
        options_block=options_block,
        evidence_json=evidence_json,
        trend_block=trend_block,
        artifacts_block=artifacts_block,
        vision_block=vision_block,
        flags_block=flags_block,
        task_block=task_block,
    )
