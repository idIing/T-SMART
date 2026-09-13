import re


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Matches a bare letter at the start of a line, optionally followed by
# punctuation/whitespace, e.g.:  "A", "A.", "A)", "A:"  "A "
_LETTER_RE = re.compile(r"^\s*([A-Z])[.):\s]", re.MULTILINE | re.IGNORECASE)

# Fallback: explicit answer-letter phrasing in the first part of the response.
# An answer keyword, then OPTIONALLY a linking verb ("is"/"was"), then the letter.
# The linking verb must follow a keyword — a bare "is X" is deliberately NOT matched
# (that caused false positives like "the variance is A"). Allowing it only after a
# keyword recovers the common "the answer is B" phrasing — which otherwise abstains,
# since it is neither letter-first (primary/secondary) nor a lone-letter line
# (quaternary) — without reopening the FP hole (#5).
_LETTER_ANYWHERE_RE = re.compile(
    r"\b(?:answer|option|letter|choose|select|selected)\b"
    r"(?:\s+(?:is|was))?\s*[:=\-]?\s*([A-Z])\b",
    re.IGNORECASE,
)


def parse_answer(raw: str) -> dict:
    """
    Extract a structured answer from raw LLM output.

    The LLM is prompted to output the answer letter on the first line followed
    by one sentence of justification. This parser is intentionally lenient to
    handle minor formatting deviations.

    Args:
        raw: raw string returned by the LLM

    Returns:
        letter:        str  "A".."Z", or None if not parseable
        justification: str  remainder of the response after the letter line
        parse_success: bool False if letter could not be extracted confidently

    Notes:
        - Letter is always returned upper-case.
        - If parse_success is False, the caller (eval runner) should record the
          raw output and treat the prediction as abstained / wrong.
    """
    if not raw or not raw.strip():
        return {"letter": None, "justification": "", "parse_success": False}

    lines = raw.strip().splitlines()
    first_line = lines[0].strip()

    # --- primary: letter at start of first line ---
    m = _LETTER_RE.match(first_line + " ")  # pad so pattern always has trailing char
    if m:
        letter = m.group(1).upper()
        justification = " ".join(lines[1:]).strip()
        return {"letter": letter, "justification": justification, "parse_success": True}

    # --- secondary: lone letter IS the first line (e.g. model outputs just "B") ---
    if len(first_line) == 1 and first_line.upper() in LETTERS:
        letter = first_line.upper()
        justification = " ".join(lines[1:]).strip()
        return {"letter": letter, "justification": justification, "parse_success": True}

    # --- tertiary: scan the first 80 chars of the full response ---
    snippet = raw[:80]
    m2 = _LETTER_ANYWHERE_RE.search(snippet)
    if m2:
        letter = m2.group(1).upper()
        # justification = everything after the matched position
        justification = raw[m2.end():].strip()
        return {"letter": letter, "justification": justification, "parse_success": True}

    # --- quaternary: check the last non-empty line ---
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        m3 = re.search(r"^[^\w]*([A-Z])[^\w]*$", last_line, re.IGNORECASE)
        if m3:
            letter = m3.group(1).upper()
            justification = raw.strip()
            return {"letter": letter, "justification": justification, "parse_success": True}

    # --- failed ---
    return {"letter": None, "justification": raw.strip(), "parse_success": False}
