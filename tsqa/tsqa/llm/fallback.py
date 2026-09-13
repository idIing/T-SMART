_FALLBACK_TEMPLATE = """\
## Question
{question}

## Answer choices
{options_block}

## Note
The automated analysis pipeline could not produce reliable evidence for this \
question (reason: {reason}). Answer using your general knowledge of time series \
analysis. Be conservative — prefer the most defensible option.

Reply with the option letter on the first line, then one sentence of \
justification.\
"""

_FALLBACK_SYSTEM = """\
You are an expert time series analyst answering a multiple-choice question. \
The automated evidence pipeline failed for this question. Reason carefully \
from first principles and pick the single best answer.\
"""


def build_fallback_prompt(question: str, options: list, reason: str = "evidence incomplete") -> str:
    """
    Build a minimal fallback prompt used when the evidence pipeline fails.

    Args:
        question: raw question string
        options:  list of answer-choice strings
        reason:   short description of why the fallback was triggered

    Returns:
        (system_prompt, user_prompt) tuple ready to pass to GeminiClient.generate()
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_block = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options[:len(letters)]))
    user_prompt = _FALLBACK_TEMPLATE.format(
        question=question,
        options_block=options_block,
        reason=reason,
    )
    return _FALLBACK_SYSTEM, user_prompt
