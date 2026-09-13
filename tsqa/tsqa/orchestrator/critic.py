"""
Critic LLM: picks the best branch candidate when quality scores are tied or all fail.

Plain GeminiClient call — no Pydantic-AI. Called at most once per row (on tie/failure).
"""
from __future__ import annotations

import json
import re
from typing import Optional

CRITIC_SYSTEM_PROMPT = """\
You are a Meta-Cognitive Critic for a time-series question-answering pipeline.
You are given evidence dicts produced by two or three different analysis branches
for the same question. Select the branch whose evidence is most complete and most
directly informative for answering the question.

Output ONLY a JSON object with two fields:
  "branch": one of the candidate branch names exactly as given
  "reason": one sentence (≤ 20 words)

No prose, no markdown, no explanation outside the JSON.
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

_COMPACT_KEYS = {
    "branch", "scope", "direction", "slope", "r2", "exp_r2", "log_r2",
    "best_fit_type", "dominant_period_fft", "period_reconciled",
    "seasonal_strength", "is_white_noise", "is_stationary",
    "adf_pval", "kpss_pval", "outlier_count", "max_deviation",
    "best_lag", "direction", "best_corr", "pval_12", "pval_21",
    "flags", "evidence_incomplete",
}


def _compact_evidence(evidence: dict) -> dict:
    return {k: v for k, v in evidence.items() if k in _COMPACT_KEYS}


def build_critic_prompt(
    question: str,
    options: list,
    candidates: list[tuple[str, dict, float]],
) -> str:
    """
    Build the user-turn prompt for the critic.

    candidates: list of (branch_name, evidence_dict, quality_score) tuples,
                sorted best-first by quality score.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_block = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options[:len(letters)])
    )
    cand_blocks = []
    for branch, evidence, score in candidates:
        compact = _compact_evidence(evidence)
        cand_blocks.append(
            f"--- Branch: {branch!r} (quality_score={score:.3f}) ---\n"
            + json.dumps(compact, default=str, indent=2)
        )

    return (
        f"Question:\n{question}\n\n"
        f"Answer choices:\n{options_block}\n\n"
        f"Candidate branch evidence:\n\n"
        + "\n\n".join(cand_blocks)
        + '\n\nSelect the best branch. Output JSON: {"branch": "...", "reason": "..."}'
    )


def parse_critic_selection(
    raw: str,
    candidate_branches: list[str],
) -> Optional[str]:
    """
    Extract the selected branch name from the critic's raw response.

    Returns a branch name (from candidate_branches) or None if unparseable.
    """
    if not raw:
        return None

    text = raw.strip()
    # strip markdown fences
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group()

    try:
        obj = json.loads(text, strict=False)
        branch = str(obj.get("branch", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        branch = ""

    if branch in {b.lower() for b in candidate_branches}:
        for b in candidate_branches:
            if b.lower() == branch:
                return b

    # fallback: scan raw for any candidate branch name
    raw_lower = raw.lower()
    for b in candidate_branches:
        if b.lower() in raw_lower:
            return b

    return None
