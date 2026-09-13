"""
Keyword-based branch candidate generator.

Ported from v2 supervisor._node1_route preset_map. Pure keyword matching — no LLM call.
Takes the primary LLM branch as first candidate, then extends with keyword matches,
deduplicates, and caps at max_candidates.
"""
from __future__ import annotations

_PRESET_MAP: list[tuple[list[str], str]] = [
    (
        ["anomaly", "outlier", "spike", "jump", "change point", "structural break", "unusual"],
        "anomaly",
    ),
    (
        [
            "noise", "stationary", "white noise", "volatility", "heteroskedastic",
            "autocorr", "random walk", "variance level", "ar(", "ma(",
        ],
        "noise",
    ),
    (
        [
            "period", "cycle", "seasonal", "frequency", "waveform", "repeating",
            "sine", "cosine", "oscillat", "harmonic",
        ],
        "periodicity",
    ),
    (
        [
            "trend", "slope", "linear", "exponential", "logarithmic",
            "upward", "downward", "increas", "decreas", "drift", "long-term",
        ],
        "trend",
    ),
    (
        ["causal", "granger", "lead-lag", "lag", "drive", "predict"],
        "causality",
    ),
    (
        ["similar", "distance", "dtw", "offset", "shape", "resemble", "look alike", "compare"],
        "similarity",
    ),
]

_ALL_BRANCHES = {"trend", "periodicity", "anomaly", "noise", "similarity", "causality"}


def get_keyword_candidates(
    question: str,
    primary_branch: str | None,
    max_candidates: int = 3,
) -> list[str]:
    """
    Return an ordered, deduplicated list of branch candidates (≤ max_candidates).

    The primary LLM branch always comes first (if valid). Keyword matches from the
    question text fill remaining slots. Unknown branch names are silently dropped.
    """
    q = (question or "").lower()
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(b: str) -> None:
        if b in _ALL_BRANCHES and b not in seen:
            ordered.append(b)
            seen.add(b)

    if primary_branch:
        _add(primary_branch.lower())

    for keywords, branch in _PRESET_MAP:
        if len(ordered) >= max_candidates:
            break
        if any(k in q for k in keywords):
            _add(branch)

    return ordered[:max_candidates]
