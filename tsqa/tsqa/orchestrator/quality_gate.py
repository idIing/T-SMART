"""
Quality gate: scores combined evidence quality to decide which branch candidate wins.

Ported from tsmart-v2-pipeline. Pydantic-AI and TSFM references removed.
"""
from dataclasses import dataclass, field
from typing import Optional

_REQUIRED_GLOBAL = {
    "trend": ["slope", "direction", "r2"],
    "periodicity": ["dominant_period_fft", "dominant_freq", "period_reconciled"],
    "anomaly": ["outlier_indices", "outlier_count", "max_deviation"],
    "noise": ["adf_pval", "kpss_pval", "is_stationary", "is_white_noise"],
    "similarity": ["amp_stats_1", "amp_stats_2", "trend_1", "trend_2"],
    "causality": ["best_lag", "pval_12", "pval_21", "direction", "best_corr"],
}

_HARD_FLAGS = {
    "low_r2", "fft_unreliable", "adf_kpss_disagree",
    "granger_not_significant", "weak_correlation",
    "arch_effects", "high_volatility",
}

QUALITY_THRESHOLD = 0.55


@dataclass
class QualityReport:
    score: float
    passed: bool
    reasons: list = field(default_factory=list)
    evidence_completeness: float = 1.0
    flag_penalty: float = 0.0
    math_confidence: float = 1.0
    vision_agreement: float = 0.0


def evaluate_quality(
    stats_evidence: dict,
    vision_evidence: Optional[dict],
    context_evidence: Optional[dict],
    branch: str,
    subtype: Optional[str],
    threshold: float = QUALITY_THRESHOLD,
) -> QualityReport:
    """
    Score combined evidence quality on [0, 1].

    Weighted components:
      0.35 * evidence_completeness  — fraction of required branch fields that are non-None
      0.30 * (1 - flag_penalty)     — -0.12/hard-flag, -0.30 for evidence_incomplete
      0.20 * math_confidence        — R² ≥ 0.75 or ADF/KPSS agreement → 1.0 else scaled
      0.15 * vision_agreement       — +1.0 if vision high-conf parse success
    """
    branch_lower = (branch or "").lower()
    flags = set(stats_evidence.get("flags", []))
    reasons = []

    # --- evidence_completeness ---
    required = _REQUIRED_GLOBAL.get(branch_lower, [])
    if required:
        present = sum(1 for f in required if stats_evidence.get(f) is not None)
        completeness = present / len(required)
    else:
        completeness = 1.0
    if completeness < 1.0:
        reasons.append(f"incomplete_evidence ({completeness:.0%} fields present)")

    # --- flag_penalty ---
    penalty = 0.0
    if "evidence_incomplete" in flags:
        penalty += 0.30
        reasons.append("evidence_incomplete")
    for flag in _HARD_FLAGS:
        if flag in flags:
            penalty += 0.12
            reasons.append(flag)
    flag_penalty = min(penalty, 1.0)

    # --- math_confidence ---
    r2 = stats_evidence.get("r2")
    adf_pval = stats_evidence.get("adf_pval")
    kpss_pval = stats_evidence.get("kpss_pval")
    best_corr = stats_evidence.get("best_corr")

    math_conf = 1.0
    if r2 is not None:
        if r2 >= 0.75:
            math_conf = 1.0
        elif r2 >= 0.50:
            math_conf = 0.7
        else:
            math_conf = 0.4
            reasons.append(f"low_r2={r2:.2f}")
    elif adf_pval is not None and kpss_pval is not None:
        adf_stat = adf_pval < 0.10
        kpss_stat = kpss_pval >= 0.05
        math_conf = 1.0 if adf_stat == kpss_stat else 0.5
    elif best_corr is not None:
        math_conf = min(1.0, abs(best_corr) * 2)

    # --- vision_agreement ---
    vision_agree = 0.0
    if vision_evidence and stats_evidence:
        v_conf = stats_evidence.get("vision_confidence")
        if stats_evidence.get("vision_answer_parse_success"):
            if v_conf == "high":
                vision_agree = 1.0
            elif v_conf == "medium":
                vision_agree = 0.5
            else:
                vision_agree = 0.25

    score = (
        0.35 * completeness
        + 0.30 * (1.0 - flag_penalty)
        + 0.20 * math_conf
        + 0.15 * vision_agree
    )
    score = max(0.0, min(1.0, score))

    return QualityReport(
        score=round(score, 4),
        passed=score >= threshold,
        reasons=reasons,
        evidence_completeness=completeness,
        flag_penalty=flag_penalty,
        math_confidence=math_conf,
        vision_agreement=vision_agree,
    )
