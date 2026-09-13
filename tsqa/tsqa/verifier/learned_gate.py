"""Learned "should-I-look" gate (#1) — a calibrated per-branch suppression policy.

Drop-in replacement for ``verifier.checks.evaluate_visual_trigger``, selected via the
config switch ``vision_gate="learned"`` (default ``"additive"`` keeps the hand-tuned gate
byte-identical). Pre-registered in ``mmts_bench/PREREGISTRATION_learned_gate.md`` (committed
*before* any fit) and fitted by ``mmts_bench/scripts/fit_learned_gate.py``.

Design (frozen by the pre-registration):

* **Pre-look features ONLY.** The look decision is made *before* the vision sensor runs, so
  it may use only information available at that point. The fit found ``branch`` to be the
  dominant, benchmark-agnostic signal (sign-stable across the Base/InWild LOSO folds), so
  the policy keys on ``branch``. ``vision_confidence`` is produced by the sensor *after*
  looking and is therefore EXCLUDED by construction.
* **A per-branch SUPPRESSION FILTER over the additive gate.** On the branches the fit
  flagged as net-negative-to-neutral for looking, the policy *suppresses* the look; on the
  net-positive branches it defers byte-identically to the additive gate (same ``trigger``
  AND same ``viz_type``). It can only REMOVE looks, never ADD them — the training label was
  measured only on additive-fired rows, so there is no counterfactual licensing a new look.
  This guarantees look-rate <= additive (the pre-reg H1 mechanism: lower look-rate at no
  accuracy cost).
* **Data-derived partition.** ``fit_learned_gate.py`` estimates the per-branch realized net
  value of looking, ``y = vision_on_correct - vision_off_correct``, and fires branch ``b``
  iff ``E[y|b] > tau`` (``tau`` chosen by leave-one-subset-out CV). The fitted artifact is
  ``gate_policy.json``, hash-pinned before the held-out TSExam evaluation.
"""

import json
import os

from .checks import evaluate_visual_trigger

_POLICY_PATH = os.path.join(os.path.dirname(__file__), "gate_policy.json")
_POLICY_CACHE = None


def _load_policy() -> dict:
    """Load (and cache) the frozen fitted policy artifact.

    Raised lazily so the default ``vision_gate="additive"`` path never touches the
    artifact and stays unaffected when the gate has not been fitted yet.
    """
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        if not os.path.exists(_POLICY_PATH):
            raise RuntimeError(
                f"vision_gate='learned' requested but {_POLICY_PATH} not found. "
                "Fit + freeze it first: python3 mmts_bench/scripts/fit_learned_gate.py"
            )
        with open(_POLICY_PATH) as f:
            _POLICY_CACHE = json.load(f)
    return _POLICY_CACHE


def evaluate_visual_trigger_learned(evidence_dict: dict, query_semantics) -> tuple:
    """Per-branch learned LOOK policy. Same signature/return as the additive gate.

    Returns ``(triggered: bool, viz_type: str | None, meta: dict)``.
    """
    policy = _load_policy()
    suppress = set(policy.get("suppress_branches", []))

    branch = (
        query_semantics.get("branch") if isinstance(query_semantics, dict) else None
    )

    # Always compute the additive decision: on fire/keep branches we defer to it
    # (byte-identical look + viz_type); on suppress branches we still record what it
    # WOULD have done so the held-out eval can count "needless looks avoided".
    add_trigger, add_viz, add_meta = evaluate_visual_trigger(
        evidence_dict, query_semantics
    )
    add_meta = add_meta or {}

    if branch in suppress:
        return (
            False,
            None,
            {
                **add_meta,
                "gate": "learned",
                "triggered": False,
                "viz_type": None,
                "branch": branch,
                "policy_decision": "suppress",
                "additive_would_fire": bool(add_trigger),
                "additive_viz_type": add_viz,
                "policy_version": policy.get("version"),
            },
        )

    # Fire/keep branches (e.g. trend, periodicity) and any unrecognised branch: defer
    # to the additive gate verbatim, so the ONLY difference between the learned and
    # additive arms is the suppressed looks on the net-negative branches.
    return (
        add_trigger,
        add_viz,
        {
            **add_meta,
            "gate": "learned",
            "branch": branch,
            "policy_decision": "keep" if add_trigger else "keep_skip",
            "additive_would_fire": bool(add_trigger),
            "policy_version": policy.get("version"),
        },
    )
