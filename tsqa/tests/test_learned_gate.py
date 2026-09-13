"""Unit tests for the learned should-I-look gate (#1).

Deterministic (no LLM): they exercise the drop-in contract of
``evaluate_visual_trigger_learned`` against the frozen ``gate_policy.json`` and prove the
default additive path is untouched. The accuracy claim itself is the held-out TSExam paired
eval, not these tests.
"""

import json
import os

import pytest

from tsqa.verifier.checks import evaluate_visual_trigger
from tsqa.verifier.learned_gate import evaluate_visual_trigger_learned, _POLICY_PATH


def _qs(branch, subtype=None, scope="global"):
    return {
        "question": "q",
        "branch": branch,
        "scope": scope,
        "subtype": subtype,
        "routing": {},
    }


# evidence that makes the ADDITIVE gate fire on each branch
_NOISE_FIRES = {"branch": "noise", "arch_effect_detected": True, "flags": ["arch_effects"], "series_length": 100}
_TREND_FIRES = {"branch": "trend", "r2": 0.30, "flags": ["low_r2"], "series_length": 100}
_PERIOD_FIRES = {"branch": "periodicity", "dominant_period_fft": 80, "flags": ["fft_unreliable"], "series_length": 100}


@pytest.fixture(scope="module")
def policy():
    assert os.path.exists(_POLICY_PATH), "fit the gate first: fit_learned_gate.py"
    with open(_POLICY_PATH) as f:
        return json.load(f)


def test_policy_partition_matches_preregistration(policy):
    assert set(policy["fire_branches"]) == {"trend", "periodicity"}
    assert set(policy["suppress_branches"]) == {"noise", "similarity", "anomaly", "causality"}
    assert policy["threshold_tau"] == 0.0


def test_suppress_branch_never_looks_even_when_additive_would(policy):
    # additive WOULD fire on this noise row...
    add_trig, _, _ = evaluate_visual_trigger(_NOISE_FIRES, _qs("noise", "variance_level"))
    assert add_trig is True
    # ...but the learned policy suppresses it (noise is a net-negative branch).
    trig, viz, meta = evaluate_visual_trigger_learned(_NOISE_FIRES, _qs("noise", "variance_level"))
    assert trig is False
    assert viz is None
    assert meta["gate"] == "learned"
    assert meta["policy_decision"] == "suppress"
    assert meta["additive_would_fire"] is True  # for the "needless looks avoided" metric


@pytest.mark.parametrize("ev,branch,sub", [
    (_TREND_FIRES, "trend", "direction"),
    (_PERIOD_FIRES, "periodicity", "period_value"),
])
def test_fire_branch_is_byte_identical_to_additive(policy, ev, branch, sub):
    add_trig, add_viz, _ = evaluate_visual_trigger(ev, _qs(branch, sub))
    trig, viz, meta = evaluate_visual_trigger_learned(ev, _qs(branch, sub))
    # On fire branches the learned gate defers verbatim to additive (look + viz_type).
    assert trig == add_trig
    assert viz == add_viz
    assert meta["gate"] == "learned"


def test_learned_lookrate_is_subset_of_additive(policy):
    # Over the three representative rows, learned must look on a (strict) subset of the
    # rows additive looks on — it can only remove looks, never add them.
    rows = [
        (_NOISE_FIRES, _qs("noise", "variance_level")),
        (_TREND_FIRES, _qs("trend", "direction")),
        (_PERIOD_FIRES, _qs("periodicity", "period_value")),
    ]
    add_looks = [evaluate_visual_trigger(ev, qs)[0] for ev, qs in rows]
    learned_looks = [evaluate_visual_trigger_learned(ev, qs)[0] for ev, qs in rows]
    assert all((not l) or a for l, a in zip(learned_looks, add_looks))  # learned ⊆ additive
    assert sum(learned_looks) < sum(add_looks)  # strictly fewer
