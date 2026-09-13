"""Frozen-config byte-identity guard — **Contract D** (gate-zero).

The whole generalization narrative rests on the frozen Gemini configs (``baseline``,
``nu_ad_fix``, ``learned_gate``) producing identical decisions before and after the Wave-0
overhaul. "Existing tests stay green" does NOT prove that — it proves no crash. This guard
proves *output identity*.

Method: a fully-canned :class:`_FrozenClient` makes every model call deterministic, so
``run_pipeline`` becomes reproducible end-to-end (router + numeric head + the additive AND
forced AND suppressed vision paths). We snapshot a behavioral *fingerprint* of each frozen
config on fixed rows into ``tests/golden/frozen_pipeline_fingerprints.json``. Any Wave-0
change that shifts a frozen decision fails this test loudly.

Regenerate the golden ONLY when a frozen-path change is intended (it never should be in
Wave 0):  ``FREEZE_UPDATE=1 python -m pytest tests/test_frozen_identity.py``
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "frozen_pipeline_fingerprints.json")


# ---------------------------------------------------------------------------
# Deterministic canned client — every call resolves to a constant.
# ---------------------------------------------------------------------------
class _FrozenClient:
    """Canned, side-effect-free client. Dispatch mirrors the proven heuristic in
    ``test_runner_v2._MockClient`` (router by "branch"/"route" in the system prompt),
    plus a vision-answer-suggestion branch ("analyst") so the forced-vision path resolves
    to a stable letter."""

    def __init__(self, routing_json, fallback_branch="trend"):
        self._routing = routing_json
        self._fallback_branch = fallback_branch

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        s = (system_prompt or "").lower()
        if "branch" in s or "route" in s:                 # router
            return self._routing
        if "analyst" in s:                                # vision answer-suggestion -> letter
            return "B\nConfidence: high\nvisual cue"
        if "critic" in s or "meta-cognitive" in s:        # multi-branch critic
            return json.dumps({"branch": self._fallback_branch, "reason": "x"})
        if artifacts or "visual" in (user_prompt or "").lower():   # vision sensor -> topology
            return json.dumps({"summary": "synthetic series", "topology": "smooth", "confidence": "high"})
        return "A\nConfidence: high\nreason"              # final answer


# ---------------------------------------------------------------------------
# Fixed rows (deterministic series via seeded RNG)
# ---------------------------------------------------------------------------
def _trend_row():
    rng = np.random.default_rng(42)
    ts = np.linspace(0, 10, 200) + rng.normal(0, 0.2, 200)
    return {"id": "f_trend", "category": "trend",
            "question": "What is the direction of the trend in this series?",
            "options": ["increasing", "decreasing", "stationary", "oscillating"],
            "answer": "increasing", "ts": ts.tolist()}


def _similarity_row():
    rng = np.random.default_rng(7)
    base = np.sin(np.linspace(0, 4 * np.pi, 150))
    return {"id": "f_sim", "category": "similarity",
            "question": "Are these two series similar in shape?",
            "options": ["yes", "no", "partially", "unknown"], "answer": "no",
            "ts1": (base + rng.normal(0, 0.05, 150)).tolist(),
            "ts2": (base[::-1] + rng.normal(0, 0.05, 150)).tolist()}


def _numeric_row():
    rng = np.random.default_rng(11)
    ts = rng.normal(5, 2, 128)
    return {"id": "f_num", "category": "basic analysis",
            "question": "Based on Time Series 1, what is the standard deviation of the values?",
            "options": [], "answer": "2.0", "ts": ts.tolist()}


def _anomaly_row():
    rng = np.random.default_rng(3)
    ts = rng.normal(0, 1, 200)
    ts[100] = 12.0
    return {"id": "f_anom", "category": "anomaly",
            "question": "Is there a point anomaly in this series?",
            "options": ["yes", "no", "maybe", "unknown"], "answer": "yes", "ts": ts.tolist()}


def _noise_row():
    rng = np.random.default_rng(5)
    ts = rng.normal(0, 1, 200)
    return {"id": "f_noise", "category": "noise",
            "question": "Is this series stationary?",
            "options": ["stationary", "non-stationary", "trend-stationary", "unknown"],
            "answer": "stationary", "ts": ts.tolist()}


_NU_AD_FIX = {"no_vision_branches": ["noise"], "forced_vision_branches": ["anomaly"]}

# (name, row builder, routing branch json, run_pipeline kwargs) — covers router, numeric
# head, additive vision, forced vision (anomaly), and suppressed vision (noise).
_CASES = [
    ("baseline_trend",      _trend_row,      '{"branch":"trend","scope":"global","subtype":"direction"}',        {}),
    ("baseline_similarity", _similarity_row, '{"branch":"similarity","scope":"global","subtype":"shape"}',       {}),
    ("baseline_numeric_std", _numeric_row,   '{"branch":"trend","scope":"global","subtype":"direction"}',        {}),
    ("nu_ad_fix_anomaly",   _anomaly_row,    '{"branch":"anomaly","scope":"global","subtype":"point_anomaly"}',  _NU_AD_FIX),
    ("nu_ad_fix_noise",     _noise_row,      '{"branch":"noise","scope":"global","subtype":"stationarity"}',     _NU_AD_FIX),
]


def _fingerprint(res: dict) -> dict:
    """Behaviorally meaningful, deterministic projection of a run_pipeline result."""
    routing = res.get("routing") or {}
    pv = res.get("predicted_value")
    return {
        "predicted_letter": res.get("predicted_letter"),
        "branch_used": res.get("branch_used"),
        "subtype_used": res.get("subtype_used"),
        "used_fallback": bool(res.get("used_fallback")),
        "vision_used": bool(res.get("vision_used")),
        "vision_letter": res.get("vision_letter"),
        "expected_schema": res.get("expected_schema"),
        "predicted_value": (round(float(pv), 6) if isinstance(pv, (int, float)) and not isinstance(pv, bool) else pv),
        "numeric_quantity": res.get("numeric_quantity"),
        "numeric_note": res.get("numeric_note"),
        "flags": sorted(res.get("flags") or []),
        "parse_success": bool(res.get("parse_success")),
        "routing_branch": routing.get("branch"),
    }


def _run_all_fingerprints() -> dict:
    from tsqa.eval.runner import run_pipeline
    out = {}
    for name, row_fn, routing, kwargs in _CASES:
        client = _FrozenClient(routing)
        res = run_pipeline(row_fn(), client, **kwargs)
        out[name] = _fingerprint(res)
    return out


class TestFrozenIdentity(unittest.TestCase):
    def test_frozen_pipeline_fingerprints(self):
        actual = _run_all_fingerprints()

        if os.environ.get("FREEZE_UPDATE"):
            os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
            with open(GOLDEN, "w") as f:
                json.dump(actual, f, indent=2, sort_keys=True)
            self.skipTest(f"golden regenerated at {GOLDEN}")

        self.assertTrue(
            os.path.exists(GOLDEN),
            "golden missing — generate it once with FREEZE_UPDATE=1 (frozen-path baseline)",
        )
        with open(GOLDEN) as f:
            golden = json.load(f)
        self.assertEqual(
            actual, golden,
            "FROZEN PIPELINE OUTPUT DRIFTED — a Wave-0 change perturbed a frozen decision. "
            "If the change to the frozen path was intentional, regenerate with FREEZE_UPDATE=1.",
        )


if __name__ == "__main__":
    unittest.main()
