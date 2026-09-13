"""Unit tests for the gate-zero contracts (sample / hooks / artifacts).

Proves the frozen interfaces behave as their docstrings promise — most importantly that
reporting-only labels are *structurally* absent from the runtime object, that the trace
hash is stable under irrelevant perturbation but sensitive to real changes, and that the
recovery arbitration is deterministic and model-call-free.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsqa.eval.sample import (
    AnswerType, OptionType, FORECASTABLE_OPTION_TYPES,
    Channel, BenchmarkSample, ReportingLabels, cluster_keys,
)
from tsqa.eval.hooks import (
    TraceFields, compute_trace_hash,
    select_top2_nocritic, null_forecast_ranker, forecast_eligible,
)
from tsqa.llm.artifacts import normalize_artifacts


class TestSample(unittest.TestCase):
    def _mk(self, n_channels):
        series = [Channel(f"c{i}", np.arange(i * 3, i * 3 + 3)) for i in range(n_channels)]
        return BenchmarkSample(
            id="s", question="q?", options=["a", "b"], answer="a",
            answer_type=AnswerType.MCQ, series=series, category="trend",
        )

    def test_projection_one_channel(self):
        row = self._mk(1).to_runner_row()
        np.testing.assert_allclose(row["ts"], [0, 1, 2])
        self.assertIsNone(row["ts1"])
        self.assertIsNone(row["ts2"])

    def test_projection_two_channels(self):
        row = self._mk(2).to_runner_row()
        self.assertIsNone(row["ts"])
        np.testing.assert_allclose(row["ts1"], [0, 1, 2])
        np.testing.assert_allclose(row["ts2"], [3, 4, 5])

    def test_projection_drops_beyond_two_channels(self):
        row = self._mk(3).to_runner_row()  # third channel must be dropped (documented loss)
        np.testing.assert_allclose(row["ts1"], [0, 1, 2])
        np.testing.assert_allclose(row["ts2"], [3, 4, 5])
        self.assertIsNone(row["ts"])

    def test_channel_coerces_and_validates(self):
        ch = Channel("x", [1, 2, 3])
        self.assertIsInstance(ch.values, np.ndarray)
        self.assertFalse(ch.values.flags.writeable)
        with self.assertRaises(ValueError):
            Channel("x", [1, 2, 3], timestamps=[0, 1])  # length mismatch

    def test_reporting_labels_are_structurally_off_the_runtime_object(self):
        s = self._mk(1)
        # The Break-5 oracle leak is unrepresentable: the runtime sample has no label fields.
        self.assertFalse(hasattr(s, "dimension"))
        self.assertFalse(hasattr(s, "task"))
        self.assertFalse(hasattr(s, "domain"))
        self.assertNotIn("dimension", s.to_runner_row())

    def test_cluster_keys_exclude_outcome_labels(self):
        labels = ReportingLabels(id="s", dimension="Prediction", task="forecast",
                                 domain="finance", series_name="AAPL", template_id="t7")
        keys = cluster_keys(labels)
        self.assertEqual(keys["domain"], "finance")
        self.assertNotIn("dimension", keys)  # outcome label, never a cluster key
        self.assertNotIn("task", keys)


class TestTraceHash(unittest.TestCase):
    def _base(self):
        return TraceFields(
            route="trend", branch="trend", tool_sequence=["ols", "fft"],
            tool_outputs={"r2": 0.123456700, "slope": 1.5}, model_id="gemini-3.1-flash-lite",
            answer="A", verifier_state="ok",
        )

    def test_stable_under_dict_reorder(self):
        a = self._base()
        b = self._base()
        b.tool_outputs = {"slope": 1.5, "r2": 0.123456700}  # reordered
        self.assertEqual(compute_trace_hash(a), compute_trace_hash(b))

    def test_stable_under_volatile_path_change(self):
        a = self._base(); a.tool_outputs = {**a.tool_outputs, "path": "/tmp/a1.png"}
        b = self._base(); b.tool_outputs = {**b.tool_outputs, "path": "/tmp/b2.png"}
        self.assertEqual(compute_trace_hash(a), compute_trace_hash(b))

    def test_stable_under_subprecision_float_noise(self):
        a = self._base()
        b = self._base(); b.tool_outputs = {"r2": 0.123456740, "slope": 1.5}
        self.assertEqual(compute_trace_hash(a), compute_trace_hash(b))  # equal at ndigits=6

    def test_sensitive_to_route_and_answer(self):
        base = compute_trace_hash(self._base())
        r = self._base(); r.route = "noise"
        ans = self._base(); ans.answer = "B"
        self.assertNotEqual(base, compute_trace_hash(r))
        self.assertNotEqual(base, compute_trace_hash(ans))


class TestRecovery(unittest.TestCase):
    def test_deterministic_argmax(self):
        res = select_top2_nocritic(["trend", "noise"], {}, {"trend": 0.4, "noise": 0.9})
        self.assertEqual(res.chosen_branch, "noise")
        self.assertEqual([b for b, _ in res.ranked], ["noise", "trend"])
        self.assertFalse(res.critic_called)

    def test_name_tiebreak_is_order_independent(self):
        s = {"trend": 0.5, "anomaly": 0.5}
        a = select_top2_nocritic(["trend", "anomaly"], {}, s)
        b = select_top2_nocritic(["anomaly", "trend"], {}, s)
        self.assertEqual(a.chosen_branch, b.chosen_branch)
        self.assertEqual(a.chosen_branch, "anomaly")  # alphabetical tie-break

    def test_empty_candidates(self):
        res = select_top2_nocritic([], {}, {})
        self.assertIsNone(res.chosen_branch)
        self.assertFalse(res.critic_called)


class TestForecast(unittest.TestCase):
    def test_default_abstains(self):
        self.assertIsNone(
            null_forecast_ranker("forecast?", OptionType.TRAJECTORY, ["a", "b"], [])
        )

    def test_eligibility_guard(self):
        self.assertTrue(forecast_eligible(OptionType.TRAJECTORY))
        self.assertTrue(forecast_eligible(OptionType.RANGE))
        for ot in (OptionType.CATEGORICAL_TEXT, OptionType.EVENT_LABEL,
                   OptionType.SCALAR, OptionType.MATRIX, OptionType.ORDERING,
                   OptionType.NONE, OptionType.UNKNOWN):
            self.assertFalse(forecast_eligible(ot))
        self.assertEqual(FORECASTABLE_OPTION_TYPES,
                         frozenset({OptionType.TRAJECTORY, OptionType.RANGE}))


class TestArtifacts(unittest.TestCase):
    def test_keeps_wellformed_drops_malformed(self):
        got = normalize_artifacts([
            {"kind": "image", "path": "/tmp/a.png"},
            {"kind": "image"},                 # no path -> dropped
            {"kind": "audio", "path": "/x"},   # wrong kind -> dropped
            "junk",                            # not a dict -> dropped
        ])
        self.assertEqual(got, [{"kind": "image", "path": "/tmp/a.png"}])

    def test_legacy_image_only_when_no_artifacts(self):
        self.assertEqual(normalize_artifacts(image={"path": "/tmp/i.png"}),
                         [{"kind": "image", "path": "/tmp/i.png"}])
        self.assertEqual(normalize_artifacts(image="/tmp/s.png"),
                         [{"kind": "image", "path": "/tmp/s.png"}])
        # artifacts take precedence; legacy image ignored when artifacts present
        self.assertEqual(
            normalize_artifacts([{"kind": "image", "path": "/tmp/a.png"}], image="/tmp/i.png"),
            [{"kind": "image", "path": "/tmp/a.png"}],
        )

    def test_empty(self):
        self.assertEqual(normalize_artifacts(), [])


if __name__ == "__main__":
    unittest.main()
