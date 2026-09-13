"""
Smoke tests for the v2 runner features:
  - single-branch path (multi_branch=False)
  - multi-branch path (multi_branch=True) with quality gate and critic
  - SA forced-vision path (enable_sa_vision=True, branch=similarity)
  - run_dataset concurrent path (max_workers=2)

All LLM calls are mocked. Numpy/scipy branch computation runs for real,
so the test also validates that the branch/verifier chain doesn't explode
on synthetic data.
"""
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import unittest
from unittest.mock import MagicMock, patch


def _load_flag_diagnostics():
    """Load tsexam/flag_diagnostics.py by path (tsexam/ is a script dir, not
    an importable package)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tsexam", "flag_diagnostics.py",
    )
    spec = importlib.util.spec_from_file_location("flag_diagnostics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# Minimal mock LLM client
# ---------------------------------------------------------------------------

class _MockClient:
    """Returns canned responses without making any API calls."""

    def __init__(self, routing_json=None, answer_letter="A"):
        self._routing_json = routing_json or '{"branch":"trend","scope":"global","subtype":"direction"}'
        self._answer_letter = answer_letter
        self.calls = []

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        # Router call: return routing JSON
        if "branch" in system_prompt.lower() or "route" in system_prompt.lower():
            return self._routing_json
        # Critic call: return first candidate
        if "critic" in system_prompt.lower() or "meta-cognitive" in system_prompt.lower():
            return '{"branch": "trend", "reason": "best evidence"}'
        # Vision calls
        if artifacts or "visual" in user_prompt.lower():
            return json.dumps({
                "summary": "synthetic series with upward trend",
                "topology": "smooth",
                "confidence": "high",
            })
        # Answer call
        return f"{self._answer_letter}\nConfidence: high\nThe trend is upward."


# ---------------------------------------------------------------------------
# Synthetic dataset rows
# ---------------------------------------------------------------------------

def _trend_row():
    rng = np.random.default_rng(42)
    ts = np.linspace(0, 10, 200) + rng.normal(0, 0.2, 200)
    return {
        "id": "r001",
        "category": "trend",
        "question": "What is the direction of the trend in this series?",
        "options": ["increasing", "decreasing", "stationary", "oscillating"],
        "answer": "increasing",
        "ts": ts.tolist(),
    }


def _similarity_row():
    rng = np.random.default_rng(7)
    base = np.sin(np.linspace(0, 4 * np.pi, 150))
    ts1 = base + rng.normal(0, 0.05, 150)
    ts2 = base[::-1] + rng.normal(0, 0.05, 150)
    return {
        "id": "r002",
        "category": "similarity",
        "question": "Are these two series similar in shape?",
        "options": ["yes", "no", "partially", "unknown"],
        "answer": "no",
        "ts1": ts1.tolist(),
        "ts2": ts2.tolist(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Import lazily to avoid torch dependency at module level
def _import_runner():
    from tsqa.eval.runner import run_pipeline, run_dataset
    return run_pipeline, run_dataset


class TestSingleBranchPath(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_trend_single_branch_returns_letter(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
            answer_letter="A",
        )
        row = _trend_row()
        res = self.run_pipeline(row, client, multi_branch=False, enable_sa_vision=False)
        self.assertIn(res["predicted_letter"], list("ABCD") + [None])
        self.assertFalse(res["multi_branch_used"])
        self.assertFalse(res["critic_called"])
        self.assertIsNone(res["quality_score"])

    def test_single_branch_sets_branch_used(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        res = self.run_pipeline(_trend_row(), client)
        self.assertEqual(res["branch_used"], "trend")

    def test_no_series_data_falls_back(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        row = dict(_trend_row())
        del row["ts"]
        res = self.run_pipeline(row, client)
        self.assertTrue(res["used_fallback"])

    def test_routing_failure_falls_back(self):
        client = _MockClient(routing_json="not json at all")
        res = self.run_pipeline(_trend_row(), client)
        self.assertTrue(res["used_fallback"])


class TestMultiBranchPath(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_multi_branch_flag_set(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        res = self.run_pipeline(_trend_row(), client, multi_branch=True)
        self.assertTrue(res["multi_branch_used"])
        self.assertIsNotNone(res["quality_score"])
        self.assertIsInstance(res["branch_candidates"], list)
        self.assertGreater(len(res["branch_candidates"]), 0)

    def test_multi_branch_branch_used_is_valid(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        from tsqa.eval.runner import ALL_BRANCHES
        res = self.run_pipeline(_trend_row(), client, multi_branch=True)
        if not res["used_fallback"]:
            self.assertIn(res["branch_used"], ALL_BRANCHES)

    def test_multi_branch_quality_score_in_range(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        res = self.run_pipeline(_trend_row(), client, multi_branch=True)
        if res["quality_score"] is not None:
            self.assertGreaterEqual(res["quality_score"], 0.0)
            self.assertLessEqual(res["quality_score"], 1.0)

    def test_multi_branch_raw_ev_guard_not_accessed_when_false(self):
        """Guard: raw_ev must NEVER be accessed when multi_branch=False."""
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        # This would NameError if the short-circuit guard is broken
        res = self.run_pipeline(_trend_row(), client, multi_branch=False)
        self.assertFalse(res["multi_branch_used"])


class TestSAVisionPath(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_sa_vision_stacked_injects_vision_fields(self):
        client = _MockClient(
            routing_json='{"branch":"similarity","scope":"global","subtype":"shape"}',
            answer_letter="B",
        )
        row = _similarity_row()
        res = self.run_pipeline(row, client, enable_sa_vision=True, sa_viz_mode="stacked")
        ev = res.get("evidence") or {}
        # SA path should have set artifacts or vision_tool
        self.assertTrue(
            ev.get("artifacts") or ev.get("vision_tool") or ev.get("vision_error"),
            "SA vision path should have attempted artifact generation"
        )

    def test_sa_vision_off_pipeline_completes(self):
        """With enable_sa_vision=False the pipeline must complete and not raise."""
        client = _MockClient(
            routing_json='{"branch":"similarity","scope":"global","subtype":"shape"}',
        )
        row = _similarity_row()
        res = self.run_pipeline(row, client, enable_sa_vision=False)
        # Pipeline must always return a predicted_letter (or fallback)
        self.assertIn("predicted_letter", res)
        # SA vision flag must not be set when enable_sa_vision=False
        # (verifier may still set artifacts via its own trigger — that's fine)
        self.assertFalse(res.get("multi_branch_used"))

    def test_sa_vision_overlay_line_mode(self):
        client = _MockClient(
            routing_json='{"branch":"similarity","scope":"global","subtype":"shape"}',
        )
        row = _similarity_row()
        res = self.run_pipeline(row, client, enable_sa_vision=True, sa_viz_mode="overlay_line")
        ev = res.get("evidence") or {}
        self.assertTrue(ev.get("artifacts") or ev.get("vision_tool") or ev.get("vision_error"))

    def test_sa_vision_not_triggered_for_non_similarity_branch(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        row = _trend_row()
        res = self.run_pipeline(row, client, enable_sa_vision=True, sa_viz_mode="stacked")
        ev = res.get("evidence") or {}
        # SA vision only fires for similarity branch
        self.assertFalse(bool(ev.get("artifacts")))


class TestRunDatasetConcurrency(unittest.TestCase):
    def setUp(self):
        _, self.run_dataset = _import_runner()

    def _make_rows(self, n=4):
        return [_trend_row() for _ in range(n)]

    def test_serial_returns_n_results(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        rows = self._make_rows(3)
        results = self.run_dataset(rows, client, max_workers=1, show_progress=False)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("predicted_letter", r)

    def test_concurrent_returns_same_count(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        rows = self._make_rows(4)
        results = self.run_dataset(rows, client, max_workers=2, show_progress=False)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIsNotNone(r)
            self.assertIn("predicted_letter", r)

    def test_concurrent_order_preserved(self):
        """Results list must be index-aligned with input, not arrival order."""
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        rows = [dict(_trend_row(), id=f"row_{i}") for i in range(6)]
        results = self.run_dataset(rows, client, max_workers=3, show_progress=False)
        self.assertEqual(len(results), 6)
        for i, r in enumerate(results):
            self.assertEqual(r.get("id"), f"row_{i}", f"Order mismatch at index {i}")

    def test_max_rows_respected_in_concurrent_mode(self):
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        rows = self._make_rows(10)
        results = self.run_dataset(rows, client, max_rows=3, max_workers=2, show_progress=False)
        self.assertEqual(len(results), 3)

    def test_concurrent_exception_per_row_does_not_kill_batch(self):
        """A single row error must not crash the whole batch."""
        good = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
        )
        rows = [_trend_row() for _ in range(4)]
        # Corrupt the second row to force a KeyError in pipeline
        rows[1] = {"ts": [1, 2, 3]}  # missing 'question'

        results = self.run_dataset(rows, good, max_workers=2, show_progress=False)
        self.assertEqual(len(results), 4)
        # Row 1 should have an error key
        self.assertIn("error", results[1])
        # Other rows should have predicted_letter
        for i in (0, 2, 3):
            self.assertIn("predicted_letter", results[i])


# ---------------------------------------------------------------------------
# Numeric head (#4a) + the MCQ isolation guarantee
# ---------------------------------------------------------------------------

def _numeric_std_row():
    rng = np.random.default_rng(3)
    ts = rng.normal(50, 7, 250)
    return {
        "id": "n001",
        "category": "basic analysis",
        "question": ("Here is a list of values in a Time Series 1. Based on the "
                     "Time Series 1 data provided, what is the standard deviation "
                     "of the sequence?"),
        "options": [],            # no options -> free-response (numerical)
        "answer": f"{float(np.std(ts)):.2f}",
        "ts": ts.tolist(),
    }, ts


def _numeric_unknown_row():
    return {
        "id": "n002",
        "category": "basic analysis",
        "question": ("Based on the Time Series 1 data provided, what is the "
                     "spectral entropy of the quantum manifold?"),  # no template
        "options": [],
        "answer": "42",
        "ts": [1.0, 2, 3, 4, 5, 6, 7, 8],
    }


def _categorical_row():
    rng = np.random.default_rng(5)
    ts = rng.normal(0, 1, 200)   # ~stationary white noise
    return {
        "id": "c001",
        "category": "stationarity",
        "question": ("Based on the Time Series 1 data provided, is the series "
                     "stationary or non-stationary?"),
        "options": [],
        "answer": "stationary",
        "ts": ts.tolist(),
    }


class TestNumericHead(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_mcq_row_is_isolated_from_numeric_head(self):
        """The key guardrail: an MCQ row (options present) keeps
        expected_schema=='mcq' and NEVER enters the numeric head — predicted_value
        and the numeric_* metadata stay None, and it still answers via a letter."""
        client = _MockClient(
            routing_json='{"branch":"trend","scope":"global","subtype":"direction"}',
            answer_letter="A",
        )
        res = self.run_pipeline(_trend_row(), client)
        self.assertEqual(res["expected_schema"], "mcq")
        self.assertIsNone(res["predicted_value"])
        self.assertIsNone(res["numeric_quantity"])      # numeric head never ran
        self.assertIsNone(res["numeric_note"])
        self.assertIn(res["predicted_letter"], list("ABCD") + [None])  # MCQ path

    def test_numerical_schema_detected_when_no_options(self):
        row, _ = _numeric_std_row()
        client = _MockClient(
            routing_json='{"branch":"noise","scope":"global","subtype":"variance_level"}',
        )
        res = self.run_pipeline(row, client)
        self.assertEqual(res["expected_schema"], "numerical")

    def test_numerical_value_equals_tool_value(self):
        """Tools own the value: predicted_value must equal numpy's std exactly,
        the MCQ interpreter is bypassed, and vision is never used."""
        row, ts = _numeric_std_row()
        client = _MockClient(
            routing_json='{"branch":"noise","scope":"global","subtype":"variance_level"}',
        )
        res = self.run_pipeline(row, client)
        self.assertIsNotNone(res["predicted_value"])
        self.assertAlmostEqual(res["predicted_value"], float(np.std(ts)), places=6)
        self.assertIsNone(res["predicted_letter"])     # no letter on numeric path
        self.assertFalse(res["vision_used"])            # vision skipped
        self.assertEqual(res["numeric_quantity"], "std")

    def test_numerical_abstains_on_unknown_quantity(self):
        """An unrecognised quantity is an honest miss (None), never a guess."""
        client = _MockClient(
            routing_json='{"branch":"noise","scope":"global","subtype":"variance_level"}',
        )
        res = self.run_pipeline(_numeric_unknown_row(), client)
        self.assertEqual(res["expected_schema"], "numerical")
        self.assertIsNone(res["predicted_value"])
        self.assertEqual(res["numeric_note"], "no_quantity_match")

    def test_categorical_maps_stationarity_determination(self):
        client = _MockClient(
            routing_json='{"branch":"noise","scope":"global","subtype":"stationarity"}',
        )
        res = self.run_pipeline(_categorical_row(), client)
        self.assertEqual(res["expected_schema"], "categorical")
        self.assertIn(res["predicted_value"], {"stationary", "non-stationary"})
        self.assertIsNone(res["predicted_letter"])


# ---------------------------------------------------------------------------
# llm_numeric COUNTERFACTUAL arm — the model-computes control for free-response
# (mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md)
# ---------------------------------------------------------------------------

class _NumericProbeClient:
    """Stub for the llm_numeric arm: returns a NUMBER for the 'calculator' system
    prompt, a routing JSON for the router, and a letter for the answer call.
    Records every system prompt so a test can assert which path executed."""

    def __init__(self, number="3.14159", answer_letter="B",
                 routing_json='{"branch":"noise","scope":"global","subtype":"variance_level"}'):
        self.number = number
        self._answer_letter = answer_letter
        self._routing_json = routing_json
        self.systems = []

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        self.systems.append(system_prompt)
        if "calculator" in system_prompt.lower():        # build_llm_numeric_prompt
            return f"{self.number}\nThat is the value."
        if "branch" in system_prompt.lower() or "route" in system_prompt.lower():
            return self._routing_json
        return f"{self._answer_letter}\none sentence."


class TestLLMNumericArm(unittest.TestCase):
    """The free-response counterfactual: the model computes the number directly,
    bypassing the deterministic numeric head. Paired @10% against the head, the gap
    is the 'tools are load-bearing on a strong backbone' measurement."""

    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_numerical_row_uses_model_not_head(self):
        """llm_numeric=True on a numerical row: the MODEL's number becomes
        predicted_value; the numeric HEAD is bypassed (numeric_quantity stays None),
        and exactly ONE LLM call is made (no router)."""
        row, _ = _numeric_std_row()
        client = _NumericProbeClient(number="123.45")
        res = self.run_pipeline(row, client, llm_numeric=True)
        self.assertEqual(res["branch_used"], "llm_numeric")
        self.assertEqual(res["expected_schema"], "numerical")
        self.assertAlmostEqual(res["predicted_value"], 123.45, places=4)
        self.assertTrue(res["parse_success"])
        self.assertIsNone(res["numeric_quantity"])   # head metadata absent => bypassed
        self.assertFalse(res["used_head"])
        self.assertFalse(res["vision_used"])
        self.assertEqual(len(client.systems), 1)      # one call, no router
        self.assertIn("calculator", client.systems[0].lower())

    def test_mcq_row_falls_back_to_letter_path(self):
        """llm_numeric=True on an MCQ row routes to the llm_only letter path:
        a letter is produced, predicted_value stays None, the numeric calculator
        prompt is never issued."""
        client = _NumericProbeClient(answer_letter="C")
        res = self.run_pipeline(_trend_row(), client, llm_numeric=True)
        self.assertEqual(res["expected_schema"], "mcq")
        self.assertEqual(res["predicted_letter"], "C")
        self.assertIsNone(res["predicted_value"])
        self.assertNotIn("calculator", " ".join(client.systems).lower())

    def test_unparseable_model_output_abstains(self):
        """A non-numeric model reply is an honest miss (None) tagged llm_unparseable
        — never a fabricated number."""
        row, _ = _numeric_std_row()
        client = _NumericProbeClient(number="unknown")
        res = self.run_pipeline(row, client, llm_numeric=True)
        self.assertIsNone(res["predicted_value"])
        self.assertFalse(res["parse_success"])
        self.assertEqual(res["numeric_note"], "llm_unparseable")

    def test_off_is_byte_identical_head_path(self):
        """With llm_numeric=False (default) a numerical row still flows through the
        deterministic numeric HEAD — predicted_value equals numpy std exactly and
        numeric_quantity=='std'. The new kwarg's default disturbs nothing."""
        row, ts = _numeric_std_row()
        client = _NumericProbeClient()
        res = self.run_pipeline(row, client)              # llm_numeric defaults False
        self.assertEqual(res["expected_schema"], "numerical")
        self.assertAlmostEqual(res["predicted_value"], float(np.std(ts)), places=6)
        self.assertEqual(res["numeric_quantity"], "std")  # head ran, not the model
        self.assertNotEqual(res["branch_used"], "llm_numeric")


# ---------------------------------------------------------------------------
# Three-arm numeric PARSER study (A0 keyword / A1 deterministic / A2 llm_fallback)
# (mmts_bench/PREREGISTRATION_numeric_parser.md, log/012)
# ---------------------------------------------------------------------------

class _PlannerStub:
    """Router prompt → routing JSON; numeric-planner prompt → a canned PLAN; else a
    letter. Records system prompts so a test can assert whether the planner fired."""

    def __init__(self, plan='{"quantity":"std","param":null}'):
        self.plan = plan
        self.systems = []

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        self.systems.append(system_prompt)
        if "into a plan" in system_prompt.lower():          # numeric planner prompt
            return "plan:\n" + self.plan
        if "branch" in system_prompt.lower() or "route" in system_prompt.lower():
            return '{"branch":"noise","scope":"global","subtype":"variance_level"}'
        return "A\njustification."

    def planner_fired(self):
        return any("into a plan" in s.lower() for s in self.systems)


def _numeric_row(question):
    rng = np.random.default_rng(0)
    ts = rng.normal(50, 7, 64)
    return ({"id": "np01", "category": "basic analysis", "question": question,
             "options": [], "answer": "0", "ts": ts.tolist()}, ts)


class TestNumericParser(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_a1_parses_composition_without_llm(self):
        """A1 deterministic resolves 'mean minus variance' via the grammar — no LLM."""
        from tsqa.eval.numeric_head import parse_plan_deterministic, evaluate_plan
        row, ts = _numeric_row("Based on Time Series 1, what is the mean value minus "
                               "the variance of the sequence?")
        plan = parse_plan_deterministic(row["question"])
        self.assertEqual(plan["op"], "subtract")
        v, _ = evaluate_plan(plan, ts, None, None, {})
        self.assertAlmostEqual(v, float(np.mean(ts)) - float(np.var(ts)), places=6)

    def test_a1_synonyms_and_typos(self):
        from tsqa.eval.numeric_head import parse_plan_deterministic, infer_numeric_quantity
        self.assertEqual(parse_plan_deterministic("the spread of the series")["quantity"], "std")
        self.assertEqual(parse_plan_deterministic("the varaince of the series")["quantity"], "var")
        # A0 has neither synonym nor typo handling (the controlled brittle baseline)
        self.assertIsNone(infer_numeric_quantity("the spread of the series"))

    def test_keyword_arm_byte_identical(self):
        """numeric_parser='keyword' == the legacy head: same value, planner never fires."""
        row, ts = _numeric_row("Based on Time Series 1, what is the standard deviation?")
        c = _PlannerStub()
        res = self.run_pipeline(row, c, numeric_parser="keyword")
        self.assertAlmostEqual(res["predicted_value"], float(np.std(ts)), places=6)
        self.assertEqual(res["numeric_quantity"], "std")
        self.assertFalse(res["numeric_llm_fired"])
        self.assertFalse(c.planner_fired())

    def test_a2_planner_skipped_on_clean_row(self):
        """A2 never calls the LLM when A1 resolves (non-regression + efficiency)."""
        row, ts = _numeric_row("Based on Time Series 1, what is the standard deviation?")
        c = _PlannerStub()
        res = self.run_pipeline(row, c, numeric_parser="llm_fallback")
        self.assertAlmostEqual(res["predicted_value"], float(np.std(ts)), places=6)
        self.assertFalse(res["numeric_llm_fired"])
        self.assertFalse(c.planner_fired())

    def test_a2_planner_fires_on_abstain_emits_plan_not_number(self):
        """When A1 abstains, A2 fires; the LLM returns a PLAN and the TOOL computes."""
        row, ts = _numeric_row("Based on Time Series 1, how concentrated around the "
                               "center is the sequence?")
        c = _PlannerStub(plan='{"quantity":"std"}')
        res = self.run_pipeline(row, c, numeric_parser="llm_fallback")
        self.assertTrue(c.planner_fired())
        self.assertTrue(res["numeric_llm_fired"])
        self.assertAlmostEqual(res["predicted_value"], float(np.std(ts)), places=6)

    def test_a2_rejects_out_of_registry_plan(self):
        """An out-of-registry op the LLM hallucinates is rejected → abstain (no fabrication)."""
        row, _ = _numeric_row("Based on Time Series 1, how concentrated is the sequence?")
        c = _PlannerStub(plan='{"op":"power","args":[{"quantity":"mean"},{"quantity":"std"}]}')
        res = self.run_pipeline(row, c, numeric_parser="llm_fallback")
        self.assertIsNone(res["predicted_value"])
        self.assertEqual(res["numeric_note"], "no_quantity_match")

    def test_validate_plan_unit(self):
        from tsqa.eval.numeric_head import validate_plan
        self.assertTrue(validate_plan({"quantity": "std"}))
        self.assertTrue(validate_plan({"op": "subtract",
                                       "args": [{"quantity": "mean"}, {"quantity": "var"}]}))
        self.assertFalse(validate_plan({"quantity": "bogus"}))
        self.assertFalse(validate_plan({"op": "power", "args": [{"quantity": "mean"}]}))
        self.assertFalse(validate_plan({"op": "subtract", "args": [{"quantity": "mean"}]}))  # arity


# ---------------------------------------------------------------------------
# Self-Consistency control arm (#4) — byte-identity off + sampling-on behavior
# ---------------------------------------------------------------------------

def _incomplete_trend_branch(series, scope="global"):
    """Trend evidence missing required fields ⇒ verify() adds evidence_incomplete."""
    return {"branch": "trend", "scope": scope, "slope": None, "direction": None, "r2": None}


class _SCClient:
    """Records the temperature of every answer-LLM call and returns a scripted
    sequence of answer letters so a majority vote is testable. Router calls return
    a fixed trend route at temp 0 (unchanged)."""

    model_name = "sc-mock"

    def __init__(self, answer_letters=("A",)):
        self._answer_letters = list(answer_letters)
        self._answer_idx = 0
        self.answer_temps = []
        self.router_temps = []

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        s = (system_prompt or "").lower()
        temp = kwargs.get("temperature")
        if "branch" in s or "route" in s:
            self.router_temps.append(temp)
            return '{"branch":"trend","scope":"global","subtype":"direction"}'
        # Any non-router text call here is the answer / fallback step.
        self.answer_temps.append(temp)
        letter = self._answer_letters[min(self._answer_idx, len(self._answer_letters) - 1)]
        self._answer_idx += 1
        return f"{letter}\nConfidence: high\nreason"


class TestSelfConsistency(unittest.TestCase):
    def setUp(self):
        self.run_pipeline, _ = _import_runner()

    def test_non_flagged_row_byte_identical_to_baseline(self):
        """THE byte-identity guardrail: on a row WITHOUT evidence_incomplete the
        self_consistency switch must not change the prediction, must not flag the
        row as sampled, and must issue exactly one (temp-0) answer call — i.e. it
        is byte-identical to baseline."""
        # Suppress vision on trend so the only answer-LLM call is the final
        # interpreter — isolating the SC path. The suppression is applied to BOTH
        # arms, so the comparison stays matched (it is not what's under test).
        # Both arms use the same client class; the ONLY difference is the
        # self_consistency switch, so any prediction drift is attributable to it.
        no_vis = ["trend"]
        base = self.run_pipeline(
            _trend_row(), _SCClient(answer_letters=("A",)), no_vision_branches=no_vis,
        )

        sc_client = _SCClient(answer_letters=("A",))
        treat = self.run_pipeline(
            _trend_row(), sc_client, no_vision_branches=no_vis,
            self_consistency=5, self_consistency_temperature=0.7,
        )

        # No evidence_incomplete on a clean trend row ⇒ control dark.
        self.assertNotIn("evidence_incomplete", treat.get("flags") or [])
        self.assertFalse(treat["sc_sampled"])
        self.assertEqual(treat["sc_k"], 0)
        # Prediction + mechanism fields identical to baseline.
        self.assertEqual(treat["predicted_letter"], base["predicted_letter"])
        self.assertEqual(treat["branch_used"], base["branch_used"])
        self.assertEqual(treat["used_fallback"], base["used_fallback"])
        # Exactly ONE answer call, and it was made at the default temp (0/None),
        # never the sampling temp 0.7.
        self.assertEqual(len(sc_client.answer_temps), 1)
        self.assertNotIn(0.7, sc_client.answer_temps)

    def test_samples_k_times_and_majority_votes_on_flagged_row(self):
        """On an evidence_incomplete row the control samples the answer LLM k
        times at temp>0 and majority-votes. Scripted letters A,B,B,B,A ⇒ B wins;
        every sample is issued at the configured temperature 0.7."""
        sc_client = _SCClient(answer_letters=("A", "B", "B", "B", "A"))
        with patch.dict(
            __import__("tsqa.eval.runner", fromlist=["_SINGLE_BRANCHES"])._SINGLE_BRANCHES,
            {"trend": _incomplete_trend_branch},
        ):
            # Vision off ⇒ the evidence_incomplete row reaches the fallback path,
            # where SC sampling happens; this keeps the answer-call count == k.
            res = self.run_pipeline(
                _trend_row(), sc_client, no_vision_branches=["trend"],
                self_consistency=5, self_consistency_temperature=0.7,
            )
        self.assertTrue(res["sc_sampled"])
        self.assertEqual(res["sc_k"], 5)
        self.assertEqual(len(sc_client.answer_temps), 5)        # k samples
        self.assertEqual(set(sc_client.answer_temps), {0.7})    # all at temp 0.7
        self.assertEqual(res["predicted_letter"], "B")          # majority vote

    def test_majority_vote_tie_break_is_alphabetic(self):
        """A 2-2 tie (B,B,C,C) resolves to the alphabetically-smaller letter (B),
        the documented deterministic tie-break."""
        from tsqa.eval.runner import _sc_majority_vote
        self.assertEqual(_sc_majority_vote(["B", "B", "C", "C"]), "B")
        self.assertEqual(_sc_majority_vote(["C", "A", "C", "A"]), "A")
        self.assertEqual(_sc_majority_vote(["D"]), "D")
        self.assertIsNone(_sc_majority_vote([None, None]))       # all parse-failed
        self.assertEqual(_sc_majority_vote([None, "A", None]), "A")  # drop failures


# ---------------------------------------------------------------------------
# flag_diagnostics numeric correctness on a small fixture
# ---------------------------------------------------------------------------

class TestFlagDiagnostics(unittest.TestCase):
    def _rows(self):
        # 5 scored rows + 1 abstention (correct=None, must be excluded).
        # low_r2 fires on 3 rows: 2 wrong, 1 correct.
        # arch_effects fires on 1 row, which is correct (precision 0 ⇒ lift 0).
        return [
            {"id": 1, "correct": False, "flags": ["low_r2"]},
            {"id": 2, "correct": False, "flags": ["low_r2"]},
            {"id": 3, "correct": True,  "flags": ["low_r2"]},
            {"id": 4, "correct": True,  "flags": ["arch_effects"]},
            {"id": 5, "correct": False, "flags": []},
            {"id": 6, "correct": None,  "flags": ["low_r2"]},   # excluded
        ]

    def test_precision_recall_counts(self):
        fd = _load_flag_diagnostics()
        stats, meta = fd.compute_flag_stats(self._rows())

        # 5 scored, 3 wrong (ids 1,2,5) ⇒ base wrong-rate 0.6.
        self.assertEqual(meta["n_scored"], 5)
        self.assertEqual(meta["n_wrong"], 3)
        self.assertAlmostEqual(meta["base_wrong_rate"], 0.6)

        lr = stats["low_r2"]
        self.assertEqual(lr["fired"], 3)            # ids 1,2,3 (id 6 excluded)
        self.assertEqual(lr["fired_wrong"], 2)      # ids 1,2
        self.assertEqual(lr["fired_right"], 1)      # id 3
        self.assertAlmostEqual(lr["fire_rate"], 3 / 5)
        self.assertAlmostEqual(lr["precision"], 2 / 3)   # P(wrong | fired)
        self.assertAlmostEqual(lr["recall"], 2 / 3)      # P(fired | wrong)
        self.assertAlmostEqual(lr["lift"], (2 / 3) / 0.6)

        arch = stats["arch_effects"]
        self.assertEqual(arch["fired"], 1)
        self.assertEqual(arch["fired_wrong"], 0)    # fired only on a correct row
        self.assertAlmostEqual(arch["precision"], 0.0)
        self.assertAlmostEqual(arch["lift"], 0.0)   # net-negative regime

        ei = stats["evidence_incomplete"]
        self.assertEqual(ei["fired"], 0)            # never fired in the fixture
        self.assertIsNone(ei["precision"])          # undefined: no fired rows
        self.assertEqual(ei["recall"], 0.0)         # caught 0 of the wrong rows
        self.assertIsNone(ei["lift"])               # undefined: precision is None

    def test_row_flags_falls_back_to_evidence(self):
        fd = _load_flag_diagnostics()
        # top-level flags missing ⇒ read evidence.flags
        r = {"id": 1, "correct": False, "evidence": {"flags": ["weak_correlation"]}}
        self.assertEqual(fd._row_flags(r), {"weak_correlation"})


# ---------------------------------------------------------------------------
# raw_pixel_vision measurement arm (Track-V structured-vs-pixel study)
# ---------------------------------------------------------------------------
class _ArtifactProbeClient:
    """Like _MockClient but records, per call, whether image artifacts were
    passed AND which dispatch role the call played. Lets a test assert the
    *answer* LLM call received the raw image on a vision-fired row."""

    def __init__(self, routing_json, answer_letter="A"):
        self._routing = routing_json
        self._answer = answer_letter
        self.calls = []  # list of {"role", "has_artifacts"}

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        s = (system_prompt or "").lower()
        # NOTE: classify the FINAL answer call FIRST by its unique "expert" marker
        # — in raw-pixel mode it carries image artifacts, so an artifacts-based
        # check would misclassify it as the vision sensor. (The answer SYSTEM_PROMPT
        # is "You are an expert time series analyst…"; the vision-suggestion prompt
        # is "You are a careful visual time-series analyst.")
        if "expert time series analyst" in s:
            role = "answer"
        elif "branch" in s or "route" in s:
            role = "router"
        elif "careful visual" in s or "analyst" in s:
            role = "vision_suggestion"
        elif "critic" in s or "meta-cognitive" in s:
            role = "critic"
        elif artifacts or "visual" in (user_prompt or "").lower():
            role = "vision_sensor"
        else:
            role = "answer"
        self.calls.append({"role": role, "has_artifacts": bool(artifacts)})

        if role == "router":
            return self._routing
        if role == "vision_suggestion":
            return "B\nConfidence: high\nvisual cue"
        if role == "critic":
            return json.dumps({"branch": "trend", "reason": "x"})
        if role == "vision_sensor":
            return json.dumps({"summary": "syn", "topology": "smooth", "confidence": "high"})
        return f"{self._answer}\nConfidence: high\nreason"


def _anomaly_row():
    """Point-anomaly row — forced-vision path fires deterministically on it."""
    rng = np.random.default_rng(3)
    ts = rng.normal(0, 1, 200)
    ts[100] = 12.0
    return {
        "id": "rp_anom", "category": "anomaly",
        "question": "Is there a point anomaly in this series?",
        "options": ["yes", "no", "maybe", "unknown"],
        "answer": "yes", "ts": ts.tolist(),
    }


class TestRawPixelVision(unittest.TestCase):
    """Track-V raw-pixel measurement arm — SVI-violating, behind the switch."""

    def setUp(self):
        self.run_pipeline, _ = _import_runner()
        # Force vision on the anomaly branch so the fired path is exercised
        # deterministically without depending on the additive gate's score.
        self._routing = '{"branch":"anomaly","scope":"global","subtype":"point_anomaly"}'
        self._kw = {"forced_vision_branches": ["anomaly"]}

    def test_off_is_byte_identical_to_baseline_via_trace_hash(self):
        """raw_pixel_vision=False ⇒ trace_hash identical to the param being absent.
        Core invariant: the switch is 100% dark when off (additive AND forced
        vision paths)."""
        # additive-gate (trend) row + forced-vision (anomaly) row.
        rng = np.random.default_rng(42)
        trend_row = {
            "id": "rp_trend", "category": "trend",
            "question": "What is the direction of the trend in this series?",
            "options": ["increasing", "decreasing", "stationary", "oscillating"],
            "answer": "increasing",
            "ts": (np.linspace(0, 10, 200) + rng.normal(0, 0.2, 200)).tolist(),
        }
        cases = [
            ('{"branch":"trend","scope":"global","subtype":"direction"}', trend_row, {}),
            (self._routing, _anomaly_row(), self._kw),
        ]
        for routing, row, kw in cases:
            base = self.run_pipeline(
                dict(row), _ArtifactProbeClient(routing), return_trace=True, **kw
            )
            off = self.run_pipeline(
                dict(row), _ArtifactProbeClient(routing), return_trace=True,
                raw_pixel_vision=False, **kw
            )
            self.assertEqual(
                base["trace_hash"], off["trace_hash"],
                "raw_pixel_vision=False must be byte-identical to baseline",
            )
            self.assertFalse(off.get("raw_pixel_vision_used"))

    def test_raw_pixel_attaches_image_to_answer_call_on_fired_row(self):
        """On a vision-fired row, raw_pixel_vision=True attaches the rendered PNG
        directly to the ANSWER LLM call (the SVI-violating measurement step)."""
        client = _ArtifactProbeClient(self._routing)
        res = self.run_pipeline(
            _anomaly_row(), client, raw_pixel_vision=True, **self._kw
        )
        # The arm fired on this row.
        self.assertTrue(res.get("raw_pixel_vision_used"))
        # Exactly the ANSWER-role call(s) must carry the image artifact.
        answer_calls = [c for c in client.calls if c["role"] == "answer"]
        self.assertTrue(answer_calls, "expected an answer-LLM call")
        self.assertTrue(
            all(c["has_artifacts"] for c in answer_calls),
            "raw-pixel answer call must receive the image artifact",
        )

    def test_raw_pixel_skips_structured_json_topology(self):
        """raw_pixel mode bypasses analyze_image ⇒ no vision_struct JSON is built
        (the raw pixels replace the topology, not augment it)."""
        res = self.run_pipeline(
            _anomaly_row(), _ArtifactProbeClient(self._routing),
            raw_pixel_vision=True, **self._kw
        )
        ev = res.get("evidence") or {}
        self.assertIsNone(ev.get("vision_struct"))
        # but the artifact path (raw PNG) must exist for the answer call to use.
        self.assertTrue(ev.get("artifacts"))

    def test_structured_arm_still_builds_json_topology(self):
        """Sanity: with the switch OFF the structured sensor still produces
        vision_struct (the SVI default path is unchanged)."""
        res = self.run_pipeline(
            _anomaly_row(), _ArtifactProbeClient(self._routing), **self._kw
        )
        ev = res.get("evidence") or {}
        self.assertIsNotNone(ev.get("vision_struct"))
        self.assertFalse(res.get("raw_pixel_vision_used"))


if __name__ == "__main__":
    unittest.main()
