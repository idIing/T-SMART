import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsqa.eval.adapter_base import tsexam_row_to_sample
from tsqa.eval.cluster_diff import cluster_bootstrap_delta, paired_permutation_p
from tsqa.eval.controls import apply_control
from tsqa.eval import runner as runner_mod
from tsqa.eval.runner import run_pipeline
from tsqa.eval.sample import AnswerType, BenchmarkSample, Channel, OptionType
from tsqa.eval.scoring import score_prediction, summarize_scores, to_float
from tsqa.llm.factory import create_llm_client
from tsqa.llm.openai_client import OpenAIClient
from tsqa.llm.parser import parse_answer
from tsqa.llm.prompt_numeric import parse_numeric_answer


class TestScoring(unittest.TestCase):
    def test_schema_denominators(self):
        results = [
            score_prediction(sample_id="m", answer_type=AnswerType.MCQ, predicted="A", gold="A"),
            score_prediction(sample_id="n", answer_type=AnswerType.NUMERICAL, predicted=11, gold=10),
            score_prediction(sample_id="c", answer_type=AnswerType.CATEGORICAL, predicted="Trend Up", gold="trend-up"),
            score_prediction(sample_id="o", answer_type=AnswerType.OPEN, predicted=None, gold="essay"),
            score_prediction(sample_id="a", answer_type=AnswerType.NUMERICAL, predicted=None, gold=5),
            score_prediction(sample_id="e", answer_type=AnswerType.MCQ, predicted="A", gold="B", error="boom"),
        ]
        summary = summarize_scores(results)
        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.eligible, 5)
        self.assertEqual(summary.numeric, 2)
        self.assertEqual(summary.open_excluded, 1)
        self.assertEqual(summary.abstention, 1)
        self.assertEqual(summary.error, 1)
        self.assertEqual(summary.incorrect, 0)
        self.assertEqual(summary.correct, 3)  # (#12) eligible=5, error=1, abstention=1, incorrect=0 → correct=3

    def test_numeric_nan_and_letter_to_option_gold(self):
        self.assertIsNone(to_float(float("nan")))
        self.assertIsNone(to_float("nan"))
        res = score_prediction(
            sample_id="az",
            answer_type=AnswerType.MCQ,
            predicted="second option",
            gold="B",
            options=["first option", "second option"],
        )
        self.assertTrue(res.correct)

    def test_numeric_parser_negative_variants(self):
        for raw in ("-3.14", "- 3.14", "\u22123.14", "minus 3.14"):
            parsed = parse_numeric_answer(raw)
            self.assertTrue(parsed["parse_success"])
            self.assertAlmostEqual(parsed["value"], -3.14)
        self.assertEqual(parse_numeric_answer("3.14")["value"], 3.14)

    def test_answer_parser_letters_without_prose_false_positive(self):
        self.assertEqual(parse_answer("E")["letter"], "E")
        self.assertEqual(parse_answer("Answer: E")["letter"], "E")
        self.assertEqual(parse_answer("Selected: E")["letter"], "E")
        self.assertFalse(parse_answer("This is A trending pattern")["parse_success"])

    def test_answer_parser_recovers_keyword_is_letter(self):
        # "<keyword> is/was <letter>" must parse even when not letter-first — this
        # phrasing otherwise silently abstains (regression #5). Recovery is
        # monotonic: the tertiary scan runs only after the primary/secondary
        # (letter-first) parsers fail.
        self.assertEqual(parse_answer("The answer is B")["letter"], "B")
        self.assertEqual(parse_answer("The correct option is C.")["letter"], "C")
        self.assertEqual(parse_answer("Reasoning here.\nThe answer is D")["letter"], "D")
        self.assertEqual(parse_answer("the selected letter was A")["letter"], "A")
        # FP guard: a bare "is <letter>" with NO answer keyword must still abstain.
        self.assertFalse(parse_answer("The variance is A bit higher")["parse_success"])
        self.assertFalse(parse_answer("This is A trending pattern")["parse_success"])


class TestAdaptersAndControls(unittest.TestCase):
    def test_tsexam_wrapper_separates_reporting_labels(self):
        adapted = tsexam_row_to_sample({
            "id": "r1",
            "question": "q",
            "options": ["yes", "no"],
            "answer": "yes",
            "ts": [1, 2, 3],
            "dimension": "oracle",
            "task": "route-me",
        })
        self.assertFalse(hasattr(adapted.sample, "dimension"))
        self.assertEqual(adapted.labels.dimension, "oracle")

    def test_metadata_only_control_cannot_see_task_dimension(self):
        sample = BenchmarkSample(
            id="s",
            question="What is the trend?",
            options=["up", "down"],
            answer="up",
            answer_type=AnswerType.MCQ,
            option_type=OptionType.CATEGORICAL_TEXT,
            series=[Channel("ts", [1, 2, 3])],
        )
        controlled = apply_control(sample, "metadata_only")
        row = controlled.to_runner_row()
        joined = json.dumps(row)
        self.assertNotIn("task", joined.lower())
        self.assertNotIn("dimension", joined.lower())
        self.assertEqual(controlled.series, [])
        self.assertEqual(controlled.options, [])

    def test_option_only_retains_options_without_series(self):
        sample = BenchmarkSample(
            id="s",
            question="What is the trend?",
            options=["up", "down"],
            answer="up",
            answer_type=AnswerType.MCQ,
            option_type=OptionType.CATEGORICAL_TEXT,
            series=[Channel("ts", [1, 2, 3])],
        )
        controlled = apply_control(sample, "option_only")
        self.assertTrue(controlled.question)
        self.assertEqual(controlled.options, ["up", "down"])
        self.assertEqual(controlled.series, [])

    def test_shuffled_series_preserves_values_not_order(self):
        sample = BenchmarkSample(
            id="s", question="q", options=[], answer="",
            answer_type=AnswerType.OPEN,
            series=[Channel("ts", np.arange(20))],
        )
        controlled = apply_control(sample, "shuffled_series", seed=3)
        self.assertCountEqual(controlled.series[0].values.tolist(), list(range(20)))
        self.assertFalse(np.array_equal(controlled.series[0].values, sample.series[0].values))
        self.assertIsNone(controlled.series[0].timestamps)

    def test_raw_row_controls_match_sample_controls(self):
        sample = BenchmarkSample(
            id="s",
            question="What is the trend?",
            options=["up", "down"],
            answer="up",
            answer_type=AnswerType.MCQ,
            option_type=OptionType.CATEGORICAL_TEXT,
            series=[Channel("ts", [1, 2, 3])],
            category="trend",
        )
        base_row = sample.to_runner_row()
        base_row["category"] = "trend"
        for mode in ("option_only", "metadata_only", "no_series", "shuffled_series"):
            controlled_sample = apply_control(sample, mode, seed=2)
            controlled_row = runner_mod._apply_row_control(base_row, mode, seed=2)
            sample_row = controlled_sample.to_runner_row()
            self.assertEqual(controlled_row.get("question"), sample_row.get("question"))
            self.assertEqual(controlled_row.get("options"), sample_row.get("options"))
            self.assertEqual(controlled_row.get("ts") is None, sample_row.get("ts") is None)
        self.assertNotIn("category", runner_mod._apply_row_control(base_row, "metadata_only"))


class TestProvider(unittest.TestCase):
    def test_factory_uses_mocked_gemini(self):
        with patch("tsqa.llm.factory.GeminiClient") as cls:
            cls.return_value = "client"
            got = create_llm_client(provider="gemini", model="m", api_key="k")
        self.assertEqual(got, "client")
        cls.assert_called_once()

    def test_openai_text_and_artifact_call(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(b"png")
            f.flush()

            message = SimpleNamespace(content="A")
            choice = SimpleNamespace(message=message)
            completions = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(choices=[choice])
            )
            chat = SimpleNamespace(completions=completions)
            client = SimpleNamespace(chat=chat)
            llm = OpenAIClient(client=client)
            self.assertEqual(llm.generate("sys", "usr", artifacts=[{"kind": "image", "path": f.name}]), "A")

    def test_openai_transient_retry_succeeds(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("429 rate limit")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = OpenAIClient(client=client, max_retries=1, min_call_interval=0)
        with patch("tsqa.llm.openai_client.time.sleep", lambda _: None):
            self.assertEqual(llm.generate("sys", "usr"), "A")
        self.assertEqual(calls["n"], 2)

    def test_openai_timeout_retry_succeeds(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("timed out")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="B"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = OpenAIClient(client=client, max_retries=1, min_call_interval=0)
        with patch("tsqa.llm.openai_client.time.sleep", lambda _: None):
            self.assertEqual(llm.generate("sys", "usr"), "B")

    def test_openai_empty_choices_raises_clean_error(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[]))
            )
        )
        llm = OpenAIClient(client=client, min_call_interval=0)
        with self.assertRaisesRegex(RuntimeError, "no choices"):
            llm.generate("sys", "usr")

    def test_factory_uses_mocked_openai(self):
        with patch("tsqa.llm.factory.OpenAIClient") as cls:
            cls.return_value = "oai"
            got = create_llm_client(provider="openai", api_key="k")
        self.assertEqual(got, "oai")
        # default model for openai is gpt-4o-mini (Track-A backbone)
        _, kwargs = cls.call_args
        self.assertEqual(kwargs.get("model_name"), "gpt-4o-mini")

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            create_llm_client(provider="not-a-provider")

    def test_openai_passes_temperature_zero(self):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = OpenAIClient(client=client, min_call_interval=0)
        llm.generate("sys", "usr")
        self.assertEqual(seen.get("temperature"), 0.0)
        self.assertEqual(seen.get("model"), "gpt-4o-mini")

    def test_openai_temp0_determinism(self):
        # At temp 0 the same prompt yields the same letter across calls (the
        # pairing assumption for the Track-A within-backbone ablation).
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="C\nbecause"))]
                    )
                )
            )
        )
        llm = OpenAIClient(client=client, min_call_interval=0)
        a = parse_answer(llm.generate("sys", "usr"))["letter"]
        b = parse_answer(llm.generate("sys", "usr"))["letter"]
        self.assertEqual(a, b)
        self.assertEqual(a, "C")


class TestLLMOnly(unittest.TestCase):
    """The no-architecture control arm (Track-A de-confound)."""

    def _row(self):
        return {
            "id": "lo",
            "category": "NU",
            "question": "What noise type is present?",
            "options": ["white", "red", "pink", "blue"],
            "answer": "red",
            "ts": np.linspace(0, 1, 80).tolist(),
        }

    def test_llm_only_single_call_no_router_no_branch(self):
        client = _CountingClient()
        res = run_pipeline(self._row(), client, llm_only=True)
        # exactly ONE LLM call; no router/critic system prompts among them
        self.assertEqual(len(client.calls), 1)
        sp = (client.calls[0] or "").lower()
        self.assertNotIn("route", sp)
        self.assertNotIn("branch", sp)
        self.assertEqual(res["branch_used"], "llm_only")
        self.assertEqual(res["initial_branch_used"], "llm_only")
        self.assertEqual(res["predicted_letter"], "A")
        self.assertTrue(res["parse_success"])

    def test_llm_only_records_correctness(self):
        # _CountingClient answers "A"; gold for option "red" is "B" → wrong.
        client = _CountingClient()
        res = run_pipeline(self._row(), client, llm_only=True)
        self.assertEqual(res["gold_letter"], "B")
        self.assertFalse(res["correct"])
        self.assertIsNone(res["evidence"])  # no tool evidence produced

    def test_llm_only_prompt_is_letter_neutral(self):
        from tsqa.llm import build_llm_only_prompt

        sys_p, usr_p = build_llm_only_prompt(
            "Which is it?", ["alpha", "beta", "gamma"]
        )
        # options block is present, no per-option steering / answer leakage
        self.assertIn("A. alpha", usr_p)
        self.assertIn("C. gamma", usr_p)
        for nudge in ("correct answer is", "the answer is", "hint:", "likely", "probably"):
            self.assertNotIn(nudge, usr_p.lower())


class TestClusterDiff(unittest.TestCase):
    def test_cluster_bootstrap_and_permutation(self):
        res = cluster_bootstrap_delta([1, 0, 1, 0], [1, 1, 1, 0], ["a", "a", "b", "b"], n_boot=50)
        self.assertEqual(res.n, 4)
        self.assertIn("a", res.sensitivity)
        p = paired_permutation_p([1, 0, 1], [1, 1, 1], n_perm=50)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class _CountingClient:
    model_name = "mock"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.calls.append(system_prompt)
        s = (system_prompt or "").lower()
        if "branch" in s or "route" in s:
            return '{"branch":"trend","scope":"global","subtype":"direction"}'
        if "critic" in s or "meta-cognitive" in s:
            raise AssertionError("critic should not be called")
        return "A\nConfidence: high"


class TestRunnerRecovery(unittest.TestCase):
    def test_top2_recovery_avoids_critic(self):
        client = _CountingClient()
        row = {
            "id": "r",
            "category": "trend",
            "question": "Does the trend go up or is it anomalous?",
            "options": ["up", "down", "flat", "unknown"],
            "answer": "up",
            "ts": np.linspace(0, 1, 80).tolist(),
        }
        res = run_pipeline(row, client, multi_branch=True, recovery="top2_nocritic", return_trace=True)
        self.assertFalse(res["critic_called"])
        self.assertIn("trace_hash", res)
        self.assertEqual(res["predicted_letter"], "A")

    def test_forecast_ranker_records_unsupported_multi_series(self):
        client = _CountingClient()
        row = {
            "id": "forecast_multi",
            "category": "prediction",
            "question": "Which forecast trajectory is closest?",
            "options": ["[1, 2]", "[3, 4]"],
            "answer": "[1, 2]",
            "ts1": [1, 2, 3, 4],
            "ts2": [2, 3, 4, 5],
            "option_type": "trajectory",
        }
        res = run_pipeline(row, client, forecast_ranker=True)
        self.assertFalse(res["used_head"])
        self.assertIn("unsupported_series_count=2", res["head_note"])

    def test_forecast_ranker_does_not_fire_for_matrix_or_ordering(self):
        client = _CountingClient()
        for option_type, options in (
            ("matrix", ["[[0, 1], [1, 0]]", "[[1, 0], [0, 1]]"]),
            ("ordering", ["A -> B", "B -> A"]),
        ):
            row = {
                "id": f"forecast_{option_type}",
                "category": "prediction",
                "question": "Choose the answer.",
                "options": options,
                "answer": options[0],
                "ts": [1, 2, 3, 4],
                "option_type": option_type,
            }
            res = run_pipeline(row, client, forecast_ranker=True)
            self.assertFalse(res["used_head"])
            self.assertIn("unsupported_option_type", res["head_note"])


class _LoopClient:
    model_name = "loop-mock"

    def __init__(self, reroute='{"branch":"noise","scope":"global","subtype":"stationarity"}'):
        self.router_calls = 0
        self.reroute = reroute

    def generate(self, system_prompt, user_prompt, **kwargs):
        if "branch" in (system_prompt or "").lower() or "route" in (system_prompt or "").lower():
            self.router_calls += 1
            if self.router_calls == 1:
                return '{"branch":"trend","scope":"global","subtype":"direction"}'
            return self.reroute
        return "A\nConfidence: high"


def _incomplete_trend(series, scope="global"):
    return {"branch": "trend", "scope": scope, "slope": None, "direction": None, "r2": None}


def _complete_noise(series, scope="global"):
    return {
        "branch": "noise",
        "scope": scope,
        "adf_pval": 0.01,
        "kpss_pval": 0.1,
        "is_stationary": True,
        "is_white_noise": True,
    }


def _incomplete_noise(series, scope="global"):
    return {
        "branch": "noise",
        "scope": scope,
        "adf_pval": None,
        "kpss_pval": None,
        "is_stationary": None,
        "is_white_noise": None,
    }


class TestStage1Loop(unittest.TestCase):
    def _row(self):
        return {
            "id": "loop-row",
            "category": "trend",
            "question": "What is happening?",
            "options": ["a", "b"],
            "answer": "a",
            "ts": [1, 2, 3, 4, 5],
        }

    def test_disabled_loop_trace_identity(self):
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _complete_noise}):
            a = run_pipeline(self._row(), _LoopClient(), return_trace=True)
            b = run_pipeline(
                self._row(),
                _LoopClient(),
                return_trace=True,
                loop_branches=["trend"],
                max_loop_depth=0,
            )
        self.assertEqual(a["trace_hash"], b["trace_hash"])
        self.assertNotIn("loop_used", b)

    def test_loop_does_not_fire_for_disallowed_branch(self):
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _incomplete_trend}):
            res = run_pipeline(
                self._row(),
                _LoopClient(),
                loop_branches=["noise"],
                max_loop_depth=1,
            )
        self.assertNotIn("loop_used", res)

    def test_loop_fires_once_on_configured_flag(self):
        with patch.dict(
            runner_mod._SINGLE_BRANCHES,
            {"trend": _incomplete_trend, "noise": _complete_noise},
        ):
            res = run_pipeline(
                self._row(),
                _LoopClient(),
                loop_branches=["trend", "noise"],
                max_loop_depth=1,
                loop_on_flags=["evidence_incomplete"],
            )
        self.assertTrue(res["loop_used"])
        self.assertEqual(res["loop_depth"], 1)
        self.assertEqual(res["loop_original_branch"], "trend")
        self.assertEqual(res["loop_final_branch"], "noise")
        self.assertEqual(res["branch_used"], "noise")

    def test_max_depth_stops_recursive_retry(self):
        with patch.dict(
            runner_mod._SINGLE_BRANCHES,
            {"trend": _incomplete_trend, "noise": _incomplete_noise},
        ):
            res = run_pipeline(
                self._row(),
                _LoopClient(),
                loop_branches=["trend", "noise"],
                max_loop_depth=1,
            )
        self.assertEqual(res["loop_depth"], 1)
        self.assertEqual(res["loop_stopped"], "max_depth")

    def test_failed_reroute_retains_original_evidence(self):
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _incomplete_trend}):
            res = run_pipeline(
                self._row(),
                _LoopClient(reroute="not json"),
                loop_branches=["trend"],
                max_loop_depth=1,
            )
        self.assertTrue(res["loop_used"])
        self.assertEqual(res["loop_final_branch"], "trend")
        self.assertEqual(res["branch_used"], "trend")
        self.assertEqual((res.get("evidence") or {}).get("branch"), "trend")
        self.assertEqual(res["loop_error"], "reroute_parse_failed")

    def test_initial_branch_used_is_pre_reroute(self):
        # No loop: initial_branch_used is present and equals the final branch.
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _complete_noise}):
            res = run_pipeline(self._row(), _LoopClient())
        self.assertEqual(res["initial_branch_used"], "trend")
        self.assertEqual(res["branch_used"], "trend")
        # Loop fires (trend -> noise): initial_branch_used stays the PRE-reroute
        # branch while branch_used becomes the final one. This is the stable,
        # collider-free stratifier the paired diffs must group by (finding #1).
        with patch.dict(
            runner_mod._SINGLE_BRANCHES,
            {"trend": _incomplete_trend, "noise": _complete_noise},
        ):
            res = run_pipeline(
                self._row(),
                _LoopClient(),
                loop_branches=["trend", "noise"],
                max_loop_depth=1,
                loop_on_flags=["evidence_incomplete"],
            )
        self.assertEqual(res["initial_branch_used"], "trend")
        self.assertEqual(res["branch_used"], "noise")

    def test_same_branch_reroute_is_noop(self):
        # The reroute router returns the SAME branch the row is already on. The
        # guard (finding #3) must stop BEFORE re-executing the branch: identical
        # tools would yield identical evidence and burn the depth budget for free.
        calls = {"trend": 0}

        def _counting_incomplete_trend(series, scope="global"):
            calls["trend"] += 1
            return _incomplete_trend(series, scope)

        same = '{"branch":"trend","scope":"global","subtype":"direction"}'
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _counting_incomplete_trend}):
            res = run_pipeline(
                self._row(),
                _LoopClient(reroute=same),
                loop_branches=["trend"],
                max_loop_depth=1,
                loop_on_flags=["evidence_incomplete"],
            )
        self.assertTrue(res["loop_used"])            # gate fired, reroute consulted
        self.assertEqual(res["loop_depth"], 0)       # but no reroute was committed
        self.assertEqual(res["loop_stopped"], "same_branch_noop")
        self.assertEqual(res["branch_used"], "trend")
        self.assertEqual(res["loop_final_branch"], "trend")
        self.assertEqual(calls["trend"], 1)          # tool NOT re-executed


# ---------------------------------------------------------------------------
# Stage-2 iterative tool refinement (loop_mode="refine")
# ---------------------------------------------------------------------------


class _RefineCountClient:
    """Single-branch router (always trend) + counts answer calls; the critic LLM
    is a hard error (refine must NEVER call it — 0 extra LLM calls)."""

    model_name = "refine-mock"

    def __init__(self):
        self.router_calls = 0
        self.answer_calls = 0

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        s = (system_prompt or "").lower()
        if "branch" in s or "route" in s:
            self.router_calls += 1
            return '{"branch":"trend","scope":"global","subtype":"direction"}'
        if "critic" in s or "meta-cognitive" in s:
            raise AssertionError("refine must not call the critic LLM")
        self.answer_calls += 1
        return "A\nConfidence: high"


def _strong_trend(series, scope="global"):
    """Complete, high-r2 trend evidence ⇒ quality clears the gate (no refine)."""
    return {
        "branch": "trend", "scope": scope, "slope": 0.5,
        "direction": "increasing", "r2": 0.95, "series_length": len(series),
    }


# A hard-flag stack that pushes evaluate_quality below 0.55 while keeping all
# REQUIRED fields present (so evidence_incomplete never fires and the row does NOT
# short-circuit to the fallback path). This is the realistic Stage-2 firing case
# on in-distribution MCQ — evidence_incomplete is dormant there, but flag stacks
# do occur. Score ≈ 0.48 (completeness 1.0, flag_penalty capped, math_conf 0.4).
_HARD_FLAG_STACK = [
    "low_r2", "fft_unreliable", "adf_kpss_disagree",
    "granger_not_significant", "weak_correlation", "arch_effects", "high_volatility",
]


def _weak_complete_trend(series, scope="global"):
    return {
        "branch": "trend", "scope": scope, "slope": 0.01,
        "direction": "increasing", "r2": 0.2, "series_length": len(series),
        "flags": list(_HARD_FLAG_STACK),
    }


_REFINE_CFG = dict(
    loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
    max_loop_depth=2,
    loop_mode="refine",
    quality_threshold=0.55,
)


class TestStage2Refine(unittest.TestCase):
    def _row(self, question="Is this series noisy white-noise/stationary or trending up?"):
        return {
            "id": "refine-row",
            "category": "trend",
            "question": question,
            "options": ["a", "b"],
            "answer": "a",
            "ts": np.linspace(0, 1, 80).tolist(),
        }

    # --- byte-identity (the core invariant), all three off-conditions ---------

    def test_byte_identical_when_loop_mode_not_refine(self):
        """loop_mode != 'refine' ⇒ refine path never entered: ALL result fields
        (incl. diagnostics) are byte-identical to baseline."""
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            base = run_pipeline(self._row(), _RefineCountClient(),
                                no_vision_branches=["trend"], return_trace=True)
            # reroute_once with depth 0 = baseline; refine inert because mode≠refine.
            treat = run_pipeline(self._row(), _RefineCountClient(),
                                 no_vision_branches=["trend"], return_trace=True,
                                 loop_branches=_REFINE_CFG["loop_branches"],
                                 max_loop_depth=2, loop_mode="reroute_once")
        self.assertEqual(base["trace_hash"], treat["trace_hash"])
        self.assertFalse(treat["refine_fired"])
        self.assertEqual(treat["refine_depth"], 0)
        self.assertIsNone(treat["refine_branches"])
        # quality_score untouched (refine never scored it) ⇒ stays None like baseline.
        self.assertIsNone(treat["quality_score"])

    def test_byte_identical_when_max_depth_le_1(self):
        """max_loop_depth<=1 ⇒ no extra branch may be accumulated: byte-identical."""
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            base = run_pipeline(self._row(), _RefineCountClient(),
                                no_vision_branches=["trend"], return_trace=True)
            treat = run_pipeline(self._row(), _RefineCountClient(),
                                 no_vision_branches=["trend"], return_trace=True,
                                 loop_branches=_REFINE_CFG["loop_branches"],
                                 max_loop_depth=1, loop_mode="refine")
        self.assertEqual(base["trace_hash"], treat["trace_hash"])
        self.assertFalse(treat["refine_fired"])
        self.assertIsNone(treat["quality_score"])

    def test_prediction_byte_identical_when_quality_clears(self):
        """quality >= threshold ⇒ accumulate NOTHING: the PREDICTION and evidence
        are byte-identical to baseline (only the diagnostic quality_score, which
        baseline never computed, is now populated — it is NOT in the answer
        prompt's math_evidence, so the letter cannot change)."""
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _strong_trend}):
            base = run_pipeline(self._row(), _RefineCountClient(),
                                no_vision_branches=["trend"])
            treat = run_pipeline(self._row(), _RefineCountClient(),
                                 no_vision_branches=["trend"], **_REFINE_CFG)
        self.assertFalse(treat["refine_fired"])
        self.assertEqual(treat["refine_depth"], 0)
        self.assertIsNone(treat["refine_branches"])
        self.assertEqual(base["predicted_letter"], treat["predicted_letter"])
        self.assertEqual(base["branch_used"], treat["branch_used"])
        self.assertEqual(base["correct"], treat["correct"])
        # No refine_* evidence keys were added (accumulation did not run).
        ev = treat["evidence"] or {}
        self.assertEqual([k for k in ev if k.startswith("refine_")], [])
        # quality_score IS now computed (the Stage-1 lesson) even on a no-op row.
        self.assertIsNotNone(treat["quality_score"])
        self.assertGreaterEqual(treat["quality_score"], 0.55)

    # --- firing behaviour ------------------------------------------------------

    def test_fires_and_accumulates_second_branch_below_threshold(self):
        """quality < 0.55 (hard-flag stack, NOT evidence_incomplete) ⇒ a 2nd
        keyword-matched branch's evidence is ACCUMULATED (merged, not replaced)."""
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            res = run_pipeline(self._row(), _RefineCountClient(),
                               no_vision_branches=["trend"], **_REFINE_CFG)
        self.assertTrue(res["refine_fired"])
        self.assertGreaterEqual(res["refine_depth"], 1)
        self.assertIn("noise", res["refine_branches"])
        ev = res["evidence"] or {}
        # Accumulated under a namespaced key; the PRIMARY branch is preserved.
        self.assertIn("refine_noise", ev)
        self.assertEqual(ev.get("branch"), "trend")        # primary not replaced
        # The primary's original flags survive the merge (accumulate, don't clobber).
        self.assertIn("low_r2", ev.get("flags") or [])
        self.assertIsNotNone(res["quality_score"])

    def test_no_extra_llm_calls_in_refine(self):
        """SVI / 0-extra-LLM-call invariant: even when refine fires, the answer LLM
        is called EXACTLY once and the critic is NEVER called."""
        client = _RefineCountClient()
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_REFINE_CFG)
        self.assertTrue(res["refine_fired"])
        self.assertEqual(client.answer_calls, 1)   # single answer call on richer evidence
        self.assertEqual(client.router_calls, 1)   # only the initial route (no reroute)

    def test_depth_strictly_bounded(self):
        """Depth is hard-bounded: max_loop_depth=2 ⇒ at most 1 accumulated branch,
        even when MANY candidates would match and all score below threshold. Asserts
        no unbounded recursion (refine_depth <= max_loop_depth - 1)."""
        # All candidate branches return weak (below-threshold) evidence so the gate
        # never clears — the ONLY thing that can stop the loop is the depth bound.
        def _weak_any(series, scope="global"):
            return {"branch": "noise", "scope": scope, "flags": list(_HARD_FLAG_STACK),
                    "adf_pval": 0.5, "kpss_pval": 0.0}
        q = ("Is this trending, periodic/seasonal cycle, anomalous spike, noisy "
             "stationary, similar shape, or causal lag?")  # matches ALL 6 branches
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _weak_any,
                         "periodicity": _weak_any, "anomaly": _weak_any}):
            res = run_pipeline(self._row(question=q), _RefineCountClient(),
                               no_vision_branches=["trend"], **_REFINE_CFG)
        self.assertTrue(res["refine_fired"])
        self.assertLessEqual(res["refine_depth"], _REFINE_CFG["max_loop_depth"] - 1)
        self.assertEqual(res["refine_depth"], 1)   # exactly one accumulated branch

    def test_depth_three_accumulates_at_most_two(self):
        """max_loop_depth=3 ⇒ up to 2 accumulated branches; still bounded."""
        def _weak_any(series, scope="global"):
            return {"branch": "x", "scope": scope, "flags": list(_HARD_FLAG_STACK),
                    "adf_pval": 0.5, "kpss_pval": 0.0}
        q = ("Is this trending, a periodic seasonal cycle, an anomalous spike, or "
             "noisy stationary white noise?")
        cfg = dict(_REFINE_CFG, max_loop_depth=3)
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _weak_any,
                         "periodicity": _weak_any, "anomaly": _weak_any}):
            res = run_pipeline(self._row(question=q), _RefineCountClient(),
                               no_vision_branches=["trend"], **cfg)
        self.assertTrue(res["refine_fired"])
        self.assertLessEqual(res["refine_depth"], cfg["max_loop_depth"] - 1)
        self.assertLessEqual(res["refine_depth"], 2)

    def test_deterministic_accumulation_on_rerun(self):
        """Temp-0 determinism: the SAME evidence is accumulated on a re-run (the
        refine path is pure deterministic tools — no temp>0 sampling)."""
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            r1 = run_pipeline(self._row(), _RefineCountClient(),
                              no_vision_branches=["trend"], **_REFINE_CFG)
            r2 = run_pipeline(self._row(), _RefineCountClient(),
                              no_vision_branches=["trend"], **_REFINE_CFG)
        self.assertEqual(r1["refine_branches"], r2["refine_branches"])
        self.assertEqual(r1["refine_depth"], r2["refine_depth"])
        self.assertEqual(r1["quality_score"], r2["quality_score"])
        self.assertEqual(r1["predicted_letter"], r2["predicted_letter"])


# ---------------------------------------------------------------------------
# Stage-3 LLM-proposed actions (loop_mode="propose")
# ---------------------------------------------------------------------------

from tsqa.orchestrator.action_proposer import (
    ACTION_REGISTRY,
    ProposedAction,
    parse_proposal,
    build_proposal_prompt,
)


class _ProposeClient:
    """Router that always routes to `trend`, answers "A", and returns a
    configurable PROPOSAL when asked to propose. The critic LLM is a hard error
    (the propose path must never call it). Counts answer + proposal calls so the
    SVI / budget invariants are assertable."""

    model_name = "propose-mock"

    def __init__(self, proposal='{"action":"noise","reason":"check stationarity"}'):
        self.router_calls = 0
        self.answer_calls = 0
        self.proposal_calls = 0
        self.proposal = proposal

    def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
        s = (system_prompt or "").lower()
        if "action proposer" in s:           # the Stage-3 proposer system prompt
            self.proposal_calls += 1
            return self.proposal
        if "branch" in s or "route" in s:
            self.router_calls += 1
            return '{"branch":"trend","scope":"global","subtype":"direction"}'
        if "critic" in s or "meta-cognitive" in s:
            raise AssertionError("propose must not call the critic LLM")
        self.answer_calls += 1
        return "A\nConfidence: high"


_PROPOSE_CFG = dict(
    loop_branches=["trend", "periodicity", "anomaly", "noise", "similarity", "causality"],
    max_loop_depth=2,
    loop_mode="propose",
    quality_threshold=0.55,
)


class TestStage3ProposerUnit(unittest.TestCase):
    """Preconditions 1+2: the typed schema + the closed registry (pure parser)."""

    def test_registry_is_the_closed_eight(self):
        self.assertEqual(
            ACTION_REGISTRY,
            frozenset({"trend", "periodicity", "anomaly", "noise", "similarity",
                       "causality", "vision", "numeric_head"}),
        )

    def test_valid_proposal_parses_to_typed_action(self):
        p = parse_proposal('{"action":"noise","reason":"stationarity unclear"}')
        self.assertIsInstance(p, ProposedAction)
        self.assertEqual(p.action, "noise")
        self.assertEqual(p.reason, "stationarity unclear")

    def test_fenced_json_parses(self):
        p = parse_proposal('```json\n{"action":"vision","reason":"shape"}\n```')
        self.assertIsNotNone(p)
        self.assertEqual(p.action, "vision")

    def test_out_of_registry_rejected(self):
        # Precondition 2: an action outside the closed 8 is invalid (fail closed).
        self.assertIsNone(parse_proposal('{"action":"forecast","reason":"x"}'))
        self.assertIsNone(parse_proposal('{"action":"web_search","reason":"x"}'))
        self.assertIsNone(parse_proposal('{"action":"","reason":"x"}'))

    def test_unparseable_or_malformed_rejected(self):
        # Precondition 4: anything not validating to the schema returns None.
        for bad in ("not json", "", "{}", '{"reason":"missing action"}',
                    '{"action":42,"reason":"non-string action"}',
                    '{"action":"noise","reason":123}', '{"action":"noise"}'):
            self.assertIsNone(parse_proposal(bad), bad)

    def test_direct_construction_cannot_smuggle_bad_action(self):
        with self.assertRaises(ValueError):
            ProposedAction(action="not_a_tool", reason="x")

    def test_prompt_omits_answer_options(self):
        # No-letter-nudging: the proposal prompt must NOT contain the options block.
        prompt = build_proposal_prompt(
            "Is the variance constant?", "noise",
            {"branch": "noise", "flags": ["arch_effects"]},
        )
        self.assertNotIn("A. ", prompt)
        self.assertNotIn("Answer choices", prompt)
        self.assertIn("noise", prompt)


class TestStage3Propose(unittest.TestCase):
    def _row(self, question="Is this trending up or noisy and stationary?"):
        return {
            "id": "propose-row",
            "category": "trend",
            "question": question,
            "options": ["a", "b"],
            "answer": "a",
            "ts": np.linspace(0, 1, 80).tolist(),
        }

    # --- byte-identity (the core invariant) -----------------------------------

    def test_byte_identical_when_loop_mode_not_propose(self):
        """loop_mode != 'propose' ⇒ propose path never entered: trace byte-identical."""
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            base = run_pipeline(self._row(), _ProposeClient(),
                                no_vision_branches=["trend"], return_trace=True)
            treat = run_pipeline(self._row(), _ProposeClient(),
                                 no_vision_branches=["trend"], return_trace=True,
                                 loop_branches=_PROPOSE_CFG["loop_branches"],
                                 max_loop_depth=2, loop_mode="reroute_once")
        self.assertEqual(base["trace_hash"], treat["trace_hash"])
        self.assertFalse(treat["propose_fired"])
        self.assertIsNone(treat["propose_executed"])
        self.assertIsNone(treat["quality_score"])  # propose never scored it

    def test_byte_identical_when_max_depth_zero(self):
        """max_loop_depth<=0 ⇒ no action may execute: byte-identical to baseline."""
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            base = run_pipeline(self._row(), _ProposeClient(),
                                no_vision_branches=["trend"], return_trace=True)
            treat = run_pipeline(self._row(), _ProposeClient(),
                                 no_vision_branches=["trend"], return_trace=True,
                                 loop_branches=_PROPOSE_CFG["loop_branches"],
                                 max_loop_depth=0, loop_mode="propose")
        self.assertEqual(base["trace_hash"], treat["trace_hash"])
        self.assertFalse(treat["propose_fired"])

    def test_byte_identical_when_quality_clears(self):
        """quality >= threshold ⇒ NO proposal call, accumulate nothing: prediction
        + evidence byte-identical to baseline (only the diagnostic quality_score,
        absent from the answer prompt, is populated)."""
        client = _ProposeClient()
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _strong_trend}):
            base = run_pipeline(self._row(), _ProposeClient(),
                                no_vision_branches=["trend"])
            treat = run_pipeline(self._row(), client,
                                 no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertFalse(treat["propose_fired"])
        self.assertEqual(client.proposal_calls, 0)        # gate didn't fire ⇒ no LLM call
        self.assertEqual(base["predicted_letter"], treat["predicted_letter"])
        ev = treat["evidence"] or {}
        self.assertEqual([k for k in ev if k.startswith("propose_")], [])
        self.assertIsNotNone(treat["quality_score"])
        self.assertGreaterEqual(treat["quality_score"], 0.55)

    # --- firing behaviour ------------------------------------------------------

    def test_fires_and_accumulates_proposed_branch_below_threshold(self):
        """quality < 0.55 ⇒ the LLM is asked to propose; a valid branch proposal's
        evidence is ACCUMULATED (namespaced, not replacing the primary)."""
        client = _ProposeClient(proposal='{"action":"noise","reason":"x"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertTrue(res["propose_fired"])
        self.assertIn("noise", res["propose_executed"])
        ev = res["evidence"] or {}
        self.assertIn("propose_noise", ev)            # accumulated, namespaced
        self.assertEqual(ev.get("branch"), "trend")   # primary NOT replaced
        self.assertIn("low_r2", ev.get("flags") or [])  # primary flags survive merge
        self.assertGreaterEqual(client.proposal_calls, 1)

    def test_answer_llm_called_once_never_reanswers(self):
        """SVI: the proposal is a ROUTING decision. Even when it fires, the answer
        LLM is called EXACTLY once (on enriched evidence), the critic is NEVER
        called, and the proposal is a SEPARATE call from the answer."""
        client = _ProposeClient(proposal='{"action":"noise","reason":"x"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertTrue(res["propose_fired"])
        self.assertEqual(client.answer_calls, 1)      # ONE answer call on richer evidence
        self.assertEqual(client.router_calls, 1)      # only the initial route
        self.assertGreaterEqual(client.proposal_calls, 1)

    def test_invalid_proposal_falls_back_deterministically(self):
        """Precondition 4: an unparseable/out-of-registry proposal ⇒ NO action
        executed, NO crash, NO retry; the row proceeds on the original evidence."""
        client = _ProposeClient(proposal="not json at all")
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertFalse(res["propose_fired"])
        self.assertIsNone(res["propose_executed"])
        self.assertIn("invalid_proposal", res["propose_fallbacks"])
        self.assertEqual(client.proposal_calls, 1)    # called ONCE, not retried
        self.assertEqual(client.answer_calls, 1)      # answered normally anyway

    def test_out_of_registry_proposal_rejected_at_runtime(self):
        """Registry closed end-to-end: a syntactically valid but out-of-registry
        action executes nothing and falls back."""
        client = _ProposeClient(proposal='{"action":"forecast","reason":"x"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertFalse(res["propose_fired"])
        self.assertIn("invalid_proposal", res["propose_fallbacks"])

    def test_already_run_primary_proposal_is_noop(self):
        """If the LLM re-proposes the already-run PRIMARY branch, it is a no-op
        (would re-run identical tools) ⇒ deterministic fallback, nothing executed."""
        client = _ProposeClient(proposal='{"action":"trend","reason":"redo"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertFalse(res["propose_fired"])
        self.assertTrue(any("already_run" in f for f in res["propose_fallbacks"]))

    def test_budget_enforced_at_most_max_depth_executions(self):
        """Precondition 3: at most max_loop_depth EXECUTED actions even when EVERY
        proposed branch scores below threshold (the only stop is the budget). The
        proposer here proposes a DIFFERENT branch each call so it never short-stops
        on already-run."""
        proposals = iter([
            '{"action":"noise","reason":"a"}',
            '{"action":"periodicity","reason":"b"}',
            '{"action":"anomaly","reason":"c"}',
        ])

        def _weak_any(series, scope="global"):
            return {"branch": "x", "scope": scope, "flags": list(_HARD_FLAG_STACK),
                    "adf_pval": 0.5, "kpss_pval": 0.0}

        class _SeqProposeClient(_ProposeClient):
            def generate(self, system_prompt, user_prompt, artifacts=None, **kwargs):
                if "action proposer" in (system_prompt or "").lower():
                    self.proposal_calls += 1
                    return next(proposals)
                return super().generate(system_prompt, user_prompt, artifacts, **kwargs)

        client = _SeqProposeClient()
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _weak_any,
                         "periodicity": _weak_any, "anomaly": _weak_any}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertTrue(res["propose_fired"])
        # max_loop_depth=2 ⇒ at most 2 executed actions and at most 2 proposal calls.
        self.assertLessEqual(len(res["propose_executed"]), 2)
        self.assertLessEqual(client.proposal_calls, 2)

    def test_vision_proposal_forces_vision_gate(self):
        """A proposed `vision` action escalates the LOOK decision: the vision gate
        fires (vision_triggered True, proposed flag set) even though the additive
        gate would not have, and NO branch evidence is accumulated for it."""
        client = _ProposeClient(proposal='{"action":"vision","reason":"shape"}')

        class _VisionStubClient(_ProposeClient):
            # answer + vision-suggestion both return a letter; analyze_image is
            # patched out below so no real image work happens.
            pass

        client = _VisionStubClient(proposal='{"action":"vision","reason":"shape"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}), \
             patch.object(runner_mod, "generate_ts_artifact",
                          return_value={"artifacts": [], "evidence": {}}), \
             patch.object(runner_mod, "analyze_image",
                          return_value={"vision_struct": {"summary": "s"},
                                        "vision_text": "t", "raw": "r"}):
            res = run_pipeline(self._row(), client, **_PROPOSE_CFG)
        self.assertTrue(res["propose_fired"])
        self.assertIn("vision", res["propose_executed"])
        vtrig = (res["evidence"] or {}).get("vision_trigger") or {}
        self.assertTrue(vtrig.get("triggered"))
        self.assertTrue(vtrig.get("proposed"))

    def test_numeric_head_proposal_is_noop_on_mcq(self):
        """numeric_head is in the registry but MCQ rows can never enter the numeric
        head ⇒ proposing it executes nothing (honest coverage gap, no fabricated
        value) and falls back; the MCQ row answers normally."""
        client = _ProposeClient(proposal='{"action":"numeric_head","reason":"exact"}')
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _weak_complete_trend}):
            res = run_pipeline(self._row(), client,
                               no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertFalse(res["propose_fired"])
        self.assertTrue(
            any("numeric_head" in f for f in res["propose_fallbacks"])
        )
        self.assertEqual(res["expected_schema"], "mcq")    # never left the MCQ path
        self.assertEqual(client.answer_calls, 1)

    def test_deterministic_accumulation_on_rerun(self):
        """Temp-0 determinism: the SAME action is proposed+executed on a re-run
        (the mock proposer is deterministic; the executed tools are deterministic)."""
        with patch.dict(runner_mod._SINGLE_BRANCHES,
                        {"trend": _weak_complete_trend, "noise": _complete_noise}):
            r1 = run_pipeline(self._row(), _ProposeClient(),
                              no_vision_branches=["trend"], **_PROPOSE_CFG)
            r2 = run_pipeline(self._row(), _ProposeClient(),
                              no_vision_branches=["trend"], **_PROPOSE_CFG)
        self.assertEqual(r1["propose_executed"], r2["propose_executed"])
        self.assertEqual(r1["quality_score"], r2["quality_score"])
        self.assertEqual(r1["predicted_letter"], r2["predicted_letter"])


# ---------------------------------------------------------------------------
# Phase-4 `agentic_tsmart` deliverable alias — cross-surface byte-identity
# ---------------------------------------------------------------------------

def _load_configs_dict(rel_path, attr):
    """Load a named CONFIGS dict from one of the three runner scripts by file path,
    without importing the whole module (avoids cross-tree sys.path side effects).

    The dicts are pure data built from builtins only — some via ``dict(k=v)`` calls
    (TSExam), some as ``{...}`` literals (MMTS/TSRBench) — so the assignment RHS is
    evaluated in a namespace exposing ONLY ``dict`` (no other names resolve)."""
    import ast
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo, rel_path)
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == attr for t in node.targets
        ):
            expr = ast.unparse(node.value)
            return eval(expr, {"__builtins__": {"dict": dict}}, {})  # noqa: S307
    raise AssertionError(f"{attr} not found in {rel_path}")


class TestAgenticTsmartAlias(unittest.TestCase):
    """The Phase-4 deliverable config `agentic_tsmart` must (1) be defined and
    byte-identical to `react_stage3_actions` in both runner scripts, and
    (2) when its gate clears, produce a prediction byte-identical to baseline
    (it is shipped DARK / for non-regression — logs 004-007)."""

    _SURFACES = [
        ("tsexam/run_eval.py", "CONFIGS"),
        ("mmts_bench/scripts/run_mmts_baseline.py", "MMTS_CONFIGS"),
    ]

    def test_alias_equals_stage3_in_every_surface(self):
        for rel_path, attr in self._SURFACES:
            cfgs = _load_configs_dict(rel_path, attr)
            self.assertIn("agentic_tsmart", cfgs, f"{rel_path}:{attr}")
            self.assertIn("react_stage3_actions", cfgs, f"{rel_path}:{attr}")
            self.assertEqual(
                cfgs["agentic_tsmart"], cfgs["react_stage3_actions"],
                f"agentic_tsmart drifted from react_stage3_actions in {rel_path}:{attr}",
            )

    def test_alias_identical_across_surfaces_on_shared_keys(self):
        """The loop switch (the keys present in the TSExam/MMTS dicts) must be
        byte-identical across both surfaces."""
        loaded = {rel: _load_configs_dict(rel, attr)["agentic_tsmart"]
                  for rel, attr in self._SURFACES}
        tsexam = loaded["tsexam/run_eval.py"]
        for rel, cfg in loaded.items():
            for k, v in tsexam.items():
                self.assertEqual(cfg.get(k), v, f"{rel} key {k}")

    def test_prediction_byte_identical_to_baseline_when_gate_clears(self):
        """Run the REAL agentic_tsmart kwargs end-to-end: on a gate-clearing row the
        proposer never fires (0 LLM proposal calls) and the predicted letter + the
        evidence match baseline exactly."""
        cfg = _load_configs_dict("tsexam/run_eval.py", "CONFIGS")["agentic_tsmart"]
        row = {
            "id": "agentic-gateclear", "category": "trend",
            "question": "Is this trending up or noisy and stationary?",
            "options": ["a", "b"], "answer": "a",
            "ts": np.linspace(0, 1, 80).tolist(),
        }
        client = _ProposeClient()
        with patch.dict(runner_mod._SINGLE_BRANCHES, {"trend": _strong_trend}):
            base = run_pipeline(dict(row), _ProposeClient(), no_vision_branches=["trend"])
            treat = run_pipeline(dict(row), client, no_vision_branches=["trend"], **cfg)
        self.assertFalse(treat["propose_fired"])
        self.assertEqual(client.proposal_calls, 0)            # gate clear ⇒ no LLM call
        self.assertEqual(base["predicted_letter"], treat["predicted_letter"])
        self.assertEqual(base["evidence"], treat["evidence"])


if __name__ == "__main__":
    unittest.main()
