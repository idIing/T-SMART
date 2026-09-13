# tests/test_report_spend.py
"""
Unit tests for scripts/report_spend.py -- the API-spend reporting helper
that reads the "usage" field run_mmts_baseline.py writes into its run
manifests.

Design philosophy
------------------
  1. A manifest with a "usage" block -> correct token totals and cost math
     against the pricing table.
  2. A manifest missing "usage" (an old run predating usage tracking) ->
     no crash, reported as "not recorded".
  3. An unknown model id -> no crash, cost reported as unknown rather than
     guessed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"

_spec = importlib.util.spec_from_file_location("report_spend", _SCRIPTS / "report_spend.py")
report_spend = importlib.util.module_from_spec(_spec)
sys.modules["report_spend"] = report_spend
_spec.loader.exec_module(report_spend)


def _write_manifest(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_cost_math_known_model():
    rates = report_spend.PRICING_USD_PER_1M_TOKENS["gemini-3.1-flash-lite"]
    cost, note = report_spend._cost("gemini-3.1-flash-lite", prompt_tokens=1_000_000,
                                     completion_tokens=1_000_000)
    assert note is None
    assert cost == pytest.approx(rates["input"] + rates["output"])


def test_cost_unknown_model_no_crash():
    cost, note = report_spend._cost("some-model-nobody-priced", prompt_tokens=100,
                                     completion_tokens=100)
    assert cost is None
    assert "cost unknown" in note
    assert "some-model-nobody-priced" in note


def test_load_manifest_with_usage(tmp_path):
    manifest = {
        "benchmark": "MMTS-Bench",
        "model": "gemini-3.1-flash-lite",
        "usage": {
            "total_tokens": 3000,
            "prompt_tokens": 2000,
            "completion_tokens": 1000,
            "calls": 10,
        },
    }
    p = _write_manifest(tmp_path, "mmts_run_test_20260101_000000.json", manifest)

    loaded = report_spend._load_manifests([str(p)])
    assert len(loaded) == 1
    path, data = loaded[0]
    usage = data["usage"]
    assert usage["calls"] == 10
    assert usage["prompt_tokens"] == 2000
    assert usage["completion_tokens"] == 1000
    assert usage["total_tokens"] == 3000

    cost, note = report_spend._cost(data["model"], usage["prompt_tokens"], usage["completion_tokens"])
    rates = report_spend.PRICING_USD_PER_1M_TOKENS["gemini-3.1-flash-lite"]
    expected = (2000 / 1_000_000.0) * rates["input"] + (1000 / 1_000_000.0) * rates["output"]
    assert note is None
    assert cost == pytest.approx(expected)


def test_manifest_without_usage_does_not_crash(tmp_path, capsys):
    manifest = {
        "benchmark": "MMTS-Bench",
        "model": "gemini-3.1-flash-lite",
        # no "usage" key -- simulates a pre-usage-tracking run
    }
    p = _write_manifest(tmp_path, "mmts_run_old_20250101_000000.json", manifest)

    sys_argv = sys.argv
    try:
        sys.argv = ["report_spend.py", str(p)]
        report_spend.main()
    finally:
        sys.argv = sys_argv

    out = capsys.readouterr().out
    assert "usage not recorded" in out


def test_glob_expansion_and_totals(tmp_path, capsys):
    m1 = {
        "model": "gemini-3.1-flash-lite",
        "usage": {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40, "calls": 2},
    }
    m2 = {
        "model": "gemini-3.1-flash-lite",
        "usage": {"total_tokens": 200, "prompt_tokens": 120, "completion_tokens": 80, "calls": 4},
    }
    _write_manifest(tmp_path, "mmts_run_a_20260101_000000.json", m1)
    _write_manifest(tmp_path, "mmts_run_b_20260101_000000.json", m2)

    sys_argv = sys.argv
    try:
        sys.argv = ["report_spend.py", str(tmp_path / "mmts_run_*.json")]
        report_spend.main()
    finally:
        sys.argv = sys_argv

    out = capsys.readouterr().out
    assert "calls:             6" in out
    assert "total tokens:      300" in out
