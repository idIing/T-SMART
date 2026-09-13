# smoke_test.py
"""
End-to-end smoke test for the MMTS-Bench evaluation pipeline.
Run from the my_project/ root:

    python smoke_test.py

Prints a PASS / FAIL report for every component.
"""

import os
import sys
import traceback

# Add tsqa/ to path so 'from eval.x' and 'from tools.x' resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tsqa"))

# Configurable via environment variable — works for everyone on the team:
#   export MMTS_BENCH_PATH="/path/to/your/MMTS-BENCH"
MMTS_BENCH_PATH = os.environ.get("MMTS_BENCH_PATH", "./MMTS-BENCH")

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append((name, False))

print("\n" + "="*60)
print("  MMTS-Bench Pipeline Smoke Test")
print("="*60 + "\n")

# ------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------
print("── Imports ──────────────────────────────────────")

def test_import_dataloaders():
    from eval.dataloaders import MMTSBenchAdapter

def test_import_baseline():
    from eval.baseline_matrix import DeltaMatrix

def test_import_anomaly():
    from tools.anomaly_score import zscore_anomaly_score, iqr_anomaly_score

def test_import_trend():
    from tools.trend_est import linear_trend, hurst_exponent

def test_import_granger():
    from tools.granger import granger_causality

check("Import dataloaders",     test_import_dataloaders)
check("Import baseline_matrix", test_import_baseline)
check("Import anomaly_score",   test_import_anomaly)
check("Import trend_est",       test_import_trend)
check("Import granger",         test_import_granger)

# ------------------------------------------------------------------
# 2. Data Adapter — each subset
# ------------------------------------------------------------------
print("\n── Data Adapter ─────────────────────────────────")
from eval.dataloaders import MMTSBenchAdapter

for subset, expected_min in [("Base", 600), ("InWild", 900), ("Match", 300), ("Align", 200)]:
    def test_subset(s=subset, e=expected_min):
        adapter = MMTSBenchAdapter(MMTS_BENCH_PATH, subset=s)
        n = len(adapter)
        assert n >= e, f"Expected >= {e} rows, got {n}"

        sample = next(iter(adapter))
        required = {"sample_id", "category", "ts_array", "query", "options", "ground_truth", "domain", "subset"}
        missing = required - set(sample.keys())
        assert not missing, f"Missing keys: {missing}"

        import numpy as np
        assert isinstance(sample["ts_array"], np.ndarray), "ts_array must be ndarray"
        assert isinstance(sample["query"],    str),        "query must be str"
        print(f"         {s}: {n} rows | sample_id={sample['sample_id']} | category='{sample['category']}'")

    check(f"Adapter subset='{subset}'", test_subset)

def test_all_subset():
    adapter = MMTSBenchAdapter(MMTS_BENCH_PATH, subset="All")
    assert len(adapter) >= 2000, f"All subset should have >= 2000 rows, got {len(adapter)}"
    print(f"         All: {len(adapter)} rows total")

check("Adapter subset='All'", test_all_subset)

# ------------------------------------------------------------------
# 3. NaN / edge-case robustness in real data
# ------------------------------------------------------------------
print("\n── Real-data Robustness ─────────────────────────")

def test_no_crash_on_full_iteration():
    import numpy as np
    adapter = MMTSBenchAdapter(MMTS_BENCH_PATH, subset="All")
    empty_ts, none_query = 0, 0
    for sample in adapter:
        if len(sample["ts_array"]) == 0:
            empty_ts += 1
        if not sample["query"].strip():
            none_query += 1
    print(f"         Iterated {len(adapter)} samples | empty ts_array: {empty_ts} | empty query: {none_query}")

check("Full iteration (no crashes)", test_no_crash_on_full_iteration)

# ------------------------------------------------------------------
# 4. Baseline Matrix
# ------------------------------------------------------------------
print("\n── Baseline Matrix ──────────────────────────────")

def test_baseline_loads():
    from eval.baseline_matrix import DeltaMatrix
    dm = DeltaMatrix()
    assert len(dm.baselines) > 0

def test_ingest_results():
    from eval.baseline_matrix import DeltaMatrix
    dm = DeltaMatrix()
    my_zs = {
        "trend analysis":       34.73,
        "seasonality analysis": 34.47,
        "noise analysis":       32.66,
        "volatility analysis":  43.17,
        "deductive reasoning":  36.08,
        "causal reasoning":     39.53,
        "caption":              74.58,
    }
    df = dm.ingest_results(my_zs, method="ZS")
    assert "Delta_vs_TSAgent" in df.columns
    assert "Delta_vs_ChatTS"  in df.columns
    print(dm.summary(method="ZS"))

def test_latex_export():
    from eval.baseline_matrix import DeltaMatrix
    dm = DeltaMatrix()
    latex = dm.export_latex_table(method="ZS")
    assert "\\begin{table}" in latex
    assert "\\end{table}"   in latex
    print(f"         LaTeX table: {len(latex)} chars generated")

check("Baseline loads",  test_baseline_loads)
check("Ingest results",  test_ingest_results)
check("LaTeX export",    test_latex_export)

# ------------------------------------------------------------------
# 5. Math Tools
# ------------------------------------------------------------------
print("\n── Math Tools ───────────────────────────────────")
import numpy as np

def test_anomaly_zscore():
    from tools.anomaly_score import zscore_anomaly_score
    ts = [0.0]*50 + [100.0] + [0.0]*50
    r  = zscore_anomaly_score(ts)
    assert isinstance(r, dict) and r["n_anomalies"] >= 1

def test_anomaly_iqr():
    from tools.anomaly_score import iqr_anomaly_score
    ts = list(range(100)) + [9999]
    r  = iqr_anomaly_score(ts)
    assert isinstance(r, dict) and r["n_anomalies"] >= 1

def test_trend_linear():
    from tools.trend_est import linear_trend
    r = linear_trend(list(range(100)))
    assert r["direction"] == "upward"

def test_trend_hurst():
    from tools.trend_est import hurst_exponent
    ts = np.arange(100, dtype=float) + np.random.normal(0, 0.1, 100)
    r  = hurst_exponent(ts)
    assert isinstance(r, dict) and "hurst" in r

def test_granger():
    from tools.granger import granger_causality
    rng = np.random.default_rng(0)
    x   = rng.normal(0, 1, 200)
    y   = np.zeros(200)
    for t in range(2, 200):
        y[t] = 0.9 * x[t-2] + rng.normal(0, 0.1)
    r = granger_causality(x, y, max_lag=3)
    assert r["granger_causes"] is True

def test_tools_return_errors_not_exceptions():
    from tools.anomaly_score import zscore_anomaly_score
    from tools.trend_est     import linear_trend
    from tools.granger       import granger_causality
    edge_cases = [
        zscore_anomaly_score([]),
        zscore_anomaly_score([float("nan")] * 20),
        zscore_anomaly_score([1.0]),
        linear_trend([]),
        linear_trend([float("nan")] * 10),
        granger_causality([1.0], [1.0]),
    ]
    for r in edge_cases:
        assert isinstance(r, str) and r.startswith("Error:"), \
            f"Expected 'Error:...' string, got: {r!r}"

check("Anomaly Z-score",                      test_anomaly_zscore)
check("Anomaly IQR",                          test_anomaly_iqr)
check("Trend linear",                         test_trend_linear)
check("Trend Hurst",                          test_trend_hurst)
check("Granger causality",                    test_granger)
check("Tools return errors (not exceptions)", test_tools_return_errors_not_exceptions)

# ------------------------------------------------------------------
# Final report
# ------------------------------------------------------------------
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("\n  Failed checks:")
    for name, ok in results:
        if not ok:
            print(f"    x {name}")
print("="*60 + "\n")
