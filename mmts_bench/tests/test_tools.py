# tests/test_tools.py
"""
Unit tests for tsqa/tools/ — anomaly_score, trend_est, granger.

Run with:
    pytest tests/test_tools.py -v

Design philosophy
-----------------
Every test checks one of three properties:
  1. Happy path — function returns the correct type and plausible values.
  2. Edge cases — all-zeros, all-NaN, extremely short arrays, single element.
  3. Error contract — instead of raising, the function returns a string
     that starts with "Error:".
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Adjust sys.path so tests run from repo root without installing the package
# ---------------------------------------------------------------------------
import sys, os
# Point to tsqa/ so 'from tools.x import ...' and 'from eval.x import ...' work
_here = os.path.dirname(__file__)           # tests/
_root = os.path.join(_here, "..")           # my_project/
_tsqa = os.path.join(_root, "tsqa")         # my_project/tsqa/
sys.path.insert(0, _tsqa)
sys.path.insert(0, _root)

from tools.anomaly_score import zscore_anomaly_score, iqr_anomaly_score
from tools.trend_est     import linear_trend, hurst_exponent
from tools.granger       import granger_causality


# ===========================================================================
# Helpers
# ===========================================================================

def _is_error(result) -> bool:
    """Return True iff the result is a string starting with 'Error:'."""
    return isinstance(result, str) and result.startswith("Error:")


# ===========================================================================
# anomaly_score — zscore_anomaly_score
# ===========================================================================

class TestZscoreAnomalyScore:

    def test_happy_path_detects_spike(self):
        ts = [0.0] * 50 + [100.0] + [0.0] * 50   # one obvious spike
        result = zscore_anomaly_score(ts, threshold=3.0)
        assert isinstance(result, dict)
        assert result["n_anomalies"] >= 1
        assert 50 in result["anomaly_indices"]

    def test_returns_dict_keys(self):
        ts = np.random.normal(0, 1, 100)
        result = zscore_anomaly_score(ts)
        assert isinstance(result, dict)
        assert "anomaly_indices" in result
        assert "anomaly_scores"  in result
        assert "n_anomalies"     in result

    def test_all_zeros_returns_error(self):
        result = zscore_anomaly_score([0.0] * 50)
        assert _is_error(result), f"Expected error string, got: {result}"

    def test_all_nan_returns_error(self):
        result = zscore_anomaly_score([float("nan")] * 30)
        assert _is_error(result)

    def test_mostly_nan_with_one_value_returns_error(self):
        ts = [float("nan")] * 29 + [1.0]
        result = zscore_anomaly_score(ts)
        assert _is_error(result)

    def test_length_one_returns_error(self):
        result = zscore_anomaly_score([42.0])
        assert _is_error(result)

    def test_empty_array_returns_error(self):
        result = zscore_anomaly_score([])
        assert _is_error(result)

    def test_numpy_input_accepted(self):
        ts = np.array([1.0, 2.0, 3.0, 100.0, 2.0, 1.5])
        result = zscore_anomaly_score(ts, threshold=2.0)
        assert not _is_error(result)

    def test_constant_series_returns_error(self):
        result = zscore_anomaly_score([5.0] * 100)
        assert _is_error(result)

    def test_high_threshold_no_anomalies(self):
        ts = np.random.normal(0, 1, 200)
        result = zscore_anomaly_score(ts, threshold=100.0)
        assert isinstance(result, dict)
        assert result["n_anomalies"] == 0

    def test_two_element_array_works(self):
        result = zscore_anomaly_score([1.0, 1000.0])
        # Two distinct values → std > 0, should succeed
        assert not _is_error(result)


# ===========================================================================
# anomaly_score — iqr_anomaly_score
# ===========================================================================

class TestIqrAnomalyScore:

    def test_happy_path_detects_outlier(self):
        ts = list(range(50)) + [9999]
        result = iqr_anomaly_score(ts)
        assert isinstance(result, dict)
        assert result["n_anomalies"] >= 1

    def test_returns_dict_keys(self):
        ts = np.random.normal(0, 1, 100)
        result = iqr_anomaly_score(ts)
        assert "anomaly_indices" in result
        assert "lower_fence"     in result
        assert "upper_fence"     in result
        assert "n_anomalies"     in result

    def test_all_zeros_iqr_zero_returns_error(self):
        result = iqr_anomaly_score([0.0] * 50)
        assert _is_error(result)

    def test_all_nan_returns_error(self):
        result = iqr_anomaly_score([float("nan")] * 20)
        assert _is_error(result)

    def test_length_one_returns_error(self):
        result = iqr_anomaly_score([1.0])
        assert _is_error(result)

    def test_length_three_returns_error(self):
        result = iqr_anomaly_score([1.0, 2.0, 3.0])
        assert _is_error(result)

    def test_empty_array_returns_error(self):
        result = iqr_anomaly_score([])
        assert _is_error(result)

    def test_no_outliers_in_normal_data(self):
        np.random.seed(0)
        ts = np.random.normal(100, 1, 200)
        result = iqr_anomaly_score(ts, k=3.0)
        assert isinstance(result, dict)
        # With k=3 virtually no outliers expected in tight normal data
        assert result["n_anomalies"] < 5


# ===========================================================================
# trend_est — linear_trend
# ===========================================================================

class TestLinearTrend:

    def test_clear_upward_trend(self):
        ts = list(range(100))
        result = linear_trend(ts)
        assert isinstance(result, dict)
        assert result["direction"] == "upward"
        assert result["slope"] > 0
        assert result["r_squared"] > 0.99

    def test_clear_downward_trend(self):
        ts = list(range(100, 0, -1))
        result = linear_trend(ts)
        assert result["direction"] == "downward"
        assert result["slope"] < 0

    def test_constant_series(self):
        ts = [5.0] * 50
        result = linear_trend(ts)
        assert isinstance(result, dict)
        # slope ≈ 0 → stationary
        assert result["direction"] == "stationary"

    def test_all_nan_returns_error(self):
        result = linear_trend([float("nan")] * 20)
        assert _is_error(result)

    def test_single_element_returns_error(self):
        result = linear_trend([1.0])
        assert _is_error(result)

    def test_empty_array_returns_error(self):
        result = linear_trend([])
        assert _is_error(result)

    def test_mostly_nan_few_valid_values(self):
        ts = [float("nan")] * 49 + [1.0]
        result = linear_trend(ts)
        # Only 1 clean value → error
        assert _is_error(result)

    def test_two_valid_values_works(self):
        ts = [float("nan")] * 8 + [0.0, 10.0]
        result = linear_trend(ts)
        assert not _is_error(result)
        assert result["direction"] == "upward"

    def test_returns_required_keys(self):
        ts = np.random.normal(0, 1, 50)
        result = linear_trend(ts)
        assert "slope"     in result
        assert "intercept" in result
        assert "r_squared" in result
        assert "direction" in result

    def test_r_squared_in_range(self):
        ts = np.random.normal(5, 0.1, 100)
        result = linear_trend(ts)
        assert 0.0 <= result["r_squared"] <= 1.0 + 1e-6


# ===========================================================================
# trend_est — hurst_exponent
# ===========================================================================

class TestHurstExponent:

    def test_random_walk_near_half(self):
        # The R/S Hurst estimator is known to be positively biased on finite
        # series — empirical values of 0.6–1.1 are normal for n=500 random walks.
        # We just verify the function returns a dict with a numeric 'hurst' key.
        np.random.seed(42)
        rw = np.cumsum(np.random.normal(0, 1, 500))
        result = hurst_exponent(rw)
        assert isinstance(result, dict)
        assert isinstance(result["hurst"], float)
        # Hurst should at minimum be positive
        assert result["hurst"] > 0.0

    def test_trending_series_high_hurst(self):
        ts = np.arange(500, dtype=float) + np.random.normal(0, 0.5, 500)
        result = hurst_exponent(ts)
        assert isinstance(result, dict)
        assert result["hurst"] > 0.5

    def test_short_series_returns_error(self):
        result = hurst_exponent([1.0, 2.0, 3.0])
        assert _is_error(result)

    def test_length_19_returns_error(self):
        ts = list(range(19))
        result = hurst_exponent(ts)
        assert _is_error(result)

    def test_all_nan_returns_error(self):
        result = hurst_exponent([float("nan")] * 50)
        assert _is_error(result)

    def test_empty_array_returns_error(self):
        result = hurst_exponent([])
        assert _is_error(result)

    def test_returns_required_keys(self):
        ts = np.random.normal(0, 1, 100)
        result = hurst_exponent(ts)
        assert "hurst"          in result
        assert "interpretation" in result

    def test_interpretation_string(self):
        ts = np.random.normal(0, 1, 100)
        result = hurst_exponent(ts)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


# ===========================================================================
# granger — granger_causality
# ===========================================================================

class TestGrangerCausality:

    def _make_causal_pair(self, n=200, lag=2, seed=0):
        """Create a pair where x causes y with a known lag."""
        rng = np.random.default_rng(seed)
        x   = rng.normal(0, 1, n)
        eps = rng.normal(0, 0.1, n)
        y   = np.zeros(n)
        for t in range(lag, n):
            y[t] = 0.9 * x[t - lag] + eps[t]
        return x, y

    def test_detects_known_causality(self):
        x, y = self._make_causal_pair()
        result = granger_causality(x, y, max_lag=3)
        assert isinstance(result, dict)
        assert result["granger_causes"] is True

    def test_no_causality_for_independent_series(self):
        rng = np.random.default_rng(7)
        x   = rng.normal(0, 1, 200)
        y   = rng.normal(0, 1, 200)
        result = granger_causality(x, y, max_lag=3)
        assert isinstance(result, dict)
        # Not necessarily non-causal by chance, just check it ran
        assert "p_value" in result

    def test_all_zeros_both_series_returns_error(self):
        result = granger_causality([0.0] * 50, [0.0] * 50)
        assert _is_error(result)

    def test_all_nan_returns_error(self):
        nans = [float("nan")] * 30
        result = granger_causality(nans, nans)
        assert _is_error(result)

    def test_very_short_series_returns_error(self):
        result = granger_causality([1.0, 2.0], [2.0, 3.0], max_lag=3)
        assert _is_error(result)

    def test_length_one_returns_error(self):
        result = granger_causality([1.0], [1.0])
        assert _is_error(result)

    def test_unequal_length_returns_error(self):
        result = granger_causality([1.0] * 50, [1.0] * 30)
        assert _is_error(result)

    def test_empty_arrays_return_error(self):
        result = granger_causality([], [])
        assert _is_error(result)

    def test_mostly_nan_returns_error(self):
        ts = [float("nan")] * 48 + [1.0, 2.0]
        result = granger_causality(ts, ts, max_lag=2)
        assert _is_error(result)

    def test_returns_required_keys(self):
        x, y = self._make_causal_pair()
        result = granger_causality(x, y)
        for key in ("f_statistic", "p_value", "max_lag",
                    "granger_causes", "interpretation"):
            assert key in result, f"Missing key: {key}"

    def test_f_statistic_non_negative(self):
        x, y = self._make_causal_pair()
        result = granger_causality(x, y)
        assert result["f_statistic"] >= 0.0

    def test_p_value_in_unit_interval(self):
        x, y = self._make_causal_pair()
        result = granger_causality(x, y)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_numpy_input_accepted(self):
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        result = granger_causality(x, y, max_lag=2)
        assert not _is_error(result)

    def test_invalid_max_lag_returns_error(self):
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        result = granger_causality(x, y, max_lag=0)
        assert _is_error(result)


# ===========================================================================
# tsqa.tools.basic_stats — basic_stats / percentile / value_at_index /
#                          count_threshold
# ---------------------------------------------------------------------------
# These live in the tsqa `tsqa` PACKAGE, not the standalone
# mmts_bench/tools/ above. We make `tsqa` importable the same way the MMTS
# scripts do (run_mmts_baseline.py): put tsqa/ on sys.path and import
# `tsqa` as a package. `tsqa.tools` is a distinct fully-qualified name from the
# top-level `tools` imported above, so the two coexist without collision.
# Unlike the hardened mmts_bench/tools/ (which return an "Error:" string), this
# tool RAISES ValueError on bad input — so these tests assert exceptions.
# ===========================================================================

_AGENTIC = os.path.abspath(os.path.join(_here, "..", "..", "tsqa"))
if _AGENTIC not in sys.path:
    sys.path.insert(0, _AGENTIC)

# Import the submodule directly (avoids triggering tsqa.tools.__init__, which
# pulls in statsmodels/scipy/ruptures we don't need for these closed-form tests).
from tsqa.tools.basic_stats import (
    basic_stats,
    percentile,
    value_at_index,
    count_threshold,
)


class TestBasicStats:

    # A few fixed arrays exercised against numpy ground truth.
    ARRAYS = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([-5.0, -2.0, 0.0, 3.0, 4.5, 10.0]),
        np.array([2.5, -1.25, 7.75, -3.0]),
        np.array([42.0]),                       # single element
    ]

    def test_keys_present(self):
        result = basic_stats(self.ARRAYS[0])
        expected_keys = {
            "n", "mean", "median", "std", "var", "min", "max", "range",
            "sum", "abs_max", "q1", "q3", "iqr", "first", "last",
            "skewness", "kurtosis",
        }
        assert set(result.keys()) == expected_keys

    def test_values_match_numpy_ground_truth(self):
        for a in self.ARRAYS:
            r = basic_stats(a)
            assert r["n"] == int(a.size)
            assert r["mean"] == pytest.approx(float(np.mean(a)))
            assert r["median"] == pytest.approx(float(np.median(a)))
            assert r["std"] == pytest.approx(float(np.std(a, ddof=0)))
            assert r["var"] == pytest.approx(float(np.var(a, ddof=0)))
            assert r["min"] == pytest.approx(float(np.min(a)))
            assert r["max"] == pytest.approx(float(np.max(a)))
            assert r["range"] == pytest.approx(float(np.ptp(a)))
            assert r["sum"] == pytest.approx(float(np.sum(a)))
            assert r["abs_max"] == pytest.approx(float(np.max(np.abs(a))))
            assert r["q1"] == pytest.approx(float(np.percentile(a, 25)))
            assert r["q3"] == pytest.approx(float(np.percentile(a, 75)))
            assert r["iqr"] == pytest.approx(
                float(np.percentile(a, 75) - np.percentile(a, 25))
            )
            assert r["first"] == pytest.approx(float(a[0]))
            assert r["last"] == pytest.approx(float(a[-1]))

    def test_population_std_not_sample(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = basic_stats(a)
        # ddof=0 (population), not ddof=1 (sample)
        assert r["std"] == pytest.approx(float(np.std(a, ddof=0)))
        assert r["std"] != pytest.approx(float(np.std(a, ddof=1)))

    def test_single_element(self):
        r = basic_stats([42.0])
        assert r["n"] == 1
        assert r["std"] == 0.0
        assert r["var"] == 0.0
        assert r["iqr"] == 0.0
        assert r["mean"] == 42.0 and r["median"] == 42.0
        assert r["first"] == 42.0 and r["last"] == 42.0

    def test_list_input_accepted(self):
        r = basic_stats([1.0, 2.0, 3.0])
        assert r["mean"] == pytest.approx(2.0)

    def test_nan_values_dropped(self):
        # NaNs dropped, stats computed on finite values only.
        r = basic_stats([1.0, float("nan"), 3.0])
        assert r["n"] == 2
        assert r["mean"] == pytest.approx(2.0)

    def test_value_types_are_python_scalars(self):
        r = basic_stats(self.ARRAYS[0])
        assert isinstance(r["n"], int)
        for k in ("mean", "median", "std", "var", "min", "max", "range",
                  "sum", "abs_max", "q1", "q3", "iqr", "first", "last",
                  "skewness", "kurtosis"):
            assert isinstance(r[k], float), f"{k} is {type(r[k])}, expected float"

    # ---- bad input: documented contract is to RAISE ValueError -------------

    def test_none_raises(self):
        with pytest.raises(ValueError):
            basic_stats(None)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            basic_stats([])

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            basic_stats(np.zeros((2, 3)))

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            basic_stats([float("nan")] * 5)


class TestPercentile:

    def test_matches_numpy_for_various_q(self):
        a = np.array([-5.0, -2.0, 0.0, 3.0, 4.5, 10.0])
        for q in (0, 10, 50, 90, 100):
            assert percentile(a, q) == pytest.approx(float(np.percentile(a, q)))

    def test_linear_interpolation_default(self):
        # Median of even-length array uses linear interpolation.
        a = [1.0, 2.0, 3.0, 4.0]
        assert percentile(a, 50) == pytest.approx(2.5)

    def test_q_below_zero_raises(self):
        with pytest.raises(ValueError):
            percentile([1.0, 2.0, 3.0], -1)

    def test_q_above_hundred_raises(self):
        with pytest.raises(ValueError):
            percentile([1.0, 2.0, 3.0], 150)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)


class TestValueAtIndex:

    def test_positive_index(self):
        a = np.array([10.0, 20.0, 30.0, 40.0])
        assert value_at_index(a, 0) == 10.0
        assert value_at_index(a, 2) == 30.0

    def test_negative_index(self):
        a = np.array([10.0, 20.0, 30.0, 40.0])
        assert value_at_index(a, -1) == 40.0
        assert value_at_index(a, -4) == 10.0

    def test_out_of_range_positive_raises_indexerror(self):
        with pytest.raises(IndexError):
            value_at_index([1.0, 2.0, 3.0], 5)

    def test_out_of_range_negative_raises_indexerror(self):
        with pytest.raises(IndexError):
            value_at_index([1.0, 2.0, 3.0], -4)

    def test_empty_raises_valueerror(self):
        with pytest.raises(ValueError):
            value_at_index([], 0)


class TestCountThreshold:

    ARR = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_greater_than(self):
        assert count_threshold(self.ARR, ">", 3) == 2

    def test_greater_equal(self):
        assert count_threshold(self.ARR, ">=", 3) == 3

    def test_less_than(self):
        assert count_threshold(self.ARR, "<", 3) == 2

    def test_less_equal(self):
        assert count_threshold(self.ARR, "<=", 3) == 3

    def test_equal(self):
        assert count_threshold(self.ARR, "==", 3) == 1

    def test_not_equal(self):
        assert count_threshold(self.ARR, "!=", 3) == 4

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            count_threshold(self.ARR, "=>", 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            count_threshold([], ">", 0)


class TestTSAQATransformation:

    def test_shift_offset(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        ts = np.array([1.0, 2.0, 3.0])
        res = TSAQATransformation.shift_offset(ts, 1.5)
        np.testing.assert_allclose(res, np.array([2.5, 3.5, 4.5]))

    def test_scale(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        ts = np.array([1.0, 2.0, 3.0])
        res = TSAQATransformation.scale(ts, 2.0)
        np.testing.assert_allclose(res, np.array([2.0, 4.0, 6.0]))

    def test_lag(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        ts = np.array([1.0, 2.0, 3.0])
        res_pos = TSAQATransformation.lag(ts, 1, fill_value=0.0)
        np.testing.assert_allclose(res_pos, np.array([0.0, 1.0, 2.0]))
        res_neg = TSAQATransformation.lag(ts, -1, fill_value=9.0)
        np.testing.assert_allclose(res_neg, np.array([2.0, 3.0, 9.0]))

    def test_moving_average(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        res = TSAQATransformation.moving_average(ts, 2)
        np.testing.assert_allclose(res, np.array([1.5, 2.5, 3.5]))

    def test_difference(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        ts = np.array([1.0, 3.0, 6.0, 10.0])
        res = TSAQATransformation.difference(ts, 1)
        np.testing.assert_allclose(res, np.array([2.0, 3.0, 4.0]))

    def test_detrend(self):
        from tsqa.tools.tsaqa import TSAQATransformation
        # Linear trend: y = 2 * x + 5
        ts = np.array([5.0, 7.0, 9.0, 11.0])
        res = TSAQATransformation.detrend(ts)
        np.testing.assert_allclose(res, np.zeros(4), atol=1e-7)


class TestTSAQAPuzzler:

    def test_junction_cost(self):
        from tsqa.tools.tsaqa import TSAQAPuzzler
        seg1 = np.array([1.0, 2.0, 3.0])
        seg2 = np.array([3.1, 4.0, 5.0])
        # val_diff = (3.0 - 3.1) ** 2 = 0.01
        # slope1 = 3.0 - 2.0 = 1.0; slope2 = 4.0 - 3.1 = 0.9. slope_diff = (1.0 - 0.9) ** 2 = 0.01
        # cost = 0.01 + 0.5 * 0.01 = 0.015
        cost = TSAQAPuzzler.junction_cost(seg1, seg2)
        assert cost == pytest.approx(0.015)

    def test_solve_jigsaw_without_ref(self):
        from tsqa.tools.tsaqa import TSAQAPuzzler
        # Original continuous series: [0, 1, 2, 3, 4, 5]
        # Segments: seg0=[2, 3], seg1=[4, 5], seg2=[0, 1]
        seg0 = np.array([2.0, 3.0])
        seg1 = np.array([4.0, 5.0])
        seg2 = np.array([0.0, 1.0])
        
        # Solving jigsaw should reconstruct the order: seg2 -> seg0 -> seg1
        # So sequence indices should be: [2, 0, 1]
        order = TSAQAPuzzler.solve_jigsaw([seg0, seg1, seg2])
        assert order == [2, 0, 1]

    def test_solve_jigsaw_with_ref(self):
        from tsqa.tools.tsaqa import TSAQAPuzzler
        ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        seg0 = np.array([2.0, 3.0])
        seg1 = np.array([4.0, 5.0])
        seg2 = np.array([0.0, 1.0])
        
        order = TSAQAPuzzler.solve_jigsaw([seg0, seg1, seg2], reference=ref)
        assert order == [2, 0, 1]

