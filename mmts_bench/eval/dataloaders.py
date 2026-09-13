# tsqa/eval/dataloaders.py
"""
MMTSBenchAdapter — loads and standardises the MMTS-Bench dataset.

Supported subsets
-----------------
'Base'   → Benchmark/Base/synthtic_qa_700.csv        (~700 rows)
'InWild' → Benchmark/InWild/tsqa_1084.csv            (~1 084 rows)
'Match'  → Benchmark/Match/qats_400.csv              (~400 rows)
'Align'  → Benchmark/Align/caption2ts_qa_120.csv +
           Benchmark/Align/ts2caption_qa_120.csv     (~240 rows)

Standardised output dict
------------------------
{
    "sample_id"        : int | str
    "category"         : str          e.g. 'trend analysis'
    "ts_array"         : np.ndarray   univariate -> 1-D float array
                                      multivariate (same length) -> 2-D float array
                                      dual-series (diff lengths) -> object array
    "ts1"              : np.ndarray   first series  (dual-series rows only)
    "ts2"              : np.ndarray   second series (dual-series rows only)
    "is_dual"          : bool         True when ts1/ts2 are populated
    "reference_series" : np.ndarray   reference series (multi-series rows only)
    "candidate_series" : list[np.ndarray]  candidate series, one per answer
                                      option, in A/B/C/D order (multi-series only)
    "is_multi"         : bool         True when reference/candidate_series populated
    "query"            : str
    "options"          : str
    "ground_truth"     : str
    "qa_type"          : str          native label: 'a multiple choice question'
                                      | 'a binary choice question' | 'numerical'
                                      (independent ground-truth for the
                                      schema-detection audit; "" if absent)
    "domain"           : str
    "subset"           : str
}
"""

import ast
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subset → relative file path(s) within a local MMTS-Bench directory
# ---------------------------------------------------------------------------
_SUBSET_FILES = {
    "Base":   ["Benchmark/Base/synthtic_qa_700.csv"],
    "InWild": ["Benchmark/InWild/tsqa_1084.csv"],
    "Match":  ["Benchmark/Match/qats_400.csv"],
    "Align":  [
        "Benchmark/Align/caption2ts_qa_120.csv",
        "Benchmark/Align/ts2caption_qa_120.csv",
    ],
}

_HF_DATASET_NAME = "LuckyLittleStar/MMTS-Bench"
_TASK_TYPE_COL   = "task_type"


# ---------------------------------------------------------------------------
# Core array parsing
# ---------------------------------------------------------------------------

def _to_array(parsed) -> np.ndarray:
    """
    Convert a parsed Python object (list / list-of-lists) to a numpy array.

    Rectangular shapes  → float64 array (1-D or 2-D)
    Inhomogeneous shapes → object array of individual float arrays
                           (dual series of different lengths)
    """
    try:
        return np.array(parsed, dtype=float)
    except (ValueError, TypeError):
        # Inhomogeneous shape — dual series with different lengths
        try:
            arrays = [np.array(sub, dtype=float) for sub in parsed]
            return np.array(arrays, dtype=object)
        except Exception:
            raise ValueError("Inhomogeneous conversion failed")


def _parse_ts_value(raw) -> np.ndarray:
    """
    Parse the 'value' column into a numpy array.
    Never raises — logs warnings on failures and returns np.array([]).
    Truncation increased to 200 chars so full values are visible in logs.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.array([])

    if isinstance(raw, (np.ndarray, list)):
        try:
            return _to_array(raw)
        except Exception as e:
            logger.warning("_parse_ts_value: could not convert list/array — %s", e)
            return np.array([])

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return np.array([])
        try:
            parsed = json.loads(raw)
            return _to_array(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(raw)
            return _to_array(parsed)
        except Exception as e:
            logger.warning(
                "_parse_ts_value: failed to parse value %r — %s. "
                "Row will have empty ts_array.", raw[:200], e
            )
    else:
        logger.warning(
            "_parse_ts_value: unexpected type %s for value %r", type(raw), raw
        )
    return np.array([])


def _split_dual_series(arr: np.ndarray):
    """
    Given a parsed ts_array, determine if it's a dual-series row and split.

    Returns (ts1, ts2, is_dual):
      - is_dual=True  when arr is an object array of 2 sub-arrays (diff lengths)
                      or a 2-D float array with exactly 2 rows
      - is_dual=False for all univariate / other multivariate rows
    """
    if arr.dtype == object and arr.ndim == 1 and len(arr) == 2:
        # Inhomogeneous dual series (different lengths)
        try:
            ts1 = np.asarray(arr[0], dtype=float)
            ts2 = np.asarray(arr[1], dtype=float)
            return ts1, ts2, True
        except Exception:
            pass
    if arr.dtype != object and arr.ndim == 2 and arr.shape[0] == 2:
        # Rectangular dual series (same length)
        return arr[0], arr[1], True
    return np.array([]), np.array([]), False


def _split_multi_series(arr: np.ndarray):
    """
    Detect rows with 3+ candidate series (e.g. MMTS-Bench Match subset
    "choose it" / "smooth it" / "find it" / "reverse it" categories, where
    the question presents one reference series plus several candidates and
    asks which candidate matches).

    Convention (matches the option ordering "Time Series 2, 3, 4, 5..."):
      - row[0]   = reference series
      - row[1:]  = candidate series, one per answer option (A, B, C, D...)

    Returns (reference, candidates, is_multi):
      - is_multi=True  when arr is a 2-D float array with 3+ rows, or an
                       object array of 3+ sub-arrays (different lengths)
      - is_multi=False otherwise (handled by _split_dual_series or as
                       single-series)
    """
    if arr.dtype != object and arr.ndim == 2 and arr.shape[0] >= 3:
        return arr[0], [arr[i] for i in range(1, arr.shape[0])], True
    if arr.dtype == object and arr.ndim == 1 and len(arr) >= 3:
        try:
            series = [np.asarray(s, dtype=float) for s in arr]
            return series[0], series[1:], True
        except Exception:
            pass
    return np.array([]), [], False


# ---------------------------------------------------------------------------
# Column normalisation (HuggingFace schema → local CSV schema)
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "id":             "num",
        "taxonomy_level": _TASK_TYPE_COL,
        "time_series":    "value",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _extract_category(row: pd.Series) -> str:
    raw = row.get(_TASK_TYPE_COL) or row.get("taxonomy_level") or ""
    if isinstance(raw, list):
        return ", ".join(str(v).strip() for v in raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return ", ".join(str(v).strip() for v in parsed)
            except Exception:
                pass
        return raw
    return str(raw)


def _parse_options(raw) -> str:
    """
    Normalise the option column — can be a Python-literal list or plain string.
    Returns a clean comma-separated string, e.g. "A, B, C, D".
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    if isinstance(raw, list):
        return ", ".join(str(o).strip() for o in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return ", ".join(str(o).strip() for o in parsed)
            except Exception:
                pass
        return s
    return str(raw)


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class MMTSBenchAdapter:
    """
    Adapter that loads MMTS-Bench data from local CSVs or HuggingFace,
    filters by subset, cleans NaNs, and yields standardised dicts.

    Parameters
    ----------
    data_path : str
        Path to the local MMTS-Bench root directory, OR "huggingface".
    subset : str
        One of 'Base', 'InWild', 'Match', 'Align', 'All'.
    """

    VALID_SUBSETS = list(_SUBSET_FILES.keys()) + ["All"]

    def __init__(self, data_path: str, subset: str = "Base"):
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. Choose from {self.VALID_SUBSETS}."
            )
        self.data_path = data_path
        self.subset    = subset
        self.dataset   = self._load_and_filter()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_local(self, subset: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for rel in _SUBSET_FILES[subset]:
            full = Path(self.data_path) / rel
            if not full.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {full}\n"
                    f"Make sure data_path points to the MMTS-Bench root directory."
                )
            df = pd.read_csv(full, low_memory=False)
            df["benchmark_set"] = subset
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def _load_huggingface(self, subset: str) -> pd.DataFrame:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install 'datasets' to use HuggingFace loading: pip install datasets"
            ) from e
        ds = load_dataset(_HF_DATASET_NAME, split=subset.lower())
        df = ds.to_pandas()
        df = _normalise_columns(df)
        df["benchmark_set"] = subset
        return df

    def _load_and_filter(self) -> pd.DataFrame:
        use_hf = str(self.data_path).strip().lower() == "huggingface"
        subsets = list(_SUBSET_FILES.keys()) if self.subset == "All" else [self.subset]

        frames = []
        for s in subsets:
            df = self._load_huggingface(s) if use_hf else self._load_local(s)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["question", "value"], how="all")

        for col in ["question", "option", "answer", _TASK_TYPE_COL, "domain"]:
            if col in combined.columns:
                combined[col] = combined[col].fillna("")

        return combined.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict]:
        """
        Yield one standardised dict per sample.

        Dual-series rows (Similarity / Causality subsets):
            is_dual = True
            ts1     = first series  (float array)
            ts2     = second series (float array)

        Multi-series rows (Match subset "choose it" / "smooth it" /
        "find it" / "reverse it" — N candidate series to compare against
        one reference, answer options map 1:1 to candidates):
            is_multi          = True
            reference_series  = the reference series (float array)
            candidate_series  = list of candidate float arrays, in the
                                same order as the answer options (A, B, C...)

        All other rows (single series):
            is_dual = False, is_multi = False
            ts_array = 1-D or 2-D float array
        """
        for idx, row in self.dataset.iterrows():
            sample_id = row.get("num", idx)
            if pd.isna(sample_id):
                sample_id = idx

            ts_array          = _parse_ts_value(row.get("value"))
            ts1, ts2, is_dual = _split_dual_series(ts_array)

            reference_series, candidate_series, is_multi = (
                np.array([]), [], False
            )
            if not is_dual:
                reference_series, candidate_series, is_multi = (
                    _split_multi_series(ts_array)
                )

            yield {
                "sample_id":        sample_id,
                "category":         _extract_category(row),
                "ts_array":         ts_array,
                "ts1":              ts1,
                "ts2":              ts2,
                "is_dual":          is_dual,
                "reference_series": reference_series,
                "candidate_series": candidate_series,
                "is_multi":         is_multi,
                "query":            str(row.get("question", "")),
                "options":          _parse_options(row.get("option")),
                "ground_truth":     str(row.get("answer", "")),
                "qa_type":          str(row.get("qa_type", "")),
                "domain":           str(row.get("domain", "")),
                "subset":           str(row.get("benchmark_set", self.subset)),
            }

    def category_counts(self) -> pd.Series:
        return self.dataset[_TASK_TYPE_COL].fillna("unknown").value_counts()
