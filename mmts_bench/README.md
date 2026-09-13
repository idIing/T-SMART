# MMTS-Bench Evaluation Module

Part of the [T-SMART](https://github.com/idIing/T-SMART) project —
**T-SMART: Mechanism-Level Attribution for Tool-Augmented Time-Series Question Answering**.

This module provides the evaluation infrastructure for [MMTS-Bench](https://huggingface.co/spaces/MMTS-Bench/MMTS-Bench-Explorer) — loading the dataset, comparing results against literature baselines, and running hardened math tools that never crash the eval loop.

---

## What's Inside

```
mmts_bench/
├── eval/
│   ├── dataloaders.py       # Loads all 4 MMTS-Bench subsets
│   └── baseline_matrix.py   # Baseline comparison + LaTeX export
├── tools/
│   ├── anomaly_score.py     # Z-score and IQR anomaly detection
│   ├── trend_est.py         # Linear trend + Hurst exponent
│   └── granger.py           # Granger causality test
├── tests/
│   └── test_tools.py        # 51 unit tests for all math tools
├── smoke_test.py            # End-to-end pipeline check
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:idIing/T-SMART.git
cd T-SMART/mmts_bench
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Obtain the MMTS-Bench data**

MMTS-Bench is a third-party benchmark and is **not** redistributed in this repo.
Get it from the authors' official release — the
[MMTS-Bench Explorer](https://huggingface.co/spaces/MMTS-Bench/MMTS-Bench-Explorer)
links the current download — and extract it so the layout is:

```
mmts_bench/
├── MMTS-BENCH/
│   └── Benchmark/
│       ├── Base/
│       ├── InWild/
│       ├── Match/
│       └── Align/
```

The harnesses read `mmts_bench/MMTS-BENCH/` by default. To keep the data outside
the repo, point `MMTS_BENCH_PATH` at wherever it lives:

```bash
export MMTS_BENCH_PATH=/path/to/MMTS-BENCH
```

---

## Verify Everything Works

```bash
python3 smoke_test.py
```
Expected: **20 passed, 0 failed**

```bash
pytest tests/test_tools.py -v
```
Expected: **51 passed**

---

## Usage

### Load the Dataset

```python
from eval.dataloaders import MMTSBenchAdapter

# subset options: 'Base', 'InWild', 'Match', 'Align', 'All'
adapter = MMTSBenchAdapter("./MMTS-BENCH", subset="Base")
print(f"Loaded {len(adapter)} samples")

for sample in adapter:
    print(sample["query"])        # question text
    print(sample["ts_array"])     # numpy array of time series values
    print(sample["ground_truth"]) # correct answer letter
    print(sample["category"])     # e.g. 'trend analysis'
    break
```

### Compare Against Baselines

```python
from eval.baseline_matrix import DeltaMatrix

dm = DeltaMatrix()

# Pass in your model's accuracy per category (0-100 scale)
dm.ingest_results({
    "trend analysis":       38.5,
    "seasonality analysis": 36.2,
    "noise analysis":       34.1,
    "volatility analysis":  44.0,
    "deductive reasoning":  37.5,
    "caption":              75.0,
}, method="ZS")  # options: ZS, ZS_CoT, FS, FS_CoT

print(dm.summary())            # readable table in terminal
print(dm.export_latex_table()) # paste directly into paper
```

### Use the Math Tools

```python
from tools.anomaly_score import zscore_anomaly_score, iqr_anomaly_score
from tools.trend_est     import linear_trend, hurst_exponent
from tools.granger       import granger_causality

ts = [1.2, 1.4, 1.3, 1.5, 99.0, 1.4, 1.3]

print(zscore_anomaly_score(ts))
# {'anomaly_indices': [4], 'anomaly_scores': [6.2], 'n_anomalies': 1}

print(linear_trend(list(range(100))))
# {'slope': 1.0, 'direction': 'upward', 'r_squared': 1.0, ...}
```

> All math tools return a descriptive `"Error: ..."` string instead of
> crashing on bad input — empty arrays, all-NaN, too-short series — so
> the eval loop never breaks mid-run.

---

## Superseded: early Qwen2.5-7B-Instruct exploration

> ⚠️ **These are not the paper's results.** This table is a preliminary
> prompting-only study on a Qwen2.5-7B-Instruct backbone, kept for the record.
> It predates the T-SMART pipeline and shares no configuration with it. The
> reported results use `gemini-3.1-flash-lite` — see [`../RESULTS.md`](../RESULTS.md)
> (Table 2) and [`GENERALIZATION_REPORT.md`](GENERALIZATION_REPORT.md).

| Method         | Base  | InWild | Match | Align | Overall |
|----------------|-------|--------|-------|-------|---------|
| Zero-Shot      | 16.71 | 40.50  | 29.50 | 74.58 | 35.19   |
| Zero-Shot+CoT  | 15.86 | 40.22  | 31.50 | 74.17 | 35.11   |
| Few-Shot       | 16.29 | 41.70  | 42.50 | 75.00 | **37.79**   |
| Few-Shot+CoT   | 17.43 | 40.22  | 40.00 | 66.25 | 36.18   |

### Gap vs Literature Baselines (Zero-Shot)

| Category            | TS-Agent | ChatTS | Ours  | Δ vs TS-Agent |
|---------------------|----------|--------|-------|----------------|
| Trend Analysis      | 47.0     | 52.0   | 34.7  | -12.3          |
| Seasonality         | 52.0     | 55.0   | 34.5  | -17.5          |
| Noise Analysis      | 38.0     | 40.0   | 32.7  | -5.3           |
| Volatility          | 45.0     | 48.0   | 43.2  | -1.8           |
| Deductive Reasoning | 40.0     | 43.0   | 36.1  | -3.9           |
| Caption/Align       | 68.0     | 72.0   | 74.6  | **+6.6** ✅    |

---

## Dataset Subsets

| Subset  | Size     | Description                                          |
|---------|----------|------------------------------------------------------|
| Base    | 700      | Synthetic QA — trend, seasonality, noise, volatility |
| InWild  | 1,084    | Real-world time series from diverse domains          |
| Match   | 400      | Pattern matching and sequence recognition            |
| Align   | 240      | Cross-modal: time series ↔ caption alignment         |
| **All** | **2,424**| All subsets combined                                 |

---

## Team

The T-SMART authors — [T-SMART](https://github.com/idIing/T-SMART).
Licensed under the MIT License; see [`../LICENSE`](../LICENSE).
