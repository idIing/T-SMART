
# 📄 `tsqa/README.md`

## TSQA: Agentic Time Series QA Pipeline

This folder contains an **agentic pipeline for time-series question answering (TSQA)**.

The system combines:

* **LLM-based reasoning (Gemini)**
* **Deterministic time-series tools (FFT, autocorr, etc.)**
* **Structured routing + verification**

The goal is to move beyond static prompting and instead:

> Let the model **decide how to analyze the time series**, gather evidence, and then reason over it.

---

# 🧠 High-Level Overview

At a high level, the pipeline works as:

```
Question + Time Series
        ↓
Router (LLM)
        ↓
Branch (analysis logic)
        ↓
Tools (deterministic features)
        ↓
Structured Evidence (JSON)
        ↓
LLM Answer
        ↓
Verifier (sanity checks)
        ↓
Final Answer (A/B/C/D)
```

---

# 🔁 End-to-End Data Flow (Concrete)

Here is what happens for **one dataset row**:

1. **Input**

   * Question
   * Options (A/B/C/D)
   * Time series (`ts`, `ts1`, `ts2`)

2. **Routing (`tsqa/router`)**

   * LLM classifies:

     * `branch`: trend / periodicity / anomaly / noise / similarity / causality
     * `scope`: global vs local
     * `series_type`: single vs dual

3. **Branch Execution (`tsqa/branches`)**

   * Each branch defines:

     * what tools to run
     * how to structure evidence

4. **Tool Computation (`tsqa/tools`)**

   * Deterministic stats:

     * trend, FFT, autocorr, DTW, Granger, etc.
   * Outputs structured metrics

5. **Evidence Construction**

   * All results are stored as JSON:

     ```
     {
       "slope": ...,
       "best_lag": ...,
       "is_white_noise": ...,
       ...
     }
     ```

6. **LLM Answer (`tsqa/llm`)**

   * Prompt built from:

     * question
     * options
     * structured evidence
   * LLM outputs A/B/C/D

7. **Verification (`tsqa/verifier`)**

   * Rule-based checks:

     * missing signals
     * contradictions
   * May trigger fallback

8. **Final Output**

   * predicted answer
   * correctness
   * logs (routing, evidence, flags)

---

# 📦 Package Structure

```
tsqa/
├── router/        # LLM-based routing
├── branches/      # branch-specific pipelines
├── tools/         # deterministic TS functions
├── verifier/      # sanity checks
├── llm/           # Gemini + prompts + parsing
└── eval/          # pipeline runner + metrics
```

---

# 🔧 Running the Pipeline

From the repo root (see the [top-level README](../README.md) for full setup):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e tsqa
```

`GEMINI_API_KEY` must be in a repo-root `.env` (git-ignored) or the environment.

Example:

```python
from tsqa.eval import run_pipeline
r = run_pipeline(row, client)
```

This returns:

* prediction
* correctness
* routing info
* evidence
* flags

Each run logs routing, evidence, and final prediction.

---

# 🧩 Where to Contribute

## 1. Tools (`tsqa/tools`)

Best place to start.

You can:

* add new statistical features
* improve existing ones

Examples:

* `change_point.py`
* `shape_features.py`

⚠️ Requirements:

* deterministic
* return structured dict
* no LLM calls

---

## 2. Branch Logic (`tsqa/branches`)

Each branch defines:

* what tools to call
* what evidence to return

Example branches:

* trend
* periodicity
* anomaly
* similarity
* causality

Good contributions:

* better feature selection
* cleaner evidence structure
* fixing failure cases

---

## 3. Router (`tsqa/router`)

LLM decides:

* which branch
* which scope

Possible improvements:

* better prompts
* error handling
* routing consistency

---

## 4. Verifier (`tsqa/verifier`)

Currently underutilized.

Opportunities:

* detect contradictions
* enforce constraints
* trigger fallback intelligently

---

## 5. LLM Prompts (`tsqa/llm`)

Controls:

* how evidence is interpreted
* how answers are selected

Possible improvements:

* clearer mapping from evidence → choices
* reducing hallucination

---

## 6. Evaluation (`tsqa/eval`)

Handles:

* dataset runs
* logging
* metrics

You can:

* add new metrics
* analyze failure modes
* improve logging

---

# ⚠️ Important Design Principles

## 1. Tools ≠ Reasoning

* Tools compute facts
* LLM interprets them

Do NOT mix these.

---

## 2. Evidence is the interface

Everything passed to the LLM must go through:

```
tools → structured JSON → LLM
```

---

## 3. No hardcoding answers

Branches should:

* compute signals
* NOT directly choose answers

---

## 4. Keep things modular

* tools should be reusable
* branches should be composable

---

# 🚨 Known Issues / Current Limitations

This file is an architectural orientation for contributors. **It is not the
results record** — for reported numbers see [`../RESULTS.md`](../RESULTS.md) and
the published paper. On TimeSeriesExam the `baseline` config
scores 65.0 macro-OA and `nu_ad_fix` 67.5 (n=746, `gemini-3.1-flash-lite`).

Standing weaknesses:

* Anomaly detection (AD) is the weakest category and motivated the
  `nu_ad_fix` configuration.
* The LLM sometimes:

  * interprets evidence incorrectly
  * maps to the wrong answer choice

---

# 💡 Future Directions

* Better tool selection (agentic decision-making)
* Stronger verification loops
* Multimodal integration (vision)
* Improved evidence → answer mapping

---

# 🧠 Mental Model

Think of the system as:

> **LLM = scientist**
>
> **Tools = instruments**
>
> **Branches = experiment protocols**
