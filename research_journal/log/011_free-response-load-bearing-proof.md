# 011 — The free-response load-bearing proof: tool numeric head vs an LLM-computes counterfactual (WIN)

---
id: 011
date: 2026-06-22
stage: 0
claim: C1
status: complete
evidence_level: mechanism-local-stratum
claim_scope: mechanism-local
overall_significance: "closed-form @10%: tool 0.997 vs model 0.679, Δ +31.8pp, 104/0 discordant, McNemar p<0.0001, CI [+26.9,+36.7]"
prereg: mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md
commit: free-response-regime gate (see git log)
artifact_present: yes
required_caveats: "SCOPE: this is tool-vs-LLM on COMPUTATION — on the numeric-head arm the LLM is BYPASSED (0 answer-LLM calls; the value is pure numpy + a deterministic keyword→quantity map). It shows deterministic tool OWNERSHIP of computable answers beats LLM computation; it is NOT evidence the reasoning/interpretation layer AUGMENTS the LLM. Where the LLM IS load-bearing in this regime (PARSING composition/paraphrase/typos, which the keyword map can't do) is the separate three-arm study (log/012). Single surface (MMTS Base) and single backbone (gemini-3.1-flash-lite). NOT a config McNemar — it is tool-vs-model on the same 386 numerical rows, both emitting predicted_value, both scored @10% (discordant pairs well-defined). Method-sensitive strata (period/count/slope) reported as bounds, never folded into the headline; slope is a genuine model win there."
---

## Hypothesis (pre-registered)
- **H1 (load-bearing):** on the **closed-form** stratum (std/var/mean/median/min/max/range/
  percentile/argmax/argmin), the deterministic numeric head's Accuracy@10% is **>>** the
  `llm_numeric` counterfactual's (the *same* gemini backbone computing the number from the raw
  series), with McNemar p<0.05 and a bootstrap CI on the gap **entirely > 0** — i.e. the tool is
  load-bearing **even on a strong backbone**, in the one regime where evidence (not interpretation)
  is the binding constraint.
- **H0 / FALSIFICATION:** if the model also reaches ~the head's @10% (no gap), the tools are
  redundant *even on free-response* and the "general framework" claim fails on this surface.
- **Decision rule:** H1 supported iff closed-form Δ>0, CI-lo>0, McNemar p<0.05.

## Setup
- Arms (same 386 MMTS-Base `numerical` rows, same `score_numeric_value` @10%):
  - **TOOL** = deterministic numeric head, REUSED `mmts_results_Base_baseline_20260620_224634.csv`
    (the head makes no answer-LLM call ⇒ drift-free reuse).
  - **MODEL** = `--config llm_numeric` (NEW): the answer LLM computes the number from the raw series,
    numeric head bypassed (`build_llm_numeric_prompt` + `_run_llm_numeric`). Run 2026-06-22, 700
    rows in 27 s, errors 0; model abstention 0.3%.
- Backbone: **gemini-3.1-flash-lite**, temp 0 — the strong backbone (the hard case for H1).

## Result — **H1 strongly SUPPORTED** (analysis: `mmts_bench/scripts/diff_numeric.py`)

| stratum | n | tool @10% | model @10% | Δ (pp) | 95% CI | b/c | McNemar p |
|---|---|---|---|---|---|---|---|
| OVERALL (numerical) | 386 | 0.938 | 0.632 | **+30.6** | [+25.4,+35.8] | 128/10 | <0.0001 |
| **closed-form (PRIMARY)** | 327 | **0.997** | 0.679 | **+31.8** | [+26.9,+36.7] | **104/0** | **<0.0001** |
| method-sensitive (bounds) | 58 | 0.621 | 0.379 | +24.1 | [+5.2,+41.4] | 24/10 | 0.024 |

Per quantity (closed-form): **mean +64.5pp** (tool 1.00 vs model 0.36), **median +51.3**, **std
+43.5**, **percentile +36.3**, **argmin +61.5**, **range +3.2** (model 0.97 — it's max−min), **min /
max +0.0** (both 1.00 — the model *can* scan for extrema). Method-sensitive bounds: **period +60.9**,
count_local_min +62.5 — but **slope −31.6** (model 0.74 > tool 0.42, n.s.): the lone reversal, exactly
where the *tool's own* definition is weak (pre-registered as a bound, not headline).

**Sensitivity sweep (closed-form):** the gap *widens* as tolerance tightens — Δ **+58.7pp @1%**,
+41.6 @5%, +31.8 @10%, +24.5 @20%. The tool is exact; the model approximates, so it loses most where
precision is demanded.

**Mechanism.** On a strong backbone the model can *read off* extrema (min/max tie at 1.00) and
near-trivial composites (range), but it **cannot compute aggregate statistics from raw values** —
mean, median, std, percentile collapse to 0.36–0.68. The deterministic tool **owns** these. This is
the regime where the architecture is load-bearing *even on gemini* — the mirror image of the MCQ
nulls (logs 004–008), where the tools were redundant and agency was inert.

## Verdict — **T-SMART is a general framework in the free-response regime, not an MCQ trick.**
The pre-registered PRIMARY is met decisively (closed-form Δ +31.8pp, 104/0 discordant, p<0.0001,
CI excludes 0). Together with the MCQ nulls this resolves the audit's reframe into one sentence:
**the architecture's value is a function of what the backbone cannot do itself** — inert on
confident-tool mono-aspect MCQ (the model already has the signal), but **+30pp load-bearing** on
free-response numeric (the model cannot compute the number). The +13.5pp-style lift the brief
predicted "should appear even on a strong backbone" appears — at **+30.6pp**.

**Scope — read precisely (added 2026-06-22 after PI scrutiny).** This contrasts *tool computation* vs
*LLM computation*; on the numeric-head arm the LLM makes **zero** answer calls (the value is pure
numpy + a deterministic keyword→quantity map), so the result licenses **"deterministic tool ownership
of computable answers,"** NOT "the reasoning/interpretation layer augments the LLM." On these rows the
framework is a deterministic dispatcher and the LLM is bypassed, not augmenting. Where the LLM *is*
load-bearing in the numeric regime — **parsing** the question into the right computation (composition,
paraphrase, typos), which the keyword map cannot do — is isolated, three-arm, in **log/012**. (Two
rows the failure-audit flagged as "bugs" — 554, 557 — are isolated CSV/data artifacts, not pipeline
bugs: the schema gate is verified **0/700 systematic MCQ leaks**.)

## What changed next
- The numeric head is validated as a **load-bearing** component (not just coverage). The honest
  bound: single surface + single backbone; method-sensitive quantities (slope especially) are where
  a *learned should-ACT gate* could choose the model over the tool — the first concrete place agency
  could pay (gated on this evidence, per the plan; not built for MCQ).
- Item-2 framing: pair this with Track V — the architecture's two defensible wins are (1) tools that
  **compute** what the model can't (free-response, this entry) and (2) **conditional structured
  sycophancy-resistant vision** on shape questions (log/009), each with its own backbone-dependence.

## Artifacts
- Runs: `mmts_bench/outputs/mmts_results_Base_llm_numeric_20260622_104342.csv` (model);
  `..._Base_baseline_20260620_224634.csv` (tool, reused).
- Analysis: `mmts_bench/scripts/diff_numeric.py`. Pre-reg:
  `mmts_bench/PREREGISTRATION_llm_numeric_counterfactual.md`.
- Code: `build_llm_numeric_prompt` (`tsqa/llm/prompt.py`), `_run_llm_numeric` (`tsqa/eval/runner.py`),
  config `llm_numeric` (both harnesses); tests `tests/test_runner_v2.py::TestLLMNumericArm` (4).
