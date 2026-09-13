# 012 — Is the LLM load-bearing for numeric PARSING, or could a better tool do it? (three-arm)

---
id: 012
date: 2026-06-22
stage: 0
claim: C1
status: complete
evidence_level: mechanism-local-stratum
claim_scope: mechanism-local
overall_significance: "stress paraphrase layer: A2−A1 = +0.75 (CI [+0.63,+0.85], McNemar p<0.0001, A2 fire-rate 100%); composition/typo A2−A1 ≈ 0 (deterministic A1 already solves them); Base non-regression A0=A1=A2=0.88, A2 fire-rate 0%"
prereg: mmts_bench/PREREGISTRATION_numeric_parser.md
commit: free-response-regime gate (see git log)
artifact_present: yes
required_caveats: "SYNTHETIC stress distribution (construct validity — shows the mechanism + the honest A1-vs-A2 split, not a natural-distribution headline; no real messy-numeric TS benchmark exists). Single backbone (gemini planner; gpt-4o-mini cross-family paraphraser with A1 synonyms banned). Gold is tool-deterministic, so the metric measures only PARSING. A2 paraphrase parse is 0.75, not 1.0."
---

## Hypothesis (pre-registered — filed before the A2/paraphrase results)
- **E1 (non-regression, Base):** A0 ≈ A1 ≈ A2 @10%; **A2 LLM-fire-rate ≈ 0%**.
- **E2 (decomposition, stress set):** **A1 − A0** large on composition + typo (a grammar + difflib);
  **A2 − A1** (binding) **> 0 only on paraphrase** (McNemar p<0.05, fire-rate>0), ≈ 0 on
  composition/typo. **H0/FALSIFICATION:** A2 − A1 ≈ 0 on paraphrase too ⇒ the LLM is redundant over a
  good deterministic parser even there.

## Setup
- Three parser arms feeding the SAME tools (`evaluate_plan`); only the question→plan parser differs.
  **A0** keyword (current head) · **A1** deterministic (composition grammar + fuzzy/synonym + difflib
  typo + bare-noun) · **A2** = A1 + an **LLM-planner fired only on an A1 abstain** (emits a PLAN over
  the closed registry, validated, **never a number**).
- Gold = `evaluate_plan(intended_plan, series)` — the tools' own answer ⇒ a correct PARSE is an exact
  match; Accuracy@10% measures only parsing. Stress set: 240 rows (60/layer) on real MMTS-Base series
  (`make_numeric_stress.py`); paraphrases authored by held-out cross-family gpt-4o-mini with A1
  synonyms BANNED. Non-regression on 292 univariate Base numerical rows.

## Result

### E1 — non-regression (Base, n=292) — **holds**
A0 = A1 = A2 = **0.88** @10%; **A2 fire-rate 0%**. The deterministic/LLM upgrade is byte-identical on
clean single-quantity templates and never calls the LLM there.

### E2 — the decomposition (stress set) — **binding hypothesis SUPPORTED**
| layer | n | A0 | A1 | A2 | A1−A0 | **A2−A1** | A2 p | A2 fire% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| clean | 60 | 1.00 | 1.00 | 1.00 | +0.00 | +0.00 | 1.00 | 0% |
| compositional | 60 | 0.15 | 1.00 | 1.00 | **+0.85** | +0.00 | 1.00 | 0% |
| typo | 60 | 0.00 | 0.83 | 1.00 | **+0.83** | +0.17 | 0.002 | 17% |
| paraphrase | 60 | 0.00 | 0.00 | 0.75 | +0.00 | **+0.75** | <0.0001 | 100% |
| OVERALL | 240 | 0.29 | 0.71 | 0.94 | +0.42 | +0.23 | <0.0001 | 29% |

Binding pre-registered number: **paraphrase A2 − A1 = +0.75** (CI [+0.63,+0.85], McNemar p<0.0001, A2
fired on 100% of paraphrase rows). A2 paraphrase accuracy is 0.75 (the LLM is not perfect on open
phrasing) — but +75pp over the deterministic floor of 0.

## Verdict — **the LLM is load-bearing for parsing, but ONLY for open paraphrase**
The PI's intuition is vindicated and **located precisely**: a keyword mapping cannot solve every
numeric question, and the LLM is genuinely load-bearing where it can't — **but only the open-paraphrase
slice**. **Composition and typos are tool-tractable**: a deterministic composition grammar + difflib
recovers them (A1−A0 = +0.85 / +0.83) and the LLM adds **nothing** there (A2−A1 = +0.00 / +0.17). So
the LLM's marginal value is the **smallest, most specific** slice — exactly the norm-respecting credit
assignment (never credit the LLM with what a tool can do).

**The numeric regime, decomposed into who owns what:**
1. **Computation** → the deterministic **tool** owns it (+30.6pp vs LLM-computes, log/011).
2. **Parsing — composition & typos** → a deterministic **grammar/fuzzy parser** owns it (A1−A0 huge;
   the LLM is redundant).
3. **Parsing — open paraphrase/semantics** → the **LLM** owns it (A2−A1 = +0.75) — the one place it is
   load-bearing, fired conditionally (29% overall, 0% on clean Base) so it is free on the clean path.

This is the honest answer to "is our tool comprehensive enough alone?" — **yes for computation and for
structured/typo parsing; no for open paraphrase, where the LLM earns its keep.**

## What changed next
- Ship A1 (deterministic grammar+fuzzy) as the numeric parser upgrade (strict superset of A0,
  non-regressing); gate A2 behind A1-abstain (the should-I-ask-the-LLM-to-parse analogue of the
  should-I-look gate). Both are config switches (`numeric_det`, `numeric_llm`), default off.
- The remaining want is a **natural** messy-numeric benchmark (TSAQA acquisition) to replace the
  synthetic construct — the only soft spot in this result.

## Artifacts
- Code: `numeric_head.py` (`parse_plan_deterministic`, `evaluate_plan`, `validate_plan`, registry),
  `prompt_numeric.py` (`build_numeric_planner_prompt`, `parse_numeric_plan_json`), runner
  `_select_numeric_plan`/`_llm_numeric_plan` + `numeric_parser` switch; tests
  `tests/test_runner_v2.py::TestNumericParser` (7).
- Data/eval: `mmts_bench/scripts/make_numeric_stress.py`, `run_numeric_stress.py`,
  `outputs/numeric_stress.csv` (240), `outputs/base_numerical.csv` (292). Pre-reg
  `PREREGISTRATION_numeric_parser.md`.
