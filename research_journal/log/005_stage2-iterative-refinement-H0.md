# 005 — Stage 2: iterative tool refinement — pre-registered H0 (gate alive, candidate generator inert)

---
id: 005
date: 2026-06-22
stage: 2
claim: none   # Stage-2 refinement mechanism test; pre-registered H0 reached, nothing promoted
status: complete
evidence_level: preregistered-H0
claim_scope: per-initial-route-branch
overall_significance: "react_s2 vs baseline ΔOA=-1.2pp McNemar p=0.122 (tie); quality<0.55 fires 123/746 (16.5%) but refine_fired=0 (no 2nd-branch candidates); no branch survives Holm."
prereg: agentic-setup/PREREGISTRATION_stage2.md
commit: agentic-tsmart Gate-2 (see git log)
artifact_present: yes
required_caveats: "the quality gate is ALIVE (123/746 sub-0.55, computed where baseline had None) but the keyword candidate generator returns 0 second-branches on every sub-threshold row ⇒ refine never acts. The −1.2pp wobble is temp-0 noise (all 29 pred-diffs on non-fired rows). Noise floor ≈ ±1.3pp."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_stage2.md)
- **H1:** bounded quality-gated refinement (accumulate a 2nd branch when `quality_score < 0.55`, depth ≤ 2, tool-layer only) lifts ≥1 initial-route branch (per-branch McNemar CI ≥ 0 post-Holm), no frozen branch regresses, effect localizes to refine-fired rows.
- **H0:** dormant/neutral — a valid logged result.
- **Decision rule:** promote branch *b* iff per-initial-route paired McNemar CI lower bound ≥ 0 after Holm AND no frozen branch regresses AND the effect localizes to refine-fired rows. Trigger = the pre-existing principled `QUALITY_THRESHOLD=0.55` (non-circular).

## Setup
- Configs: `baseline` vs `react_stage2_refine` (loop_mode="refine", max_loop_depth 2, quality threshold 0.55; computes `evaluate_quality` on the primary branch, accumulates a `get_keyword_candidates` second branch's evidence in the tool layer — **0 extra LLM calls**, SVI-preserving). Rows: TSExam test n=746, reused `baseline_full`. Backbone gemini-3.1-flash-lite temp 0.

## Result — mechanism view (Holm-corrected)
```
noise        186  0.565  0.570  +0.005  Holm 1.000
similarity   162  0.660  0.660  +0.000  Holm 1.000
periodicity  156  0.763  0.750  -0.013  Holm 1.000
trend        126  0.825  0.786  -0.040  6/1  Holm 0.875 (n.s.)
anomaly       82  0.524  0.488  -0.037  3/0  Holm 1.000
causality     30  0.933  0.933  +0.000  Holm 1.000
OVERALL      746  0.682  0.670  -0.012  McNemar p=0.122 (tie)
```
- **Gate fire-rate: `quality_score < 0.55` = 123/746 (16.5%)** — the gate is ALIVE (baseline had `quality_score=None` on all rows; Stage 2 computes it on 742/746). Distribution: min 0.464, median 0.814, mean 0.766. Sub-threshold rows: periodicity 45, trend 51, anomaly 27, noise/similarity/causality 0.
- **Action fire-rate: `refine_fired` = 0/746.** On **all 123** sub-threshold rows, `get_keyword_candidates` surfaces **no second branch** to accumulate, so refine correctly abstains (`refine_depth>0 = 0`, `refine_branches` empty everywhere). vision fired identically 568/746; migration shows 0 refine moves.
- Prediction diffs vs baseline: 29/746 (9 corrected, 18 broken) — **100% on non-fired rows** ⇒ pure temp-0 answer-LLM noise (mirrors Stage-1's 32/746). The trend −4.0pp is Holm-n.s. (0.875), not a refinement effect (refine touched 0 trend rows).

## Verdict
**H0.** Promote no branch (`react_s2 = ∅`, shipped dark — zero regression because zero activation). **Sharper diagnosis than Stage 1:** the binding constraint moved from the *gate* (Stage 1's `evidence_incomplete` never fired) to the *candidate generator* (Stage 2's gate fires on 16.5% of rows, but the keyword candidate source finds no alternative branch on single-branch in-distribution MCQ — a periodicity/trend/anomaly question carries no keywords pointing at a different analysis). The decision tree's tools are not just confident; the questions are *mono-branch by construction*, so there is nothing to accumulate.

## What changed next
- `react_s2 = ∅`. `agentic_tsmart` inherits nothing from Stage 2.
- **Open question → future entry:** a separately-pre-registered Stage-2b could test a richer (still non-LLM, non-circular) second-branch source — but only on a surface where it can be validated without re-using the rows that motivated it. Combined with log/004's OOD analysis (fallback ~0% on MMTS Base/InWild/Match), the emerging thesis is that **evidence-completeness / quality self-correction has little to act on in the structured-MCQ regime**; agentic value, if any, lives in free-response or a learned should-act gate, not these triggers.

## Artifacts
- Run: `research/outputs/react_s2_full` (746). `baseline_full` reused (untouched).
- Pre-reg: `agentic-setup/PREREGISTRATION_stage2.md`.
- Tests: `tests/test_wave1_contracts.py::TestStage2Refine` (8; byte-identity via `trace_hash`, 0-extra-LLM, depth bound, determinism). Full suite **85 passed**.
