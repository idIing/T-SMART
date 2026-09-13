# 007 — Stage 3: constrained LLM-proposed actions — pre-registered H0 (proposer acts, evidence net-neutral)

---
id: 007
date: 2026-06-22
stage: 3
claim: none   # Stage-3 LLM-proposer mechanism test; pre-registered H0 reached, nothing promoted
status: complete
evidence_level: preregistered-H0
claim_scope: per-initial-route-branch
overall_significance: "react_s3 vs baseline ΔOA=-0.7pp McNemar p=0.50 (tie); gate 129/746, actions executed on 120/746; action-executed discordance 6 corrected / 8 broken (net -2, inside noise floor); no branch survives Holm."
prereg: agentic-setup/PREREGISTRATION_stage3.md
commit: agentic-tsmart Gate-3 (see git log)
artifact_present: yes
required_caveats: "the LLM proposer DOES execute real actions on 120 rows (breaking Stage-2's 0-candidate bottleneck) and ranges across the registry (vision 83, anomaly 66, trend 43, ...), but the accumulated evidence is NET-NEUTRAL where it fires (6 vs 8). The -0.7pp overall is temp-0 noise; the trend -5.6pp (Holm 0.27, n.s.) is the SAME recurring wobble seen in the Stage-1/2 non-firing arms, not a Stage-3 effect."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_stage3.md)
- **H1:** the LLM-proposer (gated on `quality_score < 0.55`, closed 8-action registry, ≤2 actions/row) lifts ≥1 initial-route branch (per-branch McNemar CI ≥ 0 post-Holm, no frozen regress, effect localizes to action-executed rows, beats Stage-2's keyword null).
- **H0(a):** proposer adds no value because it can't find a useful action (degenerate proposals). **H0(b):** proposer executes useful-looking actions but the accumulated evidence is net-neutral. Either is a valid logged result.
- **Decision rule:** promote branch *b* iff per-initial-route paired McNemar CI lower bound ≥ 0 after Holm AND no frozen regress AND effect localizes to action-executed rows.

## Setup
- Configs: `baseline` vs `react_stage3_actions` (loop_mode="propose", max_loop_depth 2, gate `quality_score<0.55`; LLM proposes one action from the closed registry {6 branches + vision + numeric_head}, typed-schema-validated, executed deterministically with evidence accumulated in the tool layer; proposal prompt omits options; ≤1 proposal call/step). 746 TSExam rows, reused `baseline_full`. Backbone gemini-3.1-flash-lite temp 0.

## Result — mechanism rates + the action-executed headline
```
GATE  (quality<0.55)        129/746 (17.3%)   — alive, matches Stage-2's 123
PROPOSAL made (LLM called)  129/746           — every gated row
ACTION executed (>=1)       120/746 (16.1%)   — vs Stage-2 keyword null = 0  ← bottleneck broken
FALLBACK only               9/746  (numeric_head_unavailable_on_mcq — correct no-op)
budget over-runs            0     (max 2 executed/row, 1 proposal call/step)
proposed→executed: vision 83→83, anomaly 66→66, trend 43→43, periodicity 24→24, noise 15→15, causality 9→9, numeric_head 9→0
```
**Action-executed-row discordance (independently recomputed by meta):**
```
action-executed  n=120  corrected=6  broken=8  unchanged=106  net=-2   ← inside ±1.3pp noise floor
gated-no-action  n=9    corrected=0  broken=0  unchanged=9    net= 0
non-gated        n=617  corrected=9  broken=12 unchanged=596  net=-3   ← pure temp-0 noise
```
Paired (Holm): OVERALL −0.7pp (p=0.50, tie); every branch Holm p ≥ 0.27. The trend −5.6pp (Holm 0.27) is the **identical** non-firing-arm wobble from log/004 & log/005 — propose touched 57 trend rows but the discordance does **not** localize there. vision-fire 76.1%→78.2% (the +15 executed vision proposals on borderline rows).

## Verdict
**H0(b).** Promote no branch (`react_s3 = ∅`, shipped dark — zero net regression). The LLM proposer **broke Stage-2's candidate bottleneck** (executed 120 actions vs 0) and proposed sensibly across the registry, but the accumulated second-branch / extra-vision evidence is **net-neutral where it fires**. The binding constraint is no longer the trigger (Stage 1) or the candidate source (Stage 2) — it is that **a second analysis on a mono-branch MCQ has nothing decisive to add, and the additive vision gate has already looked.**

## The three-stage synthesis (the real result of the Avalanche's ReAct arc)
| stage | mechanism | trigger | fires? | acts? | net effect | bottleneck |
|---|---|---|---|---|---|---|
| 1 | re-route | `evidence_incomplete` | **0/746** | — | tie | trigger dormant (tools emit complete evidence) |
| 2 | refine (keyword 2nd branch) | `quality<0.55` | 123 | **0** | tie | candidate generator finds nothing |
| 3 | LLM-proposed actions | `quality<0.55` | 129 | **120** | **net −2 (tie)** | a 2nd tool adds nothing decisive on mono-branch MCQ |

**Conclusion (pre-registered H0, three independent ways):** verifier-/quality-gated bounded self-correction is **safe but inert** on structured time-series MCQ — in-distribution (TSExam) and, by the log/004 OOD analysis, on MMTS MCQ too (fallback ~0%). This is exactly the regime the literature warns of (ungated loops degrade; gated loops need a real signal — `03_related_work.md` §B). The agentic value, if any, lives in **free-response / numeric** rows or a **learned should-act gate**, not in evidence-completeness/quality-triggered tool self-correction. The scaffold is built, byte-identity-safe, and ready for those surfaces.

## What changed next
- `react_s3 = ∅`. **`agentic_tsmart` therefore inherits nothing from Stages 1–3** → it equals the frozen Stage-0 tree + structured vision (the ReAct switches ship dark, zero regression). Phase 4 becomes a *non-regression* validation, not a promotion.

## Artifacts
- Run: `research/outputs/react_s3_full` (746, 0 errors) + `paired_react_s3.txt`. `baseline_full` reused (untouched).
- New module `tsqa/orchestrator/action_proposer.py` (typed schema + closed registry + parser-failure contract). Pre-reg `PREREGISTRATION_stage3.md`.
- Tests: `tests/test_wave1_contracts.py::TestStage3*` (19; byte-identity, budget, registry-closed, fallback, SVI single-answer-call). Full suite **111 passed**.
