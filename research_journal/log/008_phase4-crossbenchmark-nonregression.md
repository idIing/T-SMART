# 008 — Phase 4: agentic_tsmart cross-benchmark non-regression + OOD-agency clean negative

---
id: 008
date: 2026-06-22
stage: 3
claim: none   # cross-benchmark validation of the agentic scaffold; non-regression null + OOD-agency negative
status: complete
evidence_level: OOD-transfer
claim_scope: overall
overall_significance: "agentic_tsmart TIES baseline on all 3 surfaces: TSExam -0.7pp (p=0.50), MMTS pooled -0.1pp (p=1.0, n=1859), TSRBench -0.5pp (p=0.18, n=1010). No branch/category significantly regresses anywhere (all Holm p=1.0). OOD agency fires (gate up to 21%, executes up to 16%) but nets ~0 — no promotable behavior."
prereg: agentic-setup/PREREGISTRATION_phase4.md
commit: agentic-tsmart Gate-4 (see git log)
artifact_present: yes
required_caveats: "Fresh-PAIRED drift-free (TSRBench fresh baseline 370 vs prior-day 365 = the cross-day drift the fresh pairing avoids). TSRBench overall OA flat-bootstrap CI-lo grazes -1.1pp but is temp-0 NOISE: n.s. (p=0.18), 5 net rows, mechanism net-zero on action-executed rows, no branch/category CI entirely <0. Nothing was promoted — the shipped default remains the Stage-0 tree; agentic_tsmart is validated safe-to-ship-DARK, not a new default."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_phase4.md)
- The fullest agentic config `agentic_tsmart` (= `react_stage3_actions`: gate `quality<0.55`, LLM proposes from the closed registry, ≤2 actions, deterministic fallback) ties-or-beats `baseline` across TSExam → MMTS → TSRBench with **no significant branch/category regression** (the guardrail), AND the OOD-agency test asks whether the proposer *helps* where rows are more ambiguous than TSExam's mono-branch MCQ.

## Setup
- Fresh PAIRED grid {`baseline`, `agentic_tsmart`} run in one session (drift-free), temp 0, gemini-3.1-flash-lite. MMTS `--mcq-only` (4 subsets), TSRBench `--compatible-only --sample-mode first --seed 0` (1010). TSExam reused (`react_s3_full`, log/007). Baselines run FRESH (not the 2026-06-18 grid) so the pairing carries no cross-day drift.

## Result — cross-benchmark scoreboard (treat − base)
| Surface | n | base OA | agentic OA | ΔOA | McNemar p | guardrail |
|---|---|---|---|---|---|---|
| TSExam | 746 | 0.682 | 0.676 | −0.7pp | 0.50 | PASS |
| MMTS Base | 274 | 0.456 | 0.460 | +0.4pp | 1.00 | PASS |
| MMTS InWild | 1065 | 0.485 | 0.484 | −0.2pp | 0.80 | PASS |
| MMTS Match¹ | 400 | 0.590 | 0.590 | 0.0pp | 1.00 | PASS |
| MMTS Align | 120 | 0.292 | 0.292 | 0.0pp | 1.00 | PASS |
| **MMTS pooled** | 1859 | 0.491 | 0.491 | −0.1pp | 1.00 | PASS |
| TSRBench² | 1010 | 0.366 | 0.361 | −0.5pp | 0.18 | PASS |

¹ Match = N-way `_run_multi_series_pipeline` (bypasses the loop) → exact tie, agency 0% — a built-in negative control.
² TSRBench flat-bootstrap CI-lo −1.09pp grazes −1pp but is n.s. noise (mechanism net-0); the only branch movement is `trend` −1.8pp, **Holm p=1.0**.
- Router-branch disagreement **0.0%** on every surface; **0 errors**; **0 branch migration** (initial==final on all rows). The only per-branch movement anywhere is `trend` (MMTS pooled −2.1pp, TSRBench −1.8pp) — Holm p=1.0 on both, the same recurring noise wobble as logs 004/005/007.

## OOD agency test (the interesting question) — clean negative
| Surface | gate (q<0.55) | action-executed | corrected/broken/net |
|---|---|---|---|
| MMTS Base | 6.5% | 5.9% (18) | 0 / 0 / **0** |
| MMTS InWild | 19.6% | 15.7% (170) | 5 / 9 / **−4** |
| TSRBench | 20.8% | 14.5% (146) | 1 / 1 / **0** |
The proposer is **demonstrably alive OOD** (gate fires up to ~21%, executes diverse registry actions — vision/anomaly/trend/periodicity — on up to ~16% of rows) but **net discordance is 0 or slightly negative everywhere**. InWild (most ambiguous) is where it fires hardest and nets −4 (Holm-n.s., concentrated in trend). **No promotable OOD behavior: agency does not pay anywhere.**

## Verdict
**Non-regression VALIDATED across all three benchmarks; OOD-agency hypothesis REJECTED.** `agentic_tsmart` (the full agentic scaffold) is a clean tie everywhere — safe to ship **dark**. Nothing earned per-branch promotion on any surface, so the shipped T-SMART **remains the Stage-0 decision tree + structured vision** (+ the prior-validated nu_ad_fix/learned-gate). The Avalanche's ReAct arc is, end-to-end, a rigorous **null with zero regression** — the contribution is the *method* (a fully built, byte-identity-safe, pre-registered, cross-benchmark-validated agentic scaffold) and the *negative result* (constrained self-correction is inert on structured time-series MCQ, in-distribution and OOD), not an accuracy lever.

## What changed next
- `agentic_tsmart` exists as a validated-safe config (3 harnesses, byte-identical alias of react_stage3_actions). Default ships baseline. The agency value, if any, remains a question for free-response / numeric rows or a learned should-act gate (logs 004–007 thesis).
- Honest deviation: added ReAct agency diagnostic columns to the MMTS/TSRBench result CSVs (`quality_score`, `propose_*`, `loop_used`) — purely additive (`extrasaction="ignore"`), the frozen-identity golden test passes, 0 errors, frozen baseline byte-unchanged. Without them the OOD agency analysis was unmeasurable.

## Artifacts
- Runs: `mmts_bench/outputs/phase4/` (paired grid + diffs + `agency_*.txt`), `tsr_bench/outputs/phase4/` (paired + `diff_tsrbench.json`). (Gitignored per repo convention; numbers preserved here.)
- Config `agentic_tsmart` in all 3 CONFIGS. Pre-reg `PREREGISTRATION_phase4.md`. Analyzer `mmts_bench/scripts/phase4_agency.py`. Tests: `TestAgenticTsmartAlias` (3). Full suite **114 passed**; frozen-identity golden guard green.
