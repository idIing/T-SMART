# 004 — Stage 1: bounded self-correction — pre-registered H0 (trigger dormant on TSExam)

---
id: 004
date: 2026-06-22
stage: 1
claim: none   # Stage-1 self-correction mechanism test; pre-registered H0 reached, nothing promoted
status: complete   # verdict present: clean null
evidence_level: preregistered-H0
claim_scope: per-initial-route-branch
overall_significance: "react_s1 vs baseline ΔOA=-1.3pp McNemar p=0.099 (tie); self_consistency ΔOA=-1.3pp p=0.087 (tie); no branch survives Holm."
prereg: agentic-setup/PREREGISTRATION_stage1.md
commit: agentic-tsmart Gate-1 (see git log)
artifact_present: yes
required_caveats: "registered trigger evidence_incomplete fires 0/746 on TSExam ⇒ loop + Self-Consistency both inert; the −1.3pp wobble and the trend −5.6pp stratum are temp-0 / cross-day answer-LLM NOISE (identical in both non-firing arms), NOT a Stage-1 effect. Re-run noise floor ≈ ±1.3pp / ~30 rows on 746."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_stage1.md)
- **H1:** a bounded ≤1 reroute on `evidence_incomplete` lifts ≥1 initial-route branch (per-branch McNemar CI ≥ 0 post-Holm), no frozen branch regresses, AND it beats the Self-Consistency control.
- **H0:** neutral/negative — a valid, logged result (lit: ungated self-correction degrades reasoning — CRITIC 2305.11738, Kamoi 2406.01297; spend test-time compute re-verifying not re-generating — Lightman 2305.20050).
- **Decision rule:** promote branch *b* iff its per-initial-route paired McNemar CI lower bound ≥ 0 after Holm AND no frozen branch regresses AND the loop beats `self_consistency` on *b* (gain ≠ sampling artifact).

## Setup
- Configs: `baseline` (Stage-0 additive gate) · `react_stage1_evidence_retry` (loop_branches = all 6, max_loop_depth 1, loop_on_flags `[evidence_incomplete]`, reroute_once) · `self_consistency` (k=5 answer-LLM samples @ temp 0.7 on `evidence_incomplete` rows, majority vote — the cheaper control the loop must beat).
- Rows: TSExam test, n=746. Reused `outputs/baseline_full` (2026-06-18; untouched). Backbone gemini-3.1-flash-lite temp 0 (SC samples temp 0.7).

## Result — stratify by initial route (mechanism view, Holm-corrected)
react_s1 vs baseline:
```
noise        186  0.565  0.570  +0.005  [-0.027,+0.038]  5/6  Holm 1.000
similarity   162  0.660  0.660  +0.000  [-0.025,+0.025]  2/2  Holm 1.000
periodicity  156  0.763  0.750  -0.013  [-0.038,+0.013]  3/1  Holm 1.000
trend        126  0.825  0.770  -0.056  [-0.095,-0.024]  7/0  raw 0.0156 / Holm 0.109 (n.s.)
anomaly       82  0.524  0.500  -0.024  [-0.073,+0.024]  3/1  Holm 1.000
causality     30  0.933  0.933  +0.000  [ 0.000, 0.000]  0/0  Holm 1.000
OVERALL      746  0.682  0.669  -0.013  McNemar p=0.099 (tie)
```
self_consistency vs baseline: OVERALL −0.013 (p=0.087); **the trend −0.056 / 7-0 / Holm 0.109 line is IDENTICAL** to react_s1.

- **Trigger fire-rate: `evidence_incomplete` = 0/746 (0.0%)** in all three runs (recomputed independently from raw `flags`). Loop fired 0×; SC sampled 0×. Migration: 0 loop reroutes — the only "moves" are router noise (4 rows → `None` fallback), present in baseline too. vision fired identically 568/746 (76.1%) in both arms.
- Prediction diffs vs baseline: react_s1 **32/746**, selfcons **31/746** — equal magnitude, both arms land at OA 0.6689. Since neither mechanism touched any row, **100% of these flips are temp-0 / cross-day answer-LLM nondeterminism**. The trend −5.6pp appears identically in both non-firing arms ⇒ ~7 boundary trend rows wobbling, **not a Stage-1 regression**.
- **Verifier-flag precision/recall** (constraint #2, `flag_diagnostics.py` on baseline_full): `evidence_incomplete` never fires; only `arch_effects` (lift 1.27) and `adf_kpss_disagree` (1.11) correlate with wrongness; `low_r2`/`high_volatility`/`granger_not_significant`/`fft_unreliable`/`weak_correlation` are at/below chance.

## Verdict
**H0.** Promote NO branch (`react_s1 = ∅`). The registered trigger is **dormant on in-distribution well-formed MCQ** — the deterministic tools always emit "complete" evidence, so `evidence_incomplete` never fires. Bounded self-correction is therefore **safe to ship dark** (zero regression because zero activation), and the Stage-1 *instrument* (matched Self-Consistency control + flag diagnostics + migration plumbing) is validated for surfaces where evidence is genuinely incomplete.

## What changed next
- `react_s1 = ∅`, shipped dark. `agentic_tsmart` inherits nothing from Stage 1.
- **Open question → future entry:** exercise the trigger where it fires. Two honest paths: (a) run `react_stage1_evidence_retry` on MMTS / TSRBench (free-response / harder rows may surface `evidence_incomplete`); (b) a *separately pre-registered* Stage-1b triggering on an informative flag (`arch_effects`) — but that flag was selected using baseline correctness, so it must be validated on a **held-out** surface, never re-tested on the TSExam rows that selected it (no circularity). Not done tonight.
- **Methodological caveat for all later stages:** re-run / cross-day answer-LLM noise floor ≈ ±1.3pp (~30/746). Routing is deterministic (flag counts, fallbacks, vision-fire identical across runs); the answer letter is not. Attribute effects via **mechanism-fired-row** stratification, never raw OA.

## Artifacts
- Runs: `research/outputs/{react_s1_full, selfcons_full}` (746 each). `baseline_full` reused (untouched, mtime 2026-06-18).
- Pre-reg: `agentic-setup/PREREGISTRATION_stage1.md`. Flag report: `agentic-setup/research/flag_diagnostics.py`.
- Tests: `tests/test_runner_v2.py::TestSelfConsistency` (byte-identity), `::TestFlagDiagnostics`. Full suite **77 passed**.
