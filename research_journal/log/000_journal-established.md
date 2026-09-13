# 000 — Journal established; the staged-ReAct arc begins

---
id: 000
date: 2026-06-21
stage: 0→1 (transition)
claim: C4 (and the framing for C1, C5)
status: complete (founding entry, not an experiment)
commit: 164d28f (numeric-head-visual-trust base)
---

## Why this entry exists
The program's insights were real but **uncaptured** — scattered across
`GENERALIZATION_REPORT.md`, the preregs, `T_SMART_paper_vs_main_audit.md`, sprint dumps,
and past conversations that left no durable trace. The requirement is a direct,
reproducible, **non-black-box** log of the journey. This journal is that log; this is
entry zero.

## State of the program at founding (verified 2026-06-21)
- **Stage 0 complete.** Single-shot tree + on-demand structured vision + **learned
  should-I-look gate** (frozen, commit `20ca22c`). Held-out TSExam: OA 0.6823 → 0.6944
  (+1.2pp, McNemar p=0.46 = tie) at **half** the look-rate (76.1% → 27.9%); rediscovered
  the `nu_ad_fix` wisdom from data. Numeric head: 361/392 @ 92.3% Acc@10%, MCQ
  byte-identical.
- **OOD transfer (MMTS).** Anomaly lever +16.5pp (p=0.024); noise neutral; overall
  +1.0pp n.s. Defensible claim = mechanism transfer + non-regression.
- **TSRBench bringup.** Baseline **365/1010, errors 0** (artifact
  `tsr_bench/outputs/tsrbench_results_baseline_20260621_141547`). The 371/1010 number is a
  **polluted run (error=1)** — do not cite it. Parasol sprint: **Gate A cleared; B/C/D
  open** (memory `parasol-gate-status`); three verified current-impact bugs being fixed by
  Codex.
- **Honesty caveats** (`00_thesis.md`) intact: +8.2pp is backbone-confounded (clean claim
  +3.2pp paired); the anomaly lever is ~5% of rows; the noise lever is neutral.

## The decision logged here
We **begin the staged ReAct transition** (ADR-007, `02_react_stages.md`): ReAct = additive
config switches, evaluated by per-branch paired McNemar against the frozen Stage 0. We do
**not** rewrite to a free agent. The 2026-06-21 trio split: this journal (criteria +
narrative) + Codex (Stage-0 bug fixes + Stage-1 scaffold), to merge.

## Next entries (the queue)
- `001` — Stage-1 bounded self-correction: prereg + per-branch promotion run on TSExam.
- `002` — C1 backbone-agnostic arm: the frozen config under a 2nd backbone, paired.
- `003` — TSRBench Gate D: forecast-head coverage oracle **first**, then a paired lever
  (only if coverage > 0; else pre-register zero-coverage as the result).
- `004` — TSAQA: the strong anomaly lever where it fires on ~100% of rows.

## Links
Thesis `00_thesis.md` · Lineage/ADRs `01_lineage.md` · Stages `02_react_stages.md` ·
Related work `03_related_work.md` · Memory: `project-audit-tsmart`,
`tsmart-next-phase-agenda`, `tsmart_learned_gate_built`, `parasol-gate-status`.
