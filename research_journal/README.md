# T-SMART Research Journal

This directory is the **narrative spine** of the project: the durable, reproducible log
of *why* T-SMART is built the way it is and *how* it grows. It exists because the
project's hard-won insights have been scattered across
`mmts_bench/GENERALIZATION_REPORT.md`, the `PREREGISTRATION_*.md` files,
`T_SMART_paper_vs_main_audit.md`, sprint-result dumps, and (mostly) past conversations
that left no durable trace. The journal makes the reasoning **auditable and
non-black-box at every step** — which is itself one of the project's scientific claims.

`CLAUDE.md` remains the *operational* manual (how to run things). This journal is the
*epistemic* one (what we claim, why, and what the evidence says).

## How to read it
1. `00_thesis.md` — the north star and the **falsifiable** form of each claim.
2. `01_lineage.md` — the branch history and the **Architectural Decision Records**
   (why deterministic tools, why on-demand structured vision, why a decision tree
   *first*, why we refuse a from-scratch agent rewrite).
3. `02_react_stages.md` — the **architectural trajectory**: the staged Stage 0 → 3
   transition from decision tree to a bounded ReAct loop, with per-stage promotion
   criteria and the non-regression invariant.
4. `03_related_work.md` — the literature synthesis (positioning vs prior art, the
   self-correction evidence base, TSFM integration, backbone-agnostic methodology).
5. `log/` — **append-only, one entry per experiment.** Every entry is a closed loop:
   hypothesis → pre-registration → config → result table (paired Δ, CI, McNemar) →
   pre-registered verdict → what changed. `log/TEMPLATE.md` is the schema;
   `log/000_journal-established.md` is the founding entry.

## How to use it (the contract)
- **No result enters the program without a log entry.** A run that isn't logged didn't
  happen.
- **Pre-register before you run.** The verdict (H0/H1) is declared *before* seeing the
  number — exactly as the project already does for the vision, gate, and numeric-head
  studies.
- **Link, don't duplicate.** Point to the prereg, the CSV, the commit hash. The journal
  is the index and the story; the artifacts live where they live.

## Artifacts this journal subsumes / links
- `mmts_bench/GENERALIZATION_REPORT.md` — §5 limitations, §6 PhD-level agenda
- `T_SMART_paper_vs_main_audit.md` — paper ↔ `main` discrepancies *(scoped/dated — may be
  stale on model & backbone state; verify against current code before citing)*
- `PREREGISTRATION*.md` — visual-trust, numeric head, learned gate, forecast head, top-2
- `mmts_bench/SPRINT_RESULTS_learned_gate.md` — the learned-gate sprint handoff
- Memory: `project-audit-tsmart`, `tsmart-next-phase-agenda`, `tsmart_learned_gate_built`,
  `tsmart_visual_trust_result`, `parasol-gate-status`
