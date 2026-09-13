# 002 — Stage-1 measurement prerequisites wired (collider fix, Holm, no-op guard)

---
id: 002
date: 2026-06-21
stage: 1
claim: none   # tooling remediation — wires the measurement plumbing; moves no experimental claim yet
status: note   # not an experiment; closes the gate-prerequisites ADR-007/02_react_stages flagged as open
evidence_level: scaffold
claim_scope: n/a
overall_significance: n/a
prereg: research_journal/02_react_stages.md (Stage 1 promotion criteria)
commit: numeric-head-visual-trust (this commit; see git log)
artifact_present: n/a   # no run yet — this entry ships the instrument, not a result
required_caveats: "the Stage-1 loop is still inert by default; no per-branch promotion has been measured. This only makes the eventual measurement valid."
---

## Why this entry exists
ADR-007 and `02_react_stages.md` recorded two **open promotion-gate prerequisites**
(2026-06-21 code review), both load-bearing before the Stage-1 loop may be enabled. Codex's
inert-by-default Stage-1 scaffold landed on `numeric-head-visual-trust`; this entry wires the
measurement plumbing that scaffold needs so the *first* per-branch McNemar isn't conditioned on
a collider. The scaffold is Codex's; the plumbing + tests below are the review remediation.

## What was closed
**(a) #1 — stable `initial_branch_used` on every row.** The reroute previously surfaced the
original branch only when it fired, so non-looping rows had no initial-route column to join on.
`run_pipeline` now records `initial_branch_used` for *every* row (pre-reroute; the multi_branch
orchestrator's pick when it overrides the router). Serialized by all three writers (TSExam
`slim_row`, TSRBench `RESULT_FIELDS`, MMTS `_diag_fields`). Post-reroute `branch_used` is kept as
the *final* branch — the two now differ exactly when the loop moves a row.

**(b) #2 — stratify by the initial route, with Holm/BH on the 6-branch sweep.** All three diff
tools (`paired_diff.py`, `mmts_bench/.../diff_configs.py`, `tsr_bench/.../diff_configs.py`) now
stratify the mechanism view by `initial_branch_used` (fallback to `branch_used` for pre-existing
runs, where loop-off ⇒ they are equal) and report **Holm–Bonferroni** adjusted per-branch
p-values. TSRBench's `by_branch` previously emitted discordant *counts with no p-value at all* — it
now emits per-branch McNemar + Holm. Each tool also prints the **initial→final branch-migration
matrix** (promotion criterion: the gain must not be a re-routing artifact). The cross-config
disagreement diagnostic now compares *initial* routes (deterministic at temp 0 ⇒ ~0%), so
loop-induced final-branch divergence no longer reads as router nondeterminism.

**(c) #3 — same-branch reroute no-op guard (beyond the two journaled prerequisites).** At temp 0
the reroute router can re-select the branch the row is already on; re-running identical tools
yields identical evidence and burns the depth budget for zero gain. The loop now breaks *before*
re-execution when `new_branch == current branch` (`loop_stopped="same_branch_noop"`).

## Verification
- New **value-level** tests (not shape-only): `initial_branch_used` captures the pre-reroute
  branch while `branch_used` becomes the final one; the same-branch reroute does **not**
  re-execute the branch tool (call-count asserted == 1); the TSRBench diff stratifies a rerouted
  row under its *initial* branch and reports the migration. (`test_wave1_contracts.py::TestStage1Loop`,
  `tsr_bench/tests/test_heads_and_scripts.py`.)
- Suites green: agentic-setup 71, TSRBench 20, MMTS tools 89, MMTS smoke 20/20. The MCQ
  byte-identity invariant is intact (the new key is additive; predictions unchanged).
- Both diff CLIs smoke-tested end-to-end on a synthetic reroute: Holm column + migration block
  present; the rerouted row counts under its initial branch.

## What changed next
- **Unblocked:** the Stage-1 loop can now be run as a *measured* paired experiment — the promotion
  criteria in `02_react_stages.md` are computable (per-initial-route McNemar + Holm + migration).
- Status flipped "open" → "wired" in ADR-007 (`01_lineage.md`) and the Stage-1 section of
  `02_react_stages.md`.
- **Still open** (not this entry, required before any branch is promoted): the Self-Consistency
  control arm (constraint #4) and the verifier-flag precision/recall report (constraint #2).
