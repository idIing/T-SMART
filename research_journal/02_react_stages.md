# 02 — Architectural Trajectory: Stage 0 → ReAct

*How we transition a single-shot decision tree into a bounded self-correcting loop
without a rewrite and without regressing. This is the operational form of ADR-007.*

## The reframe (read this first)
**ReAct in T-SMART is not a new architecture. It is a new family of config switches.**

Today's switches are *spatial* — `forced_vision_branches`, `no_vision_branches` decide
*which branch gets which tool*. ReAct adds *temporal* switches — `loop_branches`,
`max_loop_depth`, `loop_on_flags` decide *which branch may act → observe → re-act, and how
deep*. Same `run_pipeline`, same frozen baseline, same per-branch paired McNemar.

Self-correction is therefore **verifier-gated bounded re-entry**, not free LLM looping:
when the verifier flags incomplete/contradictory evidence, the pipeline re-enters once
(or twice) with the flag as context. It is the should-I-look gate (ADR-009) extended from
"should I look" to "should I loop." Each iteration is meant to append to the run's
reasoning trace and a `log/` entry — that is the auditability *contract*. (Reality check:
the runtime `return_trace` is optional, off by default, and emits a compact fingerprint —
not yet a full per-iteration dump; the `log/` entry is the binding record. Closing that gap
is tracked work, not a solved property.)

## The invariant (the same across every stage)
> **Paired McNemar vs the frozen Stage 0, stratified by the *initial route* (the
> pre-treatment branch fixed by temp-0 baseline routing — NOT the final `branch_used`,
> which loops make endogenous/post-treatment), with the discordant-pair cost made explicit
> and an initial→final migration matrix reported alongside. A loop is promoted on an
> initial-route branch only where the discordant math is favorable; it is suppressed
> everywhere it is neutral or negative.**

This is literally how `nu_ad_fix` was found (force vision on anomaly, kill it on noise).
"Maintain accuracy across benchmarks" is operationalized as: *measure the per-branch
regression every stage and gate promotion on it.* Non-regression on frozen branches is a
hard gate, not a hope.

## The stages
### Stage 0 — Frozen control (DONE)
Single-shot tree + on-demand structured vision under the **additive** should-I-look gate
(`vision_gate="additive"`, the runtime default — this is the `baseline` config and the
**null arm** for every paired test). Temp-0, ~0% routing drift. **Never changes.** The
**learned** gate (`vision_gate="learned"`) is a *validated treatment arm* that ties this
control at half the look-rate — a candidate *next* frozen variant, **not** the control
itself. Keep the two distinct or the paired-McNemar null is ambiguous.

### Stage 1 — Bounded self-correction (NEXT)
- **Mechanism.** On verifier flags (`evidence_incomplete`, or a contradiction such as
  vision ⟂ math), allow **one** re-route / re-branch with the flag as context. The
  smallest possible ReAct primitive: act → observe → re-act once.
- **Switch (built, inert-by-default).** `loop_branches=[…]`, `max_loop_depth=1`,
  `loop_on_flags=[…]`, `loop_mode="reroute_once"` — now exposed by
  `run_pipeline`/`run_dataset` as Codex's Stage-1 scaffold (landing on
  `numeric-head-visual-trust`). The defaults (`max_loop_depth=0`, `loop_branches=None`) keep
  it **byte-identical to Stage 0** — verified — so it ships dark until a branch is promoted.
  **Gate-prerequisites (2026-06-21 code review) — now WIRED (`log/002`):** (a) the result records
  a stable `initial_branch_used` for *every* row (pre-reroute; non-looping rows carry it too);
  (b) `diff_configs.py`/`paired_diff.py` stratify by that initial route, apply **Holm–Bonferroni**
  on the 6-branch sweep, and print the initial→final migration matrix. A same-branch reroute no-op
  guard was added for free (`loop_stopped="same_branch_noop"`). The per-branch McNemar is therefore
  no longer conditioned on a collider; the loop may be enabled for a measured paired experiment.
- **Hypothesis (H1).** On the gated branches, the loop converts a measurable share of
  failures-by-misroute (root failure #2) into corrections, with **zero regression** on
  frozen branches (byte-identical where the loop never fires).
- **Pre-registered H0.** The loop ties Stage 0 overall while strictly not regressing any
  frozen branch — itself a publishable result ("bounded self-correction is safe to add").
- **Promotion criteria.** Per-**initial-route**-branch paired McNemar vs Stage 0: promote
  branch *b* iff its discordant pairs favor the loop (CI lower bound ≥ 0 on *b*, after
  Holm/BH across the 6-branch sweep) **and** no frozen branch regresses, **and** the
  initial→final migration matrix shows the gain isn't an artifact of re-routing. Bounded
  depth preserves the Phase-0 latency win.
- **Risk.** Intrinsic self-correction often fails without a real signal
  (`03_related_work.md`); our mitigation is the *external verifier* trigger (not LLM
  self-doubt) plus a hard depth bound.

### Stage 1 — design constraints from the literature (load-bearing)
The self-correction literature (`03_related_work.md` §B) places our gated/bounded/per-branch
design in the *only* regime where self-correction reliably helps (decomposable +
tool-verifiable; CRITIC 2305.11738, Kamoi 2406.01297) — but it adds four non-negotiable
constraints:
1. **Correct in the evidence/tool layer, not the prompt.** Re-entry re-runs deterministic
   tools at adjusted scope (it changes the *evidence*); it must **never** re-prompt the
   same LLM on the same evidence. This avoids the 64.5% self-correction "blind spot"
   (2507.02778), SCoRe's behavior-collapse / over-editing (2409.12917), and sycophantic
   correct→wrong flips (~14.7%, 2502.08177). It keeps correction in the tool/evidence
   layer (ADR-001 tool grounding), never the prompt.
2. **The verifier flag is load-bearing — measure it.** Report the flag's own
   precision/recall; a flag firing on already-correct rows can only make them wrong. The
   McNemar gate tests verifier+loop *jointly*.
3. **Multiple-comparison correction.** The per-branch McNemar sweep is 6 tests → apply
   Holm/BH so we don't "discover" a spurious helpful branch.
4. **Self-consistency control arm.** Self-Consistency (2203.11171) lifts accuracy with *no
   loop*; Stage 1 must beat "sample-more-on-flagged-rows," or that is the better lever.

> Note (positioning, from `03_related_work.md` §A): **TS-Agent already runs a ReAct loop
> with a self-refinement critic and answer-verification.** Our loop is therefore *not* a
> capability novelty — sell it as **rigor + efficiency + per-branch selective promotion**,
> the things a stochastic agent trajectory cannot deliver.

### Stage 2 — Iterative tool refinement
- **Mechanism.** When a branch's evidence is weak, the loop may request a *second
  tool/branch* (the `multi_branch` orchestrator turned into a bounded loop, depth ≤2–3),
  accumulating evidence rather than re-asking the LLM.
- **Switch.** `max_loop_depth=2–3`; candidate expansion via the existing quality_gate.
- **Promotion.** Same invariant; expect gains concentrated on ambiguous / multi-aspect rows.
- **Risk.** Cost growth and error compounding; gated by quality_gate score deltas.

### Stage 3 — LLM-proposed actions (LAST)
- **Mechanism.** Only here does the LLM *propose* the next tool. Hard preconditions before
  any implementation (or it drifts into the black-box failure mode): (1) a **typed action
  schema** the proposal must validate against; (2) a **finite, closed tool registry** — no
  open-ended actions; (3) a **hard per-row call budget**; (4) a **parser-failure contract**
  — an unparseable/invalid proposal falls back deterministically, never silently retried.
  Still verifier-gated and fully logged. Closest to TS-Agent's free ReAct; ships only after
  Stages 1–2 prove agency pays.
- **Risk.** Highest black-box / overfit risk; admitted only behind the four preconditions
  above + the full logging + paired-McNemar gate, and only on branches that earned it.

## The end state
A learned route + escalate + **loop** policy, fit on labelled failure/discordance data,
**frozen**, that ties-or-beats the hand rules at lower cost and transfers across
TSExam → MMTS → TSRBench → TSAQA and across ≥2 backbones. Each promotion is one
pre-registered, paired, logged `log/` entry.

## What Codex is building now (coordination)
Per the 2026-06-21 trio split, Codex is finishing **Stage 0 hardening** (the parasol
Gate-A/C bug fixes — numeric sign, CSV `excluded_reason`, phantom channel) and
**scaffolding Stage 1** on `numeric-head-visual-trust`. This journal defines Stage 1's
**promotion criteria** so that scaffold has a gate to pass. The two efforts land
independently and merge.
