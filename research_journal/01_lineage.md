# 01 — Lineage & Architectural Decision Records

*Why T-SMART is the way it is. The branch history is the cautionary tale; the ADRs are
the decisions we commit to — and the ones we deliberately refuse.*

## Branch history (the cautionary tale)
From the PI post-mortem (memory `project-audit-tsmart`). The recurring failure pattern:
**real ideas abandoned for hardcoded heuristics under leaderboard pressure.**
- `setup-environment-initial-branch` — the real cognitive architecture (Pydantic-AI
  supervisor router with up to 3 candidate branches, verifier critic on a thread pool,
  VLM on dual-CWT spectrograms). **Never merged** — ~14 s/sample, rate-limit pressure,
  Colab/dependency complexity.
- `tsmart-dual-process-gate` — gated the VLM on math uncertainty; **14/15 on a smoke
  test, collapsed to ~60% at scale.** Classic brittle overfit.
- `t-smart-experts` — token/rate-limit control; reverted (hardcoded limits broke config).
- `main` — MMTS integration; the Match-subset multi-series path went 0%→93% by
  **bypassing the agent** (a hardcoded DTW loop) — useful as a negative control, sobering
  as a lesson.

**Three root failures the architecture must not repeat:** (1) *spectrogram deafness* —
broken math numbers override good visual descriptions because LLMs over-trust quantities;
(2) *rigid single-branch routing* — one misroute blinds the whole pipeline with no
recovery; (3) *prompt engineering as gradient descent* — editing guidelines to nudge the
answer letter, which collapses under distribution shift.

## Architectural Decision Records
Format: Context · Decision · Consequence · Status.

### ADR-001 — Tools compute facts; the LLM only interprets (tool grounding)
- **Context.** LLMs hallucinate quantities and over-trust numbers (root failure #1).
- **Decision.** Deterministic econometric tools own every numeric value
  (FFT / ADF·KPSS / Granger / change-point / autocorr / std / percentile …). The LLM
  receives a structured evidence bundle and only *interprets* it to a choice. (The
  parallel rule for the visual modality — never feed raw pixels/tensors to the reasoner,
  extract a structured topology first — is the Structured-Vision Invariant, ADR-002.)
- **Consequence.** Grounding + auditability + backbone-portability (a weaker LLM still
  gets correct facts). Schema compliance buys *scoreability*, not accuracy — the tools
  buy accuracy.
- **Status.** **Load-bearing, permanent.** Any "let the LLM compute the number" proposal
  violates this.

### ADR-002 — Vision is on-demand and structured, never raw (the Structured-Vision Invariant, SVI)
- **Context.** Vision is expensive and double-edged (it can help or induce sycophancy).
- **Decision.** A gate decides *whether* to look; if it looks, the image becomes a
  structured JSON topology before the reasoner sees it — **no raw pixels/Base64/tensors
  enter the reasoning context.** This is the **Structured-Vision Invariant (SVI)**.
- **Consequence.** Cost control + a clean place to study modality trust. Motivated the
  visual-trust study and the learned gate.
- **Naming note (grounding, added `log/001`).** This repo historically called the SVI "the
  DDI rule." That was a misappropriation: **DDI = Discrete Disentangled Interaction**, an
  unrelated *learned vector-quantization fusion* module (MADI, arXiv 2601.21436; see
  `03_related_work.md` §A). Its stance is the *opposite* of ours — it *fuses* numeric+pixel
  modalities in a discrete latent space, whereas SVI *refuses* pixel fusion and extracts
  symbolic structure first. "DDI" is now reserved for citing that paper; our invariant is
  the SVI.
- **Status.** Permanent; the *gate* is being upgraded (ADR-009).

### ADR-003 — Ship a single-shot decision tree *first*, deliberately
- **Context.** The full agent (Pydantic-AI supervisor) was abandoned for latency/complexity.
- **Decision.** Stabilize a deterministic, single-shot tree with paired-ablation
  evaluation **before** adding any loop.
- **Consequence.** Gave us temp-0 determinism (~0% routing drift), fast paired runs, and
  a trustworthy control arm. This is *why* we can now add ReAct safely — we have a frozen
  baseline to measure against.
- **Status.** Stage 0; frozen as the control arm forever (see `02_react_stages.md`).

### ADR-004 — Configs are architectural switches; the unit of evidence is the paired McNemar per branch
- **Context.** Eyeballing rows and prompt-tweaking repeatedly overfit (root failure #3).
- **Decision.** A "config" is a named dict of pipeline kwargs. Every change is evaluated
  by running baseline vs treatment over the *same rows*, joined per-row, reported as a
  bootstrap 95% CI + exact McNemar, **stratified by the routed branch**. Pre-register the
  verdict first. *(Stratify by the **initial route**, fixed by temp-0 baseline routing, not
  the final `branch_used` — once Stage-1 loops can re-route, the final branch is
  post-treatment/endogenous; see `02_react_stages.md`.)*
- **Consequence.** Effects are *located* to a mechanism and are distribution-shift-robust
  by construction. This methodology *is* the contribution — it is why we don't rewrite.
- **Status.** Permanent. Extends unchanged to ReAct (the loop is just new switches).

### ADR-005 — Backbone is `gemini-3.1-flash-lite`, temp 0 (intentional)
- **Context.** Need a fast, cheap, deterministic gold standard for multi-thousand-row
  paired runs.
- **Decision.** Fixed backbone for all reported results; concurrency tuned for it.
- **Consequence.** Reproducibility + feasibility (746-row eval: 1.9 h → ~8 min). Creates
  the backbone-confound caveat → motivates C1 (a multi-backbone test), not a backbone
  change.
- **Status.** Frozen for the main line; a *second* backbone is added only as a paired C1 arm.

### ADR-006 — Refuse the Pydantic-AI supervisor rewrite
- **Context.** It's the "real agent," but it was abandoned for concrete reasons (latency,
  complexity, rate limits).
- **Decision.** Do not resurrect it wholesale. Re-introduce agency *incrementally* and
  *measured* (ADR-007), not as a framework swap.
- **Status.** Standing refusal; revisit only if a staged loop proves agency pays and the
  latency/complexity budget is solved.

### ADR-007 — Refuse a from-scratch ReAct rewrite; implement ReAct as additive config switches
- **Context.** TS-Agent-style free ReAct (the LLM drives the loop) is the obvious move
  and the wrong one: it discards the paired-ablation machinery (ADR-004) that is our
  actual contribution, and re-opens the black-box and overfit risks.
- **Decision.** ReAct in T-SMART = **new *temporal* config switches** (`loop_branches`,
  `max_loop_depth`, `loop_on_flags`, `loop_mode` — now implemented as an **inert-by-default
  scaffold** in Codex's Stage-1 follow-up work landing on `numeric-head-visual-trust`; the
  defaults `max_loop_depth=0`/`loop_branches=None` keep `run_pipeline`/`run_dataset`
  byte-identical to Stage 0 — verified — so the switches *exist* but change nothing until a
  branch is promoted) layered on the frozen tree, evaluated by the same per-branch paired
  McNemar. Self-correction = **verifier-gated bounded re-entry**, not free LLM looping.
  Promote a loop only on branches where the discordant-pair math is favorable — exactly how
  vision got forced on anomaly and suppressed on noise.
- **Consequence.** We add agency without losing the soul, the evaluation, or the
  non-black-box property. Full design in `02_react_stages.md`.
- **Status.** **The active decision.** This is the arc. The scaffold is built and inert; the
  two promotion-gate prerequisites flagged in the 2026-06-21 code review are now **wired
  (`log/002`)**: the result emits a stable `initial_branch_used` for *every* row, and the diff
  tooling stratifies by that initial route with Holm/BH on the 6-branch sweep + an initial→final
  migration matrix (see `02_react_stages.md` and ADR-004). The loop stays inert by default;
  promoting any branch still needs the measured paired experiment (plus the Self-Consistency
  control arm, `02_react_stages.md` §constraints).

### ADR-008 — The numeric/schema head is isolated by a *structural* gate
- **Context.** ~56% of MMTS Base is free-response numeric; an MCQ classifier cannot score
  it.
- **Decision.** A closed-form numeric head answers free-response rows; it is gated by a
  structural `expected_schema` (options present ⇒ MCQ) so MCQ rows stay
  **byte-identical**. Tools own the value; the LLM only formats. Reject-loop bounded to
  ≤1 format-only retry.
- **Consequence.** Unlocks ~400 rows for scoring without touching the MCQ/vision
  experiments. Scored by the benchmark's own Accuracy@10%.
- **Status.** Built, tested (byte-identity proven), validated (361/392 @ 92.3% Acc@10%).
  **Caveat:** the structural `expected_schema` classifier is not perfect — ~8 rows land
  off-diagonal — so "MCQ byte-identical" holds *modulo* that small schema-classification
  leakage; it is a coverage win, not a flawless gate.

### ADR-009 — Learn the "should-I-look" gate; don't hand-tune it
- **Context.** The vision gate is a hand-tuned additive-score heuristic — exactly the
  brittle-rule failure mode the audit warned about.
- **Decision.** Fit a calibrated per-branch suppression policy on labelled discordance
  data; select via `vision_gate="learned"`; freeze the artifact.
- **Consequence.** On held-out TSExam it ties the hand gate at **half** the look-rate and
  **rediscovered the NU half of `nu_ad_fix` — "suppress noise vision" — from data** (its
  only significant lever, +9.7pp on noise). Note it *diverges* from the AD half: being
  suppression-only and firing on `{trend, periodicity}`, it suppresses anomaly vision
  (fit weight −0.188), whereas `nu_ad_fix` *forces* it. So it rediscovered NU, not full
  `nu_ad_fix`. This is the template for the ReAct loop gate (learn *when to act again*;
  don't hand-tune it).
- **Status.** Built, frozen, validated. The pattern generalizes to Stage 1+ (ADR-007).

## Related work
See `03_related_work.md` for the literature positioning (synthesized 2026-06-21 from a
three-reviewer pass: agentic TS reasoning, self-correction evidence, TSFMs + backbone
methodology).
