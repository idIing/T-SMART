# 001 — DDI regrounding + audit remediation

---
id: 001
date: 2026-06-21
stage: 0
claim: none   # audit/regrounding note — moves no experimental claim; it corrects the record
status: note   # not an experiment; supersedes stale statuses without rewriting 000
evidence_level: n/a
claim_scope: n/a
overall_significance: n/a
prereg: n/a
commit: research-journal branch (see git log)
artifact_present: n/a
required_caveats: "this entry REMOVES overclaims; it adds no new headline number"
---

## Why this entry exists
PI-directed regrounding ("reground in true literature; fight it like a PhD but concede
genuine points") triggered by a read-only audit of the journal. Two things had to be
verified against **primary sources**, not asserted: (a) what "DDI" actually means, and
(b) whether the audit's own DDI correction was itself grounded. Both checks landed.

## Finding 1 — "DDI" was misappropriated repo-wide (primary source: arXiv 2601.21436)
**DDI = Discrete Disentangled Interaction**, a module of **MADI** ("From Consistency to
Complementarity…", Ni, Zhang, Wang, Shao, Liu; arXiv **2601.21436**, Jan 2026). Verbatim:

> "We propose a *Discrete Disentangled Interaction (DDI)* module that performs modality
> disentanglement in a discrete latent space to enforce compact, consistent
> representations, and then integrates the isolated cross-modal unique signals."

Mechanically it is **hierarchical residual vector quantization** separating modality-common
from modality-specific features + cross-attention over each modality's *unique* signal. It
is a **learned neural fusion module**, **not** a data-flow rule, and it does **not** forbid
raw images/tensors in a loop.

Our repo used "DDI" as a label for the *opposite* idea — our rule that the vision tool must
**never** put raw pixels/Base64/tensors into the reasoning context (extract a structured
JSON topology first). We borrowed a published acronym and attached it to a rule that
**inverts** the source concept (MADI *fuses* modalities; we *refuse* to). The PI's call was
exact: "we never even use an official definition just its abbreviation."

## Finding 2 — the audit's central correction was *also* ungrounded
The audit's headline S0 declared a "canonical DDI definition: the vision/artifact data-flow
invariant only," and prescribed "reserve DDI for the vision flow." That **canonizes our
misnomer**. The auditor caught every *downstream* symptom of the error (DDI sprawled onto
self-correction and the TSFM interface) while sharing the *upstream* error (it never read
2601.21436). Lesson worth keeping: a tidy, confident audit was right on ~12 items and wrong
on the one it was most sure of — *because* it argued from internal consistency, not the
source. This is the entire case for grounding.

## Resolution (PI decision)
Our invariant is renamed the **Structured-Vision Invariant (SVI)** (ADR-002). "DDI" now
appears **only** when citing MADI (2601.21436), which is added to `03_related_work.md` §A as
the **learned-fusion foil** to SVI — a genuinely useful comparand that sharpens our novelty
(calibrated selective *symbolic* perception) and the live Plots-Unlock tension.

## Conceded corrections applied this pass (the audit was right on these)
- **DDI → SVI** everywhere it labeled our invariant: `CLAUDE.md`, `AGENTS.md`,
  `01_lineage.md` (ADR-001 retitled "tool grounding"; ADR-002 now names SVI + the grounding
  note), `02_react_stages.md` :61, `03_related_work.md` :104 & :146.
- **"rediscovered nu_ad_fix" → "rediscovered the NU half; diverges on AD."** The learned
  gate is suppression-only, fires on `{trend, periodicity}`, and **suppresses** anomaly
  vision (fit weight −0.188); `nu_ad_fix` **forces** it. (`00_thesis` C2, `01_lineage`
  ADR-009.)
- **Stage 0 control = the additive baseline**, not the learned gate (a validated *arm*).
  (`02_react_stages.md` Stage 0; `CLAUDE.md` trajectory section.)
- **C1/C2/C3 evidence-labels:** C1 reopened (the +3.2pp is a fixed-backbone *config*
  ablation, not architecture-vs-LLM-only nor backbone-invariance); C2 = pre-registered
  **H0/efficiency**, no OA claim; C3 = **mechanism-local; overall n.s.** (`00_thesis`.)
- **Loop switches marked planned, not built** (`01_lineage` ADR-007, `02_react_stages`).
- **Trace-logging claim demoted** from "every decision is logged" to "the `log/` is the
  contract; runtime `return_trace` is optional/off/fingerprint-only" (`00_thesis`,
  `02_react_stages`).
- **Stage 3 guardrails added:** typed action schema · finite registry · call budget ·
  parser-failure contract (`02_react_stages`).
- **Numeric-head schema caveat** (~8 off-diagonal rows) on ADR-008 (`01_lineage`).
- **`branch_used` → initial route + migration matrix** (see Method note) — invariant in
  `02_react_stages`, ADR-004 in `01_lineage`, and `log/TEMPLATE.md`.
- **TEMPLATE gains** `evidence_level`, `claim_scope`, `overall_significance`,
  `artifact_present`, `required_caveats`, a migration-matrix table, and a verifier-flag
  precision/recall line; `status: complete` now **requires a verdict** (notes use `note`).

## Method note this entry establishes (the most load-bearing fix)
Stratify the paired McNemar by the **initial route** (pre-treatment, fixed by temp-0
baseline routing), **not** the final `branch_used`. Once Stage-1 loops can re-route, the
final branch is post-treatment/endogenous — conditioning on it is a collider that biases
every per-branch Δ. Always report an **initial→final migration matrix** so a "gain" can be
distinguished from a re-routing artifact. This protects C4 *before* the Stage-1 scaffold
hardens the wrong stratification.

## Supersedes (do NOT edit those entries — append-only discipline)
- `log/000` frontmatter `claim: C4 / status: complete` → read it as a founding **note**;
  **C4 remains untested.** (000's body stays byte-identical; this is the correction of
  record.)
- 000's "rediscovered the `nu_ad_fix` wisdom" line → scoped to the **NU half** per above.

## What changed next (queued — NOT done this pass; needs PI greenlight)
- Two **main-tree** linked reports still carry stale handoff wording (learned gate /
  numeric head described as future work) and the old DDI name:
  `mmts_bench/GENERALIZATION_REPORT.md` and `T_SMART_paper_vs_main_audit.md`. They are
  Codex's active surface — defer to a coordinated pass rather than touch them mid-sprint.
- The Plots-Unlock vs MADI tension now has a named opponent (MADI's learned fusion). The
  pre-registered **structured-topology vs raw-pixel** ablation (`03_related_work.md` §A) is
  still the open item to settle ADR-002/SVI empirically.

## Links
`00_thesis.md` · `01_lineage.md` (ADR-001/002, ADR-004, ADR-009) · `02_react_stages.md` ·
`03_related_work.md` §A/§C · memory: `ddi-true-meaning-svi-rename`,
`tsmart-react-arc-and-journal`, `tsmart_learned_gate_built`.
