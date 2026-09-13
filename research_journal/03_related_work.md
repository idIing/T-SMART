# 03 — Related Work & Literature Synthesis

*Synthesized 2026-06-21 from a three-reviewer pass (agentic TS reasoning · self-correction
evidence · TSFMs + backbone methodology). All arXiv ids verified by the reviewers. This is
the durable home for the "earlier literature synthesis" that previously lived only in
conversation.*

---

## A. Where T-SMART sits (positioning)

T-SMART occupies the **"LLM + tools"** cell of the LLM-for-time-series taxonomy
(survey: Zhang et al. 2402.01801; position: Jin et al. 2402.02713), with a side-channel
into **"vision-as-bridge."** Its thesis — *deterministic tools compute structured
evidence, the LLM only interprets* — is the same separation-of-concerns as **PAL**
(2211.10435) / **Program-of-Thoughts** (2211.12588) and as our closest comparand,
**TS-Agent** (2510.07432). So T-SMART is **not a new paradigm in kind**; it is a
*single-shot, deterministic, rigorously-ablatable specialization* of the tool-grounded-
reasoner family.

**The TS-Agent comparison is sharper than we framed it.** TS-Agent (NeurIPS 2025 wksp,
backbone **gpt-4o-mini**) already runs a **ReAct loop** (think→tool→observe) over six
operator families (Summarize/Extract/Query/Detect/Predict/Relate), **plus a
self-refinement critic and a final answer-verification gate.** In other words, the loop
and self-correction we are building toward (`02_react_stages.md`) is *not novel relative
to TS-Agent* — TS-Agent already loops. Our defensible differentiators are: (1) **temp-0
reproducibility** → clean paired-McNemar ablations (stochastic agent trajectories can't do
this); (2) **per-branch promotion** of mechanisms; (3) keeping corrections in the
**tool/evidence layer**, not free LLM looping. Sell the loop as *rigor + efficiency +
selective promotion*, **not** as a capability TS-Agent lacks.

### Benchmark landscape
| Benchmark | Size | Taxonomy | MCQ vs numeric | Anomaly a top-level category? | Modality |
|---|---|---|---|---|---|
| **TimeSeriesExam** (2410.14752) | ~700 Q, IRT-refined | pattern, noise, similarity, anomaly, causality | **MCQ only** | **Yes** (dedicated) | numeric (synthetic) |
| **MMTS-Bench** (2602.08588) | 2,424 Q (Base/InWild/Match/Align) | structural, feature, temporal, matching, cross-modal | **Mixed** — Base ~56% numeric free-resp. (Acc@10%) | No (under feature analysis) | numeric + image |
| **TSRBench** (2601.18744, ICML 2026) | 4,125 problems, 14 domains, 15 tasks | Perception / Reasoning / Prediction / Decision | mixed incl. numerical reasoning | No (within Perception/Reasoning) | **4 modalities**: text, image, interleaved, embeddings |
| **TSAQA** (2601.23204v2, GEM@ACL 2026 wksp) | **210k**, 13 domains | anomaly, classification, characterization, comparison, transform, temporal-rel | **TF + MC + "puzzling"** | **Yes** (explicit, at scale) | numeric |

> **Citation corrections for the repo:** TSRBench is **arXiv 2601.18744** (Yu, Guo, …,
> Zhou; ICML 2026) — record this id wherever TSRBench is cited. TSAQA is **2601.23204v2**,
> a **workshop** paper (GEM@ACL 2026), not main-track. TS-Agent backbone is **gpt-4o-mini**
> (confirms our backbone-confound caveat).
> **Honesty note:** anomaly is a *first-class* category only in TSExam and TSAQA; in MMTS
> and TSRBench it's folded into broader dimensions — so cross-benchmark "anomaly lever"
> claims are not strictly apples-to-apples.

### What's genuinely distinctive vs. incremental
- **Distinctive:** (a) a *reproducible, ablatable* tool-augmented pipeline enabling paired
  cross-benchmark causal claims (rare in this literature); (b) the **conditional,
  learned, structured-vision gate** — *no* anchor paper (TS-Agent, ChatTS, TSRBench, the
  surveys) addresses **calibrated, learned selective perception** ("should I look?") for
  time series; they always-look, never-look, or loop. **This is the strongest novelty and
  the frontier to push** (ADR-009 → Stage 1+).
- **Incremental:** tools-then-interpret (TS-Agent/PAL did it), the numeric head (PAL
  principle), structured-vision-instead-of-pixels (a defensible engineering choice, not a
  new concept).

### ⚠ Open tension to resolve (recorded, not yet answered)
**"Plots Unlock Time-Series Understanding"** (2410.02637, Google) finds frontier models
understand series **far better from plots than from numeric text (up to +120–150%, ~90%
cheaper)** — *prima facie evidence against discarding the pixels.* TSRBench (2601.18744)
finds the opposite-flavored result that **current multimodal models fail to fuse
text+vision**, which is evidence *for* our structured extraction dodging the fusion
failure. These cut against each other. **T-SMART's "render plot → JSON topology → reasoner"
choice (ADR-002) is not validated against pixel-passing.** → **Action:** a pre-registered
ablation (structured-topology vs. raw-pixel vision) belongs in the `log/` queue; we cannot
claim structured-vision ≥ pixel-vision until we run it. Other paradigms we deliberately
*don't* use: raw digit serialization (LLMTime 2310.07820), native TS-encoder training
(ChatTS 2412.03104), text+series "multimodal" (MTBench 2503.16858), and **learned
numeric+plot *fusion* via vector-quantized disentanglement — MADI / the *Discrete
Disentangled Interaction (DDI)* module (2601.21436)**, which is the direct foil to our
stance: MADI *fuses* both modalities in a discrete latent space, while T-SMART's SVI
(ADR-002) *refuses* pixel fusion and extracts symbolic topology first. (This is also the
source of the "DDI" acronym this repo previously misapplied to the SVI — corrected in
`log/001`.) Branch/subtype design is corroborated by the feature-understanding taxonomy
(2404.16563).

---

## B. Self-correction: when it helps, when it hurts (validates + constrains Stage 1)

**Bottom line: the literature endorses our design — gated, bounded, per-branch — as the
*only* regime where self-correction reliably helps, but it imposes four load-bearing
constraints.**

The consistent finding across **Huang et al. 2310.01798** ("LLMs Cannot Self-Correct
Reasoning Yet" — intrinsic, ungated loops *degrade* reasoning) and the **Kamoi et al. 2024
survey (2406.01297, TACL)** is: unaided self-correction fails; it works only when (a)
responses are *decomposable*, (b) a *reliable external tool* exists, or (c)
verify-from-response is easier than generate-from-scratch. **CRITIC (2305.11738)** is the
direct analogue to T-SMART: LLMs can't critique themselves but *can* use a deterministic
tool's signal — exactly our verifier-on-evidence setup. Process-verifier work
(**Lightman 2305.20050**; **ThinkPRM 2504.16828**) shows an external step-verifier beats
end-to-end self-judgment, and that test-time compute is best spent *re-verifying*, not
re-generating.

The failure modes we must avoid: the **self-correction blind spot** (models fix identical
errors framed as *external* input but miss their *own* — 64.5% across 14 models;
2507.02778); **behavior collapse / error injection** (**SCoRe**, DeepMind 2409.12917:
untrained second passes either don't edit or over-edit; base intrinsic delta −11.2% on
MATH); **sycophancy** (correct→wrong flips ~14.7% under disagreement-shaped feedback;
2502.08177, 2509.16533); and **multi-turn noise accumulation** (2603.16244: more rounds
degrade, not converge). Note **Self-Consistency (2203.11171)** raises accuracy with *no
loop at all* — a cheaper lever that any loop must beat.

### → Four constraints now baked into Stage 1 (see `02_react_stages.md`)
1. **Correct in the evidence/tool layer, not the prompt** — re-run deterministic tools at
   adjusted scope; never re-prompt the same LLM on the same evidence (avoids blind-spot +
   SCoRe collapse + sycophancy; it keeps correction in the tool/evidence layer — ADR-001).
2. **Measure the verifier flag itself** (precision/recall) — the gate is now load-bearing;
   McNemar tests verifier+loop *jointly*.
3. **Multiple-comparison correction** (Holm/BH) on the 6-branch McNemar sweep.
4. **A self-consistency control arm** on flagged rows — the loop must beat "just sample
   more."

### The caveat to quote with any Stage-1 result
> The literature does not support expecting a self-correction loop to raise reasoning
> accuracy on its own; unaided self-correction *degrades* reasoning absent a reliable
> external signal (Huang 2310.01798; Kamoi 2406.01297), because models miss their own
> errors (the 64.5% blind spot, 2507.02778) and flip correct answers under
> disagreement-shaped feedback (~14.7%, 2502.08177). Any gain we report is attributed to
> the **external, tool-grounded verifier signal and per-branch selectivity** — not to the
> LLM "thinking again." We cap iterations at 1–2, promote per-branch only on paired
> significance, and claim nothing for branches we did not validate.

(Lineage of the loop idea: **ReAct 2210.03629** — lift comes from *external grounding*;
**Toolformer 2302.04761** — *whether to call* is a learnable utility-gated decision;
**Reflexion 2303.11366** — needs a task-feedback oracle; **Self-Refine 2303.17651** — gains
concentrate on open-ended, not reasoning, tasks.)

---

## C. TSFMs (the forecast-head seam) + backbone-agnostic method

### TSFM comparison (for the `ForecastRanker` seam, C5)
| Model | arXiv/yr | Variate | Probabilistic? | Zero-shot? | License | Fit for a ranking head |
|---|---|---|---|---|---|---|
| **Chronos** (Amazon) | 2403.07815 | univariate | **Yes** (samples paths) | Yes | **Apache-2.0** | **Best fit** — natively emits sample paths to score |
| **Lag-Llama** | 2310.08278 | univariate | **Yes** (exact log-lik) | Yes | **Apache-2.0** | **Yes** — lightweight; exact per-step likelihood |
| **TimesFM** (Google) | 2310.10688 | univariate | point (+quantile in 2.x) | Yes | **Apache-2.0** | Yes — cleanest open point-forecaster |
| **MOMENT** | 2402.03885 | univariate | point (reconstruction) | Yes | **MIT** | Maybe — best as *embedding/representation* head |
| **Moirai** (Salesforce) | 2402.02592 | **multivariate** | **Yes** | Yes | **CC-BY-NC-4.0** ⚠ | Maybe — capable but **non-commercial license** |
| **TimeGPT-1** (Nixtla) | 2310.03589 | uni+exog | **Yes** | Yes | **commercial API** ⚠ | Maybe — paid black box; bad paper artifact |
| **Time-LLM** | 2310.01728 | univariate | point | few/zero-shot | code open | **No** — reprograms a frozen LLM; nests a 2nd LLM |

**Recommendation:** prototype **Chronos** first (Apache-2.0, natively probabilistic, tiny
20M CPU-runnable checkpoints → reproducible), **Lag-Llama** second (exact log-likelihood =
cleaner ranking signal). Keep the head **univariate-first** (5/7 models); reach for Moirai
only for true multivariate ranking and gate it behind its non-commercial license.

**Interface contract** (so any backend slots in — returns a *score vector*, not raw
tensors, consistent with our structured-evidence discipline / SVI):
```
rank(context: float[T],
     candidates: float[K][H],          # K candidate continuations, horizon H
     n_samples: int = 100) -> float[K] # higher = more plausible
```
Backends fill it differently: Chronos → score-by-sample-likelihood; Lag-Llama → exact
log-likelihood; TimesFM/MOMENT → quantile-distance (degraded mode). This reconciles with
the current `ForecastRanker` Protocol (`tsqa/tsqa/eval/hooks.py`) — see the
parasol Gate-C item that the Protocol return type and impl (`HeadResult`) are disjoint.

### Backbone-agnostic recipe (C1) — the highest-value next run
The model-agnostic-framework literature (StepTool 2410.07745; Ladder 2406.15741;
generalizability survey 2509.16330; "Breaking Agent Backbones" 2510.22620) converges on:
**freeze the architecture byte-identical, sweep the backbone, report per-backbone paired
deltas.**
1. **≥3 backbones across confound axes:** our **gemini-3.1-flash-lite** + **gpt-4o-mini**
   (equalizes the TS-Agent comparison directly) + one **open-weight** (Llama-3.1-8B or
   Qwen2-7B-Instruct). Optionally a 4th frontier model to test whether the architecture's
   value *shrinks* as the backbone strengthens.
2. **Freeze everything else** (router prompts, `nu_ad_fix`, tools, temp 0, same rows).
3. **Paired per backbone:** baseline vs tool-augmented over the same rows; bootstrap CI +
   McNemar; stratify by `branch_used`.
4. **The headline becomes a vector:** "+Δ paired on each of K backbones, all CIs > 0" —
   consistent-sign significant deltas across heterogeneous backbones is what makes
   "regardless of backbone" defensible.

> **Caveat (load-bearing):** one extra backbone upgrades the claim from "anecdote" to
> "replicated once" — **not** backbone-invariance. The known risk is an interaction where
> structured tools only rescue *weak* backbones and wash out on frontier models. Defensible
> phrasing: *"the tool-augmentation effect is positive and significant across N backbones
> spanning M families and two capability tiers, with effect size attenuating/holding as
> backbone capability increases."* And: the cross-backbone "+8.2pp vs TS-Agent" stays
> confounded **until gpt-4o-mini is run inside our own architecture** — that single run is
> the highest-value experiment for the north star's goal (1).

---

## Master reference list
**TS-for-LLM systems & benchmarks:** TS-Agent 2510.07432 · TimeSeriesExam 2410.14752 ·
MMTS-Bench 2602.08588 · TSRBench 2601.18744 · TSAQA 2601.23204v2 · LLMTime 2310.07820 ·
Plots-Unlock 2410.02637 · ChatTS 2412.03104 · MTBench 2503.16858 · MADI/DDI 2601.21436 ·
Feature-Understanding 2404.16563 · Position 2402.02713 · Survey 2402.01801 · PAL 2211.10435
· PoT 2211.12588.
**Reasoning loops & self-correction:** ReAct 2210.03629 · Self-Consistency 2203.11171 ·
Toolformer 2302.04761 · Reflexion 2303.11366 · Self-Refine 2303.17651 · Huang 2310.01798 ·
Kamoi 2406.01297 · CRITIC 2305.11738 · SCoRe 2409.12917 · Blind-Spot 2507.02778 · Lightman
2305.20050 · ThinkPRM 2504.16828 · SycEval 2502.08177 · Rebuttal 2509.16533 · More-Rounds
2603.16244.
**TSFMs & backbone method:** TimesFM 2310.10688 · Chronos 2403.07815 · Moirai 2402.02592 ·
Lag-Llama 2310.08278 · MOMENT 2402.03885 · TimeGPT-1 2310.03589 · Time-LLM 2310.01728 ·
StepTool 2410.07745 · Ladder 2406.15741 · Generalizability-survey 2509.16330 ·
Breaking-Agent-Backbones 2510.22620.
