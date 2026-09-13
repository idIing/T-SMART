# 00 — Thesis & North Star

*Established 2026-06-21. This is the claim the whole program is organized to prove or
falsify. If a piece of work doesn't move one of these claims, question why we're doing
it.*

## The system in one sentence
T-SMART answers questions about time series by letting **deterministic analysis tools
compute the facts** and an **LLM only interpret** them — escalating to a structured
visual sensor on demand — so that the reasoning is grounded, auditable, and portable
across LLM backbones.

## North star
A **generalized, plug-and-play reasoning framework** that:
1. **Beats a strong agentic baseline (TS-Agent) regardless of the LLM backbone** — the
   improvement comes from the *architecture*, not from a stronger model.
2. **Is robust across benchmark varieties** — validated on TimeSeriesExam (home),
   transferred to MMTS-Bench (OOD), stressed on TSRBench (broad task mix) and TSAQA
   (anomaly at scale).
3. **Leaves room for local models and Time-Series Foundation Models (TSFMs)** — the
   backbone and the forecasting/representation head are swappable behind stable
   interfaces.
4. **Is not a black box** — every result is recorded in the `research_journal/log/` (the
   auditability *contract*), and the pipeline can emit an *optional* per-run semantic trace
   (`return_trace`, off by default — a compact fingerprint, not yet a full per-iteration
   decision dump). The journal is the binding audit trail; closing the gap to a complete
   runtime trace is itself tracked work, not a solved property.

We keep the **soul of T-SMART** (deterministic tools + structured evidence + on-demand
vision + paired, mechanism-located evaluation) while **adding bounded self-correction /
ReAct** as a *staged, measured* upgrade — never a rewrite (see `02_react_stages.md`).

## The claims, in falsifiable form
| # | Claim | Falsified if… | Status |
|---|---|---|---|
| C1 | Architecture, not backbone, drives the win | At fixed backbone, paired Δ vs an un-routed / LLM-only baseline is ≤0 and n.s. | **Supported in the free-response regime; regime/mechanism-specific (logs 006, 010, 011).** Track A (006): +13.5pp at fixed gpt-4o-mini vs the raw model. **The architecture-vs-LLM-only contrast is now met decisively on free-response (011):** the tool numeric head beats an `llm_numeric` counterfactual (the *same* gemini computing from the raw series) by **+30.6pp** (closed-form +31.8pp, **104/0**, p<0.0001) at fixed backbone — load-bearing exactly where the model can't compute. **Scope:** that contrast is *computation* — the numeric head BYPASSES the LLM (0 answer calls), so 011 shows tool ownership of computation, **not** that the reasoning layer augments; whether the LLM is load-bearing for *parsing* (composition/paraphrase/typos the keyword map can't handle) is the open three-arm study (log/012). **Boundary (010):** the *vision* lever does **not** transfer to a weak backbone (gpt-4o-mini AD +2.5pp n.s.), so C1 is **regime- and mechanism-specific** (mechanism value = f(backbone × task)), not a universal headline. On MCQ at a strong backbone the architecture is ~inert (logs 004–008) |
| C2 | The conditional "should-I-look" decision transfers OOD | Learned gate regresses some branch OOD, or fails to match the hand rules | **Pre-registered H0 supported (no OA-accuracy claim)** — H1 ("improves") was *not* met; the learned gate *ties* the hand gate (efficiency win: ~half the look-rate) on held-out TSExam, no branch regressing. It rediscovered the **NU (suppress-noise-vision) half** of `nu_ad_fix` from data — and *diverges* from the AD half (it suppresses anomaly vision; `nu_ad_fix` forces it) |
| C3 | A mechanism validated at home transfers to OOD | Anomaly/noise levers flip sign or vanish on MMTS | **Supported (mechanism-local; overall n.s.)** — the **anomaly stratum** replicates (+16.5pp, p=0.024, ~5% of rows); **overall MMTS +1.0pp n.s.**; noise neutral both places. The claim is mechanism transfer, not a headline OA lift |
| C4 | Bounded, gated self-correction improves *where the verifier fires* without regressing elsewhere | Stage-1 loop yields net-negative discordant pairs on its target branches, or regresses any frozen branch | **Untested** — the next frontier (`02_react_stages.md`) |
| C5 | The framework is backbone- and head-swappable | A second backbone or a TSFM head cannot be inserted behind the existing interfaces without touching the core | **Untested** — `ForecastRanker` / client-factory seams exist; not yet exercised |

## Honesty caveats that travel with every headline
Non-negotiable; mirrored from `mmts_bench/GENERALIZATION_REPORT.md` §5. They accompany
any number we report.
- **"+8.2pp vs TS-Agent" is backbone-confounded** (us = `gemini-3.1-flash-lite`;
  TS-Agent = `gpt-4o-mini`). The cleanest same-backbone number is the **+3.2pp paired
  config ablation** (baseline vs `nu_ad_fix`) — that isolates the *vision-routing
  mechanism* at fixed backbone; it is **not** an architecture-vs-LLM-only proof and
  **not** backbone-invariance. Closing both gaps is the point of C1.
- **The strong anomaly lever (+16.5pp) reaches ~5% of rows.** Never quote it without the
  stratum size. TSAQA exists to test it where it fires on ~100% of rows.
- **The noise lever is neutral at scale.** Its value is *removing a failure mode for
  free*, not adding accuracy.
- **The headline OA gain is small and often n.s.** The defensible framing is
  **mechanism transfer + non-regression**, not "T-SMART is +X% better."
- **Self-correction is not assumed to raise OA** (see `03_related_work.md`): the
  literature shows intrinsic self-correction often *fails* without an external signal.
  We expect Stage-1's value to be *failure-mode rescue on gated branches*, measured —
  not a blanket lift.

## What "done" looks like for the next arc
A learned route + escalate + **loop** policy, fit on labelled discordance/failure data,
**frozen**, that (a) ties or beats the hand rules at lower cost, and (b) transfers across
TSExam → MMTS → TSRBench → TSAQA **and** across ≥2 backbones — each step a
pre-registered, paired, logged entry in `log/`.
