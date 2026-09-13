# 006 — Track A: architecture-only de-confound at fixed gpt-4o-mini (+13.5pp)

---
id: 006
date: 2026-06-22
stage: 0
claim: C1
status: complete
evidence_level: fixed-backbone-ablation
claim_scope: overall
overall_significance: "+13.5pp paired ΔOA at fixed gpt-4o-mini, 95% CI [+9.5,+17.6], exact McNemar p<0.0001 (b/c=170/69)."
prereg: agentic-setup/PREREGISTRATION_c1_gpt4omini.md
commit: agentic-tsmart Track-A gate (see git log)
artifact_present: yes
required_caveats: "Tools arm is vision_OFF (the anomaly forced-vision lever is suppressed → AD flat). Our gpt-4o-mini tools-arm OA 0.548 is still −5.4pp BELOW TS-Agent's published gpt-4o-mini 0.602 on OUR harness ⇒ the original cross-backbone +8.2pp was substantially BACKBONE capability, not architecture. Replicated once at one backbone — existence proof, not invariance; harness/prompt/scoring differ from TS-Agent's own."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_c1_gpt4omini.md)
- **H1:** at a FIXED backbone (gpt-4o-mini = TS-Agent's own), T-SMART's deterministic architecture lifts OA over the raw model (the architecture-only contribution, de-confounding the backbone).
- Primary metric: paired ΔOA (tools − llm_only) with bootstrap CI + exact McNemar, stratified by the tools-arm initial route. Secondary: absolute tools OA vs TS-Agent's published gpt-4o-mini figure (existence-proof caveat).

## Setup
- Backbone: **gpt-4o-mini** for ALL LLM calls (router+answer), temp 0, via the `factory.create_llm_client(provider="openai")` seam (additive; default provider stays gemini, byte-identical — proven).
- Arms (same 746 TSExam rows, all `expected_schema==mcq`): `llm_only` (no tools — raw model answers the MCQ in one letter-neutral call) vs `vision_off` (deterministic T-SMART tools + structured reasoning, **vision suppressed** to isolate the deterministic-tool contribution and avoid cross-backbone vision risk).
- Baseline fairness verified: llm_only = **746/746 valid ABCD preds, 0 fallback** (not crippled); both arms share gpt-4o-mini's letter bias (cancels in the pairing).

## Result
```
OVERALL    746  llm_only 0.413   tools 0.548   Δ +0.135  [+0.095,+0.176]  b/c 170/69  McNemar p<0.0001 *
by category:  PR +0.169*  CA +0.264*  NU +0.167*  SA +0.058 (ns)  AD +0.000 (ns)
mechanism view (Holm): trend +0.199*  causality +0.241*  noise +0.130*  periodicity +0.142*  similarity +0.114*  anomaly +0.000 (ns)
```
**5 of 6 initial-route branches significantly improve after Holm.** Only **anomaly is flat** — expected: vision_off suppresses the forced-anomaly-vision lever (nu_ad_fix), so the architecture's anomaly value is absent here. The lift is therefore **broad** (trend/periodicity/noise/similarity/causality), not carried by one branch.

Discordance (the honest cost): 170 rows corrected by the architecture, 69 broken — net +101/746.

### Absolute placement vs TS-Agent (secondary)
| | our tools (gpt-4o-mini, vision_off) | TS-Agent (gpt-4o-mini, published) |
|---|---|---|
| OA (macro) | 0.548 | 0.602 (run_eval.py:REF, TimeSeriesExam Table I) |
| AD | 0.361 | 0.570 |

## Verdict
**C1 supported at the architecture level, with an honest boundary.** The deterministic architecture contributes a large, broadly-significant **+13.5pp at fixed gpt-4o-mini** (vs +3.2pp at fixed gemini-3.1-flash-lite reported earlier — *the architecture helps a weaker backbone more*). **But** on the same backbone and our harness, TS-Agent's published number (0.602) still exceeds our vision_off tools-arm (0.548) by 5.4pp — so the original cross-backbone "+8.2pp vs TS-Agent" was **substantially backbone capability**, not pure architecture. This is the de-confounded, defensible reading: the architecture is real and backbone-robust; the TS-Agent *headline* was confounded.

## What changed next
- **Headline for `AGENTIC_TSMART_RESULTS.md`:** architecture-only = +13.5pp @ gpt-4o-mini (de-confounded), with the TS-Agent boundary stated.
- **High-value follow-up (flagged, not run tonight):** the −5.4pp gap is dominated by AD (−20.9pp) where vision is suppressed. A FULL-pipeline gpt-4o-mini arm (with the anomaly forced-vision lever) would test whether the full architecture closes the gap to TS-Agent. `OpenAIClient` handles image artifacts (debug-confirmed); cost ≈ $0.5. Run after the core stages if time permits.

## Artifacts
- Runs: `research/outputs/{c1_gpt4o_llmonly, c1_gpt4o_tools}` (746 each, 0 errors, vision 0). Cost ≈ $0.27 OpenAI.
- Pre-reg: `agentic-setup/PREREGISTRATION_c1_gpt4omini.md`. Mechanism-view tool: `research/paired_diff_route.py` (reuses paired_diff stats; stratifies by the tools-arm route since llm_only has none).
- Tests: +7 in `test_wave1_contracts.py` (factory→openai, provider reject, temp-0 determinism, llm_only sanity). Full suite **92 passed**. Default-provider byte-identity proven.
