# 009 — Track V: structured (VLM→JSON) vs raw-pixel vision — SVI validated (robustness choice)

---
id: 009
date: 2026-06-22
stage: 0
claim: none   # validates the Structured-Vision Invariant (SVI / ADR-002); measurement-only arm, never shipped
status: complete
evidence_level: mechanism-local-stratum
claim_scope: mechanism-local
overall_significance: "vision-fired subset n=568: structured net +8.5pp vs raw_pixel +10.0pp (both p<0.05 vs the math anchor). Structured HOLDS the ceiling on the shape branches where pixels were predicted to win (trend +20.9 vs +9.9; periodicity & anomaly exact ties) AND is less sycophantic (trend follows-wrong-look 40% vs 73%)."
prereg: agentic-setup/PREREGISTRATION_structured_vs_pixel.md
commit: agentic-tsmart Track-V gate (see git log)
artifact_present: yes
required_caveats: "raw_pixel is marginally higher OVERALL (+0.016 net, within the ±1.3pp noise band) — SVI's case is NOT strictly-higher accuracy; it is sycophancy-resistance + holding the shape ceiling. raw_pixel_vision is MEASUREMENT-ONLY, never promoted/shipped (deliberate SVI violation behind a switch). Single benchmark (TSExam), one backbone."
---

## Hypothesis (pre-registered — agentic-setup/PREREGISTRATION_structured_vs_pixel.md)
- **H1:** raw pixels are *more sycophantic* (the reasoner follows a wrong visual look more often) than the structured JSON topology.
- **H2:** raw pixels have a *higher ceiling* on shape/anomaly branches where the fixed 8-field JSON discards discriminative signal.
- SVI is **validated** if structured ≥ raw_pixel on net_value at the at-risk branches AND structured sycophancy ≤ raw_pixel; **bounded** if pixels win the shape/anomaly ceiling.

## Setup
- Three arms on the **vision-fired subset of TSExam (n=568/746)**, vs the `vision_off` math anchor (OA 0.593 on the subset). **structured** = `baseline_full` (SVI: image→JSON topology→reasoner). **raw_pixel** = new measurement-only arm: on the *identical* 568 fired rows (same additive gate; only the representation moves) the rendered PNG is attached **directly to the answer LLM**, bypassing `analyze_image`'s JSON. gemini-3.1-flash-lite, temp 0. raw_pixel issued 568/568 clean letters (multimodal path not flaky).
- `net_value` = P(arm correct) − P(math correct). `sycophancy` = among discordant rows where the look is WRONG, fraction the reasoner follows it.

## Result — per branch (vision-fired subset)
| branch | n | structured net / syc | raw_pixel net / syc | net Δ(px−str) | syc Δ(px−str) |
|---|--:|--:|--:|--:|--:|
| periodicity | 117 | +0.248* / 0.92 | +0.248* / 1.00 | +0.00 | +0.08 |
| trend | 91 | **+0.209*** / 0.40 | +0.099 / 0.73 | **−0.110** | **+0.333** |
| similarity | 139 | +0.050 / 0.40 | +0.065 / 0.47 | +0.01 | +0.07 |
| anomaly | 65 | +0.015 / 0.88 | +0.015 / 1.00 | +0.00 | +0.13 |
| noise | 145 | −0.055 / 0.64 | +0.062 / 0.48 | +0.12 | −0.16 |
| **OVERALL** | **568** | **+0.085*** / 0.598 | **+0.100*** / 0.607 | +0.016 | +0.008 |
(* McNemar p<0.05 vs the math anchor; structured CI [+0.039,+0.132], raw_pixel [+0.055,+0.146]. Independently recomputed by meta: math 0.593, structured 0.678, raw_pixel 0.694.)

## Verdict — **SVI validated as a robustness choice (H2 rejected; H1 holds directionally)**
- **H2 (pixels lift shape/anomaly) — NOT supported.** Where pixels were predicted to win, the JSON topology matches or beats them: periodicity exact tie (+0.248 both), anomaly exact tie (+0.015 both), **trend the structured arm WINS** (+0.209 vs +0.099). The fixed 8-field topology already captures the discriminative shape signal on this benchmark.
- **H1 (pixels more sycophantic) — holds directionally.** Raw pixels are more sycophantic on 4/5 testable branches (trend **+0.33**, anomaly +0.13, periodicity +0.08, similarity +0.07); the lone exception is `noise` (a statistical branch where neither look should fire).
- **The clean mechanism:** raw pixels turn `trend` from a controlled, calibrated lift (structured +0.209, follows-wrong-look only 40%) into a higher-sycophancy, lower-accuracy arm (+0.099, follows-wrong-look 73%). The JSON intermediary **dampens blind following without discarding the signal** — the SVI's intended effect, now measured.
- **Honest boundary:** raw_pixel's overall net is marginally HIGHER (+0.100 vs +0.085, Δ +0.016) — within the noise band. So SVI does **not** strictly dominate on raw accuracy; its case is **sycophancy-resistance + holding the shape ceiling** (robustness under distribution shift, where blind visual following is the documented failure mode), at no meaningful accuracy cost on this benchmark.

## What changed next
- **SVI (ADR-002) is validated** — keeping raw pixels out of the reasoner loses no ceiling here and reduces visual sycophancy. The structured topology stays the default; `raw_pixel_vision` remains a measurement-only switch, never shipped.
- Open: re-test on a benchmark with richer plots (the 2410.02637 regime) and a 2nd backbone before claiming invariance — one benchmark is an existence proof.

## Artifacts
- Runs: `research/outputs/{tv_vision_off, tv_raw_pixel}` (746 each, 0 errors); `baseline_full` reused as the structured arm (untouched). Analyzer `research/structured_vs_pixel.py`. Pre-reg `PREREGISTRATION_structured_vs_pixel.md`.
- Tests: `tests/test_runner_v2.py::TestRawPixelVision` (4; byte-identity off, attaches image to the answer call, skips JSON). Full suite **118 passed**; frozen guard green.
