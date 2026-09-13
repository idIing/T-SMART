# 010 — Track A2: does the anomaly-vision lever close the TS-Agent gap on a weak backbone? (NULL)

---
id: 010
date: 2026-06-22
stage: 0
claim: C1
status: complete
evidence_level: fixed-backbone-ablation
claim_scope: mechanism-local
overall_significance: "anomaly stratum Δ +2.5pp, McNemar p=0.80 (NULL); OA 0.548→0.550 (n.s.)"
prereg: agentic-setup/PREREGISTRATION_c1_gpt4omini_vision.md
commit: free-response-regime gate (see git log)
artifact_present: yes
required_caveats: "Single backbone swap on OUR TSExam harness (existence test, not invariance); harness/prompt/scoring differ from TS-Agent's own setup. gpt-4o-mini temp-0 is not bit-exact (control is a prior-session run). The lever fires on ~7% of rows (anomaly stratum)."
---

## Hypothesis (pre-registered)
- **H1:** at fixed gpt-4o-mini, forcing anomaly vision (`ad_vision_only`) lifts the **anomaly
  stratum** vs the `vision_off` arm — paired Δ > 0, McNemar p<0.05 — i.e. the **vision-gate**, not
  more agency, is what closes the −5.4pp-vs-TS-Agent gap (which is −20.9pp on the AD branch, the
  branch Track A measured with vision suppressed).
- **H0 (pre-declared):** the lever does **not** help the weak backbone (Δ≈0 or negative) — a genuine
  result. Pre-registered mechanism to check: gpt-4o-mini may be *more* sycophantic to the look.
- **Decision rule:** H1 supported iff anomaly-stratum paired Δ>0 at McNemar p<0.05 and no other
  branch regresses.

## Setup
- Configs: control = `vision_off` (REUSED `c1_gpt4o_tools` on disk); treatment = `ad_vision_only`
  (vision_off on the five non-anomaly branches **+ forced anomaly vision**) — the single moving part
  vs control is the anomaly-vision lever.
- Dataset: TSExam, full 746 MCQ rows; backbone **gpt-4o-mini**, temp 0, router+answer+vision sensor.
- New run: `c1_gpt4o_ad_vision_only` (746 rows, errors 0, vision fired 6.7% = the anomaly rows).

## Result — **NULL** (H0 holds)
Paired `ad_vision_only − vision_off`, stratified by the tools-arm initial router branch:

| stratum (initial route) | n | vision_off | ad_vision_only | Δ (pp) | 95% CI | McNemar p (Holm) |
|---|---|---|---|---|---|---|
| OVERALL | 746 | 0.548 | 0.550 | +0.1 | [−1.3,+1.6] | 1.00 |
| **anomaly** | 79 | 0.316 | 0.342 | **+2.5** | [−7.6,+12.7] | **0.80** (1.00) |
| AD category | 108 | 0.361 | 0.380 | +1.9 | [−5.6,+9.3] | 0.81 |
| noise / similarity / trend / periodicity / causality | — | — | — | ≈0 | — | 1.00 |

**Discordant pairs on the anomaly stratum: +9 corrected, −7 broken** (net +2, n.s.) — a coin flip,
exactly like the gemini **noise** branch. The lever that delivered **+15.9pp on gemini** (TSExam
`nu_ad_fix`, p=0.029) / **+16.5pp on MMTS** (p=0.024) produces **+2.5pp n.s.** on gpt-4o-mini.

**Mechanism (pre-registered sycophancy diagnostic) — why the look doesn't help the weak backbone.**
On the 50 anomaly rows where vision fired at gpt-4o-mini:
- the gpt-4o-mini vision **sensor** is **wrong 66% of the time** (only 17/50 looks correct);
- the gpt-4o-mini **follower** is **91% sycophantic** — it follows **30/33 wrong looks** (and 16/17
  right looks): it does whatever the sensor says, almost regardless of correctness;
- ⇒ accuracy on vision-fired anomaly rows ≈ the look's accuracy ≈ **32%** (16/50). The two
  weaknesses compound: a noisy sensor + a blind follower = the look hurts as often as it helps.

(Contrast gemini: a more accurate sensor **and** a more selective follower turn the same forced look
into a clean +16pp. This is Track V's SVI/sycophancy thesis across backbones — sycophancy resistance
matters **most** on a weak backbone, which is ~91% sycophantic.)

## Verdict — pre-registered **H0 supported; H1 rejected**
Forcing anomaly vision does **not** close the weak-backbone gap. The honest implication **corrects the
working hypothesis**: the −5.4pp vs TS-Agent is **not** the suppressed vision lever — the vision-gate
strengthens a *capable* backbone (gemini), not a weak one. The "vision-gate is the headline" story is
therefore **conditional on backbone capability**, which is itself the program's central thesis
(mechanism value = f(backbone)). A clean, mechanistically-explained null.

## What changed next
- Suppressed: the idea that the vision lever rescues a weak backbone (do **not** quote vision as a
  weak-backbone gap-closer). The −5.4pp vs TS-Agent remains attributable to backbone capability +
  TS-Agent's other mechanisms + harness differences, **not** to our suppressed AD vision.
- Strengthens **Track V (log/009)**: the SVI's sycophancy resistance is most valuable exactly where
  the follower is most sycophantic (weak backbones at 0.91). Feeds the item-2 framing.
- Open → the genuine weak-backbone lever is the **deterministic tools** (Track A +13.5pp), not vision;
  and on a strong backbone the load-bearing regime is **free-response** (log/011), not MCQ vision.

## Artifacts
- Runs: `research/outputs/c1_gpt4o_ad_vision_only` (746, errors 0); control `c1_gpt4o_tools` reused.
- Analysis: `research/paired_diff_route.py`; sycophancy diagnostic inline (this entry).
- Pre-reg: `agentic-setup/PREREGISTRATION_c1_gpt4omini_vision.md`.
