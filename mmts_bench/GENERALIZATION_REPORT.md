# Does conditional visual escalation generalize? — T-SMART on TimeSeriesExam1 → MMTS-Bench

**Run:** 2026-06-18. **Model:** `gemini-3.1-flash-lite` (router + reasoner + vision
sensor), temp 0.0. **Config under test:** the **frozen** `nu_ad_fix` =
`no_vision_branches=["noise"]`, `forced_vision_branches=["anomaly"]` — the exact
switches validated on TimeSeriesExam1, copied byte-for-byte into the MMTS harness
with **no** prompt/letter tuning. Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md).

---

## 0. TL;DR

- **Infrastructure.** Removing the GeminiClient's serial 2 s lock (network call now
  runs outside the spacing lock) took the 746-row TimeSeriesExam eval from **1.9 h
  → 7.8 min** (9.27 → 0.62 s/row, 24 workers, errors=0). This is what made a paired
  multi-thousand-row MMTS experiment feasible at all.
- **TimeSeriesExam (home dataset), now with a paired test.** baseline vs `nu_ad_fix`
  on the full 746-row set: **micro OA 0.682 → 0.714 (+3.2pp, McNemar p=0.020)**.
  The effect is **surgical and mechanism-located**: the **anomaly** branch (forced
  line-plot) gains **+15.9pp (p=0.029)**; the **noise** branch (vision suppressed)
  gains +7.0pp (p=0.136, *not* significant); periodicity/trend/similarity/causality
  move <1pp. Vision firing drops 76.1% → 59.0%.
- **MMTS-Bench (never-tuned-on), pre-registered paired test (1865 scoreable MCQ
  rows).** Overall OA **0.498 → 0.508 (+1.0pp, p=0.16, n.s.)**. The result splits
  exactly along the mechanism, and **both halves point the same way they did on
  TSExam**: (a) the **noise** lever is **non-negative** — Δ+1.4pp (p=0.70), 51/56
  discordant: suppressing vision on noise questions is *safe* (PRIMARY supported);
  (b) the **anomaly** lever — which I expected to act on ~0 MMTS rows — actually
  fires on **91 router-selected rows** and delivers a **significant +16.5pp
  (p=0.024)**, *replicating its TSExam +15.9pp almost exactly on a benchmark it was
  never tuned on.* Match & Align are **exactly 0.000** (vision-bypass paths →
  perfect negative control). No category significantly regresses.
- **Honest bound.** The *overall* number is small (+1.0pp) because the strong lever
  reaches only 91/1865 rows. The claim that survives scrutiny is **mechanism
  transfer**, not a second headline landslide: the same routing+escalation rule
  produces same-signed, same-magnitude effects OOD, and never regresses. Details §4–5.

---

## 1. Infrastructure (Phase 0) — parallelization

`GeminiClient.generate` held `_call_lock` **around** the network call and slept 2.0 s
between calls → fully serial, ~30 RPM, ~9 s/row. Fix (`tsqa/llm/client.py`): keep a
tiny critical section that only enforces 0.02 s spacing and stamps the send time,
then run `generate_content` **outside** the lock so concurrent worker threads have
overlapping in-flight requests. The 90 s per-request HTTP timeout and 429/503 retry
loop are unchanged.

| eval | before | after |
|---|---|---|
| TimeSeriesExam, 746 rows | ~6914 s (9.27 s/row, serial) | **466 s (0.62 s/row, 24 workers)** |
| concurrent smoke, 20 rows | — | 16 s, errors=0 |

Throughput on MMTS is ~0.56 rows/s/config (not RPM-bound — ~200 RPM, far under the
4000 cap — but bound by per-row vision-call latency and GIL-held statsmodels/ruptures
work). The sweet spot used: subsets sequential, the two configs of a subset
concurrent (≈48 workers, ~400 RPM).

## 2. TimeSeriesExam — paired full-set result

Two separate full runs (`baseline_full`, `nu_ad_fix_full`), joined per-row on `id`.
Router-branch assignment was **identical across the two runs (0.0% disagreement)** —
temp-0 routing is deterministic, so the config change is the only moving part.
Analysis: `research/paired_diff.py`; raw output: `research/outputs/paired_tsexam_full.txt`.

```
stratum               n    base   treat       Δ            95% CI      b/c McNemar p
OVERALL             746   0.682   0.714  +0.032  [+0.005,+0.058]    37/61    0.0197 *
-- by branch_used (MECHANISM VIEW) ---------------------------------------------------
anomaly              82   0.524   0.683  +0.159  [+0.024,+0.293]     9/22    0.0294 *
noise               186   0.565   0.634  +0.070  [-0.016,+0.156]    26/39    0.1360
periodicity         156   0.763   0.756  -0.006  [-0.019,+0.000]      1/0    1.0000
trend               126   0.825   0.817  -0.008  [-0.024,+0.000]      1/0    1.0000
similarity          162   0.660   0.660  +0.000        —              0/0    1.0000
causality            30   0.933   0.933  +0.000        —              0/0    1.0000
vision fired: base 568/746 (76.1%)  ->  treat 440/746 (59.0%)
```

**Reading it honestly.** The headline fix effect (+3.2pp, significant) is carried by
the **anomaly** lever. The **noise** lever is positive but not independently
significant, and at the *category* level **NU is flat (0.571→0.571, 18/18 churn)** —
the stored claim that the fix lifts NU does **not** survive at full scale; the lift is
on AD (+11.1pp) via the anomaly branch. Macro OA 0.656 → 0.684.

**vs. TS-Agent.** `nu_ad_fix` macro 0.684 vs TS-Agent 0.602 (Table 2 of
[TS-Agent](https://arxiv.org/abs/2510.07432); PR .71/NU .61/AD .57/SA .57/CA .55) =
**+8.2pp**. **Caveat (load-bearing):** TS-Agent uses **gpt-4o-mini**; we use
**gemini-3.1-flash-lite**. That headline therefore conflates *architecture* with
*backbone*. The backbone-controlled claim is the paired ablation above (+3.2pp at
fixed model), which is the honest measure of the vision-routing idea itself.

## 3. MMTS-Bench data-quality scoping (read before believing any OA)

T-SMART is architecturally an **MCQ** system (router → evidence → A/B/C/D
interpreter). MMTS-Bench Base mixes in **free-response** items the framework cannot
answer and that score wrong by construction in *every* config:

| subset | total | scoreable MCQ | unscoreable | what the unscoreable are |
|---|---|---|---|---|
| Base | 700 | **307 (44%)** | 393 | `basic analysis` 200 (std/median/percentile → a number), `stationarity` 100, scattered |
| InWild | 1084 | 1081 (100%) | 3 | — |
| Match | 400 | 400 (100%) | 0 | multi-series "choose/find/smooth/reverse it" |
| Align | 240 | 240 (100%) | 0 | caption↔series |
| **total** | **2424** | **2028** | **396** | |

The experiment runs MCQ rows only (`--mcq-only`); the diff also defaults to dropping
any row without a valid A/B/C/D gold. **Reporting MMTS OA without this filter (as a
naïve run does — Base would read ~27–37%) measures "fraction of questions that happen
to be MCQ," not model skill.** The 396 free-response items are logged as an explicit
out-of-scope coverage gap (see §5, and handoff item on a regression head).

## 4. MMTS-Bench generalization — pre-registered paired result

Paired baseline vs `nu_ad_fix` on the **2028 scoreable MCQ rows**, joined on
`(subset, sample_id)`. Router-branch disagreement across the two runs: **0.0%**
(deterministic; the config is the only moving part). Analysis:
`scripts/diff_configs.py`. Driver: `scripts/run_generalization.sh`.

### 4a. Early partial — Base only (n=278, real, non-smoke)
```
OVERALL              278   0.471   0.489  +0.018  [-0.036,+0.068]   25/30   p=0.590
noise (branch)       142   0.345   0.366  +0.021  [-0.063,+0.106]   18/21   p=0.749   <- PRIMARY, non-negative
anomaly (branch)      26   0.308   0.423  +0.115  [-0.192,+0.385]    6/9    p=0.607   (the lever DOES fire on a few MMTS rows)
periodicity / trend  flat (surgical)
```

### 4b. Pooled across all 4 subsets (n=1865 scoreable MCQ)

Of 2028 scoreable rows, 1865 paired cleanly (163 lost to per-row skips/errors,
mostly Align's free-response `ts2caption` half and InWild parse drops). Full output:
[`outputs/generalization/diff_pooled.txt`](outputs/generalization/diff_pooled.txt).

```
stratum                 n     base    treat        Δ            95% CI      b/c   McNemar p
OVERALL              1865    0.498    0.508   +0.010   [-0.003,+0.023]    65/83      0.1621
-- by subset ------------------------------------------------------------------------------
InWild               1067    0.499    0.511   +0.012   [-0.006,+0.030]    40/53      0.2132
Match                 400    0.585    0.585   +0.000        —              0/0      1.0000   <- bypass (neg. control)
Base                  278    0.471    0.489   +0.018   [-0.036,+0.068]    25/30      0.5901
Align                 120    0.267    0.267   +0.000        —              0/0      1.0000   <- bypass (neg. control)
-- by branch_used (MECHANISM VIEW) --------------------------------------------------------
anomaly                91    0.286    0.451   +0.165   [+0.033,+0.297]    12/27      0.0237 *  <- replicates TSExam +15.9pp
noise                 355    0.420    0.434   +0.014   [-0.042,+0.070]    51/56      0.6992    <- PRIMARY: non-negative
trend                 238    0.634    0.630   -0.004   [-0.013,+0.000]     1/0      1.0000
periodicity           192    0.557    0.552   -0.005   [-0.016,+0.000]     1/0      1.0000
similarity            317    0.524    0.524   +0.000        —              0/0      1.0000
similarity_multi      620    0.484    0.484   +0.000        —              0/0      1.0000
causality              52    0.577    0.577   +0.000        —              0/0      1.0000
```

The **anomaly** branch draws its 91 rows from InWild (65) + Base (26) — categories
`index`, `basic analysis`, and various `*_reasoning` — i.e. the router finds
anomaly-shaped "where/which-point" questions even though MMTS has no anomaly
*category*. That the forced-line-plot lever then replicates its TSExam effect
(+16.5pp here vs +15.9pp there, both p≈0.02–0.03) is the single strongest piece of
generalization evidence in this study. The **noise** branch shows heavy churn
(51 helped → wrong, 56 wrong → helped) that nets slightly positive — i.e. the
visual suggestion on noise questions is ~coin-flip, so removing it is free.

### 4c. Pre-registered verdict

- **PRIMARY** (noise-branch non-regression): **SUPPORTED.** Δ+1.4pp, CI
  [-0.042,+0.070], McNemar p=0.70 — neutral-to-positive, not a regression. The
  `no_vision_branches=["noise"]` claim ("don't escalate statistical noise
  questions to a picture") holds under distribution shift.
- **GUARDRAIL** (neutrality): **HELD.** No category significantly regressed; the two
  vision-bypassing subsets (Match, Align) are byte-identical (Δ=0.000, 0/0). The fix
  is surgical, not a global perturbation.
- **SECONDARY** (headline OA): overall +1.0pp (0.498→0.508, p=0.16) — positive but
  not significant; expected, since the strong lever reaches only 5% of rows.
- **Bonus, unplanned:** the **anomaly** lever transferred *significantly* (+16.5pp,
  p=0.024) on the rows it reached — stronger than the pre-registration dared predict
  ("acts on ~0 rows"). The router's branch assignment is itself what generalizes.

## 5. Brutally honest limitations

1. **The headline OA gain is small (+1.0pp) and not significant — do not oversell
   it.** The strong, significant lever (forced line-plot on anomaly) reaches only
   91/1865 rows, so even a +16.5pp effect there barely moves the pooled average. The
   defensible claim is **mechanism transfer + non-regression**, not "T-SMART is
   +X% better on MMTS." Anyone quoting the +16.5pp anomaly number must also quote
   that it is a 5%-of-rows stratum. The noise lever, the one the fix was *named*
   for, is only neutral here (and was non-significant on TSExam too) — its value is
   *removing a failure mode for free*, not adding accuracy.
2. **The "+8.2pp vs TS-Agent" is backbone-confounded** (gemini-3.1-flash-lite vs
   gpt-4o-mini). The architecture-only claim is +3.2pp paired, at fixed model.
3. **MMTS Base coverage gap.** 56% of Base (free-response numeric / stationarity) is
   out of scope for an MCQ pipeline. We measure the 44% it can answer. A faithful TS
   reasoner needs a **regression / numeric head**, not just a letter classifier.
4. **Low absolute MMTS-noise accuracy.** The noise branch scores ~0.35 on MMTS Base
   vs ~0.57 on TSExam — the deterministic stats transfer poorly to MMTS's
   noise-level wording. The fix doesn't *hurt*, but the branch itself is weak OOD.
5. **Still a decision tree, not an agent** (per the project audit): single-shot
   routing with no self-correction; the vision gate is a **hand-tuned additive-score
   heuristic**, exactly the kind of brittle rule the audit flagged. Generalizing the
   *gate* (handoff item 2) matters more than this single frozen config transferring.
6. **Determinism caveat.** "0.0% routing disagreement" is reassuring but
   gemini temp-0 is not contractually bit-exact; tiny reasoner flips contribute to
   the discordant counts and are part of the measured noise floor.

## 6. Handoff — PhD-level agenda for the next orchestrator

(Carried from the plan; sharpened by what the data now shows. Every deliverable:
pre-registered, paired-significant, generalization-checked, **zero** prompt-letter
nudging.)

1. **A learned "should-I-look" gate** (top priority). Replace `verifier/checks.py`'s
   additive-score heuristic with a calibrated rule keyed on deterministic-evidence
   confidence/completeness. The frozen-config result shows the *value* of conditional
   escalation lives in **which branch/uncertainty** triggers it; learn that boundary
   and show it matches the hand rules **and** transfers. This is the real
   generalization claim — the single config transferring is the weak version.
2. **Quantitative visual sycophancy.** When `vision_letter ⟂ math evidence`, measure
   how often the reasoner follows vision vs math and which is right; derive the
   optimal trust policy. Directly motivated by the noise-branch story.
3. **A third benchmark + the anomaly lever's home.** Find a TS-QA benchmark *with* an
   anomaly category to test the **strong** lever OOD (MMTS can't). 
4. **Regression head for free-response** (unlocks ~400 MMTS Base rows + real
   numeric-estimation skill). Architectural, not a nudge.
5. **Two-model fidelity** (Lite router + Flash reasoner) and **multi-series routing**
   for two-series AR(1)/anomaly questions (the residual error cluster).
6. **NU recovery, clearly labeled** (Niharika's downsampling as a *named*
   preprocessing step on the noise branch; report isolated effect; do NOT fold into
   the generalization claim).

---
*Artifacts:* `PREREGISTRATION.md`, `scripts/run_generalization.sh`,
`scripts/run_mmts_baseline.py` (`--config/--mcq-only/--max-workers`),
`scripts/diff_configs.py`, `outputs/generalization/`,
`tsexam/paired_diff.py`, `.../outputs/paired_tsexam_full.txt`.
*Literature:* [MMTS-Bench](https://arxiv.org/abs/2602.08588) (2,424 pairs, 4 subsets),
[TS-Agent](https://arxiv.org/abs/2510.07432) (tool-grounded reasoner, gpt-4o-mini).

---

## 7. The Graceful Avalanche — decision-tree → agentic transition, executed (2026-06-22)

Full scoreboard: [`AGENTIC_TSMART_RESULTS.md`](../AGENTIC_TSMART_RESULTS.md); per-run detail in
`research_journal/log/004–009`. Branch `agentic-tsmart` (7 gates pushed). The Stage 1→3 ReAct
transition (ADR-007) was built and evaluated end-to-end as pre-registered, paired, cross-benchmark
experiments:

- **Agentic self-correction is safe but INERT on structured time-series MCQ** — three escalating
  pre-registered H0s: trigger dormant (S1, `evidence_incomplete` 0/746) → gate fires but no candidate
  (S2, `quality<0.55` fires 123/746, 0 second-branches) → LLM proposer executes 120 real actions but
  **net-neutral** (S3, 6 corrected/8 broken). Holds in-distribution **and** OOD (fallback ~0% on MMTS
  Base/InWild/Match).
- **`agentic_tsmart` ties baseline across all three benchmarks** (TSExam −0.7pp · MMTS pooled −0.1pp ·
  TSRBench −0.5pp; all Holm n.s.; no branch/category CI entirely <0) → the full agentic scaffold is
  **validated non-regressing / safe to ship dark**. Nothing earned per-branch promotion; the shipped
  default stays the Stage-0 tree. This **extends §4's single-config transfer to a full cross-benchmark
  non-regression guarantee** (now incl. TSRBench, a third surface — agenda item 3, partial).
- **Architecture de-confound (Track A — addresses the §5 backbone caveat):** T-SMART's deterministic
  architecture on gpt-4o-mini beats the raw model by **+13.5pp** (p<0.0001), but our gpt-4o-mini
  tools-arm trails TS-Agent's published number by −5.4pp → the cross-backbone "+8.2pp" was substantially
  backbone, not architecture.
- **Visual sycophancy quantified + SVI validated (Track V — closes agenda item 2):** on the vision-fired
  subset, raw pixels are more sycophantic than the structured JSON topology (trend follows-wrong-look
  **73% vs 40%**) with no shape-ceiling gain (trend structured **+20.9** vs pixel +9.9) — keeping pixels
  out of the reasoner is a robustness win, not a ceiling tax. `raw_pixel_vision` is measurement-only.

**Net:** the conditional-escalation generalization claim (§4) is now backed by a full cross-benchmark
non-regression validation; the architecture claim is de-confounded; agenda items 2 (sycophancy) and
partially 3 (third benchmark) are delivered. The open lever remains item 1 (the learned should-ACT gate)
+ the free-response / numeric regime, where this study shows the agentic value — if any — must live.

---

## 8. The interpretation regime — free-response load-bearing proof + weak-backbone vision null (2026-06-22)

Branch `free-response-regime` (logs 010–011). The Avalanche left one reframe to test: **the
architecture's value is a function of what the backbone cannot do itself.** Two cheap, pre-registered,
paired experiments tested it from its two load-bearing corners — and the headline is a **single unified
thesis**, not "the vision-gate is the story."

### The two results

- **Free-response is where tools are load-bearing on a STRONG backbone (log/011, the WIN).** On the
  386 MMTS-Base `numerical` rows, the deterministic numeric head beats an honest **`llm_numeric`
  counterfactual** (the *same* gemini computing the number from the raw series, head bypassed) by
  **+30.6pp overall** and **+31.8pp on the closed-form stratum** (tool 0.997 vs model 0.679; **104/0**
  discordant; McNemar p<0.0001; CI [+26.9,+36.7]). Per quantity: **mean +64.5pp, median +51.3, std
  +43.5, percentile +36.3**; min/max tie (the model *can* scan for extrema); **slope** is the lone
  reversal (model 0.74 > tool 0.42, n.s. — a method-sensitive bound, pre-registered). The gap *widens*
  as tolerance tightens (+58.7pp @1%). This is the mirror image of the MCQ nulls: where the model
  cannot compute the number, the tool **owns** it — **+30pp even on a capable backbone.** T-SMART is a
  **general framework in the numeric regime, not an MCQ trick.**

- **The vision lever does NOT rescue a WEAK backbone (log/010, the NULL — corrects the working
  hypothesis).** Forcing anomaly vision at fixed gpt-4o-mini (`ad_vision_only` vs the on-disk
  `vision_off` arm) moves the anomaly stratum only **+2.5pp (n.s., p=0.80; 9 fixed / 7 broken)**, OA
  flat (0.548→0.550). The **same** lever gave **+15.9pp on gemini**. Mechanism (pre-registered
  sycophancy diagnostic): the gpt-4o-mini vision **sensor** is wrong **66%** of the time, and the
  follower is **91% sycophantic** (follows 30/33 wrong looks) — two compounding weaknesses, so the
  look hurts as often as it helps. ⇒ The −5.4pp vs TS-Agent (Track A) is **not** the suppressed
  vision lever; **vision strengthens a capable backbone, not a weak one.**

### The unified headline (what to sell)

> **Mechanism value = f(backbone capability × task regime).** The architecture pays exactly where the
> backbone is weak *for that task*: (1) **tools that compute** rescue *any* backbone on free-response
> numeric, because no LLM reads a std off raw values (+30.6pp on strong gemini); (2) **conditional
> structured sycophancy-resistant vision** helps a *capable* backbone interpret shape (+16pp on
> gemini), but a *weak, 91%-sycophantic* backbone cannot exploit a noisy look. This is **also** the
> sharpest case for the SVI (Track V / log/009): **sycophancy resistance matters most exactly where the
> follower is most sycophantic** — weak backbones. The decision-tree→ReAct nulls (logs 004–008), the
> Track-A de-confound (006), Track V (009), and these two entries are one finding: stop optimizing
> confident-tool mono-aspect MCQ — the architecture's value lives in the regimes where the model
> can't do the job itself.

**Honesty caveats that travel:** log/011 is single-surface (MMTS Base) + single-backbone (gemini);
log/010 is a single backbone-swap on our harness (existence test, not invariance), and the AD lever
reaches ~7% of rows. The genuine next lever is a **learned should-ACT gate** keyed on the
method-sensitive quantities (e.g. `slope`, where the model beats the tool) — the first concrete place
agency could pay — built *only* on this free-response evidence, never for MCQ.

**Scope of headline (1) — do not over-hear it (added 2026-06-22 after PI scrutiny):** the +30.6pp is
*tool ownership of COMPUTATION* — on the numeric head the LLM is **bypassed** (0 answer-LLM calls; the
value is pure numpy + a deterministic keyword→quantity map), so it is **not** evidence the
reasoning/interpretation layer *augments* the LLM; on those rows the framework is a deterministic
dispatcher. The keyword map is itself brittle (single-quantity, no composition, no typo handling) and
MMTS Base — 4 templates, 0 compositional/typo/paraphrase rows — cannot expose that. Where the LLM *is*
load-bearing in the numeric regime is **parsing** (composition/paraphrase/typos), isolated as a
three-arm study (A0 keyword vs A1 deterministic-grammar+fuzzy vs A2 +LLM-planner) on a synthetic stress
set in **log/012**. Base's 6% misses are *definitional* (slope/period/count), not parsing; the two rows
once flagged as bugs (554, 557) are CSV/data artifacts (schema gate verified **0/700** systematic leaks).

**log/012 result (the decomposition):** the LLM is load-bearing for parsing **only on open paraphrase**
(A2−A1 = **+0.75**, McNemar p<0.0001). **Composition and typos are tool-tractable** — a deterministic
grammar + fuzzy/difflib parser recovers them (A1−A0 = +0.85 / +0.83) and the LLM adds ~0 there; Base
**non-regresses** (A0=A1=A2=0.88, LLM fire-rate 0%). So the numeric regime splits three ways: the
**tool** owns *computation* (log/011) **and** *structured/typo parsing* (A1); the **LLM** owns only
*open-paraphrase semantics* (A2) — the smallest, most specific slice, gated conditionally so it is free
on the clean path. The honest answer to "are the tools comprehensive enough alone?": yes for computation
and structured parsing, no for open paraphrase — exactly where the LLM earns its keep.
