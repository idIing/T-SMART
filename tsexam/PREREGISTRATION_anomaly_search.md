# Pre-registration — Autonomous architecture search: empower gpt-4o-mini on **anomaly** without losing **causality**

> **Post-hoc note (public release):** this document is the frozen scientific contract for the search
> reported in `research_journal/log/013` (a corrected null) and is preserved verbatim below for that
> record. The `orchestration/` harness it references (`score_variant.py`, `summarize_ledger.py`,
> `ORCHESTRATOR_BRIEF.md`, the variant ledger) has since been removed from the public repository and is
> no longer runnable as-is; see the post-hoc note in `log/013` for what remains.

**Date:** 2026-06-22  **Surfaces:** search = TSExam (`tsexam/`); held-out confirm = MMTS-Bench Base (`mmts_bench/`)
**Backbone:** **gpt-4o-mini** (temp 0), router + answer + vision, both arms.
**Status:** frozen **before** any variant is scored. Branch `free-response-regime`.
**Author:** master orchestrator (Opus 4.8), autonomous config-knob search.
**Design source:** plan `jazzy-petting-bentley.md`; `research_journal/log/006` (Track A de-confound),
`log/010` (why forcing AD vision fails on a weak backbone), `02_react_stages.md` (the agency switches).

---

## 1. Motivation (grounded in prior logs)

Track A (`log/006`) de-confounded architecture from backbone: at fixed gpt-4o-mini the deterministic
tools beat the raw model by **+13.5pp**, but the tools arm (OA **0.548**) still trailed TS-Agent's
published **0.602** by **−5.4pp**. That residual is **not spread evenly** — it is almost entirely the
**anomaly (AD) branch: −20.9pp** (ours ≈0.36 vs TS-Agent 0.57). Every other category is within a few
points or ahead; **causality (CA) is a strength** (TSExam ≈0.93; we beat TS-Agent on CA).

`log/010` diagnosed *why* the obvious fix (force AD vision, the gemini hero lever) fails here: at
gpt-4o-mini the **vision sensor is wrong 66%** of the time and the **follower is 91% sycophantic**, so
forcing the look feeds the reasoner garbage it then obeys. ⇒ the mechanism-grounded direction is
**deterministic-evidence accumulation / agency on anomaly** (refine / propose loops) and **suppressing**
the unreliable AD vision on this backbone — **not** more vision.

**This is a SEARCH, not a single ablation.** A master orchestrator will autonomously propose and score
many config-knob variants. The scientific hazard of any "try lots, report the best" procedure is
multiple-comparisons p-hacking. This pre-registration exists to make the search **incapable of
noise-mining**: one frozen objective, one frozen guardrail, one frozen decision rule, a **family-wise
correction across every variant tried** (none hidden — all land in `orchestration/variants_ledger.jsonl`),
and a **held-out replication** that a search-set fluke cannot survive.

## 2. Objective & guardrail (frozen — do not move after seeing results)

- **Objective (what we are trying to raise):** the **anomaly-branch** accuracy at gpt-4o-mini, toward
  TS-Agent's **AD 0.57**, measured by per-branch **paired McNemar** on the stratum
  `initial_branch_used == "anomaly"` (TSExam), confirmed on the **MMTS Base anomaly stratum**.
- **Hard guardrail (what must NOT regress):** the **causality, periodicity, trend** branches. A variant
  that buys AD by hurting any protected branch is **rejected outright**. Operationalised as per-branch
  bootstrap **CI-hi ≥ 0** on each protected branch, on *both* surfaces a variant reaches.

Why these guardrails: CA is our validated strength and the most important non-regression; PR/TREND are
strong branches whose forced-vision / agency interactions are the most plausible collateral-damage sites.

## 3. The variant space (config knobs only — all already `run_pipeline` kwargs)

Injected via the new `--config-json` arm; no pipeline code is written for the search. The orchestrator
uses judgment within this space (it is **not** a fixed grid):

- **Agency:** `loop_mode ∈ {reroute_once, refine, propose}`, `max_loop_depth ∈ {1,2,3}`,
  `loop_branches` (esp. **{anomaly}** alone vs with neighbours), `loop_on_flags`,
  `quality_threshold ∈ [0.45,0.65]`, `self_consistency ∈ {0,3,5}`.
- **Vision router:** `vision_gate ∈ {additive, learned}`, `no_vision_branches` (esp. **[anomaly]** — the
  log/010 "suppress the 66%-wrong look" insight), `forced_vision_branches`, `multi_branch`.

**Seeded hypotheses (mechanism-grounded, the orchestrator may invent others in-space):**
- **H1** `loop_mode=propose` + `loop_branches=[anomaly]` — accumulate deterministic anomaly evidence
  instead of trusting the sensor.
- **H2** `no_vision_branches=[anomaly]` (suppress the wrong look) + agency on anomaly.
- **H3** `vision_gate=learned` (the fitted gate already suppresses AD vision).
- **H4** `self_consistency` on the AD stratum.

## 4. Hypotheses (frozen, directional)

- **H_win (per-variant search-pass):** a variant raises the **AD** stratum with paired bootstrap
  **CI-lo > 0** on TSExam **AND** no protected branch regresses (each protected **CI-hi ≥ 0**).
- **H_confirm (the bar that makes a win count):** a search-passing variant's AD gain **replicates** on
  the **held-out MMTS Base anomaly stratum** (CI-lo > 0) with **CA still safe** (CI-hi ≥ 0) — *after* the
  family-wise correction across all variants tried (§5).
- **H0 (the honest null we expect and will report if true):** logs 004–009 show the agency space is
  largely **inert** on structured MCQ; the most probable outcome is a **broad, corrected null** — no
  variant clears CI-lo > 0 on AD that also survives Holm/BH and replicates on MMTS. *"This region of the
  config space cannot reliably empower a weak backbone on anomaly without collateral damage"* is a
  publishable, logged result. We will **not** widen the objective, relax the guardrail, or cherry-pick a
  variant to manufacture a win.

## 5. The *reliably-better* decision rule (stated verbatim, frozen)

> A variant counts as a **CONFIRMED-WIN** iff **all** hold:
> 1. **Search-pass (TSExam, gpt-4o-mini):** AD-stratum paired bootstrap **CI-lo > 0** AND each protected
>    branch (CA, PR, TREND) **CI-hi ≥ 0**.
> 2. **Family-wise correction:** the variant's AD McNemar p survives **Holm–Bonferroni across ALL variants
>    scored in the ledger** (`summarize_ledger.py`, reusing `diff_configs._holm`). Every variant tried is
>    logged; none is hidden from the correction.
> 3. **Held-out replication (MMTS Base, gpt-4o-mini):** the AD gain replicates (AD stratum **CI-lo > 0**)
>    with **CA CI-hi ≥ 0**.
>
> A variant failing any clause is **REJECT** (clause 1 fails) or **SEARCH-PASS but not confirmed** (clause 1
> holds, 2 or 3 fails). Only a CONFIRMED-WIN is reported as a result and considered for promotion.

The paired McNemar (not raw OA), the per-branch guardrail, the across-variants correction, and the
held-out replication are **jointly** the anti-p-hacking machine. The held-out step is decisive: a TSExam
search-set fluke from gpt-4o-mini temp-0 jitter will not replicate on an independent benchmark.

## 6. Method (paired, mechanism-located)

- **Cache the baseline ONCE** at gpt-4o-mini (`baseline` config) on the TSExam search set **and** MMTS
  Base; it is the paired anchor and is **never regenerated**. Every variant diffs against it.
- **Same rows.** Search set = TSExam at fixed `--n-per-cat 120 --seed 42` (deterministic sampling ⇒ every
  variant scores the identical rows as the cached baseline). **Sizing rationale (set before any variant is
  scored):** the AD category has exactly **108** rows; `--n-per-cat 120` captures **all of them** (and all
  of CA/NU/SA) so McNemar power on the *objective* branch is not kneecapped, while trimming only the
  oversized PR category (362→120) for cost. This is a power decision, not a hypothesis change. The confirm uses MMTS Base `--mcq-only`
  (isolates the vision/agency path from the numeric head via the structural `expected_schema` gate).
- **Stratify by `initial_branch_used`** (the pre-reroute router branch — deterministic at temp 0, so
  initial-route disagreement across arms is ~0%). Post-reroute `branch_used` is a treatment-affected
  collider and is **not** the stratifier. Stats: bootstrap 95% CI + exact McNemar (`paired_diff.stats`),
  Holm across the per-branch sweep within a run, and Holm/BH across variants in the ledger.
- **Scorer:** `orchestration/score_variant.py` shells out to the frozen harnesses (`run_eval.py`,
  `run_mmts_baseline.py`) via the `--config-json` arm, computes the per-branch paired diff, applies the
  §5 rule, and appends every stat + **token cost** to the ledger.

## 7. Stop rule & budget (autonomous, overridable)

- Stop when **either** a **CONFIRMED-WIN** is found **or** the budget is exhausted, whichever first.
- **Budget (orchestrator default, stated for transparency; the user may override):** up to **~15 scored
  variants**, with a wall-clock ceiling, stopping early on a confirmed win. Real **token cost** is summed
  per variant into the ledger (clients now accumulate `usage`; harnesses persist it to `metrics.json`).
  The honest prior (a null) means the search will not grind hundreds of variants; ~15 mechanism-seeded
  candidates exhaust the plausible region.

## 8. What would falsify / weaken a win

- AD-stratum CI-lo ≤ 0 on TSExam (no search-pass), or the gain vanishing under the across-variants Holm/BH
  correction (it was a multiple-comparisons artifact), or failing to replicate on MMTS (a search-set fluke).
- Any protected-branch CI-hi < 0 on either surface (the win was bought with collateral damage → reject).
- A gain that evaporates once the AD-stratum **coverage** (~11% of TSExam rows; the AD lever "reaches only
  ~5% of MMTS rows") is quoted — **coverage travels with every per-branch headline.**

## 9. Honesty caveats (must travel with any headline)

- **Backbone confound:** all numbers are at gpt-4o-mini; the comparison to TS-Agent's 0.57 is an
  *existence-proof placement* on our harness, **not** controlled invariance (harness/prompt/scoring differ).
- **gpt-4o-mini temp 0 is not bit-exact** (±~1.3pp floor, `log/006`). The mechanism stratifier + discordant-
  pair McNemar + held-out replication are what keep the search from being fooled by this jitter.
- The strong-gemini AD vision lever (+16.5pp on MMTS) is **not** assumed to transfer here — `log/010` says
  it backfires on this backbone; this search tests the *opposite* (suppress vision + accumulate evidence).
- Every variant → the ledger; the final result (win **or** corrected null) → a `research_journal/log/` entry.

## 10. Guardrails honored

- Default provider stays **gemini, byte-identical**; the Gemini model freeze is untouched. `--config-json`
  unused ⇒ both harnesses are byte-identical to today (tested). No frozen config (`baseline`, `nu_ad_fix`,
  …) changes behavior.
- **No prompt-letter nudging** anywhere — variants change *which evidence / route* the reasoner sees, never
  the answer letter. SVI intact (no raw pixels into the reasoner; corrections live in the tool/evidence
  layer). `OPENAI_API_KEY` read only from the repo-root `.env` / the environment.

## 11. Exact commands

```bash
# 1. cache the paired anchor ONCE (gpt-4o-mini)
python3 tsexam/run_eval.py --config baseline --provider openai \
    --n-per-cat 120 --seed 42 --tag as_baseline_4omini --max-workers 12
cd ../mmts_bench && python3 scripts/run_mmts_baseline.py --subset Base --config baseline \
    --provider openai --mcq-only --max-workers 16   # cached MMTS anchor

# 2. per variant (the orchestrator's one tool call)
python3 orchestration/score_variant.py --variant '{"no_vision_branches":["anomaly"], \
    "loop_mode":"propose","loop_branches":["anomaly"],"max_loop_depth":2}' \
    --backbone openai --label H1_propose_ad --baseline-tsexam <cached> --baseline-mmts <cached>

# 3. after the search: family-wise correction + verdict
python3 orchestration/summarize_ledger.py
```
