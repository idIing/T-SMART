# 013 — Autonomous architecture search: can agency/vision knobs empower gpt-4o-mini on anomaly? (CORRECTED NULL)

---
id: 013
date: 2026-06-23
stage: 0
claim: C1
status: complete
evidence_level: mechanism-local-stratum
claim_scope: per-initial-route-branch
overall_significance: "AD: no variant CI-lo>0 (H2 Δ−0.014 p=1.00; H3 Δ−0.061 p=0.375). BROAD CORRECTED NULL. Headline finding: gpt-4o-mini router only ~60% reproducible → AD effects are noise-limited on this backbone."
prereg: tsexam/PREREGISTRATION_anomaly_search.md
commit: T-SMART main (this commit; see git log)
artifact_present: metrics + ledger committed; raw per-row results.json gitignored, regenerable by script (see ## Artifacts)
required_caveats: "gpt-4o-mini routing is NON-reproducible: two runs differing only POST-routing agree on initial_branch_used just ~60% (40% flip, mostly in/out of fallback). Anomaly-routed n swings 61→71→49 across runs. ⇒ per-stratum AND overall paired diffs carry large routing-luck noise; only LARGE true effects are detectable. Contrast gemini (deterministic temp-0 routing) where the config is the only moving part. AD lever reaches ~10-14% of rows. Backbone confound: TS-Agent 0.57 is existence-placement on OUR harness, not controlled invariance."
---

## Hypothesis (pre-registered)
- **H_win:** a config-knob variant raises the **anomaly** stratum (paired bootstrap CI-lo > 0) at
  gpt-4o-mini WITHOUT regressing causality/periodicity/trend (each CI-hi ≥ 0), survives Holm across ALL
  variants, AND replicates on the MMTS Base anomaly stratum.
- **H0 (pre-declared):** logs 004–009 + log/010 say this space is largely inert on a weak backbone; the
  most probable outcome is a **broad, corrected null** — a genuine, logged result.
- **Decision rule (frozen in `PREREGISTRATION_anomaly_search.md` §5 before any variant scored):**
  CONFIRMED-WIN iff (1) TSExam AD CI-lo>0 & protected CI-hi≥0, (2) AD McNemar p survives Holm across the
  whole ledger, (3) replicates on MMTS Base anomaly stratum (CI-lo>0, CA safe).

## Setup
- Search surface: TSExam, gpt-4o-mini (router+answer+vision), `--n-per-cat 120 --seed 42` (504 rows;
  captures all 108 AD-category rows → AD power not kneecapped by sampling). Held-out confirm: MMTS Base
  `--mcq-only` (only reached on a SEARCH-PASS — none occurred).
- Paired anchor: `as_baseline_4omini` (504 rows, errors 0; macro OA 0.425, AD 0.333, CA 0.403,
  vision_rate 0.371, **fallback_rate 0.464**). Cached ONCE; every variant diffs against it.
- Orchestration (this commit): `orchestration/score_variant.py` shells the frozen harnesses via the new
  `--config-json` arm (validated against the `run_pipeline` signature), computes per-branch paired
  McNemar + bootstrap CI stratified by the TREATMENT arm's `initial_branch_used`, applies the §5
  guardrail, sums token cost → `variants_ledger.jsonl`; `summarize_ledger.py` applies Holm across the
  family. Tests: `orchestration/tests` (guardrail/stratify/Holm pure-function coverage). Both
  `--config-json` arms default-off ⇒ byte-identical when unused (pytest green: agentic 129, mmts 89).

## Diagnose-first (read-only, no API) — which knobs are LIVE on the AD stratum
Baseline `initial_branch_used == anomaly` stratum (n=61):
- **vision fires 83.6%** (51/61); acc | fired = 0.333 vs | not-fired = 0.600 → **−26.7pp observational**.
  BUT this is selection (the additive gate fires on the ambiguous rows), NOT causal — the paired H2 diff
  settles it (and does: ~0).
- **evidence_incomplete = 0%** → `reroute_once` AND `self_consistency` are **analytically dormant on AD**
  (both fire ONLY on that flag — runner.py:763; self_consistency=k is byte-identical to baseline on AD).
  Closed by code-read, no API spent.
- quality_score not on baseline → the **propose/refine** agency gate (`quality_score<threshold`,
  runner.py:636) live-ness is the ONE open question → H1 probe below.

⇒ live knobs = **vision routing** (H2/H3/control) + **maybe** the propose loop (H1). Everything else
dormant for free.

## Result — per-variant, stratified by **initial route** (pre-treatment). gpt-4o-mini, TSExam 504 rows.
| variant (label) | AD n | AD Δ | AD 95% CI | AD b/c | AD p | CA Δ (CI-hi) | OA Δ | verdict |
|---|---|---|---|---|---|---|---|---|
| **H2** `no_vision_branches=[anomaly]` (suppress wrong look) | 71 | **−0.014** | [−0.085,+0.056] | 4/3 | **1.000** | +0.133 (+0.300) | −0.030 | **REJECT** |
| **H3** `vision_gate=learned` (gemini-fit gate) | 49 | **−0.061** | [−0.143,+0.020] | 4/1 | 0.375 | +0.158 (+0.342) | **−0.081 (p=0.001)** | **REJECT** |
| **control** `forced_vision_branches=[anomaly]` | covered by **log/010** (full 746): forced AD vision Δ**+2.5pp, p=0.80 NULL**. On-anchor re-run started then **killed** (brutal TPM restart; redundant given log/010; noise-limited per the router finding below). | | | | | | | (cited, not re-run) |
| **H1** `loop_mode=propose,loop_branches=[anomaly],q=0.65` (n=30 probe) | 20 | **DORMANT** — loop fired **0/20**; quality_score always ≥0.814 (median 0.85) ≫ 0.65 gate ⇒ byte-identical to baseline on AD | | | | | | **dormant** |

**H2 (suppress the 66%-wrong look) = clean NULL on anomaly** (Δ−0.014, b/c=4/3, p=1.000). The −26.7pp
observational gap was **selection bias**: log/010 already showed the *causal* effect of forcing the look
is +2.5pp n.s.; suppressing it is −1.4pp n.s. → the AD look is a **coin flip in both directions** on this
backbone (sensor wrong 66%, follower 91% sycophantic — log/010). Guardrails safe.

**H3 (learned gate) = REJECT, and does NOT transfer to gpt-4o-mini.** AD Δ−0.061 n.s.; but **OA Δ−0.081
(p=0.001)** — the gemini-fit "fire {trend,periodicity}, suppress rest" gate (memory `learned_gate_built`)
*overall-regresses* on gpt-4o-mini. Part is genuine (similarity vision suppressed → SA Δ−0.111, p=0.052),
part is routing-luck (below). Either way the LOOK-decision policy fit on a capable backbone is **not**
safe to transfer to a weak one. CA never regresses on any arm (a robust non-regression of the guardrail).

## HEADLINE METHODOLOGICAL FINDING — gpt-4o-mini routing is only ~60% reproducible
H2 changes behavior **only post-routing** (AD vision), so `initial_branch_used` should be identical to
baseline IF routing were deterministic. It agrees **303/504 = 60.1%** (H3: 284/504 = 56.3%). The 40%
that flip go mostly **in/out of the fallback path** (`similarity↔None` ~30 each way; `causality/noise/
trend → None` ~20 each). The **anomaly-routed n itself swings 61→71→49** across runs.

Consequences (these travel with every number above):
1. The large "None/fallback stratum regressions" (H2 −13pp on n=137; H3 −22pp on n=167) are **largely
   routing-luck artifacts** — rows a given run's flaky router dumped to the worse fallback path — not
   config effects. (None-count: baseline 127, H2 137, H3 167.)
2. The AD objective is **noise-limited**: ±~20-row stratum-membership churn between runs means only a
   *large* true AD effect could clear CI-lo>0. A small real gain would be **undetectable** here, so each
   AD null is "≤ the noise floor," not provably exactly zero.
3. This is a result *for* the program's thesis: the decision-tree's **per-branch mechanism localization
   is clean on gemini (deterministic temp-0 routing) but degrades on a weak backbone** whose router is
   40% irreproducible. The architecture's precision is **conditional on backbone capability** — the
   value that survives is the **deterministic computation**, which does not depend on the flaky router.

## Verdict — pre-registered **H0 supported: BROAD CORRECTED NULL** (confirmed by `summarize_ledger.py`)
Scored family = {H2, H3}; both **REJECT** (AD CI-lo not > 0). Holm across the family leaves H3's raw
p=0.375 → 0.750. No variant clears AD CI-lo>0 (the search-pass gate), so none reaches the MMTS held-out
confirm. Scoreboard total: 4.35M search tokens (anchor 4.25M + H1 probe ~0.5M separate). The agency+vision-router config space **cannot reliably empower gpt-4o-mini on anomaly** —
because (a) the AD vision look is a coin flip on this backbone (both directions n.s.), (b) self-
consistency/reroute are dormant (0% evidence_incomplete), (c) the propose/refine agency loop is
**empirically dormant on AD** (fired 0/20 in the probe — anomaly evidence quality is always ≥0.81 ≫ the
0.65 gate, so the trigger never fires; the AD bottleneck is the LLM misreading *confident* evidence, not
the ambiguity agency would accumulate against), and (d) the search is **noise-limited by 40% router
irreproducibility**. A clean, mechanistically-explained null — the rigor guaranteed it could not
noise-mine a false win.

## What changed next
- **Feeds the "Computation, Not Agency" thesis pivot** (memory `tsmart-thesis-pivot`, 2026-06-23): this
  is the empirical close on the agency/vision levers for a weak backbone. The genuine weak-backbone lever
  remains the **deterministic tools** (Track A +13.5pp), not vision and not — on this evidence — agency.
- Suppressed: any claim the learned gate transfers cross-backbone (it overall-regresses on gpt-4o-mini).

> **Post-hoc note (public release):** the `orchestration/` autonomous-search harness
> (`score_variant.py`/`summarize_ledger.py`/`diagnose_stratum.py`, the ledger, and
> `ORCHESTRATOR_BRIEF.md`) that ran this search has been removed from the public repository. It
> was preregistered, Holm-corrected across the whole variant family, and held-out-gated (see
> `tsexam/PREREGISTRATION_anomaly_search.md`) — nothing about the search was hidden or
> cherry-picked, and every variant it tried is still reported in the table above. It is no longer
> shipped as reusable scaffolding for future automated searches. The per-run `metrics.json` results
> it produced (below) remain committed as the evidence for this entry.

## Artifacts
- **Committed** (the analysis is fully preserved): per-run `metrics.json` under
  `tsexam/outputs/{as_baseline_4omini, as_var_H2_no_vision_ad, as_var_H3_learned_gate, as_h1probe}/`.
  Every per-branch stat (Δ, CI, McNemar p, b/c) the now-removed ledger recorded is reproduced in the
  results table above.
- **Not committed, regenerable by script**: the raw per-row `results.json`/`failures.json` are gitignored
  (`.gitignore` §"Ad-hoc eval run outputs" — metrics-only snapshot policy). Recreate the anchor with
  `run_eval.py --config baseline --provider openai --n-per-cat 120 --seed 42 --tag as_baseline_4omini`;
  the per-variant `--config-json` scoring itself is no longer shipped (see the post-hoc note above). NB
  gpt-4o-mini temp-0 is ±~1.3pp, not bit-exact (see the router non-reproducibility caveat above) —
  regenerated numbers land within noise, not identical.
- Pre-reg: `tsexam/PREREGISTRATION_anomaly_search.md`.
