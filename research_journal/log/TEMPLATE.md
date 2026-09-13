# log/TEMPLATE — copy to `NNN_short-slug.md`

*One entry per experiment. No result enters the program without one. Pre-register the
verdict BEFORE you see the number.*

---
id: NNN
date: YYYY-MM-DD
stage: 0 | 1 | 2 | 3
claim: C1 | C2 | C3 | C4 | C5 | none   # which thesis claim this moves (00_thesis.md); "none" = founding/audit note
status: pre-registered | running | complete | abandoned | note   # "complete" REQUIRES a verdict; founding/audit entries use "note"
evidence_level: fixed-backbone-ablation | OOD-transfer | mechanism-local-stratum | preregistered-H0 | efficiency | scaffold | n/a
claim_scope: overall | mechanism-local | per-initial-route-branch | n/a
overall_significance: "<e.g. p=0.46 (tie)>" | n/a
prereg: path/to/PREREGISTRATION_*.md
commit: <short hash of the code that produced this>
artifact_present: yes | no   # do the cited CSVs/reports actually exist on disk?
required_caveats: "<honesty caveats that MUST travel with this number — stratum size, backbone confound, …>"
---

## Hypothesis (pre-registered)
- **H1:** … (what we'd believe if it works)
- **H0:** … (the pre-declared null / fallback — a genuine result, not a failure)
- **Decision rule:** e.g. "promote initial-route branch *b* iff per-branch McNemar
  (stratified by **initial route**, not final `branch_used`) CI lower bound ≥ 0 after
  Holm/BH, and no frozen branch regresses."

## Setup
- Configs: baseline = …, treatment = …
- Dataset / rows: benchmark, subset, n, filter (e.g. `--mcq-only`)
- Backbone(s): …

## Result
Stratify by **initial route** (pre-treatment), not final `branch_used`.
| stratum (initial route) | n | baseline | treatment | Δ (pp) | 95% CI | McNemar p (post Holm/BH) |
|---|---|---|---|---|---|---|
| overall | | | | | | |
| branch=… | | | | | | |

**Branch-migration matrix** (only if the treatment can re-route/loop; else "n/a — no reroute"):
| initial → final | count | of which corrected | of which broken |
|---|---|---|---|
| … → … | | | |

- Look-rate / loop-rate / head-fire-rate: …
- Discordant pairs (cost made explicit): +X corrected, −Y broken.
- Verifier-flag precision/recall (if a flag gated the action): …

## Verdict (which pre-registered branch fired)
…

## What changed next
- Promoted / suppressed: …
- New open question → next entry: …

## Artifacts
- CSVs: …
- Report: …
