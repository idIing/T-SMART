# TimeSeriesExam run outputs — what each tag is

Each directory is one evaluation run, named by its `--tag`, holding a
`metrics.json` summary. The per-row logs (`results.json`, `failures.json`) are
git-ignored: they are large and regenerable from the committed code.

**A run tag is not a config name.** `run_final.sh` invokes config `baseline`
under tag `baseline_final`. The short tags are earlier pilots that survive for
provenance, so reading the wrong one silently gives the wrong number.

## The paper's runs

These are the 746-row `gemini-3.1-flash-lite` runs behind the reported results,
all produced by `run_final.sh` at commit `d37128f31aaf`:

| Tag | Config | Macro OA | Role |
|---|---|--:|---|
| `baseline_final` | `baseline` | 65.0 | Table 1 baseline row |
| `nu_ad_fix_final` | `nu_ad_fix` | 67.5 | Table 1 `vision_override` row |
| `vision_off_final` | `vision_off` | 62.6 | vision ablation |
| `raw_pixel_final` | `raw_pixel_vision` | 67.5 | raw-pixel counterfactual |
| `learned_gate_final` | `learned_gate` | 64.5 | learned vs additive gate |
| `react_s1_final` | `react_stage1_evidence_retry` | 64.4 | agency null, reroute-once |
| `react_s2_final` | `react_stage2_refine` | 64.6 | agency null, refine |
| `react_s3_final` | `react_stage3_actions` | 64.5 | agency null, propose |

## Not the paper's runs

| Tag(s) | n | What it is |
|---|--:|---|
| `baseline`, `nu_ad_fix` | 150 | Early pilots. `baseline` scores 62.0, **not** the paper's 65.0. |
| `*_full` | 746 | Earlier full runs on a previous checkout. Still the inputs to `paired_tsexam_full.txt` and `paired_react_s3.txt`, whose headers therefore show a stale `research/outputs/...` path prefix. |
| `smoke*` | 5–20 | Wiring checks, not measurements. |
| `heldout_additive`, `heldout_learned` | 746 | Held-out replication for the learned-gate verdict. |
| `tv_raw_pixel`, `tv_vision_off` | 746 | Track-V arms feeding `structured_vs_pixel.txt`. |
| `c1_gpt4o_*` | 746 | Fixed-backbone (gpt-4o-mini) secondary analysis. |
| `as_*` | 150–504 | Anomaly search on gpt-4o-mini, per `../PREREGISTRATION_anomaly_search.md`. |

## Analysis outputs

| File | Produced by | Compares |
|---|---|---|
| `paired_baseline_nuadfix.txt` | `paired_diff.py` | `baseline_final` vs `nu_ad_fix_final` |
| `paired_react_s3.txt` | `paired_diff.py` | `baseline_full` vs `react_s3_full` |
| `paired_tsexam_full.txt` | `paired_diff.py` | `baseline_full` vs `nu_ad_fix_full` |
| `structured_vs_pixel.txt` | `structured_vs_pixel.py` | structured vs raw-pixel on the 568 vision-fired rows |
| `gate_verdict.txt` | `gate_verdict.py` | additive vs learned gate, held out |

In the paired tables, `b` is control-only-correct and `c` is
treatment-only-correct (`paired_diff.py:66-67`), so the `b/c` column is
`n01/n10` — the reverse of the `n10/n01` ordering used in the paper's prose.
