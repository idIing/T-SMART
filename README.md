# T-SMART

Mechanism-Level Attribution for Tool-Augmented Time-Series Question Answering

T-SMART answers questions about time series by delegating every numerical computation
to deterministic statistical tools and restricting a frozen language model to two roles:
selecting the computation, and reading the computed evidence into an answer. A structured
visual sensor is invoked on demand when the numerical evidence is judged unreliable.

The project is a measurement study as much as a system. We use T-SMART to locate where the
value in a tool-augmented pipeline actually sits, through pre-registered, paired,
mechanism-local experiments. The headline findings:

- Deterministic tools own computation. On free-response numerical questions a closed-form
  numeric head beats the same model computing from the raw series by a wide margin, and is
  at or near ceiling on purely tool-computed quantities.
- The language model is load-bearing only as a translator, for open paraphrase that a
  deterministic grammar cannot resolve, and nothing more.
- Conditional structured vision helps morphological questions and reduces visual
  sycophancy, at a small accuracy cost relative to feeding raw pixels.
- Bounded self-correction, evaluated under the same paired protocol, is inert on the
  closed-form benchmarks considered.

We make no state-of-the-art claim. Cross-system comparisons against prior agentic systems
are backbone-confounded; the comparison we trust is the matched-backbone paired difference.
The companion paper is *T-SMART: Mechanism-Level Attribution for Tool-Augmented Time-Series
Question Answering* (ICTAI 2026), published through IEEE and not redistributed here. This
repository is the code and the evidence behind its numbers: every reported table rebuilds
from the committed run artifacts via `make_paper_tables.py`.

## Repository layout

- `tsqa/` — the pipeline engine, an installable package (`pip install -e tsqa`). The
  architecture lives here; engine unit tests are under `tsqa/tests/`.
- `tsexam/` — the home-benchmark harness (TimeSeriesExam). Named configs are validated
  here first via `tsexam/run_eval.py`, with the paired-diff and analysis scripts alongside.
- `mmts_bench/` — the out-of-distribution generalization harness. It imports the `tsqa`
  package and runs frozen configs on MMTS-Bench. See
  [`mmts_bench/GENERALIZATION_REPORT.md`](mmts_bench/GENERALIZATION_REPORT.md).
- `research_journal/` — the documented reasoning behind the architecture: the falsifiable
  thesis, the architectural decision records, the staged plan, and one log entry per
  result. Start at [`research_journal/README.md`](research_journal/README.md).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # harness + analysis dependencies
pip install -e tsqa               # the tsqa engine package (editable), from the repo root
```

`GEMINI_API_KEY` must be present in a `.env` at the repo root (git-ignored) or in the
environment. The engine's own runtime dependencies are declared in `tsqa/setup.py`;
alternate backbones are extras (`pip install -e 'tsqa[openai]'`, `'tsqa[qwen]'`). The MMTS
harness imports the same `tsqa` package.

The backbone for all primary results is `gemini-3.1-flash-lite` at temperature 0, with one
model instance acting as router, vision sensor, and answer reader.

### Benchmark data

Neither benchmark is redistributed here.

- **TimeSeriesExam** is pulled from HuggingFace by `tsexam/run_eval.py` on first use.
- **MMTS-Bench** must be downloaded separately and placed at `mmts_bench/MMTS-BENCH/`, or
  pointed at with `MMTS_BENCH_PATH`. See [`mmts_bench/README.md`](mmts_bench/README.md).

Per-row eval logs (`tsexam/outputs/*/results.json`) are git-ignored because they are large
and regenerable; the committed `metrics.json` summaries are what the tables are built from.

## Architecture

```
run_pipeline(row, llm_client, **config)        # tsqa/eval/runner.py
  router (LLM) -> JSON route {branch, scope, subtype, series_type}
  deterministic schema check on the options
    multiple-choice path:
      branch (deterministic tools) -> evidence
      verifier -> flags + additive trust gate
      conditional structured vision sensor (on a trip)
      answer reader -> a letter
    free-response path:
      numeric head -> validated operation plan over a closed registry
      deterministic evaluation -> a value   (exits before the gate and reader)
```

The pipeline is single-pass; bounded self-correction loops exist in the codebase but are
disabled by default and are reported as null results. The structured-vision invariant holds
that the visual sensor returns a JSON description of the rendering, and the raw pixels never
reach the reasoner.

## Running an evaluation

TimeSeriesExam (home benchmark, run from the repo root):

```bash
python3 tsexam/run_eval.py --config baseline --full --tag baseline
python3 tsexam/paired_diff.py            # paired McNemar on two runs
pytest tsqa/tests/test_runner_v2.py
```

MMTS-Bench (generalization, from `mmts_bench/`):

```bash
cd mmts_bench
python3 scripts/run_mmts_baseline.py --subset Base --config baseline --max-workers 16
python3 scripts/diff_configs.py --baseline <run> --treatment <run>
pytest tests/test_tools.py
```

Configurations are named dictionaries of pipeline arguments, not prompt edits. The
canonical system is `baseline`. Every change is evaluated as a pre-registered, paired,
mechanism-located comparison: two configurations over the same rows, reporting a paired
difference with a bootstrap confidence interval and an exact McNemar test, stratified by
the pre-routing branch and corrected for multiplicity.

## Method and provenance

No result enters the project without a `research_journal/log/` entry. The journal is the
binding record of what was run, what was found, and the honesty caveats that travel with
each number.

Regenerate the paper's tables from the committed run logs with:

```bash
python3 make_paper_tables.py --out RESULTS.md
```

## Citation

```bibtex
@inproceedings{tsmart2026,
  title     = {T-SMART: Mechanism-Level Attribution for Tool-Augmented
               Time-Series Question Answering},
  booktitle = {Proceedings of the IEEE International Conference on Tools with
               Artificial Intelligence (ICTAI)},
  year      = {2026}
}
```

<!-- TODO(camera-ready): add author list, pages, publisher, and DOI once assigned. -->

## License

MIT — see [`LICENSE`](LICENSE). Redistributed third-party material (the IEEEtran LaTeX
class) and the licensing of the benchmarks are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
