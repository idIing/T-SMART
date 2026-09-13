# Third-party material

The MIT license in [`LICENSE`](LICENSE) covers the code and documentation authored
for this project. No third-party material is redistributed here; this file records
what the project depends on and where each piece comes from.

## The manuscript

The companion ICTAI 2026 paper is published through IEEE and is not included in
this repository — neither its LaTeX source, its figures, nor the typeset PDF.
Obtain it from IEEE Xplore. What the paper reports about this system can be
rebuilt here from the committed run artifacts; see [`RESULTS.md`](RESULTS.md).

## Benchmarks

Neither benchmark's data is redistributed in this repository; both are fetched by
the harnesses at run time from their upstream sources, under their own licenses.

- **TimeSeriesExam** — loaded via HuggingFace `datasets` in `tsexam/run_eval.py`.
- **MMTS-Bench** — read from a local checkout pointed at by `MMTS_BENCH_PATH`
  (see [`mmts_bench/README.md`](mmts_bench/README.md)).

Cross-system comparison rows in the paper's tables (ChatTS, TS-Agent) are numbers
quoted from the cited publications, not reruns.

## Cited papers

`sources/` records how each of the paper's citations was verified — manifests,
extracted abstracts, and verification notes. Neither the PDFs nor their extracted
full text are redistributed here; what remains are the short verbatim excerpts a
verification table needs to be checkable. `sources/citation_verification.md` and
the fetch manifests identify each source so it can be retrieved from its
publisher or arXiv.
