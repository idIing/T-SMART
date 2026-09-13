#!/usr/bin/env python3
"""Regenerate the paper's result tables from the committed run logs.

Read-only and scoring-free: it reuses the per-run artifacts the eval scripts
already wrote (TSExam ``metrics.json``; MMTS run-manifests + detail CSVs) and
formats them as the paper's Table 1, Table 2, and the MMTS Base per-category
breakdown, each with a provenance line lifted from the run itself. The paired
mechanism statistics (McNemar, CIs) come from the analyzer scripts, not here.

Run from the repo root:
    python3 make_paper_tables.py            # print to stdout
    python3 make_paper_tables.py --out RESULTS.md
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
TSX_OUT = os.path.join(REPO, "tsexam", "outputs")
MMTS_OUT = os.path.join(REPO, "mmts_bench", "outputs", "final")
CATEGORIES = ["PR", "NU", "AD", "SA", "CA"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _latest(pattern):
    files = [f for f in sorted(glob.glob(pattern)) if "checkpoint" not in f]
    return files[-1] if files else None


def _prov_line(meta):
    """One-line provenance from a metrics.json / manifest dict."""
    p = (meta or {}).get("provenance", {}) if meta else {}
    commit = (p.get("git_commit") or "?")[:12]
    model = (meta or {}).get("model", "?")
    ts = p.get("timestamp_utc") or (meta or {}).get("timestamp", "?")
    return f"_commit `{commit}` · model `{model}` · {ts}_"


# --------------------------------------------------------------------------- #
def table1():
    out = [
        "**Table 1. TimeSeriesExam, per-category accuracy (%). OA is the category mean.**",
        "",
        "| Model | Setting | OA | PR | NU | AD | SA | CA |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
        "| TS-Agent | agentic, gpt-4o-mini | 60.2 | 71 | 61 | 57 | 57 | 55 |",
    ]
    prov = None
    for tag, label in [("baseline_final", "`baseline`"), ("nu_ad_fix_final", "`nu_ad_fix`")]:
        path = os.path.join(TSX_OUT, tag, "metrics.json")
        if not os.path.exists(path):
            out.append(f"| **T-SMART** {label} | _run missing ({tag})_ | | | | | | |")
            continue
        m = _load(path)
        prov = prov or m
        cells = []
        for c in CATEGORIES:
            pc = m["per_category"].get(c)
            cells.append(f"{100 * pc['acc']:.1f}" if pc else "—")
        out.append(
            f"| **T-SMART** {label} | gemini-3.1 | **{100 * m['macro_oa']:.1f}** | "
            + " | ".join(cells)
            + " |"
        )
    out += ["", "*T-SMART rows: this work, gemini-3.1, n=746. " + _prov_line(prov) + "*"]
    return out


def _mmts_detail():
    return _latest(os.path.join(MMTS_OUT, "mmts_results_All_baseline_*.csv"))


def _mmts_manifest():
    p = _latest(os.path.join(MMTS_OUT, "mmts_run_All_baseline_*.json"))
    return _load(p) if p else None


def table2():
    detail = _mmts_detail()
    out = [
        "**Table 2. MMTS-Bench, overall accuracy per subset (%). "
        "Base uses the numeric head on free-response.**",
        "",
        "| Model | Overall | Base | InWild | Match | Align |",
        "|---|--:|--:|--:|--:|--:|",
        "| ChatTS | 49 | 39 | 50 | 37 | 80 |",
        "| TS-Agent | 60 | 21 | 71 | 77 | 97 |",
    ]
    if not detail:
        out.append("| **T-SMART** `baseline` | _run missing (mmts All baseline)_ | | | | |")
        return out
    # Match run_mmts_baseline's own rule exactly: every row counts; a parse
    # error or skip counts as wrong (correct == 'true' is the only success).
    by_sub = defaultdict(lambda: [0, 0])  # subset -> [correct, total]
    tot = [0, 0]
    with open(detail) as f:
        for r in csv.DictReader(f):
            sub = r.get("subset", "?")
            ok = str(r.get("correct")).strip().lower() == "true"
            by_sub[sub][1] += 1
            tot[1] += 1
            if ok:
                by_sub[sub][0] += 1
                tot[0] += 1

    def pct(k, n):
        return f"{100 * k / n:.1f}" if n else "—"

    order = ["Base", "InWild", "Match", "Align"]
    cells = [pct(*by_sub[s]) for s in order]
    out.append(
        f"| **T-SMART** `baseline` | **{pct(*tot)}** | " + " | ".join(cells) + " |"
    )
    # denominators spelled out so the Base figure's basis is unambiguous
    ns = "  ".join(f"{s} n={by_sub[s][1]}" for s in order)
    out += ["", f"*Per-subset denominators: {ns}  (Overall n={tot[1]}, micro-average). "
            + _prov_line(_mmts_manifest()) + "*"]
    return out


def mmts_base_by_category():
    detail = _mmts_detail()
    out = ["**MMTS Base, per-category accuracy (%) — the §4.3 mechanism view.**", ""]
    if not detail:
        return out + ["_run missing (mmts All baseline)_"]
    by_cat = defaultdict(lambda: [0, 0])
    with open(detail) as f:
        for r in csv.DictReader(f):
            if r.get("subset") != "Base":
                continue
            cat = r.get("category", "?")
            ok = str(r.get("correct")).strip().lower() == "true"
            by_cat[cat][1] += 1
            if ok:
                by_cat[cat][0] += 1
    out += ["| Category | Acc | n |", "|---|--:|--:|"]
    for cat in sorted(by_cat):
        k, n = by_cat[cat]
        out.append(f"| {cat} | {100 * k / n:.1f} | {n} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    args = ap.parse_args()

    blocks = [
        "# T-SMART — regenerated result tables",
        "",
        "Generated by `make_paper_tables.py` from the committed run logs. "
        "Cross-system rows are reproduced from source papers; T-SMART rows are this work.",
        "",
        "\n".join(table1()),
        "",
        "\n".join(table2()),
        "",
        "\n".join(mmts_base_by_category()),
        "",
    ]
    text = "\n".join(blocks)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
