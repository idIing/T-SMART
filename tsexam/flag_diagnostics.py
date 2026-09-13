"""
Verifier-flag precision / recall report (Stage-1 constraint #2).
================================================================

The Stage-1 self-correction loop is gated by a verifier flag, and the loop's
McNemar gate tests verifier+loop *jointly*. Before promoting any branch we must
know the flag's own diagnostic value against per-row correctness, because a flag
that fires on already-correct rows can only ever make them wrong (the
self-correction "blind spot", 03_related_work.md §B). This script reports, per
verifier flag, how well "flag fired" predicts "row is wrong".

Per flag, against per-row correctness (a row is *wrong* iff correct == False):
    fire_rate  = P(flag fired)                       = fired / N
    precision  = P(row wrong | flag fired)           = (fired & wrong) / fired
    recall     = P(flag fired | row wrong)           = (fired & wrong) / wrong
    lift       = precision / base_wrong_rate         (>1 ⇒ better than chance)
plus the raw 2x2 counts. A flag with HIGH fire-rate but precision at or below the
base wrong-rate is firing mostly on correct rows — exactly the regime where a
loop gated on it is net-negative.

Reads a run dir's results.json (e.g. outputs/baseline_full) — the same per-row
`flags` + `correct` fields slim_row already serializes. Rows with correct is None
(numeric-head abstentions / errors) are excluded from the correctness analysis.

Usage:
    python research/flag_diagnostics.py outputs/baseline_full
    python research/flag_diagnostics.py outputs/baseline_full --json
"""
import argparse
import json
import os
import sys

# The verifier flag universe (verifier/checks.py:_EXPLICIT_FLAGS). Reported in a
# fixed order so two runs line up column-for-column; evidence_incomplete first
# because it is the Stage-1 loop / self_consistency trigger.
FLAGS = [
    "evidence_incomplete",
    "low_r2",
    "adf_kpss_disagree",
    "granger_not_significant",
    "weak_correlation",
    "arch_effects",
    "high_volatility",
    "fft_unreliable",
]


def _load_rows(run_dir):
    path = os.path.join(run_dir, "results.json")
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if r and "error" not in r]


def _row_flags(r):
    """Per-row flag set: prefer the top-level `flags`, fall back to evidence.flags."""
    flags = r.get("flags")
    if flags is None:
        ev = r.get("evidence") or {}
        flags = ev.get("flags")
    return set(flags or [])


def compute_flag_stats(rows):
    """Return (per_flag_stats, meta). A row counts toward the correctness analysis
    only when `correct` is a real bool (numeric abstentions/errors are excluded)."""
    scored = [r for r in rows if isinstance(r.get("correct"), bool)]
    n = len(scored)
    n_wrong = sum(1 for r in scored if not r["correct"])
    base_wrong = (n_wrong / n) if n else 0.0

    stats = {}
    for flag in FLAGS:
        fired = [r for r in scored if flag in _row_flags(r)]
        n_fired = len(fired)
        n_fired_wrong = sum(1 for r in fired if not r["correct"])
        n_fired_right = n_fired - n_fired_wrong
        precision = (n_fired_wrong / n_fired) if n_fired else None
        recall = (n_fired_wrong / n_wrong) if n_wrong else None
        fire_rate = (n_fired / n) if n else 0.0
        lift = (precision / base_wrong) if (precision is not None and base_wrong > 0) else None
        stats[flag] = {
            "fired": n_fired,
            "fired_wrong": n_fired_wrong,   # true positives (flag ⇒ wrong)
            "fired_right": n_fired_right,   # false positives (flag fired on a correct row)
            "fire_rate": fire_rate,
            "precision": precision,
            "recall": recall,
            "lift": lift,
        }
    meta = {"n_scored": n, "n_wrong": n_wrong, "base_wrong_rate": base_wrong}
    return stats, meta


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def print_report(stats, meta, run_dir):
    print("=" * 92)
    print(f"VERIFIER-FLAG PRECISION/RECALL  {run_dir}")
    print("=" * 92)
    print(f"scored rows: {meta['n_scored']}   wrong: {meta['n_wrong']}   "
          f"base wrong-rate: {meta['base_wrong_rate']:.3f}")
    print("  precision = P(wrong | fired)   recall = P(fired | wrong)   "
          "lift = precision / base-wrong\n")
    hdr = (f"{'flag':<24}{'fire_rt':>9}{'fired':>7}{'wrong':>7}{'right':>7}"
           f"{'prec':>8}{'recall':>8}{'lift':>8}")
    print(hdr)
    print("-" * 92)
    for flag in FLAGS:
        s = stats[flag]
        note = ""
        # Flag the net-negative regime: it fires, but no better than chance at
        # finding wrong rows ⇒ a loop gated on it mostly perturbs correct rows.
        if s["fired"] > 0 and s["lift"] is not None and s["lift"] <= 1.0:
            note = "  <- fires on mostly-correct rows"
        if s["fired"] == 0:
            note = "  <- never fired"
        print(f"{flag:<24}{s['fire_rate']:>9.3f}{s['fired']:>7}{s['fired_wrong']:>7}"
              f"{s['fired_right']:>7}{_fmt(s['precision']):>8}{_fmt(s['recall']):>8}"
              f"{_fmt(s['lift']):>8}{note}")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run output dir containing results.json "
                                    "(e.g. outputs/baseline_full)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of the table")
    args = ap.parse_args()

    rows = _load_rows(args.run_dir)
    stats, meta = compute_flag_stats(rows)
    if args.json:
        print(json.dumps({"meta": meta, "flags": stats}, indent=2))
    else:
        print_report(stats, meta, args.run_dir)


if __name__ == "__main__":
    main()
