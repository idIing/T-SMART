#!/usr/bin/env python3
"""Phase-4 agency analysis: gate fire-rate, action-executed rate, and the
action-executed-row discordance (corrected vs broken) for the agentic_tsmart arm,
paired against a fresh baseline. Pure read-only over the result CSVs.

Usage:
  python3 scripts/phase4_agency.py --baseline <baseline.csv> --treatment <agentic.csv> [--label Base]
"""
import argparse
import csv
import json


def _truth(v):
    return str(v).strip().lower() == "true"


def _read(path):
    with open(path, newline="") as f:
        return {str(r["sample_id"]): r for r in csv.DictReader(f)}


def _qscore(r):
    v = r.get("quality_score")
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _executed(r):
    v = r.get("propose_executed")
    if v in (None, "", "None", "null", "[]"):
        return []
    try:
        parsed = json.loads(v)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base = _read(args.baseline)
    treat = _read(args.treatment)
    ids = sorted(set(base) & set(treat))

    n = len(ids)
    q_pop = gated = fired = exec_rows = 0
    corrected = broken = unchanged_exec = 0
    exec_action_counts = {}
    for sid in ids:
        t = treat[sid]
        b = base[sid]
        q = _qscore(t)
        if q is not None:
            q_pop += 1
            if q < 0.55:
                gated += 1
        if _truth(t.get("propose_fired")):
            fired += 1
        ex = _executed(t)
        if ex:
            exec_rows += 1
            for a in ex:
                exec_action_counts[a] = exec_action_counts.get(a, 0) + 1
            bc, tc = _truth(b.get("correct")), _truth(t.get("correct"))
            if not bc and tc:
                corrected += 1
            elif bc and not tc:
                broken += 1
            else:
                unchanged_exec += 1

    print(f"\n=== AGENCY [{args.label}] ===  paired n={n}")
    print(f"  quality_score populated : {q_pop}/{n}")
    print(f"  GATE fired (q<0.55)     : {gated}/{n}  ({100*gated/n:.1f}%)" if n else "  no rows")
    print(f"  PROPOSE fired           : {fired}/{n}  ({100*fired/n:.1f}%)" if n else "")
    print(f"  ACTION-EXECUTED rows    : {exec_rows}/{n}  ({100*exec_rows/n:.1f}%)" if n else "")
    if exec_rows:
        print(f"  -- among ACTION-EXECUTED rows ({exec_rows}):")
        print(f"       corrected (base wrong -> treat right): {corrected}")
        print(f"       broken    (base right -> treat wrong): {broken}")
        print(f"       unchanged                            : {unchanged_exec}")
        print(f"       net (corrected - broken)             : {corrected - broken}")
        print(f"       executed-action histogram            : {exec_action_counts}")
    else:
        print("  -- no actions executed (agency dormant on this surface)")


if __name__ == "__main__":
    main()
