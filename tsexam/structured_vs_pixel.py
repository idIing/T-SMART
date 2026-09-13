#!/usr/bin/env python3
# research/structured_vs_pixel.py
"""
Structured (VLM->JSON) vs Raw-Pixel vision — the Track-V vision-science study.
=============================================================================
MEASUREMENT ONLY. The raw_pixel arm deliberately VIOLATES the Structured-Vision
Invariant (SVI) behind a switch to measure the multimodal ceiling vs the
structured JSON sensor; it is NEVER promoted/shipped. See
PREREGISTRATION_structured_vs_pixel.md.

Three TSExam arms, joined on `id`, ALL scoped to the vision-FIRED subset (the
rows where the STRUCTURED arm triggered vision — the only place the
representation differs):

  vision_off   (math anchor)   : results.json from `--config vision_off`
  structured   (SVI default)   : results.json from `--config baseline`  (baseline_full)
  raw_pixel    (SVI-violating) : results.json from `--config raw_pixel_vision`

Per initial-route branch, for STRUCTURED and RAW_PIXEL each vs the vision_off
math anchor, it reports:

  net_value      = P(arm correct) - P(vision_off correct)   [accuracy of LOOKING]
  sycophancy     = among discordant rows (vision_letter != math_letter) where the
                   LOOK is WRONG (vision_letter != gold), the fraction where the
                   final pred FOLLOWS the look (pred == vision_letter)

with a bootstrap 95% CI + exact McNemar on the per-branch net_value contrast
(reusing paired_diff.boot_ci / mcnemar_p — import-safe, guarded by __main__).

Usage (from the repo root):
    python3 tsexam/structured_vs_pixel.py \
        --vision-off  tsexam/outputs/tv_vision_off \
        --structured  tsexam/outputs/baseline_full \
        --raw-pixel   tsexam/outputs/tv_raw_pixel
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tsqa"))  # engine package for tsqa import (paired_diff needs it)

from paired_diff import boot_ci, mcnemar_p  # noqa: E402  (import-safe; __main__ guarded)


def _load(tag_dir):
    """Load a TSExam results.json into {id: row}."""
    path = os.path.join(tag_dir, "results.json")
    rows = json.load(open(path))
    return {str(r["id"]): r for r in rows if r and r.get("id") is not None}


def _norm(x):
    """Normalise a letter cell to uppercase A/B/C/D or '' (None/NaN/empty)."""
    if x is None:
        return ""
    s = str(x).strip().upper()
    return "" if s in {"", "NAN", "NONE"} else s


def _fired(row):
    """A row is vision-FIRED iff vision_triggered AND vision_letter present."""
    vt = row.get("vision_triggered")
    truthy = bool(vt) and str(vt).strip().lower() not in {"false", "0", "none", ""}
    return truthy and _norm(row.get("vision_letter")) != ""


def _branch(struct_row):
    """Stable per-branch stratifier: prefer initial_branch_used (pre-reroute),
    fall back to branch_used (older runs predate initial_branch_used). Routing is
    deterministic at temp 0, so this is identical across all three arms."""
    return (
        struct_row.get("initial_branch_used")
        or struct_row.get("branch_used")
        or "None"
    )


# ---------------------------------------------------------------------------
# Per-arm metrics on the fired subset
# ---------------------------------------------------------------------------

def _net_value_arrays(ids, voff, arm):
    """Paired correctness vectors (vision_off_correct, arm_correct) over `ids`."""
    base = np.array([int(bool(voff[i].get("correct"))) for i in ids], int)
    treat = np.array([int(bool(arm[i].get("correct"))) for i in ids], int)
    return base, treat


def _sycophancy(ids, arm):
    """(n_disc_wrong, n_followed) over `ids` for one arm.

    discordant      : vision_letter != math_letter (math == this arm's letter when
                      vision is off; here we use the arm's OWN vision_letter vs the
                      vision_off math letter passed in via `arm[i]['_math']`).
    look is wrong    : vision_letter != gold
    followed         : final pred == vision_letter
    sycophancy rate  = n_followed / n_disc_wrong  (None if n_disc_wrong == 0)
    """
    n_disc_wrong = 0
    n_followed = 0
    for i in ids:
        r = arm[i]
        vl = _norm(r.get("vision_letter"))
        ml = _norm(r.get("_math"))          # math anchor letter (injected by caller)
        gold = _norm(r.get("gold"))
        pred = _norm(r.get("pred"))
        if not vl or not ml or vl == ml:
            continue                         # concordant or undefined
        if not gold or vl == gold:
            continue                         # look was RIGHT (or gold missing)
        n_disc_wrong += 1
        if pred == vl:
            n_followed += 1
    return n_disc_wrong, n_followed


def _fmt(x, w=8, p=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'--':>{w}}"
    spec = f"{'+' if signed else ''}{w}.{p}f"
    return f"{x:>{spec}}"


def _arm_branch_table(title, fired_ids, by_branch, voff, arm):
    """Print net_value (vs vision_off) + sycophancy per branch for one arm."""
    print(f"\n-- {title} " + "-" * max(0, 60 - len(title)))
    hdr = (f"{'branch':<14}{'n_fired':>8}{'P(math)':>9}{'P(arm)':>9}"
           f"{'net_val':>9}{'95% CI':>18}{'b/c':>8}{'McN p':>9}"
           f"{'syc_n':>7}{'syc':>8}")
    print(hdr)
    print("-" * len(hdr))

    def _row(label, ids):
        base, treat = _net_value_arrays(ids, voff, arm)
        n = len(ids)
        b = int(((base == 1) & (treat == 0)).sum())
        c = int(((base == 0) & (treat == 1)).sum())
        lo, hi = boot_ci(base, treat)
        p = mcnemar_p(b, c)
        nv = (treat.mean() - base.mean()) if n else float("nan")
        ci = f"[{lo:+.3f},{hi:+.3f}]" if n else "--"
        nd, nf = _sycophancy(ids, arm)
        syc = (nf / nd) if nd else float("nan")
        star = ""
        if n >= 8 and p < 0.05:
            star = " *" if nv > 0 else " X"
        return (f"{str(label)[:14]:<14}{n:>8}{_fmt(base.mean(),9):>0}"
                f"{_fmt(treat.mean(),9)}{_fmt(nv,9,signed=True)}{ci:>18}"
                f"{(str(b)+'/'+str(c)):>8}{p:>9.4f}{nd:>7}{_fmt(syc,8)}{star}")

    # per-branch (sorted by n)
    branches = sorted(by_branch.items(), key=lambda kv: len(kv[1]), reverse=True)
    for br, ids in branches:
        print(_row(br, ids))
    print("-" * len(hdr))
    print(_row("OVERALL", fired_ids))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vision-off", required=True,
                    help="TSExam tag dir for the vision_off (math anchor) run.")
    ap.add_argument("--structured", required=True,
                    help="TSExam tag dir for the structured/SVI run (baseline_full).")
    ap.add_argument("--raw-pixel", required=True,
                    help="TSExam tag dir for the raw_pixel_vision run.")
    args = ap.parse_args()

    voff = _load(args.vision_off)
    struct = _load(args.structured)
    rawpx = _load(args.raw_pixel)

    # Common ids across all three arms (routing is deterministic, so the sets
    # should coincide; intersect to be safe).
    ids = set(voff) & set(struct) & set(rawpx)

    # Fired subset = rows where the STRUCTURED arm triggered vision.
    fired_ids = sorted(i for i in ids if _fired(struct[i]))

    print("=" * 92)
    print("STRUCTURED vs RAW-PIXEL  —  TSExam vision-fired subset (Track-V; MEASUREMENT ONLY)")
    print("=" * 92)
    print(f"common ids across 3 arms          : {len(ids)}")
    print(f"vision-FIRED subset (structured)  : {len(fired_ids)}")

    if not fired_ids:
        print("\n[structured_vs_pixel] no fired rows — nothing to analyse.")
        return 0

    # Inject the math-anchor letter (vision_off pred) into each arm's row so
    # _sycophancy can compute discordance against the SAME math position.
    for i in fired_ids:
        math_letter = voff[i].get("pred")
        struct[i]["_math"] = math_letter
        rawpx[i]["_math"] = math_letter

    by_branch = defaultdict(list)
    for i in fired_ids:
        by_branch[_branch(struct[i])].append(i)

    print("\nnet_val = P(arm correct) - P(vision_off correct)  [accuracy of LOOKING]")
    print("syc     = among discordant rows where the LOOK is WRONG, fraction where "
          "pred FOLLOWS the look")
    print("* / X   = net_val significant gain / loss (McNemar p<0.05, n>=8)")

    _arm_branch_table("STRUCTURED (SVI: JSON topology) vs vision_off",
                      fired_ids, by_branch, voff, struct)
    _arm_branch_table("RAW_PIXEL (SVI-violating: raw PNG) vs vision_off",
                      fired_ids, by_branch, voff, rawpx)

    # ── Head-to-head verdict summary ────────────────────────────────────────
    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD (per branch): structured net_val/syc vs raw_pixel net_val/syc")
    print("=" * 92)
    hdr = (f"{'branch':<14}{'n':>6}"
           f"{'S_net':>9}{'P_net':>9}{'net_diff':>10}"
           f"{'S_syc':>8}{'P_syc':>8}{'syc_diff':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _summ(ids):
        b_s, t_s = _net_value_arrays(ids, voff, struct)
        b_p, t_p = _net_value_arrays(ids, voff, rawpx)
        s_net = (t_s.mean() - b_s.mean()) if len(ids) else float("nan")
        p_net = (t_p.mean() - b_p.mean()) if len(ids) else float("nan")
        nd_s, nf_s = _sycophancy(ids, struct)
        nd_p, nf_p = _sycophancy(ids, rawpx)
        s_syc = (nf_s / nd_s) if nd_s else float("nan")
        p_syc = (nf_p / nd_p) if nd_p else float("nan")
        return s_net, p_net, s_syc, p_syc

    def _hrow(label, ids):
        s_net, p_net, s_syc, p_syc = _summ(ids)
        nd = (p_net - s_net) if not (np.isnan(s_net) or np.isnan(p_net)) else float("nan")
        sd = (p_syc - s_syc) if not (np.isnan(s_syc) or np.isnan(p_syc)) else float("nan")
        return (f"{str(label)[:14]:<14}{len(ids):>6}"
                f"{_fmt(s_net,9,signed=True)}{_fmt(p_net,9,signed=True)}"
                f"{_fmt(nd,10,signed=True)}"
                f"{_fmt(s_syc,8)}{_fmt(p_syc,8)}{_fmt(sd,10,signed=True)}")

    for br, bids in sorted(by_branch.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(_hrow(br, bids))
    print("-" * len(hdr))
    print(_hrow("OVERALL", fired_ids))
    print("\nnet_diff = raw_pixel - structured (net_value); >0 ⇒ pixels add ceiling.")
    print("syc_diff = raw_pixel - structured (sycophancy); >0 ⇒ pixels more sycophantic.")
    print("\nVERDICT RULE: SVI VALIDATED if structured net_val >= raw_pixel AND "
          "structured syc <= raw_pixel.\n              SVI BOUNDED if raw_pixel net_val "
          "wins on shape/anomaly branches (a measurable multimodal tax).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
