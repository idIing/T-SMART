"""Gate efficiency report (learned vs additive) — pre-registered endpoint #4.

Companion to paired_diff.py for the learned should-I-look gate (#1). On a paired
additive-vs-learned run it decomposes the suppressed looks (rows the additive gate looked
at but the learned gate skipped) into:

  * harmful_suppression  — additive was right, learned wrong  → we LOST a beneficial look
  * helpful_suppression  — additive was wrong, learned right  → we AVOIDED a harmful look
  * neutral_suppression  — same correctness                  → needless look avoided (free)

"needless looks avoided" = neutral + helpful (looks removed at no accuracy cost or better).
Because the learned arm IS the no-look counterfactual on exactly these rows, this is a clean
per-row causal read of each suppressed look — no separate vision_off arm needed.

Usage:
    python research/gate_efficiency.py outputs/heldout_additive outputs/heldout_learned
    (arg order: additive[base]  learned[treat])
"""
import json
import sys
from collections import defaultdict


def _load(tag_dir):
    rows = json.load(open(f"{tag_dir}/results.json"))
    return {str(r["id"]): r for r in rows if r and r.get("id") is not None}


def _fired(r):
    return bool(r.get("vision_triggered") or r.get("vision_used") or r.get("vision_letter"))


def _correct(r):
    return bool(r.get("correct"))


def main():
    add_dir, learn_dir = sys.argv[1], sys.argv[2]
    A, L = _load(add_dir), _load(learn_dir)
    ids = sorted(set(A) & set(L))

    a_looks = sum(_fired(A[i]) for i in ids)
    l_looks = sum(_fired(L[i]) for i in ids)
    n = len(ids)

    # suppressed looks: additive fired, learned skipped
    supp = [i for i in ids if _fired(A[i]) and not _fired(L[i])]
    added = [i for i in ids if not _fired(A[i]) and _fired(L[i])]  # should be 0 by design

    cat = defaultdict(int)
    by_branch = defaultdict(lambda: defaultdict(int))
    for i in supp:
        ac, lc = _correct(A[i]), _correct(L[i])
        if ac and not lc:
            k = "harmful_suppression"
        elif lc and not ac:
            k = "helpful_suppression"
        else:
            k = "neutral_suppression"
        cat[k] += 1
        by_branch[str(A[i].get("branch_used"))][k] += 1

    needless = cat["neutral_suppression"] + cat["helpful_suppression"]
    net_acc_delta_from_supp = cat["helpful_suppression"] - cat["harmful_suppression"]

    print("=" * 80)
    print(f"GATE EFFICIENCY  additive={add_dir}  learned={learn_dir}")
    print("=" * 80)
    print(f"paired rows: {n}")
    print(f"look-rate:   additive {a_looks}/{n} ({a_looks/n:.1%})  ->  "
          f"learned {l_looks}/{n} ({l_looks/n:.1%})   "
          f"Δ {-(a_looks-l_looks)} looks ({(l_looks-a_looks)/n:+.1%})")
    if added:
        print(f"  !! WARNING: learned ADDED {len(added)} looks additive skipped "
              f"(design says it should only remove) — investigate ids {added[:8]}")
    print(f"\nsuppressed looks (additive fired ∧ learned skipped): {len(supp)}")
    print(f"  harmful_suppression  (lost a good look) : {cat['harmful_suppression']}")
    print(f"  helpful_suppression  (avoided bad look) : {cat['helpful_suppression']}")
    print(f"  neutral_suppression  (free efficiency)  : {cat['neutral_suppression']}")
    print(f"  --> needless looks avoided (neutral+helpful): {needless}/{len(supp) or 1} "
          f"({needless/(len(supp) or 1):.1%} of suppressed)")
    print(f"  --> net OA effect of all suppressions: {net_acc_delta_from_supp:+d} rows "
          f"({net_acc_delta_from_supp/n:+.3%} OA)")

    print("\n-- suppressions by branch (mechanism view) " + "-" * 36)
    print(f"{'branch':14s}{'suppressed':>11}{'harmful':>9}{'helpful':>9}{'neutral':>9}")
    for b in sorted(by_branch, key=lambda b: -sum(by_branch[b].values())):
        d = by_branch[b]
        tot = sum(d.values())
        print(f"{b:14s}{tot:>11}{d['harmful_suppression']:>9}"
              f"{d['helpful_suppression']:>9}{d['neutral_suppression']:>9}")
    print("=" * 80)


if __name__ == "__main__":
    main()
