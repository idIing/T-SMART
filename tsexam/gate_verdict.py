"""Pre-registered H1/H2 verdict for the learned should-I-look gate (#1).

Consolidates the frozen decision rule from mmts_bench/PREREGISTRATION_learned_gate.md into
one auditable output on a paired additive-vs-learned run. Reuses paired_diff.py's exact
stats (bootstrap 95% CI + exact McNemar) — this script only encodes the endpoints, it does
not redefine the statistics.

Pre-registered endpoints:
  H1 (primary): paired ΔOA (learned − additive) CI lower bound ≥ −0.005 AND look-rate
                strictly lower. PASS iff both hold.
  H2 (secondary/guardrail): per-branch_used paired Δ non-negative on every branch — read as
                "no SIGNIFICANT regression on any branch" (a non-significant negative is
                within noise and reported, not failed).

Usage:
    python research/gate_verdict.py outputs/heldout_additive outputs/heldout_learned
    (arg order: additive[base]  learned[treat])
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_diff import _load, stats  # noqa: E402

CI_FLOOR = -0.005  # pre-registered non-inferiority margin (−0.5pp)


def _fired(r):
    return bool(r.get("vision_triggered") or r.get("vision_used") or r.get("vision_letter"))


def main():
    add_dir, learn_dir = sys.argv[1], sys.argv[2]
    A, L = _load(add_dir), _load(learn_dir)
    ids = sorted(set(A) & set(L))
    n = len(ids)

    base = [int(bool(A[i].get("correct"))) for i in ids]
    treat = [int(bool(L[i].get("correct"))) for i in ids]
    ov = stats(base, treat)

    a_looks = sum(_fired(A[i]) for i in ids)
    l_looks = sum(_fired(L[i]) for i in ids)
    look_delta = (l_looks - a_looks) / n if n else float("nan")

    # routing-drift floor (temp-0 determinism check)
    drift = sum(str(A[i].get("branch_used")) != str(L[i].get("branch_used")) for i in ids) / n

    # per-branch (H2)
    groups = {}
    for i in ids:
        k = str(A[i].get("branch_used"))
        groups.setdefault(k, [[], []])
        groups[k][0].append(int(bool(A[i].get("correct"))))
        groups[k][1].append(int(bool(L[i].get("correct"))))
    per_branch = sorted(
        ((k, stats(v[0], v[1])) for k, v in groups.items()),
        key=lambda kv: kv[1]["n"], reverse=True,
    )

    print("=" * 86)
    print(f"LEARNED-GATE VERDICT (held-out)   additive={add_dir}  learned={learn_dir}")
    print("=" * 86)
    print(f"paired rows: {n}   temp-0 routing drift: {drift:.2%}")
    print(f"\nOA: additive {ov['ab']:.4f}  ->  learned {ov['at']:.4f}   "
          f"ΔOA {ov['d']:+.4f}  CI [{ov['lo']:+.4f},{ov['hi']:+.4f}]  "
          f"McNemar p={ov['p']:.4f}  (b/c {ov['b']}/{ov['c']})")
    print(f"look-rate: additive {a_looks}/{n} ({a_looks/n:.1%})  ->  "
          f"learned {l_looks}/{n} ({l_looks/n:.1%})   Δ {look_delta:+.1%} "
          f"({a_looks - l_looks} fewer looks)")

    print("\n-- per branch_used (H2 guardrail) " + "-" * 52)
    print(f"{'branch':14s}{'n':>5}{'add':>8}{'learn':>8}{'Δ':>8}{'  95% CI':>20}{'p':>9}")
    sig_regressions = []
    for k, s in per_branch:
        flag = ""
        if s["d"] < 0 and s["p"] < 0.05:
            flag = "  <-- SIG REGRESSION"
            sig_regressions.append((k, s))
        elif s["d"] < 0:
            flag = "  (neg, n.s.)"
        print(f"{k:14s}{s['n']:>5}{s['ab']:>8.3f}{s['at']:>8.3f}{s['d']:>+8.3f}"
              f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{s['p']:>9.3f}{flag}")

    # ---- verdicts ----
    h1_oa = ov["lo"] >= CI_FLOOR
    h1_look = l_looks < a_looks
    h1 = h1_oa and h1_look
    h2 = len(sig_regressions) == 0

    print("\n" + "=" * 86)
    print("PRE-REGISTERED VERDICT")
    print("-" * 86)
    print(f"H1 (primary): ΔOA CI lower bound ≥ {CI_FLOOR:+.3f}?  {ov['lo']:+.4f} -> "
          f"{'YES' if h1_oa else 'NO'}    |    look-rate strictly lower?  "
          f"{'YES' if h1_look else 'NO'}  ({a_looks}->{l_looks})")
    print(f"     => H1 {'PASS' if h1 else 'FAIL'}  "
          f"(non-inferior OA at a strictly lower look-rate)")
    print(f"H2 (guardrail): no significant per-branch regression?  "
          f"{'YES -> PASS' if h2 else 'NO -> FAIL: ' + ', '.join(k for k,_ in sig_regressions)}")
    print("=" * 86)


if __name__ == "__main__":
    main()
