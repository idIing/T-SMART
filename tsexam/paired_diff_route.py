"""
Track-A paired diff, stratified by the TREATMENT arm's initial route.

The canonical research/paired_diff.py keys its mechanism-view strata off the
BASELINE arm's branch (correct when both arms route through the tree, e.g.
baseline vs nu_ad_fix). The Track-A de-confound pairs the NO-architecture control
(`llm_only`, whose initial_branch_used is the constant "llm_only") against the
deterministic-tools arm (`vision_off`), so the informative mechanism stratifier is
the TOOLS arm's router branch — that is what this script reports. It reuses
paired_diff's exact statistics (bootstrap CI + exact McNemar + Holm) verbatim, so
the numbers are directly comparable; only the strata key changes.

Usage:
    python research/paired_diff_route.py outputs/<llm_only_run> outputs/<tools_run>
"""
import os
import sys

# Make the sibling paired_diff importable regardless of the caller's cwd (the
# canonical paired_diff.py is run the same way — both live in research/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paired_diff import _load, stats, holm_adjust, line  # noqa: E402,F401


def main():
    base_dir, treat_dir = sys.argv[1], sys.argv[2]
    B, T = _load(base_dir), _load(treat_dir)
    ids = sorted(set(B) & set(T))
    base = [int(bool(B[i].get("correct"))) for i in ids]
    treat = [int(bool(T[i].get("correct"))) for i in ids]
    overall = stats(base, treat)

    def _t_route(i):
        return str(T[i].get("initial_branch_used") or T[i].get("branch_used") or "None")

    def _t_cat(i):
        return str(T[i].get("category") or "None")

    HDR = (f"{'stratum':<18}{'n':>5}{'base':>8}{'treat':>8}{'Δ':>8}"
           f"{'  95% CI':>18}{'b/c':>9}{'McNemar p':>10}")
    print("=" * 92)
    print(f"TRACK-A PAIRED DIFF  {base_dir}  vs  {treat_dir}   (tools - llm_only)")
    print("strata keyed off the TOOLS arm's initial router branch")
    print("=" * 92)
    print(f"paired rows: {len(ids)}   (* sig gain, X sig loss; p<0.05 exact McNemar)\n")
    print(HDR); print("-" * 92)
    print(line("OVERALL", overall))

    # by category (tools arm's category == llm_only's; same rows)
    print("\n-- by category " + "-" * 77)
    print(HDR); print("-" * 92)
    cats = {}
    for i in ids:
        k = _t_cat(i)
        cats.setdefault(k, [[], []])
        cats[k][0].append(int(bool(B[i].get("correct"))))
        cats[k][1].append(int(bool(T[i].get("correct"))))
    for k, v in sorted(cats.items(), key=lambda kv: len(kv[1][0]), reverse=True):
        print(line(k, stats(v[0], v[1])))

    # by TOOLS-arm initial route (the mechanism view), Holm-corrected
    print("\n-- by tools-arm initial_branch_used (MECHANISM VIEW, Holm-corrected) " + "-" * 24)
    BHDR = HDR + f"{'Holm p':>10}"
    print(BHDR); print("-" * 102)
    routes = {}
    for i in ids:
        k = _t_route(i)
        routes.setdefault(k, [[], []])
        routes[k][0].append(int(bool(B[i].get("correct"))))
        routes[k][1].append(int(bool(T[i].get("correct"))))
    route_rows = sorted(
        ((k, stats(v[0], v[1])) for k, v in routes.items()),
        key=lambda kv: kv[1]["n"], reverse=True,
    )
    holm = holm_adjust([s["p"] for _, s in route_rows])
    for (k, s), padj in zip(route_rows, holm):
        star = (" *" if s["d"] > 0 else " X") if padj < 0.05 else ""
        bc = f"{s['b']}/{s['c']}"
        print(f"{k[:18]:<18}{s['n']:>5}{s['ab']:>8.3f}{s['at']:>8.3f}{s['d']:>+8.3f}"
              f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{bc:>9}{s['p']:>10.4f}{padj:>10.4f}{star}")
    print("=" * 92)


if __name__ == "__main__":
    main()
