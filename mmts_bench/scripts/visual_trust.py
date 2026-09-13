#!/usr/bin/env python3
# scripts/visual_trust.py
"""
Visual-Trust analysis (Workstream A3) — does the vision modality actually help?
===============================================================================
On rows where the *vision* sensor's independent suggestion CONFLICTS with the
*math-only* answer, this measures how often the final reasoner FOLLOWS vision
vs. math, and which modality is actually RIGHT — stratified per
(branch × vision-confidence). It produces:

  1. A per-branch / per-confidence TRUST TABLE (follow-rate, P(vision right),
     P(math right), net value of following vision), and
  2. A DISCORDANCE DATASET (one row per vision-fired *decision*) — the labelled
     signal a future learned "should-I-look" gate will be calibrated on.

Two modes
---------
PRIMARY (two-CSV MMTS mode) — the real experiment:
    A *permissive* run (vision allowed to fire; carries the vision suggestion)
    is joined PAIRWISE on (subset, sample_id) against a *vision_off* run (vision
    suppressed on every branch → its `predicted` is the pure "math position").
    The counterfactual lets us compute follow-rate AND net value of vision.

        python3 scripts/visual_trust.py \\
            --permissive outputs/.../mmts_results_*_baseline_*.csv \\
            --vision-off outputs/.../mmts_results_*_vision_off_*.csv \\
            [--out outputs/visual_trust_discordance.csv] [--seed 0]

ALTERNATE (single-file TSExam mode) — `--tsexam`:
    A TSExam `research/outputs/<tag>/results.json` (a JSON list with fields
    branch_used, vision_letter, vision_confidence, vision_triggered, pred, gold).
    There is NO vision_off counterfactual here, so the "math position" is
    UNAVAILABLE: follow-rate, P(math right) and net-value CANNOT be computed.
    Only the vision-vs-gold columns are reported (n_fired, P(vision right)).

        python3 scripts/visual_trust.py --tsexam path/to/results.json

Reused, byte-for-byte, from diff_configs.py: paired_stats, bootstrap_delta_ci,
mcnemar_p, _as_bool, _load_many. (diff_configs is import-safe — its body is
guarded by `if __name__ == "__main__"`.)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same-directory import pattern: put this script's own dir on sys.path, then
# pull the shared stats/IO helpers from diff_configs (no code runs on import —
# it is guarded by `if __name__ == "__main__"`).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diff_configs import (          # noqa: E402
    paired_stats,
    bootstrap_delta_ci,             # re-exported for callers; used indirectly via paired_stats
    mcnemar_p,                      # re-exported for callers
    _as_bool,
    _load_many,
)

# Silence "imported but unused" while keeping the requested public surface.
_ = (bootstrap_delta_ci, mcnemar_p)

# Confidence buckets we stratify on (anything else → "other").
_CONF_BUCKETS = ["high", "medium", "low"]

# Minimum discordant-n below which a paired contrast is flagged "tiny-n / n.s.".
_MIN_N_FOR_STATS = 8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _truthy(val) -> bool:
    """True for True/'True'/'true'/1/'1'/'1.0'; False for False/None/''/NaN/0."""
    if val is None:
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        if isinstance(val, float) and np.isnan(val):
            return False
        return val != 0
    s = str(val).strip().lower()
    return s in {"true", "1", "1.0", "yes"}


def _norm_letter(val) -> str:
    """Normalise a letter cell to an uppercase A/B/C/D, or '' if empty/None/NaN."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).strip().upper()
    if s in {"", "NAN", "NONE"}:
        return ""
    return s


def _conf_bucket(val) -> str:
    """Map a vision_confidence cell to a stratum label: high/medium/low/other."""
    if val is None:
        return "other"
    if isinstance(val, float) and np.isnan(val):
        return "other"
    s = str(val).strip().lower()
    if s in _CONF_BUCKETS:
        return s
    return "other"


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


# ---------------------------------------------------------------------------
# Trust-table reporting (fixed-width, diff_configs._row_line house style)
# ---------------------------------------------------------------------------

_TRUST_HDR = (
    f"{'stratum':<30}{'n_fired':>9}{'n_disc':>8}{'follow_v':>10}"
    f"{'P(v_rt)':>9}{'P(m_rt)':>9}{'net_v':>9}"
)


def _trust_row(label: str, st: dict) -> str:
    """One fixed-width line of the trust table from a stratum-stats dict."""
    def fmt(x, w=9, p=3, signed=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return f"{'--':>{w}}"
        # Format spec order is [sign][width].[precision]f — sign before width.
        spec = f"{'+' if signed else ''}{w}.{p}f"
        return f"{x:>{spec}}"

    return (
        f"{label[:30]:<30}{st['n_fired']:>9}{st['n_disc']:>8}"
        f"{fmt(st['follow_vision_rate'], 10)}"
        f"{fmt(st['p_vision_right'])}{fmt(st['p_math_right'])}"
        f"{fmt(st['net_value'], 9, 3, signed=True)}"
    )


def _stratum_stats(g: pd.DataFrame) -> dict:
    """Compute the trust-table cells for one stratum (subset of fired rows).

    `g` must already be restricted to vision-FIRED rows (vision_triggered &
    vision_letter present). Discordance is recomputed inside.
    """
    n_fired = len(g)
    disc = g[g["_discordant"]]
    n_disc = len(disc)
    if n_disc == 0:
        return {
            "n_fired": n_fired, "n_disc": 0,
            "follow_vision_rate": float("nan"),
            "p_vision_right": float("nan"),
            "p_math_right": float("nan"),
            "net_value": float("nan"),
        }
    fv = _safe_div(disc["_follows_vision"].sum(), n_disc)
    pv = _safe_div(disc["_vision_right"].sum(), n_disc)
    pm = _safe_div(disc["_math_right"].sum(), n_disc)
    return {
        "n_fired": n_fired, "n_disc": n_disc,
        "follow_vision_rate": fv,
        "p_vision_right": pv,
        "p_math_right": pm,
        "net_value": pv - pm,
    }


def _print_trust_table(title: str, fired: pd.DataFrame, by: str) -> None:
    """Print a grouped trust table; an ALL row is appended by the caller."""
    print(f"\n-- {title} " + "-" * max(0, (72 - len(title))))
    print(_TRUST_HDR)
    print("-" * 84)
    rows = []
    for key, g in fired.groupby(by, dropna=False):
        rows.append((str(key), _stratum_stats(g)))
    rows.sort(key=lambda kv: kv[1]["n_fired"], reverse=True)
    for label, st in rows:
        print(_trust_row(label, st))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# Columns we pull from the permissive side (final decision + vision suggestion).
_NEEDED_PERMISSIVE = [
    "sample_id", "subset", "branch_used", "subtype_used",
    "vision_letter", "vision_confidence", "vision_triggered",
    "predicted", "ground_truth", "flags", "viz_type",
]


def _ensure_cols(df: pd.DataFrame, cols, where: str) -> None:
    """Hard-fail with a clear message if a required column is missing — the
    runner is being extended to emit these; this is where a name drift shows up.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[visual_trust] {where} CSV is missing required column(s): {missing}\n"
            f"  present columns: {sorted(df.columns)}\n"
            "  (The runner emits vision_letter/vision_confidence/vision_triggered "
            "etc. once extended — re-run run_mmts_baseline.py with the updated "
            "schema, or check the column names.)"
        )


def _load_paired(permissive_paths, vision_off_paths) -> pd.DataFrame:
    """Load both runs, join PAIRWISE on (subset, sample_id), restrict to fired."""
    perm = _load_many(permissive_paths)      # drops error=='skipped' internally
    voff = _load_many(vision_off_paths)

    # Coerce join keys to str on both sides.
    for df in (perm, voff):
        df["sample_id"] = df["sample_id"].astype(str)
        if "subset" not in df.columns:
            df["subset"] = ""
        df["subset"] = df["subset"].astype(str)

    _ensure_cols(perm, _NEEDED_PERMISSIVE, "permissive")
    _ensure_cols(voff, ["sample_id", "subset", "predicted"], "vision_off")

    key = ["subset", "sample_id"]
    perm = perm.drop_duplicates(subset=key).copy()
    voff = voff.drop_duplicates(subset=key).copy()

    merged = perm.merge(
        voff[key + ["predicted"]].rename(columns={"predicted": "math_letter"}),
        on=key, how="inner",
    )
    return merged


def _load_tsexam(path: str) -> pd.DataFrame:
    """Load a TSExam results.json into a frame with the columns we report on.

    Maps the JSON field names (pred/gold) onto our internal names. There is no
    vision_off counterfactual, so `math_letter` is left empty.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"[visual_trust] --tsexam {path} is not a JSON list.")
    rows = []
    for r in data:
        rows.append({
            "sample_id":        r.get("id"),
            "subset":           "TSExam",
            "branch_used":      r.get("branch_used"),
            "subtype_used":     r.get("subtype_used"),
            "vision_letter":    r.get("vision_letter"),
            "vision_confidence": r.get("vision_confidence"),
            "vision_triggered": r.get("vision_triggered"),
            "predicted":        r.get("pred"),
            "ground_truth":     r.get("gold"),
            "flags":            r.get("flags"),
            "viz_type":         r.get("viz_type"),
            "math_letter":      "",      # no counterfactual in single-file mode
        })
    df = pd.DataFrame(rows)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Core feature derivation (shared by both modes)
# ---------------------------------------------------------------------------

def _derive(merged: pd.DataFrame, have_math: bool) -> pd.DataFrame:
    """Add normalised letters, fired flag, confidence bucket, and the per-row
    discordance/decision labels. Returns the FIRED subset only.
    """
    df = merged.copy()
    df["_vletter"] = df["vision_letter"].map(_norm_letter)
    df["_pred"]    = df["predicted"].map(_norm_letter)
    df["_gold"]    = df["ground_truth"].map(_norm_letter)
    df["_mletter"] = df["math_letter"].map(_norm_letter) if have_math else ""
    df["_conf"]    = df["vision_confidence"].map(_conf_bucket)
    df["_fired"]   = df["vision_triggered"].map(_truthy) & (df["_vletter"] != "")

    fired = df[df["_fired"]].copy()

    if have_math:
        # Discordance is only defined when both letters are present.
        fired["_discordant"]    = (fired["_vletter"] != fired["_mletter"]) \
                                  & (fired["_mletter"] != "")
        fired["_follows_vision"] = (fired["_pred"] == fired["_vletter"]).astype(int)
        fired["_vision_right"]   = (fired["_vletter"] == fired["_gold"]) \
                                   & (fired["_gold"] != "")
        fired["_math_right"]     = (fired["_mletter"] == fired["_gold"]) \
                                   & (fired["_gold"] != "")
        fired["_vision_right"]   = fired["_vision_right"].astype(int)
        fired["_math_right"]     = fired["_math_right"].astype(int)
    else:
        # Single-file (TSExam) mode: no math position → discordance/follow/math
        # are undefined. Only vision-vs-gold is computable.
        fired["_discordant"]     = False
        fired["_follows_vision"] = np.nan
        fired["_vision_right"]   = ((fired["_vletter"] == fired["_gold"])
                                    & (fired["_gold"] != "")).astype(int)
        fired["_math_right"]     = np.nan
    return fired


# ---------------------------------------------------------------------------
# Mode A — PRIMARY (two-CSV) full analysis
# ---------------------------------------------------------------------------

def _run_paired_mode(args) -> int:
    merged = _load_paired(args.permissive, args.vision_off)

    if merged.empty:
        print("\n[visual_trust] No shared (subset, sample_id) rows between the "
              "permissive and vision_off runs — nothing to analyse.")
        return 0

    n_paired = len(merged)
    fired = _derive(merged, have_math=True)

    print("\n" + "=" * 84)
    print("VISUAL-TRUST ANALYSIS  —  permissive vs vision_off (paired)")
    print("=" * 84)
    print(f"paired rows (inner join on subset,sample_id): {n_paired}")
    print(f"vision-FIRED rows (vision_triggered & vision_letter present): {len(fired)}")

    if fired.empty:
        print("\n[visual_trust] No vision-fired rows in the permissive run — "
              "the vision gate never triggered, so there is nothing to trust.")
        return 0

    # Per-branch fired/discordant census.
    n_disc_total = int(fired["_discordant"].sum())
    print(f"discordant fired rows (vision_letter != math_letter): {n_disc_total}")

    if n_disc_total == 0:
        print("\n[visual_trust] Vision fired but NEVER disagreed with the math "
              "position — vision changed no decision, so follow-rate / net value "
              "are undefined. (Trust table below shows fired counts only.)")

    # ── Trust tables ────────────────────────────────────────────────────
    # Combined branch × confidence table, with an ALL-confidence row per branch
    # and a grand-total row.
    print("\n" + "=" * 84)
    print("TRUST TABLE  —  per (branch × vision_confidence)")
    print("net_v = P(vision_right | disc) - P(math_right | disc)  "
          "[disc = discordant fired rows]")
    print("=" * 84)
    print(_TRUST_HDR)
    print("-" * 84)
    for branch, gb in sorted(fired.groupby("branch_used", dropna=False),
                             key=lambda kv: len(kv[1]), reverse=True):
        # rows for each confidence bucket within this branch
        sub_rows = []
        for conf, gc in gb.groupby("_conf", dropna=False):
            sub_rows.append((f"  {branch} / {conf}", _stratum_stats(gc)))
        sub_rows.sort(key=lambda kv: kv[1]["n_fired"], reverse=True)
        # branch ALL row first, then its confidence breakdown
        print(_trust_row(f"{branch} [ALL]", _stratum_stats(gb)))
        for label, st in sub_rows:
            print(_trust_row(label, st))
    print("-" * 84)
    print(_trust_row("GRAND TOTAL", _stratum_stats(fired)))

    # By-branch-only and by-confidence-only tables.
    _print_trust_table("by branch_used only", fired, "branch_used")
    _print_trust_table("by vision_confidence only", fired, "_conf")

    # ── Paired significance on the discordance set ───────────────────────
    # Net value of following vision = P(vision_right) - P(math_right). Treat
    # math_right as the BASE arm and vision_right as the TREAT arm so
    # paired_stats' delta == net value, and McNemar's discordant counts (b,c)
    # are exactly "math-right-only" vs "vision-right-only" rows.
    print("\n" + "=" * 84)
    print("NET VALUE OF FOLLOWING VISION  (discordance set; paired)")
    print("delta = P(vision_right) - P(math_right);  CI = bootstrap 95%;  "
          "McNemar p = exact")
    print("=" * 84)
    _STAT_HDR = (f"{'stratum':<26}{'n_disc':>8}{'P(m_rt)':>9}{'P(v_rt)':>9}"
                 f"{'delta':>9}{'95% CI':>18}{'b/c':>9}{'McNemar p':>12}")
    print(_STAT_HDR)
    print("-" * 100)

    def _stat_line(label: str, disc: pd.DataFrame) -> str:
        n = len(disc)
        if n == 0:
            return f"{label[:26]:<26}{0:>8}  (no discordant rows)"
        base  = disc["_math_right"].to_numpy()
        treat = disc["_vision_right"].to_numpy()
        s = paired_stats(base, treat, seed=args.seed)
        ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
        bc = f"{s['b']}/{s['c']}"
        flag = ""
        if n < _MIN_N_FOR_STATS:
            flag = "  (tiny-n, n.s.)"
        elif s["mcnemar_p"] < 0.05:
            flag = "  *" if s["delta"] > 0 else "  X"
        return (f"{label[:26]:<26}{s['n']:>8}{s['acc_base']:>9.3f}"
                f"{s['acc_treat']:>9.3f}{s['delta']:>+9.3f}{ci:>18}{bc:>9}"
                f"{s['mcnemar_p']:>12.4f}{flag}")

    disc_all = fired[fired["_discordant"]]
    print(_stat_line("OVERALL", disc_all))
    print("-" * 100)
    for branch, gb in sorted(fired.groupby("branch_used", dropna=False),
                             key=lambda kv: int(kv[1]["_discordant"].sum()),
                             reverse=True):
        print(_stat_line(str(branch), gb[gb["_discordant"]]))

    print("\nLegend: * = following vision is a significant net gain (p<0.05); "
          "X = significant net loss; tiny-n strata are not tested.")

    # ── Write the discordance dataset ────────────────────────────────────
    _write_discordance(disc_all, args.out, have_math=True)
    return 0


# ---------------------------------------------------------------------------
# Mode B — ALTERNATE (single-file TSExam)
# ---------------------------------------------------------------------------

def _run_tsexam_mode(args) -> int:
    df = _load_tsexam(args.tsexam)
    if df.empty:
        print(f"\n[visual_trust] --tsexam {args.tsexam} contained no rows.")
        return 0

    fired = _derive(df, have_math=False)

    print("\n" + "=" * 84)
    print("VISUAL-TRUST ANALYSIS  —  TSExam single-file mode")
    print("=" * 84)
    print(f"rows loaded: {len(df)}")
    print(f"vision-FIRED rows (vision_triggered & vision_letter present): {len(fired)}")
    print("NOTE: no vision_off counterfactual in this mode — the 'math position' "
          "is UNAVAILABLE.\n      Follow-rate, P(math_right) and net-value are "
          "NOT computable; only n_fired and P(vision_right) are shown.")

    if fired.empty:
        print("\n[visual_trust] No vision-fired rows — nothing to report.")
        return 0

    hdr = f"{'stratum':<30}{'n_fired':>9}{'P(vision_right)':>18}"
    print("\n-- by (branch × confidence) " + "-" * 40)
    print(hdr)
    print("-" * 60)

    def _vline(label, g):
        n = len(g)
        pv = _safe_div(g["_vision_right"].sum(), n)
        pvs = f"{pv:.3f}" if not np.isnan(pv) else "--"
        return f"{label[:30]:<30}{n:>9}{pvs:>18}"

    for branch, gb in sorted(fired.groupby("branch_used", dropna=False),
                             key=lambda kv: len(kv[1]), reverse=True):
        print(_vline(f"{branch} [ALL]", gb))
        confs = sorted(gb.groupby("_conf", dropna=False),
                       key=lambda kv: len(kv[1]), reverse=True)
        for conf, gc in confs:
            print(_vline(f"  {branch} / {conf}", gc))
    print("-" * 60)
    print(_vline("GRAND TOTAL", fired))

    _write_discordance(fired, args.out, have_math=False)
    return 0


# ---------------------------------------------------------------------------
# Discordance dataset writer
# ---------------------------------------------------------------------------

# Feature columns for the learned-gate training set (step 5/8 of the spec).
_DUMP_COLS = [
    "sample_id", "subset", "branch_used", "subtype_used",
    "vision_confidence", "flags", "viz_type",
    "gold", "vision_letter", "math_letter", "final_pred",
    "follows_vision", "vision_right", "math_right",
]


def _write_discordance(rows: pd.DataFrame, out_path: str, have_math: bool) -> None:
    """Write the per-decision discordance dataset to CSV.

    In paired mode `rows` is the discordance set; in TSExam mode it is the
    fired set (discordance undefined without the counterfactual). Either way one
    row per vision-fired decision, with the feature columns the gate trains on.
    """
    out = pd.DataFrame()
    out["sample_id"]         = rows["sample_id"]
    out["subset"]            = rows["subset"]
    out["branch_used"]       = rows["branch_used"]
    out["subtype_used"]      = rows.get("subtype_used")
    out["vision_confidence"] = rows["vision_confidence"]
    out["flags"]             = rows.get("flags")
    out["viz_type"]          = rows.get("viz_type")
    out["gold"]              = rows["_gold"]
    out["vision_letter"]     = rows["_vletter"]
    out["math_letter"]       = rows["_mletter"] if have_math else ""
    out["final_pred"]        = rows["_pred"]
    out["follows_vision"]    = rows["_follows_vision"] if have_math else np.nan
    out["vision_right"]      = rows["_vision_right"]
    out["math_right"]        = rows["_math_right"] if have_math else np.nan

    out = out.reindex(columns=_DUMP_COLS)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_p, index=False)
    print(f"\n[visual_trust] discordance dataset ({len(out)} rows) -> {out_p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--permissive", nargs="+",
        help="Permissive-run detail CSV(s) (config 'baseline' — vision allowed "
             "to fire; carries the vision suggestion). PRIMARY mode.",
    )
    ap.add_argument(
        "--vision-off", nargs="+", dest="vision_off",
        help="Vision-off detail CSV(s) (config 'vision_off' — vision suppressed "
             "on all branches; its `predicted` is the pure math position). "
             "PRIMARY mode.",
    )
    ap.add_argument(
        "--tsexam",
        help="ALTERNATE single-file mode: a TSExam results.json (JSON list). "
             "LIMITATION: no vision_off counterfactual, so the math position is "
             "unavailable — follow-rate, P(math_right) and net-value are NOT "
             "reported; only n_fired and P(vision_right) are.",
    )
    ap.add_argument(
        "--out", default="outputs/visual_trust_discordance.csv",
        help="Where to write the per-decision discordance dataset "
             "(default: outputs/visual_trust_discordance.csv).",
    )
    ap.add_argument("--seed", type=int, default=0,
                    help="Bootstrap seed (default 0).")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    if args.tsexam:
        if args.permissive or args.vision_off:
            print("[visual_trust] --tsexam is an alternate single-file mode; "
                  "ignoring --permissive/--vision-off.", file=sys.stderr)
        return _run_tsexam_mode(args)

    if not args.permissive or not args.vision_off:
        print("[visual_trust] PRIMARY mode needs BOTH --permissive and "
              "--vision-off (or use --tsexam for single-file mode). "
              "See --help.", file=sys.stderr)
        return 2

    return _run_paired_mode(args)


if __name__ == "__main__":
    sys.exit(main())
