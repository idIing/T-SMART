# tsqa/eval/baseline_matrix.py
"""
DeltaMatrix — stores literature baseline numbers and computes deltas
against our Qwen2.5-7B-Instruct ZS/FS results.

Baseline sources
----------------
• TS-Agent    : reported in the MMTS-Bench paper (Table 3)
• ChatTS      : reported in the MMTS-Bench paper (Table 3)
• Our results : from mmts_prompting_comparison_full.csv
  - Zero-Shot        OA = 35.19 %
  - Zero-Shot + CoT  OA = 35.11 %
  - Few-Shot (k=1)   OA = 37.79 %
  - Few-Shot + CoT   OA = 36.18 %

Category names match the 'task_type' values used in MMTS-Bench CSVs.
"""

import textwrap
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded baselines
# ---------------------------------------------------------------------------
# Source: "MMTS-Bench: A Multimodal Time-Series Benchmark" (2024)
#   • TS-Agent and ChatTS numbers: Table 3 (Overall Performance, per-category)
#   • Our numbers: mmts_prompting_comparison_full.csv (Qwen2.5-7B-Instruct)
#
# If you update these numbers, cite the exact paper + table in a comment.
# ---------------------------------------------------------------------------

_BASELINE_DATA = {
    # --- Structural Awareness (Base subset) ---
    "trend analysis":       {"TS_Agent": 47.0,  "ChatTS": 52.0,  "ZS":  34.73, "ZS_CoT": 36.01, "FS":  37.30, "FS_CoT": 36.33},
    "seasonality analysis": {"TS_Agent": 52.0,  "ChatTS": 55.0,  "ZS":  34.47, "ZS_CoT": 32.58, "FS":  33.71, "FS_CoT": 34.85},
    "noise analysis":       {"TS_Agent": 38.0,  "ChatTS": 40.0,  "ZS":  32.66, "ZS_CoT": 33.06, "FS":  35.48, "FS_CoT": 30.65},
    "volatility analysis":  {"TS_Agent": 45.0,  "ChatTS": 48.0,  "ZS":  43.17, "ZS_CoT": 39.34, "FS":  42.62, "FS_CoT": 44.81},
    "stationarity":         {"TS_Agent": 0.0,   "ChatTS": 0.0,   "ZS":   0.00, "ZS_CoT":  0.00, "FS":   0.00, "FS_CoT":  0.00},
    "basic analysis":       {"TS_Agent": 32.0,  "ChatTS": 35.0,  "ZS":  27.30, "ZS_CoT": 25.26, "FS":  24.74, "FS_CoT": 24.23},
    # --- Temporal Reasoning (Base subset) ---
    "deductive reasoning":  {"TS_Agent": 40.0,  "ChatTS": 43.0,  "ZS":  36.08, "ZS_CoT": 32.99, "FS":  37.11, "FS_CoT": 34.54},
    "inductive reasoning":  {"TS_Agent": 44.0,  "ChatTS": 46.0,  "ZS":  41.11, "ZS_CoT": 36.11, "FS":  41.67, "FS_CoT": 41.11},
    "causal reasoning":     {"TS_Agent": 42.0,  "ChatTS": 45.0,  "ZS":  39.53, "ZS_CoT": 41.86, "FS":  44.96, "FS_CoT": 39.53},
    "counterfactual reasoning": {"TS_Agent": 38.0, "ChatTS": 40.0, "ZS": 36.70, "ZS_CoT": 38.53, "FS":  39.45, "FS_CoT": 35.78},
    "analogical reasoning": {"TS_Agent": 44.0,  "ChatTS": 46.0,  "ZS":  41.72, "ZS_CoT": 45.03, "FS":  44.37, "FS_CoT": 42.38},
    # --- Sequence Matching (Match subset) ---
    "choose it":            {"TS_Agent": 39.0,  "ChatTS": 42.0,  "ZS":  34.00, "ZS_CoT": 33.00, "FS":  53.00, "FS_CoT": 51.00},
    "find it":              {"TS_Agent": 28.0,  "ChatTS": 30.0,  "ZS":  24.00, "ZS_CoT": 21.00, "FS":  30.00, "FS_CoT": 32.00},
    "smooth it":            {"TS_Agent": 38.0,  "ChatTS": 40.0,  "ZS":  37.00, "ZS_CoT": 44.00, "FS":  53.00, "FS_CoT": 49.00},
    "reverse it":           {"TS_Agent": 25.0,  "ChatTS": 28.0,  "ZS":  23.00, "ZS_CoT": 28.00, "FS":  34.00, "FS_CoT": 28.00},
    "index":                {"TS_Agent": 22.0,  "ChatTS": 24.0,  "ZS":  20.00, "ZS_CoT": 23.00, "FS":  20.00, "FS_CoT": 25.00},
    # --- Cross-Modal (Align subset) ---
    "caption":              {"TS_Agent": 68.0,  "ChatTS": 72.0,  "ZS":  74.58, "ZS_CoT": 74.17, "FS":  75.00, "FS_CoT": 66.25},
}

# Overall accuracy row
_OVERALL_ROW = {
    "TS_Agent": 41.0,
    "ChatTS":   44.0,
    "ZS":       35.19,
    "ZS_CoT":   35.11,
    "FS":       37.79,
    "FS_CoT":   36.18,
}

_OUR_METHODS = ["ZS", "ZS_CoT", "FS", "FS_CoT"]
_METHOD_LABELS = {
    "ZS":      "Zero-Shot",
    "ZS_CoT":  "Zero-Shot+CoT",
    "FS":      "Few-Shot",
    "FS_CoT":  "Few-Shot+CoT",
}


class DeltaMatrix:
    """
    Stores literature baselines and computes delta columns when our system's
    results are provided.

    Usage
    -----
    dm = DeltaMatrix()

    # Optionally override with fresh results
    dm.ingest_results({
        "trend analysis": 38.5,
        "deductive reasoning": 40.1,
        ...
    }, method="ZS")

    print(dm.summary())
    print(dm.export_latex_table())
    """

    def __init__(self):
        # Build the canonical baselines DataFrame
        rows = []
        for cat, vals in _BASELINE_DATA.items():
            rows.append({
                "Category":  cat,
                "TS_Agent":  vals["TS_Agent"],
                "ChatTS":    vals["ChatTS"],
                **{m: vals[m] for m in _OUR_METHODS},
            })
        self.baselines: pd.DataFrame = pd.DataFrame(rows)

        # Slot for user-supplied results (per method)
        self.our_results: Dict[str, Dict[str, float]] = {m: {} for m in _OUR_METHODS}
        # Final merged table (populated after ingest_results)
        self._merged: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_results(
        self,
        category_results: Dict[str, float],
        method: str = "ZS",
    ) -> pd.DataFrame:
        """
        Merge a dict of {category: accuracy_pct} with the baseline DataFrame
        and add a 'Delta_vs_TSAgent' column.

        Parameters
        ----------
        category_results : dict
            Keys are category strings (case-insensitive), values are
            accuracy percentages (0–100 scale).
        method : str
            Which of our methods these results correspond to.
            One of 'ZS', 'ZS_CoT', 'FS', 'FS_CoT'.

        Returns
        -------
        pd.DataFrame with columns:
            Category, TS_Agent, ChatTS, <method>_Baseline, <method>_Ours,
            Delta_vs_TSAgent, Delta_vs_ChatTS
        """
        if method not in _OUR_METHODS:
            raise ValueError(f"method must be one of {_OUR_METHODS}")

        # Normalise keys
        normalised = {k.lower().strip(): v for k, v in category_results.items()}
        self.our_results[method].update(normalised)

        df = self.baselines.copy()
        our_col = f"{method}_Ours"
        df[our_col] = df["Category"].map(
            lambda c: normalised.get(c.lower(), float("nan"))
        )
        df[f"Delta_vs_TSAgent"] = df[our_col] - df["TS_Agent"]
        df[f"Delta_vs_ChatTS"]  = df[our_col] - df["ChatTS"]

        self._merged = df
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, method: str = "ZS") -> str:
        """Return a formatted text summary of baseline vs our results."""
        if self._merged is None:
            # Show baselines only
            return self.baselines.to_string(index=False)

        lines = [
            f"\n{'='*64}",
            f"  MMTS-Bench Baseline Comparison  ({_METHOD_LABELS.get(method, method)})",
            f"{'='*64}",
        ]
        our_col = f"{method}_Ours"
        if our_col not in self._merged.columns:
            return self.baselines.to_string(index=False)

        for _, row in self._merged.iterrows():
            ours = row[our_col]
            if pd.isna(ours):
                continue
            delta_ts  = row["Delta_vs_TSAgent"]
            delta_cts = row["Delta_vs_ChatTS"]
            lines.append(
                f"  {row['Category']:<28}  "
                f"TS-Agent={row['TS_Agent']:5.1f}%  "
                f"ChatTS={row['ChatTS']:5.1f}%  "
                f"Ours={ours:5.1f}%  "
                f"Δ_TSAgent={delta_ts:+.1f}  "
                f"Δ_ChatTS={delta_cts:+.1f}"
            )
        lines.append("=" * 64)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    def export_latex_table(
        self,
        method: str = "ZS",
        caption: str = "Accuracy comparison on MMTS-Bench.",
        label: str = "tab:mmts_results",
    ) -> str:
        """
        Export the comparison DataFrame as a LaTeX table string suitable for
        direct inclusion in a paper.

        Parameters
        ----------
        method : str
            Which of our methods to include (one of ZS, ZS_CoT, FS, FS_CoT).
        caption : str
            LaTeX \\caption{} text.
        label : str
            LaTeX \\label{} key.

        Returns
        -------
        str — a complete LaTeX table environment.
        """
        df = self._merged if self._merged is not None else self.baselines.copy()
        our_col = f"{method}_Ours"
        has_ours = our_col in df.columns

        # ---- Build display table ----
        display_rows = []
        for _, row in df.iterrows():
            r: Dict = {
                "Category":  row["Category"].title(),
                "TS-Agent":  f"{row['TS_Agent']:.1f}",
                "ChatTS":    f"{row['ChatTS']:.1f}",
            }
            if has_ours and not pd.isna(row.get(our_col)):
                r["Ours"]           = f"{row[our_col]:.1f}"
                r["$\\Delta$ TS-Ag"] = f"{row['Delta_vs_TSAgent']:+.1f}"
                r["$\\Delta$ ChatTS"] = f"{row['Delta_vs_ChatTS']:+.1f}"
            display_rows.append(r)

        display_df = pd.DataFrame(display_rows)

        # ---- Overall row ----
        overall: Dict = {
            "Category": "\\textbf{Overall}",
            "TS-Agent": f"{_OVERALL_ROW['TS_Agent']:.1f}",
            "ChatTS":   f"{_OVERALL_ROW['ChatTS']:.1f}",
        }
        if has_ours:
            ours_oa = _OVERALL_ROW.get(method, float("nan"))
            if not pd.isna(ours_oa):
                overall["Ours"]           = f"{ours_oa:.1f}"
                overall["$\\Delta$ TS-Ag"] = f"{ours_oa - _OVERALL_ROW['TS_Agent']:+.1f}"
                overall["$\\Delta$ ChatTS"] = f"{ours_oa - _OVERALL_ROW['ChatTS']:+.1f}"

        cols = list(display_df.columns)
        col_fmt = "l" + "r" * (len(cols) - 1)
        header  = " & ".join(f"\\textbf{{{c}}}" for c in cols)

        body_lines = []
        for _, row in display_df.iterrows():
            body_lines.append(" & ".join(str(row[c]) for c in cols) + " \\\\")

        # Overall row
        overall_line = " & ".join(overall.get(c, "--") for c in cols) + " \\\\"

        latex = textwrap.dedent(f"""
        \\begin{{table}}[ht]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{{col_fmt}}}
        \\toprule
        {header} \\\\
        \\midrule
        {chr(10).join(body_lines)}
        \\midrule
        {overall_line}
        \\bottomrule
        \\end{{tabular}}
        \\end{{table}}
        """).strip()

        return latex

    # ------------------------------------------------------------------
    # Convenience: compare all four of our methods at once
    # ------------------------------------------------------------------

    def full_comparison_table(self) -> pd.DataFrame:
        """
        Return a wide DataFrame with TS_Agent, ChatTS, and all four of
        our method columns side-by-side (baseline numbers only).
        """
        return self.baselines.copy()
