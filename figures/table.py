import polars as pl
from great_tables import GT, loc, style

# 1. Define the data directly in Polars
data = {
    "Model": [
        "Phi-3.5",
        "GPT-4o",
        "Gemini-2.5",
        "Gemini-2.5",
        "Mistral-7B",
        "ChatTS",
        "TS-Agent",
        "T-SMART Gemini-2.5-lite",
        "T-SMART Qwen-3.5B",
    ],
    "Modality": [
        "Text",
        "Text",
        "Text",
        "Vision",
        "Text",
        "TS",
        "Agentic",
        "Multimodal Agentic",
        "Multimodal Agentic",
    ],
    "Overall": [
        0.3180,
        0.3880,
        0.3040,
        0.4880,
        0.3460,
        0.3900,
        0.6020,
        round((0.6298 + 0.7262 + 0.4722 + 0.5280 + 0.6667) / 5, 4),
        None,
    ],
    "Pattern Rec.": [
        0.44,
        0.35,
        0.29,
        0.51,
        0.29,
        0.37,
        0.71,
        0.6298,
        None,
    ],
    "Noise Und.": [
        0.24,
        0.32,
        0.32,
        0.54,
        0.33,
        0.39,
        0.61,
        0.7262,
        None,
    ],
    "Anomaly Det.": [
        0.25,
        0.40,
        0.26,
        0.46,
        0.38,
        0.36,
        0.57,
        0.4722,
        None,
    ],
    "Similarity": [
        0.41,
        0.51,
        0.35,
        0.69,
        0.45,
        0.53,
        0.57,
        0.5280,
        None,
    ],
    "Causality": [
        0.25,
        0.36,
        0.30,
        0.24,
        0.28,
        0.30,
        0.55,
        0.6667,
        None,
    ],
}

df = pl.DataFrame(data)

# 2. Define the specific columns we want grouped
sel_tasks = ["Pattern Rec.", "Noise Und.", "Anomaly Det.", "Similarity", "Causality"]
all_metrics = ["Overall"] + sel_tasks

our_models = ["T-SMART Gemini-2.5-lite", "T-SMART Qwen-3.5B"]

# 3. Build and style the Great Table
eval_table = (
    GT(df)
    .tab_header(
        title="TimeSeriesExam [NeurIPS '24] Benchmark Evaluation",
        subtitle="",
    )
    .tab_spanner(label="Task-Specific Accuracies", columns=sel_tasks)
    .cols_label(Overall="Overall Acc.")
    .fmt_percent(columns=all_metrics, decimals=1, rows=pl.col("Model") != "T-SMART Qwen-3.5B")
    .sub_missing(columns=all_metrics, missing_text="TBD")
    # Background for our framework rows
    .tab_style(
        style=style.fill(color="#fff7ed"),
        locations=loc.body(rows=pl.col("Model").is_in(our_models)),
    )
    # Background for TS-Agent
    .tab_style(
        style=style.fill(color="#f4f7f9"),
        locations=loc.body(rows=pl.col("Model") == "TS-Agent"),
    )
    # Brand our framework labels bold + orange
    .tab_style(
        style=style.text(weight="bold", color="#d35400"),
        locations=loc.body(
            columns=["Model", "Modality"],
            rows=pl.col("Model").is_in(our_models),
        ),
    )
    # Center align the data columns
    .tab_style(
        style=style.text(align="center"),
        locations=loc.body(columns=all_metrics),
    )
    # Center the Modality column header
    .tab_style(
        style=style.text(align="center"),
        locations=loc.column_labels(columns=["Modality"]),
    )
)

# Bold the highest score in every metric column
for col_name in all_metrics:
    max_val = df[col_name].drop_nulls().max()
    eval_table = eval_table.tab_style(
        style=style.text(weight="bold"),
        locations=loc.body(columns=[col_name], rows=pl.col(col_name) == max_val),
    )

# Underline the second-best value in every metric column
for col_name in all_metrics:
    vals = df[col_name].drop_nulls().to_list()
    distinct_vals = sorted(set(vals), reverse=True)
    if len(distinct_vals) < 2:
        continue
    second_val = distinct_vals[1]
    eval_table = eval_table.tab_style(
        style=style.text(decorate="underline"),
        locations=loc.body(columns=[col_name], rows=pl.col(col_name) == second_val),
    )

# 4. Save
eval_table.write_raw_html("agentic_benchmark_results.html")