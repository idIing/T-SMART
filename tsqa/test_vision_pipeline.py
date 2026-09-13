import os
import json
from datasets import load_dataset
from tsqa.llm.client import GeminiClient
from tsqa.eval.runner import run_pipeline
import pandas as pd
from dotenv import load_dotenv


def main():
    import time
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable.")
        return

    debug_mode = True  # Set to True to see FULL traces

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(
        os.path.dirname(__file__), f"eval_report_{timestamp}.txt"
    )
    print(f"Detailed logs will be written to: {log_file_path}")
    log_file = open(log_file_path, "w", encoding="utf-8")

    def mprint(text):
        print(text, flush=True)
        log_file.write(str(text) + "\n")
        log_file.flush()

    # 1. Initialize the LLM client
    mprint("Initializing GeminiClient...")
    client = GeminiClient(api_key=api_key, model_name="gemini-2.5-flash", max_tokens=4000)

    # 2. Load the dataset
    mprint("Loading AutonLab/TimeSeriesExam1...")
    ds = load_dataset("AutonLab/TimeSeriesExam1", split="test")
    df = ds.to_pandas()

    # 3. Select 15 samples across different categories
    categories = df["category"].unique()
    sampled_dfs = []
    samples_per_category = max(1, 15 // len(categories))

    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_samples = min(samples_per_category, len(cat_df))
        sampled_dfs.append(cat_df.sample(n=n_samples, random_state=42))

    test_df = pd.concat(sampled_dfs).head(15)

    mprint(f"\nSelected {len(test_df)} samples for testing:")
    mprint(test_df["category"].value_counts())

    # 4. Run the pipeline
    mprint("\nStarting pipeline execution...\n" + "=" * 50)

    correct_count = 0
    total_count = 0
    cat_correct = {cat: 0 for cat in categories}
    cat_total = {cat: 0 for cat in categories}

    for i, row in enumerate(test_df.itertuples(index=False), 1):
        total_count += 1
        cat_total[row.category] += 1
        mprint(
            f"\n--- Sample {i}/15 | Category: {row.category} | Task: {row.question[:50]}... ---"
        )

        row_dict = row._asdict()
        if True:
            # Run the pipeline with delay to avoid API limits
            time.sleep(1)
            result = run_pipeline(row_dict, llm_client=client)

            if debug_mode:
                # To see router output
                mprint(
                    f"[DEBUG] Router Raw: {result.get('raw_router', 'NOT PROVIDED')}"
                )
                mprint(
                    f"[DEBUG] Router Parsed: {result.get('routing', 'NOT PROVIDED')}"
                )

            # --- Verification 1: Math branches ---
            branch_used = result.get("branch_used")
            flags = result.get("flags", [])
            mprint(f"✅ Math Branch executed: {branch_used}")
            mprint(f"   Flags: {flags}")

            # --- Verification 2: Vision Trigger ---
            evidence_bundle = result.get("evidence_bundle", {})
            vision_evidence = evidence_bundle.get("vision_evidence")
            vision_text = evidence_bundle.get("vision_text")

            # Print if there was a vision parsing error
            if "vision_error" in (result.get("evidence") or {}):
                mprint(f"❌ Vision Error: {result['evidence']['vision_error']}")

            vision_triggered = bool(vision_evidence or vision_text)
            mprint(f"✅ Vision Trigger fired: {vision_triggered}")
            if vision_triggered:
                trigger_reason = result.get("evidence", {}).get("vision_trigger")
                if trigger_reason:
                    mprint(f"   Trigger Reason: {trigger_reason}")

            if debug_mode:
                if vision_triggered:
                    mprint(
                        f"\n   [DEBUG] Raw Vision Evidence: {json.dumps(vision_evidence, indent=2)}"
                    )
                    if vision_evidence is None:
                        mprint(
                            f"   [DEBUG] VISION FAILED TO PARSE JSON. RAW WAS: {result.get('evidence', {}).get('vision_raw')}"
                        )

                    artifacts_list = result.get("evidence", {}).get("artifacts", [])
                    if artifacts_list:
                        img_path = artifacts_list[0].get("path")
                        mprint(f"   [DEBUG] STACKED IMAGE SAVED TO: {img_path}")
                mprint(
                    f"\n   [DEBUG] Intermediate Evidence Bundle Math Keys: {list(evidence_bundle.get('math_evidence', {}).keys())}"
                )

            # --- Verification 3: Final LLM prompt ---
            mprint("\n>> Reasoner Answer Output:")
            q_hint = row_dict.get("question_hint")
            if q_hint:
                mprint(f"   [Ground Truth Hint]: {q_hint}")
            pred = result.get("predicted_letter")
            from tsqa.eval.runner import _gold_letter

            gold = _gold_letter(row_dict)

            mprint(f"Predicted: {pred} | True Answer: {gold}")

            if debug_mode:
                mprint(
                    f"[DEBUG] Raw response:\n{result.get('raw_answer', '').strip()}\n"
                )

            if pred == gold:
                correct_count += 1
                cat_correct[row.category] += 1
                mprint("✅ OUTCOME: CORRECT")
            else:
                mprint("❌ OUTCOME: INCORRECT")

    mprint("\n" + "=" * 50)
    mprint(
        f"OVERALL ACCURACY: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)"
    )
    mprint("\nACCURACY BY CATEGORY:")
    for cat in cat_total:
        if cat_total[cat] > 0:
            pct = cat_correct[cat] / cat_total[cat] * 100
            mprint(f"  {cat}: {cat_correct[cat]}/{cat_total[cat]} ({pct:.1f}%)")

    log_file.close()
    print(f"\nAll logs safely secured to {log_file_path}")


if __name__ == "__main__":
    main()
