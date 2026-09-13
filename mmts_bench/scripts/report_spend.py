#!/usr/bin/env python3
# scripts/report_spend.py
"""
Report actual API spend from one or more mmts_run_*.json manifests.
=====================================================================
`run_mmts_baseline.py` writes a `"usage"` field into its run manifest
(mmts_run_<tag>_<timestamp>.json), populated from the LLM client's
usage_snapshot() (calls, prompt_tokens, completion_tokens, total_tokens).
This script reads one or more of those manifests and prints a per-run and
total spend report -- tokens plus an estimated USD cost from a pricing
table below.

Manifests written before the "usage" field existed have no such key; those
runs are reported with tokens/cost as "not recorded" rather than crashing.

Usage
-----
    python scripts/report_spend.py outputs/mmts_run_*.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PRICING TABLE -- edit only with a confirmed source. Do not guess a rate.
#
# Rates are USD per 1,000,000 tokens, taken from Google's official Gemini
# API pricing page on 2026-09-06:
#   https://ai.google.dev/gemini-api/docs/pricing
# (standard paid-tier, text/image/video input; battery of other tiers --
#  batch, flex, priority, audio, cache -- are NOT reflected here).
#
# Add a model's real published rate before relying on its cost estimate.
# Unknown models get `None` and the script reports tokens without a cost.
# ---------------------------------------------------------------------------
PRICING_USD_PER_1M_TOKENS = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
}


def _load_manifests(patterns):
    paths = []
    for p in patterns:
        matches = glob.glob(p)
        paths.extend(matches if matches else [p])
    if not paths:
        raise SystemExit(f"No manifests matched: {patterns}")
    manifests = []
    for p in paths:
        with open(p) as f:
            manifests.append((p, json.load(f)))
    return manifests


def _cost(model_id, prompt_tokens, completion_tokens):
    """Return (cost_usd_or_None, note)."""
    rates = PRICING_USD_PER_1M_TOKENS.get(model_id)
    if rates is None:
        return None, f"cost unknown (no rate for {model_id})"
    in_rate, out_rate = rates.get("input"), rates.get("output")
    if in_rate is None or out_rate is None:
        return None, f"cost unknown (no rate for {model_id})"
    cost = (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
    return cost, None


def _fmt_usd(x):
    return f"${x:,.4f}"


def main():
    ap = argparse.ArgumentParser(
        description="Report API spend from MMTS-Bench run manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("manifests", nargs="+", help="mmts_run_*.json manifest path(s)/glob(s)")
    args = ap.parse_args()

    manifests = _load_manifests(args.manifests)

    print("\n" + "=" * 88)
    print("API SPEND REPORT")
    print("=" * 88)

    total_calls = 0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0
    any_cost_unknown = False
    any_missing_usage = False

    for path, manifest in manifests:
        name = Path(path).name
        model_id = manifest.get("model", "(unknown model)")
        usage = manifest.get("usage")

        print(f"\n{name}")
        print(f"  model: {model_id}")

        if not usage:
            print("  usage not recorded (manifest predates usage tracking)")
            any_missing_usage = True
            continue

        calls = usage.get("calls", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tokens = usage.get("total_tokens", 0)

        cost, note = _cost(model_id, prompt_tokens, completion_tokens)

        print(f"  calls:             {calls:,}")
        print(f"  prompt tokens:     {prompt_tokens:,}")
        print(f"  completion tokens: {completion_tokens:,}")
        print(f"  total tokens:      {tokens:,}")
        if cost is None:
            print(f"  cost:              {note}")
            any_cost_unknown = True
        else:
            print(f"  est. cost:         {_fmt_usd(cost)}")
            total_cost += cost

        total_calls += calls
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_tokens += tokens

    print("\n" + "-" * 88)
    print("TOTAL")
    print(f"  calls:             {total_calls:,}")
    print(f"  prompt tokens:     {total_prompt:,}")
    print(f"  completion tokens: {total_completion:,}")
    print(f"  total tokens:      {total_tokens:,}")
    if any_cost_unknown:
        print(f"  est. cost:         {_fmt_usd(total_cost)} (partial -- some runs have no rate)")
    elif any_missing_usage and total_calls == 0:
        print("  est. cost:         unknown (no runs had usage recorded)")
    else:
        print(f"  est. cost:         {_fmt_usd(total_cost)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
