#!/usr/bin/env bash
# scripts/run_ablation_experiment.sh
# ============================================================================
# Run overall evaluation (MCQ + numeric head) for T-SMART ablation study.
# Evaluates baseline (additive vision), learned_gate (learned should-I-look policy),
# and vision_off (pure mathematical reasoning) across all 4 MMTS-Bench subsets.
# Subsets run sequentially; the three configs of a subset run concurrently.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                      # -> mmts_bench/

OUT="./outputs/ablation"
LOGS="$OUT/logs"
WORKERS="${WORKERS:-16}"
mkdir -p "$LOGS"

echo ">>> Ablation experiment starting $(date)  | workers=$WORKERS/config | out=$OUT"

for subset in Base InWild Match Align; do
  echo ">>> subset=$subset : launching baseline, learned_gate, and vision_off concurrently"
  for cfg in baseline learned_gate vision_off; do
    python3 scripts/run_mmts_baseline.py \
      --subset "$subset" --config "$cfg" \
      --max-workers "$WORKERS" --output-dir "$OUT" \
      > "$LOGS/${subset}_${cfg}.log" 2>&1 &
  done
  wait
  echo ">>> subset=$subset done $(date)"
done

echo ">>> All runs complete $(date)"
echo ">>> Computing paired diffs..."

# Generate paired diffs using scripts/diff_configs.py
echo "=== Paired Diff: baseline vs learned_gate ==="
python3 scripts/diff_configs.py \
  --baseline  "$OUT"/mmts_results_*_baseline_*.csv \
  --treatment "$OUT"/mmts_results_*_learned_gate_*.csv \
  --include-unscoreable \
  > "$OUT/diff_baseline_vs_learned_gate.txt" 2>&1

echo "=== Paired Diff: vision_off vs learned_gate ==="
python3 scripts/diff_configs.py \
  --baseline  "$OUT"/mmts_results_*_vision_off_*.csv \
  --treatment "$OUT"/mmts_results_*_learned_gate_*.csv \
  --include-unscoreable \
  > "$OUT/diff_vision_off_vs_learned_gate.txt" 2>&1

echo ">>> Diffs written to $OUT/diff_*.txt"
