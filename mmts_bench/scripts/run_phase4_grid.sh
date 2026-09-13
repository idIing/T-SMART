#!/usr/bin/env bash
# Phase-4 fresh paired grid: {baseline, agentic_tsmart} x {Base,InWild,Match,Align}.
# Both configs of a subset run concurrently; subsets sequential. MCQ-only (honest OA).
# Drift-free: baseline is run FRESH in the same session as agentic_tsmart.
set -uo pipefail
cd "$(dirname "$0")/.."                      # -> mmts_bench/
OUT="./outputs/phase4"
LOGS="$OUT/logs"
WORKERS="${WORKERS:-24}"
mkdir -p "$LOGS"
echo ">>> Phase-4 MMTS grid starting $(date) | workers=$WORKERS/config | out=$OUT"
for subset in Base InWild Match Align; do
  echo ">>> subset=$subset : baseline + agentic_tsmart concurrently $(date)"
  for cfg in baseline agentic_tsmart; do
    python3 scripts/run_mmts_baseline.py \
      --subset "$subset" --config "$cfg" --mcq-only \
      --max-workers "$WORKERS" --output-dir "$OUT" \
      > "$LOGS/${subset}_${cfg}.log" 2>&1 &
  done
  wait
  echo ">>> subset=$subset done $(date)"
done
echo ">>> all MMTS runs complete $(date)"
