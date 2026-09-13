#!/usr/bin/env bash
# scripts/run_generalization.sh
# ============================================================================
# Pre-registered paired generalization experiment (see ../PREREGISTRATION.md).
# Runs baseline vs the FROZEN nu_ad_fix config across all 4 MMTS-Bench subsets,
# MCQ rows only, then prints the paired diff. Subsets run sequentially; the two
# configs of a subset run concurrently (bounded ~48 workers => well under the
# 4000 RPM cap). Reproducible: same rows, same seed, temp 0.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                      # -> mmts_bench/

OUT="./outputs/generalization"
LOGS="$OUT/logs"
WORKERS="${WORKERS:-24}"
mkdir -p "$LOGS"

echo ">>> Generalization run starting $(date)  | workers=$WORKERS/config | out=$OUT"

for subset in Base InWild Match Align; do
  echo ">>> subset=$subset : launching baseline + nu_ad_fix concurrently"
  for cfg in baseline nu_ad_fix; do
    python3 scripts/run_mmts_baseline.py \
      --subset "$subset" --config "$cfg" --mcq-only \
      --max-workers "$WORKERS" --output-dir "$OUT" \
      > "$LOGS/${subset}_${cfg}.log" 2>&1 &
  done
  wait
  echo ">>> subset=$subset done $(date)"
done

echo ">>> all runs complete $(date)"
echo ">>> per-subset paired diffs + pooled diff follow"

# Pooled diff across all subsets (scoreable-only is the default).
python3 scripts/diff_configs.py \
  --baseline  "$OUT"/mmts_results_*_baseline_*.csv \
  --treatment "$OUT"/mmts_results_*_nu_ad_fix_*.csv \
  | tee "$OUT/diff_pooled.txt"

echo ">>> wrote $OUT/diff_pooled.txt"
