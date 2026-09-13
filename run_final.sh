#!/usr/bin/env bash
# ===========================================================================
# Final regeneration pass — every paper number, one clean run, at HEAD.
# Gemini arms only (gpt-4o-mini arms are frozen). Resilient: a failed arm is
# logged and the driver continues. Run from the repo root:  bash run_final.sh
# ===========================================================================
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/tsqa"                                   # engine package shadows any stale install
# MMTS-Bench data and the cached paraphrased stress set live outside the repo;
# override these env vars to point at wherever they sit on your machine.
export MMTS_BENCH_PATH="${MMTS_BENCH_PATH:-$REPO/mmts_bench/MMTS-BENCH}"
STRESS_SRC="${STRESS_SRC:-$REPO/mmts_bench/outputs/numeric_stress.csv}"

MMTS_OUT="$REPO/mmts_bench/outputs/final"
mkdir -p "$MMTS_OUT"
LOG="$REPO/mmts_bench/outputs/final/final_runs.runlog"   # *.runlog is git-ignored
: > "$LOG"

# --- preflight ----------------------------------------------------------------
# Every arm below spends real API budget. Check the inputs that the later arms
# need *before* the first call, so a missing benchmark checkout fails in seconds
# rather than after an hour of paid Gemini traffic.
preflight_fail=0
note () { echo "PREFLIGHT: $*" >&2; preflight_fail=1; }

if [ -z "${GEMINI_API_KEY:-}" ] && ! grep -qs 'GEMINI_API_KEY' "$REPO/.env"; then
  note "GEMINI_API_KEY is not set and no .env at the repo root defines it."
fi
if [ ! -d "$MMTS_BENCH_PATH/Benchmark" ]; then
  note "MMTS-Bench data not found at '$MMTS_BENCH_PATH' (expected a Benchmark/ subdir)."
  note "  MMTS-Bench is not redistributed; see mmts_bench/README.md, or set MMTS_BENCH_PATH."
fi
if [ ! -f "$STRESS_SRC" ]; then
  note "Paraphrased numeric stress set not found at '$STRESS_SRC'."
  note "  Regenerate it with mmts_bench/scripts/make_numeric_stress.py, or set STRESS_SRC."
fi
if [ "$preflight_fail" -ne 0 ]; then
  echo "PREFLIGHT: aborting before spending API budget." >&2
  exit 1
fi

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
arm () { # arm "<label>" <command...>
  local label="$1"; shift
  say ">>> START $label"
  local t0=$SECONDS
  "$@" >>"$LOG" 2>&1
  local rc=$?
  say "<<< DONE  $label  rc=$rc  ($((SECONDS-t0))s)"
}

TSX () { python3 tsexam/run_eval.py --config "$1" --full --tag "$2" --max-workers 16; }
MMT () { python3 mmts_bench/scripts/run_mmts_baseline.py --subset "$1" --config "$2" \
           --output-dir "$MMTS_OUT" --max-workers 16; }

say "FINAL RUN PASS  commit=$(git rev-parse --short HEAD)  branch=$(git rev-parse --abbrev-ref HEAD)"

# --- P1: Table 1 + the headline mechanism -------------------------------------
arm "tsx/baseline"        TSX baseline        baseline_final
arm "tsx/nu_ad_fix"       TSX nu_ad_fix       nu_ad_fix_final
arm "mmts/baseline_All"   MMT All baseline
arm "mmts/llm_numeric"    MMT Base llm_numeric

# --- P2: vision, gate, generalization ----------------------------------------
arm "tsx/vision_off"      TSX vision_off      vision_off_final
arm "tsx/raw_pixel"       TSX raw_pixel_vision raw_pixel_final
arm "tsx/learned_gate"    TSX learned_gate    learned_gate_final
arm "mmts/nu_ad_fix_All"  MMT All nu_ad_fix

# --- P2: three-arm numeric parser (re-score the cached paraphrased stress set)-
cp -f "$STRESS_SRC" "$MMTS_OUT/numeric_stress.csv"
arm "numeric_stress_threearm" bash -c \
  "python3 mmts_bench/scripts/run_numeric_stress.py --input '$MMTS_OUT/numeric_stress.csv' --max-workers 16 > '$MMTS_OUT/numeric_stress_threearm.txt' 2>&1"

# --- P3: agency null loops ----------------------------------------------------
arm "tsx/react_s1"        TSX react_stage1_evidence_retry react_s1_final
arm "tsx/react_s2"        TSX react_stage2_refine         react_s2_final
arm "tsx/react_s3"        TSX react_stage3_actions        react_s3_final

say "ALL ARMS COMPLETE"
