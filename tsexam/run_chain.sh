#!/usr/bin/env bash
# Run a sequence of configs back-to-back on the same paired 150-row sample.
# Each writes tsexam/outputs/<tag>/. Diffs are computed vs the baseline tag.
set -u
cd "$(dirname "$0")/.."   # repo root
NPC=${NPC:-30}
SEED=${SEED:-42}

run () {
  local cfg=$1 tag=$2
  echo "=================== $tag ($cfg) ==================="
  python3 tsexam/run_eval.py --config "$cfg" --n-per-cat "$NPC" --seed "$SEED" \
      --tag "$tag" --baseline-tag baseline 2>&1 \
      | grep -v "UserWarning\|nperseg\|HF_TOKEN\|Warning:" | grep -v "it/s\]"
}

run vision_only  vision_only
run multi_branch multi_branch
run combined      combined
echo "=================== CHAIN DONE ==================="
