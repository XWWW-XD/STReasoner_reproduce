#!/usr/bin/env bash
# Stage 2.5 checkpoint compare: CoT -> Align -> Qwen3-8B (user order).
# Uses official inference/inference_tsmllm_vllm.py and evaluation/evaluate.py only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PHASE="${1:-paper}"  # paper | sttest | all

prepare_paper_datasets() {
  local src="00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl"
  local out="00_new_codes/repro_autodl/experiments/results/stage2.5_checkpoint_compare_paper_cases_6144/datasets"
  /root/autodl-tmp/conda/envs/str-py310/bin/python \
    00_new_codes/repro_autodl/experiments/scripts/stage2_5_prepare_paper_cases.py \
    --source "$src" --output-dir "$out"
}

run_one() {
  local model_path="$1"
  if [[ "$PHASE" == "paper" || "$PHASE" == "all" ]]; then
    bash scripts/qwen3-8b/run_paper_case_one_model.sh "$model_path"
  fi
  if [[ "$PHASE" == "sttest" || "$PHASE" == "all" ]]; then
    bash scripts/qwen3-8b/run_sttest_one_model.sh "$model_path"
  fi
}

if [[ "$PHASE" == "paper" || "$PHASE" == "all" ]]; then
  prepare_paper_datasets
fi

# 1. CoT
if [[ -d base_model/STReasoner-8B-CoT ]]; then
  run_one base_model/STReasoner-8B-CoT
else
  echo "Skip CoT: base_model/STReasoner-8B-CoT missing" >&2
fi

# 2. Align
if [[ -d base_model/STReasoner-8B-Align ]]; then
  run_one base_model/STReasoner-8B-Align
else
  echo "Skip Align: base_model/STReasoner-8B-Align missing" >&2
fi

# 3. Qwen3-8B (initialized)
if [[ -d base_model/Qwen3-8B ]]; then
  run_one base_model/Qwen3-8B
else
  echo "Skip Qwen3-8B: base_model/Qwen3-8B missing (run download_model + initial_model)" >&2
fi

echo "Checkpoint compare phase=${PHASE} finished."
