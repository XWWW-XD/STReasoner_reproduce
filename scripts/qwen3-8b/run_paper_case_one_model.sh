#!/usr/bin/env bash
# paper_cases 4 samples: one official inference+evaluate per task/category.
# Usage: bash scripts/qwen3-8b/run_paper_case_one_model.sh /path/to/model_dir [dataset_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/str-py310/bin/python}"
MODEL_PATH="${1:?model_path required}"
DATASET_DIR="${2:-00_new_codes/repro_autodl/experiments/results/stage2.5_checkpoint_compare_paper_cases_6144/datasets}"

export PYTHONPATH=.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"

MAX_TOKENS=6144
TEMPERATURE=0.2
MODEL_NAME="$(basename "$(readlink -f "$MODEL_PATH")")"

shopt -s nullglob
files=("$DATASET_DIR"/*.jsonl)
if ((${#files[@]} == 0)); then
  echo "No paper case jsonl under ${DATASET_DIR}" >&2
  exit 1
fi

for dataset in "${files[@]}"; do
  base="$(basename "$dataset")"
  # 00_reasoning_entity_*.jsonl -> reasoning_entity
  task="$(echo "$base" | sed -n 's/^[0-9]*_\(reasoning_[^_]*\)_.*/\1/p')"
  if [[ -z "$task" ]]; then
    echo "Cannot parse task from ${base}" >&2
    exit 1
  fi
  exp_name="${task}-${MODEL_NAME}"

  echo "=== paper_case inference ${task} model=${MODEL_NAME} ==="
  "$PYTHON" inference/inference_tsmllm_vllm.py \
    --task "$task" \
    --dataset "$dataset" \
    --model_path "$MODEL_PATH" \
    --num_gpus 1 \
    --num_gpus_per_process 1 \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output_name generated_answer.json

  echo "=== paper_case evaluate ${task} model=${MODEL_NAME} ==="
  "$PYTHON" evaluation/evaluate.py \
    --task "$task" \
    --dataset "$dataset" \
    --exp_path "exp/${exp_name}" \
    --pred_pattern generated_answer \
    --repo_root "$ROOT"
done

echo "Done paper_cases for ${MODEL_NAME}"
