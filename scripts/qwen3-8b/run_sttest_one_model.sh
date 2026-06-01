#!/usr/bin/env bash
# Official ST-Test inference + evaluate for one checkpoint (README-style loop).
# Usage: bash scripts/qwen3-8b/run_sttest_one_model.sh /path/to/model_dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/str-py310/bin/python}"
MODEL_PATH="${1:?model_path required}"

export PYTHONPATH=.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"

MAX_TOKENS=6144
TEMPERATURE=0.2

TASKS=(
  "reasoning_etiological:data/ST-Bench/ST-Test/etiological_test.jsonl"
  "reasoning_entity:data/ST-Bench/ST-Test/entity_test.jsonl"
  "reasoning_correlation:data/ST-Bench/ST-Test/correlation_test.jsonl"
  "reasoning_forecasting:data/ST-Bench/ST-Test/forecasting_test.jsonl"
)

MODEL_NAME="$(basename "$(readlink -f "$MODEL_PATH")")"

for entry in "${TASKS[@]}"; do
  task="${entry%%:*}"
  dataset="${entry#*:}"
  exp_name="${task}-${MODEL_NAME}"
  exp_dir="exp/${exp_name}"

  echo "=== inference ${task} model=${MODEL_NAME} ==="
  "$PYTHON" inference/inference_tsmllm_vllm.py \
    --task "$task" \
    --dataset "$dataset" \
    --model_path "$MODEL_PATH" \
    --num_gpus 1 \
    --num_gpus_per_process 1 \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output_name generated_answer.json

  echo "=== evaluate ${task} model=${MODEL_NAME} ==="
  "$PYTHON" evaluation/evaluate.py \
    --task "$task" \
    --dataset "$dataset" \
    --exp_path "$exp_dir" \
    --pred_pattern generated_answer \
    --repo_root "$ROOT"
done

echo "Done ST-Test for ${MODEL_NAME}"
