#!/usr/bin/env bash
set -euo pipefail

# Continue Stage1 from the protected checkpoint-500 model weights.
# This is model-weight continuation, not exact optimizer/scheduler-state resume,
# because the original run used --save_only_model.

cd "$(dirname "$0")/../../../.."

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-./00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/Qwen3-4B-Instruct-2507-stage1-continue-from500}"
MASTER_PORT="${MASTER_PORT:-19901}"
MAX_STEPS="${MAX_STEPS:-500}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

if [[ ! -d "$MODEL_CHECKPOINT" ]]; then
  echo "Missing MODEL_CHECKPOINT: $MODEL_CHECKPOINT" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists: $OUTPUT_DIR" >&2
  echo "Choose a fresh OUTPUT_DIR or remove it manually after backing it up." >&2
  exit 1
fi

export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"

/root/autodl-tmp/conda/envs/str-py310/bin/deepspeed \
  --num_gpus 1 \
  --master_port="${MASTER_PORT}" \
  src/train.py \
  --deepspeed 00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json \
  --stage sft \
  --model_name_or_path "${MODEL_CHECKPOINT}" \
  --dataset alignment \
  --interleave_probs 1 \
  --do_train \
  --mix_strategy interleave_over \
  --template STReasoner-Align \
  --finetuning_type full \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --timeseries_sft_lr 1e-5 \
  --warmup_ratio 0.02 \
  --num_train_epochs 0 \
  --max_steps "${MAX_STEPS}" \
  --plot_loss \
  --bf16 \
  --save_only_model \
  --save_safetensors False \
  --preprocessing_num_workers 96 \
  --trust_remote_code True \
  --cutoff_len 10000
