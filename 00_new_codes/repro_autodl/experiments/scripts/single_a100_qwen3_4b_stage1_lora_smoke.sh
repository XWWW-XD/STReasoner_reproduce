#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

export PATH="/root/autodl-tmp/conda/envs/str-py310/bin:${PATH}"
export PYTHONPATH=.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true

MODEL_DIR=base_model/Qwen3-4B-Instruct-2507
DATA_FILE=data/ST-Bench/ST-Align/alignment_train.jsonl
OUTPUT_DIR=output/single_a100_qwen3_4b_stage1_lora_smoke

test -d "${MODEL_DIR}"
test -f "${DATA_FILE}"

CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/conda/envs/str-py310/bin/python src/train.py \
  --stage sft \
  --model_name_or_path "./${MODEL_DIR}" \
  --flash_attn fa2 \
  --dataset "alignment" \
  --interleave_probs "1" \
  --do_train \
  --mix_strategy "interleave_over" \
  --template "STReasoner-Align" \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir "./${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_steps 10 \
  --learning_rate 1e-5 \
  --timeseries_sft_lr 1e-5 \
  --warmup_ratio 0.02 \
  --num_train_epochs 0 \
  --max_steps 10 \
  --plot_loss \
  --bf16 \
  --save_only_model \
  --save_safetensors false \
  --report_to none \
  --preprocessing_num_workers 4 \
  --trust_remote_code true \
  --cutoff_len 4096
