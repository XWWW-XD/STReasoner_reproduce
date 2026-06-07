#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

export PATH="/root/autodl-tmp/conda/envs/str-py310/bin:${PATH}"
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions

test -d base_model/Qwen3-4B-Instruct-2507
test -f data/ST-Bench/ST-CoT/entity_cot.jsonl
test -f data/ST-Bench/ST-CoT/etiological_cot.jsonl
test -f data/ST-Bench/ST-CoT/correlation_cot.jsonl
test -f data/ST-Bench/ST-CoT/forecasting_cot.jsonl

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=1800 /root/autodl-tmp/conda/envs/str-py310/bin/deepspeed --num_gpus 1 --master_port=19912 src/train.py \
  --deepspeed ds_config/ds_config_3_offload_all.json \
  --stage sft \
  --model_name_or_path "./base_model/Qwen3-4B-Instruct-2507" \
  --dataset "entity_cot,etiological_cot,correlation_cot,forecasting_cot" \
  --interleave_probs "0.25,0.25,0.25,0.25" \
  --do_train \
  --mix_strategy "interleave_over" \
  --template "STReasoner-CoT" \
  --finetuning_type full \
  --output_dir "./output/single_a100_qwen3_4b_stage2_smoke" \
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
  --fp16 \
  --save_only_model \
  --save_safetensors False \
  --preprocessing_num_workers 4 \
  --trust_remote_code True \
  --cutoff_len 4096
