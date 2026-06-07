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

MODEL_PATH="${1:-./output/single_a100_qwen3_4b_stage2_smoke}"

test -d "$MODEL_PATH"
test -f data/ST-Bench/ST-RL/entity_rl.jsonl
test -f data/ST-Bench/ST-RL/etiological_rl.jsonl
test -f data/ST-Bench/ST-RL/correlation_rl.jsonl
test -f data/ST-Bench/ST-RL/forecasting_rl.jsonl

/root/autodl-tmp/conda/envs/str-py310/bin/python -m src.EasyR1.verl.trainer.main \
  config=./src/EasyR1/examples/config.yaml \
  data.train_files=./data/ST-Bench/ST-RL/etiological_rl.jsonl,./data/ST-Bench/ST-RL/forecasting_rl.jsonl,./data/ST-Bench/ST-RL/correlation_rl.jsonl,./data/ST-Bench/ST-RL/entity_rl.jsonl \
  data.val_files=./data/ST-Bench/ST-Test/etiological_test.jsonl,./data/ST-Bench/ST-Test/forecasting_test.jsonl,./data/ST-Bench/ST-Test/correlation_test.jsonl,./data/ST-Bench/ST-Test/entity_test.jsonl \
  data.prompt_key=input \
  data.ts_key=timeseries \
  data.answer_key=output \
  data.format_prompt=./src/EasyR1/examples/format_prompt/str.jinja \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.rollout_batch_size=4 \
  data.val_batch_size=4 \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.model.trust_remote_code=true \
  worker.reward.reward_function=./src/EasyR1/examples/reward_function/str.py:compute_score \
  worker.rollout.n=2 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.45 \
  trainer.experiment_name=single_a100_qwen3_4b_stage3_smoke \
  trainer.n_gpus_per_node=1 \
  trainer.find_last_checkpoint=false \
  trainer.total_epochs=1 \
  trainer.max_steps=5 \
  trainer.save_freq=5 \
  trainer.save_limit=1
