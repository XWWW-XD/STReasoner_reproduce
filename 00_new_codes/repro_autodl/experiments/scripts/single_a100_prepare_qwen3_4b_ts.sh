#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

MODEL_DIR=base_model/Qwen3-4B-Instruct-2507
CONFIG_DIR=base_model/Config-Qwen3-4B-Instruct-2507

test -d "$MODEL_DIR"
if [[ -f "$MODEL_DIR/model-00001-of-00003.safetensors" ]]; then
  test -f "$MODEL_DIR/model-00002-of-00003.safetensors"
  test -f "$MODEL_DIR/model-00003-of-00003.safetensors"
elif [[ -f "$MODEL_DIR/model-00001-of-00004.safetensors" ]]; then
  test -f "$MODEL_DIR/model-00002-of-00004.safetensors"
  test -f "$MODEL_DIR/model-00003-of-00004.safetensors"
  test -f "$MODEL_DIR/model-00004-of-00004.safetensors"
else
  echo "missing model safetensors shards in $MODEL_DIR" >&2
  exit 1
fi
test -f "$CONFIG_DIR/config.json"

cp -rf "$CONFIG_DIR"/* "$MODEL_DIR"/

HF_HOME=/root/autodl-tmp/cache/huggingface \
TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface \
HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets \
TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch \
TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton \
/root/autodl-tmp/conda/envs/str-py310/bin/python initial_model.py \
  --model_path "$MODEL_DIR"
