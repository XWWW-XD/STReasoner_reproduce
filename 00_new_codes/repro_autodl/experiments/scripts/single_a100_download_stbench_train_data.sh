#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton

df -h / /root/autodl-tmp
/root/autodl-tmp/conda/envs/str-py310/bin/python download_dataset.py
df -h / /root/autodl-tmp
