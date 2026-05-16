#!/usr/bin/env bash
set -e

PROJECT_ROOT="/kaggle/working/STReasoner_reproduce"

echo "==> Entering project root: ${PROJECT_ROOT}"
cd "$PROJECT_ROOT"

echo "==> Setting Hugging Face cache paths"
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_CACHE=/kaggle/working/hf_cache/transformers
export HF_DATASETS_CACHE=/kaggle/working/hf_cache/datasets

echo "==> Creating Hugging Face cache directories"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "==> Generating requirements_no_flash.txt from requirements.txt"
grep -v -i "flash" requirements.txt > requirements_no_flash.txt

echo "==> Installing Python dependencies without flash_attn"
pip install -r requirements_no_flash.txt

echo "==> Upgrading bitsandbytes"
pip install -U bitsandbytes

echo "==> Running Kaggle environment check"
python repro_kaggle/scripts/check_kaggle_env.py

echo "==> Kaggle T4 setup finished"
