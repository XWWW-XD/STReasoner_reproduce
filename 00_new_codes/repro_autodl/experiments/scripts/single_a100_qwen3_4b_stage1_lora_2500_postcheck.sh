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

ADAPTER_DIR=output/single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state
LOG_DIR=00_new_codes/repro_autodl/experiments/logs
RESULT_DIR=00_new_codes/repro_autodl/experiments/results
PY=/root/autodl-tmp/conda/envs/str-py310/bin/python
LOAD_CHECK=00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_adapter_load_check.py
PROBE=00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_adapter_probe.py

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

test -d "${ADAPTER_DIR}"
test -f "${ADAPTER_DIR}/adapter_model.bin"
test -f "${ADAPTER_DIR}/adapter_config.json"
test -f "${ADAPTER_DIR}/train_results.json"
test -f "${ADAPTER_DIR}/trainer_state.json"

printf '[postcheck] checking latest checkpoint state files if checkpoint exists\n'
LATEST_CKPT="$(find "${ADAPTER_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -1 || true)"
if [ -n "${LATEST_CKPT}" ]; then
  test -f "${LATEST_CKPT}/optimizer.pt"
  test -f "${LATEST_CKPT}/scheduler.pt"
  test -f "${LATEST_CKPT}/rng_state.pth"
  test -f "${LATEST_CKPT}/trainer_state.json"
  printf '[postcheck] latest_checkpoint=%s\n' "${LATEST_CKPT}"
else
  printf '[postcheck] no checkpoint-* directory found under %s\n' "${ADAPTER_DIR}"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOAD_LOG="${LOG_DIR}/stage1_lora_2500steps_from2000_save_state_adapter_load_check_${TS}.log"
STAGE1_ADAPTER_DIR="/root/autodl-tmp/STReasoner_reproduce/${ADAPTER_DIR}" \
  "${PY}" "${LOAD_CHECK}" 2>&1 | tee "${LOAD_LOG}"

TS="$(date +%Y%m%d_%H%M%S)"
PROBE30_LOG="${LOG_DIR}/stage1_lora_2500steps_from2000_save_state_probe30_${TS}.log"
export STAGE1_ADAPTER_DIR="/root/autodl-tmp/STReasoner_reproduce/${ADAPTER_DIR}"
export STAGE1_SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9,71,72,73,74,75,76,77,78,79,80,121,122,123,124,125,126,127,128,129,130"
export STAGE1_MAX_NEW_TOKENS=32
export STAGE1_PROBE_OUT="/root/autodl-tmp/STReasoner_reproduce/${RESULT_DIR}/stage1_lora_2500steps_from2000_save_state_probe30_${TS}.jsonl"
"${PY}" "${PROBE}" 2>&1 | tee "${PROBE30_LOG}"

TS="$(date +%Y%m%d_%H%M%S)"
TEMPORAL_LOG="${LOG_DIR}/stage1_lora_2500steps_from2000_save_state_temporal_balanced_probe40_${TS}.log"
export STAGE1_ADAPTER_DIR="/root/autodl-tmp/STReasoner_reproduce/${ADAPTER_DIR}"
export STAGE1_SAMPLE_INDICES="0,4,8,11,14,1,5,9,12,15,2,6,10,13,16,3,31,56,61,66,7,35,57,62,67,20,48,58,63,68,24,52,60,65,70,59,64,69,498,503"
export STAGE1_MAX_NEW_TOKENS=32
export STAGE1_PROBE_OUT="/root/autodl-tmp/STReasoner_reproduce/${RESULT_DIR}/stage1_lora_2500steps_from2000_save_state_temporal_balanced_probe40_${TS}.jsonl"
"${PY}" "${PROBE}" 2>&1 | tee "${TEMPORAL_LOG}"

printf '[postcheck] adapter_load_log=%s\n' "${LOAD_LOG}"
printf '[postcheck] probe30_log=%s\n' "${PROBE30_LOG}"
printf '[postcheck] temporal_probe40_log=%s\n' "${TEMPORAL_LOG}"
printf '[postcheck] postcheck=ok\n'
