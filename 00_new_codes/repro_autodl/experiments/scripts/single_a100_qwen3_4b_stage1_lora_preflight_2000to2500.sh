#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

MODEL_DIR=base_model/Qwen3-4B-Instruct-2507
DATA_FILE=data/ST-Bench/ST-Align/alignment_train.jsonl
SOURCE_DIR=output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state
RESUME_DIR="${SOURCE_DIR}/checkpoint-500"
NEXT_OUTPUT_DIR=output/single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state
RESUME_SCRIPT=00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh

printf '[preflight] cwd=%s\n' "$(pwd)"
printf '[preflight] checking model/data/checkpoint files\n'

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/modeling_qwen3_ts.py"
test -f "${MODEL_DIR}/processing_qwen3_ts.py"
test -f "${DATA_FILE}"
test -f "${RESUME_DIR}/adapter_model.bin"
test -f "${RESUME_DIR}/adapter_config.json"
test -f "${RESUME_DIR}/optimizer.pt"
test -f "${RESUME_DIR}/scheduler.pt"
test -f "${RESUME_DIR}/rng_state.pth"
test -f "${RESUME_DIR}/trainer_state.json"
test -f "${RESUME_SCRIPT}"

printf '[preflight] resume checkpoint files are present\n'

if [ -e "${NEXT_OUTPUT_DIR}" ]; then
  printf '[preflight] next output dir already exists: %s\n' "${NEXT_OUTPUT_DIR}"
  printf '[preflight] stop here to avoid mixing old and new outputs. Move or inspect it before training.\n'
  exit 2
fi

bash -n "${RESUME_SCRIPT}"

printf '[preflight] disk status\n'
df -h / /root/autodl-tmp

printf '[preflight] gpu status\n'
nvidia-smi

printf '[preflight] ready_to_train=ok\n'
printf '[preflight] next command: %s\n' "${RESUME_SCRIPT}"
printf '[preflight] expected output: %s\n' "${NEXT_OUTPUT_DIR}"
