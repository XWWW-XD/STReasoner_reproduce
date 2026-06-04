#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/STReasoner_reproduce

echo "== GPU =="
nvidia-smi

echo
echo "== Disk =="
df -h /root/autodl-tmp

echo
echo "== Python =="
/root/autodl-tmp/conda/envs/str-py310/bin/python --version

echo
echo "== Required local config =="
test -f base_model/Config-Qwen3-4B-Instruct-2507/config.json
test -f initial_model.py
test -f src/train.py
test -f src/EasyR1/examples/config.yaml
echo "local config ok"

echo
echo "== Dataset presence =="
for path in \
  data/ST-Bench/ST-Align/alignment_train.jsonl \
  data/ST-Bench/ST-CoT/entity_cot.jsonl \
  data/ST-Bench/ST-CoT/etiological_cot.jsonl \
  data/ST-Bench/ST-CoT/correlation_cot.jsonl \
  data/ST-Bench/ST-CoT/forecasting_cot.jsonl \
  data/ST-Bench/ST-RL/entity_rl.jsonl \
  data/ST-Bench/ST-RL/etiological_rl.jsonl \
  data/ST-Bench/ST-RL/correlation_rl.jsonl \
  data/ST-Bench/ST-RL/forecasting_rl.jsonl \
  data/ST-Bench/ST-Test/entity_test.jsonl \
  data/ST-Bench/ST-Test/etiological_test.jsonl \
  data/ST-Bench/ST-Test/correlation_test.jsonl \
  data/ST-Bench/ST-Test/forecasting_test.jsonl
do
  if [[ -f "$path" ]]; then
    printf "ok      %s\n" "$path"
  else
    printf "missing %s\n" "$path"
  fi
done

echo
echo "== Qwen3-4B model =="
if [[ -d base_model/Qwen3-4B-Instruct-2507 ]]; then
  find base_model/Qwen3-4B-Instruct-2507 -maxdepth 1 -type f | sort
else
  echo "missing base_model/Qwen3-4B-Instruct-2507"
fi
