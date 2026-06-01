## ST-Bench 使用方式

### 1. 下载数据：

```bash
python download_dataset.py
```

数据默认下载到 `data/ST-Bench/`。常见子目录包括：

- `ST-Align/`：对齐数据，主要用于 Stage 1。
- `ST-CoT/`：带推理链的数据，主要用于 Stage 2 CoT/SFT。
- `ST-SFT/`：普通监督微调数据。
- `ST-RL/`：RL 训练数据，Stage 3 使用。
- `ST-Test/`：测试集，推理和评估常用。
- `ST-Causal/`：因果任务数据。

### 2. 看字段映射：

`data/dataset_info.json` 是训练数据注册表。里面多数 ST-Bench 条目都使用同一套列映射：

- `prompt` -> `input`
- `response` -> `output`
- `timeseries` -> `timeseries`

这也是本次 `PaperCases.jsonl` 采用的字段风格。

### 3. 跑推理：

官方推理入口是 `inference/inference_tsmllm_vllm.py`。通常显式传入数据文件、任务和模型路径，例如：

```bash
python inference/inference_tsmllm_vllm.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --model_path Time-HD-Anonymous/STReasoner-8B \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_tokens 512
```

输出会写到 `exp/<task>-<model>/generated_answer.json`，其中关键字段是 `idx`、`question_text`、`response`、`num_tokens`。

### 4. 跑评估：

评估入口是 `evaluation/evaluate.py`。建议显式传 `--dataset`，避免默认路径指向旧的 `data/reasoning/*.jsonl`：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp/reasoning_entity-STReasoner-8B
```

选择题任务会从 `<answer>...</answer>` 中解析 A/B/C/D 并算 accuracy；forecasting 会解析数值列表并算 MAE/MAPE。

