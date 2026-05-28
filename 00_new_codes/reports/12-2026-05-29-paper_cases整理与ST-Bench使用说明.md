# 12-2026-05-29 paper_cases 整理与 ST-Bench 使用说明

## 本次整理结果

`00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/` 已整理为两个数据文件：

- `PaperCases.json`
- `PaperCases.jsonl`

原来的 `streasoner_paper_cases_from_paper.json`、`paper_cases_manifest.json`、`paper_cases_matched.jsonl` 已合并。保留 4 条论文 case 对应的完整 ST-Test 样本，删除重复或不直接参与推理评估的说明字段。

`PaperCases.jsonl` 用于贴近 ST-Bench 原始数据和现有推理脚本，每行一条样本；`PaperCases.json` 是同一批样本的结构化版本，使用 `metadata` + `cases`，方便人工查看、整体加载和写报告。

## 为什么这样合并

仓库内 ST-Bench 主线数据基本都按 JSONL 组织，每行是一条样本，核心字段是 `input`、`timeseries`、`output`。这个映射也写在 `data/dataset_info.json` 里：训练和测试时把 `input` 当 prompt，把 `output` 当 response/gold，把 `timeseries` 当时序输入。因此保留 `PaperCases.jsonl` 作为直接可用的数据文件，同时保留一个 `PaperCases.json` 作为便于阅读的汇总文件。

旧文件里的节点、图结构、选项、问题等信息已经包含在 `input` 字符串中；完整数值序列已经包含在 `timeseries` 中。因此保留额外拆解字段会造成重复，也容易和原始 ST-Bench 格式不一致。

## `PaperCases.json` 结构说明

- `metadata`：数据集级别说明，不参与模型输入。包括样本数、来源数据集、任务列表、核心字段和溯源字段。
- `cases`：样本列表。每个元素与 `PaperCases.jsonl` 中的一行完全对应。

## `PaperCases.jsonl` / `cases` 字段说明

- `input`：模型看到的完整文本 prompt。里面包含节点占位符 `<ts><ts/>`、`Graph Structure`、问题和选项。
- `timeseries`：与 `input` 中节点占位符对应的时序数值数组。外层列表按 Node 顺序排列，内层列表是该节点的时间序列。
- `output`：标准答案。选择题通常是 `<answer>A</answer>` 这类格式；forecasting 是数值列表字符串。
- `category`：样本类别，取值包括 `etiological`、`entity`、`correlation`、`forecasting`。
- `sample_id`：本地整理后的唯一样本 ID，包含论文表格和原始行号信息。
- `task`：任务名。当前与 `category` 相同，保留它是为了兼容现有 SmartTest / paper case runner 中按 `task` 分流的逻辑。
- `source_file`：样本来自 ST-Bench 的哪个原始测试文件，例如 `ST-Test/entity_test.jsonl`。
- `original_line_index`：样本在原始 ST-Test 文件中的零基行号，方便回溯官方数据。
- `paper_case_id`：论文 case 的稳定 ID，例如 `appendix_h_table7_entity`。
- `paper_location`：论文中的位置，例如 `Appendix H, Table 7`。
- `paper_note`：整理备注。前三个 case 是严格匹配；forecasting case 说明了 ST-Test gold output 和论文展示的 STReasoner prediction 不同，本文件保留 ST-Test gold output 以便可复现实验评估。

## 删除的字段

- `case_count`、`matched_count`：可以由 JSONL 行数得到。
- `nodes`、`graph_edges`、`question`、`options`：这些内容已经并入 `input`。
- `matching_basis`、`matched`：整理阶段的人工匹配证据，不参与模型输入或评估。
- `length_metrics`：只是近似字符长度统计，不是官方 token 统计。
- `selection_seed`、`selection_note`：抽样说明，不影响样本本身。
- 论文摘录中的 `<ts><ts/>` 占位版本：没有完整数值序列，不能直接用于真实推理；已用匹配到的 ST-Test 完整样本替代。

## ST-Bench 常见使用方式

1. 下载数据：

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

2. 看字段映射：

`data/dataset_info.json` 是训练数据注册表。里面多数 ST-Bench 条目都使用同一套列映射：

- `prompt` -> `input`
- `response` -> `output`
- `timeseries` -> `timeseries`

这也是本次 `PaperCases.jsonl` 采用的字段风格。

3. 跑推理：

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

4. 跑评估：

评估入口是 `evaluation/evaluate.py`。建议显式传 `--dataset`，避免默认路径指向旧的 `data/reasoning/*.jsonl`：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp/reasoning_entity-STReasoner-8B
```

选择题任务会从 `<answer>...</answer>` 中解析 A/B/C/D 并算 accuracy；forecasting 会解析数值列表并算 MAE/MAPE。

## 使用 `PaperCases.jsonl` 的注意点

`PaperCases.jsonl` 是混合任务文件，包含 etiological、entity、correlation、forecasting 各 1 条。它适合给现有 paper-case/SmartTest 风格 runner 按行读取、按 `task` 分流。

如果直接使用官方 `inference/inference_tsmllm_vllm.py` 和 `evaluation/evaluate.py`，它们一次只接收一个 `--task`，更适合按任务拆开或单条运行。不要把混合任务文件当成某一个单独 `reasoning_entity` 或 `reasoning_forecasting` 测试集整体评估。

本次只整理数据文件和说明文档，没有重新运行模型推理。
