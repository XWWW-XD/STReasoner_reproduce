# ST-Bench 使用方式、子集规范与 Stage 2.2 复用

---

## 1. ST-Bench 下载与目录

### 1.1 下载数据

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

### 1.2 训练侧字段映射（dataset_info.json）

`data/dataset_info.json` 是训练数据注册表。里面多数 ST-Bench 条目都使用同一套列映射：

- `prompt` -> `input`
- `response` -> `output`
- `timeseries` -> `timeseries`

这也是本次 `PaperCases.jsonl` 采用的字段风格。

---

## 2. 抽取子集 JSONL 规范

### 2.1 输入：ST-Bench 风格 JSONL（唯一权威格式）

每行一条样本，**推理最少字段**：

| 字段 | 必填 | 说明 |
|---|---|---|
| `input` | 是 | 含 `<ts><ts/>` 占位符、图结构、题干 |
| `timeseries` | 是 | 与占位符数量一致的嵌套数值列表 |
| `output` | 是 | gold；选择题为 `<answer>X</answer>`，forecasting 为数值列表字符串 |
| `category` | 是 | `entity` / `etiological` / `correlation` / `forecasting`；混合 JSONL 分流字段 |

**次要字段**（不参与模型输入，便于报告）：

- `sample_id`
- `source_file`、`original_line_index`
- `paper_case_id`（仅paper_cases需要）

**不要**在 JSONL 里重复拆 `question`/`options`（已在 `input` 中）；不要改 `input` 去手写 Output Format——格式后缀由推理脚本追加。

### 2.2 两种输入文件

输入：
`json`（`metadata` + `cases`）仅供人工阅读；**跑实验只用 `.jsonl`**。

---

## 3. 官方 ST-Test 推理与评估

### 3.1 跑推理

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

### 3.2 跑评估（单任务 smoke）

评估入口是 `evaluation/evaluate.py`。建议显式传 `--dataset`，避免默认路径指向旧的 `data/reasoning/*.jsonl`：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp/reasoning_entity-STReasoner-8B
```

选择题任务会从 `<answer>...</answer>` 中解析 A/B/C/D 并算 accuracy；forecasting 会解析数值列表并算 MAE/MAPE。

### 3.3 输出：官方 ST-Test 全量

| 文件 | 路径 |
|---|---|
| 预测 | `exp/<exp_name>/generated_answer.json`（数组，字段 `idx`/`response`/`num_tokens`） |
| 指标 | 同目录或 `evaluation/evaluate.py` 打印；可另存 `evaluation_metrics.json` |

全量 ST-Test 正式实验评估示例：

```bash
PYTHONPATH=. python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp/sttest_full_entity_6144
```

---

## 4. Stage 2.2 小集输出

输出目录：`C:\Users\HUAWEI\Downloads\temp_git_clone\STReasoner_reproduce\00_new_codes\repro_autodl\results\<实验阶段+实验名>`

| 文件 | 内容 |
|---|---|
| `predictions.jsonl` | 每行一条；核心字段 `sample_id`、`category`、`official_task`、`response`（raw）、`gold_answer`、`official_metrics`（或内嵌 evaluate 结果）、`format_prompt_applied`、Run 字段（GPU/耗时） |
| `summary.json` | `run-all` 时各样本 summary 聚合 |
| `run.log` | 命令、环境、逐样本日志 |

**禁止**：评测前改写 `response`、按 sample 硬编码 parser、删失败样例。

---

## 5. 何时用 Stage 2.2，何时用官方脚本

| 场景 | 推荐入口 | 原因 |
|---|---|---|
| paper_cases / SmartTest / 任意 **混合 task** 小 JSONL | `stage2_2_run_paper_cases.py` | 按行读、按 `category` 分流、一次加载、`jsonl` 输出 |
| **全量** ST-Test 某一类任务（千条级） | `inference/inference_tsmllm_vllm.py` | 作者主路径、vLLM 吞吐、单 `--task` |
| 附录 4 条与 ST-Test 同 index 对比 | 两者都可；对比实验应 **都开 format-prompt** | 验证 HF vs vLLM 差异，而非 prompt 差异 |

官方脚本**不能**直接替代 Stage 2.2 的全部能力：

- 一次只接受一个 `--task`（混合 JSONL 需按 task 拆文件或逐条跑）；
- 输出是单个 `generated_answer.json`，不是按 sample 增量 `jsonl`；
- AutoDL 上 Stage 2.2 的 HF 单卡路径已调通 merge patch，保留有价值。

**结论：改动小，不新写脚本**——扩展 Stage 2.2 + 全量仍走官方即可。

---

## 6. 复现行动清单（更新版）

1. **已做**：Stage 2.2 默认 `--format-prompt true`；evaluate 收严为 tag-first（与报告 13、19 一致）。
2. **待跑**：用新默认重跑 paper_cases 4 条，看 **evaluate coverage / accuracy** 是否与 ST-Test 同 index 接近。
3. **报告写法**：**Run 诊断**（链路/资源）+ **evaluate 指标**（含 coverage）；附件可选 raw 标签计数，不另算一套分。
4. **全量 ST-Test**：继续以 `exp/sttest_full_*_6144/` 为准，不重复烧卡。
5. **不再做**：放宽 evaluate、用 `Answer: X` 兜底、或维护 Strict/Official 双层评测。

---

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `inference/prompt.json` | 官方 Output Format 后缀 |
| `evaluation/evaluate_qa.py` | tag-first 解析（已收严） |
| `stage2_2_run_paper_cases.py` | 小集/混合集 HF 推理（已加后缀） |
| `inference/inference_tsmllm_vllm.py` | 全量 ST-Test vLLM 推理 |
| `19-2026-05-30-evaluate与raw格式差异及paper_cases复现建议.md` | 问题根因与 ST-Test 对比 |
