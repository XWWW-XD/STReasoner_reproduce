# 08 ST-Align-append 数据生成追加计划

日期：2026-06-14  

目标：在不覆盖原 ST-Align 的前提下，追加一个 `ST-Align-append` 数据集，用于后续 STReasoner-Align 模型 LoRA 微调

重点问题：模型没有被教会从 `<ts>` embedding 复述指定 node/window 原始数值

其他条件：尽量不破坏原模型已经学到的 ST-Align 短答、图结构和下游任务格式。

---

## 0. 结论先行

推荐做一个独立追加集：

```text
data/ST-Bench/ST-Align-append/alignment_append_train.jsonl
data/ST-Bench/ST-Align-append/alignment_append_test.jsonl
data/ST-Bench/ST-Align-append/alignment_append_hard_probe.jsonl
data/ST-Bench/ST-Align-append/manifest.json
```

核心原则：

1. **不覆盖原始 ST-Align**：`alignment_train.jsonl`、`alignment_test.jsonl` 只读不改。
2. **不把 ST-Test hard cases 放进训练**：报告 37/39/43/44 里的复述错误样本只用来做 hard probe 和采样设计依据，避免污染后续 ST-Test benchmark。
3. **append 样本从数据合成管道生成**：主源应是 `data_generation` 管线产出的 complete scenario / simulation data，而不是从现有 `alignment_train.jsonl` 二次派生。现有 ST-Align jsonl 只作为格式、分布、防重和基线评测参考。
4. **所有新任务都必须依赖 `<ts>`**：不再新增 graph-only 或纯 metadata lookup 题。
5. **训练用短答案，不做 CoT**：Stage1/append 仍用 `STReasoner-Align`，输出短 JSON list、数字、枚举或小 JSON object，避免在 LoRA 阶段把模型推向长推理风格。
6. **后续 LoRA 以 append 为主**：因为后续只做 LoRA，不再全参训练，原 ST-Align 只作为 replay 防遗忘。建议第一版 LoRA 采样比例为 `原 alignment replay:alignment_append = 0.20:0.80`。其中原 20% 必须按问题类型分层抽样，不直接按原文件顺序或全量随机抽样。

---

## 1. 依据与新增事实

当前环境重新核查的 ST-Align 事实：

```text
alignment_train.jsonl 行数：194,212
alignment.jsonl 与 alignment_train.jsonl 内容相同
alignment_train.jsonl sha256：a18fb2e3e7eb0c510fe5332d02446dface32f63ce024596a71e097522764203a
alignment_test.jsonl 行数：40,512
alignment_test.jsonl sha256：879cb969e1aad0d10c574b4f1ab87cc45bdbe247c466bb23d4726923145d5091
当前 alignment_train.jsonl 字段：input / timeseries / output，无 category 字段
```

当前 `alignment_train.jsonl` 的题型粗统计：

| 题型 | 条数 |
| --- | ---: |
| graph_direct | 45,361 |
| graph_indirect | 45,361 |
| sinusoidal A / omega / phi | 各 16,059 |
| drift_type | 7,793 |
| baseline | 7,793 |
| kappa | 7,782 |
| node_type | 6,217 |
| edge_lag | 6,121 |
| sigma | 5,842 |
| edge_mod / effective | 各 5,705 |
| lambda | 2,354 |
| other | 1 |

注意：02-07 报告里的旧统计是 153,700 行版本，方向仍成立，但具体比例以后必须以当前 194,212 行版本重新生成 artifact 为准。

---

## 2. 要解决的问题

### 2.1 report06 核心问题

报告 06 的结论是：Stage1 训练后不能稳定复述 raw time series，因为：

- prompt 中没有明文 raw 数值，只有 `<ts><ts/>`；
- `processing_qwen3_ts.py` 会把每条序列做 `sp_encoding`，再由 `TimeSeriesEmbedding` 以 `patch_size=8` 进入 MLP；
- 现有 ST-Align gold 是 yes/no、类别、SDE 参数或 metadata，**没有任何题要求输出某个 node/window 的逐点原始数值列表**；
- Stage2/Stage3 的 `<think>` 数值复述不直接进 loss。

所以 append 的第一目标不是让模型“完美可逆解码整条序列”，而是增加受控的、小窗口、可评分的逐点读取监督，让 LoRA 至少学会：

```text
给定同一套 <ts> 输入和 Graph Structure，
按明确 node + window + index 口径，
输出该窗口的 raw values / point values / local statistics。
```

### 2.2 复述错误证据必须进入设计

代码库里已有复述错误证据：

```text
00_new_codes/reports/37-MLP编码器边界分析.md
00_new_codes/reports/39-MLP复述四任务分析.md
00_new_codes/reports/43-run2-run3复述错误三次对比.md
00_new_codes/reports/44-exp_STReasoner-8B-MLP复述与6144对照.md
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/*.json
```

关键口径来自 `find_bad_cases.py`：

- 只解析 `<think>` 中明确的 `Node k [a-b]: ...` 或 `Node k (steps a-b): ...` 数值列表；
- 与 ST-Test 同 idx、同 node、同 window 的 raw `timeseries` 对齐；
- 同时尝试 0-based 和 1-based，对齐后取误差更小的一种；
- `max_abs_diff_between_stated_and_raw > 0.01` 才算 mismatch。

代表 hard cases：

| 来源 | 错误形态 |
| --- | --- |
| forecasting idx 19 | Node 1 `[27-32]` raw `[425.44, 415.47, 407.05, 415.2, 425.13, 118.85]` 被复述成 `[1550.0, 188.64, 1550.0, 115.54, 1257.01, 118.85]` |
| forecasting idx 179 | Node 2 `[62-67]` raw `[756.59, 736.69, 712.12, 676.66, 642.34, 607.78]` 被复述成约 `[121.71, 122.12, ...]` |
| correlation idx 480/503 | 跨节点推理时把高波动 Node 4 窗口复述成近似平线 `370.xx` |

这些样本不能直接进训练，但它们告诉 append 应覆盖：

- 6 到 13 步左右的窗口；
- 短序列和中等序列；
- 0/clip、尖峰、平台、剧烈动态范围；
- forecasting/correlation 风格的多节点上下游窗口；
- 容易出现 0-based/1-based 混淆的窗口表达。

---

## 3. 数据源边界

### 3.1 训练生成源

append 训练集应重新走数据合成管道，而不是从最终 `alignment_train.jsonl` 复制场景再改问题。推荐复用现有生成链路：

```text
data_generation/run_pipeline.py
data_generation/generate_alignment_QA.py
data_generation/prompts/qa_generation/alignment_templates.json
```

使用方式：

- 在生成或读取 complete scenario / simulation data 后，直接从 `agent5_simulation_data` 取 raw `timeseries`；
- 复用 `build_input_prefix(...)` 风格生成 `Node k time series ... <ts><ts/>; Graph Structure: ...` 前缀；
- 在同一 complete data 上追加 raw-window / point / local-stat / event / comparison QA；
- 保留原 `input / timeseries / output` 三字段，额外写入 append metadata 方便排查；
- 对生成出的 scenario 做去重和切分，禁止同一 scenario 同时进入 train 和 test。

现有 ST-Align 文件的角色：

```text
data/ST-Bench/ST-Align/alignment_train.jsonl
data/ST-Bench/ST-Align/alignment_test.jsonl
```

只用于：

- 对齐字段格式、prompt 前缀和模板语气；
- 估计原数据分布，避免 append 分布过窄；
- 做 LoRA 混训和防遗忘评测；
- 做防重检查，避免新生成 scenario 与已有 ST-Align 完全重复。

可参考但不直接照搬：

```text
data/ST-Bench/ST-SFT/*.jsonl
data/ST-Bench/ST-CoT/*.jsonl
```

用途：

- 提取 forecasting/correlation 的 window 长度、问题措辞和下游任务风格；
- 不把 CoT 输出作为 append 训练目标。

### 3.2 评测源

append test 建议同样由数据合成管道生成，并与 append train 按 scenario_id 切分。现有 ST-Align test 不作为 append test 的直接来源，只作为原能力防遗忘评测集：

```text
data/ST-Bench/ST-Align/alignment_test.jsonl
```

hard probe 来自：

```text
data/ST-Bench/ST-Test/*.jsonl
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/*.json
```

限制：

- ST-Test 诊断样本只写入 `alignment_append_hard_probe.jsonl`；
- 不注册为训练 dataset；
- 不参与 LoRA 训练；
- 只用于确认报告 39/43/44 的 hard windows 在 append-LoRA 后是否改善。

---

## 4. 追加集任务族设计

第一版建议 `alignment_append_train.jsonl` 约 50,000 条；`alignment_append_test.jsonl` 约 4,000 条；hard probe 约 600 到 1,000 条。

### 4.1 raw window 复述，40%

目标：直接补 report06 缺口。

模板：

```text
Using 0-based indices, output the raw values of node {node_id} from index {start} to {end} inclusive. Return only a JSON list rounded to 2 decimals.
```

输出：

```json
[425.44, 415.47, 407.05, 415.2, 425.13, 118.85]
```

设计细节：

- window 长度优先 `4/6/8/10/12/16`；
- 至少 70% 为内部窗口，避免只问 `left/right`；
- 重点覆盖跨 patch window，例如 `start % 8 in {6,7}` 或 `end % 8 in {0,1}`；
- 覆盖 smooth、spike、zero/clip、plateau、high range、low range；
- 不输出解释，不使用 `<think>`。

### 4.2 index point extraction，15%

目标：训练精确索引，缓解 0-based/1-based 混淆。

模板：

```text
Using 0-based indices, what are the raw values of node {node_id} at indices [{i1}, {i2}, {i3}, {i4}]? Return only a JSON list in the same order.
```

输出：

```json
[340.75, 334.69, 328.09, 325.3]
```

设计细节：

- 不采样 `0` 和 `L-1`，因为 processor 文本里有 `left/right`；
- 采样同一 patch 内点、跨 patch 点、稀疏点三类；
- 可加入 10% 的显式 1-based prompt，但必须写清楚：

```text
Using 1-based time steps, where time step 1 is the first value, ...
```

不要使用模糊的 `time steps a-b`。

### 4.3 local statistics，15%

目标：让模型不仅会逐点列表，也会读窗口统计，连接 ST-Test forecasting/correlation 常见推理语言。

模板：

```text
Using 0-based indices, compute the mean, minimum, maximum, and delta (last minus first) of node {node_id} from index {start} to {end} inclusive. Return a JSON object with keys mean, min, max, delta.
```

输出：

```json
{"mean": 402.76, "min": 118.85, "max": 425.44, "delta": -306.59}
```

防泄漏规则：

- 不问整条序列的 min/max，因为 `sp_encoding` 文本里有全局 `max/min`；
- 若 local min/max 等于该 node 全局 min/max，该样本只允许进入 raw window 复述，不进入 local statistics；
- mean/delta 保留两位小数。

### 4.4 event / shape / saturation，15%

目标：覆盖 report07 和 report39/44 里常见的 0、clip、尖峰、平台、尺度错读问题。

模板示例：

```text
Using 0-based indices, how many zero values appear in node {node_id} from index {start} to {end} inclusive? Return only the integer count.
```

```text
Using 0-based indices, classify the local trend of node {node_id} from index {start} to {end} inclusive as one of: increasing, decreasing, flat, mixed. Return only the label.
```

```text
Using 0-based indices, what is the first index in node {node_id} from {start} to {end} where the value is greater than {threshold}? Return the index, or none.
```

输出：

```text
3
mixed
67
```

设计细节：

- `threshold` 从窗口内部分位数派生，不使用全局 `max/min`；
- `flat` 需要定义，例如 `max-min <= 0.05 * max(1, abs(mean))`；
- spike/clip 类问题要覆盖 ST-Align 中的 0/350、1250/1850 等平台，但不要让某个常数成为唯一答案。

### 4.5 cross-node local comparison，15%

目标：轻量连接 Stage2 correlation/etiological，但仍保持短答案，避免 CoT。

模板：

```text
Using 0-based indices, compare node {a} and node {b} from index {start} to {end}. Which node has the larger mean value? Return only node {a}, node {b}, or equal.
```

```text
Using 0-based indices, compare node {a} and node {b} from index {start} to {end}. Which node has the larger range (max minus min)? Return only node {a}, node {b}, or equal.
```

输出：

```text
node 2
node 4
equal
```

设计细节：

- 优先采样图上相邻节点、二跳节点、无边节点三类；
- 不用 graph-only 问题；
- 每条答案必须由两段 raw `timeseries` 决定；
- 这类样本不要超过 15%，避免把 Stage1 append 变成 Stage2 任务。

---

## 5. 采样与去重策略

### 5.1 场景去重

现有 ST-Align 是同一场景几百条 QA。append 必须先建 unique scenario pool。

建议指纹：

```text
scenario_id = sha1(
  normalized_prefix_before_question
  + graph_text
  + node_count
  + node_lengths
  + sha1(timeseries rounded to 2 decimals)
)
```

生成时按 `scenario_id` 分层：

- 每个场景最多生成 `K=80` 条 append train 样本；
- 每个 node 至少覆盖一次，如果 node 数较多则随机轮转；
- train/test/probe 按 scenario_id 切分，禁止同一 tensor 同时出现在 train 和 test。

### 5.2 hard window 采样

从报告 39/43/44 的 hard cases 抽象出 window 特征，在合成管道中新生成 hard analog scenarios / windows，不直接训练 ST-Test 样本。

训练源中优先采样以下窗口：

| 特征 | 用途 |
| --- | --- |
| window 长度 6 到 13 | 对齐 report39 代表窗口 |
| `start % 8` 接近 patch 边界 | 覆盖 patch crossing |
| high dynamic range | 覆盖 1550/1250/1850 量级错读 |
| 多 0 或 clip 值 | 覆盖 forecasting/correlation hard cases |
| plateau / low variance | 覆盖模型把尖峰复述成平线的反例 |
| graph 邻接路径上的 node pair | 覆盖 cross-node comparison |

建议给每个候选窗口打分：

```text
score =
  2.0 * crosses_patch_boundary
  + 1.5 * has_zero_or_clip
  + 1.5 * high_range_bucket
  + 1.0 * low_variance_plateau
  + 1.0 * interior_window
  + 0.5 * rare_length_bucket
```

再用加权随机抽样，避免只留下极端窗口。

### 5.3 防止答案先验

必须在 manifest 里统计：

- 输出 list 长度分布；
- 数值范围分桶；
- zero count 分布；
- trend label 分布；
- comparison answer 分布；
- node id 分布；
- start/end 分布；
- 序列长度分布；
- graph node count 分布。

硬性要求：

- 单个 trend label 不超过 45%；
- comparison 中任一 `node a/node b/equal` 不超过 55%；
- raw window 的长度 top-1 不超过 35%；
- point extraction 不能集中在低 index；
- 同一 scenario 不超过 train 的 0.3%。

---

## 6. Processor metadata 泄漏防护

`processing_qwen3_ts.py` 的 `sp_encoding` 会把 `<ts><ts/>` 替换成：

```text
[offset=...|scaling=...|length=...|max=...|min=...|left=...|right=...]<ts><ts/>
```

所以 append 不能问这些可由文本 metadata 直接回答的内容：

- 整条序列长度；
- 整条序列最大值/最小值；
- 第一个值/最后一个值；
- 全序列 mean 近似可由 offset 推出，也不要问。

允许问：

- 内部 window 的逐点 raw values；
- 内部 window 的 mean/min/max/delta，但要过滤掉 local min/max 等于全局 min/max 的样本；
- 非端点 indices 的 point values；
- 局部 trend、zero count、threshold crossing；
- 跨节点局部比较。

append 数据生成脚本应写入 `leak_check` 统计：

```json
{
  "whole_series_stat_samples": 0,
  "endpoint_point_samples": 0,
  "local_stat_equals_global_extreme_filtered": 1234
}
```

---

## 7. 文件与注册计划

建议新增脚本：

```text
00_new_codes/tools/generate_st_align_append.py
00_new_codes/tools/evaluate_st_align_append.py
```

建议新增数据：

```text
data/ST-Bench/ST-Align-append/alignment_replay_question_sampled.jsonl
data/ST-Bench/ST-Align-append/alignment_append_train.jsonl
data/ST-Bench/ST-Align-append/alignment_append_test.jsonl
data/ST-Bench/ST-Align-append/alignment_append_hard_probe.jsonl
data/ST-Bench/ST-Align-append/manifest.json
data/ST-Bench/ST-Align-append/stats.json
```

建议更新 `data/dataset_info.json`：

```json
{
  "alignment_replay": {
    "file_name": "ST-Bench/ST-Align-append/alignment_replay_question_sampled.jsonl",
    "columns": {
      "prompt": "input",
      "response": "output",
      "timeseries": "timeseries"
    }
  },
  "alignment_append": {
    "file_name": "ST-Bench/ST-Align-append/alignment_append_train.jsonl",
    "columns": {
      "prompt": "input",
      "response": "output",
      "timeseries": "timeseries"
    }
  },
  "alignment_append_test": {
    "file_name": "ST-Bench/ST-Align-append/alignment_append_test.jsonl",
    "columns": {
      "prompt": "input",
      "response": "output",
      "timeseries": "timeseries"
    }
  }
}
```

JSONL 每行沿用原 ST-Align 字段，并允许额外 metadata：

```json
{
  "input": "... <ts><ts/> ... question ...",
  "timeseries": [[...], [...]],
  "output": "[425.44, 415.47, 407.05, 415.2, 425.13, 118.85]",
  "category": "st_align_append",
  "append_type": "raw_window_restate",
  "source": "synthetic_pipeline",
  "scenario_id": "...",
  "node_id": 1,
  "window": [27, 32],
  "indexing": "0-based"
}
```

LLaMAFactory 只读取 `input/output/timeseries`，额外字段不会进入训练，但便于统计和排查。

---

## 8. LoRA 训练建议

后续不是全参训练，建议从已保留的 Stage1 checkpoint-500 开 LoRA：

```text
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
```

第一版不要只训 append，但也不用放太多原数据。推荐把原 ST-Align 先按问题类型抽成 replay 子集，再和 append 混训：

```text
--dataset alignment_replay,alignment_append
--mix_strategy interleave_over
--interleave_probs 0.20,0.80
--template STReasoner-Align
--finetuning_type lora
```

`alignment_replay` 的 20% 内部建议分层抽样：

```text
graph_direct / graph_indirect
sinusoidal A / omega / phi
drift_type / baseline / kappa / sigma / lambda
node_type / edge_lag / edge_mod / effective
```

抽样目标不是复刻原始 194,212 条的自然占比，而是保证每个原能力簇都有 replay。这样 20% 原数据也能覆盖短答、图结构和参数题，不让 append 的局部读数任务把输出格式带偏。

不建议第一轮就用 100% append，因为那会把模型从原 ST-Align 的短答分布拉走，尤其可能损伤 yes/no、枚举、SDE 参数短答。若原 ST-Align full test 或 40 条 temporal balanced 探针明显掉分，优先调整 `alignment_replay` 的分问题抽样覆盖，而不是马上大幅提高原数据比例。

---

## 9. 评测计划

### 9.1 训练前 baseline

在 checkpoint-500 上先跑：

```text
alignment_append_test
alignment_append_hard_probe
原 ST-Align 128 或全量测试
30 条健康 + 40 条 temporal balanced 面板
```

目的：得到 append 任务的真实 baseline，不预设 checkpoint-500 会很差或很好。

### 9.2 append 专用指标

`evaluate_st_align_append.py` 应输出：

| 指标 | 说明 |
| --- | --- |
| parse_success | output 能否解析为 list / object / float / label |
| length_match | raw window 输出长度是否正确 |
| point_exact_0.01 | 每个点绝对误差 <= 0.01 |
| mean_abs_error | 逐点平均绝对误差 |
| max_abs_error | 逐点最大绝对误差 |
| stat_abs_error | local stats 误差 |
| label_accuracy | trend / comparison / event 分类准确率 |
| per_family_metrics | 按 append_type 分组 |
| hard_probe_metrics | 按 report39 hard window / analog window 分组 |

raw window 复述应同时尝试严格口径和宽松口径：

```text
strict：必须同长度，按 0-based explicit window 直接比较
lenient：允许模型输出外层 <answer> 或 markdown code block，但数值仍逐点比较
```

不要再使用 report39 的“0-based/1-based 取更优”作为训练后主要分数；append prompt 已明确 indexing，评测应按 prompt 口径严格比较。report39 口径只用于兼容旧 hard artifact。

### 9.3 防遗忘指标

保留以下红线：

| 项 | 红线 |
| --- | --- |
| 原 ST-Align 全量 `overall_score` | 相比 checkpoint-500 下降不超过 0.02 |
| 原 ST-Align `exact_match` | 下降不超过 0.01 |
| 原 ST-Align `relative_accuracy` | 下降不超过 0.03 |
| 40 temporal balanced | 至少不低于原 checkpoint-500，并重点看 A/omega/phi 是否改善 |
| ST-SFT/ST-CoT 小面板 | 不出现格式崩坏、长答污染短答 |

报告 28 的 checkpoint-500 全量 ST-Align 结果可作为参考：

```text
overall_score = 0.8405
exact_match = 0.9490
relative_accuracy = 0.7031
coverage = 1.0
```

---

## 10. 分阶段执行

### V0：小样本 dry run

生成：

```text
alignment_append_train.v0.jsonl 2,000 条
alignment_append_test.v0.jsonl 400 条
alignment_append_hard_probe.v0.jsonl 100 条
```

检查：

- JSONL 可读；
- `<ts><ts/>` 数量等于 `timeseries` node 数；
- `bash` 训练入口能识别 `alignment_append`；
- `evaluate_st_align_append.py` 能解析输出；
- 20 到 50 step LoRA smoke 不报 processor/collator 错。

### V1：第一版正式 append

生成：

```text
alignment_append_train.jsonl 50,000 条
alignment_append_test.jsonl 4,000 条
alignment_append_hard_probe.jsonl 600 到 1,000 条
```

训练：

```text
checkpoint-500 + LoRA
alignment_replay:alignment_append = 0.20:0.80
先 500 到 1000 step，按 append_test + 原 ST-Align 防遗忘指标停训
```

### V2：按结果调参

如果 raw window 复述提升明显但原 ST-Align 掉分：

- 先检查原 20% replay 是否覆盖所有原 ST-Align 问题簇；
- 提高掉分题型在 `alignment_replay` 内部的抽样权重；
- append 中 raw_window 从 40% 降到 30%；
- 缩短训练步数或提前按防遗忘指标停训，不优先提高原数据总比例。

如果原 ST-Align 稳定但 append 仍不会复述：

- 保持 `0.20:0.80` 总比例，优先重采 append 内部任务分布；
- raw_window + point extraction 合计提高到 65%；
- 增加 patch-boundary window 权重；
- 用 hard_probe 错例特征找更多 analog windows。

如果模型开始在普通回答里乱写长数值：

- 降 raw_window 比例；
- 缩短输出窗口；
- 只保留 stats/event/comparison；
- 在原 20% replay 中提高短答、yes/no、参数题权重。

---

## 11. 不做的事

第一版不做：

- 不训练 ST-Test hard cases；
- 不把 raw values 明文塞进 prompt 作为训练 side-channel；
- 不生成长 CoT；
- 不要求模型复述整条序列；
- 不改 `TimeSeriesEmbedding`、patch size、processor 或模型结构；
- 不把 graph-only 和 metadata-only 题继续扩容；
- 不替换原 `alignment` dataset name。

可作为后续对照但不进第一版：

- raw numeric side-channel probe；
- encoder recoverability linear probe；
- patch size / encoder ablation；
- graph-aware encoder；
- full finetune。

---

## 12. 预期收益与风险

预期收益：

- 直接给 report06 缺失的 node/window raw value supervision；
- 让模型学会明确索引口径，减少 0-based/1-based 混乱；
- 让 LoRA 更新更集中在“读 `<ts>` 里的局部数值”而不是继续背 ST-Align 题型先验；
- 为后续 Stage2/Stage3 的 forecasting/correlation 提供更可靠的数值读取底座。

主要风险：

- LoRA 容量有限，过多 raw list 任务可能挤压原短答能力；
- 逐点复述可能诱导模型在 CoT 中更爱写数字，但不一定提高最终任务；
- processor metadata 可能泄漏部分统计答案；
- 合成管道生成的窗口与 ST-Test hard cases 分布不完全一致；
- 如果 `TimeSeriesEmbedding` 本身对逐点恢复保真不足，append 只能部分改善，不能保证完美复述。

因此第一版必须以“append probe 提升 + 原 ST-Align 不掉分”为验收，不以 train loss 为验收。

---

## 13. 推荐验收标准

V1 LoRA 可以继续进入后续实验的最低标准：

```text
append_test raw_window parse_success >= 0.95
append_test raw_window length_match >= 0.90
append_test raw_window median max_abs_error 明显低于 checkpoint-500 baseline
append_hard_probe median max_abs_error 明显低于 checkpoint-500 baseline
原 ST-Align overall_score 下降 <= 0.02
原 ST-Align exact_match 下降 <= 0.01
40 temporal balanced 不低于 baseline
无明显输出格式崩坏
```

如果 append_test 提升但 hard_probe 不提升，说明训练分布还没覆盖 report39/43/44 的错例形态，需要补 hard analog synthetic window，而不是直接加大训练步数。

如果 hard_probe 提升但原 ST-Align 掉分，说明 append 比例过高或输出格式过强，需要回退采样比例。

---

## 14. 一句话收束

`ST-Align-append` 不应该是“再堆一批 ST-Align 参数题”，而应该是一个 **embedding-only、无测试泄漏、短答案、按场景去重、明确 node/window/index 的局部时序读取集**。它的核心价值是把报告 06/39 中“模型没有被监督过逐点读数，却在下游 thinking 中硬写数值”的问题变成可训练、可评测、可防遗忘的 LoRA 追加实验。
