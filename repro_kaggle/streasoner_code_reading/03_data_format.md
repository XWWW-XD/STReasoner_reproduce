# 03 数据格式与任务类型

> 证据标记说明：
> `已从代码确认` 表示结论直接来自本仓库文件、类、函数或脚本参数。
> `根据代码推断，未由真实数据验证` 表示代码读取逻辑支持该判断，但本地没有原始 ST-Bench JSONL 样本可核对。
> `尚未确认` 表示需要下载数据、运行轻量验证或和论文描述对齐后才能下结论；这些条目也记录在 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 数据目录结构

`已从代码确认`：当前项目期望把 ST-Bench 下载到项目文件旁的 `data/ST-Bench/`。下载入口是 `download_dataset.py:11-24`，其中 `local_dir` 由 `os.path.dirname(__file__)`、`data`、`ST-Bench` 拼接得到，HuggingFace dataset repo 是 `Time-HD-Anonymous/ST-Bench`，见 `download_dataset.py:13-20`。

`已从代码确认`：本地 `data/` 目录目前只有 `data/dataset_info.json`，没有下载后的 `data/ST-Bench/` JSONL 文件。因此本文档不能抽取真实原始样本，只能从代码读取逻辑和已有推理输出反推字段。

当前可确认的目录角色如下：

| 路径 | 类型 | 作用 | 证据 |
|---|---|---|---|
| `data/dataset_info.json` | 数据集注册表 | 把 LLaMA-Factory dataset 名称映射到 ST-Bench JSONL 路径，并声明列映射 | `data/dataset_info.json:2-330` |
| `data/ST-Bench/` | 下载后数据根目录 | 预期保存 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test 等 split | `download_dataset.py:12-21` |
| `exp_STReasoner-8B/*/generated_answer.json` | 已有推理输出样例 | 可验证生成结果文件结构和题面风格，但不是原始训练/测试 JSONL | 例如 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:3-5` |
| `inference/prompt.json` | 推理输出格式配置 | 按 task 追加 `<think>` 和 `<answer>` 输出格式要求 | `inference/prompt.json:2-19` |
| `evaluation/` | 评估代码 | 读取测试 JSONL 和 `generated_answer.json`，按任务计算指标 | `evaluation/evaluate.py:100-174`, `evaluation/evaluate_qa.py:69-139` |

`尚未确认`：`evaluation/evaluate.py` 的默认数据路径是 `data/reasoning/*.jsonl`，见 `evaluation/evaluate.py:56-80`；但 `inference/inference_tsmllm_vllm.py` 默认读取 `data/ST-Bench/ST-Test/*.jsonl`，见 `inference/inference_tsmllm_vllm.py:42-61`。本地没有 `data/reasoning/`，所以后续评估时大概率需要显式传入 `--dataset data/ST-Bench/ST-Test/<task>_test.jsonl`，这一点需要下载数据后验证。

## 2. 数据集命名与用途

`已从代码确认`：ST-Bench 的主要命名集中在 `data/dataset_info.json`，训练和推理脚本再引用这些名称或路径。

| 数据命名 | 注册/路径 | 字段映射 | 用途 | 证据 |
|---|---|---|---|---|
| `ST-Bench` | 下载根目录 `data/ST-Bench/` | 无直接列映射 | 整个 benchmark 数据集根目录 | `download_dataset.py:12-21` |
| `ST-Align` | `ST-Bench/ST-Align/alignment_train.jsonl`, `alignment_test.jsonl` | `input -> prompt`, `output -> response`, `timeseries -> timeseries` | Stage 1 time-series alignment SFT；`alignment_test` 也可作为推理任务 | `data/dataset_info.json:2-17`, `scripts/qwen3-8b/train_stage1.sh:5-11`, `inference/inference_tsmllm_vllm.py:42-45` |
| `ST-CoT` | `ST-Bench/ST-CoT/{entity,etiological,correlation,forecasting}_cot.jsonl` | `input`, `output`, `timeseries` | Stage 2 cold-start reasoning SFT；四类任务均衡混合 | `data/dataset_info.json:18-49`, `scripts/qwen3-8b/train_stage2.sh:5-9` |
| `ST-SFT` | `ST-Bench/ST-SFT/{entity,etiological,correlation,forecasting}_finetune.jsonl` | `input`, `output`, `timeseries` | 注册存在；本次静态搜索未找到 Qwen3-8B 官方脚本直接使用 `entity_sft` 等名称 | `data/dataset_info.json:50-81` |
| `ST-RL` | `ST-Bench/ST-RL/{entity,etiological,correlation,forecasting}_rl.jsonl` | `input`, `output`, `timeseries` | Stage 3 GRPO / S-GRPO 训练集 | `data/dataset_info.json:82-113`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:9-14` |
| `ST-Test` | `ST-Bench/ST-Test/{entity,etiological,correlation,forecasting}_test.jsonl` | `input`, `output`, `timeseries` | 四类 reasoning 任务测试集；推理默认读取这里，RL val_files 也读取这里 | `data/dataset_info.json:114-145`, `inference/inference_tsmllm_vllm.py:46-57`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:10` |
| `ST-Causal` | `ST-Bench/ST-Causal/causal.jsonl` | `input`, `output`, `timeseries` | 额外 causal 任务；推理代码支持 `reasoning_causal` | `data/dataset_info.json:146-153`, `inference/inference_tsmllm_vllm.py:58-60` |
| `*-Text` | `ST-CoT-Text`, `ST-SFT-Text`, `ST-RL-Text`, `ST-Test-Text` | `input`, `output`，没有 `timeseries` 列 | text-only ablation | `data/dataset_info.json:154-264`, `scripts/qwen3-8b/train_stage2_only_text.sh:5-9` |
| `*-Image` | `ST-CoT-Image`, `ST-RL-Image` | `input`, `output`, `images` | image-only ablation，使用 Qwen3-VL 模板 | `data/dataset_info.json:266-329`, `scripts/qwen3-vl-8b-instruct/train_stage2_only_image.sh:5-9` |

`已从代码确认`：S-GRPO 脚本显式把四个 ST-RL 文件拼成一个训练集，把四个 ST-Test 文件拼成验证集，并把 `data.prompt_key=input`、`data.ts_key=timeseries`、`data.answer_key=output` 传给 EasyR1 dataloader，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:9-14`。

`已从代码确认`：EasyR1 的 `DataConfig` 默认支持 `prompt_key`、`answer_key`、`ts_key`，其中 `ts_key` 默认是 `timeseries`，见 `src/EasyR1/verl/trainer/config.py:35-60`。`create_dataloader()` 再把这些 key 传入 `RLHFDataset`，见 `src/EasyR1/verl/trainer/data_loader.py:26-46` 和 `src/EasyR1/verl/trainer/data_loader.py:70-87`。

## 3. 单条样本字段解释

### 3.1 原始 JSONL 样本字段

`根据代码推断，未由真实数据验证`：对于普通 time-series 版本的 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test、ST-Causal，每一行 JSONL 至少需要以下字段：

| 字段 | 类型 | 作用 | 证据 |
|---|---|---|---|
| `input` | string | 自然语言题面；推理时直接作为 `question`；训练时映射为 LLaMA-Factory 的 prompt | `data/dataset_info.json:4-7`, `inference/inference_tsmllm_vllm.py:138-140`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:11` |
| `output` | string | 标准答案；SFT 训练时作为 response；RL 训练时进入 `ground_truth`；评估时作为 target | `data/dataset_info.json:4-7`, `src/EasyR1/verl/utils/dataset.py:381-386`, `evaluation/evaluate_qa.py:157-165`, `evaluation/evaluate_qa.py:213-216`, `evaluation/evaluate_qa.py:286-300` |
| `timeseries` | list | 与题面中的 `<ts><ts/>` 占位符对应的时间序列列表 | `data/dataset_info.json:4-7`, `inference/inference_tsmllm_vllm.py:140-148`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:122-143` |

`根据代码推断，未由真实数据验证`：一个普通 time-series 任务样本可以抽象成下面的形态。注意这不是从本地 ST-Bench 抽出的真实样本。

```json
{
  "input": "You are a spatial temporal analysis expert. Node 0 time series with length of 48: <ts><ts/>; ... Graph Structure: Node 0->Node 2; ... please analyze ... Options: A. ... B. ... C. ... D. ...",
  "timeseries": [
    [/* node 0 values */],
    [/* node 1 values */]
  ],
  "output": "B"
}
```

`已从代码确认`：推理代码不读取 `output`，只读取 `input` 和 `timeseries`。`load_dataset()` 把 JSONL 每行转成 dict，见 `inference/inference_tsmllm_vllm.py:98-106`；`prepare_batches()` 将 `sample["input"]` 放入 `question_list`，将 `sample.get("timeseries", [])` 转成 float list 后放入 `ts_list`，见 `inference/inference_tsmllm_vllm.py:123-149`。

`已从代码确认`：评估代码读取 `output` 作为 ground truth。`load_jsonl_dataset()` 给没有 `idx` 的样本补 `idx`，见 `evaluation/evaluate_qa.py:69-79`；但后续评估循环主要使用 `enumerate(dataset)` 的顺序索引去匹配预测，见 `evaluation/evaluate_qa.py:157-160`, `evaluation/evaluate_qa.py:213-216`, `evaluation/evaluate_qa.py:286-288`。

`已从代码确认`：SFT 数据进入 LLaMA-Factory 后，`AlpacaDatasetConverter.__call__()` 会把 `input` 映射为 `_prompt`，把 `output` 映射为 `_response`，把 `timeseries` 映射为 `_timeseries`，见 `src/llamafactory/data/converter.py:84-135`。`SupervisedDatasetProcessor._encode_data_example()` 会把 timeseries 传给 multimodal plugin，并检查 timeseries 数量和特殊 token 数量是否匹配，见 `src/llamafactory/data/processor/supervised.py:31-74`。

`已从代码确认`：text-only 版本没有 `timeseries` 列；image-only 版本使用 `images` 列而不是 `timeseries` 列。对应列映射分别见 `data/dataset_info.json:154-264` 和 `data/dataset_info.json:266-329`。

### 3.2 已有推理输出样例字段

`已从代码确认`：本地存在 `exp_STReasoner-8B/*/generated_answer.json`，这些是推理输出，不是原始 JSONL。现有文件第一条记录包含 `idx`、`question_text`、`response` 三个字段，例如 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:3-5` 和 `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json:3-5`。

`已从代码确认`：当前推理脚本会在新生成文件中额外写入 `num_tokens`，见 `inference/inference_tsmllm_vllm.py:308-321`；但本地已有 `exp_STReasoner-8B/*/generated_answer.json` 第一条记录没有 `num_tokens`。评估代码兼容这种 flat `response` 格式，见 `evaluation/evaluate_qa.py:97-126`。

三个不同任务的本地输出侧样例可用于理解题面风格：

| 任务 | 本地输出文件 | 可确认的题面形态 | 证据 |
|---|---|---|---|
| `reasoning_entity` | `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json` | 题面包含多个 Node 的 `<ts><ts/>`、`Graph Structure`、问题 “Which (name, description) pair should Node ... correspond to?”、A-D 选项 | `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:4` |
| `reasoning_etiological` | `exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/generated_answer.json` | 题面询问 “Which etiological scenario can be inferred ...?”，输出格式要求单个大写选项字母 | `exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/generated_answer.json:4`, `inference/prompt.json:11-13` |
| `reasoning_correlation` | `exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/generated_answer.json` | 题面询问某个时间窗口内对 Node 的影响描述，A-D 选项 | `exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/generated_answer.json:4`, `inference/prompt.json:14-16` |
| `reasoning_forecasting` | `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json` | 题面包含 context、预测节点、预测步数和 `Historical observation window`，输出格式要求数字序列 | `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json:4`, `inference/prompt.json:5-7` |

## 4. 四类任务对比

`已从代码确认`：推理入口支持四个主要 reasoning task：`reasoning_forecasting`、`reasoning_entity`、`reasoning_etiological`、`reasoning_correlation`，默认数据集分别指向 ST-Test 四个 JSONL，见 `inference/inference_tsmllm_vllm.py:46-57`。README 的推理和评估循环也只遍历这四个任务，见 `README.md:142-170`。

| 任务 | 代码确认的评估类型 | 题面/任务含义 | 输出要求 | 指标 | 证据 |
|---|---|---|---|---|---|
| `reasoning_etiological` | 多选题 | 根据本地输出样例，题面要求从时空数据推断 etiological scenario，即成因/场景解释 | 单个大写选项字母 | accuracy | `inference/prompt.json:11-13`, `evaluation/evaluate_qa.py:278-329`, `exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/generated_answer.json:4` |
| `reasoning_entity` | 多选题 | 根据本地输出样例，题面要求判断某个 Node 对应的 `(name, description)` 实体 | 单个大写选项字母 | accuracy | `inference/prompt.json:8-10`, `evaluation/evaluate_qa.py:278-329`, `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:4` |
| `reasoning_correlation` | 多选题 | 根据本地输出样例，题面要求描述某时间窗口对某节点的影响/相关事件 | 单个大写选项字母 | accuracy | `inference/prompt.json:14-16`, `evaluation/evaluate_qa.py:278-329`, `exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/generated_answer.json:4` |
| `reasoning_forecasting` | 数值序列预测 | 根据本地输出样例，题面要求在给定 context 和历史窗口后预测某节点未来若干步 | `[v1, v2, ...]` 数字序列 | MAE、MAPE、target stats | `inference/prompt.json:5-7`, `evaluation/evaluate_qa.py:198-275`, `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json:4` |

四类任务的输入/输出差异如下：

| 维度 | `reasoning_entity` | `reasoning_etiological` | `reasoning_correlation` | `reasoning_forecasting` |
|---|---|---|---|---|
| 共同输入 | Node 时间序列占位符 `<ts><ts/>`、`Graph Structure`、自然语言问题 | 同左 | 同左 | 同左 |
| 问题核心 | 节点实体或地点/属性匹配 | 成因/场景解释 | 节点影响、事件或相关关系判断 | 未来值预测 |
| 选项 | A-D 多选 | A-D 多选 | A-D 多选 | 无 A-D 选项，要求数值序列 |
| `output` 预期 | `"A"` / `"B"` / `"C"` / `"D"` | `"A"` / `"B"` / `"C"` / `"D"` | `"A"` / `"B"` / `"C"` / `"D"` | 可解析为 list 的数字序列字符串 |
| 预测解析 | `_normalize_choice()` 抽取 `<answer>` 后取 A-D | 同左 | 同左 | `_parse_series()` 从 `<answer>` 或文本中解析数字 |
| 指标 | accuracy | accuracy | accuracy | MAE、MAPE |
| 证据 | `evaluation/evaluate_qa.py:23-33`, `evaluation/evaluate_qa.py:278-317` | 同左 | 同左 | `evaluation/evaluate_qa.py:47-66`, `evaluation/evaluate_qa.py:198-275` |

`根据代码推断，未由真实数据验证`：`input` 中 `<ts><ts/>` 占位符数量应与 `timeseries` 列表长度一致。Qwen3TSProcessor 会按 `<ts><ts/>` split prompt，并在数量不一致时报错，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:122-143`；SFT processor 也会在 timeseries 数量与 TS token 数量不一致时丢弃样本，见 `src/llamafactory/data/processor/supervised.py:67-74`。

## 5. 数据流图

下面是一条 ST-Test 样本在推理和评估中的数据流。这里描述的是代码路径，不代表已经跑通。

```text
JSONL line
  -> dict sample
  -> question = sample["input"]
  -> timeseries = sample.get("timeseries", [])
  -> question + task-specific prompt suffix
  -> chat template + {"multi_modal_data": {"timeseries": ...}}
  -> vLLM TS processor: replace <ts><ts/>, encode timeseries, merge TS embeddings
  -> generated text response
  -> generated_answer.json
  -> evaluation: read dataset["output"] + prediction["response"]
  -> metrics
```

逐步证据：

| 步骤 | 代码行为 | 证据 |
|---|---|---|
| JSONL line -> dict | `load_dataset()` 逐行 `json.loads(line)` | `inference/inference_tsmllm_vllm.py:98-106` |
| dict -> question / timeseries | `prepare_batches()` 读取 `sample["input"]`，遍历 `sample.get("timeseries", [])` 并转 float | `inference/inference_tsmllm_vllm.py:123-149` |
| question -> prompt | `get_prompt_suffix(task)` 从 `prompt.json` 取输出格式，推理主函数追加到 question 末尾 | `inference/prompt_utils.py:27-38`, `inference/inference_tsmllm_vllm.py:272-277` |
| prompt -> model input | `LLMClient.llm_batch_generate()` 应用 chat template，并包装 `multi_modal_data.timeseries` | `inference/llm_utils.py:311-336` |
| timeseries -> TS processor | vLLM data parser 识别 `timeseries` modality；HF processor 设置 `vllm_flag=True` | `inference/vllm/chatts_vllm.py:282-330` |
| `<ts><ts/>` -> token replacement | `_get_prompt_updates()` 根据 patch count 扩展 TS placeholder token | `inference/vllm/chatts_vllm.py:352-401`, `base_model/Config-Qwen3-8B/config.json:73-86` |
| TS tensor -> embedding | Qwen3TSForCausalLM 解析 TS 输入、经过 TS encoder、再 merge 到文本 embedding | `inference/vllm/chatts_vllm.py:628-709` |
| generated answer -> JSON | 推理脚本把 `idx`、`question_text`、`response`、`num_tokens` 写入 `generated_answer.json` | `inference/inference_tsmllm_vllm.py:308-321` |
| prediction -> evaluation | `load_prediction_files()` 读取 `response`，抽取 `<answer>`；`evaluate_predictions_for_task()` 按 task 分派指标 | `evaluation/evaluate_qa.py:82-139`, `evaluation/evaluate_qa.py:320-329` |

训练侧的数据流与推理侧不同：

```text
dataset_info name
  -> LLaMA-Factory DatasetAttr
  -> AlpacaDatasetConverter: input/output/timeseries -> _prompt/_response/_timeseries
  -> SupervisedDatasetProcessor: process_messages + encode_multiturn
  -> input_ids / labels / timeseries
```

`已从代码确认`：模板 `STReasoner-Align` 和 `STReasoner-CoT` 都使用 `get_mm_plugin("streasoner", timeseries_token="<ts>")`，见 `src/llamafactory/data/template.py:1978-2008`。`ChatTSPlugin` 调用 processor 处理 `timeseries` 并重写消息文本，见 `src/llamafactory/data/mm_plugin.py:1999-2034`。

## 6. 尚未验证的数据假设

1. `尚未确认`：本地没有 `data/ST-Bench/` 原始 JSONL，因此无法抽取 3 条真实不同任务样本。本文对 `input`、`output`、`timeseries` 的说明来自 `data/dataset_info.json`、推理/评估/dataloader 代码和已有 `generated_answer.json` 输出侧样例。

2. `根据代码推断，未由真实数据验证`：`input` 原始题面中包含 `<ts><ts/>` 占位符、`Graph Structure`、自然语言问题；已有 `generated_answer.json` 的 `question_text` 可以验证推理输出侧确实包含这些文本片段，但不能证明原始 JSONL 完全相同，因为推理脚本会追加 prompt suffix，见 `inference/inference_tsmllm_vllm.py:272-277`。

3. `根据代码推断，未由真实数据验证`：普通 time-series split 的 `timeseries` 是一个 list，内部每个 item 可被转成 float list。推理代码 `_to_float_list()` 支持 ndarray、list、tuple、标量和 None，见 `inference/inference_tsmllm_vllm.py:127-148`；真实数据是否统一为二维 list 需要下载后确认。

4. `尚未确认`：`ST-SFT` 在 `data/dataset_info.json:50-81` 注册，但本次静态搜索没有找到官方 Qwen3-8B 脚本使用 `entity_sft`、`etiological_sft`、`correlation_sft`、`forecasting_sft`。它们与 `ST-CoT`、`ST-RL` 的职责差异需要结合论文或真实数据确认。

5. `尚未确认`：`evaluation/evaluate.py` 默认数据路径与 ST-Bench 注册路径不一致。默认路径见 `evaluation/evaluate.py:56-80`，ST-Test 路径见 `data/dataset_info.json:114-145` 和 `inference/inference_tsmllm_vllm.py:46-57`。下载数据后需要确认 README 的 evaluation 命令是否需要补 `--dataset`。

6. `尚未确认`：本地已有 `exp_STReasoner-8B/*/generated_answer.json` 的字段是 `idx`、`question_text`、`response`，而当前推理代码会额外写 `num_tokens`。这说明已有输出可能来自旧版脚本或不同导出逻辑；不影响评估读取，但影响 token 统计，见 `evaluation/evaluate.py:13-53` 和 `evaluation/evaluate_qa.py:97-126`。

### 本文档核心结论

`已从代码确认`：STReasoner 的普通时序样本核心 schema 是 `input`、`output`、`timeseries`。SFT 通过 LLaMA-Factory 把它们映射为 prompt/response/timeseries；RL 通过 EasyR1 把 `input`、`output`、`timeseries` 映射为 prompt、ground truth 和 multimodal TS 输入；推理只用 `input` 和 `timeseries`；评估用 `output` 和 `generated_answer.json` 的 `response`。

`已从代码确认`：四类主任务分成两种评估范式：`reasoning_forecasting` 是数值序列预测，评估 MAE/MAPE；`reasoning_entity`、`reasoning_etiological`、`reasoning_correlation` 是 A-D 多选题，评估 accuracy。

`尚未确认`：真实 ST-Bench JSONL 没有下载，不能确认每类任务的真实字段细节、样本分布、`timeseries` 嵌套形状和 `ST-SFT` 的实际用途。

### 组会可讲版本

这周我没有跑完整训练，但已经把数据入口读清楚了。代码里 ST-Bench 不是一个单文件，而是按阶段和任务拆成 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test。普通时序版本统一用 `input/output/timeseries` 三个字段；text-only 去掉 `timeseries`，image-only 改用 `images`。

推理时，一条测试样本会从 JSONL 读成 dict，`input` 变成题面，`timeseries` 变成多模态输入；脚本再按 task 追加 `<think>/<answer>` 输出格式，最后通过 vLLM 的 time-series processor 把 `<ts><ts/>` 替换为时间序列 embedding。生成结果保存为 `generated_answer.json`，评估时从 response 里抽 `<answer>`。

四类任务里，forecasting 是未来数值序列预测，用 MAE/MAPE；entity、etiological、correlation 都是多选题，用 accuracy。当前最大的缺口是真实 ST-Bench JSONL 没下载，所以字段解释是代码级确认加代码推断，还不能说已经验证了真实样本。

### 后续需要验证的问题

1. 下载或获取少量 ST-Bench JSONL 后，抽取 `entity_test`、`etiological_test`、`correlation_test`、`forecasting_test` 各 1 条，核对 `input/output/timeseries` 的真实格式。
2. 检查每条样本中 `<ts><ts/>` 数量是否严格等于 `timeseries` 列表长度。
3. 验证 `evaluation/evaluate.py` 是否必须加 `--dataset data/ST-Bench/ST-Test/<task>_test.jsonl` 才能按 README 命令跑通。
4. 对比 `ST-CoT`、`ST-SFT`、`ST-RL` 三类文件的 `output`：确认 CoT 是否含完整 reasoning、SFT 是否是精简答案、RL 是否只保留 ground truth。
5. 确认已有 `exp_STReasoner-8B/*/generated_answer.json` 是否来自旧脚本，因为它们缺少当前代码会写入的 `num_tokens` 字段。
