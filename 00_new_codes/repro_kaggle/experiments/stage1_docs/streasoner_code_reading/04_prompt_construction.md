# 04 Prompt 构造

> 证据标记说明：
> `已从代码确认` 表示结论直接来自本仓库文件、类、函数或脚本参数。
> `根据代码推断，未由真实数据验证` 表示代码读取逻辑支持该判断，但本地没有原始 ST-Bench JSONL 样本可核对。
> `尚未确认` 表示需要下载数据、运行轻量验证或和论文描述对齐后才能下结论。

## 1. Prompt 文件位置

`已从代码确认`：推理阶段的任务级 prompt 配置在 `inference/prompt.json`。读取函数在 `inference/prompt_utils.py`：`PROMPT_CONFIG_PATH` 指向同目录的 `prompt.json`，见 `inference/prompt_utils.py:7-9`；`_load_prompt_config()` 读取 JSON 并做顶层类型检查，见 `inference/prompt_utils.py:12-20`；`get_prompt_suffix(task)` 取出对应任务的 `"prompt"` 字段并 `strip()`，见 `inference/prompt_utils.py:27-38`。

`已从代码确认`：推理主入口在 `inference/inference_tsmllm_vllm.py`。它的 `DEFAULT_TASK_CONFIG` 把任务名映射到 ST-Test 或 ST-Align 数据路径，见 `inference/inference_tsmllm_vllm.py:41-61`；`prepare_batches()` 从每条样本中读取 `sample["input"]` 和 `sample.get("timeseries", [])`，见 `inference/inference_tsmllm_vllm.py:123-149`；主函数在生成前追加 `prompt_suffix`，见 `inference/inference_tsmllm_vllm.py:272-277`。

`已从代码确认`：真正送入 vLLM 前，`LLMClient` 会加 chat template 和 system message。默认 system prompt 是 `"You are a helpful assistant."`，见 `inference/llm_utils.py:223-236`；`_apply_chat_template()` 构造 `system + user` 两条消息并调用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)`，见 `inference/llm_utils.py:270-275`；如果有 `timeseries`，`llm_batch_generate()` 会把输入包装为 `{"prompt": ..., "multi_modal_data": {"timeseries": ...}}`，见 `inference/llm_utils.py:311-336`。

训练侧还有三类 prompt/template 入口：

| 阶段 | 文件 | 作用 | 证据 |
|---|---|---|---|
| SFT Stage 1 | `src/llamafactory/data/template.py` 的 `STReasoner-Align` | Qwen 风格 user/system/assistant 模板，默认 system 是 helpful assistant，并启用 `streasoner` TS 插件 | `src/llamafactory/data/template.py:1978-1992`, `scripts/qwen3-8b/train_stage1.sh:3-11` |
| SFT Stage 2 | `src/llamafactory/data/template.py` 的 `STReasoner-CoT` | Qwen 风格模板，默认 system 写入 `<think>/<answer>` 输出格式，并启用 `streasoner` TS 插件 | `src/llamafactory/data/template.py:1994-2008`, `scripts/qwen3-8b/train_stage2.sh:3-11` |
| RL Stage 3 | `src/EasyR1/examples/format_prompt/str.jinja` | 根据原始 `input` 是否含 `Historical observation window`，为 forecasting 和多选任务追加不同输出格式 | `src/EasyR1/examples/format_prompt/str.jinja:1-7`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:7-15` |

## 2. 各任务 Prompt 模板

### 2.1 推理阶段 prompt_suffix

`已从代码确认`：`inference/prompt.json` 只定义输出格式约束，不定义完整题面。完整题面来自数据样本的 `input` 字段，见 `inference/inference_tsmllm_vllm.py:138-140`；suffix 再追加到题面末尾，见 `inference/inference_tsmllm_vllm.py:272-277`。

| task | prompt_suffix | 任务输出约束 | 证据 |
|---|---|---|---|
| `alignment` | `Output Format: <think>...</think><answer>Your answer here</answer>` | 要求先 reasoning，再输出普通答案 | `inference/prompt.json:2-4` |
| `reasoning_forecasting` | `Output Format: <think>...</think><answer>[Your predicted sequence, use comma to separate the values, use [] to enclose the sequence]</answer>` | `<answer>` 必须是预测序列 | `inference/prompt.json:5-7` |
| `reasoning_entity` | `Output Format: <think>...</think><answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>` | `<answer>` 只输出一个大写选项字母 | `inference/prompt.json:8-10` |
| `reasoning_etiological` | 同多选格式 | `<answer>` 只输出一个大写选项字母 | `inference/prompt.json:11-13` |
| `reasoning_correlation` | 同多选格式 | `<answer>` 只输出一个大写选项字母 | `inference/prompt.json:14-16` |
| `reasoning_causal` | 同多选格式 | `<answer>` 只输出一个大写选项字母 | `inference/prompt.json:17-19` |

`已从代码确认`：评估代码也围绕 `<answer>` 工作。`_extract_tag_content()` 会优先抽取 `<answer>...</answer>`，见 `evaluation/evaluate_qa.py:36-45`；多选任务用 `_normalize_choice()` 抽取 A-D 字母，见 `evaluation/evaluate_qa.py:23-33` 和 `evaluation/evaluate_qa.py:278-300`；forecasting 任务把 `output` 和预测文本都解析为数值序列，见 `evaluation/evaluate_qa.py:198-216`。

### 2.2 SFT 和 RL 阶段模板

`已从代码确认`：SFT 使用 LLaMA-Factory 模板名，而不是 `inference/prompt.json`。Stage 1 脚本使用 `--template "STReasoner-Align"`，见 `scripts/qwen3-8b/train_stage1.sh:3-11`；Stage 2 脚本使用 `--template "STReasoner-CoT"`，见 `scripts/qwen3-8b/train_stage2.sh:3-11`。

`已从代码确认`：`STReasoner-Align` 和 `STReasoner-CoT` 都用 Qwen chat 格式：user 内容包在 `<|im_start|>user ... <|im_end|><|im_start|>assistant`，assistant 内容以 `<|im_end|>` 结束，见 `src/llamafactory/data/template.py:1978-2008`。两者差异是默认 system：Align 是 `"You are a helpful assistant."`，见 `src/llamafactory/data/template.py:1988`；CoT 是包含 `<think>` 和 `<answer>` 的输出格式说明，见 `src/llamafactory/data/template.py:2004`。

`已从代码确认`：RL 阶段使用 `data.format_prompt=./src/EasyR1/examples/format_prompt/str.jinja`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:14`。`RLHFDataset._build_messages()` 会读取该 jinja 文件，并将 `content=prompt_str` 渲染成 user message，见 `src/EasyR1/verl/utils/dataset.py:180-197`。`str.jinja` 中如果 `content` 包含 `"Historical observation window"`，就使用 forecasting 的序列输出格式；否则使用多选题的大写字母输出格式，见 `src/EasyR1/examples/format_prompt/str.jinja:1-7`。

## 3. Prompt 构造流程

`已从代码确认`：推理阶段没有独立的 graph/question builder。数据样本的 `input` 字段已经是完整自然语言题面；`prepare_batches()` 只是把 `sample["input"]` 放进 `question_list`，见 `inference/inference_tsmllm_vllm.py:136-140`。本地已有推理输出显示，`question_text` 里同时包含角色说明、Node 时间序列占位符、`Graph Structure`、问题和输出格式，例如 forecasting 输出样例见 `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json:4`。

推理阶段的数据流是：

```text
sample["input"]
  -> question_list
  -> question.rstrip() + "\n\n" + prompt_suffix
  -> tokenizer.apply_chat_template([
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": question_with_suffix}
     ], add_generation_prompt=True)
  -> {"prompt": chat_text, "multi_modal_data": {"timeseries": ts_list[i]}}
  -> vLLM generation
```

逐步证据如下：

| 步骤 | 代码行为 | 证据 |
|---|---|---|
| task -> dataset | `DEFAULT_TASK_CONFIG` 选择对应 JSONL | `inference/inference_tsmllm_vllm.py:41-61` |
| JSONL -> question/timeseries | `prepare_batches()` 读取 `input` 和 `timeseries` | `inference/inference_tsmllm_vllm.py:123-149` |
| question -> question_with_suffix | `get_prompt_suffix(task)` 后追加到题面 | `inference/prompt_utils.py:27-38`, `inference/inference_tsmllm_vllm.py:272-277` |
| question_with_suffix -> chat prompt | `_apply_chat_template()` 加 system/user role 和 assistant generation prompt | `inference/llm_utils.py:270-275`, `base_model/Config-Qwen3-8B/chat_template.jinja:13-15`, `base_model/Config-Qwen3-8B/chat_template.jinja:31-32`, `base_model/Config-Qwen3-8B/chat_template.jinja:84-89` |
| chat prompt + timeseries -> vLLM input | `llm_batch_generate()` 增加 `multi_modal_data.timeseries` | `inference/llm_utils.py:311-336` |

`根据代码推断，未由真实数据验证`：原始 `input` 的文本结构通常是：

```text
You are a spatial temporal analysis expert.
Node 0 time series with length of <L0>: <ts><ts/>;
Node 1 time series with length of <L1>: <ts><ts/>;
...
Graph Structure: Node a->Node b; ...
please analyze the spatial temporal data and answer the following question:
<task-specific question>
[Options: A. ... B. ... C. ... D. ...] 或 [Historical observation window: ...]
```

这类结构由本地 `generated_answer.json` 的 `question_text` 支持，但原始 ST-Bench JSONL 未下载，不能确认所有样本都完全遵循该格式。

训练侧流程略有不同：

```text
dataset_info.json 注册的 input/output/timeseries
  -> AlpacaDatasetConverter: input -> _prompt, output -> _response, timeseries -> _timeseries
  -> template.mm_plugin.process_messages(...)
  -> template.encode_multiturn(...)
  -> input_ids / labels / timeseries
```

`已从代码确认`：`AlpacaDatasetConverter.__call__()` 把 `input` 对应的 prompt/query 合并成一条 user message，把 `output` 作为 assistant response，并保留 `_timeseries`，见 `src/llamafactory/data/converter.py:84-135`。`SupervisedDatasetProcessor._encode_data_example()` 在存在 timeseries 时调用 `template.mm_plugin.process_messages(..., timeseries=timeseries)`，再编码多轮模板，见 `src/llamafactory/data/processor/supervised.py:31-52`。

## 4. Time Series 占位符机制

`已从代码确认`：STReasoner prompt 使用 `<ts><ts/>` 作为文本占位符，真实数值不直接展开到普通文本中。Qwen3TSProcessor 的注释明确说明输入 prompt 包含 `<ts><ts/>`，timeseries 列表与这些占位符匹配，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:72-93`。

占位符处理分为两种路径：

| 路径 | 行为 | 证据 |
|---|---|---|
| 非 vLLM / SFT processor 路径 | `prompt.split("<ts><ts/>")`，每个 timeseries 经过 `sp_encoding()`，再把占位符替换成带统计信息的 `[offset=...|scaling=...|length=...|max=...|min=...|left=...|right=...]<ts><ts/>` 片段 | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:122-143`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50` |
| vLLM 路径 | prompt 修改留给 vLLM；processor 设置 `vllm_flag=True`，返回 `zip(ts_tokens, encoded_ts_arrays)`；vLLM 的 prompt replacement 再用 TS token 和 patch 数扩展 placeholder | `inference/vllm/chatts_vllm.py:312-330`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:104-118`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:164-171`, `inference/vllm/chatts_vllm.py:352-401` |

`已从代码确认`：`sp_encoding()` 会对单条时间序列做均值中心化和缩放，并生成一个统计文本片段；同时把数值序列编码成 value/mask 成对特征，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`。随后模型端 `Qwen3TSForCausalLM` 会解析 `timeseries`，拼接成 TS tensor，送入 `ts_encoder`，再把得到的 TS embedding 合并到语言模型 embedding 中，见 `inference/vllm/chatts_vllm.py:628-709`。

`已从代码确认`：vLLM 侧限制每个 prompt 最多 50 条 timeseries，见 `inference/llm_utils.py:149` 和 `inference/vllm/chatts_vllm.py:210-211`。

`已从代码确认`：SFT processor 会检查 timeseries 数量与 TS special token 数量是否一致，不一致则丢弃样本，见 `src/llamafactory/data/processor/supervised.py:64-74` 和 `src/llamafactory/data/processor/supervised.py:115-128`。

`已从代码确认`：S-GRPO 的 spatial reward 会构造一个无 graph 的 prompt 对照。`remove_graph_structure()` 删除从 `Graph Structure:` 到 `please analyze` 之前的文本，见 `src/EasyR1/verl/utils/dataset.py:35-56`；当 `enable_spatial_reward` 为真时，`RLHFDataset.__getitem__()` 会额外生成 `prompt_no_graph`、`input_ids_no_graph` 和 `attention_mask_no_graph`，见 `src/EasyR1/verl/utils/dataset.py:313-334`。

## 5. 伪样例

下面是一个结构样例，只展示 prompt 的组成方式，不代表真实数据内容。

### 5.1 原始 JSONL 中的 `input`

```text
You are a spatial temporal analysis expert.
Node 0 time series with length of <L0>: <ts><ts/>;
Node 1 time series with length of <L1>: <ts><ts/>;
Node 2 time series with length of <L2>: <ts><ts/>;
Graph Structure: Node 0->Node 2; Node 1->Node 2,
please analyze the spatial temporal data and answer the following question:
Which (name, description) pair should Node 2 correspond to?
Options:
A. <option A>
B. <option B>
C. <option C>
D. <option D>
```

对应 JSONL 结构按代码读取逻辑可表示为：

```json
{
  "input": "<上面的题面字符串>",
  "timeseries": [
    ["<node 0 numeric values, omitted>"],
    ["<node 1 numeric values, omitted>"],
    ["<node 2 numeric values, omitted>"]
  ],
  "output": "B"
}
```

这里的 `timeseries` 数值被省略，因为本地没有原始 ST-Bench JSONL；真实数值不会被编入这个伪样例。

### 5.2 推理时追加 prompt_suffix 后

```text
<原始 input>

Output Format: <think>Your step-by-step reasoning process</think><answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>
```

### 5.3 chat template 后的模型文本输入

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<原始 input>

Output Format: <think>...</think><answer>...</answer><|im_end|>
<|im_start|>assistant
```

同时，vLLM 输入中还有一个非文本字段：

```json
{
  "multi_modal_data": {
    "timeseries": ["<node 0 series>", "<node 1 series>", "<node 2 series>"]
  }
}
```

`已从代码确认`：上述 chat 包装来自 `LLMClient._apply_chat_template()` 和 Qwen chat template，见 `inference/llm_utils.py:270-275` 与 `base_model/Config-Qwen3-8B/chat_template.jinja:13-15`, `base_model/Config-Qwen3-8B/chat_template.jinja:31-32`, `base_model/Config-Qwen3-8B/chat_template.jinja:84-89`；`multi_modal_data.timeseries` 包装来自 `inference/llm_utils.py:330-336`。

## 6. 组会讲法

### 本文档核心结论

`已从代码确认`：推理阶段的 prompt 不是从多个字段临时拼 graph、question 和 node 描述，而是直接使用数据样本的 `input` 字段；这个字段已经包含 Node 时间序列占位符、graph text 和具体问题。推理脚本只额外追加任务级 `prompt_suffix`，用于强制 `<think>/<answer>` 输出格式。

`已从代码确认`：STReasoner prompt 与普通 LLM prompt 的关键区别是 `<ts><ts/>`。普通 LLM 只接收文本 token；STReasoner 同时接收文本里的 TS 占位符和 `multi_modal_data.timeseries` 中的真实数值序列，再由 TS processor 和 TS encoder 把时间序列 embedding 合并进语言模型输入。

`已从代码确认`：system prompt 有三处：推理默认 `"You are a helpful assistant."`；SFT Align 模板默认 `"You are a helpful assistant."`；SFT CoT 模板默认包含 `<think>/<answer>` 格式说明。RL 阶段主要通过 `str.jinja` 在 user content 里追加 instruction 和 output format。

### 组会可讲版本

这部分可以这样讲：STReasoner 的 prompt 分两层。第一层是数据自带的题面，它已经把“有哪些节点、每个节点对应一个 `<ts><ts/>`、图结构是什么、要回答什么问题”写在 `input` 字段里。第二层是代码追加的格式约束，比如 forecasting 要在 `<answer>` 里输出数字列表，entity/etiological/correlation 要输出一个大写选项字母。

模型实际看到的不只是普通文本。文本里保留 `<ts><ts/>`，真实时间序列放在 `multi_modal_data.timeseries`。processor 会把每条序列做归一化、切 patch、编码成 TS embedding，再替换文本里的 TS token 位置。因此 STReasoner 的 prompt 是“文本题面 + 时间序列模态输入”的组合，不是把所有数值直接写成长文本。

训练和推理的 prompt 入口不完全一样：SFT 走 LLaMA-Factory 的 `STReasoner-Align/CoT` 模板；RL 走 EasyR1 的 `str.jinja`；推理走 `inference/prompt.json` 的 suffix。但三者共同要求模型用 `<think>` 表达推理过程，用 `<answer>` 包住最终答案。

### 后续需要验证的问题

1. `尚未确认`：下载真实 ST-Bench JSONL 后，需要核对所有任务的 `input` 是否都严格包含 `You are a spatial temporal analysis expert`、Node 占位符、`Graph Structure` 和问题四段。
2. `尚未确认`：真实样本中 `<ts><ts/>` 数量是否总是等于 `timeseries` 列表长度，需要用轻量脚本逐行检查。
3. `尚未确认`：SFT CoT 数据的 `output` 是否已经包含 `<think>/<answer>`，还是主要由 template/default_system 约束生成格式，需要查看真实 `ST-CoT/*.jsonl`。
4. `尚未确认`：`prompt_suffix` 与原始 `input` 是否存在重复输出格式说明，需要下载数据后检查原始 `input` 末尾。
