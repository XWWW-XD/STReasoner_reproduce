# 05 Inference 主链路

> 证据标记说明：
> `已从代码确认` 表示结论直接来自本仓库文件、类、函数或脚本参数。
> `根据代码推断，未由真实数据验证` 表示代码读取逻辑支持该判断，但本地没有原始 ST-Bench JSONL 或本地模型可运行验证。
> `尚未确认` 表示需要下载数据、准备模型或做小规模运行后才能下结论。

## 1. 入口命令

`已从代码确认`：README 给出的官方推理入口是循环四个 reasoning task，调用 `inference/inference_tsmllm_vllm.py` 并传入 `--task` 和 `--model_path`，见 `README.md:142-156`。

```bash
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do
    python inference/inference_tsmllm_vllm.py \
        --task $task \
        --model_path checkpoints/easy_r1/qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/huggingface
done
```

`已从代码确认`：脚本顶部先 `import inference.vllm.chatts_vllm`，注释说明这个 import 必须先执行，因为它会注册自定义 ChatTS 模块和 multimodal processor，见 `inference/inference_tsmllm_vllm.py:15-24`。`chatts_vllm.py` 末尾把 `Qwen2TSForCausalLM` 和 `Qwen3TSForCausalLM` 注册到 vLLM 的 `ModelRegistry`，见 `inference/vllm/chatts_vllm.py:760-765`。

## 2. 参数说明

### 2.1 CLI 参数

`已从代码确认`：`parse_args()` 定义了 inference 脚本的全部命令行参数，见 `inference/inference_tsmllm_vllm.py:152-215`。

| 参数 | 默认值 | 作用 | 代码证据 |
|---|---:|---|---|
| `--task` | `alignment` | 选择任务；choices 来自 `DEFAULT_TASK_CONFIG` 的 key，包括 `alignment`、四个 reasoning task 和 `reasoning_causal` | `inference/inference_tsmllm_vllm.py:152-160` |
| `--dataset` | `None` | 自定义 JSONL 数据路径；不传时使用 task 默认数据集 | `inference/inference_tsmllm_vllm.py:161-166`, `inference/inference_tsmllm_vllm.py:223-230` |
| `--exp` | `None` | 实验目录名，只是 `exp/` 下的子目录名；不传时由 task 和 model name 自动生成 | `inference/inference_tsmllm_vllm.py:167-172`, `inference/inference_tsmllm_vllm.py:240-267` |
| `--model_path` | 必填 | 本地模型目录或 HuggingFace model id；脚本会检查本地路径或 HF id 形态 | `inference/inference_tsmllm_vllm.py:173-178`, `inference/inference_tsmllm_vllm.py:232-238` |
| `--num_gpus` | `8` | LLMClient 可用 GPU 总数；用于切分 worker 进程 | `inference/inference_tsmllm_vllm.py:179-184`, `inference/inference_tsmllm_vllm.py:299-305` |
| `--num_gpus_per_process` | `2` | 每个模型副本使用的 GPU 数；传给 `LLMClient(... gpus_per_model=...)` | `inference/inference_tsmllm_vllm.py:185-190`, `inference/inference_tsmllm_vllm.py:79-84` |
| `--max_tokens` | `512` | 主脚本构造 SamplingParams 时使用；但见下方注意点，当前 worker 没有实际使用这个对象 | `inference/inference_tsmllm_vllm.py:191-196`, `inference/inference_tsmllm_vllm.py:296` |
| `--temperature` | `0.2` | 同上，主脚本构造 SamplingParams 时使用；当前 worker 实际用内部 temperature | `inference/inference_tsmllm_vllm.py:197-202`, `inference/inference_tsmllm_vllm.py:296` |
| `--max_samples` | `None` | 限制本次最多处理多少条样本；传给 `prepare_batches()` | `inference/inference_tsmllm_vllm.py:203-208`, `inference/inference_tsmllm_vllm.py:274` |
| `--output_name` | `generated_answer.json` | 输出 JSON 文件名，位于 `exp/<exp_name>/` 下 | `inference/inference_tsmllm_vllm.py:209-214`, `inference/inference_tsmllm_vllm.py:319-321` |

`已从代码确认`：用户提到的 `exp_path` 不是 inference 脚本参数。inference 里对应的是 `--exp` 和内部 `exp_dir = os.path.join("exp", exp_name)`，见 `inference/inference_tsmllm_vllm.py:240-267`。`--exp_path` 是 evaluation 脚本参数，README 评估命令见 `README.md:161-170`。

### 2.2 隐含关键参数

| 参数/概念 | 是否 CLI 暴露 | 作用 | 代码证据 |
|---|---:|---|---|
| `batch_size` | 否 | `LLMClient` 默认 `BATCH_SIZE=32`，worker 每轮最多从输入队列取 32 条；`answer_question_list()` 没有暴露这个参数 | `inference/llm_utils.py:30-39`, `inference/llm_utils.py:223-224`, `inference/llm_utils.py:158-172` |
| `tensor_parallel_size` | 否 | vLLM 初始化时设为 `len(gpu_id.split(','))`；实际由 `--num_gpus_per_process` 控制 | `inference/llm_utils.py:142-150`, `inference/llm_utils.py:246-255` |
| `gpu_memory_utilization` | 否 | `vllm-ts` worker 固定为 `0.95` | `inference/llm_utils.py:149` |
| `limit_mm_per_prompt` | 否 | 每个 prompt 最多 50 条 time series | `inference/llm_utils.py:149`, `inference/vllm/chatts_vllm.py:210-211` |
| `CTX_LENGTH` | 否 | vLLM TS worker 的 `max_model_len` 和内部生成 `max_tokens` 都用 `6500` | `inference/llm_utils.py:30-39`, `inference/llm_utils.py:148-149` |
| `engine` | 否 | `answer_question_list()` 固定使用 `engine="vllm-ts"` | `inference/inference_tsmllm_vllm.py:79-84` |
| `system_prompt` | 否 | `LLMClient` 默认 system message 是 `"You are a helpful assistant."` | `inference/llm_utils.py:223-236`, `inference/llm_utils.py:270-275` |

`已从代码确认的重要注意点`：`main()` 构造了 `SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)` 并传给 `answer_question_list()`，见 `inference/inference_tsmllm_vllm.py:296-305`；`LLMClient.llm_batch_generate()` 也把该对象放进队列，见 `inference/llm_utils.py:337-340`。但是 `worker_vllm_ts()` 实际调用 `llm.generate(batch_inputs, sampling_params, ...)` 时使用的是 worker 内部定义的 `sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=CTX_LENGTH, ...)`，见 `inference/llm_utils.py:146-149` 和 `inference/llm_utils.py:171-177`。因此按当前代码阅读，CLI 的 `--max_tokens` 和 `--temperature` 对 `vllm-ts` worker 不生效；这是需要后续验证或修复的风险点。

## 3. 函数调用链

完整调用链可以按三层看：主脚本、LLMClient 多进程调度、vLLM TS multimodal 注册。

```text
python inference/inference_tsmllm_vllm.py ...
  -> import inference.vllm.chatts_vllm
       -> ModelRegistry.register_model("Qwen2TSForCausalLM", ...)
       -> ModelRegistry.register_model("Qwen3TSForCausalLM", ...)
  -> parse_args()
  -> main()
       -> multiprocessing.set_start_method("spawn", force=True)
       -> resolve task / dataset_path / model_path / exp_dir
       -> load_dataset(dataset_path)
       -> get_prompt_suffix(task)
       -> prepare_batches(dataset, max_samples)
       -> append prompt_suffix to each question
       -> AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
       -> calculate text token count + time-series token count
       -> SamplingParams(...)
       -> answer_question_list(question_list, ts_list, model_path, ...)
            -> LLMClient(model_path, engine="vllm-ts", ...)
                 -> spawn worker_vllm_ts processes
            -> LLMClient.wait_for_ready()
            -> LLMClient.llm_batch_generate(question_list, ts_list, ...)
                 -> _apply_chat_template(question)
                 -> package {"prompt": chat_text, "multi_modal_data": {"timeseries": ...}}
                 -> input_queue.put(...)
                 -> worker_vllm_ts()
                      -> vllm.LLM(... tensor_parallel_size=..., limit_mm_per_prompt={"timeseries": 50})
                      -> llm.generate(batch_inputs, sampling_params)
                      -> output_queue.put((answer, idx, ...))
                 -> collect answer_list in original order
            -> LLMClient.kill()
       -> build generated_answer list
       -> json.dump(..., exp/<exp_name>/<output_name>)
```

关键证据：

| 链路节点 | 证据 |
|---|---|
| script main guard | `inference/inference_tsmllm_vllm.py:326-327` |
| 参数解析 | `inference/inference_tsmllm_vllm.py:152-215` |
| 数据、模型、输出目录解析 | `inference/inference_tsmllm_vllm.py:223-267` |
| 加载 dataset 和 prompt suffix | `inference/inference_tsmllm_vllm.py:269-277` |
| token 统计 | `inference/inference_tsmllm_vllm.py:279-294` |
| 调用生成 | `inference/inference_tsmllm_vllm.py:296-306` |
| 写输出文件 | `inference/inference_tsmllm_vllm.py:308-323` |
| LLMClient 创建 worker | `inference/llm_utils.py:223-263` |
| vLLM TS worker 初始化 | `inference/llm_utils.py:142-150` |
| vLLM 自定义模型注册 | `inference/vllm/chatts_vllm.py:404-409`, `inference/vllm/chatts_vllm.py:760-765` |

## 4. 单样本流动过程

### 4.1 JSONL 到 question_list / ts_list

`已从代码确认`：`load_dataset()` 按行读取 JSONL，去掉空行，用 `json.loads(line)` 得到 dict，见 `inference/inference_tsmllm_vllm.py:98-106`。

`已从代码确认`：`prepare_batches()` 的行为如下：

1. 初始化 `question_list` 和 `ts_list`，见 `inference/inference_tsmllm_vllm.py:123-125`。
2. 如果 `max_samples` 为 `None`，处理全部样本；否则处理 `min(len(dataset), max_samples)` 条，见 `inference/inference_tsmllm_vllm.py:136`。
3. 对每条样本，把 `sample["input"]` 直接 append 到 `question_list`，见 `inference/inference_tsmllm_vllm.py:137-140`。
4. 对 `sample.get("timeseries", [])` 中每个 item 调用 `_to_float_list()`：如果是 ndarray 先 `.tolist()`；如果是 list/tuple 递归转换；如果是 `None` 跳过；其他值转为 `float(value)`，见 `inference/inference_tsmllm_vllm.py:127-148`。
5. 每条样本的所有时间序列组成 `ts_series`，再 append 到 `ts_list`，见 `inference/inference_tsmllm_vllm.py:140-149`。

因此单条样本大致变成：

```text
sample = {
  "input": "<完整题面，含 <ts><ts/> 和 Graph Structure>",
  "timeseries": [[...node0...], [...node1...]],
  "output": "<评估答案，推理阶段不使用>"
}

question_list[i] = sample["input"]
ts_list[i] = [
  [float, float, ...],
  [float, float, ...]
]
```

### 4.2 prompt_suffix 与 chat template

`已从代码确认`：`get_prompt_suffix(task)` 从 `inference/prompt.json` 取出任务对应的 `"prompt"` 字段，见 `inference/prompt_utils.py:27-38`。主函数把 suffix 追加为 `question.rstrip() + "\n\n" + prompt_suffix`，见 `inference/inference_tsmllm_vllm.py:272-277`。

`已从代码确认`：`LLMClient._apply_chat_template()` 将每个 question 包装成：

```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": prompt}
]
```

然后调用 tokenizer 的 chat template 并打开 `add_generation_prompt=True`，见 `inference/llm_utils.py:270-275`。

### 4.3 text token 与 time series token 统计

`已从代码确认`：文本 token 统计使用 `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)` 加载 tokenizer，然后对追加 suffix 后的 `question_list[idx]` 调用 `tokenizer.encode(..., add_special_tokens=False)`，见 `inference/inference_tsmllm_vllm.py:279-287`。

`已从代码确认`：time series token 统计不调用模型 processor，而是用脚本内的近似函数 `calculate_ts_tokens()`。它把每条 time series 长度除以 `TS_PATCH_SIZE=8` 后向上取整，再对一个样本内所有 series 求和，见 `inference/inference_tsmllm_vllm.py:37-38` 和 `inference/inference_tsmllm_vllm.py:109-120`。

```text
text_tokens = len(tokenizer.encode(question, add_special_tokens=False))
ts_tokens = sum(ceil(len(series) / 8) for series in ts_list[idx])
num_tokens = text_tokens + ts_tokens
```

`根据代码推断，未由真实运行验证`：这个 `num_tokens` 是输入 token 估计值，用于日志和输出 JSON 的 `num_tokens` 字段，不参与 vLLM 实际调度。实际 vLLM 的 TS token replacement 会根据 encoded TS array 和 `hf_config.ts["patch_size"]` 计算 patch 数，见 `inference/vllm/chatts_vllm.py:371-379`。

### 4.4 LLMClient 如何调用模型

`已从代码确认`：`answer_question_list()` 创建 `LLMClient(model_path=model_path, engine="vllm-ts", num_gpus=num_gpus, gpus_per_model=num_gpus_per_process)`，等待 worker ready，然后调用 `llm_batch_generate(question_list, ts_list, sampling_params=...)`，最后无论成功失败都会 `kill()` worker，见 `inference/inference_tsmllm_vllm.py:70-95`。

`已从代码确认`：`LLMClient.__init__()` 使用 `multiprocessing.Manager()` 创建输入/输出队列和 ready flag；根据 `num_gpus` 构造 `gpu_range=list(range(num_gpus))`；再按 `gpus_per_model` 分组启动 worker。每个 worker 的 `gpu_id_str` 如 `"0,1"`，见 `inference/llm_utils.py:223-263`。

`已从代码确认`：`worker_vllm_ts()` 会设置 `CUDA_VISIBLE_DEVICES=gpu_id`，导入 vLLM 和 `inference.vllm.chatts_vllm`，然后初始化：

```python
LLM(
  model=model_path,
  trust_remote_code=True,
  max_model_len=CTX_LENGTH,
  tensor_parallel_size=len(gpu_id.split(",")),
  gpu_memory_utilization=0.95,
  limit_mm_per_prompt={"timeseries": 50},
  enable_prefix_caching=False,
)
```

证据见 `inference/llm_utils.py:142-150`。

`已从代码确认`：`llm_batch_generate()` 对每条 prompt 先应用 chat template，再把 `timeseries` 放到 `multi_modal_data`，然后把任务放进输入队列；worker 批量取出后调用 `llm.generate(batch_inputs, sampling_params, use_tqdm=False)`，生成文本后放入输出队列；主进程根据 idx 收集结果并恢复原顺序，见 `inference/llm_utils.py:311-362` 和 `inference/llm_utils.py:158-181`。

`已从代码确认`：vLLM 侧的 `Qwen2TSMultiModalProcessor` 负责解析 `timeseries` modality，并强制 `vllm_flag=True` 调用 HF processor，见 `inference/vllm/chatts_vllm.py:282-330`。HF processor 的 vLLM 路径会对每条序列调用 `sp_encoding()`，返回 `zip(ts_tokens, encoded_ts_arrays)`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:104-118` 和 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:164-171`。

`已从代码确认`：vLLM prompt replacement 会把 `<ts><ts/>` 目标 token 替换成带 TS embedding placeholder 的 token 序列，见 `inference/vllm/chatts_vllm.py:352-401`。`Qwen3TSForCausalLM` 再把 `timeseries` 编码成 `ts_features`，并通过 `merge_multimodal_embeddings()` 合并进语言模型输入 embedding，见 `inference/vllm/chatts_vllm.py:587-709`。

## 5. 输出文件格式

`已从代码确认`：推理输出目录是 `exp/<exp_name>/`。如果传 `--exp`，`exp_name=args.exp`；否则从 `model_path` 推导 model name，再拼成 `f"{task}-{model_name}"`，见 `inference/inference_tsmllm_vllm.py:240-267`。

`已从代码确认`：输出文件名由 `--output_name` 控制，默认 `generated_answer.json`，见 `inference/inference_tsmllm_vllm.py:209-214` 和 `inference/inference_tsmllm_vllm.py:319-321`。

当前代码写出的 JSON 是一个 list，每个元素格式如下：

```json
{
  "idx": 0,
  "question_text": "<追加 prompt_suffix 后的题面>",
  "response": "<模型生成文本>",
  "num_tokens": 1234
}
```

字段含义：

| 字段 | 来源 | 证据 |
|---|---|---|
| `idx` | `answers.items()` 中的样本索引；与 `question_list[idx]` 对齐 | `inference/inference_tsmllm_vllm.py:308-316` |
| `question_text` | 已追加 prompt_suffix 的 question | `inference/inference_tsmllm_vllm.py:275-277`, `inference/inference_tsmllm_vllm.py:312-314` |
| `response` | vLLM 返回的生成文本 | `inference/inference_tsmllm_vllm.py:93-95`, `inference/inference_tsmllm_vllm.py:313-314` |
| `num_tokens` | `text_tokens + ts_tokens` 的输入 token 估计值 | `inference/inference_tsmllm_vllm.py:282-290`, `inference/inference_tsmllm_vllm.py:315` |

`已从代码确认`：本地已有 `exp_STReasoner-8B/*/generated_answer.json` 文件包含 `idx`、`question_text`、`response` 字段，但没有当前代码会写出的 `num_tokens` 字段；这说明本地示例输出可能来自旧版脚本或不同导出逻辑。评估代码兼容 flat `response` 结构，见 `evaluation/evaluate_qa.py:82-127`。

## 6. 小规模 sanity check 建议

`不实际运行大模型`：下面只是参数设置建议。当前仓库缺少 `data/ST-Bench/` 原始 JSONL 和本地合并后的模型 checkpoint，直接运行会在 dataset 或 model path 检查处失败，见 `inference/inference_tsmllm_vllm.py:223-238`。

如果后续已有本地模型和数据，只做最小 sanity check，建议：

```bash
python inference/inference_tsmllm_vllm.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --model_path checkpoints/easy_r1/qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/huggingface \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_samples 1 \
  --output_name sanity_generated_answer.json \
  --exp sanity_reasoning_entity
```

建议理由：

| 设置 | 理由 |
|---|---|
| `--max_samples 1` | 只构造和生成 1 条样本，降低等待时间和输出量 |
| `--num_gpus 1 --num_gpus_per_process 1` | 只启动 1 个 worker，vLLM `tensor_parallel_size=1`，便于定位问题 |
| `--dataset .../entity_test.jsonl` | 显式传数据路径，避免默认路径和本地数据布局不一致 |
| `--exp sanity_*` + `--output_name sanity_generated_answer.json` | 不覆盖正式实验输出 |
| `--task reasoning_entity` | 多选任务输出短，便于肉眼检查 `<answer>A-D</answer>` 格式 |

注意事项：

1. `已从代码确认`：脚本没有 CLI 参数控制 `batch_size`，默认 worker batch size 是 32，见 `inference/llm_utils.py:38` 和 `inference/llm_utils.py:223-224`。`--max_samples 1` 会让实际队列里只有 1 条样本。
2. `已从代码确认`：当前 `--max_tokens` 和 `--temperature` 对 `vllm-ts` worker 可能不生效，原因见第 2 节的 SamplingParams 注意点。
3. `根据代码推断，未由真实运行验证`：如果机器只有单卡，`--num_gpus_per_process 1` 才能让 `tensor_parallel_size=1`；默认值 2 会尝试每个 worker 使用两个 GPU。
4. `尚未确认`：如果使用 HuggingFace model id 作为 `--model_path`，vLLM/Transformers 可能触发模型下载；本周不建议这样做。优先使用已经存在的本地模型目录。

## 7. 组会讲法

### 本文档核心结论

`已从代码确认`：inference 主链路是 `JSONL -> question_list/ts_list -> prompt_suffix -> chat template -> multi_modal_data.timeseries -> vLLM TS model -> generated_answer.json`。其中 `input` 负责文本题面，`timeseries` 负责真实数值序列，`prompt_suffix` 只约束输出格式。

`已从代码确认`：`num_gpus_per_process` 实际决定 vLLM 的 `tensor_parallel_size`；`batch_size` 不在 CLI 暴露；`max_samples` 是最适合 sanity check 的限制参数。

`已从代码确认`：当前代码中 `--max_tokens` 和 `--temperature` 被主进程构造出来，但 `worker_vllm_ts()` 使用自己的内部 SamplingParams，因此这两个 CLI 参数对实际生成不一定生效。这是组会可以如实汇报的代码阅读发现。

### 组会可讲版本

推理脚本的入口很清楚：先按 `--task` 找默认测试集，或者用 `--dataset` 覆盖；再读 JSONL，每条样本的 `input` 进入 `question_list`，`timeseries` 转成 float list 进入 `ts_list`。脚本会给 question 追加任务级输出格式，例如多选题要求 `<answer>` 里只放 A-D，forecasting 要放数字序列。

真正送模型时，`LLMClient` 会启动一个或多个 vLLM worker。每条输入先被 Qwen chat template 包成 system/user/assistant 形式，再和 `multi_modal_data.timeseries` 一起交给 vLLM。vLLM 侧注册了自定义 `Qwen3TSForCausalLM`，它会把 `<ts><ts/>` 对应的时间序列编码成 embedding，再合并进语言模型输入。

输出是 `exp/<任务-模型名>/generated_answer.json`，每条包含 `idx`、`question_text`、`response` 和当前代码新增的 `num_tokens`。如果只做 sanity check，最重要是 `--max_samples 1`、单卡参数、单独的 `--exp` 和 `--output_name`，但仍然会加载大模型，所以本周只建议作为后续复现计划，不建议现在跑。

### 后续需要验证的问题

1. `尚未确认`：在真实环境跑 1 条样本，确认当前 `generated_answer.json` 是否包含 `num_tokens`，以及是否能被 evaluation 读取。
2. `尚未确认`：验证 `--max_tokens` 和 `--temperature` 是否真的不生效；代码阅读显示 worker 使用内部 SamplingParams，但需要运行日志或 patch 后对比确认。
3. `尚未确认`：下载真实 ST-Test 后，检查 `prepare_batches()` 对所有 `timeseries` 嵌套格式是否都能正确转 float。
4. `尚未确认`：单卡 `--num_gpus 1 --num_gpus_per_process 1` 能否成功加载合并后的 STReasoner-8B，需要本地模型和显存条件验证。
