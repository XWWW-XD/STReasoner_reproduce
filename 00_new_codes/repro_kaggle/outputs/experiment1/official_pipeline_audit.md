# 官方流程审计报告

## 1. 目标

本报告用于比较作者原始推理/评测链路和 `repro_kaggle` 辅助评测链路的差异，判断当前 tiny eval 结果是否能代表作者官方评测方式。审计仅基于静态代码阅读；未运行模型、未运行训练、未下载 checkpoint、未安装依赖。

## 2. 作者官方推理流程

### 入口文件

- README 将官方推理入口写为 `python inference/inference_tsmllm_vllm.py --task ... --model_path ...`，并在 `reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation` 上循环运行：`README.md:143-158`。
- `inference/inference_tsmllm_vllm.py` 自身声明需要 `vllm==0.8.5`，并且必须先 import `inference.vllm.chatts_vllm` 注册自定义 ChatTS/vLLM 模块：`inference/inference_tsmllm_vllm.py:15-24`。
- `scripts/qwen3-*` 下未发现独立推理/评测入口；这些脚本主要是训练入口。它们确实配置了 `data.val_files` 指向 ST-Test、`data.prompt_key=input`、`data.ts_key=timeseries`、`data.answer_key=output`、`data.format_prompt=.../str.jinja`，例如 `scripts/qwen3-8b/train_stage2+3_w_spatial.sh:7-22`，但不是 README 所示的官方推理/评测入口。

### 模型加载方式

- 官方推理通过 `LLMClient(model_path=model_path, engine="vllm-ts", num_gpus=..., gpus_per_model=...)` 创建 vLLM-TS 客户端：`inference/inference_tsmllm_vllm.py:70-84`。
- `LLMClient` 对 `engine == 'vllm-ts'` 启动 worker：`inference/llm_utils.py:247-255`。
- vLLM-TS worker 内部构造 `LLM(model=model_path, trust_remote_code=True, max_model_len=CTX_LENGTH, tensor_parallel_size=..., gpu_memory_utilization=0.95, limit_mm_per_prompt={"timeseries": 50}, enable_prefix_caching=False)`：`inference/llm_utils.py:142-149`。
- vLLM 注册了 `Qwen3TSForCausalLM`：`inference/vllm/chatts_vllm.py:760-765`。

### processor / tokenizer 使用方式

- 官方推理入口只显式加载 `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` 用于统计 text tokens：`inference/inference_tsmllm_vllm.py:279-287`。
- `LLMClient` 也加载 `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`，用于套 chat template：`inference/llm_utils.py:223-239`。
- 真正的时序 multimodal processor 由 vLLM 注册链路调用。vLLM processor 会设置 `mm_kwargs['vllm_flag'] = True` 再调用 HF processor：`inference/vllm/chatts_vllm.py:312-330`。
- base model 的 `Qwen3TSProcessor` 在初始化时继承 tokenizer 的 `chat_template`，但其 `__call__` 本身只是重建 `<ts><ts/>` 占位符并调用 tokenizer，不主动调用 `apply_chat_template`：`base_model/Config-Qwen3-8B/processing_qwen3_ts.py:61-70`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:157-171`。

### prompt 构造方式

- 官方推理从数据样本取 `sample["input"]`：`inference/inference_tsmllm_vllm.py:123-149`。
- 然后按 task 从 `inference/prompt.json` 读取 prompt suffix：`inference/prompt_utils.py:27-38`。
- 入口会把 suffix 追加到每个问题末尾：`question.rstrip() + "\n\n" + prompt_suffix`，见 `inference/inference_tsmllm_vllm.py:272-277`。
- `inference/prompt.json` 对 forecasting 要求 `<think>...</think><answer>[...]</answer>`，对选择题要求 `<think>...</think><answer>单个大写字母</answer>`：`inference/prompt.json:5-16`。
- 训练/rollout 的 `str.jinja` 也表达了同样意图：forecasting 根据是否含 `Historical observation window` 走数值序列 answer，否则走单个大写选项：`src/EasyR1/examples/format_prompt/str.jinja:1-7`。

### timeseries 输入如何传入

- 官方推理把每个样本的 `timeseries` 转为 float/list，并与 question 对齐成 `question_list, ts_list`：`inference/inference_tsmllm_vllm.py:123-149`。
- 调用 `llm_client.llm_batch_generate(question_list, ts_list, sampling_params=...)`：`inference/inference_tsmllm_vllm.py:85-89`。
- `LLMClient.llm_batch_generate` 默认 `use_chat_template=True`；若传入 `batch_timeseries`，它将输入包装成 `{"prompt": inputs, "multi_modal_data": {"timeseries": batch_timeseries[i]}}`：`inference/llm_utils.py:311-340`。
- base processor 的非 vLLM 路径会把 `<ts><ts/>` 替换为带 offset/scaling/length/max/min/left/right 的 `<ts>...<ts/>` token 文本，并返回 `timeseries` tensor：`base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:122-171`。
- vLLM 路径下，processor 不直接改 prompt，而是返回 `timeseries` 的 token/update 信息；vLLM prompt replacement 后续处理 `<ts><ts/>`：`base_model/Config-Qwen3-8B/processing_qwen3_ts.py:104-118`, `inference/vllm/chatts_vllm.py:352-400`。
- 模型的 HF `generate`/forward 也显式支持 `timeseries`，会在首轮 forward 中编码并 merge 时序 embedding：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:307-332`, `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-589`。

### generation 参数

- 官方入口默认 `SamplingParams(max_tokens=512, temperature=0.2)`：`inference/inference_tsmllm_vllm.py:64-68`。
- CLI 也提供 `--max_tokens` 默认 `512`、`--temperature` 默认 `0.2`：`inference/inference_tsmllm_vllm.py:191-201`。
- 实际生成前用 `SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)`：`inference/inference_tsmllm_vllm.py:296`。
- `LLMClient` 的 vLLM-TS worker 如果未传入 sampling params，会用内部默认 `temperature=0.5, top_p=0.95, max_tokens=CTX_LENGTH`；但官方入口传入了自己的 `SamplingParams`，覆盖默认值：`inference/llm_utils.py:146-149`, `inference/llm_utils.py:339-340`。

### 输出保存格式

- 官方推理输出 `exp/<exp_name>/generated_answer.json`，默认文件名来自 `--output_name generated_answer.json`：`inference/inference_tsmllm_vllm.py:209-214`, `inference/inference_tsmllm_vllm.py:319-322`。
- 每条记录字段为 `idx`, `question_text`, `response`, `num_tokens`：`inference/inference_tsmllm_vllm.py:308-317`。

### 是否使用 vLLM

- 是。README 推理入口是 `inference_tsmllm_vllm.py`：`README.md:143-158`。
- 入口 import vLLM `SamplingParams` 且注释要求 vLLM：`inference/inference_tsmllm_vllm.py:15-24`。
- 实际生成调用 vLLM worker 的 `llm.generate(...)`：`inference/llm_utils.py:171-177`。

### 是否使用 chat template

- 是。`LLMClient.llm_batch_generate(..., use_chat_template=True)` 默认开启：`inference/llm_utils.py:311`。
- `_apply_chat_template` 构造 system/user 对话并调用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`：`inference/llm_utils.py:270-275`。
- 默认 system prompt 是 `You are a helpful assistant.`：`inference/llm_utils.py:223-224`。

### 是否要求 `<answer>...</answer>`

- 官方 prompt 明确要求 `<think>...</think><answer>...</answer>`：`inference/prompt.json:2-19`。
- 训练奖励也 fullmatch `<think>.*?</think><answer>.*?</answer>`，说明格式在训练/RL 中被奖励：`src/EasyR1/examples/reward_function/str.py:12-15`。
- forecasting 奖励特别要求数值预测在 `<answer>...</answer>` 内，否则给 0：`src/EasyR1/examples/reward_function/str.py:82-90`。

## 3. 作者官方评测流程

### 入口文件

- README 官方评测入口是 `python evaluation/evaluate.py --task ... --exp_path exp/...`：`README.md:162-172`。
- `evaluation/evaluate.py` 解析 `--exp_path`, `--dataset`, `--task`, `--pred_pattern generated_answer`, `--repo_root`：`evaluation/evaluate.py:100-132`。

### 读取哪些预测文件

- `evaluate.py` 默认 `--pred_pattern` 是 `generated_answer`：`evaluation/evaluate.py:120-125`。
- `load_prediction_files` 在 `exp_dir` 中读取文件名包含 pattern 且以 `.json` 结尾的文件：`evaluation/evaluate_qa.py:82-93`。
- 预测文件支持 flat top-level `response`，也支持 `responses` 列表；最终按 `idx` 存到 predictions：`evaluation/evaluate_qa.py:101-127`。

### 如何解析答案

- 读取 response 后先 `_extract_tag_content(text)`，如果有 `<answer>...</answer>` 则取 tag 内内容，否则保留整段文本：`evaluation/evaluate_qa.py:36-45`, `evaluation/evaluate_qa.py:123-127`。
- 选择题 `_normalize_choice` 再次提取 `<answer>`，并只在字符串开头匹配 `[A-Da-d]`：`evaluation/evaluate_qa.py:23-33`。
- forecasting `_parse_series` 先尝试 JSON list/number，失败则用正则抽取所有数字：`evaluation/evaluate_qa.py:47-66`。

### 如何计算 accuracy / MAE / 其他指标

- alignment 计算 `overall_score`, `exact_match`, `relative_accuracy`：`evaluation/evaluate_qa.py:142-195`。
- forecasting 计算每样本 MAE、MAPE、target stats、missing indices：`evaluation/evaluate_qa.py:198-275`。
- multiple choice 计算 `accuracy = correct / evaluated`：`evaluation/evaluate_qa.py:278-317`。
- `evaluate_predictions_for_task` 按 task 分发：`evaluation/evaluate_qa.py:320-330`。

### forecasting 是否单独处理

- 是。`reasoning_forecasting` 进入 `evaluate_forecasting_predictions`：`evaluation/evaluate_qa.py:326-327`。
- forecasting 会把预测序列短的用最后一个值 padding，长的截断到 target 长度：`evaluation/evaluate_qa.py:223-228`。

### A/B/C/D 选择题是否单独处理

- 是。`reasoning_entity`, `reasoning_etiological`, `reasoning_correlation`, `reasoning_causal` 进入 `evaluate_multiple_choice_predictions`：`evaluation/evaluate_qa.py:328-329`。
- 选择题通过 `_normalize_choice` 对 target/prediction 归一化后比较：`evaluation/evaluate_qa.py:286-300`。

### 是否依赖 `<answer>...</answer>`

- 评测代码不硬性要求 tag；如果没有 tag，会 fallback 到原始文本：`evaluation/evaluate_qa.py:36-45`。
- 但官方推理 prompt、训练格式奖励、forecasting 奖励都强烈要求 tag：`inference/prompt.json:2-19`, `src/EasyR1/examples/reward_function/str.py:12-15`, `src/EasyR1/examples/reward_function/str.py:82-90`。

## 4. repro_kaggle 当前 tiny eval 流程

### 入口脚本

- 当前 tiny eval 入口是 `repro_kaggle/scripts/05_eval_sttest_tiny.py`；文件注释明确说它不 import 作者训练或评测入口：`repro_kaggle/scripts/05_eval_sttest_tiny.py:1-6`。
- `06_compare_single4bit_dualfp16.sh` 调用 tiny eval 三次做 single/dual 对比：`repro_kaggle/scripts/06_compare_single4bit_dualfp16.sh:21-55`。
- `07_parse_fix_experiment.sh` 调用 tiny eval 做 `64/256` 和 answer prompt 对比：`repro_kaggle/scripts/07_parse_fix_experiment.sh:21-61`。

### 模型加载方式

- `03_load_streasoner_smoke.py` 使用 `AutoProcessor`, `AutoTokenizer`, `AutoConfig`, `AutoModelForCausalLM` / `AutoModel` 直接从 HF 加载：`repro_kaggle/scripts/03_load_streasoner_smoke.py:144-165`, `repro_kaggle/scripts/03_load_streasoner_smoke.py:168-181`, `repro_kaggle/scripts/03_load_streasoner_smoke.py:201-236`。
- tiny eval 复用 smoke 脚本里的 loader，并根据 `precision` 构造 4bit 或 fp16：`repro_kaggle/scripts/05_eval_sttest_tiny.py:39-50`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:314-328`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:346-397`。
- tiny eval 支持 `single_gpu`, `dual_auto`, `dual_balanced` 三种 `device_map`，不是官方 vLLM tensor parallel 链路：`repro_kaggle/scripts/05_eval_sttest_tiny.py:331-343`。

### processor / tokenizer 使用方式

- tiny eval 调 `load_processor_and_tokenizer`，即 `AutoProcessor.from_pretrained` 与 `AutoTokenizer.from_pretrained`：`repro_kaggle/scripts/03_load_streasoner_smoke.py:144-165`。
- 生成输入由 `processor(text=prompt, timeseries=timeseries, return_tensors="pt")` 构造：`repro_kaggle/scripts/05_eval_sttest_tiny.py:573-593`。
- 解码时优先使用 tokenizer，截掉 input_ids 长度后的 generated ids，再 `decode(..., skip_special_tokens=True)`：`repro_kaggle/scripts/04_run_one_sttest_sample.py:172-182`。

### prompt 构造方式

- tiny eval 默认直接使用 HF dataset 样本的 `sample["input"]`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:577-578`。
- 默认不追加官方 `inference/prompt.json` 的 task-specific suffix；只有 `--answer_format_prompt true` 时追加临时 `ANSWER_FORMAT_INSTRUCTION`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:60-64`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:590-591`。
- 该临时 instruction 只覆盖 A/B/C/D tag，不覆盖 forecasting 的数值序列格式：`repro_kaggle/scripts/05_eval_sttest_tiny.py:60-64`。

### timeseries 输入方式

- tiny eval 从 `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train` 读取样本：`repro_kaggle/scripts/05_eval_sttest_tiny.py:54-56`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:250-259`。
- 会检查 `<ts><ts/>` placeholder 数量必须等于 `timeseries` 节点数：`repro_kaggle/scripts/05_eval_sttest_tiny.py:584-588`。
- 然后通过 HF processor 的非 vLLM 路径把 timeseries 合成 `timeseries` tensor，而不是官方 vLLM `multi_modal_data` 包装：`repro_kaggle/scripts/05_eval_sttest_tiny.py:593`。

### generation 参数

- tiny eval CLI 默认 `--max_new_tokens 64`，`--answer_format_prompt false`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:85-86`。
- 生成使用 `model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:605-619`。
- `07_parse_fix_experiment.sh` 的基线分支显式传 `--max_new_tokens 64`；其它分支传 `256`：`repro_kaggle/scripts/07_parse_fix_experiment.sh:21-61`。

### answer parser 规则

- tiny eval 的 `parse_choice` 优先解析 `<answer>...</answer>`，否则接受纯 `A/B/C/D`，再用 `CHOICE_RE` 在整段文本中找大写 A/B/C/D：`repro_kaggle/scripts/05_eval_sttest_tiny.py:67-70`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:143-170`。
- 同一 parser 同时用于 target 和 prediction：`repro_kaggle/scripts/05_eval_sttest_tiny.py:972-983`。

### forecasting 处理方式

- tiny eval 把 `forecasting` 放进 `TARGET_CATEGORIES`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:65`。
- 但 scoring 仍调用 `parse_choice`，没有单独解析数值序列、没有 MAE/MAPE：`repro_kaggle/scripts/05_eval_sttest_tiny.py:972-983`。
- 因此 forecasting 样本基本会被标成 parse fail 或被错误地当成选择题处理，不能代表官方 forecasting 评测。

### 输出 jsonl / summary 字段

- 每条 jsonl record 包含 `index`, `category`, `input_preview`, `output`, `prediction`, `parsed_prediction`, `is_correct`, latency、precision、device map、parse/error 字段等：`repro_kaggle/scripts/05_eval_sttest_tiny.py:622-651`。
- 每条样本即时 append 到 jsonl：`repro_kaggle/scripts/05_eval_sttest_tiny.py:1000-1001`。
- summary 包含 `max_new_tokens`, `answer_format_prompt`, generate/parse success/fail、parse_fail_rate、accuracy_by_category、GPU memory 等：`repro_kaggle/scripts/05_eval_sttest_tiny.py:765-800`。

## 5. 差异表

| 维度 | 作者原始代码 | repro_kaggle 当前代码 | 是否一致 | 可能影响 |
|---|---|---|---|---|
| 模型加载 | vLLM-TS `LLM(... trust_remote_code=True, tensor_parallel_size=..., limit_mm_per_prompt={"timeseries": 50})`，`inference/llm_utils.py:146-149` | HF `AutoModelForCausalLM.from_pretrained(... device_map=..., torch_dtype=float16, optional 4bit)`，`repro_kaggle/scripts/05_eval_sttest_tiny.py:370-381` | 否 | 运行时、显存、batch 行为、时序 multimodal 路径都不同 |
| processor 使用 | vLLM 内部 multimodal processor，`vllm_flag=True`，`inference/vllm/chatts_vllm.py:312-330` | 直接 `AutoProcessor(...); processor(text=..., timeseries=..., return_tensors="pt")`，`repro_kaggle/scripts/05_eval_sttest_tiny.py:573-593` | 部分一致 | 都用远端/模型 processor 语义，但一个是 vLLM 路径，一个是 HF 非 vLLM 路径 |
| tokenizer 使用 | `AutoTokenizer` 用于 chat template 和 token count，`inference/llm_utils.py:223-239`, `inference/inference_tsmllm_vllm.py:279-287` | `AutoTokenizer` 用于解码；processor 负责 tokenization，`repro_kaggle/scripts/04_run_one_sttest_sample.py:172-182` | 部分一致 | 解码一致性较接近，但 prompt encoding 路径不同 |
| timeseries 输入 | `multi_modal_data: {"timeseries": ...}` 交给 vLLM，`inference/llm_utils.py:330-335` | processor 返回 `timeseries` tensor 后直接传入 HF `generate`，`repro_kaggle/scripts/05_eval_sttest_tiny.py:593`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:618` | 否 | 可能造成输出差异；多 GPU 下还需要我们 runtime patch |
| prompt 文本 | 样本 input + task-specific `inference/prompt.json` suffix，`inference/inference_tsmllm_vllm.py:272-277` | 默认只用样本 input；可选临时 A/B/C/D answer instruction，`repro_kaggle/scripts/05_eval_sttest_tiny.py:590-591` | 否 | 最可能导致没有 `<answer>`、输出继续长推理、parse fail |
| chat template | 默认 system/user chat template，`inference/llm_utils.py:270-275`, `inference/llm_utils.py:311` | tiny eval 没有 `apply_chat_template`，直接 processor(text=prompt)，`repro_kaggle/scripts/05_eval_sttest_tiny.py:593` | 否 | 模型看到的对话格式不同，可能影响输出格式和质量 |
| max_new_tokens / max_tokens | vLLM `max_tokens=512` 默认，`inference/inference_tsmllm_vllm.py:191-201`, `inference/inference_tsmllm_vllm.py:296` | HF `max_new_tokens=64` 默认，`repro_kaggle/scripts/05_eval_sttest_tiny.py:85`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:618` | 否 | 64 很容易截断 `<answer>`，显著增加 parse fail |
| do_sample / temperature | vLLM `temperature=0.2`，未显式 `do_sample`，`inference/inference_tsmllm_vllm.py:296` | HF `do_sample=False`，未设 temperature，`repro_kaggle/scripts/05_eval_sttest_tiny.py:618` | 否 | 随机性/确定性不同，答案和长度分布可能不同 |
| answer extraction | 官方先取 `<answer>`；无 tag 则 fallback；选择题只从开头解析字母，`evaluation/evaluate_qa.py:23-45` | tiny eval 搜索 `<answer>`，无 tag 时在整段找大写 A/B/C/D，`repro_kaggle/scripts/05_eval_sttest_tiny.py:143-170` | 否 | tiny eval 可能既更宽松，也可能误抓推理中的字母；parse_fail/accuracy 不可直接对齐官方 |
| forecasting evaluation | 单独解析数值序列并算 MAE/MAPE，`evaluation/evaluate_qa.py:198-275` | 当选择题解析，`repro_kaggle/scripts/05_eval_sttest_tiny.py:972-983` | 否 | forecasting 指标不可用，parse fail 被夸大 |
| output format | `generated_answer.json` list，字段 `idx/question_text/response/num_tokens`，`inference/inference_tsmllm_vllm.py:308-322` | jsonl + summary，自定义字段，`repro_kaggle/scripts/05_eval_sttest_tiny.py:622-651`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:765-800` | 否 | 官方 `evaluation/evaluate.py` 不能直接消费 tiny eval jsonl |
| batch / vLLM | 多进程 vLLM-TS batch queue，`inference/llm_utils.py:142-181`, `inference/llm_utils.py:311-362` | 单进程逐样本 HF generate，`repro_kaggle/scripts/05_eval_sttest_tiny.py:949-1001` | 否 | 性能和输出路径不同；tiny eval 更像 smoke test |
| 数据入口 | 本地 `data/ST-Bench/ST-Test/*_test.jsonl`，`inference/inference_tsmllm_vllm.py:46-57` | HF dataset `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`，`repro_kaggle/scripts/05_eval_sttest_tiny.py:54-56`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:250-259` | 部分一致/需验证 | 样本内容可能同源，但 split/文件组织不同；当前报告未逐条校验数据等价 |

## 6. 风险评估

### 哪些部分已经接近作者官方逻辑

- 使用同一个公开模型 ID `Time-HD-Anonymous/STReasoner-8B`，并启用 `trust_remote_code=True`，接近官方 checkpoint 使用方式：`repro_kaggle/scripts/05_eval_sttest_tiny.py:370-381`。
- 使用 `AutoProcessor` 接受 `text + timeseries`，且检查 `<ts><ts/>` 数量和 timeseries 节点数一致，接近 base processor 对输入结构的要求：`repro_kaggle/scripts/05_eval_sttest_tiny.py:573-593`, `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:139-143`。
- 生成后保存 prediction、gold output、parse result、latency 和设备信息，对工程调试有用：`repro_kaggle/scripts/05_eval_sttest_tiny.py:622-651`。

### 哪些部分只是我们的临时辅助实现

- 4bit/single GPU/dual_auto/dual_balanced 评测矩阵是 Kaggle 辅助实验，不是 README 官方流程：`repro_kaggle/scripts/06_compare_single4bit_dualfp16.sh:21-55`。
- `max_new_tokens=64` 和 `answer_format_prompt` 是辅助脚本参数，不是作者官方推理参数：`repro_kaggle/scripts/05_eval_sttest_tiny.py:85-86`。
- 多 GPU device mismatch runtime patch 是我们为 HF device_map 路径加的工程补丁，不存在于作者 vLLM-TS 链路：`repro_kaggle/scripts/05_eval_sttest_tiny.py:454-570`。
- tiny eval 的 parser 和 summary 体系是辅助评测，不是 `evaluation/evaluate.py`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:143-170`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:765-800`。

### 哪些差异可能导致 parse fail

- 最大风险是没有默认追加官方 `inference/prompt.json` 的 `<think>/<answer>` 输出格式。官方会追加：`inference/inference_tsmllm_vllm.py:272-277`；tiny eval 默认不追加：`repro_kaggle/scripts/05_eval_sttest_tiny.py:590-591`。
- 第二大风险是生成长度：官方默认 `max_tokens=512`，tiny eval 默认 `max_new_tokens=64`，容易在 `<answer>` 前截断：`inference/inference_tsmllm_vllm.py:191-201`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:85`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:618`。
- 第三大风险是缺少官方 chat template。官方默认套 system/user template：`inference/llm_utils.py:270-275`；tiny eval 直接 `processor(text=prompt, ...)`：`repro_kaggle/scripts/05_eval_sttest_tiny.py:593`。
- forecasting 被当选择题 parse，本身会制造 parse fail：`repro_kaggle/scripts/05_eval_sttest_tiny.py:65`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:972-983`。

### 哪些差异可能导致 accuracy 不可信

- tiny eval 混合四类任务，但只用选择题 parser/scorer；forecasting 的 accuracy 不对应官方 MAE/MAPE：`evaluation/evaluate_qa.py:198-275`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:972-983`。
- tiny eval 的 parse_success 子集上计算 `accuracy_overall_if_applicable`，parse fail 样本被排除；官方 multiple choice 则按可读取预测覆盖率和 evaluated 计算 accuracy：`repro_kaggle/scripts/05_eval_sttest_tiny.py:718-728`, `evaluation/evaluate_qa.py:286-317`。
- tiny eval parser 在全文搜索大写 A/B/C/D，可能从推理文本中误抓选项字母；官方 `_normalize_choice` 更偏向 tag 或开头字母：`repro_kaggle/scripts/05_eval_sttest_tiny.py:143-170`, `evaluation/evaluate_qa.py:23-33`。
- tiny eval 使用 HF 非 vLLM 路径，官方使用 vLLM-TS；即便同一模型，输入包装和生成后端不同：`inference/llm_utils.py:330-335`, `repro_kaggle/scripts/05_eval_sttest_tiny.py:593`。

### 当前结果适合如何汇报

当前 tiny eval 适合汇报为“工程可运行性实验 / smoke and engineering validation”，尤其用于确认模型能否加载、timeseries processor 是否可用、不同 device strategy 是否会 OOM 或 device mismatch。

当前 tiny eval 不适合汇报为“正式复现结果”。原因是官方 prompt suffix、chat template、vLLM-TS、generation 参数、官方 evaluation parser/metrics 都没有完全复用。

## 7. 建议

1. 优先复用作者原始 `inference/inference_tsmllm_vllm.py` 生成 `generated_answer.json`，再用 `evaluation/evaluate.py` 评测。README 官方流程就是这个组合：`README.md:143-172`。
2. 如果 Kaggle 环境暂时不能跑 vLLM-TS，也应先把 `repro_kaggle` 的 prompt 改到和官方一致：按 category/task 追加 `inference/prompt.json` 的 suffix，而不是默认裸 `sample["input"]`。相关官方代码在 `inference/prompt_utils.py:27-38` 和 `inference/inference_tsmllm_vllm.py:272-277`。
3. 把 `repro_kaggle` 的 answer parser 对齐官方 `evaluation/evaluate_qa.py`，或者直接产出官方 `generated_answer.json` 格式后调用官方 evaluator。官方读取和解析逻辑在 `evaluation/evaluate_qa.py:82-127`。
4. forecasting 必须单独评估：按官方 `_parse_series`、padding/truncation、MAE/MAPE 逻辑处理，不能继续走 `parse_choice`。官方逻辑在 `evaluation/evaluate_qa.py:198-275`。
5. 将当前 tiny eval 明确保留为 smoke / engineering validation，而非正式结果。正式复现至少应满足：官方 prompt suffix、chat template、足够接近的 generation 长度、官方 output format、官方 evaluator。
6. parse fail 实验的最小修正优先级：先启用官方 prompt suffix，再把生成长度从 64 提到接近官方 512，最后再比较是否需要额外 answer-format prompt。当前自定义 prompt 只覆盖选择题，无法修正 forecasting。
