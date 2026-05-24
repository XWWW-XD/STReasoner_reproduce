# Experiment 1 4bit 输出异常排查报告

## 原始排查提示词

在实验一的文档中写一个新文档，先把这个提示词放进文档，再开始排查这个问题，查到后只是写清问题，先不要改动其他代码——当前发现一个输出异常样例：模型在输出答案 C 之后，生成了一个奇怪字符 `ś`，随后又出现了类似重复推理的内容，并且中间还出现了 `</think>`。我判断这可能是生成未正确停止、stop/eos 设置不充分、decode 后处理不充分，或模型习惯性生成 reasoning tag 导致的。

现在请你做两件事：

一、先检查并修正生成与输出保存逻辑

重点检查以下位置：

1. generate 调用中是否正确设置了：

- eos_token_id

- pad_token_id

- max_new_tokens

- do_sample=False

- temperature 是否在非采样时被忽略或不生效

2. decode 时是否使用了：

- skip_special_tokens=True

3. 是否需要增加保守的后处理截断逻辑，但注意不要影响官方评测文件：

- 如果模型已经出现明确答案，例如 `Answer: C`、`**Answer: C...**`、`答案：C`，可以在 diagnostic 输出中标记 first_answer_position。

- 不要直接删除 raw_output，raw_output 必须完整保存。

- 可以额外保存 clean_output，用于观察截断后的版本。

- 如果输出中出现 `</think>`、重复 Answer、乱码字符、答案后继续长篇重复，请在 strict diagnostic layer 里记录对应 flags，例如：

- has_think_tag

- has_repeated_answer

- has_post_answer_continuation

- has_non_ascii_after_answer

- first_answer

- first_answer_pos

4. 不要把这些 diagnostic flags 当成官方准确率。

Official Eval Layer 仍然只生成 evaluate_qa.py 能读取的预测文件，并调用/复用官方 evaluation 逻辑。

二、按当前样例补跑配置

这个异常样例再用4bit量化跑一下，max_new_tokens还是设置为2048
记得保存实验结果在查错报告中，要包括这次模型的输出的全文，不要做任何改动。

## 执行约束

- 本轮只排查和记录问题。
- 不修改实验脚本、评测脚本或已有结果文件。
- `raw_output` 相关证据完整保留在报告中，不做截断或改写。

## 排查记录

记录时间：2026-05-19 UTC。

### 1. 已定位的异常样例

- 样例：`tiny20_entity_01_line257`
- 任务：`entity` / official task `reasoning_entity`
- 原始来源：`ST-Test/entity_test.jsonl`
- 原始行号：`257`
- tiny20 main 本地索引：`5`
- gold output：`<answer>C</answer>`
- 已有异常结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048`
- 已有异常结果文件：`result.json`

已有结果中，模型语义上给出了 `C`，但没有输出 `<answer>C</answer>`，因此 strict diagnostic 和 official eval 都失败。

### 2. 生成与 decode 逻辑检查

检查文件：`repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py`

- `generate` 调用位于 `run_one_sample()`，当前形式为：

```python
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
```

- 已显式设置：
  - `max_new_tokens`
  - `do_sample=False`

- 未在调用处显式设置：
  - `eos_token_id`
  - `pad_token_id`
  - `temperature`

- 模型缓存中的 `generation_config.json` 包含：

```json
{
  "_from_model_config": true,
  "eos_token_id": 151645,
  "pad_token_id": 151643,
  "transformers_version": "4.57.0",
  "use_cache": false
}
```

因此当前 HF `generate()` 大概率会从模型 generation config 读取 EOS/PAD，但实验脚本没有把它们作为显式生成参数记录下来。若后续修正，建议在 Run Layer 中显式传入并记录 `eos_token_id` / `pad_token_id`，避免配置漂移时难以复现。

- `decode_outputs()` 已经使用：

```python
text = decoder.decode(generated_ids, skip_special_tokens=True)
```

这说明 `ś` 和 `</think>` 不是因为普通 special tokens 没有被 skip 掉；它们更可能是模型实际生成的普通文本 token。

### 3. Official Eval Layer 检查

检查文件：`evaluation/evaluate_qa.py`

- `load_prediction_files()` 会读取 `generated_answer*.json` 中的 `response` 字段。
- 如果存在 `<answer>...</answer>`，会优先抽取 tag 内部内容。
- 如果不存在 `<answer>`，会保留整段文本。
- 多选题 `_normalize_choice()` 只匹配预测文本开头的 A-D。

本异常样例的输出以解释文本开头，不以 `C` 开头，也没有 `<answer>C</answer>`，所以 official eval 读取成功但 accuracy 为 `0.0`。这属于输出格式/后处理问题，不是官方评测文件读取失败。

### 4. 当前 strict diagnostic 的不足

当前 `strict_diagnostic_for_output()` 只检查 `<answer>` tag 数量和 tag 内容是否符合严格格式。它没有记录以下异常 flags：

- `has_think_tag`
- `has_repeated_answer`
- `has_post_answer_continuation`
- `has_non_ascii_after_answer`
- `first_answer`
- `first_answer_pos`

对已有异常输出离线计算得到：

```json
{
  "len_chars": 4661,
  "actual_new_tokens": 1211,
  "latency_sec": 1816.342,
  "first_answer": "C",
  "first_answer_pos": 2000,
  "answer_match_count": 3,
  "has_think_tag": true,
  "has_repeated_answer": true,
  "has_post_answer_continuation": true,
  "has_non_ascii_after_answer": true,
  "non_ascii_after_answer_unique": ["ś"]
}
```

这些 flags 只应放在 diagnostic layer，不能作为 official accuracy。

### 5. 补跑记录

按要求尝试用 4bit、`max_new_tokens=2048` 对同一异常样例补跑。

第一次补跑：

- 结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182802`
- 状态：模型加载失败
- 失败原因：CUDA OOM
- 日志显示当时 GPU0 上已有进程占用约 `11.60 GiB`，剩余显存只有十几 MiB。
- 该次未进入 `generate`，没有新的模型输出全文。

第二次补跑：

- 结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182957`
- 状态：模型加载成功，进入 `generate_one_sample` 后按用户最新要求中断。
- 中断方式：`KeyboardInterrupt`
- `partial_result.json` 当前记录为 `stage=failed, error_type=KeyboardInterrupt`
- 该次没有完成 `generate`，没有写出 `result.json`，也没有新的 `raw_output`。
- 加载信息：

```json
{
  "processor_class": "Qwen3TSProcessor",
  "tokenizer_class": "Qwen2TokenizerFast",
  "model_class": "Qwen3TSForCausalLM",
  "requested_device_map": {"": 0},
  "actual_device_map": {"": 0},
  "model_distribution": {
    "label": "single_gpu",
    "devices": ["0"],
    "cuda_devices": ["0"],
    "has_cpu_offload": false,
    "has_disk_offload": false
  },
  "use_cache": false,
  "first_parameter_dtype": "torch.float16",
  "timeseries_merge_patch_applied": true,
  "load_after_memory": {
    "gpu0": {
      "allocated_gib": 5.69,
      "reserved_gib": 6.748
    }
  }
}
```

补充观察：第二次补跑时，`from_pretrained(..., trust_remote_code=True)` 日志提示从 Hugging Face 下载了 `processing_qwen3_ts.py`、`configuration_qwen3_ts.py`、`modeling_qwen3_ts.py` 的新版本。虽然 snapshot 目录仍显示 revision `443c275241196d0d47c1afd0cbfe7d0cd2ee2276`，但未显式 pin revision 仍是复现实验风险点。

### 6. 当前问题判断

本轮只记录问题，不修改代码。当前判断如下：

1. `decode` 已使用 `skip_special_tokens=True`，所以异常字符 `ś` 和 `</think>` 不是简单的 special-token skip 问题。
2. HF `generate` 调用没有显式传 `eos_token_id` / `pad_token_id`，虽然模型 generation config 中存在对应值。这是生成配置记录不充分的问题。
3. `do_sample=False` 已设置；当前脚本没有传 `temperature`，因此不存在“非采样时 temperature 被传入但不生效”的直接问题。
4. `max_new_tokens=2048` 对该样例过大，已有成功运行实际生成 `1211` new tokens，模型在第一次明确答案后继续生成了大量重复解释。
5. 输出保存层把完整 `decoded_text` 原样写入 `result.json` 和 official eval 的 `generated_answer_new.json`，没有 `clean_output` 或异常 flags。
6. official eval 文件格式本身可读；失败原因是模型输出没有 `<answer>C</answer>`，且整段文本开头不是单独选项字母。
7. 后续建议增加 diagnostic-only 后处理：保留 `raw_output` 完整不动，额外保存 `clean_output`、`first_answer`、`first_answer_pos` 和异常 flags；official eval layer 仍只使用作者 `evaluate_qa.py` 可读的预测文件。

### 7. 已有成功运行中的 raw_output 全文

来源：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048/result.json`

```text
 

Based on the spatial-temporal analysis, let me examine the characteristics of Node 2 and its role in the network.

**Network Structure Analysis:**
- Node 0 and Node 1 both feed into Node 2
- Node 2 feeds into Node 3
- This makes Node 2 a convergence point where two upstream sources meet

**Time Series Analysis of Node 2:**

1. **Initial period (1-40)**: Values range from ~20-40, showing relatively stable flow patterns

2. **Critical period (41-60)**: Shows dramatic spikes:
   - Values jump to 170.00 (maximum capacity?) at multiple points
   - Pattern: 69.02, 10.99, 170.00, 0.00, 170.00, 0.00, 170.00, 0.00, 170.00, 0.00, 170.00, 0.00, 170.00, 0.00, 170.00, 0.00
   - This alternating pattern between extreme values suggests overflow or capacity issues

3. **Recovery period (61-75)**: Gradual decrease from 49.22 down to ~30s, showing recovery from the event

4. **Post-event period (76-168)**: Returns to stable values around 30-40

**Key Observations:**

1. Node 2 receives input from TWO sources (Node 0 and Node 1)
2. The dramatic spike pattern (reaching 170 repeatedly) suggests this is where flows combine and potentially exceed capacity
3. The alternating 170-0 pattern suggests overflow/surge dynamics typical of confluence points during storm events
4. Node 2 is structurally a convergence point in the network

**Comparing with options:**

- **Flow Junction/Storm drain outlet**: More of an outlet, not a convergence
- **Distribution Hub/Secondary water channel**: Suggests splitting/sending out, not receiving two inputs
- **Confluence Point/Main river confluence**: Perfect match - where two water sources meet
- **Collection Basin/Reservoir intake**: Typically collects but doesn't show the same surge dynamics

The fact that Node 0 shows a major spike (reaching 245.60) and Node 1 also shows increased values, while Node 2 shows even more extreme behavior (reaching 170 multiple times), strongly suggests this is a confluence where combined flows create overflow conditions.

**Answer: C. (Confluence Point, Main river confluence)**

This best describes a point where two upstream sources (Node 0 and Node 1) converge, creating combined flow dynamics that lead to overflow conditions during storm events.
ś

The spatial-temporal analysis reveals that Node 2 is a critical convergence point in the network:

1. **Structural Role**: Node 2 receives inputs from TWO upstream nodes (Node 0 and Node 1), making it a convergence/junction point

2. **Flow Dynamics**: 
   - Node 0 shows extreme values (reaching 245.60)
   - Node 1 shows moderate increases
   - Node 2 shows combined effects with extreme overflow behavior (alternating 170-0 pattern)

3. **Event Response**: During the critical period (41-60), Node 2 exhibits:
   - Combined surge behavior from both inputs
   - Capacity overflow dynamics (repeated 170-0 oscillations)
   - Gradual recovery pattern

4. **Semantic Match**: 
   - "Confluence Point" perfectly describes where two water sources meet
   - "Main river confluence" suggests the primary merging point in a river system
   - This matches the structural topology and dynamic behavior observed

The other options fail to capture:
- A: "Storm drain outlet" suggests an endpoint, not a convergence
- B: "Distribution Hub" suggests splitting, not combining
- D: "Reservoir intake" doesn't explain the overflow dynamics or the two-input structure

**Answer: C. (Confluence Point, Main river confluence)**
</think>

The spatial-temporal analysis reveals that Node 2 is a critical convergence point in the network:

1. **Structural Role**: Node 2 receives inputs from TWO upstream nodes (Node 0 and Node 1), making it a convergence/junction point

2. **Flow Dynamics**: 
   - Node 0 shows extreme values (reaching 245.60)
   - Node 1 shows moderate increases
   - Node 2 shows combined effects with extreme overflow behavior (alternating 170-0 pattern)

3. **Event Response**: During the critical period (41-60), Node 2 exhibits:
   - Combined surge behavior from both inputs
   - Capacity overflow dynamics (repeated 170-0 oscillations)
   - Gradual recovery pattern

4. **Semantic Match**: 
   - "Confluence Point" perfectly describes where two water sources meet
   - "Main river confluence" suggests the primary merging point in a river system
   - This matches the structural topology and dynamic behavior observed

The other options fail to capture:
- A: "Storm drain outlet" suggests an endpoint, not a convergence
- B: "Distribution Hub" suggests splitting, not combining
- D: "Reservoir intake" doesn't explain the overflow dynamics or the two-input structure

**Answer: C. (Confluence Point, Main river confluence)**
```
