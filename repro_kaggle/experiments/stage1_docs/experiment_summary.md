# 实验一：不同精度推理资源测试

## 样本与目录

- 主测试样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl`，共 20 条，四类任务各 5 条。
- 论文样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl`，共 4 条。
- 压力测试样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl`，共 1 条。
- 运行脚本：`repro_kaggle/experiments/scripts/stage1_script/run_experiment1_precision_resource.py`
- 机器可读结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource`

## 证据查找

### 作者代码中的精度设置

- 训练脚本使用 fp16：
  - `scripts/qwen3-8b/train_stage1.sh:24`：`--fp16`
  - `scripts/qwen3-8b/train_stage2.sh:24`：`--fp16`
- 推理脚本中的 dtype：
  - `inference/llm_utils.py:97-98`：普通 vLLM worker 使用 `LLM(..., dtype='half')`，对应 fp16/half。
  - `inference/llm_utils.py:142` 之后的 `worker_vllm_ts` 未显式设置 dtype；实际 dtype 需要通过本实验记录加载后模型参数 dtype、量化配置和日志确认。

### 作者代码中的 max_new_tokens / max_tokens 设置

- 作者 vLLM 推理使用 `max_tokens=512`：
  - `inference/inference_tsmllm_vllm.py:64-68`：`SamplingParams(max_tokens=512, temperature=0.2)`
- 本实验使用 Hugging Face `generate`，对应参数记录为 `max_new_tokens=512`。
- Hugging Face 的 `max_new_tokens` 表示最多生成的新 token 数，不包含 prompt tokens。


## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | 主测试 20 + 论文样例 4 + 压力测试 1 |
|   batch size   |                1                |
| max_new_tokens |               512               |

|           | dtype                        | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |
| --------- | ---------------------------- | -------: | -------: | -------: | ------- |
| 配置证据 | 加载方式 | 4bit | 8bit | fp16 | fp16 |
|  | device_map | {"": 0} | {"": 0} | {"": 0} | balanced |
|  | 实际模型分布 | {"": 0} | {"": 0} |  | {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0, "model.layers.4": 0, "model.layers.5": 0, "model.layers.6": 0, "model.layers.7": 0, "model.layers.8": 0, "model.layers.9": 0, "model.layers.10": 0, "model.layers.11": 0, "model.layers.12": 0, "model.layers.13": 0, "model.layers.14": 0, "model.layers.15": 0, "model.layers.16": 1, "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1, "model.layers.20": 1, "model.layers.21": 1, "model.layers.22": 1, "model.layers.23": 1, "model.layers.24": 1, "model.layers.25": 1, "model.layers.26": 1, "model.layers.27": 1, "model.layers.28": 1, "model.layers.29": 1, "model.layers.30": 1, "model.layers.31": 1, "model.layers.32": 1, "model.layers.33": 1, "model.layers.34": 1, "model.layers.35": 1, "model.norm": 1, "model.rotary_emb": 1, "lm_head": 1, "ts_encoder": 1} |
|  | is_cpu_offload | 见 actual_device_map | 见 actual_device_map | 见 actual_device_map | 见 actual_device_map |
|  | use_cache | False | False |  | False |
| 可运行证据 | input tokens（平均值） | 745.160 | 745.160 |  | 745.160 |
|  | actual new tokens（平均值） |  |  |  |  |
|  | load 成功率 | True | True | False | True |
|  | generate 成功率 | 0.000 | 0.000 | 0.000 | 0.000 |
| 资源与速度 | GPU 总显存 | {"gpu0": 14.563} | {"gpu0": 14.563} | {"gpu0": 14.563} | {"gpu0": 14.563, "gpu1": 14.563} |
|  | load 后显存 | {"gpu0": {"allocated_gib": 5.703, "reserved_gib": 6.74}} | {"gpu0": {"allocated_gib": 8.859, "reserved_gib": 9.064}} |  | {"gpu0": {"allocated_gib": 6.909, "reserved_gib": 6.941}, "gpu1": {"allocated_gib": 8.474, "reserved_gib": 8.494}} |
|  | generate 峰值显存 | {"gpu0": {"max_allocated_gib": 6.631, "max_reserved_gib": 6.744}} | {"gpu0": {"max_allocated_gib": 9.387, "max_reserved_gib": 10.293}} | {"gpu0": {"max_allocated_gib": 14.345, "max_reserved_gib": 14.389}} | {"gpu0": {"max_allocated_gib": 6.917, "max_reserved_gib": 6.941}, "gpu1": {"max_allocated_gib": 8.491, "max_reserved_gib": 8.496}} |
|  | 平均延迟与最高延迟 |  /  |  /  |  /  |  /  |
|  | tokens/s |  |  |  |  |
|  | decode 成功率 | 0.000 | 0.000 | 0.000 | 0.000 |
|  | parse 成功率 | 0.000 | 0.000 | 0.000 | 0.000 |
|  | 平均正确率（对比失败也算错误） | 0.000 | 0.000 | 0.000 | 0.000 |
| 失败阶段、失败原因 | 失败阶段、详细失败原因；若有输出，输出是否正确（T/F） | 输入与 time-series 特征合并时张量长度不匹配，generate 失败：RuntimeError: The size of tensor a (981) must match the size of tensor b (951) at non-singleton dimension 1 | 输入与 time-series 特征合并时张量长度不匹配，generate 失败：RuntimeError: The size of tensor a (981) must match the size of tensor b (951) at non-singleton dimension 1 | 显存不足导致失败：OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 14.56 GiB of which 57.81 MiB is free. Including non-PyTorch memory, this process has 14.50 GiB memory in use. Of the allocated memory 14.35 GiB is allocated by PyTorch, and 44.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | 设备不一致导致失败：RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0! |

## 正式统计口径

正式成功率统计三组样例合计 25 条：主测试 20 条 + 论文样例 4 条 + 压力测试 1 条。报告中仍保留三组样例的分组结果用于定位问题。

## 瓶颈类型总结

详见各配置报告与 `summary.json` 中的 `bottleneck_counts`。失败原因统一归入资源瓶颈、输入/生成瓶颈、输出与评测瓶颈三类；确定的原因尽量细写，不确定时只写粗略阶段和错误信息。
