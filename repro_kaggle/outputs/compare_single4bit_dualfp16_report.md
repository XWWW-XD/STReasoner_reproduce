# 单卡 4bit 与双卡 FP16 对照实验

## 1. 目标

本实验比较当前 Kaggle T4 x2 环境下，STReasoner-8B 使用 single_gpu 4bit 与 dual_gpu FP16 推理的可行性和稳定性。

结论只适用于当前环境、当前模型和当前辅助脚本配置，不应泛化到所有双卡环境。

## 2. 背景

- 8B 模型的 FP16 权重约为 16GB 级别，而单张 Kaggle T4 约 14.56 GiB，因此单卡 FP16 不合适。
- 4bit 量化可以显著降低显存压力；已知 single_gpu smoke test 可以成功加载。
- dual_gpu FP16 理论上可以分摊权重，但可能引入 device mismatch、KV cache 放置、processor/timeseries tensor 设备不一致等问题。
- PyTorch 的 allocated 和 reserved 显存不能相加：allocated 是真实张量占用，reserved 是 PyTorch 缓存池，已经包含 allocated。

## 3. 实验设置

- model_name: `Time-HD-Anonymous/STReasoner-8B`
- dataset: `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`
- samples: 最多 20 条，每类最多 5 条
- attention backend: `sdpa`
- GPU: Kaggle Tesla T4
- A: `single_gpu + 4bit`
- B: `dual_auto + FP16`
- C: `dual_balanced + FP16`

## 4. 选中样本

- index `0`: `correlation`
- index `1`: `correlation`
- index `2`: `correlation`
- index `3`: `correlation`
- index `4`: `correlation`
- index `1592`: `entity`
- index `1593`: `entity`
- index `1594`: `entity`
- index `1595`: `entity`
- index `1596`: `entity`
- index `2786`: `etiological`
- index `2787`: `etiological`
- index `2788`: `etiological`
- index `2789`: `etiological`
- index `2790`: `etiological`
- index `2993`: `forecasting`
- index `2994`: `forecasting`
- index `2995`: `forecasting`
- index `2996`: `forecasting`
- index `2997`: `forecasting`

## 5. 结果表

| 策略 | 精度 | 可见 GPU | 实际 device map | 是否真实双卡 | 模型加载 | 生成成功数 | 生成失败数 | 解析失败率 | 准确率 | 平均延迟 | GPU0 峰值 reserved | GPU1 峰值 reserved | 主要错误 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_gpu | 4bit | 1 | {"": 0} | False | True | 20 | 0 | 0.650 | 0.714 | 50.270 | 6.744 | n/a | no_answer_tag_or_standalone_choice |
| dual_auto | fp16 | 2 | {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0, "model.layers.4": 0, "model... | True | True | 20 | 0 | 0.900 | 0.500 | 37.868 | 7.266 | 8.824 | no_answer_tag_or_standalone_choice |
| dual_balanced | fp16 | 2 | {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0, "model.layers.4": 0, "model... | True | True | 20 | 0 | 0.900 | 0.500 | 38.102 | 7.266 | 8.824 | no_answer_tag_or_standalone_choice |

## 6. 错误分析

### A. single_gpu + 4bit

- conclusion_hint: `PARTIAL`
- 模型加载是否通过: `True`
- 是否真实双卡: `False`
- failure_count_by_stage: `{"parse": 13}`
- failure_count_by_error_type: `{"ParseError": 13}`
- 首个错误: no_answer_tag_or_standalone_choice

### B. dual_auto + fp16

- conclusion_hint: `PARTIAL`
- 模型加载是否通过: `True`
- 是否真实双卡: `True`
- failure_count_by_stage: `{"parse": 18}`
- failure_count_by_error_type: `{"ParseError": 18}`
- 首个错误: no_answer_tag_or_standalone_choice

### C. dual_balanced + fp16

- conclusion_hint: `PARTIAL`
- 模型加载是否通过: `True`
- 是否真实双卡: `True`
- failure_count_by_stage: `{"parse": 18}`
- failure_count_by_error_type: `{"ParseError": 18}`
- 首个错误: no_answer_tag_or_standalone_choice

## 7. 证据片段

### A. single_gpu + 4bit

```text
MODEL_LOAD_PASS
actual_device_map: {'': 0}
gpu_memory_after_model_load: {'gpu0': {'allocated_gib': 5.703, 'reserved_gib': 6.74}}
GENERATE_PASS index=0 latency_sec=34.608
PARSE_FAIL index=0: no_answer_tag_or_standalone_choice
GENERATE_PASS index=1 latency_sec=58.234
GENERATE_PASS index=2 latency_sec=61.919
PARSE_FAIL index=2: no_answer_tag_or_standalone_choice
GENERATE_PASS index=3 latency_sec=64.873
PARSE_FAIL index=3: no_answer_tag_or_standalone_choice
GENERATE_PASS index=4 latency_sec=69.573
GENERATE_PASS index=1592 latency_sec=67.324
...
GENERATE_PASS index=2995 latency_sec=68.192
PARSE_FAIL index=2995: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2996 latency_sec=30.136
PARSE_FAIL index=2996: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2997 latency_sec=29.842
PARSE_FAIL index=2997: no_answer_tag_or_standalone_choice
  "actual_device_map": {
  "max_reserved_gib_by_gpu": {
```

### B. dual_auto + fp16

```text
MODEL_LOAD_PASS
actual_device_map: {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1, 'ts_encoder': 1}
gpu_memory_after_model_load: {'gpu0': {'allocated_gib': 6.909, 'reserved_gib': 6.941}, 'gpu1': {'allocated_gib': 8.474, 'reserved_gib': 8.494}}
GENERATE_PASS index=0 latency_sec=27.583
PARSE_FAIL index=0: no_answer_tag_or_standalone_choice
GENERATE_PASS index=1 latency_sec=48.365
GENERATE_PASS index=2 latency_sec=50.782
PARSE_FAIL index=2: no_answer_tag_or_standalone_choice
GENERATE_PASS index=3 latency_sec=52.869
PARSE_FAIL index=3: no_answer_tag_or_standalone_choice
GENERATE_PASS index=4 latency_sec=56.803
PARSE_FAIL index=4: no_answer_tag_or_standalone_choice
...
GENERATE_PASS index=2995 latency_sec=53.618
PARSE_FAIL index=2995: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2996 latency_sec=18.743
PARSE_FAIL index=2996: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2997 latency_sec=18.428
PARSE_FAIL index=2997: no_answer_tag_or_standalone_choice
  "actual_device_map": {
  "max_reserved_gib_by_gpu": {
```

### C. dual_balanced + fp16

```text
MODEL_LOAD_PASS
actual_device_map: {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1, 'ts_encoder': 1}
gpu_memory_after_model_load: {'gpu0': {'allocated_gib': 6.909, 'reserved_gib': 6.941}, 'gpu1': {'allocated_gib': 8.474, 'reserved_gib': 8.494}}
GENERATE_PASS index=0 latency_sec=28.236
PARSE_FAIL index=0: no_answer_tag_or_standalone_choice
GENERATE_PASS index=1 latency_sec=52.752
GENERATE_PASS index=2 latency_sec=52.370
PARSE_FAIL index=2: no_answer_tag_or_standalone_choice
GENERATE_PASS index=3 latency_sec=52.833
PARSE_FAIL index=3: no_answer_tag_or_standalone_choice
GENERATE_PASS index=4 latency_sec=56.216
PARSE_FAIL index=4: no_answer_tag_or_standalone_choice
...
GENERATE_PASS index=2995 latency_sec=53.612
PARSE_FAIL index=2995: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2996 latency_sec=18.713
PARSE_FAIL index=2996: no_answer_tag_or_standalone_choice
GENERATE_PASS index=2997 latency_sec=18.451
PARSE_FAIL index=2997: no_answer_tag_or_standalone_choice
  "actual_device_map": {
  "max_reserved_gib_by_gpu": {
```

## 8. 结论

在当前 Kaggle T4 x2 环境下，single_gpu 4bit 和至少一种 dual_gpu FP16 策略都可行。下一步应重点比较延迟、显存和输出稳定性；由于 single_gpu 4bit 更简单，除非 FP16 带来明确收益，否则它仍可作为默认低资源方案。
