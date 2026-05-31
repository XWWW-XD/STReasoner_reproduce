# 实验一：fp16单卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：2048
- precision：`fp16`
- requested device_map：`None`
- actual device_map：`None`
- model distribution：`None`

## A. Run Layer

- load 成功：False
- load 错误：OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 14.56 GiB of which 57.81 MiB is free. Including non-PyTorch memory, this process has 14.50 GiB memory in use. Of the allocated memory 14.35 GiB is allocated by PyTorch, and 44.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- generate 成功率：0.0
- decode 成功率：0.0
- 平均 input tokens：None
- 平均 actual new tokens：None
- 平均/最高延迟：None / None 秒
- 平均 tokens/s：None
- config-level generate 峰值显存：`{}`

## B. Strict Diagnostic Layer

- strict format 成功率：0.0
- strict error counts：`{"no_decoded_text": 2}`
- 说明：该层只诊断输出格式，不作为作者官方评测口径。

## C. Official Eval Layer

- official choice accuracy micro：None
- official forecasting MAE：None
- official forecasting MAPE：None
- official forecasting coverage：0.0
- 说明：官方指标复用 `evaluation/evaluate_qa.py`；forecasting 使用 MAE/MAPE/coverage，不计算 exact accuracy。

## 产物

- run records：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_single/main_predictions_new.jsonl`
- official eval：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_single/official_eval`
- summary：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_single/summary_new.json`
- log：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_single/run_new.log`

## 瓶颈

- failure_count_by_stage：`{"model_loading": 2}`
- bottleneck_counts：`{"资源瓶颈": 2}`
- first_error：显存不足导致失败：OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 14.56 GiB of which 57.81 MiB is free. Including non-PyTorch memory, this process has 14.50 GiB memory in use. Of the allocated memory 14.35 GiB is allocated by PyTorch, and 44.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- post_config_health_check：`{"success": false, "errors": ["load failed: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 14.56 GiB of which 57.81 MiB is free. Including non-PyTorch memory, this process has 14.50 GiB memory in use. Of the allocated memory 14.35 GiB is allocated by PyTorch, and 44.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)", "missing load_after_memory", "missing model_distribution", "main/tiny20_forecasting_01_line87: generate failed", "main/tiny20_forecasting_01_line87: decode failed", "main/tiny20_forecasting_01_line87: decoded_text missing", "main/tiny20_forecasting_01_line87: input_tokens missing", "main/tiny20_forecasting_01_line87: actual_new_tokens missing", "main/tiny20_forecasting_01_line87: gpu_peak_memory missing", "main/tiny20_entity_02_line529: generate failed", "main/tiny20_entity_02_line529: decode failed", "main/tiny20_entity_02_line529: decoded_text missing", "main/tiny20_entity_02_line529: input_tokens missing", "main/tiny20_entity_02_line529: actual_new_tokens missing", "main/tiny20_entity_02_line529: gpu_peak_memory missing"]}`

## 样例输入、实际输出和正确结果

### 样例 1: `tiny20_forecasting_01_line87`

- task：`forecasting`
- source_file：`ST-Test/forecasting_test.jsonl`
- original_line_index：`87`
- generate_success：False
- decode_success：False
- actual_new_tokens：None
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "no_decoded_text", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; not used for official metrics."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 37: <ts><ts/>; Node 1 time series with length of 37: <ts><ts/>; Node 2 time series with length of 37: <ts><ts/>; Node 3 time series with length of 37: <ts><ts/>; Node 4 time series with length of 37: <ts><ts/>; Node 5 time series with length of 37: <ts><ts/>; Node 6 time series with length of 37: <ts><ts/>; Node 7 time series with length of 37: <ts><ts/>; Node 8 time series with length of 37: <ts><ts/>; Node 9 time series with length of 37: <ts><ts/>; Graph Structure: Node 0->Node 1; Node 1->Node 3; Node 2->Node 4; Node 4->Node 3; Node 3->Node 7; Node 7->Node 6; Node 7->Node 8; Node 1->Node 5; Node 5->Node 9; Node 8->Node 9, please analyze the spatial temporal data and answer the following question: Given the context Evening shopping and leisure activities, predict the value of node 8 for the next 5 steps. Historical observation window: 27-36.
```

#### 实际输出

```text
None
```

#### 正确结果

```json
"[72.17, 101.58, 126.82, 102.51, 226.55]"
```

### 样例 2: `tiny20_entity_02_line529`

- task：`entity`
- source_file：`ST-Test/entity_test.jsonl`
- original_line_index：`529`
- generate_success：False
- decode_success：False
- actual_new_tokens：None
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "no_decoded_text", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; not used for official metrics."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 96: <ts><ts/>; Node 1 time series with length of 96: <ts><ts/>; Node 2 time series with length of 96: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 1->Node 2; Node 2->Node 0; Node 2->Node 1, please analyze the spatial temporal data and answer the following question: Which (name, description) pair should Node 1 correspond to? Options: A. Commercial District, Business area with office complexes and retail stores B. Industrial Park, Industrial park with manufacturing facilities C. Logistics Hub, Distribution center with warehouses and shipping terminals D. Technology Park, Research and development campus with tech laboratories
```

#### 实际输出

```text
None
```

#### 正确结果

```json
"<answer>B</answer>"
```
