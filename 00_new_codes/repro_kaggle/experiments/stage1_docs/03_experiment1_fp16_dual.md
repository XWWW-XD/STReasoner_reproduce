# 实验一：fp16双卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：6144
- precision：`fp16`
- requested device_map：`balanced`
- actual device_map：`{'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1, 'ts_encoder': 1}`
- model distribution：`{'label': 'multi_gpu', 'devices': ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 'cuda_devices': ['0', '1'], 'has_cpu_offload': False, 'has_disk_offload': False}`

## A. Run Layer

- load 成功：True
- load 错误：None
- generate 成功率：0.0
- decode 成功率：0.0
- 平均 input tokens：697.0
- 平均 actual new tokens：None
- 平均/最高延迟：None / None 秒
- 平均 tokens/s：None
- config-level generate 峰值显存：`{"gpu0": {"max_allocated_gib": 6.973, "max_reserved_gib": 7.0}, "gpu1": {"max_allocated_gib": 8.493, "max_reserved_gib": 8.51}}`

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

- run records：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_dual/main_predictions_new.jsonl`
- official eval：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_dual/official_eval`
- summary：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_dual/summary_new.json`
- log：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/fp16_dual/run_new.log`

## 瓶颈

- failure_count_by_stage：`{"generate": 2}`
- bottleneck_counts：`{}`
- first_error：generate 阶段失败：RuntimeError: The expanded size of the tensor (980) must match the existing size (950) at non-singleton dimension 2.  Target sizes: [1, 32, 980, 980].  Tensor sizes: [1, 1, 950, 980]
- post_config_health_check：`{"success": false, "errors": ["main/tiny20_forecasting_01_line87: generate failed", "main/tiny20_forecasting_01_line87: decode failed", "main/tiny20_forecasting_01_line87: decoded_text missing", "main/tiny20_forecasting_01_line87: actual_new_tokens missing", "main/tiny20_entity_02_line529: generate failed", "main/tiny20_entity_02_line529: decode failed", "main/tiny20_entity_02_line529: decoded_text missing", "main/tiny20_entity_02_line529: actual_new_tokens missing"]}`

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
