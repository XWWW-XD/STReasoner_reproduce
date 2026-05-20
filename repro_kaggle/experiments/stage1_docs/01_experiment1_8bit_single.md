# 实验一：8bit单卡

## 本轮结论边界

- 本轮 8bit 在 model loading 前置依赖阶段失败：`repro_kaggle/scripts/03_load_streasoner_smoke.py` 缺失。
- 静态核查发现对应文件当前位于 `repro_kaggle/00_smoke_test_scripts/03_load_streasoner_smoke.py`。
- 因此本文件记录的是脚本路径/工作区文件移动导致的加载失败，不代表 8bit 单卡显存不足或 8bit 模型能力结论。
- 按 prompt2 的停止规则，本轮在此配置停止，没有继续运行 fp16 配置。

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：2048
- precision：`8bit`
- requested device_map：`None`
- actual device_map：`None`
- model distribution：`None`

## A. Run Layer

- load 成功：False
- load 错误：FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/STReasoner_reproduce/repro_kaggle/scripts/03_load_streasoner_smoke.py'
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

- run records：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/main_predictions_new.jsonl`
- official eval：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/official_eval`
- summary：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/summary_new.json`
- log：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/run_new.log`

## 瓶颈

- failure_count_by_stage：`{"model_loading": 2}`
- bottleneck_counts：`{}`
- first_error：model_loading 阶段失败：FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/STReasoner_reproduce/repro_kaggle/scripts/03_load_streasoner_smoke.py'
- post_config_health_check：`{"success": false, "errors": ["load failed: FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/STReasoner_reproduce/repro_kaggle/scripts/03_load_streasoner_smoke.py'", "missing load_after_memory", "missing model_distribution", "main/tiny20_forecasting_01_line87: generate failed", "main/tiny20_forecasting_01_line87: decode failed", "main/tiny20_forecasting_01_line87: decoded_text missing", "main/tiny20_forecasting_01_line87: input_tokens missing", "main/tiny20_forecasting_01_line87: actual_new_tokens missing", "main/tiny20_forecasting_01_line87: gpu_peak_memory_during_generate missing", "main/tiny20_entity_02_line529: generate failed", "main/tiny20_entity_02_line529: decode failed", "main/tiny20_entity_02_line529: decoded_text missing", "main/tiny20_entity_02_line529: input_tokens missing", "main/tiny20_entity_02_line529: actual_new_tokens missing", "main/tiny20_entity_02_line529: gpu_peak_memory_during_generate missing"]}`

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
