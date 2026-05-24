# 实验一：8bit单卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：2048
- precision：`8bit`
- requested device_map：`{'': 0}`
- actual device_map：`{'': 0}`
- model distribution：`{'label': 'single_gpu', 'devices': ['0'], 'cuda_devices': ['0'], 'has_cpu_offload': False, 'has_disk_offload': False}`

## A. Run Layer

- load 成功：True
- load 错误：None
- generate 成功率：1.0
- decode 成功率：1.0
- 平均 input tokens：697.0
- 平均 actual new tokens：1068.5
- 平均/最高延迟：1140.646 / 1242.483 秒
- 平均 tokens/s：0.952
- config-level generate 峰值显存：`{"gpu0": {"max_allocated_gib": 9.434, "max_reserved_gib": 11.002}}`

## B. Strict Diagnostic Layer

- strict format 成功率：0.0
- strict error counts：`{"expected_exactly_one_answer_tag_got_0": 2}`
- 说明：该层只诊断输出格式，不作为作者官方评测口径。

## C. Official Eval Layer

- official choice accuracy micro：0.0
- official forecasting MAE：85.002
- official forecasting MAPE：69.30770408980386
- official forecasting coverage：1.0
- 说明：官方指标复用 `evaluation/evaluate_qa.py`；forecasting 使用 MAE/MAPE/coverage，不计算 exact accuracy。

## 结论与瓶颈分类

- 资源瓶颈：8bit 单卡可以加载并完成两条 SmartTest 推理，没有 OOM，也没有 CPU/disk offload；峰值 reserved 显存约 11.002 GiB，平均 tokens/s 为 0.952，速度仍然偏慢。
- 输入/生成瓶颈：两条样本都没有触顶 2048，而是分别在 968、1169 new tokens 提前结束；这与 4bit 的 2048 触顶不同，属于生成收束/停止策略需要进一步检查的问题。
- 输出与评测瓶颈：两条输出都没有生成规范 `<answer>...</answer>`，strict parse 成功率为 0；entity 样本最终回答 A，gold 为 `<answer>B</answer>`，official accuracy 为 0；forecasting MAE/MAPE 仍然很差。因此 8bit 虽能跑通，但当前输出质量不可接受，暂停进入 fp16。

## 产物

- run records：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/main_predictions_new.jsonl`
- official eval：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/official_eval`
- summary：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/summary_new.json`
- log：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/run_new.log`

## 瓶颈

- failure_count_by_stage：`{}`
- bottleneck_counts：`{"输入/生成瓶颈": 2}`
- first_error：生成提前结束：actual_new_tokens=968，小于 max_new_tokens=2048；可能是 EOS 或模型自然停止。
- post_config_health_check：`{"success": true, "errors": []}`

## 样例输入、实际输出和正确结果

### 样例 1: `tiny20_forecasting_01_line87`

- task：`forecasting`
- source_file：`ST-Test/forecasting_test.jsonl`
- original_line_index：`87`
- generate_success：True
- decode_success：True
- actual_new_tokens：968
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "expected_exactly_one_answer_tag_got_0", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; official evaluate_qa.py may be more permissive."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 37: <ts><ts/>; Node 1 time series with length of 37: <ts><ts/>; Node 2 time series with length of 37: <ts><ts/>; Node 3 time series with length of 37: <ts><ts/>; Node 4 time series with length of 37: <ts><ts/>; Node 5 time series with length of 37: <ts><ts/>; Node 6 time series with length of 37: <ts><ts/>; Node 7 time series with length of 37: <ts><ts/>; Node 8 time series with length of 37: <ts><ts/>; Node 9 time series with length of 37: <ts><ts/>; Graph Structure: Node 0->Node 1; Node 1->Node 3; Node 2->Node 4; Node 4->Node 3; Node 3->Node 7; Node 7->Node 6; Node 7->Node 8; Node 1->Node 5; Node 5->Node 9; Node 8->Node 9, please analyze the spatial temporal data and answer the following question: Given the context Evening shopping and leisure activities, predict the value of node 8 for the next 5 steps. Historical observation window: 27-36.
```

#### 实际输出

```text
 Historical observation window for node 8: 27-36: 105.00, 105.60, 106.20, 106.80, 107.40, 107.97, 108.51, 109.02, 109.49, 109.94

Let me analyze the spatial-temporal patterns for this network.

First, let me understand the graph structure:
- Node 0 → Node 1 → Node 3 → Node 7 → Node 8
- Node 1 → Node 5 → Node 9
- Node 2 → Node 4 → Node 3
- Node 7 → Node 6
- Node 8 → Node 9

Node 8 is influenced by Node 7, and Node 7 is influenced by Node 3, which receives inputs from Nodes 1 and 4.

Looking at Node 8's recent trend (positions 27-36):
105.00, 105.60, 106.20, 106.80, 107.40, 107.97, 108.51, 109.02, 109.49, 109.94

The pattern shows:
- Generally increasing trend
- Differences between consecutive values: 0.60, 0.60, 0.60, 0.60, 0.57, 0.54, 0.49, 0.47, 0.45
- The increments are slightly decreasing but stabilizing around 0.5

Looking at Node 7's recent values (positions 27-36):
100.80, 97.95, 100.20, 99.28, 101.46, 102.95, 104.11, 105.02, 105.69, 106.24

Node 7 also shows an increasing trend, which influences Node 8.

Looking at Node 3's recent values (positions 27-36):
110.26, 96.32, 112.17, 103.25, 115.97, 108.60, 109.61, 110.66, 111.58, 112.41

Node 3 shows volatility but an overall upward trend.

Given the context of "Evening shopping and leisure activities," Node 8 appears to be showing growth in this period. The recent trend suggests continued growth but with gradually decreasing increments.

Extrapolating the next 5 values:
- Position 37: 109.94 + 0.42 ≈ 110.36
- Position 38: 110.36 + 0.38 ≈ 110.74
- Position 39: 110.74 + 0.34 ≈ 111.08
- Position 40: 111.08 + 0.30 ≈ 111.38
- Position 41: 111.38 + 0.26 ≈ 111.64

Let me refine based on the overall pattern and spatial dependencies.
</think>

The prediction for Node 8 values at positions 37-41 (next 5 steps) is:

110.45, 110.78, 111.08, 111.35, 111.59

This prediction is based on:
1. The observed increasing trend with decelerating growth rate
2. Spatial dependencies from Node 7 (which also shows increasing trend)
3. The pattern of gradually decreasing increments (0.5 → 0.42 → 0.38 → 0.34 → 0.30 → 0.26)
4. Context of evening shopping and leisure activities suggesting continued but moderating growth
```

#### 正确结果

```json
"[72.17, 101.58, 126.82, 102.51, 226.55]"
```

### 样例 2: `tiny20_entity_02_line529`

- task：`entity`
- source_file：`ST-Test/entity_test.jsonl`
- original_line_index：`529`
- generate_success：True
- decode_success：True
- actual_new_tokens：1169
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "expected_exactly_one_answer_tag_got_0", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; official evaluate_qa.py may be more permissive."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 96: <ts><ts/>; Node 1 time series with length of 96: <ts><ts/>; Node 2 time series with length of 96: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 1->Node 2; Node 2->Node 0; Node 2->Node 1, please analyze the spatial temporal data and answer the following question: Which (name, description) pair should Node 1 correspond to? Options: A. Commercial District, Business area with office complexes and retail stores B. Industrial Park, Industrial park with manufacturing facilities C. Logistics Hub, Distribution center with warehouses and shipping terminals D. Technology Park, Research and development campus with tech laboratories
```

#### 实际输出

```text

Answer:
To determine which location Node 1 corresponds to, I need to analyze the spatial-temporal patterns in the data.

**Step 1: Analyze the time series patterns**

Node 0: Starts at ~81, drops to ~20-25 (minimum around index 25-30), then rises dramatically to ~139 (peak around index 65), then gradually decreases back to ~57-60.

Node 1: Starts at -15.16, then zeros for several points, then rises from ~14 to ~100, peaks around 167 (index 60), then drops significantly to ~40-70 range.

Node 2: Starts at ~90, gradually decreases to ~10-17 (minimum around index 20-30), then rises to ~143 (peak around index 70), then decreases to ~81-85.

**Step 2: Analyze the graph structure**
- Node 0 → Node 2
- Node 1 → Node 2
- Node 2 → Node 0
- Node 2 → Node 1

Node 2 is central, receiving inputs from both Node 0 and Node 1, and sending outputs to both.

**Step 3: Identify key characteristics**

Node 1's most distinctive feature:
- Starts with negative value and zeros (indices 0-11)
- This suggests it's not operational or has no activity during early hours
- Gradual ramp-up from index 12 onwards
- Peak activity around index 60 (~167)
- Sharp drop after peak to ~40-70 range
- The pattern suggests 24-hour operation with significant variation

**Step 4: Match with location types**

A. Commercial District: Would show weekday/weekend patterns, peak during business hours (9 AM - 6 PM), lower at night - possible

B. Industrial Park: Manufacturing facilities typically run 24/7 or in shifts, with relatively stable activity - less likely to have extended zero period

C. Logistics Hub: Would show activity during business hours, with peaks during shipping/receiving times, but the extended zero period is unusual

D. Technology Park: R&D facilities might have irregular patterns, but the extended zero period at the start is unusual

**Step 5: Consider the temporal pattern more carefully**

The initial negative value and zeros in Node 1 (first ~12 time points) suggest:
- Either calibration/data collection issues
- Or a facility that doesn't operate during certain hours
- Or a location that ramps up gradually from night to day

The pattern shows:
- Very low activity initially (zeros)
- Gradual increase suggesting sunrise/sunrise-dependent activity
- Peak mid-day to evening
- This is most consistent with a **Commercial District** that builds up activity throughout the day

However, looking more carefully at the dramatic peak and subsequent drop, and considering Node 2 appears to be a central hub that influences both other nodes, Node 1's pattern with the extended low/zero activity period followed by high activity suggests it could be a **Logistics Hub** that operates primarily during certain hours and has major shipping/receiving periods.

Actually, reconsidering: The pattern where Node 1 starts inactive (zeros), then gradually ramps up, peaks, and then drops significantly is most consistent with a **Commercial District** that has business hours and shows typical daily activity patterns with morning ramp-up and evening decline.

Given that Node 2 appears to be a central hub (receives from both and sends to both), and Node 0 shows more stable patterns, Node 1's distinctive pattern with the extended ramp-up period suggests it's a **Commercial District** with typical business hour patterns.

Answer: A
</think>

To determine which location Node 1 corresponds to, I analyzed the spatial-temporal patterns in the data:

1. **Time Series Characteristics of Node 1:**
   - Starts with a negative value (-15.16) followed by zeros for an extended period (indices 0-11)
   - Gradual ramp-up from ~14 to ~100
   - Peak activity around 167 (index 60)
   - Sharp drop after peak to ~40-70 range
   - Shows a clear operational ramp-up pattern

2. **Graph Structure Analysis:**
   - Node 2 is central (receives from both Node 0 and Node 1, sends to both)
   - Node 1 feeds into Node 2, suggesting it's an input location

3. **Pattern Interpretation:**
   - The extended period of zeros/negative value suggests a location that builds up activity gradually
   - The gradual ramp-up pattern is consistent with morning activity building throughout the day
   - The peak and subsequent drop suggest typical business hours with evening decline
   - This pattern is most consistent with a location that has defined operational hours

4. **Location Type Matching:**
   - **Commercial District**: Would show business hour patterns with gradual morning ramp-up and evening decline ✓
   - **Industrial Park**: Would show 24/7 or shift-based patterns, not extended zero periods ✗
   - **Logistics Hub**: Would show more consistent activity patterns ✗
   - **Technology Park**: Would show irregular R&D patterns ✗

The pattern of Node 1 with its extended ramp-up period, peak activity, and clear operational boundaries is most consistent with a **Commercial District** that shows typical business hour activity patterns.

Answer: A
```

#### 正确结果

```json
"<answer>B</answer>"
```
