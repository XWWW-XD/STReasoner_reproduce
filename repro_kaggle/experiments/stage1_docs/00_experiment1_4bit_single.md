# 实验一：4bit单卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：2048
- precision：`4bit`
- requested device_map：`{'': 0}`
- actual device_map：`{'': 0}`
- model distribution：`{'label': 'single_gpu', 'devices': ['0'], 'cuda_devices': ['0'], 'has_cpu_offload': False, 'has_disk_offload': False}`

## A. Run Layer

- load 成功：True
- load 错误：None
- generate 成功率：1.0
- decode 成功率：1.0
- 平均 input tokens：697.0
- 平均 actual new tokens：2048.0
- 平均/最高延迟：3507.756 / 4060.895 秒
- 平均 tokens/s：0.599
- config-level generate 峰值显存：`{"gpu0": {"max_allocated_gib": 6.038, "max_reserved_gib": 6.744}}`

## B. Strict Diagnostic Layer

- strict format 成功率：0.0
- strict error counts：`{"expected_exactly_one_answer_tag_got_0": 2}`
- 说明：该层只诊断输出格式，不作为作者官方评测口径。

## C. Official Eval Layer

- official choice accuracy micro：0.0
- official forecasting MAE：113.526
- official forecasting MAPE：87.61772173544392
- official forecasting coverage：1.0
- 说明：官方指标复用 `evaluation/evaluate_qa.py`；forecasting 使用 MAE/MAPE/coverage，不计算 exact accuracy。

## 结论与瓶颈分类

- 资源瓶颈：4bit 单卡可以加载并完成两条 SmartTest 推理，没有 OOM，也没有 CPU/disk offload；但速度非常慢，平均延迟约 3507.756 秒，平均 tokens/s 只有 0.599。
- 输入/生成瓶颈：两条样本的 actual new tokens 都等于 2048，说明均触顶 `max_new_tokens`，没有自然 EOS 收束。
- 输出与评测瓶颈：两条输出都没有生成规范 `<answer>...</answer>`，strict parse 成功率为 0；entity 样本预测为 C，gold 为 B，official choice accuracy 为 0；forecasting MAE/MAPE 也很差。

## 产物

- run records：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/main_predictions_new.jsonl`
- official eval：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/official_eval`
- summary：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/summary_new.json`
- log：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/run_new.log`

## 瓶颈

- failure_count_by_stage：`{}`
- bottleneck_counts：`{}`
- first_error：None
- post_config_health_check：`{"success": true, "errors": []}`

## 样例输入、实际输出和正确结果

### 样例 1: `tiny20_forecasting_01_line87`

- task：`forecasting`
- source_file：`ST-Test/forecasting_test.jsonl`
- original_line_index：`87`
- generate_success：True
- decode_success：True
- actual_new_tokens：2048
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "expected_exactly_one_answer_tag_got_0", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; official evaluate_qa.py may be more permissive."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 37: <ts><ts/>; Node 1 time series with length of 37: <ts><ts/>; Node 2 time series with length of 37: <ts><ts/>; Node 3 time series with length of 37: <ts><ts/>; Node 4 time series with length of 37: <ts><ts/>; Node 5 time series with length of 37: <ts><ts/>; Node 6 time series with length of 37: <ts><ts/>; Node 7 time series with length of 37: <ts><ts/>; Node 8 time series with length of 37: <ts><ts/>; Node 9 time series with length of 37: <ts><ts/>; Graph Structure: Node 0->Node 1; Node 1->Node 3; Node 2->Node 4; Node 4->Node 3; Node 3->Node 7; Node 7->Node 6; Node 7->Node 8; Node 1->Node 5; Node 5->Node 9; Node 8->Node 9, please analyze the spatial temporal data and answer the following question: Given the context Evening shopping and leisure activities, predict the value of node 8 for the next 5 steps. Historical observation window: 27-36.
```

#### 实际输出

```text
 The observation window is 27-36, which means we need to look at the data from indices 27 to 36 (inclusive).

Let me analyze the data for Node 8 during the observation window (indices 27-36):
Node 8 values [27-36]: 90.70, 89.11, 89.56, 89.72, 89.41, 103.28, 119.00, 134.73, 147.86, 139.21

I notice:
1. There's a significant upward trend in the later part of this window (from index 32 onwards): 89.41 → 103.28 → 119.00 → 134.73 → 147.86 → 139.21
2. The peak is at 147.86 (index 35), then there's a slight decline to 139.21 (index 36)
3. This suggests a peak has occurred and there might be a slight reversal

Looking at the broader pattern of Node 8:
- Early values (0-10): around 90-117
- Middle values (11-21): fluctuating around 80-110
- Later values (22-36): showing an upward trend, ending at 139.21

Given the context is "Evening shopping and leisure activities," and considering:
1. The recent strong upward trend
2. The slight pullback after the peak
3. The general pattern suggests continued elevated activity

For prediction, I expect:
- A recovery from the slight dip at 139.21
- Continuing the upward trend or stabilization at elevated levels
- Values likely in the 120-145 range based on the trend

Based on the momentum and the context of evening shopping/leisure (which typically shows higher activity), I predict a continuation of the upward trend with some fluctuation around the elevated levels.

**Prediction:**
138, 140, 142, 144, 145
</reasoning>

Let me refine based on the pattern:
The trend shows strong momentum upward, but given typical shopping/leisure patterns would stabilize, I'll adjust for realistic fluctuations.

138.5, 140.2, 142.1, 143.8, 145.0
</reasoning>

Rounding to two decimal places as in the original data:
138.50, 140.20, 142.10, 143.80, 145.00
</reasoning>

Actually, looking at the original data format, it seems to often show two decimal places but sometimes one. Let me match the style:

Original Node 8 ending: ...92.18, 91.10, 76.46

Most values have 2 decimal places, some have 1.

Final prediction with realistic fluctuation:
138.50, 140.20, 142.10, 143.80, 145.00
</reasoning>

Actually, let me make the last value slightly lower to reflect typical mean reversion even after a strong uptrend.

138.50, 140.20, 142.10, 143.80, 144.80
</reasoning>
</answer>

I'll provide the final answer with the reasoning completed.

</answer>
</revised_answer>

Let me analyze the spatial-temporal data for Node 8 with the given context.

**Observation Window Analysis (indices 27-36):**
90.70, 89.11, 89.56, 89.72, 89.41, 103.28, 119.00, 134.73, 147.86, 139.21

**Key Observations:**
1. Sharp increase starting at index 32 (89.41 → 103.28)
2. Strong upward momentum: 103.28 → 119.00 → 134.73 → 147.86
3. Peak at 147.86 (index 35)
4. Recent pullback to 139.21 (index 36)

**Context Consideration:**
"Evening shopping and leisure activities" typically shows:
- Increased activity compared to mid-day
- Tendency to peak and stabilize rather than continue sharp upward trend
- Mean reversion after peaks

**Graph Structure Context:**
Node 7 → Node 8, so Node 8 is influenced by Node 7's pattern. Checking Node 7's recent behavior:
- Shows similar peak pattern around the same time period
- Recent value: 73.12 (after peak around 381.65)

**Prediction Strategy:**
1. The strong upward trend suggests continued elevated activity
2. Mean reversion expected after peak
3. Values should stabilize around 140-150 range
4. Gradual decline from peak with fluctuations

**Prediction:**
Based on the momentum, mean reversion, and context, I predict:
139.5, 141.2, 142.8, 144.1, 145.0

</answer>
</revised_answer> 

Actually, let me reconsider the pattern. The data shows:
- Sharp spike from 89.41 to 147.86
- Recent pullback to 139.21
- The context suggests "evening" which typically shows elevated but stabilizing values

Given the strong momentum and "shopping/leisure" context, I'll predict continued upward trend with gradual stabilization:

140.5, 142.0, 143.5, 144.8, 145.5

</answer>
</revised_answer> 

Let me finalize with a prediction that shows recovery from the pullback and continuation of the upward trend typical of evening shopping/leisure activities:

140.2, 142.1, 143.8, 145.2, 146.1

</answer>
</revised_answer> 

Actually, let me provide the answer in the requested format with proper formatting:

**Answer:** \boxed{140.2, 142.1, 143.8, 145.2, 146.1}
</answer>
</revised_answer> 

Let me provide the final answer in the requested format:

**Answer:** \boxed{140.2, 142.1, 143.8, 145.2, 146.1}
</answer>
</revised_answer> 

Actually, let me reconsider based on the actual data patterns. Looking at Node 8's complete history, after peaks it tends to:
1. Show some pullback
2. Then stabilize at elevated levels
3. Gradually trend upward

Given the recent peak at 147.86 and pullback to 139.21, I expect:
- Recovery toward 142-145
- Continued gradual upward trend
- Values reflecting strong evening activity

**Final Prediction:** \boxed{142.3, 144.1, 145.6, 146.8, 147.9}
</answer>
</revised_answer> 

Actually, let me make this more conservative based on typical mean reversion:

**Answer:** \boxed{141.5, 143.2, 144.8, 146.0, 147.0}
</answer>
</revised_answer> 

Let me provide a prediction that shows gradual recovery and continuation of the upward trend:

**Answer:** \boxed{141.8, 143.5, 145.0, 146.2, 147.1}
</answer>
</revised_answer> 

Actually, let me make this more realistic based on the data's typical behavior:

**Answer:** \boxed{142.1, 143.8, 145.2, 146.3, 147.2}
</answer>
</revised_answer> 

Let me provide the final answer with realistic fluctuations:

**Answer:** \boxed{142.1, 143.8, 145.2, 146.3, 147.2}
</answer>
</revised_answer> 

Actually, let me reconsider one more time. Looking at the pattern more carefully, after a strong peak, we typically see:
- Some stabilization
- Gradual increase
- Values settling in a new range

Given the context of "
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
- actual_new_tokens：2048
- strict_diagnostic：`{"strict_format_success": false, "strict_error": "expected_exactly_one_answer_tag_got_0", "parsed_value": null, "answer_tag_count": 0, "note": "Diagnostic only; official evaluate_qa.py may be more permissive."}`

#### 输入

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 96: <ts><ts/>; Node 1 time series with length of 96: <ts><ts/>; Node 2 time series with length of 96: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 1->Node 2; Node 2->Node 0; Node 2->Node 1, please analyze the spatial temporal data and answer the following question: Which (name, description) pair should Node 1 correspond to? Options: A. Commercial District, Business area with office complexes and retail stores B. Industrial Park, Industrial park with manufacturing facilities C. Logistics Hub, Distribution center with warehouses and shipping terminals D. Technology Park, Research and development campus with tech laboratories
```

#### 实际输出

```text

Answer: C. Logistics Hub

The spatial-temporal analysis reveals the following patterns:

1. **Node 1 characteristics:**
   - Starts with a negative value (-15.16)
   - Shows a steady, consistent growth from near zero (11.47 → 20.74 → 29.41...)
   - Reaches a peak around 140-141
   - Then shows dramatic oscillating spikes (167.42, 121.08, 121.42, 121.29, 121.47, 121.72, 121.77, 121.72)
   - These oscillations suggest periodic high-intensity activity
   - Eventually stabilizes at a moderate level (70-90 range)

2. **Graph structure analysis:**
   - Node 2 → Node 1 (Node 1 receives from Node 2)
   - Node 1 → Node 2 (Node 1 sends to Node 2)
   - This bidirectional relationship suggests Node 1 is a distribution or flow point

3. **Pattern interpretation:**
   - The initial ramp-up from near zero suggests preparation/accumulation phase
   - The dramatic oscillating spikes (rapid alternating high-low pattern) suggest:
     * Batch processing
     * Loading/unloading cycles
     * Periodic delivery/receiving patterns
   - The oscillations at ~121 level suggest repetitive loading/unloading cycles typical of logistics operations

4. **Comparison with options:**
   - **Commercial District**: Would show daily patterns with peaks during business hours, not dramatic oscillations
   - **Industrial Park**: Would show more consistent production patterns, not these specific oscillating delivery patterns
   - **Logistics Hub**: Matches perfectly - the ramp-up represents inventory buildup, oscillations represent loading/unloading cycles, and the eventual stabilization represents steady-state operations
   - **Technology Park**: Would show research activity patterns, not these distribution-like oscillations

The dramatic oscillating pattern in Node 1, starting from zero and showing rapid alternating peaks, is characteristic of a logistics hub where goods are periodically received, sorted, and dispatched.
</reasoning>

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, starting from near zero and showing rapid alternating peaks, is characteristic of a logistics hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near zero, ramping up gradually, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and its bidirectional connection with Node 2, strongly indicates it represents a Logistics Hub with distribution center characteristics.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched in batches. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels, is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels, is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels, is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels, is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The spatial-temporal pattern of Node 1 - starting from near-zero activity, followed by dramatic oscillating spikes that suggest periodic loading/unloading cycles, and eventually stabilizing at moderate levels - is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

Node 1's pattern of starting from near-zero activity, followed by dramatic oscillating spikes (characteristic of loading/unloading cycles), and eventually stabilizing at moderate levels, is most consistent with a Logistics Hub where goods are periodically received, stored, and shipped.
</answer> 

Answer: C
</answer> 

The dramatic oscillating pattern in Node 1, with rapid alternating high-low values, is characteristic of a Logistics Hub where goods are periodically received, sorted, and dispatched. This matches the behavior of a distribution center with warehouses and shipping terminals.
</answer> 

Answer: C
</answer> 

Node 1's pattern - starting from near-zero activity, showing dramatic oscillating spikes (characteristic
```

#### 正确结果

```json
"<answer>B</answer>"
```
