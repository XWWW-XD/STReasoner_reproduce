# paper_cases × ST-Test 6144 同 index 对照报告

日期：2026-06-01

## 说明

ST-Test 四类任务已按 `max_tokens=6144` 全量跑完（见 `10-ST-Test数据集效果.md`）。
本报告不再单独跑 paper_cases，而是从全量附件
`artifacts/sttest_full_6144_outputs_with_gold.jsonl` 中，按 `PaperCases.jsonl` 记录的
`source_file` + `original_line_index` 抽取同一条样本的 response。

推理配置（全量 ST-Test 正式）：

- 入口：`inference/inference_tsmllm_vllm.py`
- 模型：`base_model/STReasoner-8B`
- `max_tokens=6144`，`temperature=0.2`
- 格式后缀：`inference/prompt.json`
- 评估口径：tag-first `evaluation/evaluate.py`

## 索引对照

| # | paper_case_id | category | ST-Test task | idx | gold | parsed | 正确 | response tokens | 触顶 6144 | `<answer>` 1 对 |
|---:|---|---|---|---:|---|---|---|---:|---:|---:|
| 0 | `appendix_h_table6_etiological` | etiological | `reasoning_etiological` | 118 | `<answer>D</answer>` | `D` | True | 756 | 否 | 是 |
| 1 | `appendix_h_table7_entity` | entity | `reasoning_entity` | 982 | `<answer>C</answer>` | `C` | True | 613 | 否 | 是 |
| 2 | `appendix_h_table8_correlation` | correlation | `reasoning_correlation` | 547 | `<answer>D</answer>` | `D` | True | 857 | 否 | 是 |
| 3 | `appendix_h_table9_forecasting` | forecasting | `reasoning_forecasting` | 9 | `[19.86, 19.97, 20.05]` | `[19.88, 19.9, 19.91]` | MAE≈0.077 | 761 | 否 | 是 |

## 汇总

- paper_cases 4 条均能在全量 ST-Test 6144 附件中找到同 index 记录。
- 选择题 tag-first 判对：3/3。
- 严格 1 对 `<answer>` 标签：4/4。
- 达到 `max_tokens=6144`：0/4。
- response token 范围：613–857（四题长度接近，均未进入 5000+ 长尾）。

## 指标补充

| 指标 | 值 |
|---|---:|
| 选择题 accuracy | 3/3 |
| forecasting parsed | `[19.88, 19.9, 19.91]` |
| forecasting gold | `[19.86, 19.97, 20.05]` |
| forecasting MAE（手工按 parsed vs gold） | ≈0.0767 |
| forecasting MAPE（同上） | ≈0.38% |

## Case 0：`appendix_h_table6_etiological`（idx=118）

- task：`reasoning_etiological`
- 来源：`ST-Test/etiological_test.jsonl` 第 118 行（0-based idx=118）
- 预测文件：`exp/sttest_full_etiological_6144/generated_answer.json` → key `"118"`
- gold：`<answer>D</answer>`
- parsed_prediction：`D`
- choice_correct：`True`
- input_tokens_recorded：293
- response_tokens_tokenizer：**756**
- response 字符数：2664
- reached_max_tokens_6144：False
- answer_tag：open=1 close=1

### input（题干，TS 为 `<ts><ts/>` 占位）

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 48: <ts><ts/>; Node 1 time series with length of 48: <ts><ts/>; Node 2 time series with length of 48: <ts><ts/>; Node 3 time series with length of 48: <ts><ts/>; Node 4 time series with length of 48: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 1->Node 2; Node 2->Node 3; Node 3->Node 4, please analyze the spatial temporal data and answer the following question: Which etiological scenario can be inferred from the spatio-temporal data? Options: A. Factory ventilation system with chemical processes spreading through connected chambers based on pressure differentials. B. Coastal emission system with marine and port activities dispersing through atmospheric layers toward inland regions. C. Agricultural runoff network with pesticide sources flowing through water channels, affected by seasonal discharge patterns. D. Urban pollution network with industrial and traffic sources dispersing through nodes, modulated by time-based transport patterns.
```

### raw_response（完整）

```text
<think>
Let me analyze the spatial-temporal data systematically:

1. **Graph Structure Analysis:**
   - Node 0 → Node 2
   - Node 1 → Node 2
   - Node 2 → Node 3
   - Node 3 → Node 4
   - This shows a convergent-divergent network where Node 2 is a central hub receiving inputs from Nodes 0 and 1, then flowing to Nodes 3 and 4.

2. **Temporal Pattern Analysis:**

   **Node 0:** Shows a clear rise from ~20 (t=0-12) to peak at ~56 (t=20), then gradual decline back to ~29 (t=47). Has a notable spike around t=20.

   **Node 1:** Starts at ~34-35, decreases to ~25 (t=12-19), then rises to ~46 (t=32), with elevated values in middle period (t=32-36) around 40-46.

   **Node 2:** Relatively stable at ~29-31 (t=0-19), then shows extreme spikes (60.27 at t=24, 41.01 at t=26, 36.95 at t=28), suggesting it receives combined inputs. Later stabilizes around 32-33.

   **Node 3:** Stable at ~24-25 (t=0-20), then shows increases with peaks around t=23-28 (reaching 44.25 at t=28), followed by gradual decline to ~25-26.

   **Node 4:** Most stable initially at ~20-22, gradual increase starting around t=22-23, reaching ~35 at t=35, then declining to ~24-25.

3. **Key Observations:**
   - Two source nodes (0 and 1) feed into a central node (2)
   - Sequential propagation through the network (0,1 → 2 → 3 → 4)
   - Time-lagged responses: peaks in upstream nodes precede peaks in downstream nodes
   - Node 2 shows combined effects of two sources
   - The pattern suggests accumulation and transport over time

4. **Scenario Matching:**

   **Option A (Factory ventilation):** Would expect more regular oscillatory patterns based on operational cycles, not the smooth rise-peak-decline pattern observed.

   **Option B (Coastal emission):** Marine systems typically show more cyclical patterns related to tides and would have different temporal characteristics.

   **Option C (Agricultural runoff):** Seasonal patterns would be much longer (annual cycles), not 48 time steps representing a single event with rapid rise and fall.

   **Option D (Urban pollution):** 
   - Two sources (industrial and traffic) = Nodes 0 and 1
   - Central collection/dispersion point = Node 2
   - Sequential downstream dispersion = Nodes 3 and 4
   - Time-based transport patterns = the lagged responses and peak propagation
   - The rise and peak pattern matches daily/periodic urban activity (industrial operations ramping up, traffic increases)
   - The subsequent decline matches reduced emissions

The convergent structure (two sources → hub → sequential nodes) and time-based transport with lagged peaks strongly suggest an urban pollution network.

</think>

<answer>D</answer>
```

## Case 1：`appendix_h_table7_entity`（idx=982）

- task：`reasoning_entity`
- 来源：`ST-Test/entity_test.jsonl` 第 982 行（0-based idx=982）
- 预测文件：`exp/sttest_full_entity_6144/generated_answer.json` → key `"982"`
- gold：`<answer>C</answer>`
- parsed_prediction：`C`
- choice_correct：`True`
- input_tokens_recorded：396
- response_tokens_tokenizer：**613**
- response 字符数：2169
- reached_max_tokens_6144：False
- answer_tag：open=1 close=1

### input（题干，TS 为 `<ts><ts/>` 占位）

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 48: <ts><ts/>; Node 1 time series with length of 48: <ts><ts/>; Node 2 time series with length of 48: <ts><ts/>; Node 3 time series with length of 48: <ts><ts/>; Node 4 time series with length of 48: <ts><ts/>; Node 5 time series with length of 48: <ts><ts/>; Node 6 time series with length of 48: <ts><ts/>; Node 7 time series with length of 48: <ts><ts/>; Node 8 time series with length of 48: <ts><ts/>; Node 9 time series with length of 48: <ts><ts/>; Graph Structure: Node 0->Node 1; Node 1->Node 3; Node 2->Node 3; Node 3->Node 4; Node 4->Node 5; Node 5->Node 6; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9, please analyze the spatial temporal data and answer the following question: Which (name, description) pair should Node 9 correspond to? Options: A. Endpoint Alpha, Last measurement checkpoint B. Exit Node, Terminal collection station C. Final Point, Final monitoring point D. Terminal Hub, Network termination sensor
```

### raw_response（完整）

```text
<think>
Let me analyze the spatial-temporal data systematically.

1. **Graph Structure Analysis:**
   - The graph forms a linear chain: Node 0 → Node 1 → Node 3 → Node 4 → Node 5 → Node 6 → Node 7 → Node 8 → Node 9
   - Node 2 also connects to Node 3
   - Node 9 is the last node in this chain with no outgoing connections

2. **Time Series Pattern Analysis:**

   Looking at the disturbance propagation:
   - Nodes 0-2: Relatively stable values around 17.5
   - Node 3: Shows disturbance starting at index 18 (drops to 11.25, then 0.00, then spikes to 48.03)
   - Nodes 4-9: Show progressive disturbance patterns with increasing delay

   Node 9's time series shows:
   - Starts stable around 17.4-17.5 for the first 33 time points
   - Disturbance appears around index 33-34 (drops to 1.23, 5.95)
   - Shows recovery pattern: 11.23, 16.22, 18.21, 19.21, 20.00, 20.56, 20.96, 21.21, 21.34, 21.41, 21.44, 17.50
   - Eventually stabilizes back to around 17.5

3. **Positional Analysis:**
   - Node 9 is at the END of the propagation chain
   - It receives information that has propagated through 8 preceding nodes
   - It has no outgoing connections to other nodes

4. **Evaluating Options:**
   - All options describe a terminal/endpoint position
   - The key is understanding the specific role
   - "Endpoint Alpha" and "Exit Node" suggest an exit/terminal point
   - "Final Point" and "Final monitoring point" emphasize the final monitoring aspect
   - "Terminal Hub" and "Network termination sensor" suggest sensor functionality at network end

5. **Contextual Clue:**
   - The graph shows a clear monitoring/flow network
   - Node 9 is the ultimate destination where propagated effects are finally measured
   - The term "monitoring point" fits well with the temporal propagation pattern observed
   - "Final" emphasizes it's the last point in the sequence

Given that Node 9 is the last node in a linear monitoring chain with no outgoing connections and receives the most delayed response, **Option C "Final Point, Final monitoring point"** best describes its role as the ultimate destination for monitoring the propagated phenomenon.

</think>

<answer>C</answer>
```

## Case 2：`appendix_h_table8_correlation`（idx=547）

- task：`reasoning_correlation`
- 来源：`ST-Test/correlation_test.jsonl` 第 547 行（0-based idx=547）
- 预测文件：`exp/sttest_full_correlation_6144/generated_answer.json` → key `"547"`
- gold：`<answer>D</answer>`
- parsed_prediction：`D`
- choice_correct：`True`
- input_tokens_recorded：729
- response_tokens_tokenizer：**857**
- response 字符数：1935
- reached_max_tokens_6144：False
- answer_tag：open=1 close=1

### input（题干，TS 为 `<ts><ts/>` 占位）

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 240: <ts><ts/>; Node 1 time series with length of 240: <ts><ts/>; Node 2 time series with length of 240: <ts><ts/>; Node 3 time series with length of 240: <ts><ts/>; Node 4 time series with length of 240: <ts><ts/>; Node 5 time series with length of 240: <ts><ts/>; Node 6 time series with length of 240: <ts><ts/>; Node 7 time series with length of 240: <ts><ts/>; Node 8 time series with length of 240: <ts><ts/>; Node 9 time series with length of 240: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 0->Node 6; Node 1->Node 2; Node 2->Node 3; Node 2->Node 5; Node 3->Node 4; Node 4->Node 8; Node 5->Node 7; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9, please analyze the spatial temporal data and answer the following question: Which statement best describes the relationship between Node 2 and Node 4 during time steps 177-180 (1 time step = 1 hour)? Options: A. There is no significant thermal signal transmission between Node 2 and Node 4 B. Thermal signal travels directly from Node 2 to Node 4 without intermediate nodes C. Thermal signal flows from Node 4 back to Node 2 through a feedback loop D. Thermal signal propagates from Node 2 to Node 4 via Node 3, following a sequential central-to-eastern pathway
```

### raw_response（完整）

```text
<think>
Let me analyze the spatial-temporal data for nodes 2, 3, and 4 during time steps 177-180.

First, let me extract the values:
- Node 2 at steps 177-180: 21.40, 21.41, 21.41, 21.42
- Node 3 at steps 177-180: 21.02, 21.02, 21.02, 21.02
- Node 4 at steps 177-180: 20.51, 20.51, 20.51, 20.51

Now let me look at the graph structure:
- Node 2 -> Node 3
- Node 3 -> Node 4

This shows a sequential pathway: Node 2 → Node 3 → Node 4

Let me examine the pattern more carefully by looking at a few time steps before and after to see if there's a propagation pattern:

Node 2 (steps 177-180): 21.40, 21.41, 21.41, 21.42
Node 3 (steps 177-180): 21.02, 21.02, 21.02, 21.02
Node 4 (steps 177-180): 20.51, 20.51, 20.51, 20.51

Looking at earlier time steps:
Node 2 (steps 173-176): 21.31, 21.31, 21.31, 21.31
Node 3 (steps 173-176): 20.96, 20.96, 20.96, 20.96
Node 4 (steps 173-176): 20.42, 20.42, 20.42, 20.42

Looking at later time steps:
Node 2 (steps 181-184): 21.43, 21.43, 21.43, 21.43
Node 3 (steps 181-184): 21.03, 21.03, 21.03, 21.03
Node 4 (steps 181-184): 20.52, 20.52, 20.52, 20.52

Now let me check if there's a thermal signal propagation pattern. Looking at the overall trends:
- Node 2 shows an increase from around 21.31 to 21.42
- Node 3 shows an increase from around 20.96 to 21.02
- Node 4 shows an increase from around 20.42 to 20.51

The values are increasing sequentially, and given the graph structure (Node 2 → Node 3 → Node 4), this suggests thermal signal propagation through the pathway.

Looking at the options:
A. No significant thermal signal - This doesn't seem right as there is a gradual increase
B. Direct transmission - The graph shows Node 2 doesn't connect directly to Node 4
C. Flow from Node 4 back to Node 2 - The graph shows no backward connection
D. Propagates from Node 2 to Node 4 via Node 3 - This matches the graph structure and the observed pattern

The answer is D.
</think>

<answer>D</answer>
```

## Case 3：`appendix_h_table9_forecasting`（idx=9）

- task：`reasoning_forecasting`
- 来源：`ST-Test/forecasting_test.jsonl` 第 9 行（0-based idx=9）
- 预测文件：`exp/sttest_full_forecasting_6144/generated_answer.json` → key `"9"`
- gold：`[19.86, 19.97, 20.05]`
- parsed_prediction：`[19.88, 19.9, 19.91]`
- choice_correct：`None`
- input_tokens_recorded：182
- response_tokens_tokenizer：**761**
- response 字符数：2051
- reached_max_tokens_6144：False
- answer_tag：open=1 close=1

### input（题干，TS 为 `<ts><ts/>` 占位）

```text
You are a spatial temporal analysis expert. Node 0 time series with length of 36: <ts><ts/>; Node 1 time series with length of 36: <ts><ts/>; Node 2 time series with length of 36: <ts><ts/>; Graph Structure: Node 0->Node 2; Node 1->Node 2; Node 2->Node 1, please analyze the spatial temporal data and answer the following question: Given the context Maximum solar heating period enhances thermal transfer, predict the value of node 2 for the next 3 steps. Historical observation window: 30-35.
```

### raw_response（完整）

```text
<think>
Let me analyze the spatial-temporal data step by step.

## Step 1: Understanding the Graph Structure
- Node 0 → Node 2 (Node 0 influences Node 2)
- Node 1 → Node 2 (Node 1 influences Node 2)
- Node 2 → Node 1 (Node 2 influences Node 1)

## Step 2: Analyzing the Historical Window (positions 30-35)
- Node 0 [30-35]: 19.27, 19.41, 19.51, 19.61, 19.69, 18.97
- Node 1 [30-35]: 19.39, 19.52, 19.61, 19.69, 19.78, 19.71
- Node 2 [30-35]: 19.74, 19.81, 19.85, 19.88, 19.91, 19.87

## Step 3: Identifying Trends
Looking at the entire series:
- **Node 0**: Shows an increasing trend until position 23 (around 20.72), then starts decreasing, with a drop to 18.97 at position 35
- **Node 1**: Shows a strong increasing trend from 17.92 to around 19.78 (position 34), then a slight drop to 19.71
- **Node 2**: Shows a relatively stable increasing trend from around 19.64 to 19.91, then a slight drop to 19.87

## Step 4: Analyzing Relationships
Node 2 is influenced by both Node 0 and Node 1. 
- Node 0 is declining (from 19.69 to 18.97)
- Node 1 is high and stable (around 19.69-19.78 range)
- Node 2 shows slight decline at the end (19.91 to 19.87)

## Step 5: Context - Solar Heating Period
The context mentions "Maximum solar heating period enhances thermal transfer". This suggests we're looking at temperature data during peak solar hours. Node 2 appears to be stabilizing around 19.85-19.90 range.

## Step 6: Prediction Logic
Given:
- Node 1 is at high values (19.69-19.78) and provides upward influence
- Node 0 is declining (18.97), which would reduce its influence
- Node 2 recently peaked around 19.91 and slightly declined to 19.87
- The solar heating period suggests stabilization at high values

For the next 3 steps, I expect Node 2 to:
1. Stabilize or slightly recover after the small drop (around 19.88-19.90)
2. Maintain stability due to Node 1's high values providing thermal support
3. Possibly slight increase due to the solar heating context

Predicted values: 19.88, 19.90, 19.91

</think>

<answer>[19.88, 19.90, 19.91]</answer>
```

## 质量分析

**Case 0（etiological, idx=118, 756 tokens）**  
CoT 完整闭合（`</think>` + `<answer>D</answer>`），parsed 与 gold 一致。推理链按图结构、各节点时序峰值与传播时序展开，属于正常高质量样本。

**Case 1（entity, idx=982, 613 tokens）**  
同样格式合规，答案 C 正确。对 Node 9 作为链式网络末端监测点的解释与 gold 一致，长度适中。

**Case 2（correlation, idx=547, 857 tokens）**  
四题中最长但仍远低于 6144；明确写出 Node2→Node3→Node4 传播路径，答案 D 正确，末尾有规范 `<answer>`。

**Case 3（forecasting, idx=9, 761 tokens）**  
预测 `[19.88, 19.90, 19.91]` vs gold `[19.86, 19.97, 20.05]`，MAE≈0.077，与 report 15 单条 vLLM 结果同量级；有完整 thinking 与 `<answer>[...]</answer>`，**不是** idx=838 那种重复 loop 退化。

### 总体

1. **四题在全量 ST-Test 6144 中表现正常**：4/4 唯一 `<answer>`、0/4 触顶、选择题全对；paper_cases 对应 index **未落入**全量长尾异常（对比 entity max idx=838 的 6043 tokens 退化，见 `10.1-超长response样例.md`）。

2. **长度**：本 4 条 response 均在 613–857 tokens，彼此接近；说明论文附录样例在全量跑批里属于「中等长度、正常收束」子集，不能代表 ST-Test 里 forecasting/entity 的极端长尾分布。

3. **与 report 15 的关系**：report 15 为同 index 单独 vLLM 重跑；本报告为全量 `exp/sttest_full_*_6144/` 快照。当前抽取结果与 report 15 在答案与 MAE 上高度一致，可互相印证「官方链路 + prompt.json 下 paper_cases 合规」。

4. **使用建议**：写 PPT/周报时，paper_cases 可直接引用本报告全文 response；全量 benchmark 数字仍以 `10-ST-Test数据集效果.md` 为准；个案异常以 `10-1` 为例说明长尾风险即可。

## 附件

```text
00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl
00_new_codes/reports/artifacts/paper_case_00_etiological_idx118_response.txt
00_new_codes/reports/artifacts/paper_case_01_entity_idx982_response.txt
00_new_codes/reports/artifacts/paper_case_02_correlation_idx547_response.txt
00_new_codes/reports/artifacts/paper_case_03_forecasting_idx9_response.txt
```
