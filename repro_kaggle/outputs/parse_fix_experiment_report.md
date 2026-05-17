# 输出格式修复实验

## 1. 目标

本实验的目标是降低由 `no_answer_tag_or_standalone_choice` 导致的解析失败。前面的实验已经证明模型可以生成文本；这里进一步检查两种修复方向：一是增加生成长度，二是追加明确的最终答案格式要求，看看模型输出是否更容易被解析为 `<answer>A</answer>` / `<answer>B</answer>` / `<answer>C</answer>` / `<answer>D</answer>`。

## 2. 实验设置

- model_name: `Time-HD-Anonymous/STReasoner-8B`
- dataset: `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`
- device_strategy: `single_gpu`，使用 `CUDA_VISIBLE_DEVICES=0`
- precision: `4bit`
- attention backend: `sdpa`
- samples: 最多 20 条，每类最多 5 条；如果存在则复用 `compare_single4bit_dualfp16_selected_indices.json`
- 基线: `max_new_tokens=64`, `answer_format_prompt=false`
- 加长生成: `max_new_tokens=256`, `answer_format_prompt=false`
- 强制答案格式: `max_new_tokens=256`, `answer_format_prompt=true`

## 3. 结果表

| 实验组 | 配置 | 解析失败率 | 解析成功数 | 准确率 | 平均延迟 | 主要错误 |
|---|---|---:|---:|---:|---:|---|
| 基线 | max_new_tokens=64, answer_format_prompt=false | 0.650 | 7 | 0.714 | 50.042 | no_answer_tag_or_standalone_choice |
| 加长生成 | max_new_tokens=256, answer_format_prompt=false | 0.550 | 9 | 0.556 | 225.799 | no_answer_tag_or_standalone_choice |
| 强制答案格式 | max_new_tokens=256, answer_format_prompt=true | 0.900 | 2 | 0.500 | 237.769 | no_answer_tag_or_standalone_choice |

## 4. 证据

### 基线

解析失败样例：

```text
index=0 category=correlation gold=<answer>A</answer>
error=no_answer_tag_or_standalone_choice


Based on the spatial-temporal analysis of the data:

**Node 2 behavior during time steps 15-19:**
- Time 15: 28.69
- Time 16: 17.24
- Time 17: 30.
```

解析成功样例：

```text
index=1 category=correlation gold=<answer>A</answer> parsed=C correct=False

Answer: C

Node 5 is influenced by Node 4, which is influenced by Node 3, which is influenced by Node 2. Node 2 receives inputs from Node 0 and Node 1. 

Looking at the time series patterns:
- Node 0 shows a major spike around time steps
```

### 加长生成

解析失败样例：

```text
index=0 category=correlation gold=<answer>A</answer>
error=no_answer_tag_or_standalone_choice


Based on the spatial-temporal analysis of the data:

**Node 2 behavior during time steps 15-19:**
- Time 15: 28.69
- Time 16: 17.24
- Time 17: 30.11
- Time 18: 12.10
- Time 19: 36.10

This shows extreme volatility with values dropping from 28.69 to 17.24, spiking to 30.11, dropping to 12.10, then spiking to 36.10.

**Key observations:**

1. **Upstream influence (Nodes 0 and 1):**
   - Node 0 at times 15-19: 20.34, 18.38, 19.62, 22.06, 24.28 (gradual increase, smooth pattern)
   - Node 1 at times 15-19: 12.13, 12.68, 13.77,
```

解析成功样例：

```text
index=1 category=correlation gold=<answer>A</answer> parsed=C correct=False

Answer: C

Node 5 is influenced by Node 4, which is influenced by Node 3, which is influenced by Node 2. Node 2 receives inputs from Node 0 and Node 1. 

Looking at the time series patterns:
- Node 0 shows a major spike around time steps 13-19 (values jumping from ~99 to peaks of 149-205)
- Node 1 shows a major spike around time steps 36-40 (values jumping from ~79 to peaks of 168)
- Node 2 shows extreme spikes and drops to 0 at multiple points (time steps 14, 16, 18, 20, 22)
- Node 3 shows moderate variations with peaks around time steps 13-19
- Node 4 shows relatively stable values mostly between 94-99 with slight increases around time steps 13-19
- Node 5 shows relatively stable values mostly between 97-101 with a small peak around time steps 13-19

The pattern suggests:
- Node 0's spike appears to be a controlled release pattern (
```

### 强制答案格式

解析失败样例：

```text
index=0 category=correlation gold=<answer>A</answer>
error=no_answer_tag_or_standalone_choice
 Do not use any other text.

Based on the data analysis:

**Step 1: Examine Node 2's behavior during time steps 15-19**
Node 2 values: 28.69, 26.21, 29.00, 18.18, 43.02
This shows extreme volatility with a sudden drop at step 18 (18.18) followed by a massive spike at step 19 (43.02)

**Step 2: Check upstream nodes (Node 0 and Node 1) during steps 15-19**
- Node 0: 16.90, 17.99, 19.79, 21.80, 23.42 (gradual, steady increase)
- Node 1: 12.13, 12.79, 13.70, 14.95, 16.28 (gradual, steady increase)

Both upstream nodes show smooth, gradual increases - no sudden spikes.

**Step 3: Examine downstream nodes (Node 3
```

解析成功样例：

```text
index=2787 category=etiological gold=<answer>C</answer> parsed=C correct=True
 Do not use any other text.
[Answer: C]

The spatial-temporal data shows three nodes (0, 1, and 2) with time series of 168 observations each. Let me analyze the characteristics:

1. **Graph Structure Analysis:**
   - Node 0 → Node 2
   - Node 1 → Node 2
   - Node 2 → Node 0
   This suggests Node 2 is a central node influenced by both Node 0 and Node 1, with feedback to Node 0.

2. **Time Series Characteristics:**

   **Node 0:** Shows clear diurnal patterns with values ranging from ~11°C to ~25°C. There are notable spikes (e.g., 23.06, 25.44) suggesting rapid temperature changes, likely solar heating effects.

   **Node 1:** Shows more moderate variations (14.59-21.27°C) with gentler fluctuations, suggesting deeper water or less direct solar influence.

   **Node 2:** Shows intermediate behavior with some extreme spikes (e.g., 23.01, 25.37, even a drop to 9.92
```

## 5. 结论

本次实验中效果最好的是 `加长生成` (`max_new_tokens=256, answer_format_prompt=false`)：parse_fail_rate=0.550，parse_success=9，avg_latency=225.799 秒。
下一轮 tiny eval 可优先尝试 `max_new_tokens=256`；本次实验里，显式答案格式提示没有带来足够改善。
