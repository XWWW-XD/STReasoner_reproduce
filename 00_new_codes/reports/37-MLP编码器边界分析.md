# STReasoner five-layer MLP time-series encoder 边界分析报告（修订版）

## 0. 结论先行

- **样本证据**：STReasoner-8B 的 ST-Test full 6144 response 中，存在大量 `<think>` 内复述历史窗口数值不保真的样本。代表证据必须是一段时间窗口级别的对比：原始真实时间序列 vs 模型推理中复述的时间序列。
- **合理推测**：这些现象和 patch-level MLP embedding 的信息压缩风险相容，但不能直接证明 MLP 是唯一原因；LLM 解读 embedding、prompt 格式、训练数据覆盖和生成策略都可能参与。
- **尚未验证**：还没有做 raw numeric side-channel、patch size ablation、embedding recoverability probe 或 encoder 替换实验，因此不能下结论说“MLP 导致数值压缩”。

## 1. 本版证据标准

本轮“数值压缩/信息压缩”不再用 MAE 大、最终预测、标注答案来判断。更接近问题本意的证据是：

- 输入原始序列包含细粒度数值，例如 `425.44, 415.47, 407.05`。
- 模型在 `<think>` 中明确复述了某个 node/window，例如 `Node 1 [27-32]`。
- 把模型复述的这一段数值，和原始 JSONL 中同一 node/window 的真实时间序列直接对齐。
- 如果两段序列不一致，才作为“推理过程复述时间序列错误”的证据。

跨 patch / 跨节点 / 延迟传播的证据标准也改成：

- 必须是 STReasoner-8B full ST-Test 输出。
- 必须记录 `<think>` 中模型复述的节点、时间段和数值。
- 必须记录原始 JSONL 中同一节点、同一时间段的真实数值。
- 可以说明它发生在跨节点/跨图结构推理语境中，但不能只用最终选项或最终预测来当证据。

不再纳入主证据链：

- Stage 1 自训 / LoRA probe 输出。
- 只按 MAE 排序的大误差 forecasting 样本。
- 没有 `<think>` 复述值和原始时间序列对齐的错例。

## 3. 样本证据：只比较推理过程复述的时间序列与原始时间序列

本节只讨论“复述时间序列错误”。判断标准是：

- 原始 JSONL 中某个节点某段时间序列是真实输入；
- 模型在 `<think>` 中复述了这个节点这个时间段的数值；
- 把模型复述的序列和原始 time series 对齐比较；
- 如果两者不一致，才记为“推理中复述时间序列错误”。

这里不看最终预测，也不看标注答案；不使用预测误差来证明信息压缩。

本版只使用已有 STReasoner-8B full ST-Test 6144 输出：

- `exp/sttest_full_forecasting_6144/generated_answer.json`
- `exp/sttest_full_correlation_6144/generated_answer.json`
- `exp/sttest_full_entity_6144/generated_answer.json`
- `exp/sttest_full_etiological_6144/generated_answer.json`

自动筛选结果：

| 证据类型 | 数量 | 说明 |
| --- | ---: | --- |
| forecasting reasoning numeric reconstruction cases | 478 | forecasting response 的 `<think>` 中显式复述某节点某时间段数值，但和 raw time series 对不上 |
| cross-node cases with numeric reconstruction check | 347 | correlation/entity/etiological response 的跨节点推理中，存在可对齐的节点时间段复述错误 |

代表样本表：

| 证据类型 | task type | sample idx | data path:line | 节点 | 时间段 | 原始真实时间序列 | 模型推理中复述的时间序列 | 错误说明 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 复述时间序列错误 | forecasting | 19 | `data/ST-Bench/ST-Test/forecasting_test.jsonl`:20 | Node 1 | 27-32 | [425.44, 415.47, 407.05, 415.2, 425.13, 118.85] | [1550.0, 188.64, 1550.0, 115.54, 1257.01, 118.85] | 模型在推理中复述该时间段时读数错误 |
| 复述时间序列错误 | forecasting | 179 | `data/ST-Bench/ST-Test/forecasting_test.jsonl`:180 | Node 2 | 62-67 | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] | [121.71, 122.12, 122.29, 122.29, 122.25, 122.12] | 模型在推理中复述该时间段时读数错误 |
| 复述时间序列错误 | forecasting | 44 | `data/ST-Bench/ST-Test/forecasting_test.jsonl`:45 | Node 0 | 46-63 | [500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0, 0.0] | [69.97, 70.09, 66.15, 60.11, 51.02, 38.71, 25.11, 9.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.88, 50.66, 106.29, 133.8] | 模型在推理中复述该时间段时读数错误 |
| 跨节点推理中的时间序列复述错误 | correlation | 480 | `data/ST-Bench/ST-Test/correlation_test.jsonl`:481 | Node 4 | 67-73 | [407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85, 3942.12] | [370.15, 370.1, 370.09, 370.11, 370.11, 370.12, 370.12] | 模型在跨节点推理中复述该节点时间段时读数错误 |

这张表是一段时间一行，不是逐点表。每一行都说明：模型复述的是哪个样本、哪个节点、哪个时间段；这一段真实时间序列是什么；模型在推理中写成了什么。

## 4. 代表样本详解

### 4.1 forecasting idx=19：Node 1 时间段 27-32 复述错误

样本：`data/ST-Bench/ST-Test/forecasting_test.jsonl` 第 20 行，idx=19。  
输出：`exp/sttest_full_forecasting_6144/generated_answer.json`。

任务：forecasting。  
图结构：`Node 0->Node 1; Node 1->Node 2; Node 2->Node 1; Node 1->Node 0`。  
模型在 `<think>` 中复述的是 Node 1 的 27-32 时间段。

原始 JSONL 中 Node 1 的真实时间序列片段是：

```text
[425.44, 415.47, 407.05, 415.2, 425.13, 118.85]
```

模型 `<think>` 中复述为：

```text
Node 1 [27-32]: 1550.00, 188.64, 1550.00, 115.54, 1257.01, 118.85
```

也就是同一个样本、同一个 Node 1、同一个 27-32 时间段，模型推理中使用的数值和真实输入明显不一致。这个样本支持的不是“预测错”，而是“模型在推理过程中读取/复述时间序列不保真”。

### 4.2 forecasting idx=179：Node 2 时间段 62-67 复述错误

样本：`data/ST-Bench/ST-Test/forecasting_test.jsonl` 第 180 行，idx=179。  
输出：`exp/sttest_full_forecasting_6144/generated_answer.json`。

模型在 `<think>` 中复述的是 Node 2 的 62-67 时间段。

原始 JSONL 中 Node 2 的真实时间序列片段是：

```text
[756.59, 736.69, 712.12, 676.66, 642.34, 607.78]
```

模型 `<think>` 中复述为：

```text
Node 2 [62-67]: 121.71, 122.12, 122.29, 122.29, 122.25, 122.12
```

这里不是小数位差异，而是整段数值尺度都错了。这个样本说明模型在推理中可能把某段 time-series embedding 解读成了完全不同的数值轨迹。

### 4.3 correlation idx=480：跨节点推理中 Node 4 时间段 67-73 复述错误

样本：`data/ST-Bench/ST-Test/correlation_test.jsonl` 第 481 行，idx=480。  
输出：`exp/sttest_full_correlation_6144/generated_answer.json`。

任务：correlation。  
图结构包含 10 个节点、10 条边，Node 1 位于 `Node 0 -> Node 1 -> Node 4` 这条局部传播路径上：

```text
Node 0->Node 1; Node 1->Node 4; Node 4->Node 3; Node 3->Node 6; Node 2->Node 9; Node 9->Node 8; Node 8->Node 7; Node 7->Node 5; Node 5->Node 4; Node 6->Node 3
```

模型在 `<think>` 中沿图结构分析 Node 1、Node 0 和 Node 4，并复述 Node 4 的 67-73 时间段。

原始 JSONL 中 Node 4 的真实时间序列片段是：

```text
[407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85, 3942.12]
```

模型 `<think>` 中复述为：

```text
Node 4 [67-73]: 370.15, 370.10, 370.09, 370.11, 370.11, 370.12, 370.12
```

这个样本用于说明：在跨节点图推理过程中，模型在推理链里使用的某个节点时间段数值本身已经和原始输入不一致。

## 5. 三个问题的修订结论

### 5.1 信息/数值压缩是否可能影响推理

**代码事实**：原始 time series 不作为完整文本进入 prompt，而是 normalize 后经 8 步 patchify，再由 MLP 压成 patch embedding。prompt 只保留 metadata。

**样本证据**：STReasoner-8B full ST-Test response 中存在 `478` 条 forecasting 样本，模型在 `<think>` 中复述某节点某时间段数值时，与 raw time series 不一致。代表样本 idx=19 中，Node 1 的真实 27-32 时间段是 `[425.44, 415.47, 407.05, 415.2, 425.13, 118.85]`，模型复述为 `[1550.00, 188.64, 1550.00, 115.54, 1257.01, 118.85]`。

**合理推测**：模型在需要从 `<ts>` embedding 中读取具体数值时可能不稳定。patch-level MLP embedding 可能是风险来源之一，因为每 8 个点被压成 1 个向量，细粒度数值无法像文本 token 一样直接逐项引用。

**尚未验证**：还不能证明是 MLP 压缩导致。需要做 raw numeric side-channel 或 embedding recoverability probe，比较模型有无 raw 数值文本时的复述正确率。

### 5.2 跨 patch、跨节点、带延迟因果链是否可能是边界

**代码事实**：MLP encoder 内不做 cross-patch / cross-node 建模。所有 patch 独立过 MLP，关系推理只能交给 LLM self-attention。

**样本证据**：在 correlation/entity/etiological 的 response 中，有 `347` 条跨节点推理样本能抽取到可对齐的节点时间段复述错误。代表样本 idx=480 中，模型在分析 `Node 0 -> Node 1 -> Node 4` 的局部传播路径时，把 Node 4 的 67-73 时间段从真实 `[407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85, 3942.12]` 复述为 `[370.15, 370.10, 370.09, 370.11, 370.11, 370.12, 370.12]`。

**合理推测**：跨节点任务需要同时读取多个节点的时间段信息并沿图结构组合。当前 encoder 不提供图传播或时延对齐，LLM 后续对 patch embeddings 的读取和组合可能不稳定。

**尚未验证**：还需要按图直径、入度、路径长度、时间窗口长度分桶，看复述错误是否随结构复杂度上升；也需要对比显式 raw numeric prompt 或 graph-aware encoder。

### 5.3 精确数值预测是否可能是边界

**代码事实**：时间序列先被压缩成 embedding，最终推理文本由 LLM 生成。若模型在 `<think>` 中要引用具体数值，它必须从 embedding 中恢复或近似读取。

**样本证据**：本报告不再用最终预测和标注答案之间的误差证明精确数值边界；只使用 `<think>` 中复述数值与原始 time series 的对比。现有样本显示，模型可能在推理阶段就把某段真实数值复述成另一段数值。

**合理推测**：如果模型连历史窗口中的具体数值都不能稳定复述，那么需要精确数值推理或预测的任务自然会受到影响。

**尚未验证**：需要专门设计“给定 `<ts>`，要求复述指定 node/window 原始数值”的最小测试，而不是用预测误差间接推断。

## 6. 目前证据不足的地方

- 这些样本证明模型 `<think>` 中存在时间序列复述不保真，但不能单独证明错误发生在 MLP encoder。
- 复述错误可能来自 MLP patch embedding 压缩、LLM 解读 embedding 失败、训练目标没有要求逐点复述、或生成 hallucination。
- 当前统计来自自动正则抽取，仍需要人工复核更多样本。
- 还没有做 raw numeric side-channel 对照，因此不能确认如果把原始数值直接写进 prompt，复述错误是否会明显减少。

## 7. 下一轮最小实验建议

1. **复述任务最小测试**：从 ST-Test 选 20 条样本，问题不要求预测，只要求“复述 Node k 在时间段 a-b 的数值”，评估 `<answer>` 中复述值和 raw time series 的误差。
2. **raw numeric side-channel 对照**：同样样本，一版只给 `<ts>`，一版额外在 prompt 中明文加入目标 node/window 的 raw values，比较复述正确率。
3. **embedding recoverability probe**：冻结 `TimeSeriesEmbedding`，训练小线性 probe 从 patch embedding 还原 patch 内 8 个原始值，直接测 embedding 是否保留细粒度数值。
4. **跨节点复述分桶**：对 cross-node 样本按节点数、边数、target 上游路径数量分桶，看复述错误是否随结构复杂度上升。

## 8. 本轮生成文件

```text
00_new_codes/scripts/mlp_encoder_focused_analysis/inspect_mlp_encoder.py
00_new_codes/scripts/mlp_encoder_focused_analysis/find_bad_cases.py
00_new_codes/scripts/mlp_encoder_focused_analysis/summarize_bad_cases.py
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/mlp_encoder_facts.json
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/reasoning_numeric_fidelity_cases.json
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/cross_node_reasoning_error_cases.json
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/cross_node_numeric_reconstruction_cases.json
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/boundary_cases_summary.json
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/boundary_cases_table.md
00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/boundary_period_alignment.md
00_new_codes/reports/37-MLP编码器边界分析.md
```
