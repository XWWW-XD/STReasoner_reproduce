# 上游 bundled（`exp_STReasoner-8B`）MLP 复述 strict mismatch 与 run1 对照

## 0. 结论先行

在 **report 39 同一 strict mismatch 口径**下，对上游作者 repo 打包的 `exp_STReasoner-8B/reasoning_*-STReasoner-8B/` 四任务输出重新统计（不重新推理）：

1. **复述现象在 bundled 中同样成立**：forecasting **147/280**、correlation **344/1592** 存在 strict node-window 数值复述与 raw time series 不一致；合计 **491/3273（15.00%）**，与 run1 的 **488/3273（14.91%）** 同量级。
2. **bundled 与 run1 不是同一次推理**：同 idx 的最终答案、任务 metrics、mismatch idx 集均可分叉（见 §4、§3）；mismatch 计数差 **±6 样本/任务** 与 report 43 三 run 波动尺度一致，**不能**解读为工具 bug。
3. **任务指标分层**：bundled 的 `evaluation_metrics.json` 与论文 Table 对齐（correlation **0.8712**、forecasting MAE **65.59**）；run1（`exp/sttest_full_*_6144`）在 correlation / forecasting 上 **未复现** bundled 数值（corr **0.8317**、MAE **68.32**），这是 **复现 run vs 上游参考输出** 的差异，与 strict mismatch 统计口径无关。

**禁止结论（本报告遵守）**：bundled ≈ run1 同 run；mismatch 差 ⇒ 口径错；entity/etiological strict=0 ⇒ 无复述问题。

---

## 1. 口径与路径

### 1.1 统计口径（同 report 39 §0）

- 只解析 `<think>` 内 `Node k [a-b]:` / `(steps a-b):` 数值列表（**不**匹配 `Node k values at steps …` 类 prose）。
- 与 ST-Test 同 idx 的 `timeseries` 对齐；0-based / 1-based 取 `mean_abs_diff` 更小者。
- `max_abs_diff > 0.01` 记 mismatch；至少 3 个数值点。

脚本：

- `00_new_codes/scripts/mlp_encoder_focused_analysis/find_bad_cases.py`（`--exp-layout bundled`）
- `00_new_codes/scripts/mlp_encoder_focused_analysis/summarize_task_level_stats.py`

### 1.2 数据源

| 标签 | 路径 | 角色 |
| --- | --- | --- |
| **upstream-bundled** | `exp_STReasoner-8B/reasoning_{task}-STReasoner-8B/` | 作者 GitHub 打包；论文指标参考 |
| **repro-canonical run1** | `exp/sttest_full_{task}_6144/` | 本仓库远端 full ST-Test 6144（report 39/43） |

bundled `generated_answer.json` schema 为 `{idx, question_text, response}`，**无** `num_tokens` 字段；mismatch 统计**不读**该字段。

### 1.3 Artifact

| run | 目录 |
| --- | --- |
| bundled | `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/upstream_bundled/` |
| run1 回归 | `.../run1_recheck/` |

---

## 2. 四任务复述对不上统计（bundled）

| 任务 | 对不上样本数 a/b | 对不上窗口数 | 可抽取复述样本数 | 备注 |
| --- | ---: | ---: | ---: | --- |
| forecasting | 147/280 | 525 | 147 | strict node-window 数值列表 |
| correlation | 344/1592 | 1162 | 344 | 同上 |
| entity | 0/1194 | 0 | 0 | strict 口径下无可抽取列表 |
| etiological | 0/207 | 0 | 0 | 同上 |

**合计**：491/3273 = **15.00%**

分组任务效果（bundled 自身 mismatch 切分，parser 重算）：

| 任务 | 分组 | 指标 | 数值 |
| --- | --- | --- | ---: |
| forecasting | all | MAE / MAPE | 65.59 / 132.46 |
| forecasting | mismatch | MAE / MAPE | 46.47 / 91.44 |
| forecasting | other | MAE / MAPE | 86.72 / 177.80 |
| correlation | all | accuracy | 0.8712 |
| correlation | mismatch | accuracy | 0.9070 |
| correlation | other | accuracy | 0.8614 |

与 report 39 一致：**mismatch 组任务指标并不差于 other 组**，不能推出「复述对不上 ⇒ 任务失败」。

---

## 3. bundled vs run1 对照

**先读 §4 再解读本节**：二者任务 metrics 已分叉，本节只比较 **同一 strict 口径下的 mismatch 形态**。

### 3.1 四任务 mismatch 两列

| 任务 | bundled | run1（39） | bundled−run1 |
| --- | ---: | ---: | ---: |
| forecasting 样本 | 147/280 | 141/280 | +6 |
| forecasting 窗口 | 525 | 474 | +51 |
| forecasting 可抽取 | 147 | 142 | +5 |
| correlation 样本 | 344/1592 | 347/1592 | −3 |
| correlation 窗口 | 1162 | 1179 | −17 |
| correlation 可抽取 | 344 | 347 | −3 |
| entity 样本 | 0/1194 | 0/1194 | 0 |
| etiological 样本 | 0/207 | 0/207 | 0 |
| **合计样本** | **491/3273 (15.00%)** | **488/3273 (14.91%)** | **+3** |

### 3.2 idx 交集（mismatch 样本）

| 任务 | \|bundled∩run1\| | 仅 bundled | 仅 run1 | bundled | run1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forecasting | 105 | 42 | 36 | 147 | 141 |
| correlation | 209 | 135 | 138 | 344 | 347 |
| entity | 0 | 0 | 0 | 0 | 0 |
| etiological | 0 | 0 | 0 | 0 | 0 |

- forecasting：约 **71%**（105/147）mismatch 样本与 run1 重叠。
- correlation：约 **61%**（209/344）重叠；单次 snapshot 独有样本各 **135 / 138** 条，与 report 43 run 间 ±20~70 波动同量级。

---

## 4. 任务 metrics 对照（bundled vs 论文 Table vs run1）

| 任务 | bundled（`exp_STReasoner-8B`） | run1（`exp/sttest_full_*_6144`） | 论文/报告 10 参考 |
| --- | --- | --- | --- |
| forecasting MAE | **65.5935** | 68.3171 | run1 见 [`10-ST-Test数据集效果.md`](10-ST-Test数据集效果.md) |
| correlation acc | **0.8712** | 0.8317 | run1 同上 |
| entity acc | **0.7571** | 0.7479 | run1 同上 |
| etiological acc | **0.9565** | 0.9565 | 一致 |

解读：

- **bundled** 为上游打包、与论文 Table 对齐的参考输出（[`09-stage2.2效果调试.md`](09-stage2.2效果调试.md) / [`10-ST-Test数据集效果.md`](10-ST-Test数据集效果.md) 中 STReasoner-8B 指标来源）。
- **run1** 为本仓库 canonical 复现 run；correlation / forecasting **未复现** bundled 任务分，但 etiological 一致。
- idx 0（correlation）bundled 答 **A**、run1 答 **C**，证明二者为 **不同 inference snapshot**。

---

## 5. 样例核对

### 5.1 report 39 §5 代表 idx 在 bundled 上的 membership

| idx | 任务 | run1 mismatch | bundled mismatch | 说明 |
| ---: | --- | --- | --- | --- |
| 19 | forecasting | 是 | **否** | bundled 未进入 mismatch 集（推理文本中 strict 窗口不同或未超阈值） |
| 179 | forecasting | 是 | **是** | 两 run 均为 hard mismatch |
| 503 | correlation | 是 | **否** | bundled 未复述 39 §5 代表窗口或未 mismatch |
| 480 | correlation | 是 | **是** | 两 run 均为 mismatch |
| 27 | forecasting | **否**（39 负例） | **是** | bundled 将该 idx 记为 mismatch；run1 为 39 负例 |

### 5.2 bundled-only 样例

- **forecasting idx 1**：仅在 bundled mismatch 集（run1 无）；artifact 见 `upstream_bundled/manual_audit_top10_reconstruction_mismatch.json`。
- **correlation idx 10**：仅在 bundled mismatch 集；同上。

---

## 6. 限制

- **strict 口径**：不统计 prose 式「values at steps …」；entity/etiological 的 `0` 仅表示无可对齐数值列表。
- **top-5 窗口 cap**：`find_bad_cases` 对 correlation 等每样本最多写入 5 个 mismatch 窗口 → `mismatch_window_count` 可能低于全窗口重算；bundled 与 run1 **同限**，横比公平。
- **bundled 生成版本未知**：无法从本地 git 区分 clone 带入 vs 后写；以上游 GitHub 为准。
- **不得**将 bundled 当作 run1 替身，或把 run1 任务 metrics 差距解释为 mismatch 工具错误。

---

## 7. artifact 索引

| 文件 | 路径 |
| --- | --- |
| mismatch 汇总 | `upstream_bundled/task_level_reconstruction_mismatch.json` |
| 分组 metrics | `upstream_bundled/task_level_metric_comparison.json` |
| 严重度 | `upstream_bundled/reconstruction_error_severity_by_task.json` |
| Superpowers plan | `superpowers/_plans/wave-report44-bundled-mlp.md` |
| Code review | `superpowers/_reviews/wave-report44-review.md` |

---

## 8. 附录：forecasting idx 179 输入与输出

§5.1 中 bundled 与 run1 均为 hard mismatch 的代表样例。材料来自 `data/ST-Bench/ST-Test/forecasting_test.jsonl`（line 180，idx 179）与 `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json`。

**题干**（`input` 字段；各 node 的 `<ts><ts/>` 为 MLP embedding 占位，推理时不展开为明文数值）：

```
You are a spatial temporal analysis expert. Node 0 time series with length of 68: <ts><ts/>; Node 1 time series with length of 68: <ts><ts/>; Node 2 time series with length of 68: <ts><ts/>; Node 3 time series with length of 68: <ts><ts/>; Node 4 time series with length of 68: <ts><ts/>; Node 5 time series with length of 68: <ts><ts/>; Node 6 time series with length of 68: <ts><ts/>; Node 7 time series with length of 68: <ts><ts/>; Node 8 time series with length of 68: <ts><ts/>; Node 9 time series with length of 68: <ts><ts/>; Graph Structure: Node 0->Node 1; Node 1->Node 3; Node 1->Node 4; Node 2->Node 1; Node 3->Node 6; Node 3->Node 7; Node 4->Node 5; Node 5->Node 8; Node 8->Node 9; Node 2->Node 9, please analyze the spatial temporal data and answer the following question: Given the context Evening exodus from business district, predict the value of node 1 for the next 3 steps. Historical observation window: 62-67.
```

**Gold**（`output`）：`[0.0, 349.77, 0.0]`

### 8.1 数据集 raw time series（10 nodes × 68 steps）

| Node | values |
| ---: | --- |
| 0 | [986.25, 977.97, 971.2, 967.87, 943.46, 908.75, 896.84, 856.82, 821.53, 798.19, 769.53, 735.18, 695.21, 630.93, 587.81, 547.9, 513.85, 470.33, 420.52, 359.66, 312.12, 265.2, 213.06, 159.04, 109.34, 58.29, 6.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.68, 37.82, 71.28] |
| 1 | [219.65, 232.7, 302.83, 338.51, 366.91, 381.45, 377.17, 370.07, 366.97, 354.59, 352.54, 344.35, 333.24, 321.78, 307.41, 291.76, 279.54, 258.57, 240.23, 229.58, 215.29, 197.14, 184.8, 163.16, 147.74, 134.37, 113.06, 104.67, 91.42, 208.25, 107.7, 348.35, 124.91, 526.12, 380.82, 298.0, 261.74, 249.05, 245.68, 253.45, 263.64, 270.3, 278.11, 288.93, 299.75, 312.92, 321.22, 323.89, 338.24, 342.01, 346.71, 354.23, 357.25, 357.99, 364.83, 367.86, 366.99, 364.41, 357.79, 357.12, 349.74, 340.06, 340.75, 334.69, 328.09, 325.3, 314.06, 303.07] |
| 2 | [-493.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21.29, 53.82, 96.21, 144.02, 172.91, 221.07, 273.26, 316.37, 351.77, 399.52, 439.56, 489.48, 536.71, 575.75, 607.16, 645.94, 694.43, 710.41, 741.81, 761.52, 783.11, 807.15, 831.35, 840.87, 851.19, 856.45, 867.98, 885.13, 886.66, 887.2, 871.59, 880.74, 847.47, 841.26, 827.19, 794.03, 775.47, 756.59, 736.69, 712.12, 676.66, 642.34, 607.78] |
| 3 | [199.76, 204.07, 210.07, 222.2, 232.35, 243.26, 255.59, 263.44, 268.21, 267.49, 267.65, 272.07, 278.09, 272.62, 272.42, 267.91, 264.04, 260.63, 249.35, 242.6, 235.58, 225.77, 215.97, 208.84, 202.91, 190.45, 178.99, 170.69, 164.0, 17.49, 446.1, 0.0, 761.53, 0.0, 1134.85, 878.9, 693.23, 558.79, 458.33, 387.08, 341.97, 309.36, 286.21, 270.63, 262.61, 267.41, 265.79, 269.08, 263.79, 269.73, 265.83, 270.03, 272.36, 275.64, 272.82, 280.44, 283.0, 280.72, 274.27, 271.49, 273.26, 276.75, 274.81, 274.52, 272.06, 269.96, 268.96, 264.0] |
| 4 | [199.13, 199.06, 203.29, 214.68, 235.06, 252.5, 270.45, 277.1, 278.19, 278.38, 274.68, 274.83, 269.78, 266.94, 260.78, 255.89, 250.37, 241.62, 238.33, 232.51, 227.62, 225.08, 216.61, 211.04, 201.19, 197.5, 188.8, 176.59, 174.09, 170.13, 176.06, 170.96, 198.26, 189.95, 238.79, 253.37, 250.95, 248.14, 239.08, 230.51, 230.92, 232.53, 232.86, 237.87, 244.84, 240.66, 243.41, 245.2, 251.1, 257.31, 263.92, 269.19, 277.1, 275.77, 279.47, 278.85, 277.74, 278.23, 277.97, 280.39, 276.62, 272.77, 271.49, 272.01, 267.57, 264.37, 266.65, 261.19] |
| 5 | [192.59, 192.73, 189.33, 192.24, 189.62, 188.53, 197.69, 204.52, 213.83, 217.9, 226.37, 230.55, 234.2, 231.45, 227.43, 227.27, 222.23, 223.38, 224.15, 225.3, 217.84, 215.03, 213.56, 208.27, 204.69, 195.76, 195.86, 190.41, 181.84, 184.04, 180.4, 183.94, 186.68, 190.61, 192.45, 198.49, 202.88, 212.08, 213.86, 214.15, 214.88, 210.5, 209.98, 207.7, 216.25, 216.68, 221.73, 221.15, 220.37, 220.87, 218.97, 217.73, 222.09, 223.87, 228.33, 229.93, 229.7, 237.42, 239.44, 236.54, 234.67, 238.79, 238.59, 238.76, 236.06, 240.78, 235.45, 235.18] |
| 6 | [180.65, 182.91, 193.22, 195.22, 198.4, 199.7, 208.91, 208.39, 215.74, 217.43, 220.52, 221.8, 224.81, 229.26, 229.61, 226.08, 217.64, 217.37, 217.15, 211.28, 205.52, 206.29, 204.22, 202.98, 200.41, 200.76, 196.34, 192.87, 179.46, 175.6, 9.25, 493.84, 0.0, 826.61, 0.0, 1218.59, 1013.26, 845.69, 702.39, 586.87, 492.43, 430.37, 374.98, 330.6, 296.48, 279.64, 269.25, 256.5, 248.08, 237.98, 236.39, 235.23, 233.62, 228.89, 225.09, 223.57, 224.08, 225.91, 225.42, 225.7, 228.26, 221.71, 222.73, 222.42, 219.81, 218.52, 220.37, 218.6] |
| 7 | [178.79, 179.17, 180.5, 187.62, 190.97, 193.4, 200.7, 208.83, 213.56, 213.97, 213.85, 215.7, 217.05, 211.3, 212.71, 218.81, 223.2, 221.17, 217.49, 213.41, 214.25, 211.85, 212.68, 208.45, 207.03, 203.01, 194.42, 187.3, 182.37, 182.53, 6.43, 494.41, 0.0, 826.61, 0.0, 1218.59, 1019.05, 849.15, 711.02, 605.06, 509.01, 435.5, 384.95, 347.31, 313.86, 284.78, 264.81, 253.03, 242.53, 238.3, 239.53, 235.7, 231.97, 227.89, 228.33, 229.95, 228.44, 227.98, 225.7, 221.47, 224.12, 221.1, 223.91, 226.39, 220.05, 227.26, 226.6, 228.88] |
| 8 | [190.28, 183.06, 187.36, 184.79, 184.79, 186.96, 188.27, 187.34, 187.73, 196.6, 201.1, 205.33, 208.96, 208.78, 210.38, 208.93, 208.42, 204.67, 206.1, 205.22, 207.51, 206.08, 203.64, 198.47, 195.6, 198.13, 200.15, 197.65, 196.63, 191.21, 188.01, 187.86, 187.9, 186.35, 188.32, 190.52, 194.4, 198.62, 200.07, 202.38, 200.71, 207.25, 208.59, 207.34, 207.83, 203.82, 204.8, 206.61, 204.14, 200.24, 200.38, 206.41, 211.98, 210.73, 211.46, 208.52, 208.55, 208.31, 213.18, 207.82, 207.09, 211.22, 210.26, 206.43, 208.2, 209.64, 209.11, 215.22] |
| 9 | [189.45, 96.11, 109.25, 117.79, 116.38, 121.42, 122.9, 123.79, 120.36, 122.81, 127.88, 127.41, 127.23, 132.67, 128.55, 134.32, 134.73, 135.46, 133.58, 126.91, 132.58, 133.22, 135.02, 132.9, 132.54, 135.06, 141.15, 142.56, 59.79, 275.9, 39.2, 466.19, 0.0, 719.18, 519.77, 417.88, 355.54, 327.07, 319.87, 318.64, 320.52, 329.14, 346.51, 348.81, 363.35, 377.35, 383.29, 385.78, 392.6, 395.24, 395.87, 402.49, 409.03, 416.21, 423.05, 423.44, 420.74, 412.11, 409.19, 409.45, 412.08, 399.51, 399.64, 390.98, 387.58, 374.82, 364.92, 357.04] |

观测窗口 62–67（0-based 索引 61–66）raw 摘录（与复述对不上窗口对应）：

| node | window | raw time series |
| ---: | --- | --- |
| 0 | 62–67 | [0.0, 0.0, 0.0, 13.68, 37.82, 71.28] |
| 1 | 62–67 | [340.75, 334.69, 328.09, 325.3, 314.06, 303.07] |
| 2 | 62–67 | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] |

### 8.2 bundled 复述 mismatch（观测窗口 62–67）

| node | window | 对齐 | raw time series | bundled 复述 | max_abs_diff | 原文行 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 2 | 62–67 | 0-based | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] | [210.31, 210.97, 212.19, 213.26, 213.7, 214.98] | 546.28 | `Node 2 [62-67]: 210.31, 210.97, 212.19, 213.26, 213.70, 214.98` |
| 1 | 62–67 | 0-based | [340.75, 334.69, 328.09, 325.3, 314.06, 303.07] | [108.37, 110.96, 112.23, 113.98, 114.99, 116.77] | 232.38 | `Node 1 [62-67]: 108.37, 110.96, 112.23, 113.98, 114.99, 116.77` |
| 0 | 62–67 | 0-based | [0.0, 0.0, 0.0, 13.68, 37.82, 71.28] | [29.16, 31.39, 33.52, 37.79, 41.23, 42.88] | 33.52 | `Node 0 [62-67]: 29.16, 31.39, 33.52, 37.79, 41.23, 42.88` |

对比 run1（report 39 §9.2）：run1 对 Node 2 复述为 `[121.71, …]`，bundled 复述为 `[210.31, …]`；两 run 均与 raw 严重偏离，但复述数值本身也不同。

### 8.3 bundled 完整 response

```
<think>

Let me analyze this spatial-temporal data step by step.

First, I'll identify the relevant nodes and their relationships:
- Node 0 → Node 1
- Node 2 → Node 1
- Node 1 → Node 3
- Node 1 → Node 4

So Node 1 receives input from Node 0 and Node 2, and outputs to Node 3 and Node 4.

The observation window is time steps 62-67 (indices 61-66 in 0-based indexing). Let me extract the relevant values:

Node 0 [62-67]: 29.16, 31.39, 33.52, 37.79, 41.23, 42.88
Node 1 [62-67]: 108.37, 110.96, 112.23, 113.98, 114.99, 116.77
Node 2 [62-67]: 210.31, 210.97, 212.19, 213.26, 213.70, 214.98

Now I need to predict Node 1 for steps 68, 69, 70.

Let me analyze the trends:
1. Node 1 shows a clear increasing trend in the observation window: from 108.37 to 116.77
2. Node 0 is also increasing: from 29.16 to 42.88
3. Node 2 is increasing: from 210.31 to 214.98

Looking at the pattern:
- Node 1 appears to have a steady upward trend
- The differences in Node 1: 
  - 110.96-108.37=2.59
  - 112.23-110.96=1.27
  - 113.98-112.23=1.75
  - 114.99-113.98=1.01
  - 116.77-114.99=1.78
  - Average increment ≈ 1.5

The context is "Evening exodus from business district", which suggests traffic/flow increasing as people leave work.

For the next 3 predictions (steps 68, 69, 70):
- Step 68: 116.77 + 1.5 ≈ 118.27
- Step 69: 118.27 + 1.5 ≈ 119.77
- Step 70: 119.77 + 1.5 ≈ 121.27

Let me also check what happened at step 67 (index 67) in the original data:
Node 1 at position 67 (7th value in the window) is 303.07 - wait, let me recount.

Actually, looking at the data again: The sequence given for Node 1 is of length 68. Positions 62-67 (counting from 1) would be indices 61-66.

Node 1[61] = 292.17
Node 1[62] = 270.70
Node 1[63] = 253.79
Node 1[64] = 239.21
Node 1[65] = 226.71
Node 1[66] = 215.35
Node 1[67] = 205.14

Wait, I think I miscounted earlier. Let me recalculate with correct indices:

Observation window 62-67 (indices 61-66):
Node 0: 29.16, 31.39, 33.52, 37.79, 41.23, 42.88
Node 1: 292.17, 270.70, 253.79, 239.21, 226.71, 215.35

Now I see - Node 1 is actually decreasing in this window, but the last value at position 67 is 205.14.

So I need to predict positions 68, 69, 70 (indices 67, 68, 69):
Node 1[67] = 205.14 (given)
Node 1[68] = ?
Node 1[69] = ?
Node 1[70] = ?

The trend shows Node 1 is decreasing during the observation window (62-67), but this contradicts the "evening exodus" context. Let me recheck:

Looking at the full Node 1 series: 219.65, 222.21, 223.26, 224.76, 226.91, 228.10, 229.66, 231.35, 233.21, 234.91, 236.81, 238.62, 240.67, 242.10, 243.90, 245.22, 246.91, 292.17, 270.70, 253.79, 239.21, 226.71, 215.35, 205.14, 213.01, 204.35, 212.73, 203.93, 195.99, 190.27, 186.05, 183.15, 181.20, 179.76, 178.71, 178.15, 177.67, 177.35, 177.15, 177.02, 176.97, 176.98, 176.99, 303.07, 103.95, 343.21, 92.71, 399.16, 68.91, 461.41, 35.90, 526.12, 100.39, 526.06, 132.41, 524.11, 167.59, 240.31, 203.70, 232.90, 211.59, 243.35

The pattern shows Node 1 was increasing up to around index 18-19, then decreased until around index 40, then shows spikes.

Given the context of "evening exodus from business district" and that we're at position 67 (205.14), which is in the later part of the series where there were decreases followed by spikes, I need to consider the recent trend.

Recent values around position 67: 213.01, 204.35, 212.73, 203.93, 195.99, 190.27, 186.05, 183.15, 181.20, 179.76, 178.71, 178.15, 177.67, 177.35, 177.15, 177.02, 176.97, 176.98, 176.99, 176.99, 303.07 (spike at 70)

At position 67: 205.14
At position 68: 213.01 (actually exists)
At position 69: 204.35 (actually exists)
At position 70: 212.73 (actually exists)

Wait, I need to predict positions 68, 69, 70, which are already in the data:
Position 68: 213.01
Position 69: 204.35
Position 70: 212.73

But the question says "predict the value of node 1 for the next 3 steps" after the observation window 62-67. If the observation window ends at 67, then the next 3 steps would be 68, 69, 70.

Looking at the actual values at these positions:
68: 213.01
69: 204.35
70: 212.73

But I think the question might be asking to predict beyond what's already given. Let me reconsider.

The last value given is at position 67 (205.14), so predicting 68, 69, 70.

Looking at the pattern around this area, there are alternating high and low values. The trend seems to be oscillating.

Given the recent pattern and the context of evening exodus, I would expect an increase. Looking at the values:
- Position 67: 205.14
- Position 68: 213.01 (actual exists)
- Position 69: 204.35 (actual exists)
- Position 70: 212.73 (actual exists)

If I need to predict these, I can use the actual values or extrapolate the trend.

Looking at the overall pattern after the major spike around position 70 (303.07), the values settle around 177-178 range, then spike again.

Given the evening exodus context and the recent upward movement before the drop to 205.14, I predict a continuation of increase:
- Step 68: ~213
- Step 69: ~220
- Step 70: ~228

</think>

<answer>[213, 220, 228]</answer>
```

bundled 预测 `[213, 220, 228]`，gold 为 `[0.0, 349.77, 0.0]`；模型在 thinking 中段已意识到 Node 1 窗口应为递减序列（292.17→215.35），但首轮复述与最终预测仍沿用错误的低幅递增值。

---

## 8. 附录：forecasting idx 179 输入 time series 与 bundled response

§5.1 中 idx 179 为 bundled 与 run1 共同 hard mismatch 样例。数据源：`data/ST-Bench/ST-Test/forecasting_test.jsonl` 第 180 行；bundled 输出：`exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/generated_answer.json`。

**题干**：Given the context Evening exodus from business district, predict the value of node 1 for the next 3 steps. Historical observation window: 62-67.（10 节点，每节点长度 68。）

### 8.1 输入 time series（raw，观测窗口 62-67，0-based 对齐）

| node | raw [62-67] |
| ---: | --- |
| 0 | [0.0, 0.0, 0.0, 13.68, 37.82, 71.28] |
| 1 | [340.75, 334.69, 328.09, 325.3, 314.06, 303.07] |
| 2 | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] |
| 3 | [274.81, 274.52, 272.06, 269.96, 268.96, 264.0] |
| 4 | [271.49, 272.01, 267.57, 264.37, 266.65, 261.19] |
| 5 | [238.59, 238.76, 236.06, 240.78, 235.45, 235.18] |
| 6 | [222.73, 222.42, 219.81, 218.52, 220.37, 218.6] |
| 7 | [223.91, 226.39, 220.05, 227.26, 226.6, 228.88] |
| 8 | [210.26, 206.43, 208.2, 209.64, 209.11, 215.22] |
| 9 | [399.64, 390.98, 387.58, 374.82, 364.92, 357.04] |

全序列见 jsonl `timeseries` 字段（10×68）；上表为题目指定观测窗。

### 8.2 gold 与 bundled 输出

| 项 | 内容 |
| --- | --- |
| gold（`output`） | `[0.0, 349.77, 0.0]` |
| bundled `<answer>` | `[213, 220, 228]` |

### 8.3 bundled thinking 复述 vs raw（代表窗口）

| node | window | raw | bundled 复述 | max_abs_diff |
| ---: | --- | --- | --- | ---: |
| 2 | 62-67 | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] | [210.31, 210.97, 212.19, 213.26, 213.70, 214.98] | 546.28 |
| 1 | 62-67 | [340.75, 334.69, 328.09, 325.3, 314.06, 303.07] | [108.37, 110.96, 112.23, 113.98, 114.99, 116.77] | 232.38 |
| 0 | 62-67 | [0.0, 0.0, 0.0, 13.68, 37.82, 71.28] | [29.16, 31.39, 33.52, 37.79, 41.23, 42.88] | 41.23 |

**bundled thinking 摘录**：

```
Node 0 [62-67]: 29.16, 31.39, 33.52, 37.79, 41.23, 42.88
Node 1 [62-67]: 108.37, 110.96, 112.23, 113.98, 114.99, 116.77
Node 2 [62-67]: 210.31, 210.97, 212.19, 213.26, 213.70, 214.98
```

与 report 39 §9.2（run1）对比：同一 idx 的 strict mismatch 形态一致（Node 2 窗口 max_abs_diff 量级 ~500+），但 bundled 与 run1 的复述数值、最终 `<answer>` 仍可分叉（run1：`[131.5, 136.8, 142.0]`）。
