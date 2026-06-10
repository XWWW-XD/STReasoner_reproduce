# 三次 ST-Test 6144 复述错误对比（run1 / run2 / run3）

## 0. 结论先行

在 **report 39 同一 strict mismatch 口径**下，对三次官方 STReasoner-8B full ST-Test 6144 输出重新统计（不重新推理）：

1. **复述现象在三次推理中均存在**：forecasting / correlation 均有大量「`<think>` 中显式 node-window 数值列表与 raw time series 不一致」样本；entity / etiological 在 strict 口径下仍几乎抽不到可对齐数值列表（run2 出现 1 条例外）。
2. **三次 run 数字不必相同，且确实不同**：同口径下 mismatch 样本数在 run 间有 ±20 量级波动，说明「是否写出可对齐复述 + 写出后是否对得上」对采样随机性敏感；这不是工具 bug，而是三次独立推理的正文结论。
3. **脚本审计通过**：run1 回归与 report 39 完全一致（141/280、347/1592 等）；39 §5 四例正例 + 负例 idx 27 核对 OK。run2/run3 统计可信。

**脚本审计结论：通过（带口径限制）**——entity/etiological 的 `0` 仍表示 strict 规则下无可抽取数值列表，不等价于模型无复述问题。

---

## 1. 口径、路径与审计摘要

### 1.1 统计口径（同 report 39 §0）

- 只解析 `<think>` 中 `Node k [a-b]:` / `(steps a-b):` 数值列表。
- 与 ST-Test 同 idx 的 `timeseries` 对齐；0-based / 1-based 取 `mean_abs_diff` 更小者。
- `max_abs_diff > 0.01` 记 mismatch；对齐后至少 3 个数值点。

脚本：

- `00_new_codes/tools/mlp_encoder_focused_analysis/find_bad_cases.py`
- `00_new_codes/tools/mlp_encoder_focused_analysis/summarize_task_level_stats.py`

参数：`--exp-suffix` 区分 run；`--out-subdir` 区分 artifact。

### 1.2 三次 exp 目录

| task | run1 | run2 | run3 |
| --- | --- | --- | --- |
| forecasting | `exp/sttest_full_forecasting_6144/` | `..._run2_official/` | `..._run3_official/` |
| correlation | `exp/sttest_full_correlation_6144/` | 同上 | 同上 |
| entity | `exp/sttest_full_entity_6144/` | 同上 | 同上 |
| etiological | `exp/sttest_full_etiological_6144/` | 同上 | 同上 |

### 1.3 步骤 0 审计

| 检查项 | 结果 |
| --- | --- |
| 抽取 / 对齐 / 阈值 / 最小长度 | 与 39 文字口径一致，核心函数未改 |
| run1 回归（`run1_recheck/`） | forecasting 141/280、474 窗；correlation 347/1592、1179 窗；entity 0/1194；etiological 0/207 — **与 39 §2 一致** |
| 39 §5 正例 | idx 19、179（forecasting）、503、480（correlation）均在 run1 mismatch_indices |
| 负例 | forecasting idx 27 有可抽取复述且 `max_abs_diff=0.0`，**不在** mismatch_indices |
| 环境修复（非逻辑） | `ROOT` 改为 `parents[2]`；Windows 下 jsonl 读入显式 `utf-8` |

artifact：

- run1：`.../mlp_encoder_focused_analysis/run1_recheck/`
- run2：`.../run2/`
- run3：`.../run3/`

---

## 2. 四任务 mismatch 三列对比

| 任务 | run1（39） | run2 | run3 | run2−run1 | run3−run1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forecasting 样本 | 141/280 | 141/280 | 137/280 | 0 | −4 |
| forecasting 窗口 | 474 | 482 | 474 | +8 | 0 |
| correlation 样本 | 347/1592 | 326/1592 | 348/1592 | −21 | +1 |
| correlation 窗口 | 1179 | 1096 | 1169 | −83 | −10 |
| entity 样本 | 0/1194 | **1/1194** | 0/1194 | +1 | 0 |
| entity 窗口 | 0 | 4 | 0 | +4 | 0 |
| etiological 样本 | 0/207 | 0/207 | 0/207 | 0 | 0 |
| **合计样本** | **488/3273 (14.91%)** | **468/3273 (14.30%)** | **485/3273 (14.82%)** | −20 | −3 |

解读：

- forecasting：**样本数** run1/run2 同为 141，run3 略少（137），但 **窗口数** run2 多 8 条（同一样本内多窗口 mismatch）。
- correlation：三次在 326–348 之间波动；run2 最少，run3 略高于 run1。
- entity：run2 唯一出现 strict mismatch（idx **1175**，4 个窗口）；run1/run3 仍为 0 — 说明 strict 口径在 entity 上极稀疏，单次 run 多 1 条即显著变化。
- etiological：三次均为 0（strict 口径限制，同 39）。

可抽取复述样本数（extractable）：run1 forecasting 142；run2 forecasting 141、correlation 326；run3 forecasting 137、correlation 348 — 与 mismatch 样本数接近，说明主要波动来自「是否写出可对齐列表」+「列表是否对不上」的组合，而非规则漏抽。

---

## 3. 分组任务效果三列对比

分组在 **各 run 自己的 mismatch_indices** 上切分；forecasting 用当前 parser 重算 MAE/MAPE（同 39 说明）。

### 3.1 forecasting（MAE）

| 分组 | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| all | 67.82 / 123.52 | 64.30 / 135.20 | 64.75 / 133.54 |
| mismatch | 51.89 / 91.01 | 43.78 / 97.43 | 50.91 / 105.55 |
| other | 83.98 / 156.50 | 85.41 / 174.08 | 78.11 / 160.54 |

三次均呈现 **mismatch 组 MAE 低于 other 组**（与 39 一致），不能推出「复述对不上 ⇒ 任务更差」。

### 3.2 correlation（accuracy）

| 分组 | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| all | 0.8317 | 0.8310 | 0.8222 |
| mismatch | 0.8905 | 0.8742 | 0.8448 |
| other | 0.8153 | 0.8199 | 0.8159 |

run1/run2 仍见 mismatch 组 accuracy 高于 other；run3 差距缩小（0.8448 vs 0.8159），与 run3 整体 correlation accuracy 下降一致。

---

## 4. 严重度三列对比（median）

| 任务 | 指标 | run1 | run2 | run3 |
| --- | --- | ---: | ---: | ---: |
| forecasting | window_length | 10.0 | 10.0 | 10.0 |
| forecasting | max_abs_diff | 12.10 | 11.37 | 11.36 |
| correlation | window_length | 13.0 | 13.0 | 13.0 |
| correlation | max_abs_diff | 47.91 | 47.91 | 44.31 |
| entity | max_abs_diff | — | 156.01 (n=4) | — |

correlation 的 max_abs_diff 中位数 run3 略低于 run1/run2；entity 仅 run2 有 4 个窗口可算严重度。

---

## 5. idx 稳定性（mismatch 样本交集）

| 任务 | \|run1∩run2\| | \|run1∩run3\| | \|run2∩run3\| | 三 run 交集 | 仅 run1 | 仅 run2 | 仅 run3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forecasting | 114 | 114 | 118 | **104** | 17 | 13 | 9 |
| correlation | 230 | 238 | 230 | **190** | 69 | 56 | 70 |
| entity | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

- forecasting：约 **74%**（104/141）mismatch 样本在三次 run 中稳定出现。
- correlation：约 **55%**（190/347）稳定；单次 run 独有样本约 56–70 条。
- 39 §5 代表样例：idx 19、179 在 **三次 run 均为 mismatch**；idx 503、480 在 run1/run2 为 mismatch，**run3 未进入** mismatch 集（run3 推理文本中该窗口可能未写出 strict 数值列表，或对齐后 ≤0.01）。

---

## 6. 给汇报用的一句话

> 在 report 39 同一 strict 口径下，三次官方 ST-Test 6144 全量推理均存在大量 forecasting/correlation 复述数值与 raw time series 不一致；合计约 14.3%–14.9% 样本至少一处 mismatch，且三次 run 间有合理波动。分组效果仍不支持「复述对不上 ⇒ 任务更差」。entity/etiological 在 strict 规则下证据仍极稀疏（run2 仅 1 样本例外）。工具经 run1 回归与样例核对，统计可信。

---

## 7. artifact 索引

| run | 目录 |
| --- | --- |
| run1 回归 | `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/run1_recheck/` |
| run2 | `.../run2/` |
| run3 | `.../run3/` |

各目录含 `task_level_reconstruction_mismatch.json`、`task_level_metric_comparison.json`、`reconstruction_error_severity_by_task.json` 及对应 `.md`。

---

## 8. §5 代表样例 idx 19 / 179 在 run2、run3 的表现

以下与 report 39 §5 同一代表窗口：**idx 19 → Node 1 [27-32]**，**idx 179 → Node 2 [62-67]**。raw time series 来自 ST-Test，三次 run 相同；模型复述取自各 run 的 `generated_answer.json` 中 `<think>` 抽取结果。

**raw（固定）**

| idx | node / window | raw time series |
| ---: | --- | --- |
| 19 | Node 1, 27-32 | [425.44, 415.47, 407.05, 415.2, 425.13, 118.85] |
| 179 | Node 2, 62-67 | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] |

**三次 run 对比（§5 代表窗口）**

| idx | 代表窗口 | run | 计入 mismatch | 对齐 | max_abs_diff | 模型复述 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 19 | Node 1 [27-32] | run1 | 是 | 0-based | 1142.95 | [1550.0, 188.64, 1550.0, 115.54, 1257.01, 118.85] |
| 19 | Node 1 [27-32] | run2 | 是 | 1-based | 1142.95 | [1550.0, 1550.0, 1550.0, 1550.0, 1550.0, 1151.27] |
| 19 | Node 1 [27-32] | run3 | 是 | 0-based | 1142.95 | [1550.0, 1550.0, 1550.0, 1550.0, 1550.0, 115.37] |
| 179 | Node 2 [62-67] | run1 | 是 | 0-based | 634.88 | [121.71, 122.12, 122.29, 122.29, 122.25, 122.12] |
| 179 | Node 2 [62-67] | run2 | 是 | 0-based | 634.77 | [121.82, 130.91, 138.09, 143.29, 147.50, 152.23] |
| 179 | Node 2 [62-67] | run3 | **未写出该窗口** | — | — | run3 推理中 Node 2 写为 **[61-66]**，见下表 |

**idx 179 / run3 补充**（未使用 §5 的 [62-67]，但同 idx 仍有 strict mismatch）：

| run | node / window | 对齐 | max_abs_diff | 模型复述 |
| --- | --- | --- | ---: | --- |
| run3 | Node 2 [61-66] | 0-based | 562.78 | [212.69, 212.33, 212.92, 213.55, 213.27, 213.61] |

**解读**

- **idx 19**：三次 run 均在 §5 代表窗口上 mismatch；`max_abs_diff` 均为 **1142.95**（仍主要由 1550 vs ~425 量级误差主导）。run2/run3 复述更「平」（多写 1550.0），末点数值与 run1 不同，但不改变「严重对不上」结论。
- **idx 179**：run2 仍在 **[62-67]** 上 mismatch，`max_abs_diff` 与 run1 几乎相同（634.77 vs 634.88），但 **6 个点的复述序列与 run1 不同**（run2 呈递增，run1 近乎平坦 ~122）。run3 **未复述 [62-67]**，改为 [61-66] 且复述 ~212，与 raw ~700+ 仍严重偏离；该 idx 在 run3 仍为 mismatch 样本，只是代表窗口文字与 §5 不完全一致。
- 两例说明：三次 run 对 **同一 hard case** 的 mismatch **定性稳定**（均为 mismatch），但 **复述的具体数字与窗口边界** 会随采样变化；不宜把 §5 某一 run 的逐点复述当作三次 run 的固定模板。

