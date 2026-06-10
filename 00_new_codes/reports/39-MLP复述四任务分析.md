# MLP 时间序列复述错误四任务定量分析报告

## 0. 结论先行

本报告重新核查 `37-MLP编码器边界分析.md` 中“模型在 `<think>` 中复述时间序列和 raw time series 对不上”的统计。这里的“对不上”只按一个标准判断：

- 模型 response 的 `<think>` 中显式写出 `Node k [a-b]: 数值列表` 或 `Node k (steps a-b): 数值列表`。
- 在原始 ST-Test JSONL 中取同一 sample、同一 node、同一 window 的 raw time series。
- 同时尝试 0-based 和 1-based 两种窗口对齐，保留误差更小的对齐方式。
- 若最优对齐后 `max_abs_diff_between_stated_and_raw > 0.01`，才记为“复述对不上”。

按这个更保守的口径，当前已有 STReasoner-8B full ST-Test 6144 输出中：

- forecasting：`141/280` 个样本至少有一个复述窗口对不上，共 `474` 个对不上窗口。
- correlation：`347/1592` 个样本至少有一个复述窗口对不上，共 `1179` 个对不上窗口。
- entity：`0/1194`，当前严格 node-window 数值列表口径下未发现可统计的对不上样本。
- etiological：`0/207`，当前严格 node-window 数值列表口径下未发现可统计的对不上样本。

这里的 `0` 不能解释为“模型没有问题”。entity / etiological 的 response 更多是定性描述，例如“values around 37-42”“shows spikes”，不是严格的 `Node k [a-b]: [v1, v2, ...]` 数值列表，因此当前自动规则几乎没有可对齐对象。

更重要的是，分组效果没有显示“复述对不上样本一定更差”：

- forecasting 中，复述对不上组当前 parser 重算 MAE 为 `51.8912`，其余组为 `83.9794`。
- correlation 中，复述对不上组 accuracy 为 `0.8905`，其余组为 `0.8153`。

因此，这个现象可以说明模型有时不能稳定复述 embedding 中的具体时间序列数值，但不能直接推出“5 层 MLP 导致任务失败”。它更适合作为下一步复述 probe / raw numeric side-channel 对照实验的动机。

## 1. 数据与口径

使用的输出和数据：

- `exp/sttest_full_forecasting_6144/generated_answer.json`
- `exp/sttest_full_correlation_6144/generated_answer.json`
- `exp/sttest_full_entity_6144/generated_answer.json`
- `exp/sttest_full_etiological_6144/generated_answer.json`
- `data/ST-Bench/ST-Test/{forecasting,correlation,entity,etiological}_test.jsonl`

统计脚本：

- `00_new_codes/scripts/mlp_encoder_focused_analysis/find_bad_cases.py`
- `00_new_codes/scripts/mlp_encoder_focused_analysis/summarize_task_level_stats.py`

本轮重新生成的 artifact：

- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/task_level_reconstruction_mismatch.json`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/task_level_reconstruction_mismatch.md`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/task_level_metric_comparison.json`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/task_level_metric_comparison.md`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/reconstruction_error_severity_by_task.json`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/reconstruction_error_severity_by_task.md`
- `00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/manual_audit_top10_reconstruction_mismatch.json`

注意：forecasting 的分组指标使用当前仓库 parser 重算。已有 `exp/sttest_full_forecasting_6144/evaluation_metrics.json` 记录的是 `evaluated_samples=280, MAE=68.3171, MAPE=123.2891`；当前 parser 重算为 `evaluated_samples=278, MAE=67.8199, MAPE=123.5208`，缺失 idx 为 `[23, 206]`。因此本报告的 forecasting 分组 MAE/MAPE 只在“当前 parser 重算”内部比较，不混用旧 metrics 文件。

## 2. 四任务复述对不上统计

比例表

| 任务          | 失配比例 |
| ----------- | ---------: |
| forecasting |    141/280 |
| correlation |   347/1592 |
| entity      |     0/1194 |
| etiological |      0/207 |


具体样例表

| 任务 | 对不上样本数 a/b | 对不上窗口数 | 可抽取复述样本数 | 对不上样本 idx 全列表 | 备注 |
| --- | ---: | ---: | ---: | --- | --- |
| forecasting | 141/280 | 474 | 142 | [0, 2, 3, 4, 6, 7, 9, 10, 13, 15, 19, 22, 25, 26, 30, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 50, 54, 57, 60, 62, 64, 65, 67, 76, 78, 79, 80, 86, 87, 89, 90, 94, 96, 98, 100, 101, 104, 105, 106, 108, 109, 110, 112, 114, 116, 124, 125, 126, 127, 128, 132, 134, 135, 136, 137, 138, 139, 140, 142, 143, 148, 153, 154, 155, 157, 164, 165, 167, 168, 169, 170, 172, 173, 174, 175, 179, 183, 188, 189, 193, 194, 195, 198, 200, 201, 202, 204, 206, 207, 209, 210, 211, 214, 217, 218, 219, 220, 223, 224, 225, 226, 227, 229, 230, 233, 234, 235, 236, 238, 239, 241, 242, 243, 244, 245, 246, 248, 249, 252, 253, 255, 257, 258, 259, 266, 269, 270, 273, 274, 275, 276] | 严格 node-window 数值列表抽取，已尝试 0-based/1-based 保守对齐 |
| correlation | 347/1592 | 1179 | 347 | [14, 15, 16, 21, 25, 37, 41, 65, 73, 78, 86, 87, 92, 97, 99, 104, 109, 115, 117, 127, 138, 141, 145, 149, 158, 170, 175, 180, 181, 191, 198, 200, 201, 212, 216, 218, 219, 220, 222, 223, 224, 230, 237, 256, 258, 261, 264, 265, 266, 267, 269, 275, 276, 279, 284, 285, 288, 295, 296, 297, 300, 313, 321, 328, 333, 340, 344, 351, 360, 375, 377, 378, 379, 387, 392, 396, 397, 398, 405, 414, 417, 421, 422, 426, 427, 429, 431, 432, 437, 438, 439, 448, 453, 458, 464, 473, 474, 477, 480, 482, 487, 489, 490, 503, 504, 520, 526, 532, 537, 540, 547, 548, 550, 554, 556, 557, 561, 562, 563, 564, 566, 567, 579, 588, 590, 593, 594, 598, 601, 607, 622, 627, 635, 636, 639, 642, 644, 646, 656, 659, 660, 661, 664, 666, 668, 675, 699, 701, 703, 705, 711, 712, 715, 717, 725, 737, 740, 750, 753, 755, 768, 778, 779, 785, 787, 791, 793, 794, 796, 800, 805, 810, 812, 814, 815, 816, 817, 823, 842, 849, 855, 861, 863, 872, 873, 875, 876, 881, 886, 892, 897, 898, 913, 916, 917, 918, 934, 938, 941, 948, 956, 957, 959, 963, 970, 975, 985, 986, 987, 989, 998, 999, 1004, 1009, 1011, 1016, 1019, 1020, 1021, 1025, 1027, 1029, 1031, 1036, 1041, 1042, 1044, 1059, 1060, 1064, 1069, 1075, 1080, 1105, 1106, 1108, 1113, 1118, 1119, 1124, 1128, 1129, 1138, 1142, 1144, 1145, 1146, 1155, 1156, 1163, 1171, 1173, 1180, 1188, 1190, 1196, 1197, 1202, 1207, 1218, 1221, 1223, 1225, 1227, 1228, 1231, 1232, 1235, 1236, 1238, 1258, 1260, 1267, 1268, 1275, 1278, 1288, 1295, 1302, 1303, 1305, 1309, 1312, 1314, 1333, 1347, 1350, 1358, 1360, 1362, 1368, 1374, 1383, 1384, 1385, 1389, 1396, 1397, 1398, 1411, 1420, 1423, 1428, 1440, 1442, 1445, 1448, 1453, 1475, 1476, 1479, 1480, 1483, 1484, 1487, 1489, 1490, 1493, 1499, 1500, 1503, 1507, 1508, 1510, 1517, 1530, 1531, 1534, 1537, 1538, 1539, 1546, 1548, 1552, 1559, 1560, 1568, 1569, 1571, 1574, 1575, 1577, 1579, 1580, 1585, 1586, 1590] | 严格 node-window 数值列表抽取，已尝试 0-based/1-based 保守对齐 |
| entity | 0/1194 | 0 | 0 | [] | 当前严格 node-window 数值列表抽取口径下未发现对不上；不等价于模型无此类问题 |
| etiological | 0/207 | 0 | 0 | [] | 当前严格 node-window 数值列表抽取口径下未发现对不上；不等价于模型无此类问题 |

总体比例：488/3273 = 14.66%

解读：

- forecasting 和 correlation 的可统计证据确实比较多，值得做下一步复述 probe。
- entity / etiological 不是“对得都很好”，而是当前 response 很少写出严格可对齐的 node-window 数值列表；需要另设复述任务才能判断。

## 3. 复述错误样本与任务效果的关系

| 任务 | 分组 | 样本数 | evaluated | missing | 指标 | 数值 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| forecasting | all | 280 | 278 | 2 | MAE / MAPE | 67.8199 / 123.5208 |
| forecasting | mismatch | 141 | 140 | 1 | MAE / MAPE | 51.8912 / 91.0088 |
| forecasting | other | 139 | 138 | 1 | MAE / MAPE | 83.9794 / 156.5040 |
| correlation | all | 1592 | 1592 | 0 | accuracy | 0.8317 |
| correlation | mismatch | 347 | 347 | 0 | accuracy | 0.8905 |
| correlation | other | 1245 | 1245 | 0 | accuracy | 0.8153 |
| entity | all | 1194 | 1194 | 0 | accuracy | 0.7479 |
| entity | mismatch | 0 | 0 | 0 | accuracy | N/A |
| entity | other | 1194 | 1194 | 0 | accuracy | 0.7479 |
| etiological | all | 207 | 207 | 0 | accuracy | 0.9565 |
| etiological | mismatch | 0 | 0 | 0 | accuracy | N/A |
| etiological | other | 207 | 207 | 0 | accuracy | 0.9565 |

这个表说明：复述错误和最终任务效果不是简单同向关系。

- forecasting 里，复述错误组的 MAE/MAPE 反而低于其余组。
- correlation 里，复述错误组 accuracy 也高于其余组。
- 一个可能原因是：模型越倾向于在 `<think>` 中显式列数值，越容易被当前规则抽到“复述错误”；而没有显式复述的样本不会进入 mismatch 组。
- 因此，本报告不能把“复述对不上”直接解释为“任务错误原因”。它只能说明模型显式复述 raw time series 时存在数值不保真现象。

## 4. 复述错误严重度

| 任务 | 指标 | count | median | mean | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| forecasting | window_length | 474 | 10.0000 | 12.5211 | 4.0000 | 50.0000 |
| forecasting | window_ratio | 474 | 0.1714 | 0.2501 | 0.0495 | 0.8372 |
| forecasting | mean_abs_diff | 474 | 6.0488 | 25.7588 | 0.0048 | 604.3133 |
| forecasting | max_abs_diff | 474 | 12.0950 | 49.9802 | 0.0119 | 1142.9500 |
| correlation | window_length | 1179 | 13.0000 | 14.9966 | 3.0000 | 61.0000 |
| correlation | window_ratio | 1179 | 0.1458 | 0.1712 | 0.0125 | 0.7708 |
| correlation | mean_abs_diff | 1179 | 20.4647 | 64.3803 | 0.0400 | 1383.1000 |
| correlation | max_abs_diff | 1179 | 47.9100 | 148.3719 | 0.0600 | 3571.9200 |
| entity | window_length | 0 | N/A | N/A | N/A | N/A |
| entity | window_ratio | 0 | N/A | N/A | N/A | N/A |
| entity | mean_abs_diff | 0 | N/A | N/A | N/A | N/A |
| entity | max_abs_diff | 0 | N/A | N/A | N/A | N/A |
| etiological | window_length | 0 | N/A | N/A | N/A | N/A |
| etiological | window_ratio | 0 | N/A | N/A | N/A | N/A |
| etiological | mean_abs_diff | 0 | N/A | N/A | N/A | N/A |
| etiological | max_abs_diff | 0 | N/A | N/A | N/A | N/A |

严重度表说明：

- forecasting 对不上窗口的长度中位数为 10，占原序列长度比例中位数为 0.1714。
- correlation 对不上窗口的长度中位数为 13，占原序列长度比例中位数为 0.1458。
- correlation 的数值偏差整体更大：`mean_abs_diff` 中位数 20.4647，`max_abs_diff` 中位数 47.9100。

## 5. 人工核查样例

完整 top10 核查结果在：

`00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/manual_audit_top10_reconstruction_mismatch.json`

代表样例：

| 任务 | idx | node/window | 对齐口径 | raw time series | 模型 `<think>` 复述 |
| --- | ---: | --- | --- | --- | --- |
| forecasting | 19 | Node 1, 27-32 | 0-based | [425.44, 415.47, 407.05, 415.2, 425.13, 118.85] | [1550.0, 188.64, 1550.0, 115.54, 1257.01, 118.85] |
| forecasting | 179 | Node 2, 62-67 | 0-based | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] | [121.71, 122.12, 122.29, 122.29, 122.25, 122.12] |
| correlation | 503 | Node 4, 68-78 | 1-based | [407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85, 3942.12, 519.68, 1339.75, 2971.15, 0.0] | [370.16, 370.19, 370.21, 370.19, 370.2, 370.19, 370.2, 370.2, 370.2, 370.2, 370.2] |
| correlation | 480 | Node 4, 67-73 | 1-based | [411.61, 407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85] | [370.15, 370.1, 370.09, 370.11, 370.11, 370.12, 370.12] |

## 6. 是否真的是 5 层 MLP 导致

目前不能下这个结论。

能确认的代码事实是：

- time series 不作为完整 raw numeric 文本进入 prompt。
- `TimeSeriesEmbedding` 先对数值序列做 normalize / patchify。
- `patch_size=8`，每个 patch 经 5 层 MLP 得到 1 个 2560 维 embedding。
- MLP encoder 内没有 cross-patch attention、RNN、卷积、state-space 或图传播。
- cross-patch / cross-node 推理只能在 patch embeddings 进入 Qwen3 主干后，由 LLM self-attention 间接完成。

能确认的现象是：

- 模型有时会在 `<think>` 中显式复述某个 node/window 的数值。
- 这些复述值有时和 raw time series 明显不一致。

不能确认的是：

- 错误一定发生在 5 层 MLP encoder。
- 错误一定导致最终任务失败。
- 错误一定是信息压缩导致，而不是 LLM 生成幻觉、训练目标不要求逐点复述、prompt 中缺少 raw numeric 明文、或者抽取规则只覆盖显式数值列表导致的样本选择偏差。

更稳妥的说法是：复述错误现象与 patch-level MLP embedding 的信息压缩风险相容，但还没有因果证据证明 5 层 MLP 是唯一原因。

## 7. 下一步最小复述 probe 设计

建议单独做一个小型模型复述实验，而不是继续从自然 `<think>` 中间接推断。

设计：

- 模型：同一个 STReasoner-8B。
- 数据：四任务分别抽样。
- forecasting：从复述错误样本中抽 20 条，从非复述错误样本中抽 20 条。
- correlation：从复述错误样本中抽 20 条，从非复述错误样本中抽 20 条。
- entity / etiological：由于当前严格证据不足，先各抽 10 条 response 中有定性时间序列描述的样本，人工指定 node/window。
- prompt：保留原始 `<ts><ts/>` 输入和 graph，只把问题改为“请精确复述 Node k 在时间段 a-b 的原始数值，并用 `<answer>[...]</answer>` 输出”。
- max_tokens：保持 6144。
- 指标：parse_success、长度是否匹配、mean_abs_diff、max_abs_diff、按任务聚合复述误差。

如果要进一步判断 5 层 MLP 的因果作用，再做 raw numeric side-channel 对照：

- A 版：只给 `<ts><ts/>`。
- B 版：额外把目标 node/window 的 raw values 明文写入 prompt。
- 如果 B 版复述显著更准，说明当前 embedding-only 输入对精确复述不友好；但仍需进一步区分是 MLP 压缩、LLM 解码 embedding，还是训练目标问题。

## 8. 本轮结论给汇报用

可以汇报为：

> 我们重新核查了 STReasoner-8B 在 full ST-Test 6144 输出中的 `<think>` 时间序列复述。按严格 node-window 数值列表口径，并同时尝试 0-based/1-based 保守对齐后，forecasting 有 141/280 个样本至少出现一处复述值和 raw time series 不一致，correlation 有 347/1592 个样本出现此现象；entity 和 etiological 当前没有足够严格可抽取的数值列表证据，不能说明没有问题。分组效果显示，有复述错误的样本并不一定任务效果更差，因此该现象目前只能说明模型显式复述时间序列时存在数值不保真，不能直接证明 5 层 MLP 是最终任务错误的原因。下一步需要做受控的 node/window 复述 probe 和 raw numeric side-channel 对照。

## 9. §5 代表样例完整材料

以下四例与 §5 表格一一对应，材料来自 `exp/sttest_full_*_6144/` 与 `mismatch_sample_cards.json`（`00_new_codes/reports/artifacts/mlp_encoder_focused_analysis/`）。题干中的 `<ts><ts/>` 为原始 embedding 占位，未展开为明文数值。

### 9.1 forecasting，idx 19（§5 代表窗口：Node 1，27-32）

**题干**：Given the context Evening commute flow reaches suburb, predict the value of node 0 for the next 3 steps. Historical observation window: 27-32.

**复述对不上窗口**（0-based/1-based 取误差更小对齐；`max_abs_diff > 0.01` 记为 mismatch）：

| node | window | 对齐 | raw time series | 模型复述 | max_abs_diff | 原文行 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | 27-32 | 0-based | [425.44, 415.47, 407.05, 415.2, 425.13, 118.85] | [1550.0, 188.64, 1550.0, 115.54, 1257.01, 118.85] | 1142.95 | `Node 1 [27-32]: 1550.00, 188.64, 1550.00, 115.54, 1257.01, 118.85` |
| 2 | 27-32 | 1-based | [254.38, 267.09, 293.57, 280.34, 272.95, 260.38] | [1250.0, 112.33, 1250.0, 125.0, 1091.32, 261.61] | 995.62 | `Node 2 [27-32]: 1250.00, 112.33, 1250.00, 125.00, 1091.32, 261.61` |
| 0 | 27-32 | 1-based | [719.68, 685.18, 709.34, 760.08, 821.91, 920.55] | [390.1, 482.22, 551.37, 636.13, 731.99, 890.67] | 329.58 | `Node 0 [27-32]: 390.10, 482.22, 551.37, 636.13, 731.99, 890.67` |

**模型输出（摘录）**：

```
Node 0 [27-32]: 390.10, 482.22, 551.37, 636.13, 731.99, 890.67
Node 1 [27-32]: 1550.00, 188.64, 1550.00, 115.54, 1257.01, 118.85
Node 2 [27-32]: 1250.00, 112.33, 1250.00, 125.00, 1091.32, 261.61
```

最终答案：`<answer>[1210, 1330, 1450]</answer>`（gold：`[0.0, 1250.0, 0.0]`）。

### 9.2 forecasting，idx 179（§5 代表窗口：Node 2，62-67）

**题干**：Given the context Evening exodus from business district, predict the value of node 1 for the next 3 steps. Historical observation window: 62-67.

| node | window | 对齐 | raw time series | 模型复述 | max_abs_diff | 原文行 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 2 | 62-67 | 0-based | [756.59, 736.69, 712.12, 676.66, 642.34, 607.78] | [121.71, 122.12, 122.29, 122.29, 122.25, 122.12] | 634.88 | `Node 2 [62-67]: 121.71, 122.12, 122.29, 122.29, 122.25, 122.12` |
| 1 | 62-67 | 0-based | [340.75, 334.69, 328.09, 325.3, 314.06, 303.07] | [106.95, 110.39, 113.28, 116.75, 122.93, 126.21] | 233.8 | `Node 1 [62-67]: 106.95, 110.39, 113.28, 116.75, 122.93, 126.21` |
| 0 | 62-67 | 0-based | [0.0, 0.0, 0.0, 13.68, 37.82, 71.28] | [29.5, 31.89, 39.92, 47.96, 56.91, 66.59] | 39.92 | `Node 0 [62-67]: 29.50, 31.89, 39.92, 47.96, 56.91, 66.59` |

**模型输出（摘录）**：

```
Node 0 [62-67]: 29.50, 31.89, 39.92, 47.96, 56.91, 66.59
Node 1 [62-67]: 106.95, 110.39, 113.28, 116.75, 122.93, 126.21
Node 2 [62-67]: 121.71, 122.12, 122.29, 122.29, 122.25, 122.12
```

最终答案：`<answer>[131.5, 136.8, 142.0]</answer>`（gold：`[0.0, 349.77, 0.0]`）。

### 9.3 correlation，idx 503（§5 代表窗口：Node 4，68-78）

**题干**：Which statement best describes the relationship between Node 0 and Node 4 during time steps 68-78 (1 time step = 15 minutes)? Options: A. Morning commuters travel from outer area to CBD via multiple transit points B. Evening commuters flow from CBD (Node 0) to outer area (Node 4) via transit hub (Node 1) during peak exodus C. There is direct traffic flow between CBD and outer area without intermediate stops D. Local traffic circulates between neighboring nodes without clear directional pattern

| node | window | 对齐 | raw time series | 模型复述 | max_abs_diff | 原文行 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 4 | 68-78 | 1-based | [407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85, 3942.12, 519.68, 1339.75, 2971.15, 0.0] | [370.16, 370.19, 370.21, 370.19, 370.2, 370.19, 370.2, 370.2, 370.2, 370.2, 370.2] | 3571.92 | `Node 4 (steps 68-78): 370.16, 370.19, ...` |
| 1 | 68-78 | 0-based | [1658.61, 1041.98, 1254.24, 1545.15, 2347.4, 2457.98, 1027.98, 2450.53, 49.46, 1592.19, 18.25] | [660.91, 1109.0, 975.1, 1220.21, 1002.29, 1429.15, 709.36, 1711.49, 152.9, 2457.98, 14.74] | 1345.11 | `Node 1 (steps 68-78): 660.91, 1109.00, ...` |
| 0 | 68-78 | 0-based | [1371.17, 1262.46, 1421.44, 1985.19, 2534.61, 1908.89, 1811.26, 1101.14, 854.71, 694.87, 721.11] | [999.22, 1103.71, 1201.5, 1212.26, 1205.01, 1209.06, 1027.21, 909.06, 776.99, 662.11, 553.58] | 1329.6 | `Node 0 (steps 68-78): 999.22, 1103.71, ...` |

**模型输出（摘录）**：

```
Node 0 (steps 68-78): 999.22, 1103.71, 1201.50, 1212.26, 1205.01, 1209.06, 1027.21, 909.06, 776.99, 662.11, 553.58
Node 1 (steps 68-78): 660.91, 1109.00, 975.10, 1220.21, 1002.29, 1429.15, 709.36, 1711.49, 152.90, 2457.98, 14.74
Node 4 (steps 68-78): 370.16, 370.19, 370.21, 370.19, 370.20, 370.19, 370.20, 370.20, 370.20, 370.20, 370.20
```

最终答案：`<answer>B</answer>`（gold：B，选对）。

### 9.4 correlation，idx 480（§5 代表窗口：Node 4，67-73）

**题干**：During time steps 68-74 (1 time step = 15 minutes), which of the following best describes the influence experienced by Node 1? Options: A. Receives from Industrial Zone B. Transmits to CBD C. Receives from CBD D. Connects with Suburban Area

| node | window | 对齐 | raw time series | 模型复述 | max_abs_diff | 原文行 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 4 | 67-73 | 1-based | [411.61, 407.67, 822.89, 2101.15, 0.0, 2341.63, 345.85] | [370.15, 370.1, 370.09, 370.11, 370.11, 370.12, 370.12] | 1971.51 | `Node 4 [67-73]: 370.15, 370.10, ...` |
| 1 | 67-73 | 1-based | [691.73, 616.12, 1658.61, 1041.98, 1254.24, 1545.15, 2347.4] | [601.11, 601.94, 602.2, 602.66, 602.91, 603.21, 603.4] | 1744.0 | `Node 1 [67-73]: 601.11, 601.94, ...` |
| 0 | 67-73 | 1-based | [1022.91, 1119.11, 1371.17, 1262.46, 1421.44, 1985.19, 2534.61] | [221.26, 491.26, 709.9, 921.22, 1136.02, 1327.76, 1501.5] | 1033.11 | `Node 0 [67-73]: 221.26, 491.26, ...` |

**模型输出（摘录）**：

```
Node 0 [67-73]: 221.26, 491.26, 709.90, 921.22, 1136.02, 1327.76, 1501.50
Node 1 [67-73]: 601.11, 601.94, 602.20, 602.66, 602.91, 603.21, 603.40
Node 4 [67-73]: 370.15, 370.10, 370.09, 370.11, 370.11, 370.12, 370.12
```

最终答案：`<answer>C</answer>`（gold：C，选对）。

四例共同特征：模型在 `<think>` 中会写出较完整的 node-window 数值列表并据此推理，但多处数值与 raw time series 明显偏离；correlation 两例尽管复述偏差大，最终选择题仍答对。

## 10. 时间序列提及次数统计（broad 口径）

### 10.1 口径说明

§1–§5 的「复述对不上」沿用 **strict mismatch**：`find_bad_cases.py` 抽取 `Node k [a-b]:` / `Node k (steps a-b):` 数值列表，再与 ST-Test raw time series 做 0-based/1-based 保守对齐。

本节另用 **broad 提及** 统计模型在 `<think>` 中「把 Node 与一段数值列表绑在一起」的次数，用于观察复述行为密度，**不与 strict mismatch 窗口数直接对比**。Broad 规则（`count_ts_mentions.py`）：在 thinking 中去重计数以下 span——严格 window 行、`Node k (steps a-b):` 行内列表、`Node k [a-b]:` 行内列表、以及带足够数字的 `Node k` 数值行（排除图边 `->` 行）；同一文本区间只计一次。

数据源：`exp/sttest_full_*_6144/generated_answer.json`（3273 条 `response`）。分组标签来自 `task_level_reconstruction_mismatch.json` 的 mismatch idx 列表（与 §2 一致）。均值 = 提及次数总和 / 样本数；mismatch 样本数为 0 时均值记 N/A。

### 10.2 四任务 × 三行统计表

| 任务 | 分组 | 样本数 | 提及次数总和 | 均值 (总和/样本数) |
| --- | --- | ---: | ---: | ---: |
| forecasting | all | 280 | 1054 | 3.7643 |
| forecasting | mismatch | 141 | 602 | 4.2695 |
| forecasting | other | 139 | 452 | 3.2518 |
| correlation | all | 1592 | 5438 | 3.4158 |
| correlation | mismatch | 347 | 1649 | 4.7522 |
| correlation | other | 1245 | 3789 | 3.0434 |
| entity | all | 1194 | 2408 | 2.0168 |
| entity | mismatch | 0 | 0 | N/A |
| entity | other | 1194 | 2408 | 2.0168 |
| etiological | all | 207 | 516 | 2.4928 |
| etiological | mismatch | 0 | 0 | N/A |
| etiological | other | 207 | 516 | 2.4928 |

artifact：`mention_stats_mismatch_vs_other.json` / `.md`，脚本 `00_new_codes/scripts/mlp_encoder_focused_analysis/count_ts_mentions.py`。

### 10.3 解读

- forecasting / correlation 的 **mismatch 组 broad 均值高于 other 组**（4.27 vs 3.25；4.75 vs 3.04），说明被 strict 规则标为「复述对不上」的样本，往往也在 thinking 里更密集地写出 node-数值列表——与 §3「复述错误组任务指标未必更差」并存，更像样本选择（更爱显式列数的 response 更容易被检出 mismatch），而非「少复述就更准」。
- entity / etiological 在 strict 口径下 mismatch 数为 0，但 broad 均值仍有 2.0 / 2.5，表明模型仍会偶发写出带数字的 Node 描述；只是多数不是可对齐的 `Node k [a-b]:` 严格列表，故进不了 §2 的 mismatch 集合。
- 本表只统计提及密度，不判断数值是否正确；是否与 raw 一致仍以 §1–§5 strict 对齐为准。

