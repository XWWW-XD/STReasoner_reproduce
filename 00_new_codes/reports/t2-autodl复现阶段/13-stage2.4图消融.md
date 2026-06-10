# Stage 2.4 ST-Test graph ablation 最终报告

## 结论

本轮完成了 ST-Test 全量 **w/o graph** 实验，并与已有的 **w/ graph 6144** 结果进行对照。

两组实验使用同一模型和同一评估流程：

- 模型：`base_model/STReasoner-8B`
- 推理脚本：`inference/inference_tsmllm_vllm.py`
- `max_tokens=6144`
- 评估脚本：`evaluation/evaluate.py`
- 未修改 `raw response`、`gold` 或 `timeseries`

### 结论

- **graph 结构有起作用**，总体来看，graph 对 ST-Test 的选择类空间推理任务有稳定帮助
- `提升` = 加入 graph 后相对基线的增益（准确率、spatial/edge 为 w/ − w/o；MAE/MAPE 为 w/o − w/，数值越大表示误差下降越多）。
- 加入 graph 后，三个选择类任务整体准确率从 **74.04%** 提升到 **80.69%**，提升 **6.65 个百分点**。
- 分任务看，`etiological`、`entity`、`correlation` 三类任务加入 graph 后准确率均有提升，说明 graph 对空间关系类推理确实有帮助。这也和论文 Figure 6 的结论一致：加入空间结构后，模型更容易使用空间传播、节点关系和边关系进行推理。
- forecasting 的结果不完全一致。加入 graph 后，MAE 略有上升，说明按绝对误差看没有变好；但 MAPE 下降了 **14.361 个百分点**，说明按相对误差看有所改善。
- raw response 诊断也支持 graph 起作用。相比 w/o graph，w/ graph 下模型回答中的 spatial term 和 explicit edge mention 明显更多；其中 explicit edge mention 在 w/o graph 中接近 0，加入 graph 后明显增加。

## 指标对比（基线 = w/o graph）


| task                  | samples | 基线 w/o graph               | w/ graph                   | 指标提升（相对基线）                           | 基线 spatial terms | w/ spatial terms | spatial 提升 | 基线 edge mentions | w/ edge mentions | edge 提升 |
| --------------------- | ------- | -------------------------- | -------------------------- | ------------------------------------ | ---------------- | ---------------- | ---------- | ---------------- | ---------------- | ------- |
| reasoning_etiological | 207     | 94.20%                     | 95.65%                     | accuracy **+1.45 pp**                | 20.0             | 32.12            | +12.12     | 0.06             | 2.44             | +2.38   |
| reasoning_entity      | 1194    | 65.41%                     | 74.79%                     | accuracy **+9.38 pp**                | 16.84            | 37.84            | +21.00     | 0.04             | 3.77             | +3.73   |
| reasoning_correlation | 1592    | 77.89%                     | 83.17%                     | accuracy **+5.28 pp**                | 21.64            | 41.08            | +19.44     | 0.33             | 4.75             | +4.42   |
| reasoning_forecasting | 280     | MAE 67.570 / MAPE 137.650% | MAE 68.317 / MAPE 123.289% | MAE **−0.747** / MAPE **+14.361 pp** | 12.36            | 31.19            | +18.83     | 0.0              | 3.2              | +3.20   |


说明：

- **基线** = w/o graph（删图）；**提升**一律相对该基线计算。
- 选择题 **accuracy 提升** = w/ − 基线（pp）；越大越好。
- forecasting：**MAE 提升** = 基线 MAE − w/ MAE（越大越好；本实验为负表示 w/ 的 MAE 略高）；**MAPE 提升** = 基线 MAPE − w/ MAPE（越大越好）。
- **spatial / edge 提升** = w/ − 基线（每条样例平均 mention 次数）。
- 选择类加权准确率：基线 `74.04%` → w/ `80.69%`，**总提升 +6.65 pp**。

## w/o graph 实际耗时


| task                  | samples | start               | end                 | minutes |
| --------------------- | ------- | ------------------- | ------------------- | ------- |
| reasoning_etiological | 207     | 2026-05-31T18:54:53 | 2026-05-31T18:59:18 | 4.41    |
| reasoning_entity      | 1194    | 2026-05-31T18:59:18 | 2026-05-31T19:28:54 | 29.60   |
| reasoning_correlation | 1592    | 2026-05-31T19:28:54 | 2026-05-31T20:44:16 | 75.36   |
| reasoning_forecasting | 280     | 2026-05-31T20:44:16 | 2026-05-31T20:50:34 | 6.29    |
| total inference       | 3273    | 2026-05-31T18:54:53 | 2026-05-31T20:50:34 | 115.66  |


- 后续同机同配置全量 ST-Test 单变体可按约 **2 小时** 规划。

## 显存记录

- 单张 A100-PCIE-40GB 上，w/o graph full ST-Test 巡查峰值为 `38.14GiB / 40.00GiB`（2026-05-31 19:28:07 UTC，entity/correlation 阶段附近）。
- 原因：官方 `inference/llm_utils.py` 的 `worker_vllm_ts` 使用 `gpu_memory_utilization=0.95`，**vLLM 会预留接近整卡显存**

## 产物


| 产物                       | 路径                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------ |
| w/o graph 根目录（数据集、日志、汇总） | `00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_sttest_6144/` |
| w/o graph 汇总             | `.../summary.json`                                                                   |
| w/o vs w/ 对照             | `.../comparison_with_existing_graph_baseline.json`                                   |
| w/o graph 命令记录           | `.../command_records.json`                                                           |
| w/o graph 官方推理输出         | `exp/stage2.4_graph_ablation_sttest_6144_without_graph_reasoning`_*                  |
| w/ graph 官方输出（复用）        | `exp/sttest_full_*_6144/`                                                            |
| w/ graph 汇总快照（复用）        | `00_new_codes/reports/artifacts/sttest_full_6144_summary.json`                       |


`.../` = `stage2.4_graph_ablation_sttest_6144/`。

## 判断

paper_cases 与 full ST-Test 方向一致：以 w/o graph 为基线时，加入 graph 后显式空间结构推理显著增多，选择类准确率整体提升约 6.65 pp。该结果可作为 Figure 6 相关论述的复现实验补充。