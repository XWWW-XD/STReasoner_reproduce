# Stage 2.4 ST-Test graph ablation 最终报告

## 结论

本轮完成了 ST-Test 全量 w/o graph 四任务实验，并复用已有 ST-Test 全量 w/ graph 6144 官方结果作对照。实验使用同一 `base_model/STReasoner-8B`，推理入口为官方 `inference/inference_tsmllm_vllm.py`，`max_tokens=6144`，评估入口为 `evaluation/evaluate.py`，没有改 raw response、gold 或 timeseries。

**指标对比以 w/o graph（删图）为原基线**，`提升` = 加入 graph 后相对基线的增益（准确率、spatial/edge 为 w/ − w/o；MAE/MAPE 为 w/o − w/，数值越大表示误差下降越多）。

- 三个选择类任务加权准确率：基线 w/o `74.04%` → w/ graph `80.69%`，**提升 `+6.65` 个百分点**。
- 分任务看：etiological、entity、correlation 加入 graph 后准确率均有提升；与 Figure 6「空间结构促进空间推理使用」一致。
- forecasting：加入 graph 后 MAE 略升（`+0.747`），但 MAPE **下降 `14.361` 个百分点**（相对更好）；不宜只看 MAE 单项。
- raw 诊断：相对 w/o 基线，w/ graph 下 spatial term / explicit edge mention 平均次数明显升高，explicit edge mention 在 w/o 侧接近 0。

## 论文预期

论文 Figure 6 报告 STReasoner 经过 S-GRPO 后，回答中显式使用空间信息的比例高于 vanilla GRPO；论文正文强调这说明模型不仅提升最终分数，也把推理行为推向 spatially grounded strategies。对应本实验，相对 w/o 基线，保留 graph 应带来更高的选择题准确率与更显式的空间结构推理。

## 指标对比（基线 = w/o graph）

| task | samples | 基线 w/o graph | w/ graph | 指标提升（相对基线） | 基线 spatial terms | w/ spatial terms | spatial 提升 | 基线 edge mentions | w/ edge mentions | edge 提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reasoning_etiological | 207 | 94.20% | 95.65% | accuracy **+1.45 pp** | 20.0 | 32.12 | +12.12 | 0.06 | 2.44 | +2.38 |
| reasoning_entity | 1194 | 65.41% | 74.79% | accuracy **+9.38 pp** | 16.84 | 37.84 | +21.00 | 0.04 | 3.77 | +3.73 |
| reasoning_correlation | 1592 | 77.89% | 83.17% | accuracy **+5.28 pp** | 21.64 | 41.08 | +19.44 | 0.33 | 4.75 | +4.42 |
| reasoning_forecasting | 280 | MAE 67.570 / MAPE 137.650% | MAE 68.317 / MAPE 123.289% | MAE **−0.747** / MAPE **+14.361 pp** | 12.36 | 31.19 | +18.83 | 0.0 | 3.2 | +3.20 |

说明：

- **基线** = w/o graph（删图）；**提升**一律相对该基线计算。
- 选择题 **accuracy 提升** = w/ − 基线（pp）；越大越好。
- forecasting：**MAE 提升** = 基线 MAE − w/ MAE（越大越好；本实验为负表示 w/ 的 MAE 略高）；**MAPE 提升** = 基线 MAPE − w/ MAPE（越大越好）。
- **spatial / edge 提升** = w/ − 基线（每条样例平均 mention 次数）。
- 选择类加权准确率：基线 `74.04%` → w/ `80.69%`，**总提升 +6.65 pp**。

## w/o graph 实际耗时

| task | samples | start | end | minutes |
|---|---:|---|---|---:|
| reasoning_etiological | 207 | 2026-05-31T18:54:53 | 2026-05-31T18:59:18 | 4.41 |
| reasoning_entity | 1194 | 2026-05-31T18:59:18 | 2026-05-31T19:28:54 | 29.60 |
| reasoning_correlation | 1592 | 2026-05-31T19:28:54 | 2026-05-31T20:44:16 | 75.36 |
| reasoning_forecasting | 280 | 2026-05-31T20:44:16 | 2026-05-31T20:50:34 | 6.29 |
| total inference | 3273 | 2026-05-31T18:54:53 | 2026-05-31T20:50:34 | 115.66 |

前期估算随样本集动态修正：etiological 结束后估计 entity 约 30-40 分钟，实际 29.60 分钟；correlation 中途估计 45-75 分钟，实际 75.36 分钟；forecasting 估计 10-20 分钟，实际 6.29 分钟。后续同机同配置全量 ST-Test 单变体可按约 2 小时规划，correlation 是最大耗时项。

## 显存记录

单张 A100-PCIE-40GB 上，w/o graph full ST-Test 巡查峰值为 `39053MiB / 40960MiB`（2026-05-31 19:28:07 UTC，entity/correlation 阶段附近）。原因与 paper_cases 一致：官方 `inference/llm_utils.py` 的 `worker_vllm_ts` 使用 `gpu_memory_utilization=0.95`，vLLM 会预留接近整卡显存。不要在同一张 40GB A100 上并行启动多个 STReasoner-8B vLLM 进程。

## 产物

- w/o graph 数据、日志、summary：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_sttest_6144/`
- 对比汇总：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_sttest_6144/comparison_with_existing_graph_baseline.json`
- w/o graph 官方输出：`exp/stage2.4_graph_ablation_sttest_6144_without_graph_reasoning_*`
- 复用的 w/ graph 对照：`exp/sttest_full_*_6144/`，以及 `00_new_codes/reports/artifacts/sttest_full_6144_summary.json`

## 判断

paper_cases 与 full ST-Test 方向一致：以 w/o graph 为基线时，加入 graph 后显式空间结构推理显著增多，选择类准确率整体提升约 6.65 pp。该结果可作为 Figure 6 相关论述的复现实验补充。
