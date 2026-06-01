# Stage 2.4 ST-Test graph ablation 最终报告

## 结论

本轮完成了 ST-Test 全量 w/o graph 四任务实验，并复用已有 ST-Test 全量 w/ graph 6144 官方结果作为基线。实验使用同一 `base_model/STReasoner-8B`，推理入口为官方 `inference/inference_tsmllm_vllm.py`，`max_tokens=6144`，评估入口为 `evaluation/evaluate.py`，没有改 raw response、gold 或 timeseries。

- 三个选择类任务加权准确率：w/ graph `80.69%`，w/o graph `74.04%`，下降 `6.65` 个百分点。
- 分任务看：etiological 小幅下降，entity 和 correlation 明显下降；这与 Figure 6 对“空间结构促进空间推理使用”的预期一致。
- forecasting：w/o graph 的 MAE 略低，但 MAPE 更高，不能简单解读为删除 graph 有利；它与选择题整体趋势不同，需单独看误差分布。
- raw 诊断也支持 graph 有效：w/o graph 后平均 spatial term / explicit edge mention 明显下降，尤其 explicit edge mention 基本接近 0。

## 论文预期

论文 Figure 6 报告 STReasoner 经过 S-GRPO 后，回答中显式使用空间信息的比例高于 vanilla GRPO；论文正文强调这说明模型不仅提升最终分数，也把推理行为推向 spatially grounded strategies。对应本实验，保留 graph 应更容易带来空间结构推理，删除 graph 后选择题表现和显式空间推理应下降。

## 指标对比

| task | samples | w/ graph | w/o graph | delta | w/ spatial terms | w/o spatial terms | w/ edge mentions | w/o edge mentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reasoning_etiological | 207 | 95.65% | 94.20% | -1.45 pp | 32.12 | 20.0 | 2.44 | 0.06 |
| reasoning_entity | 1194 | 74.79% | 65.41% | -9.38 pp | 37.84 | 16.84 | 3.77 | 0.04 |
| reasoning_correlation | 1592 | 83.17% | 77.89% | -5.28 pp | 41.08 | 21.64 | 4.75 | 0.33 |
| reasoning_forecasting | 280 | MAE 68.317 / MAPE 123.289 | MAE 67.570 / MAPE 137.650 | MAE -0.747 / MAPE 14.361 | 31.19 | 12.36 | 3.2 | 0.0 |

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
- 复用的 w/ graph 基线：`exp/sttest_full_*_6144/`，以及 `00_new_codes/reports/artifacts/sttest_full_6144_summary.json`

## 判断

paper_cases 与 full ST-Test 方向一致：删除 graph 后，回答仍可保持格式合规，但显式空间结构推理明显减少；在全量选择任务上准确率整体下降。这个结果可以作为 Figure 6 相关论述的复现实验补充。
