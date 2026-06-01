# Stage 2.4 paper_cases graph ablation 报告

## 结论

本次按 Stage 2.4 要求完成 paper_cases 4 条样例的 paired 对比：同一 STReasoner-8B、官方 vLLM 推理、`max_tokens=6144`、`inference/prompt.json` 后缀、tag-first evaluate。

- w/ graph：4/4 生成，选择题 3/3，forecasting 指标见下表。
- w/o graph：4/4 生成，选择题 3/3，forecasting 指标见下表。
- 本实验未改 raw response、未改 gold、未改 timeseries；w/o graph 仅使用 EasyR1 同款正则删除 `Graph Structure:...` 到 `please analyze` 前。

## 论文预期

论文 Figure 6 衡量的是“显式使用空间信息的回答比例”。论文 5.4 写到：S-GRPO 后模型在各任务中比 vanilla GRPO 有更高的 spatial reasoning usage ratio，说明它不仅提高最终分数，也把推理行为推向 spatially grounded strategies。
对应到本次输入消融，预期是：保留 graph 时，回答中应更容易出现 graph/node/edge/path/upstream/downstream 等空间结构推理；删除 graph 后，这类显式空间推理会减少，且部分任务性能可能下降。注意：本次没有调用 GPT-5.2 judge 复现 Figure 6 的人工/模型判别比例，只记录 tag-first evaluate 与 raw 诊断。

## 实验配置

| 项目 | 值 |
|---|---|
| 模型 | `base_model/STReasoner-8B` |
| 推理入口 | `inference/inference_tsmllm_vllm.py` |
| 评估入口 | `evaluation/evaluate.py` |
| max_tokens | `6144` |
| temperature | `0.2` |
| prompt 后缀 | `inference/prompt.json` 由官方推理脚本追加 |
| remove graph 正则 | `Graph Structure:.*?(?=please analyze)` |
| 输出目录 | `00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144` |

## 汇总指标

| variant | generated | exact answer tag | empty | reach 6144 | choice | avg input tokens | avg response tokens | avg spatial term mentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with_graph | 4/4 | 4/4 | 0 | 0 | 3/3 | 400.000000 | 711.500000 | 38.000000 |
| without_graph | 4/4 | 4/4 | 0 | 0 | 3/3 | 343.000000 | 620.750000 | 19.250000 |

## 显存记录

本轮在单张 A100-PCIE-40GB 上运行。巡查时观察到最高显存占用为 `38989MiB / 40960MiB`（2026-05-31 18:30:25 UTC，with_graph correlation 推理中）。这不是 paper_cases 单样例本身需要 39GB，而是官方 `inference/llm_utils.py` 的 `worker_vllm_ts` 使用 `gpu_memory_utilization=0.95`，vLLM 会为模型、运行时和 KV cache 预留接近整卡的显存。后续 full ST-Test 仍应按“单进程接近占满 40GB A100”来规划，不要并行启动多个 STReasoner-8B vLLM 进程。

## 每条样例结果

| variant | task | sample_id | gold | parsed | metric | input tokens | response tokens | spatial terms | tags |
|---|---|---|---|---|---|---:|---:|---:|---|
| with_graph | reasoning_etiological | `paper_appendix_h_table6_etiological_line118` | `<answer>D</answer>` | `D` | accuracy=1.000000 | 293 | 783 | 36 | 1/1 |
| without_graph | reasoning_etiological | `paper_appendix_h_table6_etiological_line118` | `<answer>D</answer>` | `D` | accuracy=1.000000 | 258 | 635 | 23 | 1/1 |
| with_graph | reasoning_entity | `paper_appendix_h_table7_entity_line982` | `<answer>C</answer>` | `C` | accuracy=1.000000 | 396 | 636 | 42 | 1/1 |
| without_graph | reasoning_entity | `paper_appendix_h_table7_entity_line982` | `<answer>C</answer>` | `C` | accuracy=1.000000 | 321 | 576 | 20 | 1/1 |
| with_graph | reasoning_correlation | `paper_appendix_h_table8_correlation_line547` | `<answer>D</answer>` | `D` | accuracy=1.000000 | 729 | 675 | 40 | 1/1 |
| without_graph | reasoning_correlation | `paper_appendix_h_table8_correlation_line547` | `<answer>D</answer>` | `D` | accuracy=1.000000 | 638 | 632 | 25 | 1/1 |
| with_graph | reasoning_forecasting | `paper_appendix_h_table9_forecasting_line9` | `[19.86, 19.97, 20.05]` | `[19.85, 19.83, 19.84]` | MAE=0.120000, MAPE=0.599595 | 182 | 752 | 34 | 1/1 |
| without_graph | reasoning_forecasting | `paper_appendix_h_table9_forecasting_line9` | `[19.86, 19.97, 20.05]` | `[19.88, 19.90, 19.92]` | MAE=0.073333, MAPE=0.366537 | 155 | 640 | 9 | 1/1 |

## 产物

- 数据与汇总：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144/`
- manifest：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144/datasets/manifest.json`
- paired 结果：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144/paired_results.jsonl`
- summary：`00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144/summary.json`
- 官方推理输出：`exp/stage2.4_graph_ablation_paper_cases_6144_*`

## 过程记录

- 18:12：确认 A100 空闲，规则要求使用根目录官方 `inference/`、`evaluation/`，不再把旧 Stage 2.2 runner 当主流程。
- 18:19：生成 8 个 paired JSONL：4 个 w/ graph、4 个 w/o graph；w/o graph 只删除 `Graph Structure:...` 到 `please analyze` 前，timeseries/gold 不变。
- 18:19-18:36：逐条调用官方 `inference/inference_tsmllm_vllm.py`，每条随后调用 `evaluation/evaluate.py`；共 8 次 inference、8 次 evaluate，全部 returncode=0。
- 18:30：巡查显存，观察到 `38989MiB / 40960MiB`，判断为 vLLM `gpu_memory_utilization=0.95` 预留策略导致。
- 18:36：生成 `summary.json`、`paired_results.jsonl` 和本报告；未启动 ST-Test。

## 下一步

按用户要求，本轮只完成 paper_cases 4 条 paired 对比；是否进入完整 ST-Test graph ablation，等你看完本报告效果后再继续。
