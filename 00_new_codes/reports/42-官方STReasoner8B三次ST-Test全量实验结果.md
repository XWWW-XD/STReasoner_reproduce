# 官方 STReasoner-8B 三次 ST-Test 全量实验结果

## 0. 结论

本报告修正此前误写成 forecasting-only 的版本。这里的“ST-Test 全量”严格按 `00_new_codes/reports/10-ST-TEST数据集效果.md` 的口径：

- `reasoning_entity`：`1194` 条
- `reasoning_etiological`：`207` 条
- `reasoning_correlation`：`1592` 条
- `reasoning_forecasting`：`280` 条
- 合计：`3273` 条

现在已经有三次完整 ST-Test 6144 结果：

| run | 四任务是否完整 | 总 prediction 行数 | 说明 |
| --- | --- | ---: | --- |
| run1 | 完整 | 3273 | 原已有 `sttest_full_*_6144/` |
| run2 | 完整 | 3273 | 本轮新增 `*_run2_official/` |
| run3 | 完整 | 3273 | 本轮新增 `*_run3_official/` |

三次均使用官方链路：

- 推理：`inference/inference_tsmllm_vllm.py`
- 评估：`evaluation/evaluate.py`
- 模型：`base_model/STReasoner-8B/`
- `max_tokens=6144`
- 不修改 raw response，不补格式，不改 gold，不改官方 evaluate/parser。

## 1. 输出目录

| task | run1 | run2 | run3 |
| --- | --- | --- | --- |
| entity | `exp/sttest_full_entity_6144/` | `exp/sttest_full_entity_6144_run2_official/` | `exp/sttest_full_entity_6144_run3_official/` |
| etiological | `exp/sttest_full_etiological_6144/` | `exp/sttest_full_etiological_6144_run2_official/` | `exp/sttest_full_etiological_6144_run3_official/` |
| correlation | `exp/sttest_full_correlation_6144/` | `exp/sttest_full_correlation_6144_run2_official/` | `exp/sttest_full_correlation_6144_run3_official/` |
| forecasting | `exp/sttest_full_forecasting_6144/` | `exp/sttest_full_forecasting_6144_run2_official/` | `exp/sttest_full_forecasting_6144_run3_official/` |

每个目录中至少有：

- `generated_answer.json`
- `evaluation_metrics.json`

本轮新增的 run2/run3 目录还保留：

- `run.log`
- `evaluate.log`

## 2. 三次完整结果

### 2.1 选择类任务 accuracy

| task | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| entity | 0.747906 | 0.733668 | 0.731156 |
| etiological | 0.956522 | 0.927536 | 0.927536 |
| correlation | 0.831658 | 0.831030 | 0.822236 |

### 2.2 forecasting 指标

| run | evaluated/total | missing | MAE | MAPE | missing idx |
| --- | ---: | ---: | ---: | ---: | --- |
| run1 | 280/280 | 0 | 68.317056 | 123.289149 | [] |
| run2 | 278/280 | 2 | 64.297454 | 135.203768 | [84, 262] |
| run3 | 279/280 | 1 | 64.754811 | 133.539721 | [82] |

说明：

- run2/run3 的 forecasting 都生成了 `280` 条 prediction，idx 完整、无空 response。
- evaluate missing 是官方 parser 未抽到有效预测，不是没有生成。
- 三次 forecasting 的 MAE/MAPE 有波动；run2/run3 的 MAE 低于 run1，但 MAPE 高于 run1，说明不同 run 的误差分布不同。

## 3. 完整性核验

| run | task | prediction 行数 | expected | unique idx | empty response | evaluated | missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run1 | entity | 1194 | 1194 | 1194 | 0 | 1194 | 0 |
| run1 | etiological | 207 | 207 | 207 | 0 | 207 | 0 |
| run1 | correlation | 1592 | 1592 | 1592 | 0 | 1592 | 0 |
| run1 | forecasting | 280 | 280 | 280 | 0 | 280 | 0 |
| run2 | entity | 1194 | 1194 | 1194 | 0 | 1194 | 0 |
| run2 | etiological | 207 | 207 | 207 | 0 | 207 | 0 |
| run2 | correlation | 1592 | 1592 | 1592 | 0 | 1592 | 0 |
| run2 | forecasting | 280 | 280 | 280 | 0 | 278 | 2 |
| run3 | entity | 1194 | 1194 | 1194 | 0 | 1194 | 0 |
| run3 | etiological | 207 | 207 | 207 | 0 | 207 | 0 |
| run3 | correlation | 1592 | 1592 | 1592 | 0 | 1592 | 0 |
| run3 | forecasting | 280 | 280 | 280 | 0 | 279 | 1 |

结论：三次 ST-Test 都已经完整生成 `3273` 条 response。只有 forecasting 在 run2/run3 存在少量官方 evaluate missing。

## 4. 运行时间估计

本轮 run2 / run3 的任务运行顺序不是完全连续同一时刻开始，因为最开始误先补了 forecasting，之后再补齐其他三类任务。因此这里给两个口径：

1. **纯生成耗时**：从各 `run.log` 的 `Generating: 100%` 进度条读取。
2. **端到端估算**：纯生成耗时 + 每个任务单独加载模型 / 初始化 vLLM / evaluate 的开销。

### 4.1 纯生成耗时

| task | run2 | run3 |
| --- | ---: | ---: |
| entity | 27:30 | 25:18 |
| etiological | 2:32 | 2:36 |
| correlation | 1:04:38 | 1:03:04 |
| forecasting | 8:33 | 6:49 |
| 合计 | 1:43:13 | 1:37:47 |

### 4.2 单次完整 ST-Test 预计耗时

如果按当前方式每个任务单独启动一次 vLLM，即四任务各加载一次模型：

- run2 纯生成约 `1小时43分`。
- run3 纯生成约 `1小时38分`。
- 每个任务还要额外花约 `1分钟` 做模型加载、vLLM 初始化、graph capture 和评估收尾。
- 因此一次完整 ST-Test 四任务端到端大约是 `1小时45分-1小时50分`。

这个估计基于单张 `A100-PCIE-40GB`、官方 vLLM 链路、`max_tokens=6144`、四任务分开跑。

## 5. 和报告 39 的关系

报告 39 关注的是 `<think>` 中复述时间序列与 raw time series 是否对得上。本报告只补齐三次 ST-Test 全量实验结果，不直接重做 39 的复述错误统计。

后续如果继续分析稳定性，应该基于三次完整输出回答：

- run1 中 forecasting/correlation 的复述错误样本，在 run2/run3 是否仍然出现。
- 复述错误样本的 accuracy / MAE / MAPE 分组差异是否在三次 run 中稳定。
- run2/run3 中新的格式失败样本是否对应更长 response 或 `<answer>` 缺失。

注意：不要再把 forecasting 单任务写成 ST-Test 全量。ST-Test 全量必须包含四个任务，共 `3273` 条。

## 6. 当前断点

可以直接继续分析的文件：

```text
exp/sttest_full_entity_6144/generated_answer.json
exp/sttest_full_etiological_6144/generated_answer.json
exp/sttest_full_correlation_6144/generated_answer.json
exp/sttest_full_forecasting_6144/generated_answer.json

exp/sttest_full_entity_6144_run2_official/generated_answer.json
exp/sttest_full_etiological_6144_run2_official/generated_answer.json
exp/sttest_full_correlation_6144_run2_official/generated_answer.json
exp/sttest_full_forecasting_6144_run2_official/generated_answer.json

exp/sttest_full_entity_6144_run3_official/generated_answer.json
exp/sttest_full_etiological_6144_run3_official/generated_answer.json
exp/sttest_full_correlation_6144_run3_official/generated_answer.json
exp/sttest_full_forecasting_6144_run3_official/generated_answer.json
```

本轮队列日志：

```text
/tmp/run_sttest_remaining.master.log
```

