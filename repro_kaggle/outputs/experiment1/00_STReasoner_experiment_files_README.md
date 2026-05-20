# STReasoner 阶段一实验结果文件索引说明
> 用途：放在实验结果文件夹中，作为 README/索引，帮助区分每个文件是什么、属于哪组实验、应该如何阅读。  
> 当前版本基于本文件夹内已有的 report / log / json / jsonl 文件整理；后续 SmartTest 或正式实验跑完后，可以在本文最后追加新条目。
## 0. 先看一句话结论
这些文件不是一组单一实验，而是几组阶段性小实验的产物：
- `official_pipeline_audit.md`：解释**作者官方流程**和当前 `repro_kaggle` 辅助流程哪里不同。
- `compare_single4bit_dualfp16_*`：比较 **single_gpu 4bit** 与 **dual_gpu fp16** 在 Kaggle T4×2 上是否能加载/生成。
- `single4bit_*`：早期 **single_gpu + 4bit** 的 tiny eval 结果。
- `parsefix_*`：围绕 parse 失败做的输出格式修复/生成长度对比实验。
- `*_summary.json` 适合看指标表；
- `*_predictions.jsonl` 适合看逐条输入输出；
`*_eval.log` 适合看运行日志和报错。

## 1. 文件命名规则
| 文件后缀/模式 | 含义 | 什么时候看 |
|---|---|---|
| `*_report.md` | 人类可读报告，通常已经有目标、设置、结果表、结论 | 写汇报、快速理解实验目的 |
| `*_summary.json` | 机器可读汇总指标，例如 parse 成功数、平均延迟、显存峰值 | 整理表格、核对数字 |
| `*_predictions.jsonl` | 每条样本一行，包含 raw prediction、parse 结果、latency 等 | 人工检查模型到底输出了什么 |
| `*_eval.log` | 控制台日志，记录加载、逐样本生成、parse pass/fail | 排查脚本、加载、显存、报错 |
| `*_selected_indices.json` | 被抽中的 ST-Test 样本 index | 确认不同实验是否用了同一批样本 |

## 2. 推荐阅读顺序
1. 先读 `official_pipeline_audit.md`：弄清楚当前结果为什么不能直接叫“官方复现结果”。
2. 再读 `compare_single4bit_dualfp16_report(1).md`：看 4bit 单卡和 fp16 双卡在资源层面是否可行。
3. 再读 `parse_fix_experiment_report(1).md`：看 parse 失败是否能通过加长输出或追加格式提示缓解。
4. 需要精确数字时看对应 `summary.json`。
5. 需要判断模型输出是否真的合理时看对应 `predictions.jsonl`。
6. 需要排查脚本运行细节时看对应 `eval.log`。

## 3. 文件分组索引
### A. 官方流程审计
| 文件 | 类型 | 作用 | 关键点 |
|---|---|---|---|
| `official_pipeline_audit.md` | 审计报告 | 对比作者官方 vLLM-TS 推理/评测流程和当前 `repro_kaggle` 辅助流程 | 说明当前 tiny eval 是工程验证，不是完整官方评测；重要差异包括 vLLM vs HF generate、官方 prompt suffix、chat template、parser/evaluator 口径 |

### B. 20 条 tiny eval 样本选择
| 文件 | 类型 | 作用 | 关键点 |
|---|---|---|---|
| `compare_single4bit_dualfp16_selected_indices.json` | 样本索引 | 记录 20 条 ST-Test 样本 index | 4 类任务各 5 条：correlation、entity、etiological、forecasting。后续多组实验复用了这批样本，保证可比性 |

### C. 单卡 4bit 基线实验
| 文件 | 类型 | 作用 | 关键点 |
|---|---|---|---|
| `single4bit_eval.log` | 日志 | 早期 single_gpu + 4bit 运行日志 | 20/20 generate 成功，但 parse 失败较多 |
| `single4bit_predictions.jsonl` | 逐样本输出 | 每条样本的 prediction、parse 结果、latency、显存等 | 用于人工检查 raw output 为什么 parse fail |
| `single4bit_summary.json` | 汇总指标 | single_gpu + 4bit 总结 | generate 20/20；parse 7/20；parse_fail_rate 0.65；avg_latency 约 50.27s；gpu0 max_reserved 约 6.744GiB |

### D. 单卡 4bit 与双卡 fp16 对照实验
| 文件 | 类型 | 作用 | 关键点 |
|---|---|---|---|
| `compare_single4bit_dualfp16_report(1).md` | 对照报告 | 比较 single_gpu 4bit、dual_auto fp16、dual_balanced fp16 | 三组都能加载并 generate 20/20；主要失败仍在 parse；双卡 fp16 确实发生层切分，但输出稳定性没有明显优于 4bit |

### E. parse fix / 输出格式修复实验
| 实验组 | 相关文件 | max_new_tokens | answer_format_prompt | 目的 | 结果摘要 |
|---|---|---:|---|---|---|
| 基线 | `parsefix_baseline_eval.log` / `parsefix_baseline_predictions.jsonl` / `parsefix_baseline_summary.json` | 64 | False | 原始短输出基线 | generate 20/20；parse 7/20；parse_fail_rate 0.65；avg_latency 约 50.04s |
| 加长生成 | `parsefix_longer_eval.log` / `parsefix_longer_predictions.jsonl` / `parsefix_longer_summary.json` | 256 | False | 测试增加输出 token 是否改善 parse | generate 20/20；parse 9/20；parse_fail_rate 0.55；avg_latency 约 225.80s |
| 强制答案格式 | `parsefix_forced_eval.log` / `parsefix_forced_predictions.jsonl` / `parsefix_forced_summary.json` | 256 | True | 测试追加 answer 格式提示是否改善 parse | generate 20/20；parse 2/20；parse_fail_rate 0.90；avg_latency 约 237.77s；效果变差 |
| 汇总报告 | `parse_fix_experiment_report(1).md` | — | — | 对上面三组做人工总结 | 结论：加长生成略有改善；显式 answer prompt 在这版脚本中没有改善，反而更差 |

## 4. 关键实验数字速查
| 实验文件 | 精度 | 设备策略 | max_new_tokens | answer_format_prompt | 样本数 | generate 成功 | parse 成功 | parse_fail_rate | avg_latency_sec | max_reserved_gib |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `single4bit_summary.json` | 4bit | single_gpu | — | — | 20 | 20 | 7/20 | 0.650 | 50.270 | gpu0:6.744 |
| `parsefix_baseline_summary.json` | 4bit | single_gpu | 64 | False | 20 | 20 | 7/20 | 0.650 | 50.042 | gpu0:6.744 |
| `parsefix_longer_summary.json` | 4bit | single_gpu | 256 | False | 20 | 20 | 9/20 | 0.550 | 225.799 | gpu0:6.744 |
| `parsefix_forced_summary.json` | 4bit | single_gpu | 256 | True | 20 | 20 | 2/20 | 0.900 | 237.769 | gpu0:6.744 |

## 5. 当前可以得出的结论边界
1. **资源层面**：single_gpu + 4bit 可以在 Kaggle T4 单卡上加载并完成 20 条 generate；显存峰值 reserved 约 6.744GiB。
2. **双卡 fp16 层面**：dual_auto / dual_balanced fp16 在当前辅助脚本中可加载并 generate，但该链路是 HF `device_map` 路径，不等于作者官方 vLLM-TS 流程。
3. **parse 层面**：主要失败类型是 `no_answer_tag_or_standalone_choice`，说明模型经常生成长推理文本但没有稳定输出当前 parser 需要的答案格式。
4. **输出长度层面**：从 64 提到 256 可以略微提高 parse 成功数，但延迟显著增加；单纯追加当前版本的 answer-format prompt 没有改善。
5. **正式评测边界**：这些 tiny eval / parsefix 实验适合做工程链路诊断，不能直接等同于论文 ST-Bench 官方评测结果。正式对齐应使用官方 prompt suffix、官方 generated_answer 格式和官方 evaluator。

## 6. 常见混淆说明
| 容易混淆的问题 | 正确理解 |
|---|---|
| `accuracy_overall_if_applicable` 是不是官方 accuracy？ | 不是。它是当前辅助脚本在 parse 成功样本上的 strict 判断，不是作者官方 `evaluation/evaluate.py` 的完整结果。 |
| parse 失败是不是模型 generate 失败？ | 不是。多数组显示 generate 已经成功；parse 失败通常是输出格式不符合 parser。 |
| `allocated` 和 `reserved` 能不能相加？ | 不能。`reserved` 是 PyTorch 缓存池保留量，已经包含 allocated。 |
| 4bit 成功是否说明 fp16 单卡也可以？ | 不能。4bit 是量化推理，显存需求显著低于 fp16。 |
| 双卡 fp16 成功是否说明完全复现作者推理？ | 不能。当前是 HF generate + device_map 辅助链路，作者官方是 vLLM-TS。 |
| forecasting 为什么没有 accuracy？ | 官方 forecasting 应该看 MAE/MAPE；当前选择题 parser 对 forecasting 不适用。 |

## 7. 后续新增文件建议命名
如果后面继续跑 SmartTest 或完整实验，建议按下面规则命名，避免和已有文件混淆：

```text
experiment1_smarttest_2048_records.jsonl
experiment1_smarttest_2048_summary.json
experiment1_smarttest_2048_report.md
experiment1_smarttest_4bit_single.md
experiment1_smarttest_8bit_single.md
experiment1_smarttest_fp16_single.md
experiment1_smarttest_fp16_dual.md
```

新增文件后，建议在本 README 中追加：
- 实验目的；
- 使用样本；
- 配置；
- 输出文件三件套；
- 哪些结果可用于汇报，哪些只是调试证据。

## 8. 待确认信息
下面几项需要后续人工确认后再补入索引：
1. 当前文件夹最终在项目中的路径名称；
2. `single4bit_*` 与 `parsefix_baseline_*` 是否为同一脚本/同一设置的重复运行，还是保留为两个独立基线；
3. 后续 SmartTest 2 条样例文件及四配置 2048 结果是否加入本文件夹；
4. 是否要把 `official_pipeline_audit.md` 单独放到 `stage1_docs`，还是和结果文件放在同一层。


## 分步版

### A

1. selected_indices.json
= 这 20 条样本是哪 20 条。

2. single4bit_*
= 单卡 4bit 基线：能跑，generate 成功，但 parse 失败多。

3. compare_single4bit_dualfp16_report
= 4bit 单卡 vs fp16 双卡：都能生成，但 parse 仍是主问题。

4. parsefix_*
= 尝试修 parse：加长生成和强制格式都没有彻底解决。

### B

allocated ≈ 实际张量占用

reserved ≈ PyTorch 预留池占用

看能不能放下模型时，reserved 更接近实际占坑情况

### single4bit实验结果

目的：验证 STReasoner-8B 在 Kaggle 单张 T4 上使用 4bit 量化时，是否能完成加载和推理。

主要结论：
模型加载成功，20 条样本全部 generate 成功，说明单卡 4bit 可以跑通推理链路；但 parse 失败 13/20，主要问题不是显存或生成崩溃，而是输出格式与 parser 不匹配。

优先阅读：
1. single4bit_summary.json：看整体统计；
2. single4bit_eval.log：查运行过程；
3. single4bit_predictions.jsonl：查具体样本输出。