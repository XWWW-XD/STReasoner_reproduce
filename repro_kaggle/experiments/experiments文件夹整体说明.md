# 定位

本文档是 `repro_kaggle/experiments/` 的总索引，用来快速定位 STReasoner 复现实验中的样本、脚本、运行结果和阶段性分析。

不需要：阅读顺序、后续建议、文件内容摘要

## 目录总览

| 路径 | 内容 | 建议用途 |
| --- | --- | --- |
| [`scripts/stage1_script/`](scripts/stage1_script/) | 实验一运行脚本、旧脚本和日志 | 复跑或核对运行参数 |
| [`stage1_subsets/`](stage1_subsets/) | 实验样本子集，包括 tiny20、paper cases、stress case、SmartTest | 核对分母、任务类型和样本来源 |
| [`stage1_results/`](stage1_results/) | 模型运行结果、预测文件、summary、run log、official eval 输出 | 读最终机器结果和失败原因 |
| [`stage1_docs/`](stage1_docs/) | 实验设计、结果分析、诊断记录、代码阅读笔记 | 读人类可解释结论 |

## 实验一：不同精度推理资源测试

核心模型：`Time-HD-Anonymous/STReasoner-8B`

旧版脚本与运行日志：

- 旧版脚本：[`scripts/stage1_script/run_experiment1_precision_resource_old.py`](scripts/stage1_script/run_experiment1_precision_resource_old.py)
- 旧版运行日志：[`scripts/stage1_script/run_experiment1_precision_resource2.log`](scripts/stage1_script/run_experiment1_precision_resource2.log)

新版脚本：

- 新版脚本：[`scripts/stage1_script/run_experiment1_new_version.py`](scripts/stage1_script/run_experiment1_new_version.py)
- Run Layer：记录模型加载、输入构造、generate/decode、资源占用。
- Strict Diagnostic Layer：检查 `<answer>...</answer>` 是否满足调试用机器格式。
- Official Eval Layer：生成兼容 `evaluation/evaluate_qa.py` 的评测文件和指标。

## 实验结果

索引到实验结果的文档、具体输出等相关文档。
但不写文档具体内容，最多20个字之内概括文档功能
这里你需要重写

## 样本索引

实验样本主要放在 [`stage1_subsets/exp1_resource_tiny20/`](stage1_subsets/exp1_resource_tiny20/)。

| 文件 | 说明 |
| --- | --- |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl) | tiny20 主测试，共 20 条 |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/forecasting_5.jsonl`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/forecasting_5.jsonl) | forecasting 任务 5 条 |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/entity_5.jsonl`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/entity_5.jsonl) | entity 任务 5 条 |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/etiological_5.jsonl`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/etiological_5.jsonl) | etiological 任务 5 条 |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/correlation_5.jsonl`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/correlation_5.jsonl) | correlation 任务 5 条 |
| [`stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/manifest.json`](stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/manifest.json) | tiny20 抽样 manifest |
| [`stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl`](stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl) | 论文样例中可匹配回 ST-Test 的 4 条 |
| [`stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_manifest.json`](stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_manifest.json) | 论文样例匹配记录 |
| [`stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl`](stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl) | 资源压力测试样例 1 条 |
| [`stage1_subsets/exp1_resource_tiny20/stress_case/stress_manifest.json`](stage1_subsets/exp1_resource_tiny20/stress_case/stress_manifest.json) | stress case 选择记录 |
| [`stage1_subsets/exp1_small_test_2/smart_test/SmartTest.jsonl`](stage1_subsets/exp1_small_test_2/smart_test/SmartTest.jsonl) | 另一个小样本实验入口 |

注意：旧版 `summary.json` 的统计口径包含 tiny20 / paper / stress；新版脚本当前文档口径更偏向 SmartTest 小样本。读结果时要先确认对应运行到底使用了哪个数据文件。

#### 样本构造文档

- [`stage1_docs/exp1_resource_tiny20/README.md`](stage1_docs/exp1_resource_tiny20/README.md)：tiny20 / paper cases / stress case 总说明。
- [`stage1_docs/exp1_resource_tiny20/st_test_tiny20_seed20260519/README.md`](stage1_docs/exp1_resource_tiny20/st_test_tiny20_seed20260519/README.md)：tiny20 主测试说明。
- [`stage1_docs/exp1_resource_tiny20/st_test_tiny20_seed20260519/field_inspection.md`](stage1_docs/exp1_resource_tiny20/st_test_tiny20_seed20260519/field_inspection.md)：字段检查记录。
- [`stage1_docs/exp1_resource_tiny20/paper_cases/matching_notes.md`](stage1_docs/exp1_resource_tiny20/paper_cases/matching_notes.md)：论文样例匹配说明。
- [`stage1_docs/exp1_resource_tiny20/stress_case/README.md`](stage1_docs/exp1_resource_tiny20/stress_case/README.md)：stress case 说明。
- [`stage1_docs/exp1_resource_tiny20/stress_case/selection_notes.md`](stage1_docs/exp1_resource_tiny20/stress_case/selection_notes.md)：stress case 选择依据。

## 结果文件索引

### 4bit 单卡

- 汇总：[`stage1_results/experiment1_precision_resource/4bit_single/summary.json`](stage1_results/experiment1_precision_resource/4bit_single/summary.json)
- 日志：
  - [`run.log`](stage1_results/experiment1_precision_resource/4bit_single/run.log)
  - [`run2.log`](stage1_results/experiment1_precision_resource/4bit_single/run2.log)
  - [`run_new.log`](stage1_results/experiment1_precision_resource/4bit_single/run_new.log)
- 预测：
  - [`main_predictions.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/main_predictions.jsonl)
  - [`main_predictions2.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/main_predictions2.jsonl)
  - [`main_predictions_new.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/main_predictions_new.jsonl)
  - [`paper_predictions.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/paper_predictions.jsonl)
  - [`paper_predictions2.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/paper_predictions2.jsonl)
  - [`stress_predictions.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/stress_predictions.jsonl)
  - [`stress_predictions2.jsonl`](stage1_results/experiment1_precision_resource/4bit_single/stress_predictions2.jsonl)
- smoke / official eval：
  - [`smoke_nonforecasting_4bit_2048/`](stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048/)
  - [`smoke_nonforecasting_4bit_2048/official_eval/official_metrics.json`](stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048/official_eval/official_metrics.json)
  - [`smoke_nonforecasting_4bit_2048_rerun_20260519_182802/`](stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182802/)
  - [`smoke_nonforecasting_4bit_2048_rerun_20260519_182957/`](stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182957/)
- 分析文档：
  - [`stage1_docs/experiment1_4bit_single.md`](stage1_docs/experiment1_4bit_single.md)
  - [`stage1_docs/experiment1_4bit_output_anomaly_diagnostic.md`](stage1_docs/experiment1_4bit_output_anomaly_diagnostic.md)
  - [`stage1_docs/experiment1_4bit_main20_issue_analysis_old.md`](stage1_docs/experiment1_4bit_main20_issue_analysis_old.md)
  - [`stage1_docs/experiment1_smoke_nonforecasting_4bit_2048_report.md`](stage1_docs/experiment1_smoke_nonforecasting_4bit_2048_report.md)

### 8bit 单卡

- 汇总：[`stage1_results/experiment1_precision_resource/8bit_single/summary.json`](stage1_results/experiment1_precision_resource/8bit_single/summary.json)
- 日志：[`stage1_results/experiment1_precision_resource/8bit_single/run.log`](stage1_results/experiment1_precision_resource/8bit_single/run.log)
- 预测：
  - [`main_predictions.jsonl`](stage1_results/experiment1_precision_resource/8bit_single/main_predictions.jsonl)
  - [`paper_predictions.jsonl`](stage1_results/experiment1_precision_resource/8bit_single/paper_predictions.jsonl)
  - [`stress_predictions.jsonl`](stage1_results/experiment1_precision_resource/8bit_single/stress_predictions.jsonl)
- smoke 诊断：
  - [`smoke_nonforecasting_8bit_2048/`](stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048/)
  - [`smoke_nonforecasting_8bit_2048_attempt2/`](stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048_attempt2/)
  - [`smoke_nonforecasting_8bit_2048_attempt3_gpu1/`](stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048_attempt3_gpu1/)
- 分析文档：[`stage1_docs/experiment1_8bit_single.md`](stage1_docs/experiment1_8bit_single.md)

### fp16 单卡

- 汇总：[`stage1_results/experiment1_precision_resource/fp16_single/summary.json`](stage1_results/experiment1_precision_resource/fp16_single/summary.json)
- 日志：[`stage1_results/experiment1_precision_resource/fp16_single/run.log`](stage1_results/experiment1_precision_resource/fp16_single/run.log)
- 预测：
  - [`main_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_single/main_predictions.jsonl)
  - [`paper_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_single/paper_predictions.jsonl)
  - [`stress_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_single/stress_predictions.jsonl)
- 分析文档：[`stage1_docs/experiment1_fp16_single.md`](stage1_docs/experiment1_fp16_single.md)

### fp16 双卡

- 汇总：[`stage1_results/experiment1_precision_resource/fp16_dual/summary.json`](stage1_results/experiment1_precision_resource/fp16_dual/summary.json)
- 日志：[`stage1_results/experiment1_precision_resource/fp16_dual/run.log`](stage1_results/experiment1_precision_resource/fp16_dual/run.log)
- 预测：
  - [`main_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_dual/main_predictions.jsonl)
  - [`paper_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_dual/paper_predictions.jsonl)
  - [`stress_predictions.jsonl`](stage1_results/experiment1_precision_resource/fp16_dual/stress_predictions.jsonl)
- 分析文档：[`stage1_docs/experiment1_fp16_dual.md`](stage1_docs/experiment1_fp16_dual.md)

## outputs 文件索引

`outputs` 是现场运行产物和诊断缓存，`stage1_results` 是整理后的实验一结果归档。

| 子目录 | 作用 |
| --- | --- |
| [`../outputs/early_smoke_tests/`](../outputs/early_smoke_tests/) | 早期冒烟测试脚本结果 |
| [`../outputs/experiment1/`](../outputs/experiment1/) | 实验一相关辅助结果 |

| 文件 | 作用 |
| --- | --- |
| [`../outputs/early_smoke_tests/stbench_inspect.log`](../outputs/early_smoke_tests/stbench_inspect.log) | ST-Bench 各 subset 加载与字段检查日志 |
| [`../outputs/early_smoke_tests/load_streasoner_smoke.log`](../outputs/early_smoke_tests/load_streasoner_smoke.log) | STReasoner-8B 加载和最小生成 smoke test |
| [`../outputs/early_smoke_tests/run_one_sttest_sample.log`](../outputs/early_smoke_tests/run_one_sttest_sample.log) | 单条 ST-Test 样本推理诊断日志 |
| [`../outputs/early_smoke_tests/one_sttest_prediction.json`](../outputs/early_smoke_tests/one_sttest_prediction.json) | 单条样本的输入、目标、预测和错误栈 |
| [`../outputs/early_smoke_tests/sttest_tiny_predictions.jsonl`](../outputs/early_smoke_tests/sttest_tiny_predictions.jsonl) | 早期 tiny20 推理逐样本预测 |
| [`../outputs/early_smoke_tests/sttest_tiny_summary.json`](../outputs/early_smoke_tests/sttest_tiny_summary.json) | 早期 tiny20 汇总指标 |
| [`../outputs/early_smoke_tests/sttest_tiny_eval.log`](../outputs/early_smoke_tests/sttest_tiny_eval.log) | 早期 tiny20 评测运行日志 |
| [`../outputs/experiment1/single4bit_predictions.jsonl`](../outputs/experiment1/single4bit_predictions.jsonl) | 单卡 4bit 对照实验逐样本预测 |
| [`../outputs/experiment1/single4bit_summary.json`](../outputs/experiment1/single4bit_summary.json) | 单卡 4bit 对照实验汇总指标 |
| [`../outputs/experiment1/single4bit_eval.log`](../outputs/experiment1/single4bit_eval.log) | 单卡 4bit 对照实验运行日志 |
| [`../outputs/experiment1/dualfp16_auto_predictions.jsonl`](../outputs/experiment1/dualfp16_auto_predictions.jsonl) | 双卡 FP16 auto device map 逐样本预测 |
| [`../outputs/experiment1/dualfp16_auto_summary.json`](../outputs/experiment1/dualfp16_auto_summary.json) | 双卡 FP16 auto device map 汇总指标 |
| [`../outputs/experiment1/dualfp16_auto_eval.log`](../outputs/experiment1/dualfp16_auto_eval.log) | 双卡 FP16 auto device map 运行日志 |
| [`../outputs/experiment1/dualfp16_balanced_predictions.jsonl`](../outputs/experiment1/dualfp16_balanced_predictions.jsonl) | 双卡 FP16 balanced device map 逐样本预测 |
| [`../outputs/experiment1/dualfp16_balanced_summary.json`](../outputs/experiment1/dualfp16_balanced_summary.json) | 双卡 FP16 balanced device map 汇总指标 |
| [`../outputs/experiment1/dualfp16_balanced_eval.log`](../outputs/experiment1/dualfp16_balanced_eval.log) | 双卡 FP16 balanced device map 运行日志 |
| [`../outputs/experiment1/compare_single4bit_dualfp16_selected_indices.json`](../outputs/experiment1/compare_single4bit_dualfp16_selected_indices.json) | 4bit/双卡对照实验复用的 20 条样本索引 |
| [`../outputs/experiment1/compare_single4bit_dualfp16_report.md`](../outputs/experiment1/compare_single4bit_dualfp16_report.md) | 单卡 4bit 与双卡 FP16 对照报告 |
| [`../outputs/experiment1/parsefix_baseline_predictions.jsonl`](../outputs/experiment1/parsefix_baseline_predictions.jsonl) | 输出格式修复基线组逐样本预测 |
| [`../outputs/experiment1/parsefix_baseline_summary.json`](../outputs/experiment1/parsefix_baseline_summary.json) | 输出格式修复基线组汇总指标 |
| [`../outputs/experiment1/parsefix_baseline_eval.log`](../outputs/experiment1/parsefix_baseline_eval.log) | 输出格式修复基线组运行日志 |
| [`../outputs/experiment1/parsefix_longer_predictions.jsonl`](../outputs/experiment1/parsefix_longer_predictions.jsonl) | 加长生成组逐样本预测 |
| [`../outputs/experiment1/parsefix_longer_summary.json`](../outputs/experiment1/parsefix_longer_summary.json) | 加长生成组汇总指标 |
| [`../outputs/experiment1/parsefix_longer_eval.log`](../outputs/experiment1/parsefix_longer_eval.log) | 加长生成组运行日志 |
| [`../outputs/experiment1/parsefix_forced_predictions.jsonl`](../outputs/experiment1/parsefix_forced_predictions.jsonl) | 强制答案格式组逐样本预测 |
| [`../outputs/experiment1/parsefix_forced_summary.json`](../outputs/experiment1/parsefix_forced_summary.json) | 强制答案格式组汇总指标 |
| [`../outputs/experiment1/parsefix_forced_eval.log`](../outputs/experiment1/parsefix_forced_eval.log) | 强制答案格式组运行日志 |
| [`../outputs/experiment1/parse_fix_experiment_report.md`](../outputs/experiment1/parse_fix_experiment_report.md) | 输出格式修复实验报告 |
| [`../outputs/experiment1/official_pipeline_audit.md`](../outputs/experiment1/official_pipeline_audit.md) | 作者官方推理/评测流程审计 |


## 文档索引

### 实验设计与总结

- [`stage1_docs/experiment_summary_2.md`](stage1_docs/experiment_summary_2.md)：当前实验一新版总览。
- [`stage1_docs/experiment_summary_old.md`](stage1_docs/experiment_summary_old.md)：旧版总结，适合追溯早期口径。
- [`stage1_docs/experiment1_session_note_20260519.md`](stage1_docs/experiment1_session_note_20260519.md)：2026-05-19 当天现场记录，包括未完成 / 中断任务。



### 代码阅读笔记

这里等其他全部重构完了再写
