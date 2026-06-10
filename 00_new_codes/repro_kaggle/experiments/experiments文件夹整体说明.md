# 定位

本文档是 `repro_kaggle/experiments/` 的总索引，用来快速定位 STReasoner 复现实验中的脚本、历史文档与（已归档的）运行结果。

**现行小样本数据**不在本目录：paper_cases 真源为 `00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl`（4 条，仍在使用）。

**归档策略（2026-06，策略一）**：Kaggle 阶段 **脚本与 json/jsonl 已删**；人类可读结论保留在 `stage1_docs/` 与 `outputs/experiment1/*.md`。`stage1_subsets`、`stage1_results`、`00_smoke_test_scripts` 不再入库。

## 目录总览

| 路径 | 内容 | 建议用途 |
| --- | --- | --- |
| [`stage1_docs/`](stage1_docs/) | 实验设计、tiny20 说明、结果分析、代码阅读笔记 | **唯一** Kaggle 历史结论入口 |
| [`../outputs/experiment1/`](../outputs/experiment1/) | 实验一辅助 **markdown** 报告 | 管线审计与 parsefix 文字记录 |

## 现行复现入口（AutoDL）

| 数据 | 路径 |
| --- | --- |
| paper_cases（4 条） | `00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl` |
| Stage2.2 运行脚本 | `00_new_codes/repro_autodl/experiments/scripts/stage2_2_run_paper_cases.py` |
| Stage2.4 graph 消融 | `00_new_codes/repro_autodl/experiments/scripts/stage2_4_graph_ablation_paper_cases.py` |
| 全量 ST-Test | `data/ST-Bench/ST-Test/` + `inference/inference_tsmllm_vllm.py` |

端到端路线图见 [`00_new_codes/guides/pipeline_map.md`](../../guides/pipeline_map.md)。

## 实验一（历史，已归档）

曾用 Kaggle 比较 fp16 / 8bit / 4bit 资源与 SmartTest 小样本；脚本与 `stage1_results/experiment1_precision_resource/` 已删除。结论与样例分析仍见：

- [`stage1_docs/experiment_summary_2.md`](stage1_docs/experiment_summary_2.md)
- [`stage1_docs/experiment_summary_old.md`](stage1_docs/experiment_summary_old.md)
- 各精度分析：`stage1_docs/00_experiment1_4bit_single.md` 等

## outputs 文件索引

仅保留 `outputs/experiment1/` 下 markdown（如 `official_pipeline_audit.md`、`parse_fix_experiment_report.md`）。`early_smoke_tests/` 与全部 json/jsonl 已删除。

## 文档索引

### 实验设计与总结

- [`stage1_docs/experiment_summary_2.md`](stage1_docs/experiment_summary_2.md)：实验一新版总览（历史）
- [`stage1_docs/experiment_summary_old.md`](stage1_docs/experiment_summary_old.md)：旧版总结
- [`stage1_docs/experiment1_session_note_20260519.md`](stage1_docs/experiment1_session_note_20260519.md)：2026-05-19 现场记录

### 代码阅读笔记

- [`stage1_docs/streasoner_code_reading/`](stage1_docs/streasoner_code_reading/)：官方 pipeline 与模块导读
