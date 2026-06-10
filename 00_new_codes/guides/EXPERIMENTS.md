# 实验产物索引（单轨登记）

> 完整机器可读登记：`00_new_codes/reports/superpowers/artifacts/experiment_registry.json`（71 条）。  
> 本页只列 **主轨道** 与 git 追踪策略；不搬产物、不删目录。

## 四档标签

| 标签 | 含义 |
| --- | --- |
| `upstream-bundled` | 作者 GitHub 打包参考输出 |
| `repro-canonical` | 本仓库主复现 run（结论优先引用） |
| `repro-archive` | 历史 repro 轨（`archive-data-keep`，只读） |
| `official-readme-path` | README 写的 `exp/$task-...` 推理路径 |

## 主轨道（必读）

| 标签 | 路径 | git 最小文件集 | 角色 |
| --- | --- | --- | --- |
| `upstream-bundled` | `exp_STReasoner-8B/reasoning_{entity,etiological,correlation,forecasting}-STReasoner-8B/` | `generated_answer.json`, `evaluation_metrics.json` | 论文 Table 对齐参考；**非** run1 替身（见报告 44） |
| `repro-canonical` | `exp/sttest_full_{task}_6144/` | 同上 | run1 full ST-Test 6144（report 39/42/43） |
| `repro-canonical` | `exp/sttest_full_{task}_6144_run2_official/` | 同上 | 三次 run 对照 run2 |
| `repro-canonical` | `exp/sttest_full_{task}_6144_run3_official/` | 同上 | 三次 run 对照 run3 |
| `repro-canonical` | `exp/sttest_full_{task}_512/` | 同上 | token budget 512 对照 |
| `official-readme-path` | `exp/reasoning_{task}-STReasoner-8B-CoT/` 等 | 同上 | README 示例/ smoke |
| `repro-archive` | `00_new_codes/repro_autodl/experiments/results/` | `*.json`, `*.jsonl` | AutoDL 阶段结果（`archive-data-keep`） |
| `repro-archive` | `00_new_codes/repro_kaggle/outputs/`, `.../stage1_results/` | 同上 | Kaggle 历史（`archive-code` + `archive-data-keep`） |

## 任务 metrics 速查（6144 full ST-Test）

| 任务 | upstream-bundled | repro-canonical run1 | 报告 |
| --- | --- | --- | --- |
| forecasting MAE | 65.59 | 68.32 | [`44`](../reports/44-exp_STReasoner-8B-MLP复述与6144对照.md) §4 |
| correlation acc | 0.8712 | 0.8317 | 同上 |
| entity acc | 0.7571 | 0.7479 | 同上 |
| etiological acc | 0.9565 | 0.9565 | 同上 |

## 远端 push checklist

每个正式 run 结束：**至少** push `generated_answer.json` + `evaluation_metrics.json`。详见 [`pipeline_map.md`](pipeline_map.md) 与 [`修改文件必读规则.md`](修改文件必读规则.md)。

## 相关文档

- 数据通路：[`pipeline_map.md`](pipeline_map.md)
- 分析入口：[`分析入口说明.md`](分析入口说明.md)
- MLP 复述：报告 [`39`](../reports/39-MLP复述四任务分析.md)、[`43`](../reports/43-run2-run3复述错误三次对比.md)、[`44`](../reports/44-exp_STReasoner-8B-MLP复述与6144对照.md)
- Artifact 溯源：[`reports/artifacts/README.md`](../reports/artifacts/README.md)
- 模型安装/卸载与调用路径：报告 [`45`](../reports/45-模型安装卸载与调用路径.md)
