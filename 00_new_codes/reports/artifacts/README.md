# `reports/artifacts/` 溯源

> 来源以 **报告正文** 为准（写代码 session 一般在报告里自证）。  
> 与 `reports/superpowers/artifacts/` **独立**：后者为 Superpowers wave ledger，不自动同步到本目录。

## 子目录 / 文件 → 报告 → 工具

| artifacts 路径 | 主报告 | 脚本 | 说明 |
| --- | --- | --- | --- |
| `mlp_encoder_focused_analysis/boundary_*`, `cross_node_*`, `mlp_encoder_facts` | [`37-MLP编码器边界分析.md`](../37-MLP编码器边界分析.md) §8 | `scripts/mlp_encoder_focused_analysis/{inspect,find,summarize}_*.py` | 首建子目录 |
| `mlp_encoder_focused_analysis/task_level_*`, `manual_audit_*`（根下） | [`38`](../38-encoder代码细节.md)、[`39`](../39-MLP复述四任务分析.md) | 同上 | 39 为 canonical |
| `mlp_encoder_focused_analysis/run1_recheck/` | [`39`](../39-MLP复述四任务分析.md)、[`43`](../43-run2-run3复述错误三次对比.md) | 同上 | run1 回归 |
| `mlp_encoder_focused_analysis/run2/`, `run3/` | [`43`](../43-run2-run3复述错误三次对比.md) | 同上 + `--exp-suffix` | 三 run 对照 |
| `mlp_encoder_focused_analysis/upstream_bundled/` | [`44`](../44-exp_STReasoner-8B-MLP复述与6144对照.md) | 同上 + `--exp-layout bundled` | upstream vs run1 |
| `sttest_full_6144_summary.json` 等 | Stage 2.3 / 2.4 报告 | — | 小指标快照；逐条 raw 读 `exp/sttest_full_*_6144/` |
| `entity_idx838_*` | [`10.1-超长response样例.md`](../10.1-超长response样例.md) | — | 单样例 |
| `st_causal_preview.jsonl` | [`25-ST-Causal数据集调研.md`](../25-ST-Causal数据集调研.md) | 预览切片 | 非 ST-Test |
| `sttest_full_6144_response_token_*.csv` | **待补报告链接** | 可能 `tools/columns.py` | 执行时 grep 补链 |

## `superpowers/artifacts/` 边界

| 来源 | 路径 | 报告 |
| --- | --- | --- |
| wave 0–7 稳定性 | `superpowers/_analysis/compute.py` | [`superpowers/01`](../superpowers/01-三次ST-Test6144推理链路与本体的稳定性.md) 等 |
| wave 8–26 | `superpowers/_analysis/compute_*.py` | 各 wave 报告 |
| 实验登记 | `experiment_registry.json` | Superpowers 多波 |

## 今后规则

- 新分析脚本产出 json：**同一 session 报告**须列出 artifact 路径（沿用 37 §8 / 39 §1）。
- 新 session 日志命名：`NN-主题-YYYYMMDD.md`。
- **不**建 `reports/INDEX.md`；**不**合并 superpowers 子树。
