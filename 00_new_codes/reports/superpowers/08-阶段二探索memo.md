# 阶段二探索 memo（Wave 0 + 8–14）

**Strict redo**: 2026-05-30 — 每 wave 见 [`_plans/README.md`](_plans/README.md) + [`_reviews/`](_reviews/)。

**范围**：在阶段一（6144 三 run strict 稳定性）之上，只读 `exp/` 与 `repro_*` 既有产物。**不改** `tools/`、`inference/`、`evaluation/`。

**工作流**：[`SUPERPOWERS-WORKFLOW.md`](SUPERPOWERS-WORKFLOW.md)（skill 循环已逐 wave 执行并留痕）。

---

## Wave 映射（与 plan 一致）

| Wave | 报告 | 脚本 / artifact |
| --- | --- | --- |
| 0 | WORKFLOW + 本 memo | — |
| 8 | 本 memo §缺口 | `compute_registry.py` → `experiment_registry.json` (71) |
| 9 | [`09-512与6144同题对照.md`](09-512与6144同题对照.md) | `token_budget_512_vs_6144.json` |
| 10 | [`10-graph消融样本级.md`](10-graph消融样本级.md) | `graph_ablation_sample_ledger.json` |
| 11 | [`11-etiological与paper_cases.md`](11-etiological与paper_cases.md) | `etiological_and_paper_cases.json` |
| 12 | [`12-Kaggle精度链路审计.md`](12-Kaggle精度链路审计.md) | `kaggle_precision_audit.json` |
| 13 | [`08-阶段二总览.md`](08-阶段二总览.md) | 汇总 + `03` §6 |
| 14 | [`14-后续实验与优化计划.md`](14-后续实验与优化计划.md) | Tier0–3 roadmap |

---

## 与 reports/ 主报告的缺口（Wave8 brainstorming 验证）

| 缺口 | 本阶段补全 |
| --- | --- |
| 512 vs 6144 同 idx | wave9：flip 163/223；512 corr acc 更高需 T0-2 分解 |
| Graph 仅 aggregate | wave10：rescued 161/226/9 样本级 |
| Etiological 一笔带过 | wave11：27 flip / 13%；strict mismatch≈0 |
| Paper cases 多 arm | wave11b：4 题 baseline/graph± 已填（`etiological_and_paper_cases.json`） |
| Kaggle 不可比 official | wave12：4×20，0 tag，pipeline≠vLLM |

---

## 阶段二 headline（artifact 对账 2026-05-30 verify PASS）

1. **512 vs 6144**：correlation / entity 同题 flip **163 / 223**；6144 无 tag 率约为 512 的 2×（correlation 2.95% vs 1.44%）。
2. **Graph w/o reasoning**：correlation **+161 / −77**；entity **+226 / −114**；etiological **+9 / −6**。
3. **Etiological 三 run**：207 题 **27 flip**（13.0%）；thinking median **2700** chars。
4. **Paper cases**：4 题 × baseline + graph with/without — 见 artifact `cases[]`（forecasting 有 MAE，choice 四题 gold 对齐）。
5. **Kaggle**：4 配置各 20 条、**0** `<answer>` tag — **禁止**与 official 6144 acc 硬比。

---

## 终止与交接

阶段二 **只读挖掘完成**（CPU 项 E2/T0-2/T0-3/Wave25 见 `22`–`25`）。Tier1+ GPU 见 [`14`](14-后续实验与优化计划.md)。阶段三见 [`15-阶段三探索memo.md`](15-阶段三探索memo.md)。
