# Superpowers 工作流（本仓库）

路径 override：产出均在 [`00_new_codes/reports/superpowers/`](.)，不改 `tools/`、`inference/`、`evaluation/`、`exp/**`。

**Strict redo（2026-05-30）**：21 wave 计划 + review 见 [`_plans/README.md`](_plans/README.md)、[`_reviews/`](_reviews/)；compute 重跑 [`_plans/compute_run_log.txt`](_plans/compute_run_log.txt) **11/11 OK**；终验 `verify_reports.py` **26/26 PASS**。

## 每 Wave 循环（强制）

1. **using-superpowers** → 确认 skill 列表  
2. **brainstorming** → 探索 memo 更新  
3. **writing-plans** → [`_plans/wave*.md`](_plans/)  
4. **executing-plans** → `_analysis/compute_*.py` 或 `run_all_computes.py`  
5. **verification-before-completion** → 重跑 + json 对账  
6. **requesting-code-review** → [`_reviews/wave*-review.md`](_reviews/)  
7. **systematic-debugging** → headline 与 artifact 矛盾时先查脚本  

## 阶段

| 阶段 | Wave | 主轴 |
| --- | --- | --- |
| 一 | 0–7 | run1/2/3 6144 稳定性 |
| 二 | 8–14 | 跨实验族 artifact |
| 三 | 15–22 | 源码 trace + crosswalk |
| CPU | E2/T0-2/T0-3/25/E4 | 重 parse / flip 账本 / idx 列表 |

## 执行边界

**Superpowers 可执行**：只读 + `_analysis/*.py`（CPU）。**已完成**：阶段二/三 + E2/T0-2/T0-3/Wave25/E4 idx。

**不在此队列**：E1/E3/E4 GPU 重跑、Tier1+ → [`14-后续实验与优化计划.md`](14-后续实验与优化计划.md)。

## 防再犯

- strict vs loose 双口径  
- 已抽取 pred 勿再套 `<answer>` 正则  
- `generated_answer.json` 的 `num_tokens` = **input**  
- 后波可推翻前波 headline  

## Checklist（strict redo 已勾选）

- [x] Read 相关 skill（每 wave 见 `_plans` frontmatter）  
- [x] compute exit 0（`run_all_computes.py` 11/11）  
- [x] artifact ↔ 报告一致（`verify_reports.py` 26/26）  
- [x] 因果句带 artifact / path:line（16–19 已扩写）  
- [x] code review 留痕（`_reviews/*`）  
- [x] Plan frontmatter todo 与 `_plans/README` 对齐  

## 一键重跑

```bash
cd 00_new_codes/reports/superpowers/_analysis
python run_all_computes.py
python verify_reports.py
```
