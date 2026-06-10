# Batch script audit（requesting-code-review）

**Scope**: `_analysis/compute_*.py` phase2/3/CPU（11 脚本）  
**Verify**: `run_all_computes.py` 11/11 OK；`verify_reports.py` 26/26 PASS

## Strengths

- 统一 `ROOT = Path(__file__).resolve().parents[4]` 指向 repo 根 — 路径一致  
- 只读 `exp/` / 源码，无写回 exp  
- `verify_reports.py` 集中对账 — 可 falsify headline  
- strict/loose 在 E2/T0-2 分离清晰  

## Findings

| 级别 | 项 | 说明 |
| --- | --- | --- |
| Important | `compute_code_phase3.py` | trace 为 regex 扫描，**非** AST/调用图 — 报告 16–19 已用人工 path:line 互证；trace json 仅索引 |
| Important | `compute_eti_paper.py` | 依赖 `paired_results.jsonl` 路径 — 缺失时 paper case 为空；当前 4/4 filled |
| Minor | `_reviews/*-review.md` | 模板化自审；深度逻辑见本文件 |
| Minor | `generate_wave_plans.py` | 生成 plan/review 骨架 — 数字仍以 verify 为准 |

## Assessment

**Ready** — 无 Critical；Important 项已在报告层补偿。后续若改 `_analysis` 口径须重跑 `run_all_computes.py` + `verify_reports.py`。
