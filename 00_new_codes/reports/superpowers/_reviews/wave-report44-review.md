# Code review: wave-report44 bundled MLP

**Reviewer session**: 2026-05-30  
**Skill**: requesting-code-review

## 口径一致性

| 检查项 | 结果 |
| --- | --- |
| strict mismatch 定义与 report 39 §0 一致 | ✅ |
| 阈值 0.01、双对齐、≥3 点 | ✅ |
| `--exp-layout bundled` 路径 `exp_STReasoner-8B/reasoning_{task}-STReasoner-8B/` | ✅ |
| run1 回归 forecasting 141/280、correlation 347/1592 | ✅（双脚本重跑） |

## 逻辑 / 结论

| 检查项 | 结果 |
| --- | --- |
| §4 先于 §3 解读（metrics 分叉） | ✅ |
| 未写「bundled ≈ run1」 | ✅ |
| idx 交集表与 artifact 一致 | ✅ |
| 39 §5 代表 idx membership 逐条记录 | ✅ |

## 代码改动

| 文件 | 评价 |
| --- | --- |
| `find_bad_cases.py` | `--exp-layout` + `exp_dir` 分支；ROOT=`_SCRIPT_DIR.parents[2]`（仓库根） |
| `summarize_task_level_stats.py` | 同步 CLI；forecasting MAPE 缺失时 validation 不 KeyError |

## 遗留（可接受）

- bundled `evaluation_metrics.json` 无 MAPE 字段 → validation 仅比 MAE（已 note）
- `_report44_intersection.py` 为一次性 helper，可删

## 结论

**通过** — 可进入整洁化建议 1–5。
