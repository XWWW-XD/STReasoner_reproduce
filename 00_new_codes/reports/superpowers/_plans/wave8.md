# Wave8: experiment_registry

**Strict redo session**: 2026-05-30  
**Skills invoked**: brainstorming, writing-plans, executing-plans, verification-before-completion, requesting-code-review

## Brainstorming（本波事实）

- 见 [`08-阶段二探索memo.md`](../08-阶段二探索memo.md) 与对应 artifact
- 边界：只写 `superpowers/`；只读 `exp/` 或源码

## Files

| 类型 | 路径 |
| --- | --- |
| 脚本 | `_analysis/compute_registry.py` |
| Artifact | `artifacts/`（见报告） |
| 报告 | `08-阶段二探索memo.md` |

## Steps

1. Read 相关 Superpowers skill（using-superpowers → 本波 skill）
2. 更新探索 memo 小节（若本波有新事实）
3. 运行 compute（若适用）
4. 写/核对报告 headline 与 json 一致
5. `requesting-code-review` → [`_reviews/wave8-review.md`](../_reviews/wave8-review.md)

## Verify

```bash
cd 00_new_codes/reports/superpowers/_analysis
python compute_registry.py
python verify_reports.py
```

## Expected output

- experiment_registry.json (71 entries)
- compute exit 0；verify 相关字段 PASS

## Execution log

| 项 | 结果 |
| --- | --- |
| compute | ✅ 见 [`compute_run_log.txt`](compute_run_log.txt) |
| verify_reports | ✅ 26/26 PASS（全量重跑 session） |
| code review | ✅ 见 `_reviews/wave8-review.md` |
