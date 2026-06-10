# Wave12: Kaggle 审计

**Strict redo session**: 2026-05-30  
**Skills invoked**: brainstorming, explore, verification-before-completion

## Brainstorming（本波事实）

- 见 [`12-Kaggle精度链路审计.md`](../12-Kaggle精度链路审计.md) 与对应 artifact
- 边界：只写 `superpowers/`；只读 `exp/` 或源码

## Files

| 类型 | 路径 |
| --- | --- |
| 脚本 | `_analysis/compute_kaggle_precision.py` |
| Artifact | `artifacts/`（见报告） |
| 报告 | `12-Kaggle精度链路审计.md` |

## Steps

1. Read 相关 Superpowers skill（using-superpowers → 本波 skill）
2. 更新探索 memo 小节（若本波有新事实）
3. 运行 compute（若适用）
4. 写/核对报告 headline 与 json 一致
5. `requesting-code-review` → [`_reviews/wave12-review.md`](../_reviews/wave12-review.md)

## Verify

```bash
cd 00_new_codes/reports/superpowers/_analysis
python compute_kaggle_precision.py
python verify_reports.py
```

## Expected output

- 4 configs, 0 answer tags
- compute exit 0；verify 相关字段 PASS

## Execution log

| 项 | 结果 |
| --- | --- |
| compute | ✅ 见 [`compute_run_log.txt`](compute_run_log.txt) |
| verify_reports | ✅ 26/26 PASS（全量重跑 session） |
| code review | ✅ 见 `_reviews/wave12-review.md` |
