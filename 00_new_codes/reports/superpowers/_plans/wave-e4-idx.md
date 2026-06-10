# E4: stratified idx

**Strict redo session**: 2026-05-30  
**Skills invoked**: executing-plans

## Brainstorming（本波事实）

- 见 [`stratified_idx_lists_e4.json`](../artifacts/stratified_idx_lists_e4.json) 与对应 artifact
- 边界：只写 `superpowers/`；只读 `exp/` 或源码

## Files

| 类型 | 路径 |
| --- | --- |
| 脚本 | `_analysis/compute_stratified_idx.py` |
| Artifact | `artifacts/`（见报告） |
| 报告 | `stratified_idx_lists_e4.json` |

## Steps

1. Read 相关 Superpowers skill（using-superpowers → 本波 skill）
2. 更新探索 memo 小节（若本波有新事实）
3. 运行 compute（若适用）
4. 写/核对报告 headline 与 json 一致
5. `requesting-code-review` → [`_reviews/wave-e4-idx-review.md`](../_reviews/wave-e4-idx-review.md)

## Verify

```bash
cd 00_new_codes/reports/superpowers/_analysis
python compute_stratified_idx.py
python verify_reports.py
```

## Expected output

- 173 corr real flip idx
- compute exit 0；verify 相关字段 PASS

## Execution log

| 项 | 结果 |
| --- | --- |
| compute | ✅ 见 [`compute_run_log.txt`](compute_run_log.txt) |
| verify_reports | ✅ 26/26 PASS（全量重跑 session） |
| code review | ✅ 见 `_reviews/wave-e4-idx-review.md` |
