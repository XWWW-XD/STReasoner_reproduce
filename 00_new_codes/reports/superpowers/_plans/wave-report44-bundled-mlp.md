# Report 44: upstream bundled MLP strict mismatch

**Session**: 2026-05-30  
**Skills invoked**: using-superpowers, brainstorming, writing-plans, executing-plans, verification-before-completion, requesting-code-review

## Brainstorming（RQ 与禁止结论）

### 研究问题

1. bundled 四任务是否存在 forecasting/correlation 大量 strict mismatch？（同 39 pipeline）
2. bundled vs run1 的 mismatch 计数与 idx 交集差多少？（不同 inference snapshot）
3. bundled 任务 metrics 是否对齐论文 Table；run1 是否复现 bundled metrics？（指标可分叉，先 §4 后 §3）

### 禁止结论

- bundled ≈ run1 同一次 run
- mismatch 差 ⇒ 工具 bug
- bundled entity/etiological strict=0 ⇒ 无复述问题

## 预检（blocking）

| 检查项 | 状态 |
| --- | --- |
| 四任务 `exp_STReasoner-8B/.../generated_answer.json` | ✅ 本地齐全 |
| schema `{idx, question_text, response}`，无 num_tokens | ✅ |
| bundled vs run1 metrics 分叉已知 | ✅ corr 0.8712 vs 0.8317 等 |
| run1_recheck artifact 基线 | ✅ 43 已建 |

## Files

| 类型 | 路径 |
| --- | --- |
| 脚本 | `00_new_codes/tools/mlp_encoder_focused_analysis/{find_bad_cases,summarize_task_level_stats}.py` |
| Artifact | `reports/artifacts/mlp_encoder_focused_analysis/upstream_bundled/` |
| 报告 | `reports/44-exp_STReasoner-8B-MLP复述与6144对照.md` |
| 对照 | `run1_recheck/`、`39`、`43` |

## Steps

1. 扩 `--exp-layout bundled`（`configure_paths` + `exp_dir`）
2. 跑 bundled：find → summarize
3. 跑 run1 回归：find → summarize → `run1_recheck/`
4. 写报告 44（§4 先于 §3 解读；含 idx 交集）
5. `_reviews/wave-report44-review.md`

## Verify

```bash
cd 00_new_codes/tools/mlp_encoder_focused_analysis
python find_bad_cases.py --exp-layout bundled --out-subdir upstream_bundled
python summarize_task_level_stats.py --exp-layout bundled --out-subdir upstream_bundled
python find_bad_cases.py --out-subdir run1_recheck
python summarize_task_level_stats.py --out-subdir run1_recheck
```

## Expected output

- bundled + run1 artifact json 存在
- run1_recheck：forecasting 141/280、correlation 347/1592、entity 0/1194、etiological 0/207
- bundled mismatch 与 run1 **同量级**（~14% 合计），不必 idx 完全一致

## Execution log

| 项 | 结果 |
| --- | --- |
| bundled compute | ✅ forecasting 147/280、correlation 344/1592、合计 491/3273 |
| run1 回归 | ✅ 141/280、347/1592、488/3273 |
| 报告 44 | ✅ `44-exp_STReasoner-8B-MLP复述与6144对照.md` |
| code review | ✅ `_reviews/wave-report44-review.md` |
