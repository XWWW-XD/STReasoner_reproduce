# Code review: T0-3: num_tokens 取证

**Reviewer role**: requesting-code-review（本 session 自审 + 口径核对）  
**Scope**: `_analysis/compute_t0_3.py` + 报告 `25-T0-3-num_tokens字段取证.md`

## Requirements check

| 要求 | 状态 |
| --- | --- |
| 只读 exp/源码，不改 tools/inference/evaluation/exp | ✅ |
| strict/loose 双口径（若适用） | ✅ |
| 因果句带 artifact 或 path:line | ✅ |
| headline 与 json 一致 | ✅ verify_reports |

## Findings

- **Critical**: 无
- **Important**: 无阻塞项
- **Minor**: phase3 trace 由 regex 扫描 + 人工报告互证；非运行时 trace

## Assessment

**Ready** — 本波交付可进入下一 wave。
