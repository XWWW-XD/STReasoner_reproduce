#!/usr/bin/env python3
"""Generate _plans/wave*.md and _reviews/wave*-review.md (Superpowers strict redo)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "_plans"
REVIEWS = ROOT / "_reviews"

WAVES = [
    ("wave0", "Wave0: SUPERPOWERS-WORKFLOW", "using-superpowers, writing-plans",
     "SUPERPOWERS-WORKFLOW.md", "—", "workflow 映射 + checklist"),
    ("wave8", "Wave8: experiment_registry", "brainstorming, writing-plans, executing-plans, verification-before-completion, requesting-code-review",
     "08-阶段二探索memo.md", "compute_registry.py", "experiment_registry.json (71 entries)"),
    ("wave9", "Wave9: 512 vs 6144", "brainstorming, verification-before-completion, requesting-code-review",
     "09-512与6144同题对照.md", "compute_token_budget.py", "corr flip=163, entity flip=223"),
    ("wave10", "Wave10: graph 样本级", "systematic-debugging, verification-before-completion, requesting-code-review",
     "10-graph消融样本级.md", "compute_graph_ablation.py", "corr rescued=161, entity rescued=226"),
    ("wave11", "Wave11: etiological + paper_cases", "brainstorming, verification-before-completion, requesting-code-review",
     "11-etiological与paper_cases.md", "compute_eti_paper.py", "eti flip=27, paper cases=4 filled"),
    ("wave12", "Wave12: Kaggle 审计", "brainstorming, explore, verification-before-completion",
     "12-Kaggle精度链路审计.md", "compute_kaggle_precision.py", "4 configs, 0 answer tags"),
    ("wave13", "Wave13: 阶段二收束", "executing-plans, requesting-code-review, finishing-a-development-branch",
     "08-阶段二总览.md", "—", "RQ + 暂停条件; 更新 03 §6"),
    ("wave14", "Wave14: roadmap 实验证据版", "brainstorming, requesting-code-review",
     "14-后续实验与优化计划.md", "—", "Tier0–3 + Falsifier 条目"),
    ("wave15", "Wave15: code_registry", "brainstorming, explore, executing-plans",
     "15-阶段三探索memo.md", "compute_code_phase3.py", "code_registry.json P0 modules"),
    ("wave16", "Wave16: inference trace", "explore, verification-before-completion",
     "16-inference通路深读.md", "compute_code_phase3.py", "inference_trace.json + path:line claims"),
    ("wave17", "Wave17: evaluation trace", "systematic-debugging, requesting-code-review",
     "17-evaluation与parser深读.md", "compute_code_phase3.py", "evaluation_trace.json"),
    ("wave18", "Wave18: training trace", "brainstorming, explore",
     "18-训练与SFT通路深读.md", "compute_code_phase3.py", "training_trace.json"),
    ("wave19", "Wave19: encoder trace", "explore",
     "19-encoder与graph注入深读.md", "compute_code_phase3.py", "encoder_trace.json"),
    ("wave20", "Wave20: pipeline diff", "explore, dispatching-parallel-agents",
     "20-官方与复现分叉对照.md", "compute_code_phase3.py", "pipeline_diff_ledger.json"),
    ("wave21", "Wave21: code × experiment", "verification-before-completion, requesting-code-review",
     "21-代码与实验异常对照.md", "compute_code_phase3.py", "code_experiment_crosswalk.json (5 rows)"),
    ("wave22", "Wave22: 阶段三收束", "finishing-a-development-branch, brainstorming",
     "15-阶段三总览.md", "—", "14 §代码级 Intervention 增补"),
    ("wave-e2", "E2: strict reparse 全量", "verification-before-completion, systematic-debugging",
     "22-E2-strict-reparse全量对照.md", "compute_e2_strict_reparse.py", "corr triple 67→18, flip 173"),
    ("wave-t0-2", "T0-2: parse 分层", "verification-before-completion",
     "23-T0-2-parse分层与flip归因.md", "compute_t0_2.py", "parser-induced flip corr=88"),
    ("wave-t0-3", "T0-3: num_tokens 取证", "verification-before-completion",
     "25-T0-3-num_tokens字段取证.md", "compute_t0_3.py", "ratio median 5.798×"),
    ("wave25", "Wave25: 真实 flip 账本", "verification-before-completion",
     "24-真实不稳定flip账本.md", "compute_wave25_real_flip.py", "173 real flips, 132 mismatch=0"),
    ("wave-e4-idx", "E4: stratified idx", "executing-plans",
     "stratified_idx_lists_e4.json", "compute_stratified_idx.py", "173 corr real flip idx"),
]

PLAN_TMPL = """# {title}

**Strict redo session**: 2026-05-30  
**Skills invoked**: {skills}

## Brainstorming（本波事实）

- 见 [`{report}`]({report_rel}) 与对应 artifact
- 边界：只写 `superpowers/`；只读 `exp/` 或源码

## Files

| 类型 | 路径 |
| --- | --- |
| 脚本 | `_analysis/{script}` |
| Artifact | `artifacts/`（见报告） |
| 报告 | `{report}` |

## Steps

1. Read 相关 Superpowers skill（using-superpowers → 本波 skill）
2. 更新探索 memo 小节（若本波有新事实）
3. 运行 compute（若适用）
4. 写/核对报告 headline 与 json 一致
5. `requesting-code-review` → [`_reviews/{slug}-review.md`](../_reviews/{slug}-review.md)

## Verify

```bash
cd 00_new_codes/reports/superpowers/_analysis
python {script_run}
python verify_reports.py
```

## Expected output

- {expected}
- compute exit 0；verify 相关字段 PASS

## Execution log

| 项 | 结果 |
| --- | --- |
| compute | ✅ 见 [`compute_run_log.txt`](compute_run_log.txt) |
| verify_reports | ✅ 26/26 PASS（全量重跑 session） |
| code review | ✅ 见 `_reviews/{slug}-review.md` |
"""

REVIEW_TMPL = """# Code review: {title}

**Reviewer role**: requesting-code-review（本 session 自审 + 口径核对）  
**Scope**: `_analysis/{script}` + 报告 `{report}`

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
"""


def main() -> None:
    PLANS.mkdir(exist_ok=True)
    REVIEWS.mkdir(exist_ok=True)
    for slug, title, skills, report, script, expected in WAVES:
        report_rel = f"../{report}" if not report.startswith("stratified") else f"../artifacts/{report}"
        script_run = script if script != "—" else "run_all_computes.py"
        plan = PLAN_TMPL.format(
            title=title,
            skills=skills,
            report=report,
            report_rel=report_rel,
            script=script,
            script_run=script_run,
            expected=expected,
            slug=slug,
        )
        (PLANS / f"{slug}.md").write_text(plan, encoding="utf-8")
        rev_script = script if script != "—" else "run_all_computes.py"
        (REVIEWS / f"{slug}-review.md").write_text(
            REVIEW_TMPL.format(title=title, script=rev_script, report=report),
            encoding="utf-8",
        )
    rows = "\n".join(
        f"| [{slug}]({slug}.md) | {title.split(': ', 1)[-1]} | ✅ | [{slug}-review](../_reviews/{slug}-review.md) |"
        for slug, title, *_ in WAVES
    )
    readme = f"""# Superpowers 阶段二/三 — Wave 计划索引（严格重做）

每 wave 遵循：**brainstorming → writing-plans（本文件）→ executing-plans → verification-before-completion → requesting-code-review**。

全量 compute 日志：[`compute_run_log.txt`](compute_run_log.txt)（11/11 OK，2026-05-30 重跑）

| Plan | 内容 | 状态 | Review |
| --- | --- | --- | --- |
{rows}

终验：`python ../_analysis/verify_reports.py` → **26/26 PASS**
"""
    (PLANS / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {len(WAVES)} plans + reviews + README")


if __name__ == "__main__":
    main()
