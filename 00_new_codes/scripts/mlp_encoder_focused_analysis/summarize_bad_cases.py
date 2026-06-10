#!/usr/bin/env python3
"""Create a markdown table for the revised MLP encoder boundary evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from find_bad_cases import find_repo_root

ROOT = find_repo_root(_SCRIPT_DIR)
OUT_DIR = ROOT / "00_new_codes/reports/artifacts/mlp_encoder_focused_analysis"


def clip(value, limit: int = 90) -> str:
    text = str(value).replace("|", "/").replace("\n", " ")
    return text[:limit]


def fmt_numeric_fidelity(case: dict) -> str:
    return (
        f"| 复述时间序列错误 | {case['task_type']} | {case['idx']} | "
        f"`{case['data_file']}`:{case['line_no']} | "
        f"Node {case['node']} | {case['stated_window'][0]}-{case['stated_window'][1]} | "
        f"{clip(case['raw_values'], 150)} | {clip(case['model_stated_values'], 150)} | "
        f"模型在推理中复述该时间段时读数错误 |"
    )


def fmt_cross_node(case: dict) -> str:
    check = case.get("top_reasoning_numeric_check")
    if not check:
        return ""
    return (
        f"| 跨节点推理中的时间序列复述错误 | {case['task_type']} | {case['idx']} | "
        f"`{case['data_file']}`:{case['line_no']} | Node {check['node']} | "
        f"{check['stated_window'][0]}-{check['stated_window'][1]} | "
        f"{clip(check['raw_values'], 150)} | {clip(check['model_stated_values'], 150)} | "
        f"模型在跨节点推理中复述该节点时间段时读数错误 |"
    )


def distinct_by_idx(cases: list[dict], limit: int) -> list[dict]:
    picked = []
    seen = set()
    for case in cases:
        if case["idx"] in seen:
            continue
        picked.append(case)
        seen.add(case["idx"])
        if len(picked) >= limit:
            break
    return picked


def period_alignment_row(case: dict, label: str) -> str:
    check = case.get("top_reasoning_numeric_check") if "top_reasoning_numeric_check" in case else case
    if not check:
        return ""
    return (
        f"| {label} | {case['task_type']} | {case['idx']} | `{case['data_file']}`:{case['line_no']} | Node {check['node']} | "
        f"{check['stated_window'][0]}-{check['stated_window'][1]} | "
        f"{clip(check['raw_values'], 150)} | {clip(check['model_stated_values'], 150)} |"
    )


def main() -> None:
    summary = json.loads((OUT_DIR / "boundary_cases_summary.json").read_text())
    numeric_reps = distinct_by_idx(summary["representatives"]["numeric_fidelity_in_reasoning"], 3)
    rows = [
        "| 证据类型 | task type | sample idx | data path:line | 节点 | 时间段 | 原始真实时间序列 | 模型推理中复述的时间序列 | 错误说明 |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for case in numeric_reps:
        rows.append(fmt_numeric_fidelity(case))
    for case in summary["representatives"]["cross_patch_cross_node_reasoning_error"]:
        row = fmt_cross_node(case)
        if row:
            rows.append(row)
            break
    out = OUT_DIR / "boundary_cases_table.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(out)

    period_rows = [
        "| 证据类型 | task type | sample idx | data path:line | 节点 | 时间段 | 真实时间序列 | 模型推理中复述的时间序列 |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for i, case in enumerate(numeric_reps, start=1):
        period_rows.append(period_alignment_row(case, f"数值复述错误-{i}"))
    for case in summary["representatives"]["cross_patch_cross_node_reasoning_error"][:1]:
        row = period_alignment_row(case, "跨节点推理读数错误")
        if row:
            period_rows.append(row)
    period_out = OUT_DIR / "boundary_period_alignment.md"
    period_out.write_text("\n".join(period_rows) + "\n", encoding="utf-8")
    print(period_out)


if __name__ == "__main__":
    main()
