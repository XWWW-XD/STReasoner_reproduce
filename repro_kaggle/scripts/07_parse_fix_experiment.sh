#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p repro_kaggle/outputs

run_experiment() {
  local label="$1"
  shift
  echo "================================================================================"
  echo "${label}"
  echo "================================================================================"
  "$@"
  local status=$?
  echo "${label} exit_status=${status}"
  return 0
}

run_experiment "A. 基线: max_new_tokens=64, answer_format_prompt=false" \
  env CUDA_VISIBLE_DEVICES=0 python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision 4bit \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy single_gpu \
    --max_new_tokens 64 \
    --answer_format_prompt false \
    --output_path repro_kaggle/outputs/parsefix_baseline_predictions.jsonl \
    --summary_path repro_kaggle/outputs/parsefix_baseline_summary.json \
    --log_path repro_kaggle/outputs/parsefix_baseline_eval.log

run_experiment "B. 加长生成: max_new_tokens=256, answer_format_prompt=false" \
  env CUDA_VISIBLE_DEVICES=0 python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision 4bit \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy single_gpu \
    --max_new_tokens 256 \
    --answer_format_prompt false \
    --output_path repro_kaggle/outputs/parsefix_longer_predictions.jsonl \
    --summary_path repro_kaggle/outputs/parsefix_longer_summary.json \
    --log_path repro_kaggle/outputs/parsefix_longer_eval.log

run_experiment "C. 强制答案格式: max_new_tokens=256, answer_format_prompt=true" \
  env CUDA_VISIBLE_DEVICES=0 python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision 4bit \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy single_gpu \
    --max_new_tokens 256 \
    --answer_format_prompt true \
    --output_path repro_kaggle/outputs/parsefix_forced_predictions.jsonl \
    --summary_path repro_kaggle/outputs/parsefix_forced_summary.json \
    --log_path repro_kaggle/outputs/parsefix_forced_eval.log

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT = Path("repro_kaggle/outputs")
REPORT_PATH = OUT / "parse_fix_experiment_report.md"

RUNS = [
    {
        "key": "baseline",
        "label": "基线",
        "config": "max_new_tokens=64, answer_format_prompt=false",
        "summary": OUT / "parsefix_baseline_summary.json",
        "predictions": OUT / "parsefix_baseline_predictions.jsonl",
    },
    {
        "key": "longer",
        "label": "加长生成",
        "config": "max_new_tokens=256, answer_format_prompt=false",
        "summary": OUT / "parsefix_longer_summary.json",
        "predictions": OUT / "parsefix_longer_predictions.jsonl",
    },
    {
        "key": "forced",
        "label": "强制答案格式",
        "config": "max_new_tokens=256, answer_format_prompt=true",
        "summary": OUT / "parsefix_forced_summary.json",
        "predictions": OUT / "parsefix_forced_predictions.jsonl",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{exc.__class__.__name__}: {exc}", "path": str(path)}


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def short(value: Any, limit: int = 160) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", "\\n").replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def block_text(value: Any, limit: int = 900) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return short(value)


def find_example(records: list[dict[str, Any]], parse_failed: bool) -> dict[str, Any] | None:
    for record in records:
        if record.get("parse_failed") is parse_failed:
            return record
    return None


summaries = {run["key"]: read_json(run["summary"]) for run in RUNS}
records_by_key = {run["key"]: read_records(run["predictions"]) for run in RUNS}

ranked = sorted(
    RUNS,
    key=lambda run: (
        summaries[run["key"]].get("parse_fail_rate") is None,
        summaries[run["key"]].get("parse_fail_rate", 1.0),
        -summaries[run["key"]].get("num_parse_success", 0),
    ),
)
best = ranked[0]
best_summary = summaries[best["key"]]

lines: list[str] = []
lines.append("# 输出格式修复实验")
lines.append("")
lines.append("## 1. 目标")
lines.append("")
lines.append(
    "本实验的目标是降低由 `no_answer_tag_or_standalone_choice` 导致的解析失败。"
    "前面的实验已经证明模型可以生成文本；这里进一步检查两种修复方向："
    "一是增加生成长度，二是追加明确的最终答案格式要求，看看模型输出是否更容易被解析为 "
    "`<answer>A</answer>` / `<answer>B</answer>` / `<answer>C</answer>` / `<answer>D</answer>`。"
)
lines.append("")
lines.append("## 2. 实验设置")
lines.append("")
lines.append("- model_name: `Time-HD-Anonymous/STReasoner-8B`")
lines.append("- dataset: `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`")
lines.append("- device_strategy: `single_gpu`，使用 `CUDA_VISIBLE_DEVICES=0`")
lines.append("- precision: `4bit`")
lines.append("- attention backend: `sdpa`")
lines.append("- samples: 最多 20 条，每类最多 5 条；如果存在则复用 `compare_single4bit_dualfp16_selected_indices.json`")
lines.append("- 基线: `max_new_tokens=64`, `answer_format_prompt=false`")
lines.append("- 加长生成: `max_new_tokens=256`, `answer_format_prompt=false`")
lines.append("- 强制答案格式: `max_new_tokens=256`, `answer_format_prompt=true`")
lines.append("")
lines.append("## 3. 结果表")
lines.append("")
lines.append("| 实验组 | 配置 | 解析失败率 | 解析成功数 | 准确率 | 平均延迟 | 主要错误 |")
lines.append("|---|---|---:|---:|---:|---:|---|")
for run in RUNS:
    summary = summaries[run["key"]]
    lines.append(
        "| "
        + " | ".join(
            [
                run["label"],
                run["config"],
                fmt(summary.get("parse_fail_rate")),
                fmt(summary.get("num_parse_success")),
                fmt(summary.get("accuracy_overall_if_applicable")),
                fmt(summary.get("avg_latency_sec")),
                short(summary.get("first_error_message")),
            ]
        )
        + " |"
    )
lines.append("")
lines.append("## 4. 证据")
lines.append("")
for run in RUNS:
    records = records_by_key[run["key"]]
    fail = find_example(records, True)
    success = find_example(records, False)
    lines.append(f"### {run['label']}")
    lines.append("")
    if fail:
        lines.append("解析失败样例：")
        lines.append("")
        lines.append("```text")
        lines.append(f"index={fail.get('index')} category={fail.get('category')} gold={fail.get('output')}")
        lines.append(f"error={fail.get('error_message')}")
        lines.append(block_text(fail.get("prediction")))
        lines.append("```")
        lines.append("")
    else:
        lines.append("未找到解析失败样例。")
        lines.append("")
    if success:
        lines.append("解析成功样例：")
        lines.append("")
        lines.append("```text")
        lines.append(
            f"index={success.get('index')} category={success.get('category')} "
            f"gold={success.get('output')} parsed={success.get('parsed_prediction')} "
            f"correct={success.get('is_correct')}"
        )
        lines.append(block_text(success.get("prediction")))
        lines.append("```")
        lines.append("")
    else:
        lines.append("未找到解析成功样例。")
        lines.append("")
lines.append("## 5. 结论")
lines.append("")
if best_summary.get("parse_fail_rate") is None:
    lines.append("没有任何实验组产出可用的 `parse_fail_rate`，因此本次格式修复实验暂时无法下结论。")
else:
    lines.append(
        f"本次实验中效果最好的是 `{best['label']}` "
        f"({best['config']})：parse_fail_rate={fmt(best_summary.get('parse_fail_rate'))}，"
        f"parse_success={fmt(best_summary.get('num_parse_success'))}，"
        f"avg_latency={fmt(best_summary.get('avg_latency_sec'))} 秒。"
    )
    if best["key"] == "forced":
        lines.append(
            "下一轮 tiny eval 建议使用 `max_new_tokens=256`，并启用 `answer_format_prompt=true`。"
        )
    elif best["key"] == "longer":
        lines.append(
            "下一轮 tiny eval 可优先尝试 `max_new_tokens=256`；本次实验里，显式答案格式提示没有带来足够改善。"
        )
    else:
        lines.append(
            "下一轮 tiny eval 可以继续使用基线生成长度；除非人工检查表明更应该先改 parser。"
        )
lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入报告: {REPORT_PATH}")
PY

echo "解析修复实验完成。报告: repro_kaggle/outputs/parse_fix_experiment_report.md"
