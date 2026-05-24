#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p repro_kaggle/outputs/experiment1

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

run_experiment "Experiment A: single_gpu + 4bit" \
  env CUDA_VISIBLE_DEVICES=0 python repro_kaggle/00_smoke_test_scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision 4bit \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy single_gpu \
    --output_path repro_kaggle/outputs/experiment1/single4bit_predictions.jsonl \
    --summary_path repro_kaggle/outputs/experiment1/single4bit_summary.json \
    --log_path repro_kaggle/outputs/experiment1/single4bit_eval.log

run_experiment "Experiment B: dual_auto + FP16" \
  python repro_kaggle/00_smoke_test_scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision fp16 \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy dual_auto \
    --output_path repro_kaggle/outputs/experiment1/dualfp16_auto_predictions.jsonl \
    --summary_path repro_kaggle/outputs/experiment1/dualfp16_auto_summary.json \
    --log_path repro_kaggle/outputs/experiment1/dualfp16_auto_eval.log

run_experiment "Experiment C: dual_balanced + FP16" \
  python repro_kaggle/00_smoke_test_scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision fp16 \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy dual_balanced \
    --output_path repro_kaggle/outputs/experiment1/dualfp16_balanced_predictions.jsonl \
    --summary_path repro_kaggle/outputs/experiment1/dualfp16_balanced_summary.json \
    --log_path repro_kaggle/outputs/experiment1/dualfp16_balanced_eval.log

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT = Path("repro_kaggle/outputs/experiment1")
REPORT_PATH = OUT / "compare_single4bit_dualfp16_report.md"
SELECTED_PATH = OUT / "compare_single4bit_dualfp16_selected_indices.json"

RUNS = [
    {
        "key": "single4bit",
        "label": "A",
        "strategy": "single_gpu",
        "precision": "4bit",
        "summary": OUT / "single4bit_summary.json",
        "log": OUT / "single4bit_eval.log",
    },
    {
        "key": "dualfp16_auto",
        "label": "B",
        "strategy": "dual_auto",
        "precision": "fp16",
        "summary": OUT / "dualfp16_auto_summary.json",
        "log": OUT / "dualfp16_auto_eval.log",
    },
    {
        "key": "dualfp16_balanced",
        "label": "C",
        "strategy": "dual_balanced",
        "precision": "fp16",
        "summary": OUT / "dualfp16_balanced_summary.json",
        "log": OUT / "dualfp16_balanced_eval.log",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report generation should still complete.
        return {"read_error": f"{exc.__class__.__name__}: {exc}", "path": str(path)}


def short(value: Any, limit: int = 140) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return short(value)


def metric(summary: dict[str, Any], name: str, default: Any = None) -> Any:
    return summary.get(name, default)


def max_reserved(summary: dict[str, Any], gpu: str) -> str:
    values = summary.get("max_reserved_gib_by_gpu") or {}
    return fmt_num(values.get(gpu))


def evidence(log_path: Path) -> list[str]:
    if not log_path.exists():
        return [f"未找到日志: {log_path}"]
    needles = (
        "actual_device_map",
        "model device map",
        "MODEL_LOAD_PASS",
        "LOAD_FAIL",
        "SINGLE_GPU_OOM",
        "BALANCED_NOT_SUPPORTED",
        "DUAL_AUTO_NOT_ACTUALLY_MULTI_GPU",
        "DUAL_BALANCED_NOT_ACTUALLY_MULTI_GPU",
        "GENERATE_PASS",
        "GENERATE_FAIL",
        "PARSE_FAIL",
        "OutOfMemory",
        "out of memory",
        "Expected all tensors",
        "same device",
        "gpu_memory_after_model_load",
        "max_reserved_gib_by_gpu",
    )
    lines = []
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(needle in raw for needle in needles):
            lines.append(raw)
    if not lines:
        return ["没有匹配到报告筛选条件的关键证据片段。"]
    head = lines[:12]
    tail = lines[-8:] if len(lines) > 20 else []
    if tail:
        return head + ["..."] + tail
    return lines[:20]


def selected_samples() -> list[dict[str, Any]]:
    payload = read_json(SELECTED_PATH)
    samples = payload.get("samples")
    return samples if isinstance(samples, list) else []


summaries = {run["key"]: read_json(run["summary"]) for run in RUNS}
selected = selected_samples()

single = summaries["single4bit"]
dual_auto = summaries["dualfp16_auto"]
dual_balanced = summaries["dualfp16_balanced"]

single_ok = bool(single.get("model_load_pass")) and single.get("num_generate_success", 0) > 0
dual_auto_real = bool(dual_auto.get("is_actually_multi_gpu"))
dual_balanced_real = bool(dual_balanced.get("is_actually_multi_gpu"))
dual_auto_ok = bool(dual_auto.get("model_load_pass")) and dual_auto_real and dual_auto.get("num_generate_failed", 0) == 0
dual_balanced_ok = (
    bool(dual_balanced.get("model_load_pass"))
    and dual_balanced_real
    and dual_balanced.get("num_generate_failed", 0) == 0
)
any_dual_ok = dual_auto_ok or dual_balanced_ok

if not single_ok:
    conclusion = (
        "当前问题并不主要是 FP16 或双卡问题；基础的 single_gpu 4bit 推理链路失败或没有成功生成，"
        "因此应先修复 single_gpu 4bit。"
    )
elif any_dual_ok:
    conclusion = (
        "在当前 Kaggle T4 x2 环境下，single_gpu 4bit 和至少一种 dual_gpu FP16 策略都可行。"
        "下一步应重点比较延迟、显存和输出稳定性；由于 single_gpu 4bit 更简单，"
        "除非 FP16 带来明确收益，否则它仍可作为默认低资源方案。"
    )
else:
    blockers = []
    for item in (dual_auto, dual_balanced):
        if item.get("first_error_message"):
            blockers.append(short(item.get("first_error_message"), 180))
    blocker_text = blockers[0] if blockers else "没有产出成功的真实双卡 FP16 运行结果"
    conclusion = (
        "在当前 Kaggle T4 x2 + STReasoner-8B + 当前脚本配置下，single_gpu 4bit 是更稳定的"
        f"低资源复现方案。dual_gpu FP16 在本次实验中不可用或不稳定；主要观察到的瓶颈是：{blocker_text}。"
    )

if bool(dual_auto.get("model_load_pass")) and not dual_auto_real:
    conclusion += " dual_auto 没有形成真实双卡切分，因此不能作为 dual_gpu FP16 成功证据。"

lines: list[str] = []
lines.append("# 单卡 4bit 与双卡 FP16 对照实验")
lines.append("")
lines.append("## 1. 目标")
lines.append("")
lines.append(
    "本实验比较当前 Kaggle T4 x2 环境下，STReasoner-8B 使用 single_gpu 4bit 与 "
    "dual_gpu FP16 推理的可行性和稳定性。"
)
lines.append("")
lines.append(
    "结论只适用于当前环境、当前模型和当前辅助脚本配置，不应泛化到所有双卡环境。"
)
lines.append("")
lines.append("## 2. 背景")
lines.append("")
lines.append("- 8B 模型的 FP16 权重约为 16GB 级别，而单张 Kaggle T4 约 14.56 GiB，因此单卡 FP16 不合适。")
lines.append("- 4bit 量化可以显著降低显存压力；已知 single_gpu smoke test 可以成功加载。")
lines.append("- dual_gpu FP16 理论上可以分摊权重，但可能引入 device mismatch、KV cache 放置、processor/timeseries tensor 设备不一致等问题。")
lines.append("- PyTorch 的 allocated 和 reserved 显存不能相加：allocated 是真实张量占用，reserved 是 PyTorch 缓存池，已经包含 allocated。")
lines.append("")
lines.append("## 3. 实验设置")
lines.append("")
lines.append(f"- model_name: `{single.get('model_name') or dual_auto.get('model_name') or 'Time-HD-Anonymous/STReasoner-8B'}`")
lines.append("- dataset: `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`")
lines.append("- samples: 最多 20 条，每类最多 5 条")
lines.append("- attention backend: `sdpa`")
lines.append("- GPU: Kaggle Tesla T4")
lines.append("- A: `single_gpu + 4bit`")
lines.append("- B: `dual_auto + FP16`")
lines.append("- C: `dual_balanced + FP16`")
lines.append("")
lines.append("## 4. 选中样本")
lines.append("")
if selected:
    for sample in selected:
        lines.append(f"- index `{sample.get('index')}`: `{sample.get('category')}`")
else:
    lines.append("- 未找到选中样本文件。")
lines.append("")
lines.append("## 5. 结果表")
lines.append("")
lines.append("| 策略 | 精度 | 可见 GPU | 实际 device map | 是否真实双卡 | 模型加载 | 生成成功数 | 生成失败数 | 解析失败率 | 准确率 | 平均延迟 | GPU0 峰值 reserved | GPU1 峰值 reserved | 主要错误 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    summary = summaries[run["key"]]
    lines.append(
        "| "
        + " | ".join(
            [
                run["strategy"],
                run["precision"],
                short(summary.get("visible_gpu_count")),
                short(summary.get("actual_device_map")),
                short(summary.get("is_actually_multi_gpu")),
                short(summary.get("model_load_pass")),
                short(summary.get("num_generate_success")),
                short(summary.get("num_generate_failed")),
                fmt_num(summary.get("parse_fail_rate")),
                fmt_num(summary.get("accuracy_overall_if_applicable")),
                fmt_num(summary.get("avg_latency_sec")),
                max_reserved(summary, "gpu0"),
                max_reserved(summary, "gpu1"),
                short(summary.get("first_error_message"), 160),
            ]
        )
        + " |"
    )
lines.append("")
lines.append("## 6. 错误分析")
lines.append("")
for run in RUNS:
    summary = summaries[run["key"]]
    lines.append(f"### {run['label']}. {run['strategy']} + {run['precision']}")
    lines.append("")
    lines.append(f"- conclusion_hint: `{summary.get('conclusion_hint', 'n/a')}`")
    lines.append(f"- 模型加载是否通过: `{summary.get('model_load_pass', 'n/a')}`")
    lines.append(f"- 是否真实双卡: `{summary.get('is_actually_multi_gpu', 'n/a')}`")
    lines.append(f"- failure_count_by_stage: `{short(summary.get('failure_count_by_stage'))}`")
    lines.append(f"- failure_count_by_error_type: `{short(summary.get('failure_count_by_error_type'))}`")
    if summary.get("first_error_message"):
        lines.append(f"- 首个错误: {short(summary.get('first_error_message'), 500)}")
    if run["strategy"] == "dual_auto" and summary.get("model_load_pass") and not summary.get("is_actually_multi_gpu"):
        lines.append("- dual_auto 没有实际使用两张 GPU，因此不能算作 dual-GPU FP16 成功证据。")
    if run["strategy"] == "dual_balanced" and summary.get("failure_count_by_error_type", {}).get("BALANCED_NOT_SUPPORTED"):
        lines.append("- 当前软件栈报告 dual_balanced 不受支持。")
    lines.append("")
lines.append("## 7. 证据片段")
lines.append("")
for run in RUNS:
    lines.append(f"### {run['label']}. {run['strategy']} + {run['precision']}")
    lines.append("")
    lines.append("```text")
    lines.extend(evidence(run["log"]))
    lines.append("```")
    lines.append("")
lines.append("## 8. 结论")
lines.append("")
lines.append(conclusion)
lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入报告: {REPORT_PATH}")
PY

echo "对照实验完成。报告: repro_kaggle/outputs/experiment1/compare_single4bit_dualfp16_report.md"
