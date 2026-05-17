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

run_experiment "Experiment A: single_gpu + 4bit" \
  env CUDA_VISIBLE_DEVICES=0 python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision 4bit \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy single_gpu \
    --output_path repro_kaggle/outputs/single4bit_predictions.jsonl \
    --summary_path repro_kaggle/outputs/single4bit_summary.json \
    --log_path repro_kaggle/outputs/single4bit_eval.log

run_experiment "Experiment B: dual_auto + FP16" \
  python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision fp16 \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy dual_auto \
    --output_path repro_kaggle/outputs/dualfp16_auto_predictions.jsonl \
    --summary_path repro_kaggle/outputs/dualfp16_auto_summary.json \
    --log_path repro_kaggle/outputs/dualfp16_auto_eval.log

run_experiment "Experiment C: dual_balanced + FP16" \
  python repro_kaggle/scripts/05_eval_sttest_tiny.py \
    --model_name Time-HD-Anonymous/STReasoner-8B \
    --precision fp16 \
    --max_samples 20 \
    --samples_per_category 5 \
    --attn_backend sdpa \
    --device_strategy dual_balanced \
    --output_path repro_kaggle/outputs/dualfp16_balanced_predictions.jsonl \
    --summary_path repro_kaggle/outputs/dualfp16_balanced_summary.json \
    --log_path repro_kaggle/outputs/dualfp16_balanced_eval.log

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT = Path("repro_kaggle/outputs")
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
        return [f"log not found: {log_path}"]
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
        return ["No key evidence snippets matched the report filters."]
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
        "Current problem is not primarily an FP16 or dual-GPU question. The base single_gpu 4bit "
        "inference path failed or did not generate successfully, so that path should be fixed first."
    )
elif any_dual_ok:
    conclusion = (
        "In the current Kaggle T4 x2 environment, both single_gpu 4bit and at least one dual_gpu FP16 "
        "strategy are feasible. The next comparison should focus on latency, memory, and output stability; "
        "because single_gpu 4bit is simpler, it can remain the default low-resource path unless FP16 gives "
        "a clear benefit."
    )
else:
    blockers = []
    for item in (dual_auto, dual_balanced):
        if item.get("first_error_message"):
            blockers.append(short(item.get("first_error_message"), 180))
    blocker_text = blockers[0] if blockers else "no successful real dual-GPU FP16 run was produced"
    conclusion = (
        "In the current Kaggle T4 x2 + STReasoner-8B + current script configuration, single_gpu 4bit is "
        "the more stable low-resource reproduction strategy. dual_gpu FP16 was unavailable or unstable in "
        f"this experiment; the main observed bottleneck was: {blocker_text}."
    )

if bool(dual_auto.get("model_load_pass")) and not dual_auto_real:
    conclusion += " dual_auto did not form a real two-GPU split, so it is not evidence of dual_gpu FP16 success."

lines: list[str] = []
lines.append("# Single-GPU 4bit vs Dual-GPU FP16 Experiment")
lines.append("")
lines.append("## 1. Goal")
lines.append("")
lines.append(
    "This experiment compares the feasibility and stability of single_gpu 4bit and dual_gpu FP16 "
    "inference for STReasoner-8B in the current Kaggle T4 x2 environment."
)
lines.append("")
lines.append(
    "The conclusion only applies to the current environment, model, and helper script configuration. "
    "It should not be generalized to all dual-GPU systems."
)
lines.append("")
lines.append("## 2. Background")
lines.append("")
lines.append("- An 8B model in FP16 has roughly 16GB-scale weights, while one Kaggle T4 exposes about 14.56 GiB, so single-card FP16 is not a good fit.")
lines.append("- 4bit quantization can greatly reduce VRAM pressure, and single_gpu smoke testing was already known to load successfully.")
lines.append("- dual_gpu FP16 can theoretically split weights, but it can introduce device mismatch, KV cache placement, and processor/timeseries tensor placement issues.")
lines.append("- PyTorch allocated and reserved memory should not be added together: allocated is live tensor memory, while reserved is the caching pool that already contains allocated memory.")
lines.append("")
lines.append("## 3. Experimental Setup")
lines.append("")
lines.append(f"- model_name: `{single.get('model_name') or dual_auto.get('model_name') or 'Time-HD-Anonymous/STReasoner-8B'}`")
lines.append("- dataset: `Time-HD-Anonymous/ST-Bench`, subset `ST-Test`, split `train`")
lines.append("- samples: 20 max, each category at most 5")
lines.append("- attention backend: `sdpa`")
lines.append("- GPU: Kaggle Tesla T4")
lines.append("- A: `single_gpu + 4bit`")
lines.append("- B: `dual_auto + FP16`")
lines.append("- C: `dual_balanced + FP16`")
lines.append("")
lines.append("## 4. Selected Samples")
lines.append("")
if selected:
    for sample in selected:
        lines.append(f"- index `{sample.get('index')}`: `{sample.get('category')}`")
else:
    lines.append("- No selected-sample file was available.")
lines.append("")
lines.append("## 5. Results Table")
lines.append("")
lines.append("| strategy | precision | visible GPUs | actual device map | actually multi-GPU | model load | generate success | generate failed | parse fail rate | accuracy | avg latency | max reserved GPU0 | max reserved GPU1 | main error |")
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
lines.append("## 6. Error Analysis")
lines.append("")
for run in RUNS:
    summary = summaries[run["key"]]
    lines.append(f"### {run['label']}. {run['strategy']} + {run['precision']}")
    lines.append("")
    lines.append(f"- conclusion_hint: `{summary.get('conclusion_hint', 'n/a')}`")
    lines.append(f"- model loading pass: `{summary.get('model_load_pass', 'n/a')}`")
    lines.append(f"- actually multi-GPU: `{summary.get('is_actually_multi_gpu', 'n/a')}`")
    lines.append(f"- failure_count_by_stage: `{short(summary.get('failure_count_by_stage'))}`")
    lines.append(f"- failure_count_by_error_type: `{short(summary.get('failure_count_by_error_type'))}`")
    if summary.get("first_error_message"):
        lines.append(f"- first error: {short(summary.get('first_error_message'), 500)}")
    if run["strategy"] == "dual_auto" and summary.get("model_load_pass") and not summary.get("is_actually_multi_gpu"):
        lines.append("- dual_auto did not actually use both GPUs, so it cannot count as successful dual-GPU FP16 evidence.")
    if run["strategy"] == "dual_balanced" and summary.get("failure_count_by_error_type", {}).get("BALANCED_NOT_SUPPORTED"):
        lines.append("- dual_balanced was reported as unsupported by the current stack.")
    lines.append("")
lines.append("## 7. Evidence Snippets")
lines.append("")
for run in RUNS:
    lines.append(f"### {run['label']}. {run['strategy']} + {run['precision']}")
    lines.append("")
    lines.append("```text")
    lines.extend(evidence(run["log"]))
    lines.append("```")
    lines.append("")
lines.append("## 8. Conclusion")
lines.append("")
lines.append(conclusion)
lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote report: {REPORT_PATH}")
PY

echo "Compare experiment finished. Report: repro_kaggle/outputs/compare_single4bit_dualfp16_report.md"
