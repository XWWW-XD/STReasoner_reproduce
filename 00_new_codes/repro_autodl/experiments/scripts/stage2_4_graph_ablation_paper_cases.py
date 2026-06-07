#!/usr/bin/env python3
"""Run Stage 2.4 paper_cases graph ablation with official vLLM/evaluate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_ROOT_DEFAULT = (
    REPO_ROOT
    / "00_new_codes"
    / "repro_autodl"
    / "experiments"
    / "results"
    / "stage2.4_graph_ablation_paper_cases_6144"
)
SOURCE_DATASET_DEFAULT = (
    REPO_ROOT
    / "00_new_codes"
    / "repro_autodl"
    / "experiments"
    / "stage2_subsets"
    / "paper_cases"
    / "PaperCases.jsonl"
)
MODEL_PATH_DEFAULT = REPO_ROOT / "base_model" / "STReasoner-8B"
PYTHON_DEFAULT = Path("/root/autodl-tmp/conda/envs/str-py310/bin/python")
GRAPH_PATTERN = re.compile(r"Graph Structure:.*?(?=please analyze)", re.DOTALL | re.IGNORECASE)
MAX_TOKENS = 6144
TASK_MAP = {
    "etiological": "reasoning_etiological",
    "entity": "reasoning_entity",
    "correlation": "reasoning_correlation",
    "forecasting": "reasoning_forecasting",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "sample"


def remove_graph_structure(text: str) -> str:
    return GRAPH_PATTERN.sub("", text)


def extract_answer_content(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip().replace("```", "").strip() if match else ""


def response_diag(response: str, tokenizer: Any = None) -> Dict[str, Any]:
    response = response or ""
    lower = response.lower()
    out = {
        "response_chars": len(response),
        "response_tokens_tokenizer": None,
        "answer_tag_open_count": len(re.findall(r"<answer>", response, flags=re.IGNORECASE)),
        "answer_tag_close_count": len(re.findall(r"</answer>", response, flags=re.IGNORECASE)),
        "final_answer_tag_open_count": len(re.findall(r"<final_answer>", response, flags=re.IGNORECASE)),
        "final_answer_tag_close_count": len(re.findall(r"</final_answer>", response, flags=re.IGNORECASE)),
        "empty_response": not response.strip(),
        "reached_max_tokens_6144": False,
        "spatial_term_mentions": sum(
            lower.count(term)
            for term in [
                "graph",
                "structure",
                "node",
                "edge",
                "path",
                "upstream",
                "downstream",
                "spatial",
                "propagate",
                "connected",
                "connection",
            ]
        ),
        "explicit_edge_mentions": len(re.findall(r"Node\s+\d+\s*(?:->|→|to)\s*Node\s+\d+", response)),
    }
    if tokenizer is not None:
        try:
            out["response_tokens_tokenizer"] = len(tokenizer.encode(response, add_special_tokens=False))
            out["reached_max_tokens_6144"] = out["response_tokens_tokenizer"] >= MAX_TOKENS
        except Exception:
            pass
    return out


def run_command(cmd: List[str], log_path: Path, env: Dict[str, str], cwd: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("$ " + " ".join(cmd) + "\n\n")
        log_fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    end = time.time()
    return {
        "command": cmd,
        "log_path": str(log_path),
        "returncode": proc.returncode,
        "start_time": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end_time": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
        "latency_sec": round(end - start, 3),
    }


def prepare_datasets(source_dataset: Path, output_root: Path) -> List[Dict[str, Any]]:
    rows = load_jsonl(source_dataset)
    if len(rows) != 4:
        raise ValueError(f"paper_cases should contain 4 rows, got {len(rows)} from {source_dataset}")

    manifest: List[Dict[str, Any]] = []
    datasets_dir = output_root / "datasets"
    for line_no, row in enumerate(rows):
        category = row.get("category")
        task = TASK_MAP.get(category)
        if not task:
            raise ValueError(f"Unsupported category at line {line_no}: {category}")
        sample_id = row.get("sample_id", f"line_{line_no}")
        case_id = row.get("paper_case_id", sample_id)
        original_input = row["input"]
        no_graph_input = remove_graph_structure(original_input)
        removed_graph_chars = len(original_input) - len(no_graph_input)
        if removed_graph_chars <= 0:
            raise ValueError(f"Graph removal did not change sample {sample_id}")
        for variant, input_text in [
            ("with_graph", original_input),
            ("without_graph", no_graph_input),
        ]:
            out_row = dict(row)
            out_row["input"] = input_text
            out_row["graph_variant"] = variant
            out_row["stage2_4_remove_graph_regex"] = GRAPH_PATTERN.pattern
            out_row["stage2_4_timeseries_unchanged"] = True
            out_row["stage2_4_gold_unchanged"] = True
            rel_name = (
                f"{line_no:02d}_{task}_{slug(case_id)}_{variant}.jsonl"
            )
            dataset_path = datasets_dir / variant / rel_name
            write_jsonl(dataset_path, [out_row])
            manifest.append(
                {
                    "line_no": line_no,
                    "variant": variant,
                    "task": task,
                    "category": category,
                    "sample_id": sample_id,
                    "paper_case_id": case_id,
                    "source_file": row.get("source_file"),
                    "original_line_index": row.get("original_line_index"),
                    "dataset": str(dataset_path),
                    "graph_structure_present": "Graph Structure:" in input_text,
                    "please_analyze_present": "please analyze" in input_text.lower(),
                    "timeseries_count": len(row.get("timeseries", [])),
                    "gold_output": row.get("output"),
                    "removed_graph_chars": removed_graph_chars if variant == "without_graph" else 0,
                    "input_chars": len(input_text),
                }
            )
    write_json(output_root / "datasets" / "manifest.json", manifest)
    return manifest


def get_tokenizer(model_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:
        return None


def read_generated_response(exp_dir: Path) -> Optional[Dict[str, Any]]:
    path = exp_dir / "generated_answer.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not data:
        return None
    return data[0]


def summarize_row(item: Dict[str, Any], exp_dir: Path, tokenizer: Any) -> Dict[str, Any]:
    generated = read_generated_response(exp_dir)
    metrics_path = exp_dir / "evaluation_metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    dataset_row = load_jsonl(Path(item["dataset"]))[0]
    response = (generated or {}).get("response", "")
    parsed = extract_answer_content(response)
    gold = dataset_row.get("output")
    diag = response_diag(response, tokenizer=tokenizer)
    return {
        **item,
        "exp": str(exp_dir.relative_to(REPO_ROOT)),
        "prediction_file": str((exp_dir / "generated_answer.json").relative_to(REPO_ROOT)),
        "metrics_file": str(metrics_path.relative_to(REPO_ROOT)),
        "generated": generated is not None,
        "input_tokens": (generated or {}).get("num_tokens"),
        "parsed_answer": parsed,
        "gold_output": gold,
        "metrics": metrics,
        **diag,
    }


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": ".",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/root/autodl-tmp/cache/huggingface",
            "HF_HUB_CACHE": "/root/autodl-tmp/cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/autodl-tmp/cache/huggingface",
            "HF_DATASETS_CACHE": "/root/autodl-tmp/cache/huggingface/datasets",
            "TORCH_HOME": "/root/autodl-tmp/cache/huggingface/torch",
            "TRITON_CACHE_DIR": "/root/autodl-tmp/cache/triton",
        }
    )
    return env


def run_experiment(args: argparse.Namespace, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    env = build_env()
    logs_dir = args.output_root / "logs"
    command_records: List[Dict[str, Any]] = []
    for item in manifest:
        variant = item["variant"]
        task = item["task"]
        exp_name = f"stage2.4_graph_ablation_paper_cases_6144_{variant}_{item['line_no']:02d}_{item['category']}"
        exp_dir = REPO_ROOT / "exp" / exp_name
        infer_cmd = [
            str(args.python),
            "inference/inference_tsmllm_vllm.py",
            "--task",
            task,
            "--dataset",
            item["dataset"],
            "--exp",
            exp_name,
            "--model_path",
            str(args.model_path),
            "--num_gpus",
            "1",
            "--num_gpus_per_process",
            "1",
            "--max_tokens",
            str(MAX_TOKENS),
            "--temperature",
            "0.2",
            "--output_name",
            "generated_answer.json",
        ]
        eval_cmd = [
            str(args.python),
            "evaluation/evaluate.py",
            "--task",
            task,
            "--dataset",
            item["dataset"],
            "--exp_path",
            str(exp_dir),
            "--pred_pattern",
            "generated_answer",
            "--repo_root",
            str(REPO_ROOT),
        ]
        infer_record = run_command(
            infer_cmd,
            logs_dir / f"{exp_name}_inference.log",
            env=env,
            cwd=REPO_ROOT,
        )
        command_records.append({"stage": "inference", "exp": exp_name, **infer_record})
        if infer_record["returncode"] != 0:
            raise RuntimeError(f"Inference failed for {exp_name}; see {infer_record['log_path']}")
        eval_record = run_command(
            eval_cmd,
            logs_dir / f"{exp_name}_evaluate.log",
            env=env,
            cwd=REPO_ROOT,
        )
        command_records.append({"stage": "evaluate", "exp": exp_name, **eval_record})
        if eval_record["returncode"] != 0:
            raise RuntimeError(f"Evaluate failed for {exp_name}; see {eval_record['log_path']}")
    write_json(args.output_root / "command_records.json", command_records)
    return {"commands": command_records}


def aggregate(args: argparse.Namespace, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokenizer = get_tokenizer(args.model_path)
    rows: List[Dict[str, Any]] = []
    for item in manifest:
        exp_name = f"stage2.4_graph_ablation_paper_cases_6144_{item['variant']}_{item['line_no']:02d}_{item['category']}"
        rows.append(summarize_row(item, REPO_ROOT / "exp" / exp_name, tokenizer=tokenizer))

    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in ["with_graph", "without_graph"]:
        subset = [row for row in rows if row["variant"] == variant]
        choice_rows = [row for row in subset if row["task"] != "reasoning_forecasting"]
        forecast_rows = [row for row in subset if row["task"] == "reasoning_forecasting"]
        choice_correct = sum(
            1
            for row in choice_rows
            if (row.get("metrics") or {}).get("accuracy") == 1.0
        )
        by_variant[variant] = {
            "cases": len(subset),
            "generated_cases": sum(1 for row in subset if row["generated"]),
            "exact_answer_tag_pair_cases": sum(
                1
                for row in subset
                if row["answer_tag_open_count"] == 1 and row["answer_tag_close_count"] == 1
            ),
            "empty_response_cases": sum(1 for row in subset if row["empty_response"]),
            "reached_max_tokens_cases": sum(1 for row in subset if row["reached_max_tokens_6144"]),
            "choice_correct": choice_correct,
            "choice_total": len(choice_rows),
            "avg_input_tokens": round(
                sum(row.get("input_tokens") or 0 for row in subset) / len(subset), 2
            )
            if subset
            else None,
            "avg_response_tokens": round(
                sum(row.get("response_tokens_tokenizer") or 0 for row in subset) / len(subset), 2
            )
            if subset
            else None,
            "avg_spatial_term_mentions": round(
                sum(row.get("spatial_term_mentions") or 0 for row in subset) / len(subset), 2
            )
            if subset
            else None,
            "forecasting_metrics": forecast_rows[0]["metrics"] if forecast_rows else {},
        }
    summary = {
        "experiment": "stage2.4_graph_ablation_paper_cases_6144",
        "date": datetime.now().date().isoformat(),
        "model_path": str(args.model_path.relative_to(REPO_ROOT))
        if args.model_path.is_relative_to(REPO_ROOT)
        else str(args.model_path),
        "source_dataset": str(args.source_dataset),
        "inference_entry": "inference/inference_tsmllm_vllm.py",
        "evaluation_entry": "evaluation/evaluate.py",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "remove_graph_regex": GRAPH_PATTERN.pattern,
        "paper_figure6_expectation": (
            "The paper reports that S-GRPO increases explicit spatial reasoning usage across tasks. "
            "For this input-level ablation, with_graph is expected to preserve more graph-grounded reasoning "
            "and may perform better than without_graph, while without_graph should reduce explicit graph usage."
        ),
        "by_variant": by_variant,
        "rows": rows,
    }
    write_json(args.output_root / "summary.json", summary)
    write_jsonl(args.output_root / "paired_results.jsonl", rows)
    return summary


def md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(args: argparse.Namespace, summary: Dict[str, Any]) -> Path:
    report_path = REPO_ROOT / "00_new_codes" / "reports" / "16-stage2.4paper消融.md"
    rows = summary["rows"]
    lines: List[str] = []
    lines.append("# Stage 2.4 paper_cases graph ablation 报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    wg = summary["by_variant"]["with_graph"]
    wog = summary["by_variant"]["without_graph"]
    lines.append(
        f"本次按 Stage 2.4 要求完成 paper_cases 4 条样例的 paired 对比：同一 STReasoner-8B、官方 vLLM 推理、`max_tokens=6144`、`inference/prompt.json` 后缀、tag-first evaluate。"
    )
    lines.append("")
    lines.append(f"- w/ graph：4/4 生成，选择题 {wg['choice_correct']}/{wg['choice_total']}，forecasting 指标见下表。")
    lines.append(f"- w/o graph：4/4 生成，选择题 {wog['choice_correct']}/{wog['choice_total']}，forecasting 指标见下表。")
    lines.append(
        "- 本实验未改 raw response、未改 gold、未改 timeseries；w/o graph 仅使用 EasyR1 同款正则删除 `Graph Structure:...` 到 `please analyze` 前。"
    )
    lines.append("")
    lines.append("## 论文预期")
    lines.append("")
    lines.append(
        "论文 Figure 6 衡量的是“显式使用空间信息的回答比例”。论文 5.4 写到：S-GRPO 后模型在各任务中比 vanilla GRPO 有更高的 spatial reasoning usage ratio，说明它不仅提高最终分数，也把推理行为推向 spatially grounded strategies。"
    )
    lines.append(
        "对应到本次输入消融，预期是：保留 graph 时，回答中应更容易出现 graph/node/edge/path/upstream/downstream 等空间结构推理；删除 graph 后，这类显式空间推理会减少，且部分任务性能可能下降。注意：本次没有调用 GPT-5.2 judge 复现 Figure 6 的人工/模型判别比例，只记录 tag-first evaluate 与 raw 诊断。"
    )
    lines.append("")
    lines.append("## 实验配置")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 模型 | `{summary['model_path']}` |")
    lines.append("| 推理入口 | `inference/inference_tsmllm_vllm.py` |")
    lines.append("| 评估入口 | `evaluation/evaluate.py` |")
    lines.append("| max_tokens | `6144` |")
    lines.append("| temperature | `0.2` |")
    lines.append("| prompt 后缀 | `inference/prompt.json` 由官方推理脚本追加 |")
    lines.append(f"| remove graph 正则 | `{summary['remove_graph_regex']}` |")
    lines.append(f"| 输出目录 | `{args.output_root.relative_to(REPO_ROOT)}` |")
    lines.append("")
    lines.append("## 汇总指标")
    lines.append("")
    lines.append("| variant | generated | exact answer tag | empty | reach 6144 | choice | avg input tokens | avg response tokens | avg spatial term mentions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant, item in summary["by_variant"].items():
        lines.append(
            f"| {variant} | {item['generated_cases']}/{item['cases']} | {item['exact_answer_tag_pair_cases']}/{item['cases']} | {item['empty_response_cases']} | {item['reached_max_tokens_cases']} | {item['choice_correct']}/{item['choice_total']} | {md_value(item['avg_input_tokens'])} | {md_value(item['avg_response_tokens'])} | {md_value(item['avg_spatial_term_mentions'])} |"
        )
    lines.append("")
    lines.append("## 每条样例结果")
    lines.append("")
    lines.append("| variant | task | sample_id | gold | parsed | metric | input tokens | response tokens | spatial terms | tags |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---|")
    for row in rows:
        if row["task"] == "reasoning_forecasting":
            metrics = row.get("metrics") or {}
            metric = f"MAE={md_value(metrics.get('mae'))}, MAPE={md_value(metrics.get('mape'))}"
            parsed = row.get("parsed_answer", "")
        else:
            metrics = row.get("metrics") or {}
            metric = f"accuracy={md_value(metrics.get('accuracy'))}"
            parsed = row.get("parsed_answer", "")
        lines.append(
            f"| {row['variant']} | {row['task']} | `{row['sample_id']}` | `{row['gold_output']}` | `{parsed}` | {metric} | {md_value(row.get('input_tokens'))} | {md_value(row.get('response_tokens_tokenizer'))} | {md_value(row.get('spatial_term_mentions'))} | {row['answer_tag_open_count']}/{row['answer_tag_close_count']} |"
        )
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append(f"- 数据与汇总：`{args.output_root.relative_to(REPO_ROOT)}/`")
    lines.append(f"- manifest：`{(args.output_root / 'datasets' / 'manifest.json').relative_to(REPO_ROOT)}`")
    lines.append(f"- paired 结果：`{(args.output_root / 'paired_results.jsonl').relative_to(REPO_ROOT)}`")
    lines.append(f"- summary：`{(args.output_root / 'summary.json').relative_to(REPO_ROOT)}`")
    lines.append("- 官方推理输出：`exp/stage2.4_graph_ablation_paper_cases_6144_*`")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("按用户要求，本轮只完成 paper_cases 4 条 paired 对比；是否进入完整 ST-Test graph ablation，等你看完本报告效果后再继续。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output_root = args.output_root.resolve()
    args.model_path = args.model_path.resolve()
    args.python = args.python.resolve()

    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.source_dataset.is_file():
        raise FileNotFoundError(args.source_dataset)

    manifest = prepare_datasets(args.source_dataset, args.output_root)
    if args.prepare_only:
        print(f"Prepared datasets under {args.output_root}")
        return
    if not args.skip_run:
        run_experiment(args, manifest)
    summary = aggregate(args, manifest)
    report_path = write_report(args, summary)
    print(f"Summary: {args.output_root / 'summary.json'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
