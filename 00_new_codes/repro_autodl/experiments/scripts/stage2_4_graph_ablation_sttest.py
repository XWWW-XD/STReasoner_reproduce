#!/usr/bin/env python3
"""Thin runner for Stage 2.4 ST-Test graph ablation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
    / "stage2.4_graph_ablation_sttest_6144"
)
MODEL_PATH_DEFAULT = REPO_ROOT / "base_model" / "STReasoner-8B"
PYTHON_DEFAULT = Path("/root/autodl-tmp/conda/envs/str-py310/bin/python")
MAX_TOKENS = 6144
GRAPH_PATTERN = re.compile(r"Graph Structure:.*?(?=please analyze)", re.DOTALL | re.IGNORECASE)
TASKS = [
    ("reasoning_etiological", "data/ST-Bench/ST-Test/etiological_test.jsonl"),
    ("reasoning_entity", "data/ST-Bench/ST-Test/entity_test.jsonl"),
    ("reasoning_correlation", "data/ST-Bench/ST-Test/correlation_test.jsonl"),
    ("reasoning_forecasting", "data/ST-Bench/ST-Test/forecasting_test.jsonl"),
]


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_graph_structure(text: str) -> str:
    return GRAPH_PATTERN.sub("", text)


def prepare_without_graph(output_root: Path) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for task, rel_dataset in TASKS:
        src = REPO_ROOT / rel_dataset
        rows = load_jsonl(src)
        out_rows: List[Dict[str, Any]] = []
        changed = 0
        unchanged = 0
        removed_chars = 0
        for idx, row in enumerate(rows):
            original_input = row.get("input", "")
            no_graph_input = remove_graph_structure(original_input)
            if no_graph_input != original_input:
                changed += 1
                removed_chars += len(original_input) - len(no_graph_input)
            else:
                unchanged += 1
            out_row = dict(row)
            out_row["input"] = no_graph_input
            out_row["stage2_4_graph_variant"] = "without_graph"
            out_row["stage2_4_remove_graph_regex"] = GRAPH_PATTERN.pattern
            out_row["stage2_4_timeseries_unchanged"] = True
            out_row["stage2_4_gold_unchanged"] = True
            out_rows.append(out_row)
        out_path = output_root / "datasets" / "without_graph" / Path(rel_dataset).name
        write_jsonl(out_path, out_rows)
        manifest.append(
            {
                "variant": "without_graph",
                "task": task,
                "source_dataset": rel_dataset,
                "dataset": str(out_path),
                "samples": len(rows),
                "changed_inputs": changed,
                "unchanged_inputs": unchanged,
                "removed_graph_chars_total": removed_chars,
            }
        )
    for task, rel_dataset in TASKS:
        rows = load_jsonl(REPO_ROOT / rel_dataset)
        manifest.append(
            {
                "variant": "with_graph",
                "task": task,
                "source_dataset": rel_dataset,
                "dataset": rel_dataset,
                "samples": len(rows),
                "changed_inputs": 0,
                "unchanged_inputs": len(rows),
                "removed_graph_chars_total": 0,
            }
        )
    write_json(output_root / "datasets" / "manifest.json", manifest)
    return manifest


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


def run_command(cmd: List[str], log_path: Path, cwd: Path, env: Dict[str, str]) -> Dict[str, Any]:
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


def selected_manifest(output_root: Path, batch: str) -> List[Dict[str, Any]]:
    manifest_path = output_root / "datasets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [item for item in manifest if item["variant"] == batch]


def exp_name(variant: str, task: str) -> str:
    return f"stage2.4_graph_ablation_sttest_6144_{variant}_{task}"


def run_batch(args: argparse.Namespace, batch: str) -> None:
    env = build_env()
    records_path = args.output_root / "command_records.json"
    if records_path.is_file():
        records = json.loads(records_path.read_text(encoding="utf-8"))
    else:
        records = []
    for item in selected_manifest(args.output_root, batch):
        task = item["task"]
        name = exp_name(batch, task)
        exp_dir = REPO_ROOT / "exp" / name
        dataset = item["dataset"]
        infer_cmd = [
            str(args.python),
            "inference/inference_tsmllm_vllm.py",
            "--task",
            task,
            "--dataset",
            dataset,
            "--exp",
            name,
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
            dataset,
            "--exp_path",
            str(exp_dir),
            "--pred_pattern",
            "generated_answer",
            "--repo_root",
            str(REPO_ROOT),
        ]
        infer_record = run_command(
            infer_cmd,
            args.output_root / "logs" / f"{name}_inference.log",
            cwd=REPO_ROOT,
            env=env,
        )
        records.append({"stage": "inference", "variant": batch, "task": task, "exp": name, **infer_record})
        write_json(records_path, records)
        if infer_record["returncode"] != 0:
            raise RuntimeError(f"Inference failed for {name}; see {infer_record['log_path']}")
        eval_record = run_command(
            eval_cmd,
            args.output_root / "logs" / f"{name}_evaluate.log",
            cwd=REPO_ROOT,
            env=env,
        )
        records.append({"stage": "evaluate", "variant": batch, "task": task, "exp": name, **eval_record})
        write_json(records_path, records)
        if eval_record["returncode"] != 0:
            raise RuntimeError(f"Evaluate failed for {name}; see {eval_record['log_path']}")


def extract_answer_content(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip().replace("```", "").strip() if match else ""


def response_diag(response: str) -> Dict[str, Any]:
    lower = (response or "").lower()
    return {
        "response_chars": len(response or ""),
        "answer_tag_open_count": len(re.findall(r"<answer>", response or "", flags=re.IGNORECASE)),
        "answer_tag_close_count": len(re.findall(r"</answer>", response or "", flags=re.IGNORECASE)),
        "empty_response": not (response or "").strip(),
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
        "explicit_edge_mentions": len(re.findall(r"Node\s+\d+\s*(?:->|→|to)\s*Node\s+\d+", response or "")),
    }


def read_generated(exp_dir: Path) -> List[Dict[str, Any]]:
    path = exp_dir / "generated_answer.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return data if isinstance(data, list) else []


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads((args.output_root / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for item in manifest:
        variant = item["variant"]
        task = item["task"]
        name = exp_name(variant, task)
        exp_dir = REPO_ROOT / "exp" / name
        metrics_path = exp_dir / "evaluation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        generated = read_generated(exp_dir)
        diag_rows = [response_diag(str(row.get("response", ""))) for row in generated]
        rows.append(
            {
                **item,
                "exp": str(exp_dir.relative_to(REPO_ROOT)),
                "prediction_file": str((exp_dir / "generated_answer.json").relative_to(REPO_ROOT)),
                "metrics_file": str(metrics_path.relative_to(REPO_ROOT)),
                "metrics": metrics,
                "generated_rows": len(generated),
                "exact_answer_tag_pair": sum(
                    1 for row in diag_rows if row["answer_tag_open_count"] == 1 and row["answer_tag_close_count"] == 1
                ),
                "empty_response": sum(1 for row in diag_rows if row["empty_response"]),
                "avg_response_chars": round(sum(row["response_chars"] for row in diag_rows) / len(diag_rows), 2)
                if diag_rows
                else None,
                "avg_spatial_term_mentions": round(sum(row["spatial_term_mentions"] for row in diag_rows) / len(diag_rows), 2)
                if diag_rows
                else None,
                "avg_explicit_edge_mentions": round(sum(row["explicit_edge_mentions"] for row in diag_rows) / len(diag_rows), 2)
                if diag_rows
                else None,
            }
        )
    summary = {
        "experiment": "stage2.4_graph_ablation_sttest_6144",
        "date": datetime.now().date().isoformat(),
        "model_path": str(args.model_path.relative_to(REPO_ROOT))
        if args.model_path.is_relative_to(REPO_ROOT)
        else str(args.model_path),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "remove_graph_regex": GRAPH_PATTERN.pattern,
        "rows": rows,
    }
    write_json(args.output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument(
        "action",
        choices=["prepare", "run-with-graph", "run-without-graph", "aggregate"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.model_path = args.model_path.resolve()
    args.python = args.python.resolve()
    if args.action == "prepare":
        manifest = prepare_without_graph(args.output_root)
        print(f"Prepared ST-Test graph ablation datasets: {args.output_root}")
        print(f"Manifest rows: {len(manifest)}")
        return
    if args.action == "run-with-graph":
        run_batch(args, "with_graph")
        return
    if args.action == "run-without-graph":
        run_batch(args, "without_graph")
        return
    if args.action == "aggregate":
        summary = aggregate(args)
        print(f"Summary: {args.output_root / 'summary.json'}")
        print(f"Rows: {len(summary['rows'])}")


if __name__ == "__main__":
    main()
