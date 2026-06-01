#!/usr/bin/env python3
"""Stage 2.5 paper_cases checkpoint comparison (CoT -> Align -> Qwen3-8B)."""

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
    REPO_ROOT / "00_new_codes" / "repro_autodl" / "experiments" / "results"
    / "stage2.5_checkpoint_compare_paper_cases_6144"
)
SOURCE_DATASET_DEFAULT = (
    REPO_ROOT / "00_new_codes" / "repro_autodl" / "experiments"
    / "stage2_subsets" / "paper_cases" / "PaperCases.jsonl"
)
PYTHON_DEFAULT = Path("/root/autodl-tmp/conda/envs/str-py310/bin/python")
MAX_TOKENS = 6144
CHECKPOINT_ORDER = ["cot", "align", "qwen3-8b"]
CHECKPOINT_PATHS = {
    "cot": REPO_ROOT / "base_model" / "STReasoner-8B-CoT",
    "align": REPO_ROOT / "base_model" / "STReasoner-8B-Align",
    "qwen3-8b": REPO_ROOT / "base_model" / "Qwen3-8B",
}
CHECKPOINT_LABELS = {
    "cot": "Time-HD-Anonymous/STReasoner-8B-CoT",
    "align": "Time-HD-Anonymous/STReasoner-8B-Align",
    "qwen3-8b": "Qwen/Qwen3-8B (initial_model)",
}
TASK_MAP = {
    "etiological": "reasoning_etiological",
    "entity": "reasoning_entity",
    "correlation": "reasoning_correlation",
    "forecasting": "reasoning_forecasting",
}

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

def slug(text):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "sample"

def extract_answer_content(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip().replace("```", "").strip() if match else ""

def response_diag(response, tokenizer=None):
    response = response or ""
    out = {
        "response_chars": len(response),
        "response_tokens_tokenizer": None,
        "answer_tag_open_count": len(re.findall(r"<answer>", response, flags=re.IGNORECASE)),
        "answer_tag_close_count": len(re.findall(r"</answer>", response, flags=re.IGNORECASE)),
        "empty_response": not response.strip(),
        "reached_max_tokens_6144": False,
    }
    if tokenizer is not None:
        try:
            out["response_tokens_tokenizer"] = len(tokenizer.encode(response, add_special_tokens=False))
            out["reached_max_tokens_6144"] = out["response_tokens_tokenizer"] >= MAX_TOKENS
        except Exception:
            pass
    return out

def run_command(cmd, log_path, env, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("$ " + " ".join(cmd) + "\n\n")
        log_fh.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
    end = time.time()
    return {
        "command": cmd, "log_path": str(log_path), "returncode": proc.returncode,
        "start_time": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end_time": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
        "latency_sec": round(end - start, 3),
    }

def prepare_datasets(source_dataset, output_root):
    rows = load_jsonl(source_dataset)
    if len(rows) != 4:
        raise ValueError(f"paper_cases should contain 4 rows, got {len(rows)}")
    manifest = []
    datasets_dir = output_root / "datasets"
    for line_no, row in enumerate(rows):
        category = row.get("category")
        task = TASK_MAP.get(category)
        if not task:
            raise ValueError(f"Unsupported category: {category}")
        sample_id = row.get("sample_id", f"line_{line_no}")
        case_id = row.get("paper_case_id", sample_id)
        rel_name = f"{line_no:02d}_{task}_{slug(case_id)}.jsonl"
        dataset_path = datasets_dir / rel_name
        write_jsonl(dataset_path, [row])
        manifest.append({
            "line_no": line_no, "task": task, "category": category,
            "sample_id": sample_id, "paper_case_id": case_id,
            "dataset": str(dataset_path), "gold_output": row.get("output"),
        })
    write_json(output_root / "datasets" / "manifest.json", manifest)
    return manifest

def build_env():
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1",
        "HF_HOME": "/root/autodl-tmp/cache/huggingface",
        "HF_HUB_CACHE": "/root/autodl-tmp/cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/autodl-tmp/cache/huggingface",
        "HF_DATASETS_CACHE": "/root/autodl-tmp/cache/huggingface/datasets",
        "TORCH_HOME": "/root/autodl-tmp/cache/huggingface/torch",
        "TRITON_CACHE_DIR": "/root/autodl-tmp/cache/triton",
    })
    return env

def verify_model(path):
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory missing: {path}")
    has_weights = any(path.glob("model*.safetensors")) or (path / "model.safetensors.index.json").is_file()
    if not has_weights:
        raise FileNotFoundError(f"Incomplete model weights under {path}")

def exp_name(checkpoint, item):
    return f"stage2.5_ckpt_paper_{checkpoint}_{item['line_no']:02d}_{item['category']}"

def run_checkpoint(args, manifest, checkpoint, records):
    model_path = CHECKPOINT_PATHS[checkpoint]
    verify_model(model_path)
    env = build_env()
    logs_dir = args.output_root / "logs"
    for item in manifest:
        task, name = item["task"], exp_name(checkpoint, item)
        exp_dir = REPO_ROOT / "exp" / name
        infer_cmd = [str(args.python), "inference/inference_tsmllm_vllm.py",
            "--task", task, "--dataset", item["dataset"], "--exp", name,
            "--model_path", str(model_path), "--num_gpus", "1", "--num_gpus_per_process", "1",
            "--max_tokens", str(MAX_TOKENS), "--temperature", "0.2", "--output_name", "generated_answer.json"]
        eval_cmd = [str(args.python), "evaluation/evaluate.py", "--task", task,
            "--dataset", item["dataset"], "--exp_path", str(exp_dir),
            "--pred_pattern", "generated_answer", "--repo_root", str(REPO_ROOT)]
        infer_record = run_command(infer_cmd, logs_dir / f"{name}_inference.log", env, REPO_ROOT)
        records.append({"checkpoint": checkpoint, "stage": "inference", "exp": name, **infer_record})
        write_json(args.output_root / "command_records.json", records)
        if infer_record["returncode"] != 0:
            raise RuntimeError(f"Inference failed for {name}")
        eval_record = run_command(eval_cmd, logs_dir / f"{name}_evaluate.log", env, REPO_ROOT)
        records.append({"checkpoint": checkpoint, "stage": "evaluate", "exp": name, **eval_record})
        write_json(args.output_root / "command_records.json", records)
        if eval_record["returncode"] != 0:
            raise RuntimeError(f"Evaluate failed for {name}")

def get_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:
        return None

def read_generated_response(exp_dir):
    path = exp_dir / "generated_answer.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return data[0] if data else None

def summarize_row(checkpoint, item, tokenizer):
    exp_dir = REPO_ROOT / "exp" / exp_name(checkpoint, item)
    generated = read_generated_response(exp_dir)
    metrics_path = exp_dir / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    response = (generated or {}).get("response", "")
    parsed = extract_answer_content(response)
    gold = item.get("gold_output")
    gold_parsed = extract_answer_content(gold) if gold else gold
    if item["task"] != "reasoning_forecasting":
        correct = parsed.upper() == (gold_parsed or "").upper() if parsed and gold_parsed else False
    else:
        correct = metrics.get("mae") is not None
    diag = response_diag(response, tokenizer=tokenizer)
    return {**item, "checkpoint": checkpoint, "checkpoint_label": CHECKPOINT_LABELS[checkpoint],
        "model_path": str(CHECKPOINT_PATHS[checkpoint].relative_to(REPO_ROOT)),
        "exp": str(exp_dir.relative_to(REPO_ROOT)), "generated": generated is not None,
        "input_tokens": (generated or {}).get("num_tokens"), "parsed_answer": parsed,
        "gold_output": gold, "correct": correct, "metrics": metrics, **diag}

def aggregate(args, manifest, checkpoints):
    rows = []
    by_checkpoint = {}
    for checkpoint in checkpoints:
        tokenizer = get_tokenizer(CHECKPOINT_PATHS[checkpoint])
        ckpt_rows = [summarize_row(checkpoint, item, tokenizer) for item in manifest]
        rows.extend(ckpt_rows)
        choice_rows = [r for r in ckpt_rows if r["task"] != "reasoning_forecasting"]
        forecast_rows = [r for r in ckpt_rows if r["task"] == "reasoning_forecasting"]
        by_checkpoint[checkpoint] = {
            "label": CHECKPOINT_LABELS[checkpoint],
            "model_path": str(CHECKPOINT_PATHS[checkpoint].relative_to(REPO_ROOT)),
            "cases": len(ckpt_rows),
            "generated_cases": sum(1 for r in ckpt_rows if r["generated"]),
            "exact_answer_tag_pair_cases": sum(1 for r in ckpt_rows if r["answer_tag_open_count"] == 1 and r["answer_tag_close_count"] == 1),
            "empty_response_cases": sum(1 for r in ckpt_rows if r["empty_response"]),
            "choice_correct": sum(1 for r in choice_rows if r["correct"]),
            "choice_total": len(choice_rows),
            "forecasting_metrics": forecast_rows[0]["metrics"] if forecast_rows else {},
        }
    summary = {
        "experiment": "stage2.5_checkpoint_compare_paper_cases_6144",
        "date": datetime.now().date().isoformat(),
        "checkpoint_order": checkpoints,
        "source_dataset": str(args.source_dataset),
        "max_tokens": MAX_TOKENS, "temperature": 0.2,
        "by_checkpoint": by_checkpoint, "rows": rows,
    }
    write_json(args.output_root / "summary.json", summary)
    write_jsonl(args.output_root / "paired_results.jsonl", rows)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_ORDER, default=None)
    parser.add_argument("action", choices=["prepare", "run-checkpoint", "run-all", "aggregate"])
    args = parser.parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output_root = args.output_root.resolve()
    args.python = args.python.resolve()
    manifest = prepare_datasets(args.source_dataset, args.output_root)
    if args.action == "prepare":
        print(f"Prepared {args.output_root}")
        return
    checkpoints = [args.checkpoint] if args.checkpoint else list(CHECKPOINT_ORDER)
    records_path = args.output_root / "command_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    if args.action in {"run-checkpoint", "run-all"}:
        for checkpoint in checkpoints:
            print(f"Running checkpoint: {checkpoint}")
            run_checkpoint(args, manifest, checkpoint, records)
    if args.action in {"run-checkpoint", "run-all", "aggregate"}:
        summary = aggregate(args, manifest, list(CHECKPOINT_ORDER))
        print(f"Summary: {args.output_root / 'summary.json'}, rows={len(summary['rows'])}")

if __name__ == "__main__":
    main()
