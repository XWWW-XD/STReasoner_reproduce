#!/usr/bin/env python3
"""Stage 2.5 ST-Test checkpoint comparison (CoT -> Align -> Qwen3-8B)."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_ROOT_DEFAULT = REPO_ROOT / "00_new_codes" / "repro_autodl" / "experiments" / "results" / "stage2.5_checkpoint_compare_sttest_6144"
PYTHON_DEFAULT = Path("/root/autodl-tmp/conda/envs/str-py310/bin/python")
MAX_TOKENS = 6144
CHECKPOINT_ORDER = ["cot", "align", "qwen3-8b"]
CHECKPOINT_PATHS = {"cot": REPO_ROOT / "base_model" / "STReasoner-8B-CoT", "align": REPO_ROOT / "base_model" / "STReasoner-8B-Align", "qwen3-8b": REPO_ROOT / "base_model" / "Qwen3-8B"}
CHECKPOINT_LABELS = {"cot": "STReasoner-8B-CoT", "align": "STReasoner-8B-Align", "qwen3-8b": "Qwen3-8B"}
TASKS = [("reasoning_etiological", "data/ST-Bench/ST-Test/etiological_test.jsonl"), ("reasoning_entity", "data/ST-Bench/ST-Test/entity_test.jsonl"), ("reasoning_correlation", "data/ST-Bench/ST-Test/correlation_test.jsonl"), ("reasoning_forecasting", "data/ST-Bench/ST-Test/forecasting_test.jsonl")]
CHOICE_TASKS = {"reasoning_etiological", "reasoning_entity", "reasoning_correlation"}
TASK_SAMPLES = {"reasoning_etiological": 207, "reasoning_entity": 1194, "reasoning_correlation": 1592, "reasoning_forecasting": 280}

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def build_env():
    env = os.environ.copy()
    env.update({"PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "HF_HOME": "/root/autodl-tmp/cache/huggingface", "HF_HUB_CACHE": "/root/autodl-tmp/cache/huggingface", "TRANSFORMERS_CACHE": "/root/autodl-tmp/cache/huggingface", "TORCH_HOME": "/root/autodl-tmp/cache/huggingface/torch", "TRITON_CACHE_DIR": "/root/autodl-tmp/cache/triton"})
    return env

def verify_model(path):
    if not path.is_dir(): raise FileNotFoundError(path)
    if not (any(path.glob("model*.safetensors")) or (path / "model.safetensors.index.json").is_file()):
        raise FileNotFoundError(f"Incomplete weights: {path}")

def run_command(cmd, log_path, cwd, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n"); proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=fh, stderr=subprocess.STDOUT, check=False)
    end = time.time()
    return {"command": cmd, "log_path": str(log_path), "returncode": proc.returncode, "start_time": datetime.fromtimestamp(start).isoformat(timespec="seconds"), "end_time": datetime.fromtimestamp(end).isoformat(timespec="seconds"), "latency_sec": round(end - start, 3)}

def exp_name(checkpoint, task):
    return f"stage2.5_ckpt_sttest_{checkpoint}_{task}"

def run_checkpoint(args, checkpoint):
    model_path = CHECKPOINT_PATHS[checkpoint]; verify_model(model_path)
    env = build_env(); records_path = args.output_root / "command_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    for task, rel_dataset in TASKS:
        dataset = str(REPO_ROOT / rel_dataset); name = exp_name(checkpoint, task); exp_dir = REPO_ROOT / "exp" / name
        infer_cmd = [str(args.python), "inference/inference_tsmllm_vllm.py", "--task", task, "--dataset", dataset, "--exp", name, "--model_path", str(model_path), "--num_gpus", "1", "--num_gpus_per_process", "1", "--max_tokens", str(MAX_TOKENS), "--temperature", "0.2", "--output_name", "generated_answer.json"]
        eval_cmd = [str(args.python), "evaluation/evaluate.py", "--task", task, "--dataset", dataset, "--exp_path", str(exp_dir), "--pred_pattern", "generated_answer", "--repo_root", str(REPO_ROOT)]
        ir = run_command(infer_cmd, args.output_root / "logs" / f"{name}_inference.log", REPO_ROOT, env)
        records.append({"checkpoint": checkpoint, "stage": "inference", "task": task, "exp": name, **ir}); write_json(records_path, records)
        if ir["returncode"] != 0: raise RuntimeError(f"Inference failed: {name}")
        er = run_command(eval_cmd, args.output_root / "logs" / f"{name}_evaluate.log", REPO_ROOT, env)
        records.append({"checkpoint": checkpoint, "stage": "evaluate", "task": task, "exp": name, **er}); write_json(records_path, records)
        if er["returncode"] != 0: raise RuntimeError(f"Evaluate failed: {name}")

def response_diag(response):
    return {"answer_tag_open_count": len(re.findall(r"<answer>", response or "", flags=re.I)), "answer_tag_close_count": len(re.findall(r"</answer>", response or "", flags=re.I)), "empty_response": not (response or "").strip()}

def read_generated(exp_dir):
    path = exp_dir / "generated_answer.json"
    if not path.is_file(): return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data: data = data["results"]
    return data if isinstance(data, list) else []

def aggregate(args):
    rows = []
    for checkpoint in CHECKPOINT_ORDER:
        for task, rel_dataset in TASKS:
            name = exp_name(checkpoint, task); exp_dir = REPO_ROOT / "exp" / name
            mp = exp_dir / "evaluation_metrics.json"; metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
            generated = read_generated(exp_dir); diag_rows = [response_diag(str(r.get("response", ""))) for r in generated]
            rows.append({"checkpoint": checkpoint, "checkpoint_label": CHECKPOINT_LABELS[checkpoint], "task": task, "dataset": rel_dataset, "samples": TASK_SAMPLES[task], "exp": str(exp_dir.relative_to(REPO_ROOT)), "metrics": metrics, "generated_rows": len(generated), "exact_answer_tag_pair": sum(1 for d in diag_rows if d["answer_tag_open_count"]==1 and d["answer_tag_close_count"]==1), "empty_response": sum(1 for d in diag_rows if d["empty_response"]), "tag_pair_rate": round(sum(1 for d in diag_rows if d["answer_tag_open_count"]==1 and d["answer_tag_close_count"]==1)/len(diag_rows), 4) if diag_rows else None})
    comparison = {"weighted_choice_accuracy": {}, "by_checkpoint": {}}
    for checkpoint in CHECKPOINT_ORDER:
        ckpt_rows = [r for r in rows if r["checkpoint"] == checkpoint]
        acc_sum, acc_w = 0.0, 0
        for row in ckpt_rows:
            if row["task"] in CHOICE_TASKS:
                acc = (row.get("metrics") or {}).get("accuracy")
                if acc is not None: acc_sum += acc * TASK_SAMPLES[row["task"]]; acc_w += TASK_SAMPLES[row["task"]]
        comparison["weighted_choice_accuracy"][checkpoint] = round(acc_sum / acc_w, 6) if acc_w else None
        comparison["by_checkpoint"][checkpoint] = {row["task"]: row["metrics"] for row in ckpt_rows}
    summary = {"experiment": "stage2.5_checkpoint_compare_sttest_6144", "date": datetime.now().date().isoformat(), "checkpoint_order": CHECKPOINT_ORDER, "max_tokens": MAX_TOKENS, "temperature": 0.2, "rows": rows, "comparison": comparison}
    write_json(args.output_root / "summary.json", summary); write_json(args.output_root / "comparison.json", comparison); return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_ORDER, default=None)
    parser.add_argument("action", choices=["run-checkpoint", "run-all", "aggregate"])
    args = parser.parse_args(); args.output_root = args.output_root.resolve(); args.python = args.python.resolve(); args.output_root.mkdir(parents=True, exist_ok=True)
    if args.action == "run-checkpoint":
        if not args.checkpoint: raise ValueError("--checkpoint required")
        run_checkpoint(args, args.checkpoint); return
    if args.action == "run-all":
        for ck in CHECKPOINT_ORDER: print(f"Running ST-Test: {ck}"); run_checkpoint(args, ck)
        aggregate(args); return
    aggregate(args)

if __name__ == "__main__":
    main()
