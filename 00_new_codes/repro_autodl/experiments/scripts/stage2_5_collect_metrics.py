#!/usr/bin/env python3
"""Collect official evaluation_metrics.json into comparison summary (no custom scoring)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

CHECKPOINTS = [
    ("cot", "STReasoner-8B-CoT"),
    ("align", "STReasoner-8B-Align"),
    ("qwen3-8b", "Qwen3-8B"),
]
TASKS = [
    "reasoning_etiological",
    "reasoning_entity",
    "reasoning_correlation",
    "reasoning_forecasting",
]
CHOICE = {"reasoning_etiological", "reasoning_entity", "reasoning_correlation"}
SAMPLES = {"reasoning_etiological": 207, "reasoning_entity": 1194, "reasoning_correlation": 1592, "reasoning_forecasting": 280}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--mode", choices=["paper", "sttest"], default="sttest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root
    rows = []
    for ckpt_key, model_name in CHECKPOINTS:
        for task in TASKS:
            exp = repo / "exp" / f"{task}-{model_name}"
            mp = exp / "evaluation_metrics.json"
            metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
            rows.append(
                {
                    "checkpoint": ckpt_key,
                    "model_name": model_name,
                    "task": task,
                    "exp": str(exp.relative_to(repo)),
                    "metrics": metrics,
                    "present": mp.is_file(),
                }
            )
    weighted = {}
    for ckpt_key, model_name in CHECKPOINTS:
        acc_sum, w = 0.0, 0
        for task in CHOICE:
            mp = repo / "exp" / f"{task}-{model_name}" / "evaluation_metrics.json"
            if not mp.is_file():
                continue
            acc = json.loads(mp.read_text(encoding="utf-8")).get("accuracy")
            if acc is not None:
                acc_sum += acc * SAMPLES[task]
                w += SAMPLES[task]
        weighted[ckpt_key] = round(acc_sum / w, 6) if w else None
    out = {
        "experiment": f"stage2.5_checkpoint_compare_{args.mode}",
        "date": date.today().isoformat(),
        "checkpoint_order": [c[0] for c in CHECKPOINTS],
        "weighted_choice_accuracy": weighted,
        "rows": rows,
        "note": "All metrics from official evaluation/evaluate.py output only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
