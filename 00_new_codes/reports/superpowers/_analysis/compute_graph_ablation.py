#!/usr/bin/env python3
"""Wave10: graph ablation sample-level ledger."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00_new_codes/scripts/mlp_encoder_focused_analysis"))

from compute import (  # noqa: E402
    OUT,
    RUNS,
    TASKS,
    collect_sample_outcomes,
    exp_dir,
    exp_dir_graph_without,
    load_jsonl,
    load_mismatch_indices,
    task_correct,
)
from evaluation.evaluate_qa import load_prediction_files  # noqa: E402

CHOICE = ["correlation", "entity", "etiological"]
EDGE_RE = re.compile(r"\bedge\b|\->|arrow|connect", re.I)
SPATIAL_RE = re.compile(r"\bspatial\b|\bnode\b|\bgraph\b|\btopolog", re.I)


def mention_counts(text: str) -> dict:
    t = text or ""
    return {
        "spatial": len(SPATIAL_RE.findall(t)),
        "edge": len(EDGE_RE.findall(t)),
    }


def flip_set(outcomes: dict, task: str) -> set[int]:
    s = set()
    for key, o in outcomes.items():
        if o["task"] != task:
            continue
        cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
        if len(cor) >= 2 and len(set(cor)) > 1:
            s.add(o["idx"])
    return s


def ledger_task(task: str, outcomes: dict) -> dict:
    if task not in CHOICE:
        return {"task": task, "skipped": "forecasting_choice_metrics_only"}
    datasets = load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl")
    with_graph = load_prediction_files(str(exp_dir(task, "")))
    without = load_prediction_files(str(exp_dir_graph_without(task)))
    mm = load_mismatch_indices("run1")[task]
    flips = flip_set(outcomes, task)

    rescued = []
    harmed = []
    delta_cells = Counter()
    for idx, sample in enumerate(datasets):
        gold = sample.get("output", "")
        cw = task_correct(task, gold, with_graph.get(idx))
        cwo = task_correct(task, gold, without.get(idx))
        if cw is None or cwo is None:
            continue
        delta_cells[f"{'R' if cwo else 'W'}->{'R' if cw else 'W'}"] += 1
        if not cwo and cw:
            rescued.append(idx)
        if cwo and not cw:
            harmed.append(idx)

    return {
        "task": task,
        "n": len(datasets),
        "delta_cells": dict(delta_cells),
        "graph_rescued_count": len(rescued),
        "graph_harmed_count": len(harmed),
        "rescued_and_flip": len(set(rescued) & flips),
        "rescued_and_mismatch_run1": len(set(rescued) & mm),
        "flip_total": len(flips),
        "rescued_sample_indices": rescued[:25],
        "harmed_sample_indices": harmed[:25],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes, _ = collect_sample_outcomes()
    rows = [ledger_task(t, outcomes) for t in CHOICE]
    payload = {"tasks": rows, "summary": {r["task"]: r.get("graph_rescued_count", 0) for r in rows}}
    path = OUT / "graph_ablation_sample_ledger.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
