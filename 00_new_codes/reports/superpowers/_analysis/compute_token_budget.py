#!/usr/bin/env python3
"""Wave9: 512 vs 6144 paired comparison."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import (  # noqa: E402
    OUT,
    TASKS,
    exp_dir,
    exp_dir_512,
    load_jsonl,
    load_responses,
    task_correct,
)
from response_diag import diagnose_response  # noqa: E402

CHOICE = ["correlation", "entity", "etiological"]


def load_full_ga(task: str, base: Path) -> dict[int, str]:
    p = base / "generated_answer.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {x["idx"]: x.get("response", "") for x in data}


def compare_task(task: str) -> dict:
    datasets = load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl")
    p512 = exp_dir_512(task)
    p6144 = exp_dir(task, "")
    pred512 = load_responses(task, "") if task in CHOICE else {}
    # 512 uses different dir
    from evaluation.evaluate_qa import load_prediction_files

    pred512 = load_prediction_files(str(p512))
    pred6144 = load_prediction_files(str(p6144))
    resp512 = load_full_ga(task, p512)
    resp6144 = load_full_ga(task, p6144)

    cells = Counter()
    no_tag = {"512": 0, "6144": 0}
    rows = []
    for idx, sample in enumerate(datasets):
        gold = sample.get("output", "")
        c512 = task_correct(task, gold, pred512.get(idx))
        c6144 = task_correct(task, gold, pred6144.get(idx))
        if c512 is None or c6144 is None:
            continue
        key = ("R" if c512 else "W") + ("R" if c6144 else "W")
        cells[key] += 1
        d512 = diagnose_response(resp512.get(idx, ""))
        d6144 = diagnose_response(resp6144.get(idx, ""))
        if d512["answer_tag_count"] == 0:
            no_tag["512"] += 1
        if d6144["answer_tag_count"] == 0:
            no_tag["6144"] += 1
        if c512 != c6144:
            rows.append({"idx": idx, "cell": key, "512_root": d512["root_cause"], "6144_root": d6144["root_cause"]})

    m512 = json.loads((p512 / "evaluation_metrics.json").read_text(encoding="utf-8"))
    m6144 = json.loads((p6144 / "evaluation_metrics.json").read_text(encoding="utf-8"))
    return {
        "task": task,
        "n": len(datasets),
        "metrics_512": {"accuracy": m512.get("accuracy"), "mae": m512.get("mae")},
        "metrics_6144": {"accuracy": m6144.get("accuracy"), "mae": m6144.get("mae")},
        "correctness_cells": dict(cells),
        "only_512_correct": cells["WR"],
        "only_6144_correct": cells["RW"],
        "both_wrong": cells["WW"],
        "both_right": cells["RR"],
        "no_answer_tag_rate": {k: round(v / len(datasets), 4) for k, v in no_tag.items()},
        "flip_indices_sample": rows[:30],
        "flip_count": len(rows),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {t: compare_task(t) for t in TASKS}
    path = OUT / "token_budget_512_vs_6144.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
