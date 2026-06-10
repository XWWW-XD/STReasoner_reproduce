#!/usr/bin/env python3
"""Wave 25: full ledger of strict-flip (real instability) samples."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, RUNS, load_jsonl, load_mismatch_indices, task_correct  # noqa: E402
from compute_e2_strict_reparse import load_loose_preds, load_strict_preds  # noqa: E402
from compute_t0_2 import flip_indices  # noqa: E402

CHOICE_TASKS = ["correlation", "entity", "etiological"]


def flip_pattern(task: str, idx: int, preds_by_run: dict, dataset: list[dict]) -> str | None:
    gold = dataset[idx].get("output", "")
    labels = []
    for rn in RUNS:
        pred = preds_by_run[rn].get(idx)
        if pred is None:
            return None
        c = task_correct(task, gold, pred)
        if c is None:
            return None
        labels.append("R" if c else "W")
    return "".join(labels) if len(labels) == 3 else None


def main() -> None:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in CHOICE_TASKS}
    mm = {run: load_mismatch_indices(run) for run in RUNS}

    loose_by = {t: {rn: load_loose_preds(t, s) for rn, s in RUNS.items()} for t in CHOICE_TASKS}
    strict_by = {t: {rn: load_strict_preds(t, s)[0] for rn, s in RUNS.items()} for t in CHOICE_TASKS}

    catalog = {}
    for task in CHOICE_TASKS:
        loose_f = flip_indices(task, datasets[task], loose_by[task])
        strict_f = flip_indices(task, datasets[task], strict_by[task])
        parser_only = sorted(loose_f - strict_f)
        real = sorted(strict_f)
        patterns = Counter()
        mismatch_any = 0
        mismatch_all3 = 0
        for idx in real:
            pat = flip_pattern(task, idx, strict_by[task], datasets[task])
            if pat:
                patterns[pat] += 1
            in_mm = [idx in mm[rn][task] for rn in RUNS]
            if all(in_mm):
                mismatch_all3 += 1
            if any(in_mm):
                mismatch_any += 1
        catalog[task] = {
            "loose_flip_count": len(loose_f),
            "strict_flip_count": len(strict_f),
            "parser_induced_count": len(parser_only),
            "real_instability_indices": real,
            "parser_induced_indices": parser_only,
            "flip_patterns_real": dict(sorted(patterns.items(), key=lambda x: -x[1])),
            "real_flip_mismatch_any_run": mismatch_any,
            "real_flip_mismatch_all3": mismatch_all3,
            "real_flip_mismatch_zero": len(real) - mismatch_any,
        }

    out_path = OUT / "real_instability_flip_ledger.json"
    out_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
