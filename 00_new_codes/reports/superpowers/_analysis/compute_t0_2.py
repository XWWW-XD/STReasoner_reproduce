#!/usr/bin/env python3
"""T0-2: parse_ok/parse_fail dashboard + flip attribution (CPU, post-E2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, RUNS, TASKS, load_jsonl, task_correct  # noqa: E402
from compute_e2_strict_reparse import (  # noqa: E402
    load_loose_preds,
    load_strict_preds,
)

CHOICE_TASKS = ["correlation", "entity", "etiological"]


def flip_indices(task: str, dataset: list[dict], preds_by_run: dict[str, dict]) -> set[int]:
    flips: set[int] = set()
    for idx, sample in enumerate(dataset):
        gold = sample.get("output", "")
        labels = []
        for rn in RUNS:
            pred = preds_by_run[rn].get(idx)
            if pred is None:
                break
            c = task_correct(task, gold, pred)
            if c is None:
                break
            labels.append(c)
        else:
            if len(set(labels)) > 1:
                flips.add(idx)
    return flips


def build_dashboard(e2: dict) -> list[dict]:
    rows = []
    for task in TASKS:
        for rn in RUNS:
            off = e2["official_metrics"][task][rn]
            loose = e2["loose_reparse"][task][rn]
            strict = e2["strict_reparse"][task][rn]
            pf = e2["parse_fail"][task]["per_run"][rn]
            row = {
                "task": task,
                "run": rn,
                "parse_fail": pf["parse_fail"],
                "parse_ok": pf["parse_ok"],
                "parse_fail_rate": pf["parse_fail_rate"],
            }
            if task == "forecasting":
                row["mae_official"] = off.get("mae")
                row["mae_loose"] = loose.get("mae")
                row["mae_strict_parse_ok"] = strict.get("mae")
                row["evaluated_official"] = off.get("evaluated_samples")
                row["evaluated_strict"] = strict.get("evaluated")
            else:
                row["acc_official"] = off.get("accuracy")
                row["acc_loose"] = loose.get("accuracy")
                row["acc_strict_parse_ok"] = strict.get("accuracy")
                row["evaluated_official"] = off.get("evaluated_samples")
                row["evaluated_strict"] = strict.get("evaluated")
            rows.append(row)
    return rows


def main() -> None:
    e2_path = OUT / "e2_strict_reparse_summary.json"
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}

    loose_by_run = {t: {} for t in TASKS}
    strict_by_run = {t: {} for t in TASKS}
    for task in TASKS:
        for rn, suffix in RUNS.items():
            loose_by_run[task][rn] = load_loose_preds(task, suffix)
            strict_by_run[task][rn], _ = load_strict_preds(task, suffix)

    flip_attr = {}
    for task in CHOICE_TASKS:
        loose_f = flip_indices(task, datasets[task], loose_by_run[task])
        strict_f = flip_indices(task, datasets[task], strict_by_run[task])
        parser_only = sorted(loose_f - strict_f)
        real = sorted(strict_f)
        flip_attr[task] = {
            "loose_flip_count": len(loose_f),
            "strict_flip_count": len(strict_f),
            "parser_induced_flip_count": len(parser_only),
            "parser_induced_share": round(len(parser_only) / len(loose_f), 4) if loose_f else 0,
            "real_instability_flip_count": len(real),
            "parser_induced_indices_sample": parser_only[:30],
            "real_instability_indices_sample": real[:30],
        }

    unclosed_path = OUT / "unclosed_thinking_forensics.json"
    unclosed_overlap = {}
    if unclosed_path.exists():
        unc = json.loads(unclosed_path.read_text(encoding="utf-8"))
        for task in ["correlation", "entity"]:
            fs = unc.get("focus_samples", {}).get(task, {})
            if "samples" in fs:
                focus = {s["idx"] for s in fs["samples"]}
            else:
                focus = {int(k) for k in fs.keys()} if isinstance(fs, dict) else set()
            ok_maps = {}
            for rn, suffix in RUNS.items():
                _, ok_maps[rn] = load_strict_preds(task, suffix)
            pf_any = {
                idx
                for idx in range(len(datasets[task]))
                if any(not ok_maps[rn].get(idx, False) for rn in RUNS)
            }
            unclosed_overlap[task] = {
                "unclosed_focus_count": len(focus),
                "parse_fail_any_run_count": e2["parse_fail"][task]["any_run_fail"],
                "focus_in_parse_fail": len(focus & pf_any),
                "focus_subset_of_parse_fail": focus <= pf_any if focus else None,
            }

    out = {
        "experiment": "T0_2_parse_dashboard_flip_attribution",
        "dashboard": build_dashboard(e2),
        "flip_attribution": flip_attr,
        "unclosed_vs_parse_fail": unclosed_overlap,
    }
    out_path = OUT / "t0_2_parse_dashboard.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
