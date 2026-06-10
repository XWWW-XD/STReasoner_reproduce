#!/usr/bin/env python3
"""E2: T0-1 strict reparse of three-run 6144 ST-Test (no new inference)."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_qa import (  # noqa: E402
    _normalize_choice,
    _parse_series,
    load_prediction_files,
)

from compute import (  # noqa: E402
    OUT,
    RUNS,
    TASKS,
    exp_dir,
    forecasting_mae,
    load_full_responses,
    load_jsonl,
    load_metrics,
    response_features,
    task_correct,
)
from compute_wave4 import (  # noqa: E402
    CHOICE_TASKS,
    extract_answer,
    triple_diverse_indices,
    triple_diverse_indices_loose,
)

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)


def strict_extract(full_response: str) -> tuple[str | None, bool]:
    """Return (extracted answer text, parse_ok). No fallback to full response."""
    if not full_response:
        return None, False
    m = ANSWER_TAG_RE.search(full_response)
    if not m:
        return None, False
    ans = m.group(1).strip().replace("```", "").strip()
    if not ans:
        return None, False
    return ans, True


def load_loose_preds(task: str, suffix: str) -> dict[int, str]:
    return load_prediction_files(str(exp_dir(task, suffix)))


def load_strict_preds(task: str, suffix: str) -> tuple[dict[int, str | None], dict[int, bool]]:
    full = load_full_responses(task, suffix)
    preds: dict[int, str | None] = {}
    ok: dict[int, bool] = {}
    for idx, text in full.items():
        extracted, parse_ok = strict_extract(text)
        ok[idx] = parse_ok
        preds[idx] = extracted if parse_ok else None
    return preds, ok


def acc_for_task(task: str, dataset: list[dict], preds: dict[int, str | None]) -> dict:
    if task == "forecasting":
        maes = []
        missing = 0
        for idx, sample in enumerate(dataset):
            pred = preds.get(idx)
            if pred is None:
                missing += 1
                continue
            mae = forecasting_mae(sample.get("output", ""), pred)
            if mae is None:
                missing += 1
            else:
                maes.append(mae)
        return {
            "evaluated": len(maes),
            "missing_or_fail": missing,
            "mae": sum(maes) / len(maes) if maes else None,
        }
    correct = 0
    evaluated = 0
    missing = 0
    for idx, sample in enumerate(dataset):
        pred = preds.get(idx)
        if pred is None:
            missing += 1
            continue
        evaluated += 1
        c = task_correct(task, sample.get("output", ""), pred)
        if c:
            correct += 1
    return {
        "evaluated": evaluated,
        "missing_or_fail": missing,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else None,
    }


def correctness_flip(task: str, dataset: list[dict], preds_by_run: dict[str, dict[int, str | None]]) -> int:
    n = len(dataset)
    flips = 0
    for idx in range(n):
        gold = dataset[idx].get("output", "")
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
                flips += 1
    return flips


def parse_fail_stats(task: str, ok_by_run: dict[str, dict[int, bool]]) -> dict:
    n = len(load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl"))
    per_run = {}
    all_three_fail = 0
    any_fail = 0
    for idx in range(n):
        fails = [not ok_by_run[rn].get(idx, False) for rn in RUNS]
        if any(fails):
            any_fail += 1
        if all(fails):
            all_three_fail += 1
    for rn in RUNS:
        fails = sum(1 for i in range(n) if not ok_by_run[rn].get(i, False))
        per_run[rn] = {
            "parse_fail": fails,
            "parse_ok": n - fails,
            "parse_fail_rate": round(fails / n, 4) if n else 0,
        }
    return {"n": n, "per_run": per_run, "any_run_fail": any_fail, "all_three_fail": all_three_fail}


def triple_diverse_from_preds(task: str, preds_by_run: dict[str, dict[int, str | None]], strict_letters: bool) -> dict:
    n = len(load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl"))
    count = 0
    indices: list[int] = []
    for idx in range(n):
        letters = []
        for rn in RUNS:
            pred = preds_by_run[rn].get(idx)
            if pred is None:
                break
            norm = _normalize_choice(pred)
            if strict_letters:
                if len(norm) != 1 or norm not in "ABCD":
                    break
            letters.append(norm)
        else:
            if len(set(letters)) == 3:
                count += 1
                indices.append(idx)
    return {"count": count, "indices": indices}


def overlap_with_no_tag(task: str, ok_by_run: dict[str, dict[int, bool]]) -> dict:
    """parse_fail vs answer_tag_count==0 on full response."""
    overlap = 0
    parse_fail_only = 0
    for rn in RUNS:
        full = load_full_responses(task, RUNS[rn])
        for idx, text in full.items():
            feats = response_features(text)
            failed = not ok_by_run[rn].get(idx, False)
            if failed and feats["answer_tag_count"] == 0:
                overlap += 1
            elif failed and feats["answer_tag_count"] > 0:
                parse_fail_only += 1
    return {"parse_fail_and_no_tag": overlap, "parse_fail_but_has_tag": parse_fail_only}


def main() -> None:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    loose_by_run: dict[str, dict[str, dict[int, str]]] = {t: {} for t in TASKS}
    strict_by_run: dict[str, dict[str, dict[int, str | None]]] = {t: {} for t in TASKS}
    ok_by_run: dict[str, dict[str, dict[int, bool]]] = {t: {} for t in TASKS}

    for task in TASKS:
        for rn, suffix in RUNS.items():
            loose_by_run[task][rn] = load_loose_preds(task, suffix)
            strict_by_run[task][rn], ok_by_run[task][rn] = load_strict_preds(task, suffix)

    official_metrics = {}
    loose_metrics = {}
    strict_metrics = {}
    for task in TASKS:
        official_metrics[task] = {rn: load_metrics(task, RUNS[rn]) for rn in RUNS}
        loose_metrics[task] = {
            rn: acc_for_task(task, datasets[task], loose_by_run[task][rn]) for rn in RUNS
        }
        strict_metrics[task] = {
            rn: acc_for_task(task, datasets[task], strict_by_run[task][rn]) for rn in RUNS
        }

    flip_comparison = {}
    triple_comparison = {}
    parse_fail = {}
    tag_overlap = {}

    for task in TASKS:
        parse_fail[task] = parse_fail_stats(task, ok_by_run[task])
        tag_overlap[task] = overlap_with_no_tag(task, ok_by_run[task])
        flip_comparison[task] = {
            "loose_flip": correctness_flip(task, datasets[task], loose_by_run[task]),
            "strict_flip": correctness_flip(
                task, datasets[task], strict_by_run[task]
            ),
        }
        if task in CHOICE_TASKS:
            triple_comparison[task] = {
                "wave4_loose_fn": len(triple_diverse_indices_loose(task)),
                "wave4_strict_fn": len(triple_diverse_indices(task)),
                "e2_loose_recomputed": triple_diverse_from_preds(
                    task, loose_by_run[task], strict_letters=False
                ),
                "e2_strict_recomputed": triple_diverse_from_preds(
                    task, strict_by_run[task], strict_letters=True
                ),
            }

    summary = {
        "experiment": "E2_T0-1_strict_reparse",
        "runs": list(RUNS.keys()),
        "tasks": TASKS,
        "official_metrics": official_metrics,
        "loose_reparse": loose_metrics,
        "strict_reparse": strict_metrics,
        "parse_fail": parse_fail,
        "parse_fail_vs_no_tag": tag_overlap,
        "flip_comparison": flip_comparison,
        "triple_diverse": triple_comparison,
        "falsifier_check": {
            "correlation_loose_triple_target_drop": "67 -> ~18",
            "correlation_e2_loose": triple_comparison.get("correlation", {}).get("e2_loose_recomputed", {}).get("count"),
            "correlation_e2_strict": triple_comparison.get("correlation", {}).get("e2_strict_recomputed", {}).get("count"),
            "entity_e2_loose": triple_comparison.get("entity", {}).get("e2_loose_recomputed", {}).get("count"),
            "entity_e2_strict": triple_comparison.get("entity", {}).get("e2_strict_recomputed", {}).get("count"),
        },
    }

    out_path = OUT / "e2_strict_reparse_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
