#!/usr/bin/env python3
"""Wave-4: triple-diverse catalog, mention cross, volatility rankings, excerpts."""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00_new_codes/scripts/mlp_encoder_focused_analysis"))

from evaluation.evaluate_qa import _normalize_choice  # noqa: E402

from compute import (  # noqa: E402
    OUT,
    RUNS,
    TASKS,
    collect_sample_outcomes,
    load_full_responses,
    load_jsonl,
    load_mismatch_indices,
    load_responses,
    task_correct,
)
from compute_deep import persistent_indices
from compute_wave3 import load_all_preds, load_all_responses
from count_ts_mentions import count_broad_ts_mentions, extract_think, iter_broad_mention_spans
from find_bad_cases import extract_numeric_reconstructions

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
CHOICE_TASKS = ["correlation", "entity", "etiological"]


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def normalized_answer(task: str, idx: int, run_name: str) -> tuple[str | None, bool]:
    """Return (choice letter or None, parser_failed)."""
    pred = load_all_preds()[task][run_name].get(idx)
    if not pred:
        return None, True
    ans = extract_answer(pred) or pred.strip()
    norm = _normalize_choice(ans)
    if len(norm) > 4:
        return None, True
    return norm, False


def triple_diverse_indices(task: str) -> list[int]:
    n = len(load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl"))
    out = []
    for idx in range(n):
        answers = []
        ok = True
        for rn in RUNS:
            ans, failed = normalized_answer(task, idx, rn)
            if failed or ans is None:
                ok = False
                break
            answers.append(ans)
        if ok and len(set(answers)) == 3:
            out.append(idx)
    return out


def flip_pattern(task: str, idx: int, outcomes: dict) -> str | None:
    key = f"{task}:{idx}"
    o = outcomes.get(key)
    if not o:
        return None
    labels = []
    for rn in RUNS:
        c = o[rn]["correct"]
        if c is None:
            return None
        labels.append("R" if c else "W")
    return "".join(labels) if len(labels) == 3 else None


def triple_diverse_indices_loose(task: str) -> list[int]:
    """Same as wave3: unique normalized extracted preds (may include multi-char)."""
    n = len(load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl"))
    out = []
    for idx in range(n):
        answers = []
        for rn in RUNS:
            p = load_all_preds()[task][rn].get(idx)
            if not p:
                break
            answers.append(_normalize_choice(p))
        else:
            if len(answers) == 3 and len(set(answers)) == 3:
                out.append(idx)
    return out


def build_triple_diverse_stats() -> dict:
    loose = {}
    strict = {}
    for task in ["correlation", "entity"]:
        loose_idxs = triple_diverse_indices_loose(task)
        strict_idxs = triple_diverse_indices(task)
        loose[task] = {"count": len(loose_idxs), "indices": loose_idxs[:20]}
        strict[task] = {"count": len(strict_idxs), "indices": strict_idxs}
    return {"loose_extracted_pred": loose, "strict_single_letter": strict}


def build_triple_diverse_catalog(outcomes: dict) -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in CHOICE_TASKS}
    resp = load_all_responses()
    mm = load_mismatch_indices("run1")
    catalog = {}
    for task in ["correlation", "entity"]:
        rows = []
        for idx in triple_diverse_indices_loose(task):
            sample = datasets[task][idx]
            gold = _normalize_choice(sample.get("output", ""))
            run_answers = {}
            run_correct = {}
            parser_failed = {}
            think_lens = {}
            for rn in RUNS:
                ans, failed = normalized_answer(task, idx, rn)
                run_answers[rn] = ans
                parser_failed[rn] = failed
                pred = load_all_preds()[task][rn].get(idx)
                run_correct[rn] = task_correct(task, sample.get("output", ""), pred)
                think_lens[rn] = len(extract_think(resp[task][rn].get(idx, "")))
            rows.append(
                {
                    "idx": idx,
                    "gold": gold,
                    "answers": run_answers,
                    "parser_failed": parser_failed,
                    "correct": run_correct,
                    "flip_pattern": flip_pattern(task, idx, outcomes),
                    "mismatch_run1": idx in mm[task],
                    "think_len": think_lens,
                    "any_run_correct": any(run_correct[r] for r in RUNS if run_correct[r] is not None),
                    "runs_correct_count": sum(1 for r in RUNS if run_correct[r]),
                }
            )
        pattern_counter = Counter(r["flip_pattern"] for r in rows if r["flip_pattern"])
        catalog[task] = {
            "count": len(rows),
            "flip_patterns": dict(pattern_counter),
            "mismatch_run1_count": sum(1 for r in rows if r["mismatch_run1"]),
            "any_run_correct_count": sum(1 for r in rows if r["any_run_correct"]),
            "samples": rows,
        }
    return catalog


def build_triple_diverse_cross(outcomes: dict) -> dict:
    """Cross triple-diverse (loose) with flip and mismatch."""
    flip_sets = {t: set() for t in CHOICE_TASKS}
    for key, o in outcomes.items():
        if o["task"] not in CHOICE_TASKS:
            continue
        cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
        if len(cor) >= 2 and len(set(cor)) > 1:
            flip_sets[o["task"]].add(o["idx"])
    result = {"stats": build_triple_diverse_stats()}
    for task in ["correlation", "entity"]:
        triple = set(triple_diverse_indices_loose(task))
        strict = set(triple_diverse_indices(task))
        flip = flip_sets[task]
        mm = load_mismatch_indices("run1")[task]
        result[task] = {
            "triple_diverse_loose": len(triple),
            "triple_diverse_strict_single_letter": len(strict),
            "flip_total": len(flip),
            "triple_loose_and_flip": len(triple & flip),
            "triple_loose_not_flip": len(triple - flip),
            "flip_not_triple_loose": len(flip - triple),
            "triple_loose_and_mismatch_run1": len(triple & mm),
            "parser_anomaly_only": len(triple - strict),
        }
    return result


def build_mention_strict_crosstab() -> dict:
    """Run1: broad mention vs strict mismatch sample 2x2."""
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in ["forecasting", "correlation"]}
    resp = load_all_responses()
    mm = load_mismatch_indices("run1")
    result = {}
    for task in ["forecasting", "correlation"]:
        cells = Counter()
        jaccard_rows = []
        for idx, sample in enumerate(datasets[task]):
            think = extract_think(resp[task]["run1"].get(idx, ""))
            broad_n = count_broad_ts_mentions(think)
            strict_wins = {
                (c["node"], tuple(c["stated_window"]))
                for c in extract_numeric_reconstructions(sample, resp[task]["run1"].get(idx, ""), True)
            }
            in_mm = idx in mm[task]
            has_broad = broad_n > 0
            has_strict = len(strict_wins) > 0
            if has_broad and in_mm:
                cells["broad_and_mismatch"] += 1
            elif has_broad and not in_mm:
                cells["broad_no_mismatch"] += 1
            elif not has_broad and in_mm:
                cells["no_broad_mismatch"] += 1
            else:
                cells["neither"] += 1
            if in_mm and has_broad:
                # proxy: strict window count vs broad span count
                jaccard_rows.append(min(len(strict_wins), broad_n) / max(len(strict_wins), broad_n, 1))
        result[task] = {
            "cells": dict(cells),
            "total": len(datasets[task]),
            "mention_mismatch_overlap_ratio": (
                round(cells["broad_and_mismatch"] / len(mm[task]), 4) if mm[task] else None
            ),
            "proxy_span_jaccard_median": round(statistics.median(jaccard_rows), 4) if jaccard_rows else None,
        }
    return result


def build_drift_vs_fill_excerpts() -> dict:
    classes_path = OUT / "persistent_core_classification.json"
    data = json.loads(classes_path.read_text(encoding="utf-8"))
    datasets = load_jsonl(ROOT / "data/ST-Bench/ST-Test/correlation_test.jsonl")
    resp = load_all_responses()
    picks = {"constant_fill": [], "local_drift": []}
    for s in data["correlation"]["samples"]:
        cls = s["primary_class"]
        if cls not in picks or len(picks[cls]) >= 5:
            continue
        idx = s["idx"]
        sample = datasets[idx]
        checks = extract_numeric_reconstructions(sample, resp["correlation"]["run1"].get(idx, ""), True)
        line = checks[0]["reasoning_line"][:200] if checks else "(no strict line)"
        picks[cls].append(
            {
                "idx": idx,
                "max_diff": s.get("max_diff"),
                "reasoning_line": line,
                "gold": _normalize_choice(sample.get("output", "")),
                "answers": {
                    rn: (normalized_answer("correlation", s["idx"], rn)[0] or "?")
                    for rn in RUNS
                },
            }
        )
    return picks


def build_forecasting_volatile_pairs() -> dict:
    outcomes, _ = collect_sample_outcomes()
    persistent = persistent_indices("forecasting")
    rows = []
    for key, o in outcomes.items():
        if o["task"] != "forecasting":
            continue
        maes = {rn: o[rn]["mae"] for rn in RUNS if o[rn]["mae"] is not None}
        if len(maes) < 3:
            continue
        vals = list(maes.values())
        rows.append(
            {
                "idx": o["idx"],
                "mae_range": round(max(vals) - min(vals), 2),
                "mae_run1": round(maes["run1"], 2),
                "mae_run2": round(maes["run2"], 2),
                "mae_run3": round(maes["run3"], 2),
                "persistent": o["idx"] in persistent,
            }
        )
    rows.sort(key=lambda r: -r["mae_range"])
    idx19 = next(r for r in rows if r["idx"] == 19)
    idx147 = next(r for r in rows if r["idx"] == 147)
    return {
        "top10_mae_range": rows[:10],
        "persistent_in_top10": sum(1 for r in rows[:10] if r["persistent"]),
        "pair_compare": {"persistent_hard_19": idx19, "volatile_non_persistent_147": idx147},
        "persistent_top10": [r for r in rows if r["persistent"]][:10],
    }


def export_volatility_rankings(outcomes: dict) -> None:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    path = OUT / "volatility_rankings.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "idx",
                "volatility_kind",
                "score",
                "run1_metric",
                "run2_metric",
                "run3_metric",
                "persistent_mismatch_all3",
                "mismatch_run1",
            ]
        )
        persistent = {t: persistent_indices(t) for t in ["forecasting", "correlation"]}
        mm1 = load_mismatch_indices("run1")
        for task in CHOICE_TASKS:
            for idx in range(len(datasets[task])):
                key = f"{task}:{idx}"
                o = outcomes[key]
                cor = [1 if o[r]["correct"] else 0 for r in RUNS if o[r]["correct"] is not None]
                if len(cor) == 3:
                    score = statistics.pstdev(cor)
                    if score > 0:
                        w.writerow(
                            [
                                task,
                                idx,
                                "correctness_std",
                                round(score, 4),
                                cor[0],
                                cor[1],
                                cor[2],
                                "",
                                idx in mm1[task],
                            ]
                        )
        for key, o in outcomes.items():
            if o["task"] != "forecasting":
                continue
            maes = [o[r]["mae"] for r in RUNS if o[r]["mae"] is not None]
            if len(maes) < 3:
                continue
            score = max(maes) - min(maes)
            if score > 0.5:
                idx = o["idx"]
                w.writerow(
                    [
                        "forecasting",
                        idx,
                        "mae_range",
                        round(score, 2),
                        round(maes[0], 2),
                        round(maes[1], 2),
                        round(maes[2], 2),
                        idx in persistent["forecasting"],
                        idx in mm1["forecasting"],
                    ]
                )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    outcomes, _ = collect_sample_outcomes()
    load_all_responses()
    load_all_preds()
    payloads = {
        "triple_diverse_stats.json": build_triple_diverse_stats(),
        "triple_diverse_catalog.json": build_triple_diverse_catalog(outcomes),
        "triple_diverse_cross.json": build_triple_diverse_cross(outcomes),
        "mention_strict_crosstab.json": build_mention_strict_crosstab(),
        "drift_vs_fill_excerpts.json": build_drift_vs_fill_excerpts(),
        "forecasting_volatile_pairs.json": build_forecasting_volatile_pairs(),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")
    export_volatility_rankings(outcomes)
    print(f"wrote {OUT / 'volatility_rankings.csv'}")


if __name__ == "__main__":
    main()
