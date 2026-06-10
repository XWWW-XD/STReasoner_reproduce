#!/usr/bin/env python3
"""Wave-5: strict excerpts, parser taxonomy, volatility top labels."""
from __future__ import annotations

import csv
import json
import re
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
    collect_sample_outcomes,
    load_jsonl,
    load_mismatch_indices,
    response_features,
    task_correct,
)
from compute_deep import persistent_indices
from compute_wave4 import (
    ANSWER_RE,
    CHOICE_TASKS,
    extract_answer,
    flip_pattern,
    normalized_answer,
    triple_diverse_indices,
    triple_diverse_indices_loose,
)
from compute_wave3 import load_all_preds, load_all_responses
from count_ts_mentions import extract_think

THINK_TAIL = 500
ANSWER_SNIP = 120


def classify_run_parser(full_response: str, pred: str | None) -> str:
    feats = response_features(full_response or "")
    tag_count = feats["answer_tag_count"]
    if tag_count == 0:
        return "no_answer_tag"
    if tag_count > 1:
        return "multiple_answer_tags"
    ans = extract_answer(full_response or "")
    if ans is None or not ans.strip():
        return "empty_answer_tag"
    norm = _normalize_choice(ans)
    if len(norm) > 4:
        return "extracted_too_long"
    if norm and len(norm) == 1 and norm in "ABCD":
        return "ok_single_letter"
    if pred:
        pnorm = _normalize_choice(pred)
        if len(pnorm) > 4:
            return "pred_string_too_long"
        if pnorm in "ABCD" and len(pnorm) == 1:
            return "ok_from_pred_not_tag"
    return "non_single_letter"


def build_strict_excerpts(outcomes: dict) -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in CHOICE_TASKS}
    resp = load_all_responses()
    preds = load_all_preds()
    mm = load_mismatch_indices("run1")
    result = {}
    for task in ["correlation", "entity"]:
        rows = []
        for idx in triple_diverse_indices(task):
            sample = datasets[task][idx]
            gold = _normalize_choice(sample.get("output", ""))
            per_run = {}
            for rn in RUNS:
                full = resp[task][rn].get(idx, "")
                think = extract_think(full)
                tail = think[-THINK_TAIL:] if think else ""
                ans, failed = normalized_answer(task, idx, rn)
                pred = preds[task][rn].get(idx, "")
                tag_m = ANSWER_RE.search(full or "")
                answer_snip = (tag_m.group(0)[:ANSWER_SNIP] if tag_m else (pred or "")[:ANSWER_SNIP])
                per_run[rn] = {
                    "answer": ans,
                    "correct": task_correct(task, sample.get("output", ""), pred),
                    "parser_class": classify_run_parser(full, pred),
                    "think_tail": tail,
                    "answer_snippet": answer_snip,
                }
            rows.append(
                {
                    "idx": idx,
                    "gold": gold,
                    "flip_pattern": flip_pattern(task, idx, outcomes),
                    "mismatch_run1": idx in mm[task],
                    "runs": per_run,
                    "all_wrong": all(
                        per_run[r]["correct"] is False for r in RUNS if per_run[r]["correct"] is not None
                    ),
                    "any_correct": any(per_run[r]["correct"] for r in RUNS if per_run[r]["correct"]),
                }
            )
        result[task] = {"count": len(rows), "samples": rows}
    return result


def build_parser_anomaly_taxonomy() -> dict:
    """Loose triple-diverse but not strict: per-run parser classes + sample-level bucket."""
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in CHOICE_TASKS}
    resp = load_all_responses()
    preds = load_all_preds()
    summary = {"correlation": {}, "entity": {}}
    for task in ["correlation", "entity"]:
        loose = set(triple_diverse_indices_loose(task))
        strict = set(triple_diverse_indices(task))
        anomaly_idxs = sorted(loose - strict)
        per_run_counter = Counter()
        sample_buckets = Counter()
        rows = []
        for idx in anomaly_idxs:
            run_classes = {}
            failed_runs = []
            for rn in RUNS:
                full = resp[task][rn].get(idx, "")
                pred = preds[task][rn].get(idx, "")
                cls = classify_run_parser(full, pred)
                run_classes[rn] = cls
                per_run_counter[cls] += 1
                if cls != "ok_single_letter":
                    failed_runs.append(rn)
            n_fail = len(failed_runs)
            if n_fail == 3:
                bucket = "all_runs_parser_issue"
            elif n_fail == 2:
                bucket = "two_runs_parser_issue"
            elif n_fail == 1:
                bucket = "one_run_parser_issue"
            else:
                bucket = "all_ok_but_not_three_letters"
            sample_buckets[bucket] += 1
            rows.append(
                {
                    "idx": idx,
                    "bucket": bucket,
                    "failed_runs": failed_runs,
                    "run_parser_class": run_classes,
                    "loose_preds_norm": {
                        rn: _normalize_choice(preds[task][rn].get(idx, "") or "")[:40]
                        for rn in RUNS
                    },
                }
            )
        summary[task] = {
            "count": len(anomaly_idxs),
            "sample_buckets": dict(sample_buckets),
            "per_run_parser_class_totals": dict(per_run_counter),
            "samples": rows,
        }
    return summary


def build_strict_aggregate(outcomes: dict) -> dict:
    """Summary stats on strict triple-diverse set."""
    mm = load_mismatch_indices("run1")
    flip_sets = defaultdict(set)
    for key, o in outcomes.items():
        if o["task"] not in CHOICE_TASKS:
            continue
        cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
        if len(cor) >= 2 and len(set(cor)) > 1:
            flip_sets[o["task"]].add(o["idx"])
    agg = {}
    for task in ["correlation", "entity"]:
        idxs = set(triple_diverse_indices(task))
        patterns = Counter()
        all_wrong = 0
        any_correct = 0
        for idx in idxs:
            fp = flip_pattern(task, idx, outcomes)
            if fp:
                patterns[fp] += 1
            key = f"{task}:{idx}"
            o = outcomes[key]
            cor = [o[r]["correct"] for r in RUNS]
            if all(c is False for c in cor if c is not None):
                all_wrong += 1
            if any(c for c in cor if c):
                any_correct += 1
        agg[task] = {
            "count": len(idxs),
            "flip_intersection": len(idxs & flip_sets[task]),
            "mismatch_run1": len(idxs & mm[task]),
            "flip_patterns": dict(patterns),
            "all_three_wrong": all_wrong,
            "any_run_correct": any_correct,
        }
    return agg


def auto_tag_row(
    task: str,
    idx: int,
    kind: str,
    outcomes: dict,
    strict_loose: dict,
) -> str:
    key = f"{task}:{idx}"
    o = outcomes[key]
    if kind == "mae_range":
        if idx in persistent_indices("forecasting"):
            return "persistent_mismatch_mae_swing"
        if idx in load_mismatch_indices("run1")["forecasting"]:
            return "mismatch_not_persistent"
        return "high_mae_non_persistent"
    # correctness_std
    if task in CHOICE_TASKS:
        if idx in strict_loose["strict"].get(task, set()):
            return "strict_triple_diverse"
        if idx in strict_loose["loose"].get(task, set()):
            return "loose_triple_parser_noise"
    fp = flip_pattern(task, idx, outcomes)
    if fp in ("WRW", "RWR"):
        return "two_run_flip_symmetric"
    if fp == "WWW":
        return "always_wrong"
    if idx in load_mismatch_indices("run1")[task]:
        return "flip_with_mismatch"
    return "flip_no_mismatch"


def build_volatility_top_labels(outcomes: dict) -> dict:
    csv_path = OUT / "volatility_rankings.csv"
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    strict_loose = {
        "strict": {t: set(triple_diverse_indices(t)) for t in ["correlation", "entity"]},
        "loose": {t: set(triple_diverse_indices_loose(t)) for t in ["correlation", "entity"]},
    }
    choice_rows = [r for r in rows if r["volatility_kind"] == "correctness_std"]
    mae_rows = [r for r in rows if r["volatility_kind"] == "mae_range"]
    choice_rows.sort(key=lambda r: -float(r["score"]))
    mae_rows.sort(key=lambda r: -float(r["score"]))

    def enrich(r: dict) -> dict:
        task, idx_s = r["task"], int(r["idx"])
        tag = auto_tag_row(task, idx_s, r["volatility_kind"], outcomes, strict_loose)
        extra = {"auto_tag": tag, "flip_pattern": flip_pattern(task, idx_s, outcomes) if task in CHOICE_TASKS else None}
        if task in CHOICE_TASKS:
            extra["strict_triple"] = idx_s in strict_loose["strict"][task]
            extra["loose_triple"] = idx_s in strict_loose["loose"][task]
        return {**r, **extra}

    top_choice = [enrich(r) for r in choice_rows[:30]]
    top_mae = [enrich(r) for r in mae_rows[:20]]
    tag_counts = Counter(r["auto_tag"] for r in top_choice + top_mae)
    return {
        "top30_correctness_std": top_choice,
        "top20_mae_range": top_mae,
        "auto_tag_counts_in_top50": dict(tag_counts),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    outcomes, _ = collect_sample_outcomes()
    load_all_responses()
    load_all_preds()
    payloads = {
        "strict_triple_excerpts.json": build_strict_excerpts(outcomes),
        "parser_anomaly_taxonomy.json": build_parser_anomaly_taxonomy(),
        "strict_triple_aggregate.json": build_strict_aggregate(outcomes),
        "volatility_top50_labels.json": build_volatility_top_labels(outcomes),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
