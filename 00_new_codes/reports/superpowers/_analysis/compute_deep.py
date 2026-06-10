#!/usr/bin/env python3
"""Phase-2 deep exploration for superpowers study. Writes only to superpowers/artifacts/."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00_new_codes/scripts/mlp_encoder_focused_analysis"))

from find_bad_cases import (  # noqa: E402
    MISMATCH_TOLERANCE,
    extract_numeric_reconstructions,
)
from compute import (  # noqa: E402
    ARTIFACT_MLP,
    OUT,
    RUNS,
    TASKS,
    collect_sample_outcomes,
    exp_dir,
    forecasting_mae,
    jaccard,
    load_full_responses,
    load_jsonl,
    load_mismatch_indices,
    load_responses,
    task_correct,
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
LOOSE_NODE_NUMERIC_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?Node\s+(\d+)\b[^\n]{0,80}?(\d[\d.,\s\-+]{6,})",
    re.I,
)


def persistent_indices(task: str) -> set[int]:
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    return mm["run1"][task] & mm["run2"][task] & mm["run3"][task]


def classify_window(check: dict) -> str:
    stated = check["model_stated_values"]
    raw = check["raw_values"]
    if len(stated) < 3 or len(raw) < 3:
        return "too_short"
    max_diff = check["max_abs_diff_between_stated_and_raw"]
    mean_diff = check["mean_abs_diff_between_stated_and_raw"]
    st_abs = [abs(x) for x in stated]
    raw_abs = [abs(x) for x in raw]
    st_mean = statistics.mean(st_abs) or 1e-6
    raw_mean = statistics.mean(raw_abs) or 1e-6
    st_std = statistics.pstdev(stated) if len(stated) > 1 else 0.0
    raw_std = statistics.pstdev(raw) if len(raw) > 1 else 0.0
    unique_st = len({round(x, 1) for x in stated})

    if unique_st <= 3 and raw_std > 5 and max_diff > 100:
        return "constant_fill"
    if st_std / st_mean < 0.03 and raw_std / (statistics.mean(raw_abs) + 1e-6) > 0.15 and max_diff > 20:
        return "constant_fill"
    ratio = st_mean / raw_mean
    if ratio > 2.5 or ratio < 0.35:
        return "magnitude_offset"
    if any(abs(s) > max(raw_abs) * 3 + 50 for s in stated):
        return "spike_hallucination"
    if mean_diff < 25 and max_diff < 80:
        return "local_drift"
    return "other_large_error"


def worst_checks_for_sample(sample: dict, responses: dict[int, str]) -> list[dict]:
    checks = []
    for run_name, text in responses.items():
        for check in extract_numeric_reconstructions(sample, text, mismatch_only=True):
            check = dict(check)
            check["run"] = run_name
            checks.append(check)
    return checks


def build_persistent_core_classification() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    result = {}
    for task in ["forecasting", "correlation"]:
        idxs = sorted(persistent_indices(task))
        per_sample = []
        window_classes = Counter()
        severity_buckets = Counter()
        for idx in idxs:
            sample = datasets[task][idx]
            by_run = {rn: load_full_responses(task, suf).get(idx, "") for rn, suf in RUNS.items()}
            checks = worst_checks_for_sample(sample, by_run)
            if not checks:
                per_sample.append({"idx": idx, "primary_class": "no_strict_window", "max_diff": None})
                window_classes["no_strict_window"] += 1
                continue
            worst = max(checks, key=lambda c: c["max_abs_diff_between_stated_and_raw"])
            primary = classify_window(worst)
            per_sample.append(
                {
                    "idx": idx,
                    "primary_class": primary,
                    "max_diff": round(worst["max_abs_diff_between_stated_and_raw"], 2),
                    "worst_run": worst["run"],
                    "node": worst["node"],
                    "window": worst["stated_window"],
                }
            )
            window_classes[primary] += 1
            md = worst["max_abs_diff_between_stated_and_raw"]
            if md > 500:
                severity_buckets["catastrophic_gt500"] += 1
            elif md > 100:
                severity_buckets["large_100_500"] += 1
            else:
                severity_buckets["moderate_le100"] += 1
        result[task] = {
            "persistent_count": len(idxs),
            "window_class_counts": dict(window_classes),
            "severity_buckets": dict(severity_buckets),
            "top_examples": sorted(
                [s for s in per_sample if s.get("max_diff")],
                key=lambda s: -s["max_diff"],
            )[:10],
            "samples": per_sample,
        }
    return result


def build_forecasting_mae_volatility(outcomes: dict[str, dict]) -> dict:
    persistent = persistent_indices("forecasting")
    mm_any = set()
    for run in RUNS:
        mm_any |= load_mismatch_indices(run)["forecasting"]
    buckets = {
        "persistent_all3": {"n": 0, "mae_range_gt1": 0, "mae_std_median": []},
        "mismatch_not_persistent": {"n": 0, "mae_range_gt1": 0, "mae_std_median": []},
        "never_mismatch": {"n": 0, "mae_range_gt1": 0, "mae_std_median": []},
    }
    for key, o in outcomes.items():
        if o["task"] != "forecasting":
            continue
        maes = [o[r]["mae"] for r in RUNS if o[r]["mae"] is not None]
        if len(maes) < 2:
            continue
        idx = o["idx"]
        mrange = max(maes) - min(maes)
        mstd = statistics.pstdev(maes) if len(maes) > 1 else 0.0
        if idx in persistent:
            label = "persistent_all3"
        elif idx in mm_any:
            label = "mismatch_not_persistent"
        else:
            label = "never_mismatch"
        b = buckets[label]
        b["n"] += 1
        if mrange > 1.0:
            b["mae_range_gt1"] += 1
        b["mae_std_median"].append(mstd)
    for label, b in buckets.items():
        vals = b.pop("mae_std_median")
        b["mae_std_median"] = round(statistics.median(vals), 3) if vals else None
        b["mae_range_gt1_ratio"] = round(b["mae_range_gt1"] / b["n"], 4) if b["n"] else None
    return buckets


def build_loose_numeric_mentions() -> dict:
    """Entity/etiological: broad Node+numeric lines in thinking (run1)."""
    result = {}
    for task in ["entity", "etiological"]:
        data = load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl")
        full = load_full_responses(task, "")
        strict_mm = load_mismatch_indices("run1")[task]
        counters = Counter()
        for idx, sample in enumerate(data):
            think = THINK_RE.search(full.get(idx, "") or "")
            think_text = think.group(1) if think else ""
            strict_checks = extract_numeric_reconstructions(sample, full.get(idx, ""), mismatch_only=False)
            strict_mismatch = extract_numeric_reconstructions(sample, full.get(idx, ""), mismatch_only=True)
            loose = LOOSE_NODE_NUMERIC_RE.findall(think_text)
            has_loose = len(loose) > 0
            has_strict_line = len(strict_checks) > 0
            if has_strict_line and strict_mismatch:
                counters["strict_mismatch"] += 1
            elif has_strict_line:
                counters["strict_line_ok"] += 1
            elif has_loose:
                counters["loose_only_numeric"] += 1
            else:
                counters["no_numeric_narration"] += 1
        result[task] = {
            "total": len(data),
            "counts": dict(counters),
            "strict_mismatch_indices": sorted(strict_mm),
        }
    return result


def build_persistent_answer_stability(outcomes: dict[str, dict]) -> dict:
    """For persistent correlation mismatch: do final answers stay stable?"""
    result = {}
    for task in ["correlation", "forecasting"]:
        idxs = persistent_indices(task)
        same_answer = diff_answer = 0
        same_correct = diff_correct = 0
        for idx in idxs:
            key = f"{task}:{idx}"
            o = outcomes[key]
            preds = [load_responses(task, RUNS[r]).get(idx) for r in RUNS]
            preds = [p for p in preds if p is not None]
            if len(set(preds)) == 1:
                same_answer += 1
            elif len(preds) >= 2:
                diff_answer += 1
            cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
            if len(cor) == 3 and len(set(cor)) == 1:
                same_correct += 1
            elif len(cor) >= 2 and len(set(cor)) > 1:
                diff_correct += 1
        result[task] = {
            "persistent_n": len(idxs),
            "same_parsed_answer_across_runs": same_answer,
            "diff_parsed_answer_across_runs": diff_answer,
            "same_correctness_label": same_correct,
            "diff_correctness_label": diff_correct,
        }
    return result


def build_cross_run_window_jaccard() -> dict:
    """Among persistent idx, how stable are mismatch *window sets* across runs?"""
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    result = {}
    for task in ["forecasting", "correlation"]:
        idxs = persistent_indices(task)
        jaccards = []
        exact_match = 0
        for idx in idxs:
            sample = datasets[task][idx]
            sets = []
            for rn, suf in RUNS.items():
                resp = load_full_responses(task, suf).get(idx, "")
                wins = {
                    (c["node"], tuple(c["stated_window"]))
                    for c in extract_numeric_reconstructions(sample, resp, mismatch_only=True)
                }
                sets.append(wins)
            j12 = jaccard(sets[0], sets[1])
            j13 = jaccard(sets[0], sets[2])
            j23 = jaccard(sets[1], sets[2])
            jaccards.append((j12 + j13 + j23) / 3)
            if sets[0] == sets[1] == sets[2]:
                exact_match += 1
        result[task] = {
            "persistent_n": len(idxs),
            "mean_pairwise_window_jaccard": round(statistics.mean(jaccards), 4) if jaccards else None,
            "exact_same_mismatch_windows_all3": exact_match,
            "exact_same_ratio": round(exact_match / len(idxs), 4) if idxs else None,
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes, _ = collect_sample_outcomes()
    payloads = {
        "persistent_core_classification.json": build_persistent_core_classification(),
        "forecasting_mae_volatility.json": build_forecasting_mae_volatility(outcomes),
        "loose_numeric_mentions.json": build_loose_numeric_mentions(),
        "persistent_answer_stability.json": build_persistent_answer_stability(outcomes),
        "cross_run_window_jaccard.json": build_cross_run_window_jaccard(),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
