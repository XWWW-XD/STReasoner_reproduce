#!/usr/bin/env python3
"""Report-only analysis for superpowers study. Does not modify exp or tools."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00_new_codes/scripts/mlp_encoder_focused_analysis"))

from evaluation.evaluate_qa import (  # noqa: E402
    _normalize_choice,
    _parse_series,
    load_prediction_files,
)

TASKS = ["forecasting", "correlation", "entity", "etiological"]
TASK_TO_REASONING = {
    "forecasting": "reasoning_forecasting",
    "correlation": "reasoning_correlation",
    "entity": "reasoning_entity",
    "etiological": "reasoning_etiological",
}
RUNS = {
    "run1": "",
    "run2": "_run2_official",
    "run3": "_run3_official",
}
OUT = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_MLP = ROOT / "00_new_codes/reports/artifacts/mlp_encoder_focused_analysis"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exp_dir(task: str, suffix: str) -> Path:
    return ROOT / "exp" / f"sttest_full_{task}_6144{suffix}"


def exp_dir_512(task: str) -> Path:
    return ROOT / "exp" / f"sttest_full_{task}_512"


def exp_dir_graph_without(task: str) -> Path:
    reasoning = TASK_TO_REASONING[task]
    return ROOT / "exp" / f"stage2.4_graph_ablation_sttest_6144_without_graph_{reasoning}"


EXPERIMENT_PRESETS = {
    "6144_run1": "",
    "6144_run2": "_run2_official",
    "6144_run3": "_run3_official",
    "512_run1": "512",  # special: use exp_dir_512
}


def load_metrics(task: str, suffix: str) -> dict:
    p = exp_dir(task, suffix) / "evaluation_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_responses(task: str, suffix: str) -> dict[int, str]:
    return load_prediction_files(str(exp_dir(task, suffix)))


def load_full_responses(task: str, suffix: str) -> dict[int, str]:
    path = exp_dir(task, suffix) / "generated_answer.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["idx"]: item.get("response", "") for item in data}


def task_correct(task: str, gold: str, pred: str | None) -> bool | None:
    if pred is None:
        return None
    if task == "forecasting":
        target = _parse_series(gold)
        pred_series = _parse_series(pred)
        if not target or not pred_series:
            return None
        n = len(target)
        if len(pred_series) < n:
            pred_series = pred_series + [pred_series[-1]] * (n - len(pred_series))
        else:
            pred_series = pred_series[:n]
        return pred_series == target  # strict equality rare; use MAE threshold for "close"
    return _normalize_choice(pred) == _normalize_choice(gold)


def forecasting_mae(gold: str, pred: str | None) -> float | None:
    if pred is None:
        return None
    target = _parse_series(gold)
    pred_series = _parse_series(pred)
    if not target or not pred_series:
        return None
    n = len(target)
    if len(pred_series) < n:
        pred_series = pred_series + [pred_series[-1]] * (n - len(pred_series))
    else:
        pred_series = pred_series[:n]
    return sum(abs(a - b) for a, b in zip(pred_series, target)) / n


def response_features(text: str) -> dict:
    think = THINK_RE.search(text or "")
    think_text = think.group(1) if think else ""
    answers = ANSWER_RE.findall(text or "")
    return {
        "response_len": len(text or ""),
        "think_len": len(think_text),
        "answer_tag_count": len(answers),
        "has_think": bool(think),
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load_mismatch_indices(run_key: str) -> dict[str, set[int]]:
    sub = {"run1": "run1_recheck", "run2": "run2", "run3": "run3"}[run_key]
    path = ARTIFACT_MLP / sub / "task_level_reconstruction_mismatch.json"
    if not path.exists():
        return {t: set() for t in TASKS}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {t: set(data["tasks"][t]["mismatch_indices"]) for t in TASKS}


def build_metrics_table() -> dict:
    rows = []
    for run_name, suffix in RUNS.items():
        for task in TASKS:
            m = load_metrics(task, suffix)
            row = {
                "run": run_name,
                "task": task,
                "total_samples": m.get("total_samples"),
                "evaluated_samples": m.get("evaluated_samples"),
                "missing_predictions": m.get("missing_predictions"),
                "missing_indices": m.get("missing_indices", []),
            }
            if task == "forecasting":
                row["mae"] = m.get("mae")
                row["mape"] = m.get("mape")
            else:
                row["accuracy"] = m.get("accuracy")
            rows.append(row)
    return {"rows": rows}


def collect_sample_outcomes() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Per idx: correctness and features across runs."""
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    outcomes: dict[str, dict] = {}

    for task in TASKS:
        data = datasets[task]
        for run_name, suffix in RUNS.items():
            preds = load_responses(task, suffix)
            full = load_full_responses(task, suffix)
            for idx, sample in enumerate(data):
                key = f"{task}:{idx}"
                if key not in outcomes:
                    outcomes[key] = {"task": task, "idx": idx}
                pred = preds.get(idx)
                gold = sample.get("output", "")
                correct = task_correct(task, gold, pred)
                feats = response_features(full.get(idx) or "")
                entry = {
                    "correct": correct,
                    "pred_missing": pred is None,
                    "mae": forecasting_mae(gold, pred) if task == "forecasting" else None,
                    **feats,
                }
                outcomes[key][run_name] = entry

    return outcomes, datasets


def build_sample_outcomes(outcomes: dict[str, dict], datasets: dict[str, list[dict]]) -> dict:
    flips = {t: {"answer_flip": 0, "correctness_flip": 0, "total": len(datasets[t])} for t in TASKS}
    flip_examples = defaultdict(list)
    for key, o in outcomes.items():
        task = o["task"]
        idx = o["idx"]
        cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
        if len(cor) >= 2 and len(set(cor)) > 1:
            flips[task]["correctness_flip"] += 1
            if len(flip_examples[task]) < 5:
                flip_examples[task].append(idx)
        if task == "forecasting":
            maes = [o[r]["mae"] for r in RUNS if o[r]["mae"] is not None]
            if len(maes) >= 2 and max(maes) - min(maes) > 1.0:
                flips[task]["answer_flip"] += 1

    return {"outcomes_count": len(outcomes), "flips": flips, "flip_examples": dict(flip_examples)}


def build_flip_patterns(outcomes: dict[str, dict]) -> dict:
    """Choice tasks: R/W patterns across three runs."""
    choice_tasks = ["correlation", "entity", "etiological"]
    result = {}
    for task in choice_tasks:
        patterns = defaultdict(int)
        for key, o in outcomes.items():
            if o["task"] != task:
                continue
            labels = []
            for run in RUNS:
                c = o[run]["correct"]
                if c is None:
                    continue
                labels.append("R" if c else "W")
            if len(labels) == 3:
                patterns["".join(labels)] += 1
        volatile = sum(v for k, v in patterns.items() if len(set(k)) > 1)
        stable = sum(v for k, v in patterns.items() if len(set(k)) == 1)
        result[task] = {
            "patterns": dict(sorted(patterns.items(), key=lambda x: -x[1])),
            "volatile_count": volatile,
            "stable_count": stable,
            "volatile_ratio": round(volatile / (volatile + stable), 4) if (volatile + stable) else None,
        }
    return result


def build_flip_mismatch_cross(outcomes: dict[str, dict]) -> dict:
    """Among correctness-flip samples, overlap with strict mismatch labels."""
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    result = {}
    for task in ["correlation", "forecasting"]:
        flip_idxs = []
        for key, o in outcomes.items():
            if o["task"] != task:
                continue
            cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
            if len(cor) >= 2 and len(set(cor)) > 1:
                flip_idxs.append(o["idx"])
        flip_set = set(flip_idxs)
        buckets = {
            "flip_total": len(flip_set),
            "mismatch_all3": 0,
            "mismatch_any_run": 0,
            "mismatch_no_run": 0,
            "stable_correct_not_flip": 0,
            "stable_wrong_not_flip": 0,
        }
        r1, r2, r3 = mm["run1"][task], mm["run2"][task], mm["run3"][task]
        for idx in flip_set:
            in1, in2, in3 = idx in r1, idx in r2, idx in r3
            if in1 and in2 and in3:
                buckets["mismatch_all3"] += 1
            elif in1 or in2 or in3:
                buckets["mismatch_any_run"] += 1
            else:
                buckets["mismatch_no_run"] += 1
        for key, o in outcomes.items():
            if o["task"] != task or o["idx"] in flip_set:
                continue
            cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
            if len(cor) == 3 and len(set(cor)) == 1:
                if cor[0]:
                    buckets["stable_correct_not_flip"] += 1
                else:
                    buckets["stable_wrong_not_flip"] += 1
        for k in ("mismatch_all3", "mismatch_any_run", "mismatch_no_run"):
            if buckets["flip_total"]:
                buckets[f"{k}_ratio"] = round(buckets[k] / buckets["flip_total"], 4)
        result[task] = buckets
    return result


def build_pairwise_acc_transitions(outcomes: dict[str, dict], datasets: dict[str, list[dict]]) -> dict:
    """Pairwise run transitions for choice tasks (run1 baseline)."""
    choice_tasks = ["correlation", "entity", "etiological"]
    pairs = [("run1", "run2"), ("run1", "run3"), ("run2", "run3")]
    result = {}
    for task in choice_tasks:
        n = len(datasets[task])
        task_result = {}
        for a, b in pairs:
            rr = rw = wr = ww = missing = 0
            for key, o in outcomes.items():
                if o["task"] != task:
                    continue
                ca, cb = o[a]["correct"], o[b]["correct"]
                if ca is None or cb is None:
                    missing += 1
                    continue
                if ca and cb:
                    rr += 1
                elif ca and not cb:
                    rw += 1
                elif not ca and cb:
                    wr += 1
                else:
                    ww += 1
            comparable = rr + rw + wr + ww
            acc_a = (rr + rw) / comparable if comparable else None
            acc_b = (rr + wr) / comparable if comparable else None
            task_result[f"{a}_to_{b}"] = {
                "RR": rr,
                "RW": rw,
                "WR": wr,
                "WW": ww,
                "missing": missing,
                "acc_delta": round(acc_b - acc_a, 6) if acc_a is not None else None,
                "net_correct_change": wr - rw,
                "net_correct_change_pct_of_N": round((wr - rw) / n, 6),
            }
        result[task] = task_result
    return result


def build_mismatch_drift_vs_flip(outcomes: dict[str, dict]) -> dict:
    """Run1 mismatch samples: persistent vs drift, and correctness flip rate."""
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    result = {}
    for task in ["forecasting", "correlation"]:
        r1, r2, r3 = mm["run1"][task], mm["run2"][task], mm["run3"][task]
        all3 = r1 & r2 & r3
        drift = r1 - all3
        buckets = {
            "run1_mismatch_count": len(r1),
            "persistent_all3": len(all3),
            "drift_from_run1": len(drift),
        }
        for label, idx_set in [("persistent_all3", all3), ("drift_from_run1", drift)]:
            flip_n = 0
            for idx in idx_set:
                key = f"{task}:{idx}"
                o = outcomes.get(key)
                if not o:
                    continue
                cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
                if len(cor) >= 2 and len(set(cor)) > 1:
                    flip_n += 1
            buckets[f"{label}_correctness_flip"] = flip_n
            buckets[f"{label}_correctness_flip_ratio"] = (
                round(flip_n / len(idx_set), 4) if idx_set else None
            )
        result[task] = buckets
    return result


def build_forecasting_gaps() -> dict:
    """Parser-missing predictions vs strict mismatch overlap."""
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    rows = []
    for run_name, suffix in RUNS.items():
        m = load_metrics("forecasting", suffix)
        missing = set(m.get("missing_indices", []))
        mismatch = mm[run_name]["forecasting"]
        rows.append(
            {
                "run": run_name,
                "missing_indices": sorted(missing),
                "missing_in_mismatch": sorted(missing & mismatch),
                "missing_not_in_mismatch": sorted(missing - mismatch),
                "mismatch_count": len(mismatch),
            }
        )
    all_missing = set()
    for r in rows:
        all_missing |= set(r["missing_indices"])
    return {
        "by_run": rows,
        "all_missing_union": sorted(all_missing),
        "all_missing_in_any_mismatch": sorted(
            all_missing
            & (mm["run1"]["forecasting"] | mm["run2"]["forecasting"] | mm["run3"]["forecasting"])
        ),
    }


def build_response_length_stats() -> dict:
    stats = {}
    for run_name, suffix in RUNS.items():
        stats[run_name] = {}
        for task in TASKS:
            full = load_full_responses(task, suffix)
            resp_lens = []
            think_lens = []
            for text in full.values():
                feats = response_features(text or "")
                resp_lens.append(feats["response_len"])
                if feats["has_think"]:
                    think_lens.append(feats["think_len"])
            stats[run_name][task] = {
                "response_len_median": median(resp_lens) if resp_lens else None,
                "response_len_mean": round(mean(resp_lens), 1) if resp_lens else None,
                "think_len_median": median(think_lens) if think_lens else None,
                "think_len_mean": round(mean(think_lens), 1) if think_lens else None,
                "n_with_think": len(think_lens),
            }
    return stats


def build_mismatch_stability() -> dict:
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    result = {}
    for task in TASKS:
        r1, r2, r3 = mm["run1"][task], mm["run2"][task], mm["run3"][task]
        all3 = r1 & r2 & r3
        result[task] = {
            "run1_count": len(r1),
            "run2_count": len(r2),
            "run3_count": len(r3),
            "jaccard_12": round(jaccard(r1, r2), 4),
            "jaccard_13": round(jaccard(r1, r3), 4),
            "jaccard_23": round(jaccard(r2, r3), 4),
            "intersection_all3": len(all3),
            "only_run1": len(r1 - r2 - r3),
            "only_run2": len(r2 - r1 - r3),
            "only_run3": len(r3 - r1 - r2),
            "intersection_all3_ratio_of_run1": round(len(all3) / len(r1), 4) if r1 else None,
        }
    return result


def build_mismatch_vs_correct() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    tables = {}
    for run_name in RUNS:
        tables[run_name] = {}
        for task in TASKS:
            preds = load_responses(task, RUNS[run_name])
            mismatch_set = mm[run_name][task]
            cells = {"mismatch_correct": 0, "mismatch_wrong": 0, "other_correct": 0, "other_wrong": 0, "m_missing": 0, "o_missing": 0}
            for idx, sample in enumerate(datasets[task]):
                pred = preds.get(idx)
                c = task_correct(task, sample.get("output", ""), pred)
                if c is None:
                    key = "m_missing" if idx in mismatch_set else "o_missing"
                    cells[key] += 1
                    continue
                if idx in mismatch_set:
                    cells["mismatch_correct" if c else "mismatch_wrong"] += 1
                else:
                    cells["other_correct" if c else "other_wrong"] += 1
            tables[run_name][task] = cells
    return tables


def build_thinking_stats() -> dict:
    """Alias: same as response_length_stats think_len fields (from full response)."""
    full_stats = build_response_length_stats()
    stats = {}
    for run_name in RUNS:
        stats[run_name] = {}
        for task in TASKS:
            row = full_stats[run_name][task]
            stats[run_name][task] = {
                "n": row["n_with_think"],
                "think_len_median": row["think_len_median"],
                "think_len_mean": row["think_len_mean"],
            }
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes, datasets = collect_sample_outcomes()
    payloads = {
        "metrics_table.json": build_metrics_table(),
        "sample_outcomes.json": build_sample_outcomes(outcomes, datasets),
        "mismatch_stability.json": build_mismatch_stability(),
        "mismatch_vs_correct.json": build_mismatch_vs_correct(),
        "thinking_stats.json": build_thinking_stats(),
        "response_length_stats.json": build_response_length_stats(),
        "flip_patterns.json": build_flip_patterns(outcomes),
        "flip_mismatch_cross.json": build_flip_mismatch_cross(outcomes),
        "pairwise_acc_transitions.json": build_pairwise_acc_transitions(outcomes, datasets),
        "mismatch_drift_vs_flip.json": build_mismatch_drift_vs_flip(outcomes),
        "forecasting_gaps.json": build_forecasting_gaps(),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
