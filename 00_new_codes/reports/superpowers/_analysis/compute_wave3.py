#!/usr/bin/env python3
"""Wave-3: further mining of existing three-run exp data only."""
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

from compute import (  # noqa: E402
    OUT,
    RUNS,
    TASKS,
    collect_sample_outcomes,
    forecasting_mae,
    load_full_responses,
    load_jsonl,
    load_mismatch_indices,
    load_responses,
    task_correct,
)
from compute_deep import (  # noqa: E402
    LOOSE_NODE_NUMERIC_RE,
    THINK_RE,
    classify_window,
    persistent_indices,
)
from count_ts_mentions import (  # noqa: E402
    count_broad_ts_mentions,
    count_strict_window_lines,
    extract_think,
)
from find_bad_cases import (  # noqa: E402
    GRAPH_RE,
    extract_numeric_reconstructions,
    graph_info,
)

CHOICE_TASKS = ["correlation", "entity", "etiological"]
_RESPONSE_CACHE: dict[str, dict[str, dict[int, str]]] | None = None
_PRED_CACHE: dict[str, dict[str, dict[int, str | None]]] | None = None


def load_all_responses() -> dict[str, dict[str, dict[int, str]]]:
    global _RESPONSE_CACHE
    if _RESPONSE_CACHE is None:
        _RESPONSE_CACHE = {}
        for task in TASKS:
            _RESPONSE_CACHE[task] = {rn: load_full_responses(task, suf) for rn, suf in RUNS.items()}
    return _RESPONSE_CACHE


def load_all_preds() -> dict[str, dict[str, dict[int, str | None]]]:
    global _PRED_CACHE
    if _PRED_CACHE is None:
        _PRED_CACHE = {}
        for task in TASKS:
            _PRED_CACHE[task] = {rn: load_responses(task, RUNS[rn]) for rn in RUNS}
    return _PRED_CACHE


def load_persistent_classes() -> dict[str, dict[int, str]]:
    path = OUT / "persistent_core_classification.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[int, str]] = {}
    for task in ["forecasting", "correlation"]:
        out[task] = {s["idx"]: s["primary_class"] for s in data[task]["samples"]}
    return out


def build_loose_vs_accuracy() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    resp_cache = load_all_responses()
    result = {}
    for task in ["entity", "etiological"]:
        groups = {"loose_only": [], "no_numeric": [], "strict_mismatch": []}
        for idx, sample in enumerate(datasets[task]):
            think = extract_think(resp_cache[task]["run1"].get(idx, ""))
            strict_mm = extract_numeric_reconstructions(sample, resp_cache[task]["run1"].get(idx, ""), True)
            strict_any = extract_numeric_reconstructions(sample, resp_cache[task]["run1"].get(idx, ""), False)
            loose = bool(LOOSE_NODE_NUMERIC_RE.search(think))
            if strict_mm:
                key = "strict_mismatch"
            elif loose:
                key = "loose_only"
            else:
                key = "no_numeric"
            for rn in RUNS:
                pred = load_all_preds()[task][rn].get(idx)
                c = task_correct(task, sample.get("output", ""), pred)
                groups[key].append(c)
        result[task] = {}
        for g, vals in groups.items():
            valid = [v for v in vals if v is not None]
            n_samples = len(valid) // 3 if valid else 0
            per_run_acc = {}
            for i, rn in enumerate(RUNS):
                chunk = valid[i::3][:n_samples] if n_samples else []
                per_run_acc[rn] = round(sum(chunk) / len(chunk), 4) if chunk else None
            all_valid = [v for v in vals if v is not None]
            result[task][g] = {
                "sample_count": n_samples,
                "accuracy_run1": per_run_acc["run1"],
                "accuracy_run2": per_run_acc["run2"],
                "accuracy_run3": per_run_acc["run3"],
                "accuracy_mean_3run": round(sum(all_valid) / len(all_valid), 4) if all_valid else None,
            }
    return result


def build_thinking_density_profiles(outcomes: dict[str, dict]) -> dict:
    resp_cache = load_all_responses()
    mm = {run: load_mismatch_indices(run) for run in RUNS}
    profiles = {}
    for task in TASKS:
        buckets = {
            "all": [],
            "mismatch_run1": [],
            "persistent_all3": [],
            "correctness_flip": [],
            "stable_correct": [],
        }
        persistent = persistent_indices(task) if task in ("forecasting", "correlation") else set()
        for key, o in outcomes.items():
            if o["task"] != task:
                continue
            idx = o["idx"]
            feats = []
            for rn in RUNS:
                think = extract_think(resp_cache[task][rn].get(idx, ""))
                feats.append(
                    {
                        "broad": count_broad_ts_mentions(think),
                        "strict_lines": count_strict_window_lines(think),
                        "think_len": len(think),
                    }
                )
            row = {
                "broad_mean": statistics.mean(f["broad"] for f in feats),
                "strict_lines_mean": statistics.mean(f["strict_lines"] for f in feats),
                "think_len_mean": statistics.mean(f["think_len"] for f in feats),
            }
            buckets["all"].append(row)
            if idx in mm["run1"][task]:
                buckets["mismatch_run1"].append(row)
            if idx in persistent:
                buckets["persistent_all3"].append(row)
            cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
            if len(cor) >= 2 and len(set(cor)) > 1:
                buckets["correctness_flip"].append(row)
            elif len(cor) == 3 and cor[0]:
                buckets["stable_correct"].append(row)

        def summarize(rows: list[dict]) -> dict | None:
            if not rows:
                return None
            return {
                "n": len(rows),
                "broad_mean_median": round(statistics.median(r["broad_mean"] for r in rows), 2),
                "strict_lines_mean_median": round(statistics.median(r["strict_lines_mean"] for r in rows), 2),
                "think_len_mean_median": round(statistics.median(r["think_len_mean"] for r in rows), 1),
            }

        profiles[task] = {k: summarize(v) for k, v in buckets.items()}
    return profiles


def build_persistent_severity_stability() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    resp_cache = load_all_responses()
    result = {}
    for task in ["forecasting", "correlation"]:
        idxs = sorted(persistent_indices(task))
        max_diffs_per_idx = []
        for idx in idxs:
            sample = datasets[task][idx]
            per_run = []
            for rn in RUNS:
                checks = extract_numeric_reconstructions(
                    sample, resp_cache[task][rn].get(idx, ""), mismatch_only=True
                )
                per_run.append(max((c["max_abs_diff_between_stated_and_raw"] for c in checks), default=0.0))
            max_diffs_per_idx.append(
                {
                    "idx": idx,
                    "run1": round(per_run[0], 2),
                    "run2": round(per_run[1], 2),
                    "run3": round(per_run[2], 2),
                    "range": round(max(per_run) - min(per_run), 2),
                    "cv": round(statistics.pstdev(per_run) / (statistics.mean(per_run) + 1e-6), 4),
                }
            )
        ranges = [x["range"] for x in max_diffs_per_idx]
        result[task] = {
            "n": len(idxs),
            "max_diff_range_median": round(statistics.median(ranges), 2),
            "max_diff_range_gt50": sum(1 for r in ranges if r > 50),
            "max_diff_range_gt200": sum(1 for r in ranges if r > 200),
            "top_unstable": sorted(max_diffs_per_idx, key=lambda x: -x["range"])[:8],
        }
    return result


def build_class_vs_correctness() -> dict:
    classes = load_persistent_classes()
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    result = {}
    for task in ["correlation"]:
        by_class: dict[str, list[bool]] = defaultdict(list)
        for idx, cls in classes[task].items():
            sample = datasets[task][idx]
            for rn in RUNS:
                pred = load_all_preds()[task][rn].get(idx)
                c = task_correct(task, sample.get("output", ""), pred)
                if c is not None:
                    by_class[cls].append(c)
        result[task] = {
            cls: {
                "n_samples": len(vals) // 3,
                "accuracy_mean_3run": round(sum(vals) / len(vals), 4) if vals else None,
            }
            for cls, vals in sorted(by_class.items())
        }
    return result


def build_graph_complexity_vs_mismatch() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    mm = load_mismatch_indices("run1")
    result = {}
    for task in ["forecasting", "correlation"]:
        bins: dict[str, dict] = defaultdict(lambda: {"total": 0, "mismatch": 0})
        for idx, sample in enumerate(datasets[task]):
            _, edges, nodes = graph_info(sample.get("input", ""))
            n_nodes = len(nodes) or len(sample.get("timeseries", []))
            n_edges = len(edges)
            if n_nodes <= 3:
                b = "nodes_le3"
            elif n_nodes <= 6:
                b = "nodes_4_6"
            else:
                b = "nodes_ge7"
            key = f"{b}_edges_{'le5' if n_edges <= 5 else 'gt5'}"
            bins[key]["total"] += 1
            if idx in mm[task]:
                bins[key]["mismatch"] += 1
        result[task] = {
            k: {
                **v,
                "mismatch_rate": round(v["mismatch"] / v["total"], 4) if v["total"] else None,
            }
            for k, v in sorted(bins.items())
        }
    return result


def build_answer_diversity() -> dict:
    result = {}
    for task in CHOICE_TASKS:
        counter = Counter()
        triple_diff = []
        for idx in range(len(load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{task}_test.jsonl"))):
            answers = []
            for rn in RUNS:
                pred = load_all_preds()[task][rn].get(idx)
                if pred:
                    from evaluation.evaluate_qa import _normalize_choice

                    answers.append(_normalize_choice(pred))
            uniq = len(set(answers))
            counter[uniq] += 1
            if uniq == 3:
                triple_diff.append(idx)
        result[task] = {
            "unique_answer_counts": dict(counter),
            "three_different_answers_count": counter[3],
            "three_different_examples": triple_diff[:10],
        }
    return result


def build_forecasting_mae_ranking(outcomes: dict[str, dict]) -> dict:
    rows = []
    for key, o in outcomes.items():
        if o["task"] != "forecasting":
            continue
        idx = o["idx"]
        maes = {rn: o[rn]["mae"] for rn in RUNS if o[rn]["mae"] is not None}
        if len(maes) < 3:
            continue
        vals = list(maes.values())
        best_run = min(maes, key=lambda r: maes[r])
        rows.append(
            {
                "idx": idx,
                "mae_run1": round(maes["run1"], 2),
                "mae_run2": round(maes["run2"], 2),
                "mae_run3": round(maes["run3"], 2),
                "mae_mean": round(statistics.mean(vals), 2),
                "mae_range": round(max(vals) - min(vals), 2),
                "best_run": best_run,
                "persistent": idx in persistent_indices("forecasting"),
            }
        )
    run_wins = Counter(r["best_run"] for r in rows)
    high_vol = sorted(rows, key=lambda r: -r["mae_range"])[:15]
    return {
        "best_run_wins": dict(run_wins),
        "high_mae_range_top15": high_vol,
        "run2_beats_run1_count": sum(1 for r in rows if maes_lt(r, "run2", "run1")),
        "run3_beats_run1_count": sum(1 for r in rows if maes_lt(r, "run3", "run1")),
        "n_with_3_mae": len(rows),
    }


def maes_lt(r: dict, a: str, b: str) -> bool:
    return r[f"mae_{a}"] < r[f"mae_{b}"]


def build_flip_thinking_contrast(outcomes: dict[str, dict]) -> dict:
    resp_cache = load_all_responses()
    result = {}
    for task in CHOICE_TASKS:
        flip_broad, stable_broad = [], []
        for key, o in outcomes.items():
            if o["task"] != task:
                continue
            cor = [o[r]["correct"] for r in RUNS if o[r]["correct"] is not None]
            idx = o["idx"]
            broads = [
                count_broad_ts_mentions(extract_think(resp_cache[task][rn].get(idx, ""))) for rn in RUNS
            ]
            avg_broad = statistics.mean(broads)
            if len(cor) >= 2 and len(set(cor)) > 1:
                flip_broad.append(avg_broad)
            elif len(cor) == 3:
                stable_broad.append(avg_broad)
        result[task] = {
            "flip_n": len(flip_broad),
            "stable_n": len(stable_broad),
            "flip_broad_median": round(statistics.median(flip_broad), 2) if flip_broad else None,
            "stable_broad_median": round(statistics.median(stable_broad), 2) if stable_broad else None,
            "delta_flip_minus_stable": (
                round(statistics.median(flip_broad) - statistics.median(stable_broad), 2)
                if flip_broad and stable_broad
                else None
            ),
        }
    return result


def build_series_length_vs_mismatch() -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in TASKS}
    mm = load_mismatch_indices("run1")
    result = {}
    for task in ["forecasting", "correlation"]:
        bins = {"short_le48": [0, 0], "mid_49_72": [0, 0], "long_ge73": [0, 0]}
        for idx, sample in enumerate(datasets[task]):
            ts = sample.get("timeseries") or []
            length = max((len(s) for s in ts), default=0)
            if length <= 48:
                b = "short_le48"
            elif length <= 72:
                b = "mid_49_72"
            else:
                b = "long_ge73"
            bins[b][0] += 1
            if idx in mm[task]:
                bins[b][1] += 1
        result[task] = {
            k: {"total": t, "mismatch": m, "rate": round(m / t, 4) if t else None}
            for k, (t, m) in bins.items()
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading outcomes...", flush=True)
    outcomes, _ = collect_sample_outcomes()
    print("caching responses...", flush=True)
    load_all_responses()
    load_all_preds()
    payloads = {
        "loose_vs_accuracy.json": build_loose_vs_accuracy(),
        "thinking_density_profiles.json": build_thinking_density_profiles(outcomes),
        "persistent_severity_stability.json": build_persistent_severity_stability(),
        "class_vs_correctness.json": build_class_vs_correctness(),
        "graph_complexity_vs_mismatch.json": build_graph_complexity_vs_mismatch(),
        "answer_diversity.json": build_answer_diversity(),
        "forecasting_mae_ranking.json": build_forecasting_mae_ranking(outcomes),
        "flip_thinking_contrast.json": build_flip_thinking_contrast(outcomes),
        "series_length_vs_mismatch.json": build_series_length_vs_mismatch(),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
