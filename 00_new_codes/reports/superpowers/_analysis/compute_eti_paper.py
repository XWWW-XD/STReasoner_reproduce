#!/usr/bin/env python3
"""Wave11: etiological stability + paper cases ledger (plan-aligned)."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, RUNS, collect_sample_outcomes, exp_dir, load_full_responses, load_jsonl, response_features  # noqa: E402
from evaluation.evaluate_qa import _normalize_choice, load_prediction_files  # noqa: E402


def etiological_analysis(outcomes: dict) -> dict:
    task = "etiological"
    patterns = Counter()
    flip_idxs = []
    think_lens = []
    no_tag = 0
    for key, o in outcomes.items():
        if o["task"] != task:
            continue
        labels = []
        for rn in RUNS:
            c = o[rn]["correct"]
            labels.append("R" if c else "W")
            think_lens.append(o[rn].get("think_len", 0))
            if o[rn].get("answer_tag_count", 0) == 0:
                no_tag += 1
        pat = "".join(labels)
        patterns[pat] += 1
        if len(set(labels)) > 1:
            flip_idxs.append(o["idx"])
    n = 207
    return {
        "n": n,
        "flip_count": len(flip_idxs),
        "flip_rate": round(len(flip_idxs) / n, 4),
        "flip_patterns": dict(patterns),
        "flip_indices": flip_idxs,
        "no_answer_tag_rate_run123": round(no_tag / (n * 3), 4),
        "think_len_median": statistics.median(think_lens) if think_lens else None,
        "think_len_mean": round(statistics.mean(think_lens), 1) if think_lens else None,
        "strict_mismatch_run1_count": 0,
        "note": "High acc + 13% flip: answer-layer instability; strict mismatch≈0 per stage1",
    }


def _load_parsed_answer(exp_rel: str) -> str | None:
    path = ROOT / exp_rel.replace("\\", "/") / "generated_answer.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return None
    item = data[0]
    preds = load_prediction_files(str(path.parent))
    ans = preds.get(item.get("idx", 0))
    return _normalize_choice(ans) if ans else None


def paper_cases_ledger() -> dict:
    pc_path = ROOT / "00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl"
    cases = load_jsonl(pc_path)
    by_id = {c.get("paper_case_id"): c for c in cases}

    paired_path = ROOT / "00_new_codes/repro_autodl/experiments/results/stage2.4_graph_ablation_paper_cases_6144/paired_results.jsonl"
    grouped: dict[str, dict] = {}
    if paired_path.exists():
        for line in paired_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["paper_case_id"]
            if pid not in grouped:
                grouped[pid] = {"with_graph": None, "without_graph": None}
            arm = grouped[pid]
            slot = "with_graph" if rec.get("variant") == "with_graph" else "without_graph"
            arm[slot] = rec

    rows = []
    for pid, meta in by_id.items():
        gold_raw = meta.get("output", "")
        gold = _normalize_choice(gold_raw) if "<answer>" in str(gold_raw) else gold_raw.strip()
        arms = grouped.get(pid, {})
        wg = arms.get("with_graph") or {}
        wog = arms.get("without_graph") or {}
        wg_ans = wg.get("parsed_answer") or _load_parsed_answer(wg.get("exp", ""))
        wog_ans = wog.get("parsed_answer") or _load_parsed_answer(wog.get("exp", ""))
        oidx = meta.get("original_line_index")
        baseline = None
        task = meta.get("category", "correlation")
        if oidx is not None and task in ("correlation", "entity", "etiological", "forecasting"):
            preds = load_prediction_files(str(exp_dir(task, "")))
            raw = preds.get(int(oidx))
            baseline = raw if task == "forecasting" else _normalize_choice(raw) if raw else None
        rows.append(
            {
                "paper_case_id": pid,
                "original_index": oidx,
                "task": task,
                "gold": gold,
                "baseline_6144_run1": baseline,
                "graph_with": wg_ans,
                "graph_without": wog_ans,
                "with_graph_correct": wg_ans == gold if wg_ans and task != "forecasting" else None,
                "without_graph_correct": wog_ans == gold if wog_ans and task != "forecasting" else None,
                "with_graph_mae": wg.get("metrics", {}).get("mae") if task == "forecasting" else None,
                "without_graph_mae": wog.get("metrics", {}).get("mae") if task == "forecasting" else None,
                "spatial_mentions_with": wg.get("spatial_term_mentions"),
                "spatial_mentions_without": wog.get("spatial_term_mentions"),
            }
        )
    return {"paper_cases_count": len(rows), "cases": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes, _ = collect_sample_outcomes()
    eti = etiological_analysis(outcomes)
    paper = paper_cases_ledger()

    (OUT / "etiological_stability.json").write_text(
        json.dumps(eti, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combined = {"etiological": eti, "paper_cases": paper}
    (OUT / "etiological_and_paper_cases.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'etiological_stability.json'}")
    print(f"wrote {OUT / 'etiological_and_paper_cases.json'}")


if __name__ == "__main__":
    main()
