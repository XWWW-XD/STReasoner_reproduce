#!/usr/bin/env python3
"""Count broad time-series mentions in ST-Test 6144 thinking traces.

Broad mentions are used for report 39 appendix §10 (all/mismatch/other averages).
Strict node-window reconstruction remains in find_bad_cases.py for mismatch evidence.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import find_bad_cases as fbc
from find_bad_cases import configure_paths, find_repo_root, generated_path, load_jsonl

ROOT = find_repo_root(_SCRIPT_DIR)
sys.path.insert(0, str(ROOT))
configure_paths(root=ROOT)
OUT_DIR = fbc.OUT_DIR
MISMATCH_JSON = OUT_DIR / "task_level_reconstruction_mismatch.json"
CHOICE_TASKS = {"correlation", "entity", "etiological"}

TASKS = ["forecasting", "correlation", "entity", "etiological"]

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

STRICT_WINDOW_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?Node\s+(\d+)\s*"
    r"(?:"
    r"\[(\d+)-(\d+)\]"
    r"|\((?:steps?|timesteps?|positions?)\s*(\d+)-(\d+)\)"
    r"|\((\d+)-(\d+)\)"
    r")"
    r"(?:\*\*)?\s*:\s*([^\n]+)",
    re.I,
)

NODE_NUMERIC_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?"
    r"Node\s+(\d+)\b"
    r"(?:\s*(?:\(steps?\s*\d+-\d+\)|\[\d+-\d+\]|at steps?\s*\d+-\d+|during time steps?\s*\d+-\d+))?"
    r"\s*[:：\-–—]?\s*"
    r"([^\n]{8,})",
    re.I,
)

INLINE_NODE_STEPS_RE = re.compile(
    r"Node\s+(\d+)\s*\((?:steps?|timesteps?)\s*(\d+)-(\d+)\)\s*:\s*([^\n]+)",
    re.I,
)

NODE_BRACKET_INLINE_RE = re.compile(
    r"Node\s+(\d+)\s*\[(\d+)-(\d+)\]\s*[:：]?\s*([0-9.,\s\-+]+)",
    re.I,
)

MIN_NUMERIC_VALUES = 3


def extract_think(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _has_enough_numbers(tail: str) -> bool:
    return len(NUM_RE.findall(tail or "")) >= MIN_NUMERIC_VALUES


def _is_graph_edge_line(text: str) -> bool:
    stripped = text.strip()
    return "->" in stripped or "→" in stripped


def iter_broad_mention_spans(think: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    for match in STRICT_WINDOW_LINE_RE.finditer(think):
        if _has_enough_numbers(match.group(7) or ""):
            spans.append((match.start(), match.end(), "strict_window_line"))

    for match in INLINE_NODE_STEPS_RE.finditer(think):
        if _has_enough_numbers(match.group(4)):
            spans.append((match.start(), match.end(), "inline_node_steps"))

    for match in NODE_BRACKET_INLINE_RE.finditer(think):
        if _has_enough_numbers(match.group(4)):
            spans.append((match.start(), match.end(), "node_bracket_inline"))

    for match in NODE_NUMERIC_LINE_RE.finditer(think):
        line = match.group(0)
        tail = match.group(2) or ""
        if _is_graph_edge_line(line):
            continue
        if not _has_enough_numbers(tail):
            continue
        spans.append((match.start(), match.end(), "node_numeric_line"))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, kind in spans:
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        kept.append((start, end, kind))
        occupied.append((start, end))
    return kept


def count_broad_ts_mentions(think: str) -> int:
    return len(iter_broad_mention_spans(think))


def count_strict_window_lines(think: str) -> int:
    count = 0
    for match in STRICT_WINDOW_LINE_RE.finditer(think):
        if _has_enough_numbers(match.group(7) or ""):
            count += 1
    return count


def load_row_index() -> dict[tuple[str, int], dict]:
    from evaluation.evaluate_qa import _normalize_choice, _parse_series

    index: dict[tuple[str, int], dict] = {}
    for task in TASKS:
        gold_rows = load_jsonl(fbc.ST_TEST / f"{task}_test.jsonl")
        gen_path = generated_path(task)
        if not gen_path.exists():
            continue
        gen_by_idx = {int(g["idx"]): g for g in json.loads(gen_path.read_text(encoding="utf-8"))}
        rel_pred = str(gen_path.relative_to(ROOT)).replace("\\", "/")
        for idx, gold in enumerate(gold_rows):
            gen = gen_by_idx.get(idx)
            if not gen:
                continue
            response = gen.get("response", "")
            inp = gen.get("question_text", gold.get("input", ""))
            gold_out = gold.get("output", "")
            if task in CHOICE_TASKS:
                parsed_pred = _normalize_choice(response)
                parsed_gold = _normalize_choice(gold_out)
                choice_correct = bool(parsed_pred) and parsed_pred == parsed_gold
            else:
                parsed_pred = _parse_series(response)
                parsed_gold = _parse_series(gold_out)
                choice_correct = parsed_pred == parsed_gold and parsed_pred
            index[(task, idx)] = {
                "task": f"reasoning_{task}",
                "idx": idx,
                "dataset_path": f"data/ST-Bench/ST-Test/{task}_test.jsonl",
                "prediction_path": rel_pred,
                "input": inp,
                "gold_output": gold_out,
                "raw_response": response,
                "parsed_prediction": parsed_pred,
                "parsed_gold": parsed_gold,
                "choice_correct": choice_correct,
            }
    return index


def load_outputs_by_task() -> dict[str, list[dict]]:
    index = load_row_index()
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    for (task, idx), row in index.items():
        by_task[task].append(row)
    for task in TASKS:
        by_task[task].sort(key=lambda r: int(r["idx"]))
    return by_task


def load_mismatch_indices() -> dict[str, set[int]]:
    payload = json.loads(MISMATCH_JSON.read_text(encoding="utf-8"))
    return {task: set(payload["tasks"][task]["mismatch_indices"]) for task in TASKS}


def build_mention_stats_by_group() -> dict:
    by_task = load_outputs_by_task()
    mismatch_by_task = load_mismatch_indices()
    strict_totals = {t: 0 for t in TASKS}
    broad_totals = {t: 0 for t in TASKS}

    result: dict = {
        "definition": {
            "broad_mention": (
                "Deduped spans in <think> where a Node is tied to a numeric list "
                "(>=3 numbers): strict window lines, inline (steps a-b) lists, Node-prefixed numeric lines."
            ),
            "strict_window_line_only": "NODE_WINDOW_LINE_RE matches with >=3 numbers (no raw alignment).",
            "mean": "mention_sum / sample_count",
        },
        "tasks": {},
    }

    for task in TASKS:
        rows = by_task[task]
        mismatch_set = mismatch_by_task[task]
        per_sample: dict[int, dict] = {}
        for row in rows:
            idx = int(row["idx"])
            think = extract_think(row.get("raw_response", ""))
            broad = count_broad_ts_mentions(think)
            strict = count_strict_window_lines(think)
            per_sample[idx] = {"broad_mentions": broad, "strict_window_lines": strict}
            broad_totals[task] += broad
            strict_totals[task] += strict

        task_payload = {"total_samples": len(rows), "groups": {}}
        for group_name, indices in [
            ("all", [int(r["idx"]) for r in rows]),
            ("mismatch", sorted(mismatch_set)),
            ("other", sorted(int(r["idx"]) for r in rows if int(r["idx"]) not in mismatch_set)),
        ]:
            mention_sum = sum(per_sample[i]["broad_mentions"] for i in indices)
            strict_sum = sum(per_sample[i]["strict_window_lines"] for i in indices)
            sample_count = len(indices)
            task_payload["groups"][group_name] = {
                "sample_count": sample_count,
                "mention_sum": mention_sum,
                "strict_window_line_sum": strict_sum,
                "mean_mentions_per_sample": (
                    round(mention_sum / sample_count, 4) if sample_count else None
                ),
            }
        result["tasks"][task] = task_payload

    result["sanity"] = {
        "strict_window_line_totals_by_task": strict_totals,
        "broad_mention_totals_by_task": broad_totals,
    }
    return result


def audit_mention_patterns(sample_per_task: int = 40) -> str:
    by_task = load_outputs_by_task()
    lines = [
        "# Broad time-series mention pattern audit",
        "",
        "Source: `exp/sttest_full_*_6144/generated_answer.json` + `data/ST-Bench/ST-Test/`",
        "",
    ]
    for task in TASKS:
        kind_counter: Counter[str] = Counter()
        broad_counts: list[int] = []
        strict_counts: list[int] = []
        for row in by_task[task][:sample_per_task]:
            think = extract_think(row.get("raw_response", ""))
            spans = iter_broad_mention_spans(think)
            for _, _, kind in spans:
                kind_counter[kind] += 1
            broad_counts.append(len(spans))
            strict_counts.append(count_strict_window_lines(think))
        lines.extend(
            [
                f"## {task}",
                f"- broad mean (sample): {sum(broad_counts)/len(broad_counts):.2f}",
                f"- strict window line mean (sample): {sum(strict_counts)/len(strict_counts):.2f}",
                f"- span kinds: {dict(kind_counter)}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def export_mismatch_sample_cards() -> dict:
    cards_spec = [
        ("forecasting", 19),
        ("forecasting", 179),
        ("correlation", 503),
        ("correlation", 480),
    ]
    fidelity = json.loads((OUT_DIR / "reasoning_numeric_fidelity_cases.json").read_text(encoding="utf-8"))
    cross = json.loads((OUT_DIR / "cross_node_numeric_reconstruction_cases.json").read_text(encoding="utf-8"))
    cross_by_idx = {int(c["idx"]): c for c in cross if c["task_type"] == "correlation"}

    by_task_idx: dict[tuple[str, int], list] = defaultdict(list)
    for item in fidelity:
        by_task_idx[(item["task_type"], int(item["idx"]))].append(item)

    row_index = load_row_index()

    cards = []
    for task, idx in cards_spec:
        row = row_index.get((task, idx))
        windows = []
        cross_extra = None
        if task == "forecasting":
            for item in by_task_idx[(task, idx)]:
                windows.append(
                    {
                        "node": item["node"],
                        "window": item["stated_window"],
                        "alignment_basis": item["alignment_basis"],
                        "raw_values": item["raw_values"],
                        "model_stated_values": item["model_stated_values"],
                        "max_abs_diff": item["max_abs_diff_between_stated_and_raw"],
                        "reasoning_line": item["reasoning_line"],
                    }
                )
        else:
            c = cross_by_idx.get(idx)
            if c:
                cross_extra = {
                    "reasoning_excerpt": c.get("reasoning_excerpt"),
                    "graph": c.get("graph"),
                }
                for check in c.get("reasoning_numeric_checks", []):
                    windows.append(
                        {
                            "node": check["node"],
                            "window": check["stated_window"],
                            "alignment_basis": check.get("alignment_basis"),
                            "raw_values": check["raw_values"],
                            "model_stated_values": check["model_stated_values"],
                            "max_abs_diff": check["max_abs_diff_between_stated_and_raw"],
                            "reasoning_line": check["reasoning_line"],
                        }
                    )
        inp = row.get("input", "") if row else ""
        cards.append(
            {
                "task": task,
                "idx": idx,
                "line_no": idx + 1,
                "dataset_path": row.get("dataset_path") if row else None,
                "prediction_path": row.get("prediction_path") if row else None,
                "input_excerpt": inp[:900] + ("..." if len(inp) > 900 else ""),
                "gold_output": row.get("gold_output") if row else None,
                "parsed_prediction": row.get("parsed_prediction") if row else None,
                "choice_correct": row.get("choice_correct") if row else None,
                "mismatch_windows": windows,
                "cross_case": cross_extra,
            }
        )
    return {"cards": cards}


def write_mention_markdown(stats: dict) -> str:
    rows = [
        "| 任务 | 分组 | 样本数 | 提及次数总和 | 均值 (总和/样本数) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        for group in ["all", "mismatch", "other"]:
            g = stats["tasks"][task]["groups"][group]
            mean = g["mean_mentions_per_sample"]
            mean_str = f"{mean:.4f}" if mean is not None else "N/A"
            rows.append(
                f"| {task} | {group} | {g['sample_count']} | {g['mention_sum']} | {mean_str} |"
            )
    return "\n".join(rows) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mention_pattern_audit.md").write_text(audit_mention_patterns(), encoding="utf-8")
    stats = build_mention_stats_by_group()
    (OUT_DIR / "mention_stats_mismatch_vs_other.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "mention_stats_mismatch_vs_other.md").write_text(
        write_mention_markdown(stats), encoding="utf-8"
    )
    (OUT_DIR / "mismatch_sample_cards.json").write_text(
        json.dumps(export_mismatch_sample_cards(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats["tasks"], ensure_ascii=False, indent=2))
    print(json.dumps(stats["sanity"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
