#!/usr/bin/env python3
"""Analyze ST-Align alignment_train.jsonl by actual question content."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Q_MARKER = "answer the following question: "
GRAPH_MARKER = "Graph Structure: "

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "ST-Bench" / "ST-Align" / "alignment_train.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "00_new_codes"
    / "reports"
    / "t3-autodl2-三阶段训练复现"
    / "ST-Align详细分析"
    / "artifacts"
    / "st_align_stats.json"
)

QUESTION_TYPES: dict[str, dict[str, str]] = {
    "graph_direct": {
        "label_zh": "直连边判断",
        "needs_ts": "no",
        "answer_kind": "yes/no",
        "ability_group": "graph_text_only",
    },
    "graph_indirect": {
        "label_zh": "间接路径判断",
        "needs_ts": "no",
        "answer_kind": "yes/no",
        "ability_group": "graph_text_only",
    },
    "st_node_role": {
        "label_zh": "节点角色",
        "needs_ts": "mostly_no",
        "answer_kind": "enum",
        "ability_group": "scenario_metadata",
    },
    "st_edge_lag": {
        "label_zh": "传播时滞",
        "needs_ts": "mostly_no",
        "answer_kind": "numeric",
        "ability_group": "scenario_metadata",
    },
    "st_edge_mod": {
        "label_zh": "边权调制 multiplier",
        "needs_ts": "mostly_no",
        "answer_kind": "numeric",
        "ability_group": "scenario_metadata",
    },
    "st_edge_effective": {
        "label_zh": "有效耦合强度",
        "needs_ts": "mostly_no",
        "answer_kind": "numeric",
        "ability_group": "scenario_metadata",
    },
    "ts_drift_type": {
        "label_zh": "演化模式",
        "needs_ts": "yes",
        "answer_kind": "category",
        "ability_group": "ts_pattern_classification",
    },
    "ts_baseline": {
        "label_zh": "长期基线",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_kappa": {
        "label_zh": "均值回归 kappa",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_sigma": {
        "label_zh": "噪声 sigma",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_lambda": {
        "label_zh": "耦合 lambda",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_diffusion_shape": {
        "label_zh": "扩散形状",
        "needs_ts": "yes",
        "answer_kind": "category",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_sin_amp": {
        "label_zh": "正弦振幅 A",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_sin_freq": {
        "label_zh": "正弦频率 omega",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
    "ts_sin_phase": {
        "label_zh": "正弦相位 phi",
        "needs_ts": "yes",
        "answer_kind": "numeric",
        "ability_group": "ts_numeric_inversion",
    },
}

ABILITY_GROUPS: dict[str, str] = {
    "graph_text_only": "纯图结构（Graph Structure 文本即可答）",
    "scenario_metadata": "场景 metadata（仿真设定，大多不需读波形）",
    "ts_numeric_inversion": "时序参数反演（需读 timeseries）",
    "ts_pattern_classification": "时序模式分类（需读 timeseries）",
}

STEM_NORMALIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"node \d+"), "node {id}"),
    (re.compile(r"edge \(\d+,\d+\)"), "edge ({src},{tgt})"),
    (re.compile(r"time range \[\d+, \d+\]"), "time range [{start}, {end}]"),
]


def extract_question(input_text: str) -> str:
    if Q_MARKER in input_text:
        return input_text.split(Q_MARKER, 1)[1].strip()
    return input_text.strip()


def extract_prefix(input_text: str) -> str:
    if Q_MARKER in input_text:
        return input_text.split(Q_MARKER, 1)[0].strip()
    return input_text.strip()


def extract_graph_text(prefix: str) -> str:
    if GRAPH_MARKER not in prefix:
        return ""
    return prefix.split(GRAPH_MARKER, 1)[1].split(", please analyze", 1)[0].strip()


def classify_question(question: str) -> str:
    question_lower = question.lower()
    if question_lower.startswith("is there a direct connection"):
        return "graph_direct"
    if question_lower.startswith("is there an indirect path"):
        return "graph_indirect"
    if "what is the type of node" in question_lower:
        return "st_node_role"
    if question_lower.startswith("what is the time lag between node"):
        return "st_edge_lag"
    if "modulation multiplier" in question_lower:
        return "st_edge_mod"
    if "effective coupling strength" in question_lower:
        return "st_edge_effective"
    if "evolution pattern" in question_lower:
        return "ts_drift_type"
    if "long-term baseline" in question_lower:
        return "ts_baseline"
    if "mean reversion speed" in question_lower:
        return "ts_kappa"
    if "random fluctuation intensity" in question_lower:
        return "ts_sigma"
    if "coupling strength" in question_lower and "(lambda)" in question_lower:
        return "ts_lambda"
    if "diffusion shape" in question_lower:
        return "ts_diffusion_shape"
    if "sinusoidal amplitude" in question_lower:
        return "ts_sin_amp"
    if "sinusoidal frequency" in question_lower:
        return "ts_sin_freq"
    if "sinusoidal phase" in question_lower:
        return "ts_sin_phase"
    return "unknown"


def normalize_stem(question: str) -> str:
    stem = question
    for pattern, replacement in STEM_NORMALIZE_PATTERNS:
        stem = pattern.sub(replacement, stem)
    return stem


def scenario_fingerprint(prefix: str, timeseries: list[list[float]]) -> str:
    node_count = len(timeseries)
    ts_len = len(timeseries[0]) if timeseries and timeseries[0] else 0
    graph_text = extract_graph_text(prefix)
    payload = f"{node_count}|{ts_len}|{graph_text}|{prefix[:120]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def summarize_counter(counter: Counter[str], total: int, top_n: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common(top_n):
        rows.append({"value": key, "count": count, "pct": round(100 * count / total, 4) if total else 0})
    return rows


def summarize_numeric_list(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p90": round(statistics.quantiles(values, n=10)[8], 2) if len(values) >= 10 else max(values),
    }


def analyze(path: Path) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    ability_counts: Counter[str] = Counter()
    unknown_questions: list[str] = []
    examples: dict[str, dict[str, Any]] = {}
    answer_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    stems: set[str] = set()
    scenario_question_counts: Counter[str] = Counter()
    prefix_lengths: list[int] = []
    graph_edge_counts: list[int] = []
    node_counts: Counter[int] = Counter()
    ts_lengths: Counter[int] = Counter()
    time_ranges: Counter[str] = Counter()

    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            input_text = row.get("input", "")
            output_text = str(row.get("output", ""))
            question = extract_question(input_text)
            prefix = extract_prefix(input_text)
            qtype = classify_question(question)
            type_counts[qtype] += 1
            category_counts[row.get("category", "missing")] += 1
            answer_by_type[qtype][output_text] += 1
            stems.add(normalize_stem(question))

            if qtype == "unknown" and len(unknown_questions) < 5:
                unknown_questions.append(question)

            if qtype not in examples:
                timeseries = row.get("timeseries") or []
                examples[qtype] = {
                    "question": question,
                    "answer": output_text,
                    "category": row.get("category"),
                    "node_count": len(timeseries),
                    "ts_length_node0": len(timeseries[0]) if timeseries and timeseries[0] else 0,
                    "graph_structure": extract_graph_text(prefix),
                    "prefix_chars": len(prefix),
                }

            meta = QUESTION_TYPES.get(qtype)
            if meta:
                ability_counts[meta["ability_group"]] += 1

            scenario_id = scenario_fingerprint(prefix, row.get("timeseries") or [])
            scenario_question_counts[scenario_id] += 1

            prefix_lengths.append(len(prefix))
            graph_text = extract_graph_text(prefix)
            graph_edge_counts.append(len([part for part in graph_text.split(";") if "->" in part]))

            timeseries = row.get("timeseries") or []
            node_counts[len(timeseries)] += 1
            if timeseries and timeseries[0]:
                ts_lengths[len(timeseries[0])] += 1

            time_range_match = re.search(r"time range \[(\d+), (\d+)\]", question)
            if time_range_match:
                time_ranges[f"[{time_range_match.group(1)}, {time_range_match.group(2)}]"] += 1

    per_scenario_values = list(scenario_question_counts.values())
    question_types_out: list[dict[str, Any]] = []
    for qtype, count in type_counts.most_common():
        meta = QUESTION_TYPES.get(qtype, {})
        unique_answers = len(answer_by_type[qtype])
        question_types_out.append(
            {
                "id": qtype,
                "label_zh": meta.get("label_zh", qtype),
                "count": count,
                "pct": round(100 * count / total, 4),
                "needs_ts": meta.get("needs_ts", "unknown"),
                "answer_kind": meta.get("answer_kind", "unknown"),
                "ability_group": meta.get("ability_group", "unknown"),
                "unique_answers": unique_answers,
                "top_answers": summarize_counter(answer_by_type[qtype], count, top_n=8),
                "example": examples.get(qtype),
            }
        )

    ability_out: list[dict[str, Any]] = []
    for group_id, label in ABILITY_GROUPS.items():
        count = ability_counts[group_id]
        ability_out.append(
            {
                "id": group_id,
                "label_zh": label,
                "count": count,
                "pct": round(100 * count / total, 4),
            }
        )

    category_out = [
        {"category": key, "count": value, "pct": round(100 * value / total, 4)}
        for key, value in category_counts.most_common()
    ]

    return {
        "meta": {
            "input_path": str(path),
            "total_rows": total,
            "file_bytes": path.stat().st_size,
            "unique_question_stems_normalized": len(stems),
            "unique_scenarios_fingerprint": len(scenario_question_counts),
        },
        "question_types": question_types_out,
        "ability_groups": ability_out,
        "category_field": category_out,
        "redundancy": {
            "questions_per_scenario": summarize_numeric_list([float(v) for v in per_scenario_values]),
            "scenario_count_distribution_top": summarize_counter(scenario_question_counts, total, 10),
        },
        "structure": {
            "prefix_chars": summarize_numeric_list([float(v) for v in prefix_lengths]),
            "graph_edge_count": summarize_numeric_list([float(v) for v in graph_edge_counts]),
            "node_count": summarize_counter(node_counts, total, 10),
            "ts_length_node0": summarize_counter(ts_lengths, total, 10),
            "time_range_in_question": summarize_counter(time_ranges, sum(time_ranges.values()) or 1, 10),
        },
        "graph_yes_no": {
            "graph_direct": summarize_counter(answer_by_type["graph_direct"], type_counts["graph_direct"], 2),
            "graph_indirect": summarize_counter(answer_by_type["graph_indirect"], type_counts["graph_indirect"], 2),
        },
        "unknown_questions": unknown_questions,
    }


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    stats = analyze(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Analyzed {stats['meta']['total_rows']} rows from {input_path}")
    print(f"Wrote {output_path}")
    print("Ability groups:")
    for row in stats["ability_groups"]:
        print(f"  {row['id']}: {row['count']} ({row['pct']}%)")


if __name__ == "__main__":
    main()
