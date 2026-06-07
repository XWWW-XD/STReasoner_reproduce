#!/usr/bin/env python3
"""Find STReasoner-8B boundary cases from existing ST-Test responses.

This script intentionally does not use Stage1/LoRA probe outputs. It only uses
existing full ST-Test 6144-token responses under exp/sttest_full_*_6144.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/root/autodl-tmp/STReasoner_reproduce")
OUT_DIR = ROOT / "00_new_codes/reports/artifacts/mlp_encoder_focused_analysis"
ST_TEST = ROOT / "data/ST-Bench/ST-Test"
EXP = ROOT / "exp"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
GRAPH_RE = re.compile(r"Graph Structure: (.*?), please analyze", re.S)
WINDOW_RE = re.compile(r"Historical observation window: (\d+)-(\d+)")
FORECAST_TARGET_RE = re.compile(r"predict the value of node (\d+) for the next (\d+) steps", re.I)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
MISMATCH_TOLERANCE = 0.01

# Explicit numeric reconstruction lines in model reasoning, for example:
# - Node 2 [27-32]: 81.32, 77.18, ...
# Node 1 (steps 18-31): 211.94, 172.27, ...
NODE_WINDOW_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?Node\s+(\d+)\s*"
    r"(?:"
    r"\[(\d+)-(\d+)\]"
    r"|\((?:steps?|timesteps?|positions?)\s*(\d+)-(\d+)\)"
    r"|\((\d+)-(\d+)\)"
    r")"
    r"(?:\*\*)?\s*:\s*([^\n]+)",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_think(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def graph_info(input_text: str) -> tuple[str, list[tuple[str, str]], list[int]]:
    match = GRAPH_RE.search(input_text)
    graph = " ".join(match.group(1).split()) if match else ""
    edges = re.findall(r"Node\s+(\d+)\s*->\s*Node\s+(\d+)", graph)
    nodes = sorted({int(node) for edge in edges for node in edge}) if edges else []
    return graph, edges, nodes


def is_int_like(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def line_no(idx: int) -> int:
    return idx + 1


def short(text: str, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def compare_stated_to_raw(sample: dict, node: int, start: int, end: int, stated_values: list[float]) -> dict | None:
    """Align a stated node/window against raw values.

    The responses mix 0-based index wording and 1-based "time step" wording.
    To avoid over-counting false mismatches, try both alignments and keep the
    smaller-difference alignment.
    """
    if node >= len(sample.get("timeseries", [])) or start < 0:
        return None

    series = [float(x) for x in sample["timeseries"][node]]
    candidates: list[tuple[str, list[float]]] = []
    if start < len(series):
        candidates.append(("0-based", series[start : end + 1]))
    if start > 0 and start - 1 < len(series):
        candidates.append(("1-based", series[start - 1 : end]))

    best: dict | None = None
    for basis, raw_values in candidates:
        n = min(len(stated_values), len(raw_values))
        if n < 3:
            continue
        stated = stated_values[:n]
        raw = raw_values[:n]
        diffs = [abs(a - b) for a, b in zip(stated, raw)]
        candidate = {
            "alignment_basis": basis,
            "node": node,
            "stated_window": [start, end],
            "raw_values": raw,
            "model_stated_values": stated,
            "mean_abs_diff_between_stated_and_raw": sum(diffs) / n,
            "max_abs_diff_between_stated_and_raw": max(diffs),
        }
        if best is None or candidate["mean_abs_diff_between_stated_and_raw"] < best["mean_abs_diff_between_stated_and_raw"]:
            best = candidate
    return best


def extract_numeric_reconstructions(sample: dict, response: str, mismatch_only: bool = True) -> list[dict]:
    """Compare model-stated node-window values in <think> with raw time series."""
    think = extract_think(response)
    checks: list[dict] = []
    for match in NODE_WINDOW_LINE_RE.finditer(think):
        node = int(match.group(1))
        groups = match.groups()
        start = next(int(x) for x in (groups[1], groups[3], groups[5]) if x is not None)
        end = next(int(x) for x in (groups[2], groups[4], groups[6]) if x is not None)
        tail = groups[7].strip()
        # Keep direct value-list reconstructions, not vague prose like
        # "Shows values around 100-110".
        if not re.match(r"[-+]?\d", tail):
            continue
        stated_values = [float(x) for x in NUM_RE.findall(tail)]
        check = compare_stated_to_raw(sample, node, start, end, stated_values)
        if not check:
            continue
        check["reasoning_line"] = match.group(0).strip()
        if mismatch_only and check["max_abs_diff_between_stated_and_raw"] <= MISMATCH_TOLERANCE:
            continue
        checks.append(check)
    return sorted(checks, key=lambda c: -c["max_abs_diff_between_stated_and_raw"])


def forecasting_data_and_outputs() -> tuple[list[dict], list[dict]]:
    data = load_jsonl(ST_TEST / "forecasting_test.jsonl")
    outputs = json.loads((EXP / "sttest_full_forecasting_6144/generated_answer.json").read_text())
    return data, outputs


def build_reasoning_numeric_fidelity_cases() -> list[dict]:
    """Find cases where model reasoning reconstructs an input window incorrectly."""
    data, outputs = forecasting_data_and_outputs()
    cases: list[dict] = []
    for item in outputs:
        idx = int(item["idx"])
        sample = data[idx]
        response = item.get("response", "")
        think = extract_think(response)
        prompt_window = WINDOW_RE.search(sample["input"])
        prompt_window_pair = [int(prompt_window.group(1)), int(prompt_window.group(2))] if prompt_window else None
        target_match = FORECAST_TARGET_RE.search(sample["input"])
        target_node = int(target_match.group(1)) if target_match else None
        horizon = int(target_match.group(2)) if target_match else None
        for check in extract_numeric_reconstructions(sample, response):
            raw_values = check["raw_values"]
            stated_values = check["model_stated_values"]
            n = len(raw_values)
            raw_nonint_ratio = sum(not is_int_like(x) for x in raw_values) / n
            stated_int_ratio = sum(is_int_like(x) for x in stated_values) / n
            cases.append(
                {
                    "boundary": "numeric_fidelity_in_reasoning",
                    "evidence_type": "model_reconstructs_input_window_incorrectly",
                    "idx": idx,
                    "line_no": line_no(idx),
                    "data_file": "data/ST-Bench/ST-Test/forecasting_test.jsonl",
                    "generated_file": "exp/sttest_full_forecasting_6144/generated_answer.json",
                    "task_type": "forecasting",
                    "node": check["node"],
                    "target_node": target_node,
                    "horizon": horizon,
                    "prompt_window": prompt_window_pair,
                    "alignment_basis": check["alignment_basis"],
                    "stated_window": check["stated_window"],
                    "raw_values": raw_values,
                    "model_stated_values": stated_values,
                    "mean_abs_diff_between_stated_and_raw": check["mean_abs_diff_between_stated_and_raw"],
                    "max_abs_diff_between_stated_and_raw": check["max_abs_diff_between_stated_and_raw"],
                    "raw_nonint_ratio": raw_nonint_ratio,
                    "model_stated_int_ratio": stated_int_ratio,
                    "reasoning_line": check["reasoning_line"],
                    "why_this_supports_boundary": (
                        "The model explicitly reconstructs numeric values from the <ts> embedding in its reasoning, "
                        "but those values do not match the raw time series window in the dataset."
                    ),
                }
            )
    return sorted(cases, key=lambda c: -c["max_abs_diff_between_stated_and_raw"])


def build_cross_node_reasoning_error_cases() -> list[dict]:
    """Find cross-node responses where reasoning reconstructs a node-window incorrectly."""
    cases: list[dict] = []
    for task in ["correlation", "entity", "etiological"]:
        data_path = ST_TEST / f"{task}_test.jsonl"
        gen_path = EXP / f"sttest_full_{task}_6144/generated_answer.json"
        if not data_path.exists() or not gen_path.exists():
            continue
        data = load_jsonl(data_path)
        outputs = json.loads(gen_path.read_text())
        for item in outputs:
            idx = int(item["idx"])
            sample = data[idx]
            graph, edges, nodes = graph_info(sample["input"])
            numeric_checks = extract_numeric_reconstructions(sample, item.get("response", ""))[:5]
            if not numeric_checks:
                continue
            cases.append(
                {
                    "boundary": "cross_patch_cross_node_reasoning_error",
                    "evidence_type": "cross_node_reasoning_reconstructs_input_window_incorrectly",
                    "idx": idx,
                    "line_no": line_no(idx),
                    "data_file": f"data/ST-Bench/ST-Test/{task}_test.jsonl",
                    "generated_file": f"exp/sttest_full_{task}_6144/generated_answer.json",
                    "task_type": task,
                    "graph": graph,
                    "edge_count": len(edges),
                    "node_count": len(sample.get("timeseries", [])),
                    "reasoning_excerpt": short(extract_think(item.get("response", "")), 1800),
                    "reasoning_numeric_checks": numeric_checks,
                    "top_reasoning_numeric_check": numeric_checks[0] if numeric_checks else None,
                    "question_excerpt": short(sample["input"], 900),
                    "why_this_supports_boundary": (
                        "The reasoning trace is in a graph/time-series task and explicitly reconstructs a node-window with values that do not match the raw time series."
                    ),
                }
            )
    return sorted(
        cases,
        key=lambda c: (
            -c["top_reasoning_numeric_check"]["max_abs_diff_between_stated_and_raw"],
            -c["edge_count"],
            -c["node_count"],
            c["task_type"],
            c["idx"],
        ),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric_fidelity = build_reasoning_numeric_fidelity_cases()
    cross_node = build_cross_node_reasoning_error_cases()
    cross_node_with_numeric_checks = [c for c in cross_node if c.get("top_reasoning_numeric_check")]

    summary = {
        "scope_note": "Only existing STReasoner-8B full ST-Test 6144-token outputs are used. Stage1/LoRA probe outputs are intentionally excluded.",
        "mismatch_definition": {
            "alignment": "For each model-stated node/window, both 0-based and 1-based raw alignments are tried; the smaller-difference alignment is used.",
            "max_abs_diff_threshold": MISMATCH_TOLERANCE,
        },
        "counts": {
            "reasoning_numeric_reconstruction_cases": len(numeric_fidelity),
            "cross_node_reasoning_numeric_reconstruction_cases": len(cross_node),
            "cross_node_cases_with_numeric_reconstruction_check": len(cross_node_with_numeric_checks),
        },
        "representatives": {
            "numeric_fidelity_in_reasoning": numeric_fidelity[:5],
            "cross_patch_cross_node_reasoning_error": cross_node[:5],
            "cross_node_numeric_reconstruction": cross_node_with_numeric_checks[:5],
        },
        "similar_indices": {
            "numeric_fidelity_in_reasoning": [c["idx"] for c in numeric_fidelity[:80]],
            "cross_patch_cross_node_reasoning_error": [c["idx"] for c in cross_node[:120]],
            "cross_node_numeric_reconstruction": [c["idx"] for c in cross_node_with_numeric_checks[:80]],
        },
    }

    (OUT_DIR / "reasoning_numeric_fidelity_cases.json").write_text(
        json.dumps(numeric_fidelity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "cross_node_reasoning_error_cases.json").write_text(
        json.dumps(cross_node, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "cross_node_numeric_reconstruction_cases.json").write_text(
        json.dumps(cross_node_with_numeric_checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "boundary_cases_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(OUT_DIR / "boundary_cases_summary.json")


if __name__ == "__main__":
    main()
