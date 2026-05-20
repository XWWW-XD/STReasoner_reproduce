#!/usr/bin/env python3
"""Prepare fixed sample sets for stage 1 experiment 1 without running models."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_config import HF_HUB_CACHE_PATH, apply_cache_config
from huggingface_hub import HfApi, hf_hub_download

apply_cache_config()

DATASET_REPO = "Time-HD-Anonymous/ST-Bench"
SEED = 20260519
EXPERIMENT = "exp1_resource_tiny20"
TASKS = ("forecasting", "entity", "etiological", "correlation")
ST_TEST_FILES = {
    "forecasting": "ST-Test/forecasting_test.jsonl",
    "entity": "ST-Test/entity_test.jsonl",
    "etiological": "ST-Test/etiological_test.jsonl",
    "correlation": "ST-Test/correlation_test.jsonl",
}
SUBSET_ROOT = PROJECT_ROOT / "repro_kaggle/experiments/stage1_subsets" / EXPERIMENT
DOC_ROOT = PROJECT_ROOT / "repro_kaggle/experiments/stage1_docs" / EXPERIMENT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl_from_hf(repo: str, revision: str, filename: str) -> list[dict[str, Any]]:
    local_path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        cache_dir=HF_HUB_CACHE_PATH,
    )
    rows: list[dict[str, Any]] = []
    with Path(local_path).open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample["_source_file"] = filename
            sample["_original_line_index"] = line_index
            rows.append(sample)
    return rows


def summarize_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row if not key.startswith("_")})
    category_counts = Counter(str(row.get("category", "<missing>")) for row in rows)
    node_counts: Counter[int] = Counter()
    series_lengths: Counter[int] = Counter()
    for row in rows:
        series = row.get("timeseries")
        if isinstance(series, list):
            node_counts[len(series)] += 1
            for values in series:
                if isinstance(values, list):
                    series_lengths[len(values)] += 1
    return {
        "columns": keys,
        "category_counts": dict(sorted(category_counts.items())),
        "timeseries_node_count_distribution": dict(sorted(node_counts.items())),
        "timeseries_length_distribution": dict(sorted(series_lengths.items())),
    }


def length_metrics(sample: dict[str, Any]) -> dict[str, int]:
    input_text = str(sample.get("input", ""))
    timeseries_text = json.dumps(sample.get("timeseries"), ensure_ascii=False, separators=(",", ":"))
    output_text = json.dumps(sample.get("output"), ensure_ascii=False, separators=(",", ":"))
    return {
        "input_char_length": len(input_text),
        "timeseries_serialized_char_length": len(timeseries_text),
        "output_char_length": len(output_text),
        "combined_input_timeseries_char_length": len(input_text) + len(timeseries_text),
    }


def make_record(
    sample: dict[str, Any],
    sample_id: str,
    task: str,
    dataset_revision: str,
    selection_note: str,
) -> dict[str, Any]:
    record = {key: value for key, value in sample.items() if not key.startswith("_")}
    record.update(
        {
            "sample_id": sample_id,
            "task": task,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": dataset_revision,
            "source_file": sample["_source_file"],
            "original_line_index": sample["_original_line_index"],
            "selection_seed": SEED,
            "selection_note": selection_note,
            "length_metrics": length_metrics(sample),
        }
    )
    return record


def normalise_text(text: str) -> str:
    text = text.replace("→", "->")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([.;:,])\s*", r"\1 ", text)
    return text.strip().lower()


def find_unique_match(
    rows_by_task: dict[str, list[dict[str, Any]]],
    task: str,
    required_snippets: list[str],
    expected_output: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    required = [normalise_text(snippet) for snippet in required_snippets]
    candidates: list[dict[str, Any]] = []
    for row in rows_by_task[task]:
        haystack = normalise_text(str(row.get("input", "")) + "\n" + str(row.get("output", "")))
        if all(snippet in haystack for snippet in required):
            if expected_output is None or str(row.get("output")) == expected_output:
                candidates.append(row)
    if len(candidates) == 1:
        return candidates[0], []
    notes = [
        {
            "source_file": row["_source_file"],
            "original_line_index": row["_original_line_index"],
            "output": row.get("output"),
        }
        for row in candidates[:20]
    ]
    return None, notes


def prepare_tiny20(rows_by_task: dict[str, list[dict[str, Any]]], revision: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir = SUBSET_ROOT / "st_test_tiny20_seed20260519"
    doc_dir = DOC_ROOT / "st_test_tiny20_seed20260519"
    rng = random.Random(SEED)
    all_records: list[dict[str, Any]] = []
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    selection_rule = (
        "对每个 ST-Test 任务文件，使用 Python random.Random(20260519).sample(range(num_rows), 5) "
        "抽取 5 个原始零基行号；随后按行号升序写出，保证文件顺序稳定。"
    )

    for task in TASKS:
        rows = rows_by_task[task]
        selected_indices = sorted(rng.sample(range(len(rows)), 5))
        task_records = [
            make_record(
                rows[index],
                sample_id=f"tiny20_{task}_{position:02d}_line{index}",
                task=task,
                dataset_revision=revision,
                selection_note=selection_rule,
            )
            for position, index in enumerate(selected_indices, start=1)
        ]
        selected_by_task[task] = task_records
        all_records.extend(task_records)
        write_jsonl(out_dir / f"{task}_5.jsonl", task_records)

    write_jsonl(out_dir / "tiny20_all.jsonl", all_records)

    manifest = {
        "experiment": EXPERIMENT,
        "subset_name": "st_test_tiny20_seed20260519",
        "created_at_utc": utc_now(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": revision,
        "seed": SEED,
        "selection_rule": selection_rule,
        "length_metric_note": "由于本地没有完整 Qwen/STReasoner tokenizer 文件，当前记录的是近似字符长度。",
        "source_files": ST_TEST_FILES,
        "field_summary": {task: summarize_fields(rows_by_task[task]) for task in TASKS},
        "task_counts": {task: len(selected_by_task[task]) for task in TASKS},
        "total_count": len(all_records),
        "samples": [
            {
                "sample_id": record["sample_id"],
                "task": record["task"],
                "source_file": record["source_file"],
                "original_line_index": record["original_line_index"],
                "category": record.get("category"),
                "output": record.get("output"),
                "length_metrics": record["length_metrics"],
            }
            for record in all_records
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    readme = f"""# ST-Test tiny20 固定样本集

本文档说明 `{EXPERIMENT}` 的 `tiny20` 主测试样本集。

- 数据集：`{DATASET_REPO}`
- 数据集 revision：`{revision}`
- 随机种子：`{SEED}`
- 抽样规则：{selection_rule}
- 数据目录：`{out_dir.relative_to(PROJECT_ROOT)}`
- 主文件：`tiny20_all.jsonl`

该样本集共 20 条 ST-Test 样本：forecasting、entity、etiological、correlation 四类各 5 条。
每个 JSONL 行保留原始 `input`、`timeseries`、`output`、`category` 字段，并额外加入可追溯元数据。

长度统计使用近似字符长度。ST-Test 的 `input` 中时间序列位置是 `<ts><ts/>` 占位符，因此 manifest 还额外记录序列化后的 `timeseries` 长度，以及 `input + timeseries` 的合并近似长度。
"""
    (doc_dir / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (doc_dir / "README.md").write_text(readme, encoding="utf-8")

    field_lines = [
        "# 字段检查",
        "",
        f"数据集仓库：`{DATASET_REPO}`",
        f"数据集 revision：`{revision}`",
        "",
        "ST-Test 直接从原始 JSONL 文件逐行读取，而不是使用 `datasets` 合并后的 split；这样可以保留 source file 和原始行号，方便后续追溯。",
        "",
    ]
    for task in TASKS:
        field_lines.append(f"## {task}")
        field_lines.append("")
        field_lines.append(f"- 源文件：`{ST_TEST_FILES[task]}`")
        field_lines.append(f"- 行数：{len(rows_by_task[task])}")
        field_lines.append(f"- 字段摘要：")
        field_lines.append("```json")
        field_lines.append(json.dumps(summarize_fields(rows_by_task[task]), ensure_ascii=False, indent=2))
        field_lines.append("```")
        field_lines.append("")
    (doc_dir / "field_inspection.md").write_text("\n".join(field_lines), encoding="utf-8")
    return all_records, manifest


def prepare_paper_cases(rows_by_task: dict[str, list[dict[str, Any]]], revision: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = SUBSET_ROOT / "paper_cases"
    doc_dir = DOC_ROOT / "paper_cases"
    paper_cases = [
        {
            "paper_case_id": "appendix_h_table6_etiological",
            "paper_location": "Appendix H，Table 6（PDF 抽取文本）；用户提示中提到 Table 9-12，但这份 PDF 中四个附录 case 编号为 Table 6-9。",
            "task": "etiological",
            "expected_output": "<answer>D</answer>",
            "snippets": [
                "Which etiological scenario can be inferred from the spatio-temporal data?",
                "Node 0->Node 2; Node 1->Node 2; Node 2->Node 3; Node 3->Node 4",
                "Urban pollution network with industrial and traffic sources dispersing through nodes",
            ],
        },
        {
            "paper_case_id": "appendix_h_table7_entity",
            "paper_location": "Appendix H, Table 7",
            "task": "entity",
            "expected_output": "<answer>C</answer>",
            "snippets": [
                "Which (name, description) pair should Node 9 correspond to?",
                "Node 0->Node 1; Node 1->Node 3; Node 2->Node 3; Node 3->Node 4; Node 4->Node 5; Node 5->Node 6; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9",
                "Final Point, Final monitoring point",
            ],
        },
        {
            "paper_case_id": "appendix_h_table8_correlation",
            "paper_location": "Appendix H, Table 8",
            "task": "correlation",
            "expected_output": "<answer>D</answer>",
            "snippets": [
                "Which statement best describes the relationship between Node 2 and Node 4 during time steps 177-180",
                "Node 0->Node 2; Node 0->Node 6; Node 1->Node 2; Node 2->Node 3; Node 2->Node 5; Node 3->Node 4; Node 4->Node 8; Node 5->Node 7; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9",
                "Thermal signal propagates from Node 2 to Node 4 via Node 3",
            ],
        },
        {
            "paper_case_id": "appendix_h_table9_forecasting",
            "paper_location": "Appendix H, Table 9",
            "task": "forecasting",
            "expected_output": None,
            "snippets": [
                "Given the context Maximum solar heating period enhances thermal transfer, predict the value of node 2 for the next 3 steps",
                "Historical observation window: 30-35",
                "Node 0->Node 2; Node 1->Node 2; Node 2->Node 1",
            ],
            "special_note": "原始 ST-Test input 与论文 prompt 匹配，但数据集 gold output 是 [19.86, 19.97, 20.05]，论文展示的 STReasoner prediction 是 [19.88, 19.89, 19.90]。该样例保留为可复跑原始输入，并在文档中记录答案差异。",
        },
    ]

    matched_records: list[dict[str, Any]] = []
    matched_manifest_entries: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = [
        {
            "paper_case_id": "figure1_traffic_flow_illustration",
            "paper_location": "Figure 1",
            "reason": "Figure 1 是交通流说明性示例，只展示了简化图、缩略时间点，没有完整 ST-Bench input/timeseries/options 记录；用精确 query 文本检索公开 ST-Test 和 ST-CoT-Text 未找到匹配。",
        },
        {
            "paper_case_id": "requested_table10",
            "paper_location": "Table 10",
            "reason": "在提供的 PDF 抽取文本中没有找到 Table 10 的 case-study 条目；这份 PDF 的附录 case-study 表编号为 Table 6 到 Table 9。",
        },
        {
            "paper_case_id": "requested_table11",
            "paper_location": "Table 11",
            "reason": "在提供的 PDF 抽取文本中没有找到 Table 11 的 case-study 条目；这份 PDF 的附录 case-study 表编号为 Table 6 到 Table 9。",
        },
        {
            "paper_case_id": "requested_table12",
            "paper_location": "Table 12",
            "reason": "在提供的 PDF 抽取文本中没有找到 Table 12 的 case-study 条目；这份 PDF 的附录 case-study 表编号为 Table 6 到 Table 9。",
        },
    ]

    for case in paper_cases:
        match, candidates = find_unique_match(
            rows_by_task,
            task=case["task"],
            required_snippets=case["snippets"],
            expected_output=case["expected_output"],
        )
        if match is None:
            unmatched.append(
                {
                    "paper_case_id": case["paper_case_id"],
                    "paper_location": case["paper_location"],
                    "reason": "按任务、graph/question/options 片段，以及指定时的 expected answer 检索后，没有找到唯一严格匹配。",
                    "candidate_preview": candidates,
                }
            )
            continue
        record = make_record(
            match,
            sample_id=f"paper_{case['paper_case_id']}_line{match['_original_line_index']}",
            task=case["task"],
            dataset_revision=revision,
            selection_note="根据论文 case-study 文本中的 question/graph/options/answer 片段匹配；未运行模型。",
        )
        record["paper_case_id"] = case["paper_case_id"]
        record["paper_location"] = case["paper_location"]
        record["paper_matching_note"] = case.get("special_note", "按任务、graph/question/options 片段和论文答案严格匹配。")
        matched_records.append(record)
        matched_manifest_entries.append(
            {
                "paper_case_id": case["paper_case_id"],
                "paper_location": case["paper_location"],
                "matched": True,
                "sample_id": record["sample_id"],
                "task": record["task"],
                "source_file": record["source_file"],
                "original_line_index": record["original_line_index"],
                "dataset_output": record.get("output"),
                "matching_basis": case["snippets"],
                "note": record["paper_matching_note"],
            }
        )

    write_jsonl(out_dir / "paper_cases_matched.jsonl", matched_records)
    manifest = {
        "experiment": EXPERIMENT,
        "subset_name": "paper_cases",
        "created_at_utc": utc_now(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": revision,
        "matched_count": len(matched_records),
        "unmatched_count": len(unmatched),
        "matched_cases": matched_manifest_entries,
        "unmatched_cases": unmatched,
        "length_metric_note": "由于本地没有完整 Qwen/STReasoner tokenizer 文件，当前记录的是近似字符长度。",
    }
    write_json(out_dir / "paper_cases_manifest.json", manifest)

    doc_dir.mkdir(parents=True, exist_ok=True)
    readme = f"""# 论文样例匹配

本文档说明 `{EXPERIMENT}` 中论文案例 / 示例样例的匹配情况。

可严格复跑的已匹配样例保存在 `{(out_dir / 'paper_cases_matched.jsonl').relative_to(PROJECT_ROOT)}`。
无法严格还原的样例单独记录，不计入主测试成功率统计。

- 已匹配：{len(matched_records)}
- 未匹配 / 无法严格还原：{len(unmatched)}

提供的 PDF 抽取文本中，Appendix H 的案例表编号为 Table 6 到 Table 9。用户要求关注的 Table 10-12 已检查，但在该 PDF 中未发现对应的案例表。
"""
    (doc_dir / "README.md").write_text(readme, encoding="utf-8")

    notes = ["# 匹配记录", ""]
    for entry in matched_manifest_entries:
        notes.append(f"## {entry['paper_case_id']}")
        notes.append("")
        notes.append(f"- 论文位置：{entry['paper_location']}")
        notes.append(f"- 匹配来源：`{entry['source_file']}` 第 `{entry['original_line_index']}` 行")
        notes.append(f"- 数据集输出：`{entry['dataset_output']}`")
        notes.append(f"- 备注：{entry['note']}")
        notes.append("- 匹配依据：")
        for snippet in entry["matching_basis"]:
            notes.append(f"  - `{snippet}`")
        notes.append("")
    (doc_dir / "matching_notes.md").write_text("\n".join(notes), encoding="utf-8")

    unmatched_lines = ["# 未匹配论文样例", ""]
    for entry in unmatched:
        unmatched_lines.append(f"## {entry['paper_case_id']}")
        unmatched_lines.append("")
        unmatched_lines.append(f"- 论文位置：{entry['paper_location']}")
        unmatched_lines.append(f"- 原因：{entry['reason']}")
        if entry.get("candidate_preview"):
            unmatched_lines.append("- 候选预览：")
            unmatched_lines.append("```json")
            unmatched_lines.append(json.dumps(entry["candidate_preview"], ensure_ascii=False, indent=2))
            unmatched_lines.append("```")
        unmatched_lines.append("")
    (doc_dir / "paper_cases_unmatched.md").write_text("\n".join(unmatched_lines), encoding="utf-8")
    return matched_records, unmatched


def prepare_stress_case(
    rows_by_task: dict[str, list[dict[str, Any]]],
    revision: str,
    tiny20_records: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = SUBSET_ROOT / "stress_case"
    doc_dir = DOC_ROOT / "stress_case"
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for task in TASKS:
        all_rows.extend((task, row) for row in rows_by_task[task])
    task, longest = max(all_rows, key=lambda item: length_metrics(item[1])["combined_input_timeseries_char_length"])
    record = make_record(
        longest,
        sample_id=f"stress_longest_{task}_line{longest['_original_line_index']}",
        task=task,
        dataset_revision=revision,
        selection_note="从全部 ST-Test 原始文件中选择 `input` 字符长度 + 序列化 `timeseries` 字符长度合计最大的样本。",
    )
    tiny20_ids = {
        (row["source_file"], row["original_line_index"])
        for row in tiny20_records
    }
    record["appears_in_tiny20"] = (record["source_file"], record["original_line_index"]) in tiny20_ids
    write_jsonl(out_dir / "stress_longest_input_1.jsonl", [record])

    manifest = {
        "experiment": EXPERIMENT,
        "subset_name": "stress_case",
        "created_at_utc": utc_now(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": revision,
        "selection_rule": "选择全部 ST-Test 样本中 `input` 字符长度 + 序列化 `timeseries` 字符长度合计最大的 1 条。",
        "length_metric_note": "由于本地没有完整 Qwen/STReasoner tokenizer 文件，当前记录的是近似字符长度。",
        "stress_count": 1,
        "sample": {
            "sample_id": record["sample_id"],
            "task": record["task"],
            "source_file": record["source_file"],
            "original_line_index": record["original_line_index"],
            "category": record.get("category"),
            "output": record.get("output"),
            "appears_in_tiny20": record["appears_in_tiny20"],
            "length_metrics": record["length_metrics"],
        },
    }
    write_json(out_dir / "stress_manifest.json", manifest)

    doc_dir.mkdir(parents=True, exist_ok=True)
    readme = f"""# 压力测试样本

压力测试样本为 `{record['sample_id']}`，来自 `{record['source_file']}` 第 `{record['original_line_index']}` 行。

- 任务类型：`{record['task']}`
- 选择指标：近似 `input` 字符长度 + 序列化 `timeseries` 字符长度
- 合并长度：{record['length_metrics']['combined_input_timeseries_char_length']}
- Input 长度：{record['length_metrics']['input_char_length']}
- 序列化 timeseries 长度：{record['length_metrics']['timeseries_serialized_char_length']}
- 是否出现在 tiny20：{record['appears_in_tiny20']}

该样本只用于资源压力测试，不进入 tiny20 主测试成功率分母。
"""
    (doc_dir / "README.md").write_text(readme, encoding="utf-8")

    selection_notes = f"""# 选择说明

本次扫描了四个任务文件中的全部 ST-Test 原始样本。由于本地没有完整 Qwen/STReasoner tokenizer 词表，未计算正式 token 数；每个候选样本按以下近似指标排序：

`len(input) + len(json.dumps(timeseries, compact separators))`

这个指标比只看 `input` 更保守，因为 ST-Test 把时间序列数值保存在单独的 `timeseries` 字段中，而 prompt 里只有 `<ts><ts/>` 占位符。

最终选中的样本：

```json
{json.dumps(manifest['sample'], ensure_ascii=False, indent=2)}
```
"""
    (doc_dir / "selection_notes.md").write_text(selection_notes, encoding="utf-8")
    return record


def write_overall_report(
    tiny20_manifest: dict[str, Any],
    matched_records: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    stress_record: dict[str, Any],
) -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    task_counts = tiny20_manifest["task_counts"]
    report = f"""# {EXPERIMENT}

## 实验目标

准备固定、可复用的样本集，用于后续比较 STReasoner/Qwen-8B 在 fp16、8bit、4bit 三种推理配置下的资源占用、generate 成功率、decode 成功率、parse 成功率和速度。本阶段只准备数据和文档，不运行模型。

## 目录对应关系

- tiny20 主测试数据：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/st_test_tiny20_seed20260519/`
- tiny20 主测试文档：`repro_kaggle/experiments/stage1_docs/{EXPERIMENT}/st_test_tiny20_seed20260519/`
- 论文样例数据：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/paper_cases/`
- 论文样例文档：`repro_kaggle/experiments/stage1_docs/{EXPERIMENT}/paper_cases/`
- 压力测试数据：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/stress_case/`
- 压力测试文档：`repro_kaggle/experiments/stage1_docs/{EXPERIMENT}/stress_case/`

## 验收摘要

- tiny20 总数：{tiny20_manifest['total_count']} 条
- forecasting：{task_counts.get('forecasting', 0)} 条
- entity：{task_counts.get('entity', 0)} 条
- etiological：{task_counts.get('etiological', 0)} 条
- correlation：{task_counts.get('correlation', 0)} 条
- 论文样例 matched：{len(matched_records)} 条
- 论文样例 unmatched / 无法严格还原：{len(unmatched)} 条
- stress case：已选出 1 条

## 后续 fp16 / 8bit / 4bit 应读取的文件

- 主测试成功率分母：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/st_test_tiny20_seed20260519/tiny20_all.jsonl`
- 分任务主测试文件：`forecasting_5.jsonl`、`entity_5.jsonl`、`etiological_5.jsonl`、`correlation_5.jsonl`
- 论文样例额外复跑：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/paper_cases/paper_cases_matched.jsonl`
- 资源压力测试：`repro_kaggle/experiments/stage1_subsets/{EXPERIMENT}/stress_case/stress_longest_input_1.jsonl`

只有 `tiny20_all.jsonl` 中的 20 条样本计入主测试的 generate/decode/parse/速度成功率比较。matched 论文样例只作为额外定性检查或回归检查。unmatched 论文样例仅作记录。stress case 只用于资源压力测试，不计入 tiny20 成功率。

## 已知限制和注意事项

- 长度为近似字符数，不是正式 tokenizer token 数，因为本地没有完整 Qwen/STReasoner tokenizer 词表。
- ST-Test 将时间序列保存在单独的 `timeseries` 字段，而 `input` 里使用 `<ts><ts/>` 占位符，因此 stress case 使用 prompt + 序列化 time-series 的合并长度排序。
- Figure 1 是说明性示例，无法从公开 ST-Bench 文件中严格还原完整输入。
- 提供的 PDF 抽取文本显示 Appendix H 的 case-study 表为 Table 6-9；未发现 Table 10-12 的 case-study 条目。
- Appendix H 的 forecasting 样例 input 能匹配到一条 ST-Test 样本，但论文展示的 STReasoner prediction 与数据集 gold output 不一致；细节记录在 `paper_cases/matching_notes.md`。
"""
    (DOC_ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> int:
    api = HfApi()
    revision = api.dataset_info(DATASET_REPO).sha
    rows_by_task = {
        task: load_jsonl_from_hf(DATASET_REPO, revision, filename)
        for task, filename in ST_TEST_FILES.items()
    }

    tiny20_records, tiny20_manifest = prepare_tiny20(rows_by_task, revision)
    matched_records, unmatched = prepare_paper_cases(rows_by_task, revision)
    stress_record = prepare_stress_case(rows_by_task, revision, tiny20_records)
    write_overall_report(tiny20_manifest, matched_records, unmatched, stress_record)

    print(json.dumps(
        {
            "tiny20_count": len(tiny20_records),
            "tiny20_task_counts": tiny20_manifest["task_counts"],
            "paper_matched": len(matched_records),
            "paper_unmatched": len(unmatched),
            "stress_count": 1,
            "stress_sample_id": stress_record["sample_id"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
