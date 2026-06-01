#!/usr/bin/env python3
"""Split PaperCases.jsonl into per-task single-sample files for official inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASK_MAP = {
    "etiological": "reasoning_etiological",
    "entity": "reasoning_entity",
    "correlation": "reasoning_correlation",
    "forecasting": "reasoning_forecasting",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_") or "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 4:
        raise SystemExit(f"expected 4 paper cases, got {len(rows)}")
    manifest = []
    for i, row in enumerate(rows):
        cat = row["category"]
        task = TASK_MAP[cat]
        case_id = row.get("paper_case_id", row.get("sample_id", f"line_{i}"))
        out = args.output_dir / f"{i:02d}_{task}_{slug(case_id)}.jsonl"
        out.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append({"line_no": i, "task": task, "category": cat, "dataset": str(out), "gold_output": row.get("output")})
    (args.output_dir.parent / "datasets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(manifest)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
