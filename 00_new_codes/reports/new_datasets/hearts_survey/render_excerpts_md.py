#!/usr/bin/env python3
"""Render sample_qa_excerpts.json to Markdown for reports."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.inp.read_text(encoding="utf-8"))
    lines = [
        "# HeaRTS 真实样例摘录（HF frozen test cases）",
        "",
        data.get("disclaimer", ""),
        "",
    ]
    for e in data.get("entries", []):
        if e.get("error"):
            continue
        n = e.get("index", 0) + 1
        lines.append(f"### 样例 {n}：`{e['dataset']}` / `{e['task']}` / `{e['source_path'].split('/')[-1]}`")
        lines.append(f"- **能力维**：{e.get('capability')} · {e.get('subtask')}")
        lines.append(f"- **题干/任务（原文）**：")
        lines.append("")
        lines.append(e.get("prompt_or_query", ""))
        lines.append("")
        lines.append(f"- **题干来源**：{e.get('prompt_source')}")
        lines.append(f"- **输入**：{e.get('input_summary')}")
        gt = e.get("ground_truth")
        lines.append(f"- **正确答案（GT）**：`{json.dumps(gt, ensure_ascii=False)}`")
        lines.append(f"- **评测指标**：{e.get('metric')}")
        lines.append(f"- **备注**：{e.get('notes')}")
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {len(data.get('entries', []))} entries to {args.out}")


if __name__ == "__main__":
    main()
