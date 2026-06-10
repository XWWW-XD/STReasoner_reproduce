#!/usr/bin/env python3
"""T0-3: audit generated_answer num_tokens field vs response length (read-only)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, RUNS, TASKS, exp_dir, response_features  # noqa: E402

WAVE6_UNCLOSED: dict[str, list[int]] = {}


def load_records(task: str, suffix: str) -> list[dict]:
    path = exp_dir(task, suffix) / "generated_answer.json"
    return json.loads(path.read_text(encoding="utf-8"))


def audit_task_run(task: str, suffix: str) -> dict:
    records = load_records(task, suffix)
    ratios = []
    rows_with_nt = 0
    unclosed_rows = []
    for item in records:
        resp = item.get("response") or ""
        nt = item.get("num_tokens")
        resp_len = len(resp)
        feats = response_features(resp)
        if nt is not None:
            rows_with_nt += 1
            ratio = resp_len / nt if nt > 0 else None
            if ratio is not None:
                ratios.append(ratio)
        idx = item.get("idx")
        if idx in WAVE6_UNCLOSED.get(task, []):
            unclosed_rows.append(
                {
                    "idx": idx,
                    "num_tokens": nt,
                    "response_len": resp_len,
                    "ratio": round(resp_len / nt, 2) if nt else None,
                    "answer_tag_count": feats["answer_tag_count"],
                }
            )
    return {
        "n": len(records),
        "with_num_tokens": rows_with_nt,
        "response_len_median": statistics.median(len(r.get("response") or "") for r in records),
        "num_tokens_median": statistics.median(r["num_tokens"] for r in records if r.get("num_tokens") is not None)
        if rows_with_nt
        else None,
        "ratio_resp_over_num_tokens_median": round(statistics.median(ratios), 3) if ratios else None,
        "ratio_p95": round(sorted(ratios)[int(len(ratios) * 0.95)] if ratios else 0, 3),
        "unclosed_focus_samples": unclosed_rows,
    }


def main() -> None:
    global WAVE6_UNCLOSED
    unclosed_path = OUT / "unclosed_thinking_forensics.json"
    if unclosed_path.exists():
        unc = json.loads(unclosed_path.read_text(encoding="utf-8"))
        for task, fs in unc.get("focus_samples", {}).items():
            if isinstance(fs, dict) and "samples" in fs:
                WAVE6_UNCLOSED[task] = [s["idx"] for s in fs["samples"]]

    summary = {task: {} for task in TASKS}
    for task in TASKS:
        for rn, suffix in RUNS.items():
            summary[task][rn] = audit_task_run(task, suffix)

    global_ratios = []
    for task in TASKS:
        for rn in RUNS:
            recs = load_records(task, RUNS[rn])
            for item in recs:
                nt = item.get("num_tokens")
                if nt and nt > 0:
                    global_ratios.append(len(item.get("response") or "") / nt)

    unclosed_ratios = [
        r["ratio"]
        for task in summary
        for rn in RUNS
        for r in summary[task][rn].get("unclosed_focus_samples", [])
        if r.get("ratio") is not None
    ]
    out = {
        "experiment": "T0_3_num_tokens_field_audit",
        "claim": "num_tokens in generated_answer.json is input-side count, not output length",
        "evidence": {
            "global_ratio_resp_over_num_tokens_median": round(statistics.median(global_ratios), 3),
            "global_ratio_p95": round(sorted(global_ratios)[int(len(global_ratios) * 0.95)], 3),
            "unclosed_samples_ratio_min": min(unclosed_ratios) if unclosed_ratios else None,
        },
        "by_task_run": summary,
        "code_anchor": "inference/inference_tsmllm_vllm.py L331 input_token_counts[idx]",
    }
    out_path = OUT / "t0_3_num_tokens_audit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
