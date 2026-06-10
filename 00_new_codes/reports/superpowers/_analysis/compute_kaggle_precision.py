#!/usr/bin/env python3
"""Wave12: Kaggle precision pipeline audit (non-official)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, load_jsonl  # noqa: E402

BASE = ROOT / "00_new_codes/repro_kaggle/experiments/stage1_results/experiment1_precision_resource"
CONFIGS = ["4bit_single", "8bit_single", "fp16_single", "fp16_dual"]


def audit_config(name: str) -> dict:
    d = BASE / name
    preds = list(d.glob("*predictions*.jsonl"))
    if not preds:
        return {"config": name, "found": False}
    p = preds[0]
    rows = load_jsonl(p)
    has_answer_tag = 0
    forecasting_as_choice = 0
    for r in rows:
        text = r.get("decoded_text") or r.get("prediction") or ""
        if "<answer>" in str(text).lower():
            has_answer_tag += 1
        if r.get("task") == "forecasting" and r.get("parsed_choice"):
            forecasting_as_choice += 1
    return {
        "config": name,
        "found": True,
        "prediction_file": str(p.relative_to(ROOT)).replace("\\", "/"),
        "n_records": len(rows),
        "with_answer_tag": has_answer_tag,
        "pipeline": "hf_kaggle_not_official_vllm",
        "comparable_to_sttest_6144": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "warning": "Results NOT comparable to official ST-Test 6144 vLLM runs",
        "configs": [audit_config(c) for c in CONFIGS],
    }
    path = OUT / "kaggle_precision_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
