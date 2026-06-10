#!/usr/bin/env python3
"""Wave12: Kaggle precision pipeline audit (non-official).

2026-06: repro_kaggle stage1_results jsonl removed (strategy-1 archive). This script
re-emits the frozen artifact or a stub; it cannot re-audit from raw predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT  # noqa: E402

FROZEN = OUT / "kaggle_precision_audit.json"
ARCHIVE_NOTE = (
    "Kaggle stage1_results deleted 2026-06; see repro_kaggle/stage1_docs and "
    "reports/07-四配置SmartTest输出.md for text archives."
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN.exists():
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        payload["archive_note"] = ARCHIVE_NOTE
        payload["source"] = "frozen_artifact"
    else:
        payload = {
            "warning": "Results NOT comparable to official ST-Test 6144 vLLM runs",
            "archive_note": ARCHIVE_NOTE,
            "source": "stub_no_raw_data",
            "configs": [
                {"config": c, "found": False, "pipeline": "hf_kaggle_not_official_vllm"}
                for c in ["4bit_single", "8bit_single", "fp16_single", "fp16_dual"]
            ],
        }
    path = OUT / "kaggle_precision_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
