#!/usr/bin/env python3
"""Wave8: experiment registry across exp/ and repro_*."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS = ROOT / "00_new_codes/reports"

EXP_ROOT = ROOT / "exp"
REPRO_KAGGLE = ROOT / "00_new_codes/repro_kaggle/experiments/stage1_results"
REPRO_AUTODL = ROOT / "00_new_codes/repro_autodl/experiments/results"


def classify_pipeline(path: Path) -> str:
    s = str(path).replace("\\", "/")
    if "repro_kaggle" in s:
        return "hf_kaggle"
    if "repro_autodl" in s or "stage2" in s:
        return "hf_autodl_or_mixed"
    if s.startswith(str(EXP_ROOT).replace("\\", "/")):
        return "official_vllm_exp"
    return "other"


def infer_max_tokens(name: str) -> int | None:
    if "6144" in name or "6144" in name.lower():
        return 6144
    if "_512" in name or name.endswith("512"):
        return 512
    if "2048" in name:
        return 2048
    return None


def scan_exp_dir(d: Path) -> dict | None:
    ga = d / "generated_answer.json"
    em = d / "evaluation_metrics.json"
    if not ga.exists():
        return None
    try:
        data = json.loads(ga.read_text(encoding="utf-8"))
        n = len(data)
    except (json.JSONDecodeError, OSError):
        return None
    metrics = {}
    if em.exists():
        try:
            metrics = json.loads(em.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    task_match = re.search(r"reasoning_(\w+)|sttest_full_(\w+)", d.name)
    task = None
    if task_match:
        task = task_match.group(1) or task_match.group(2)
    return {
        "path": str(d.relative_to(ROOT)).replace("\\", "/"),
        "name": d.name,
        "task": task,
        "n_samples": n,
        "max_tokens_inferred": infer_max_tokens(d.name),
        "pipeline": classify_pipeline(d),
        "has_evaluation_metrics": em.exists(),
        "accuracy": metrics.get("accuracy"),
        "mae": metrics.get("mae"),
        "evaluated_samples": metrics.get("evaluated_samples"),
    }


def cited_in_reports(name: str) -> list[str]:
    hits = []
    for md in REPORTS.glob("*.md"):
        if name in md.read_text(encoding="utf-8", errors="ignore"):
            hits.append(md.name)
    return hits[:5]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for d in sorted(EXP_ROOT.iterdir()):
        if d.is_dir():
            row = scan_exp_dir(d)
            if row:
                row["cited_in_reports"] = cited_in_reports(d.name)
                entries.append(row)

    repro_dirs = []
    for base in [REPRO_KAGGLE, REPRO_AUTODL]:
        if base.exists():
            for p in base.rglob("evaluation_metrics.json"):
                repro_dirs.append(p.parent)
            for p in base.rglob("generated_answer.json"):
                repro_dirs.append(p.parent)
            for p in base.rglob("*summary.json"):
                repro_dirs.append(p.parent)

    seen = set()
    for d in repro_dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        ga = list(d.glob("generated_answer.json")) or list(d.glob("*predictions.jsonl"))
        if not ga and not list(d.glob("*.jsonl")):
            continue
        entries.append(
            {
                "path": str(d.relative_to(ROOT)).replace("\\", "/"),
                "name": d.name,
                "task": None,
                "n_samples": None,
                "max_tokens_inferred": infer_max_tokens(d.name),
                "pipeline": classify_pipeline(d),
                "has_evaluation_metrics": (d / "evaluation_metrics.json").exists(),
                "cited_in_reports": cited_in_reports(d.name),
            }
        )

    payload = {
        "exp_official_count": sum(1 for e in entries if e["pipeline"] == "official_vllm_exp"),
        "total_entries": len(entries),
        "entries": entries,
    }
    out_path = OUT / "experiment_registry.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
