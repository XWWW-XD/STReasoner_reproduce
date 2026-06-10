#!/usr/bin/env python3
"""Run all phase2/3/CPU compute scripts; log exit codes (verification-before-completion)."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "compute_registry.py",
    "compute_token_budget.py",
    "compute_graph_ablation.py",
    "compute_eti_paper.py",
    "compute_kaggle_precision.py",
    "compute_code_phase3.py",
    "compute_e2_strict_reparse.py",
    "compute_t0_2.py",
    "compute_t0_3.py",
    "compute_wave25_real_flip.py",
    "compute_stratified_idx.py",
]
LOG = HERE.parent / "_plans" / "compute_run_log.txt"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# compute run {datetime.now(timezone.utc).isoformat()}", ""]
    failed = []
    for name in SCRIPTS:
        path = HERE / name
        if not path.exists():
            lines.append(f"SKIP {name}: missing")
            failed.append(name)
            continue
        r = subprocess.run([sys.executable, str(path)], cwd=str(HERE), capture_output=True, text=True)
        status = "OK" if r.returncode == 0 else "FAIL"
        lines.append(f"{status}\t{name}\texit={r.returncode}")
        if r.stdout.strip():
            lines.append(f"  stdout: {r.stdout.strip()[:200]}")
        if r.stderr.strip():
            lines.append(f"  stderr: {r.stderr.strip()[:200]}")
        if r.returncode != 0:
            failed.append(name)
    lines.append("")
    lines.append(f"SUMMARY: {len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} OK")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
