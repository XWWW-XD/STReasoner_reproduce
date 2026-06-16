#!/usr/bin/env python3
"""Task 0: disk + HF/GitHub openness and size preflight."""
import argparse
import json
import re
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path("/root/autodl-tmp/datasets/HEARTS")
MIN_AVAIL_GB = 0.5
BUFFER_MB = 50


def parse_avail_gb(df_line: str) -> float:
    parts = df_line.split()
    avail = parts[3]
    if avail.endswith("G"):
        return float(avail[:-1])
    if avail.endswith("T"):
        return float(avail[:-1]) * 1024
    if avail.endswith("M"):
        return float(avail[:-1]) / 1024
    return 0.0


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(DATA_ROOT),
        "pass": False,
        "clone_code": True,
    }

    df_out = subprocess.check_output(["df", "-h", "/root/autodl-tmp"], text=True)
    avail_line = [l for l in df_out.strip().splitlines() if "/root/autodl-tmp" in l][0]
    avail_gb = parse_avail_gb(avail_line)
    result["disk"] = {"df_line": avail_line, "avail_gb": avail_gb}

    hf = fetch_json("https://huggingface.co/api/datasets/yang-ai-lab/HEARTS")
    used_bytes = hf.get("usedStorage") or 0
    hf_mb = used_bytes / (1024 * 1024)
    result["hf"] = {
        "id": hf.get("id"),
        "private": hf.get("private"),
        "gated": hf.get("gated"),
        "disabled": hf.get("disabled"),
        "used_storage_bytes": used_bytes,
        "used_storage_mb": round(hf_mb, 3),
        "siblings_count": len(hf.get("siblings") or []),
    }

    gh = fetch_json("https://api.github.com/repos/yang-ai-lab/HEARTS")
    code_mb = (gh.get("size") or 0) / 1024
    result["github"] = {"size_kb": gh.get("size"), "size_mb": round(code_mb, 2)}

    estimated_mb = hf_mb + code_mb + BUFFER_MB
    result["estimated_total_mb"] = round(estimated_mb, 2)

    checks = {
        "disk_ok": avail_gb > MIN_AVAIL_GB,
        "hf_public": not hf.get("private") and not hf.get("gated"),
        "size_ok": estimated_mb < avail_gb * 1024,
    }
    result["checks"] = checks
    result["pass"] = all(checks.values())
    result["clone_code"] = result["pass"] and avail_gb > 1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
