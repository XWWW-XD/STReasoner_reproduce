#!/usr/bin/env python3
"""Download HF HEARTS via git clone + curl LFS resolve (hub snapshot fails behind proxy)."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

DATA_ROOT = Path("/root/autodl-tmp/datasets/HEARTS")
PREFLIGHT = DATA_ROOT / "preflight_result.json"
FROZEN_DIR = DATA_ROOT / "frozen_test_cases"
CODE_DIR = DATA_ROOT / "code"
PROXY = "http://127.0.0.1:17997"
MANIFEST_DEFAULT = Path(
    "/root/autodl-tmp/STReasoner_reproduce/00_new_codes/reports/new_datasets/artifacts/download_manifest.json"
)


def is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:20]
        return head.startswith(b"version https://git-lfs")
    except OSError:
        return False


def fetch_hf_siblings() -> list[str]:
    with urlopen("https://huggingface.co/api/datasets/yang-ai-lab/HEARTS", timeout=30) as resp:
        data = json.loads(resp.read().decode())
    paths = []
    for item in data.get("siblings") or []:
        name = item.get("rfilename") or item.get("path")
        if name and name.endswith(".pkl"):
            paths.append(name)
    return paths


def curl_download(relpath: str) -> None:
    url = f"https://huggingface.co/datasets/yang-ai-lab/HEARTS/resolve/main/{relpath}"
    dest = FROZEN_DIR / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "-x", PROXY, url, "-o", str(dest)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-out", type=Path, default=MANIFEST_DEFAULT)
    args = parser.parse_args()

    if not PREFLIGHT.exists():
        print("preflight_result.json missing", file=sys.stderr)
        sys.exit(1)
    preflight = json.loads(PREFLIGHT.read_text())
    if not preflight.get("pass"):
        print("preflight did not pass", file=sys.stderr)
        sys.exit(1)

    if not (FROZEN_DIR / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://huggingface.co/datasets/yang-ai-lab/HEARTS",
                str(FROZEN_DIR),
            ],
            check=True,
            env={**dict(__import__("os").environ), "https_proxy": PROXY, "http_proxy": PROXY},
        )

    pkl_paths = fetch_hf_siblings()
    if not pkl_paths:
        pkl_paths = [
            f"cgmacros/cgm_stat_calculation/{i}.pkl" for i in range(10)
        ]

    downloaded = []
    for rel in pkl_paths:
        dest = FROZEN_DIR / rel
        if not dest.exists() or is_lfs_pointer(dest):
            curl_download(rel)
        downloaded.append(rel)

    if preflight.get("clone_code") and not CODE_DIR.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/yang-ai-lab/HEARTS",
                str(CODE_DIR),
            ],
            check=True,
            env={**dict(__import__("os").environ), "https_proxy": PROXY, "http_proxy": PROXY},
        )

    pkl_files = list(FROZEN_DIR.rglob("*.pkl"))
    lfs_left = [str(p) for p in pkl_files if is_lfs_pointer(p)]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "git_clone + curl_resolve (huggingface_hub blocked by proxy/xet)",
        "frozen_dir": str(FROZEN_DIR),
        "pkl_count": len(pkl_files),
        "lfs_pointer_remaining": lfs_left,
        "total_bytes": sum(p.stat().st_size for p in pkl_files),
        "readme_exists": (FROZEN_DIR / "README.md").exists(),
        "code_cloned": CODE_DIR.exists(),
        "pkl_paths": [str(p.relative_to(FROZEN_DIR)) for p in sorted(pkl_files)],
        "downloaded_via_curl": downloaded,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if manifest["lfs_pointer_remaining"] or manifest["pkl_count"] < 1:
        print("download verification failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
