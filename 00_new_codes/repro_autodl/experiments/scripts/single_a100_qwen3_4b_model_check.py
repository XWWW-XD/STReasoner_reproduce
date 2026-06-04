#!/usr/bin/env python3
"""Check Qwen3-4B local files before STReasoner TS initialization."""

from __future__ import annotations

import json
import os
from pathlib import Path


MODEL_DIR = Path("/root/autodl-tmp/STReasoner_reproduce/base_model/Qwen3-4B-Instruct-2507")
REQUIRED = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]


def main() -> None:
    print(f"model_dir={MODEL_DIR}")
    missing = []
    for name in REQUIRED:
        path = MODEL_DIR / name
        if not path.is_file():
            missing.append(name)
            print(f"missing {name}")
        else:
            print(f"ok      {name} {path.stat().st_size}")

    index_path = MODEL_DIR / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        print(f"index_shards={shards}")
        for shard in shards:
            path = MODEL_DIR / shard
            print(f"index_shard_present {shard}={path.is_file()}")
            if not path.is_file():
                missing.append(shard)

    cache_dir = MODEL_DIR / ".cache" / "huggingface" / "download"
    if cache_dir.is_dir():
        incomplete = sorted(p.name for p in cache_dir.glob("*.incomplete"))
        locks = sorted(p.name for p in cache_dir.glob("*.lock"))
        print(f"incomplete_files={incomplete}")
        print(f"lock_files={locks}")

    print("HF_HOME=" + os.environ.get("HF_HOME", ""))
    print("TRANSFORMERS_CACHE=" + os.environ.get("TRANSFORMERS_CACHE", ""))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
