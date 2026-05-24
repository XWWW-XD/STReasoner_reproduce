#!/usr/bin/env python3
"""Print a lightweight Kaggle GPU/Python dependency environment report."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_config import apply_cache_config

apply_cache_config()


def print_header(title: str) -> None:
    print(f"\n==> {title}")


def check_import(module_name: str, warning_only: bool = False) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report import failures without hiding context.
        level = "WARNING" if warning_only else "MISSING"
        print(f"{level}: {module_name} import failed: {exc}")
        return

    version = getattr(module, "__version__", "version unknown")
    print(f"OK: {module_name} import succeeded ({version})")


def print_torch_info() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch failures should be visible in the report.
        print(f"MISSING: torch import failed: {exc}")
        return

    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"cuda available: {cuda_available}")

    gpu_count = torch.cuda.device_count() if cuda_available else 0
    print(f"gpu count: {gpu_count}")

    for index in range(gpu_count):
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        print(f"gpu {index}: {name} ({total_gb:.2f} GiB)")


def main() -> int:
    print_header("Python")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")

    print_header("Torch / CUDA / GPU")
    print_torch_info()

    print_header("Package imports")
    for module_name in (
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "vllm",
    ):
        check_import(module_name)

    check_import("flash_attn", warning_only=True)
    print("NOTE: flash_attn being unavailable is expected for this Kaggle setup.")

    print_header("Cache paths")
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"):
        print(f"{key}={os.environ.get(key)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
