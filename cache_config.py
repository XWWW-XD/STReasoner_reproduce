"""Central cache configuration for Hugging Face, datasets, and torch.

Import this module before importing transformers, datasets, huggingface_hub,
or torch. The module applies the cache environment variables immediately.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachePaths:
    hf_home: str
    hf_hub_cache: str
    transformers_cache: str
    hf_datasets_cache: str
    torch_home: str

    def as_env(self) -> dict[str, str]:
        return {
            "HF_HOME": self.hf_home,
            "HF_HUB_CACHE": self.hf_hub_cache,
            "TRANSFORMERS_CACHE": self.transformers_cache,
            "HF_DATASETS_CACHE": self.hf_datasets_cache,
            "TORCH_HOME": self.torch_home,
        }


def _is_windows() -> bool:
    return os.name == "nt" or platform.system().lower() == "windows"


def select_cache_paths() -> CachePaths:
    if _is_windows():
        return CachePaths(
            hf_home=r"D:\hf_cache",
            hf_hub_cache=r"D:\hf_cache\hub",
            transformers_cache=r"D:\hf_cache\transformers",
            hf_datasets_cache=r"D:\hf_cache\datasets",
            torch_home=r"D:\torch_cache",
        )

    return CachePaths(
        hf_home="/kaggle/working/hf_cache",
        hf_hub_cache="/kaggle/working/hf_cache/hub",
        transformers_cache="/kaggle/working/hf_cache/transformers",
        hf_datasets_cache="/kaggle/working/hf_cache/datasets",
        torch_home="/kaggle/working/torch_cache",
    )


def _validate_no_c_drive(paths: CachePaths) -> None:
    for key, value in paths.as_env().items():
        if is_forbidden_c_drive_path(value):
            raise RuntimeError(f"{key} points to forbidden C drive cache path: {value}")


def is_forbidden_c_drive_path(path: str | os.PathLike[str] | None) -> bool:
    if path is None:
        return False
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(path).strip()))
    normalized = expanded.replace("/", "\\").lower()
    return normalized.startswith("c:")


def reject_forbidden_cache_dir(path: str | os.PathLike[str] | None, label: str = "cache_dir") -> str | None:
    if is_forbidden_c_drive_path(path):
        raise RuntimeError(f"{label} points to forbidden C drive cache path: {path}")
    return os.fspath(path) if path is not None else None


def resolve_hub_cache_dir(path: str | os.PathLike[str] | None = None) -> str:
    return reject_forbidden_cache_dir(path, "hub cache_dir") or HF_HUB_CACHE_PATH


def resolve_transformers_cache_dir(path: str | os.PathLike[str] | None = None) -> str:
    return reject_forbidden_cache_dir(path, "transformers cache_dir") or TRANSFORMERS_CACHE_PATH


def resolve_datasets_cache_dir(path: str | os.PathLike[str] | None = None) -> str:
    return reject_forbidden_cache_dir(path, "datasets cache_dir") or HF_DATASETS_CACHE_PATH


def apply_cache_config(create_dirs: bool = True) -> CachePaths:
    paths = select_cache_paths()
    _validate_no_c_drive(paths)

    for key, value in paths.as_env().items():
        os.environ[key] = value

    if create_dirs:
        for value in paths.as_env().values():
            Path(value).mkdir(parents=True, exist_ok=True)

    return paths


CACHE_PATHS = apply_cache_config()
HF_HOME_PATH = CACHE_PATHS.hf_home
HF_HUB_CACHE_PATH = CACHE_PATHS.hf_hub_cache
TRANSFORMERS_CACHE_PATH = CACHE_PATHS.transformers_cache
HF_DATASETS_CACHE_PATH = CACHE_PATHS.hf_datasets_cache
TORCH_HOME_PATH = CACHE_PATHS.torch_home
