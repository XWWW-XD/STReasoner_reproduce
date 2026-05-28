#!/usr/bin/env python3
"""Stage 2.2 paper-cases runner for AutoDL A100 fp16 single-GPU inference.

This script was copied from the stage2 SmartTest runner, then adapted to:
- use the Stage 1 paper-case dataset from the paper examples;
- run by paper-case sample_id or sample_index instead of forecasting/non_forecasting;
- use evaluation/evaluate_qa.py parsing helpers rather than strict local parsing;
- write one model text field named "response", matching evaluation/load_prediction_files.
"""

from __future__ import annotations

"""
    base_prediction_record()
    先建一个 record，里面字段默认 None
    ↓
    run_one_sample()
    Generate the model text and store it in response.
    ↓
    output_paths()
    Write to paper_cases_prediction.jsonl.
    ↓
    append_jsonl()
    一行一行追加写入结果文件
"""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = PROJECT_ROOT.parent
for import_root in (PROJECT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

REPRO_KAGGLE_ROOT = PROJECT_ROOT / "repro_kaggle"
REPRO_AUTODL_ROOT = PROJECT_ROOT / "repro_autodl"
SCRIPT_PATH = Path(__file__).resolve()

PAPER_CASE_SOURCE_DIR = (
    REPRO_KAGGLE_ROOT
    / "experiments/stage1_subsets/exp1_resource_tiny20/paper_cases"
)
PAPER_CASE_SOURCE_PATH = PAPER_CASE_SOURCE_DIR / "paper_cases_matched.jsonl"
STAGE22_DATA_DIR = REPRO_AUTODL_ROOT / "experiments/stage2_2_subsets/experiment1_paper_cases"
STAGE22_DATA_PATH = STAGE22_DATA_DIR / "paper_cases_matched.jsonl"
RESULT_ROOT = REPRO_AUTODL_ROOT / "experiments/stage2_2_paper_cases"
AUTHOR_EVALUATE_QA = REPO_ROOT / "evaluation/evaluate_qa.py"
SMOKE_PATCH_SOURCE = REPRO_KAGGLE_ROOT / "00_smoke_test_scripts/05_eval_sttest_tiny.py"

MODEL_NAME = "Time-HD-Anonymous/STReasoner-8B"
CONFIG_NAME = "stage2.2_fp16_a100_single"
DEFAULT_MAX_NEW_TOKENS = 6144
DEFAULT_AUTODL_CACHE = "/root/autodl-tmp/cache/huggingface"
DEFAULT_ATTN_BACKEND = "flash_attention_2"
PATCH_SIZE = 8


TASK_TO_OFFICIAL = {
    "forecasting": "reasoning_forecasting",
    "entity": "reasoning_entity",
    "etiological": "reasoning_etiological",
    "correlation": "reasoning_correlation",
    "causal": "reasoning_causal",
    "alignment": "alignment",
}


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._fh.write(message + "\n")
        self._fh.flush()

    def exception(self, title: str, exc: BaseException) -> None:
        self.log(f"[ERROR] {title}: {exc.__class__.__name__}: {exc}")
        self.log(traceback.format_exc())

    def close(self) -> None:
        self._fh.close()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)


def preview(value: Any, limit: int = 600) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def short_error(exc_or_message: BaseException | str, limit: int = 1500) -> str:
    text = str(exc_or_message)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def run_command_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}: {exc}"
    output = result.stdout.strip()
    if not output:
        output = "<empty>"
    if result.returncode != 0:
        output = f"(exit {result.returncode}) {output}"
    return output


def ensure_autodl_cache_env() -> dict[str, str]:
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
        if not os.environ.get(key):
            os.environ[key] = DEFAULT_AUTODL_CACHE

    if not os.environ.get("HF_DATASETS_CACHE"):
        os.environ["HF_DATASETS_CACHE"] = str(Path(os.environ["HF_HOME"]) / "datasets")
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(Path(os.environ["HF_HOME"]) / "torch")

    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    return {
        "HF_HOME": os.environ["HF_HOME"],
        "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
        "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
        "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
        "TORCH_HOME": os.environ["TORCH_HOME"],
    }


def ensure_single_gpu_env() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log_environment(logger: TeeLogger) -> None:
    logger.log("=== Stage 2.2 Environment Check ===")
    logger.log(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"pwd: {Path.cwd()}")
    logger.log(f"script: {rel(SCRIPT_PATH)}")
    logger.log(f"project_root: {PROJECT_ROOT}")
    logger.log(f"repo_root: {REPO_ROOT}")
    logger.log(f"git branch: {run_command_text(['git', 'branch', '--show-current'], REPO_ROOT)}")
    logger.log("git status --short:")
    logger.log(run_command_text(["git", "status", "--short"], REPO_ROOT))
    logger.log(f"python --version: {run_command_text([sys.executable, '--version'])}")

    try:
        import torch

        logger.log(f"torch.__version__: {torch.__version__}")
        logger.log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.log(f"torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            try:
                logger.log(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
            except Exception as exc:
                logger.log(f"torch.cuda.get_device_name(0): unavailable ({exc})")
        else:
            logger.log("torch.cuda.get_device_name(0): unavailable (CUDA is not available)")
    except Exception as exc:
        logger.log(f"torch environment unavailable: {exc.__class__.__name__}: {exc}")

    logger.log("nvidia-smi:")
    logger.log(
        run_command_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ]
        )
    )
    logger.log("df -h /root/autodl-tmp:")
    logger.log(run_command_text(["df", "-h", "/root/autodl-tmp"]))
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
        logger.log(f"{key}={os.environ.get(key)}")
    logger.log(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
    logger.log(f"TORCH_HOME={os.environ.get('TORCH_HOME')}")


def sample_task(sample: dict[str, Any]) -> str:
    return str(sample.get("task") or sample.get("category") or "unknown").lower()


def official_task_name(task: str) -> str:
    task_key = task.lower()
    if task_key not in TASK_TO_OFFICIAL:
        raise ValueError(f"Unsupported task for official evaluation: {task}")
    return TASK_TO_OFFICIAL[task_key]


def validate_sample(sample: dict[str, Any], row_name: str) -> None:
    for key in ("input", "timeseries", "output"):
        if key not in sample:
            raise ValueError(f"{row_name} missing field: {key}")
    prompt = sample["input"]
    timeseries = sample["timeseries"]
    if not isinstance(prompt, str):
        raise TypeError(f"{row_name} input must be a string")
    if not isinstance(timeseries, list):
        raise TypeError(f"{row_name} timeseries must be a list")
    placeholder_count = prompt.count("<ts><ts/>")
    if placeholder_count != len(timeseries):
        raise ValueError(
            f"{row_name} placeholder/timeseries mismatch: {placeholder_count} vs {len(timeseries)}"
        )
    official_task_name(sample_task(sample))


def validate_paper_case_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage 2.2 paper cases file is empty")
    for idx, sample in enumerate(rows):
        validate_sample(sample, f"paper_cases row {idx}")
    sample_ids = [row.get("sample_id") for row in rows]
    duplicate_ids = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"paper_cases has duplicate sample_id values: {duplicate_ids}")


def stage22_data_exists_and_valid(logger: TeeLogger) -> bool:
    if not STAGE22_DATA_PATH.exists():
        return False
    rows = load_jsonl(STAGE22_DATA_PATH)
    validate_paper_case_rows(rows)
    logger.log(f"Existing stage 2.2 paper cases file is valid: {rel(STAGE22_DATA_PATH)}")
    return True


def prepare_paper_cases(overwrite: bool, logger: TeeLogger) -> dict[str, Any]:
    ensure_autodl_cache_env()
    log_environment(logger)
    logger.log("=== Prepare Stage 2.2 Paper Cases ===")
    logger.log(f"source paper cases dir: {rel(PAPER_CASE_SOURCE_DIR)}")
    logger.log(f"source paper cases file: {rel(PAPER_CASE_SOURCE_PATH)}")
    logger.log(f"stage 2.2 paper cases file: {rel(STAGE22_DATA_PATH)}")

    if STAGE22_DATA_PATH.exists() and not overwrite:
        rows = load_jsonl(STAGE22_DATA_PATH)
        validate_paper_case_rows(rows)
        return {
            "paper_cases_path": rel(STAGE22_DATA_PATH),
            "source_path": rel(PAPER_CASE_SOURCE_PATH),
            "status": "existing_valid",
            "rows": len(rows),
            "task_counts": dict(Counter(sample_task(row) for row in rows)),
            "sample_ids": [row.get("sample_id") for row in rows],
        }

    if not PAPER_CASE_SOURCE_PATH.exists():
        raise FileNotFoundError(f"paper_cases source file not found: {PAPER_CASE_SOURCE_PATH}")
    selected = load_jsonl(PAPER_CASE_SOURCE_PATH)
    validate_paper_case_rows(selected)
    write_jsonl(STAGE22_DATA_PATH, selected)
    logger.log(f"Wrote stage 2.2 paper cases: {rel(STAGE22_DATA_PATH)}")
    return {
        "paper_cases_path": rel(STAGE22_DATA_PATH),
        "source_path": rel(PAPER_CASE_SOURCE_PATH),
        "status": "written",
        "selection_method": "copy_stage1_paper_cases_matched",
        "rows": len(selected),
        "task_counts": dict(Counter(sample_task(row) for row in selected)),
        "sample_ids": [row.get("sample_id") for row in selected],
    }


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_qa_module() -> Any:
    return load_module(AUTHOR_EVALUATE_QA, "stage22_author_evaluate_qa")


def gpu_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"visible_gpu_count": 0, "gpu_names": [], "gpu_total_memory": {}}

    names: list[str] = []
    totals: dict[str, float] = {}
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        names.append(props.name)
        totals[f"gpu{idx}"] = round(props.total_memory / 1024**3, 3)
    return {
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": names,
        "gpu_total_memory": totals,
    }


def gpu_memory_snapshot() -> dict[str, dict[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return {}
    snapshot: dict[str, dict[str, float]] = {}
    for idx in range(torch.cuda.device_count()):
        snapshot[f"gpu{idx}"] = {
            "allocated_gib": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
        }
    return snapshot


def gpu_peak_snapshot() -> dict[str, dict[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return {}
    snapshot: dict[str, dict[str, float]] = {}
    for idx in range(torch.cuda.device_count()):
        snapshot[f"gpu{idx}"] = {
            "max_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / 1024**3, 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved(idx) / 1024**3, 3),
        }
    return snapshot


def reset_gpu_peak_stats() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except RuntimeError:
            continue


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def first_model_device(model: Any) -> Any:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def collect_device_values(value: Any) -> list[str]:
    devices: list[str] = []
    if value is None:
        return devices
    if isinstance(value, dict):
        for item in value.values():
            devices.extend(collect_device_values(item))
        return devices
    if isinstance(value, (list, tuple, set)):
        for item in value:
            devices.extend(collect_device_values(item))
        return devices
    devices.append(str(value).lower())
    return devices


def model_distribution(actual_device_map: Any) -> dict[str, Any]:
    devices = collect_device_values(actual_device_map)
    cuda_devices = sorted({item for item in devices if item.startswith("cuda") or item.isdigit()})
    has_cpu = any(item == "cpu" for item in devices)
    has_disk = any("disk" in item for item in devices)
    if has_disk:
        label = "disk_offload"
    elif has_cpu:
        label = "cpu_offload"
    elif len(cuda_devices) >= 2:
        label = "multi_gpu"
    elif len(cuda_devices) == 1:
        label = "single_gpu"
    else:
        label = "unknown"
    return {
        "label": label,
        "devices": devices,
        "cuda_devices": cuda_devices,
        "has_cpu_offload": has_cpu,
        "has_disk_offload": has_disk,
    }


def class_name(obj: Any | None) -> str:
    return "None" if obj is None else obj.__class__.__name__


def check_flash_attn_environment(logger: TeeLogger) -> None:
    """Validate AutoDL flash_attn before importing remote model code."""

    logger.log("=== Flash Attention Check ===")
    try:
        from importlib import metadata

        logger.log(f"flash_attn package version: {metadata.version('flash_attn')}")
    except Exception as exc:
        raise RuntimeError(f"flash_attn package is not available: {exc}") from exc

    try:
        import flash_attn  # noqa: F401
        import flash_attn_2_cuda  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "flash_attn_import_failed: AutoDL flash_attn is installed, but its CUDA extension "
            f"cannot be imported in this Python/PyTorch/CUDA environment: {exc}"
        ) from exc

    logger.log("flash_attn import check: PASS")


def load_processor_and_tokenizer(model_name: str, cache_dir: str, logger: TeeLogger) -> tuple[Any | None, Any | None]:
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    tokenizer = None
    logger.log("=== Processor / Tokenizer ===")
    try:
        logger.log("Trying AutoProcessor.from_pretrained(...)")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.log(f"AutoProcessor loaded: {class_name(processor)}")
    except Exception as exc:
        logger.exception("AutoProcessor load failed", exc)

    try:
        logger.log("Trying AutoTokenizer.from_pretrained(...)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.log(f"AutoTokenizer loaded: {class_name(tokenizer)}")
    except Exception as exc:
        logger.exception("AutoTokenizer load failed", exc)

    return processor, tokenizer


def load_config(model_name: str, cache_dir: str, attn_backend: str, logger: TeeLogger) -> Any:
    from transformers import AutoConfig

    logger.log("=== Config ===")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    logger.log(f"config class: {class_name(config)}")
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(config, attr):
            logger.log(f"config.{attr}: {getattr(config, attr)!r} -> {attn_backend!r}")
            setattr(config, attr, attn_backend)
    return config


def assert_no_cpu_or_disk_offload(model: Any) -> None:
    actual_map = getattr(model, "hf_device_map", None)
    distribution = model_distribution(actual_map)
    if distribution["has_cpu_offload"] or distribution["has_disk_offload"]:
        raise RuntimeError(f"CPU/disk offload is not allowed in stage2: {distribution}")


def load_model_and_processors(
    logger: TeeLogger,
    attn_backend: str = DEFAULT_ATTN_BACKEND,
) -> tuple[Any, Any, Any, float, dict[str, Any]]:
    import torch

    if attn_backend == "flash_attention_2":
        check_flash_attn_environment(logger)

    from transformers import AutoModel, AutoModelForCausalLM

    cache_dir = os.environ["TRANSFORMERS_CACHE"]
    start = time.perf_counter()
    processor, tokenizer = load_processor_and_tokenizer(MODEL_NAME, cache_dir, logger)
    config = load_config(MODEL_NAME, cache_dir, attn_backend, logger)

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": 0},
        "torch_dtype": torch.float16,
        "config": config,
        "cache_dir": cache_dir,
    }

    logger.log("=== Run Layer: Model Load ===")
    logger.log(f"model: {MODEL_NAME}")
    logger.log(f"config: {CONFIG_NAME}")
    logger.log("precision: fp16")
    logger.log(f"attn_backend: {attn_backend}")
    logger.log("requested device_map: {'': 0}")
    logger.log("quantization_config: None")
    logger.log("cpu_offload: disabled")
    logger.log("disk_offload: disabled")

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    except Exception as first_exc:
        logger.exception("AutoModelForCausalLM load failed; trying AutoModel", first_exc)
        model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)

    model.eval()
    assert_no_cpu_or_disk_offload(model)
    patch_timeseries_merge_device(model, logger)

    load_time = time.perf_counter() - start
    actual_map = getattr(model, "hf_device_map", None)
    load_info = {
        "processor_class": class_name(processor),
        "tokenizer_class": class_name(tokenizer),
        "model_class": class_name(model),
        "requested_device_map": {"": 0},
        "actual_device_map": json_safe(actual_map),
        "model_distribution": model_distribution(actual_map),
        "use_cache": getattr(getattr(model, "config", None), "use_cache", None),
        "first_parameter_dtype": str(next(model.parameters()).dtype),
        "attn_backend": attn_backend,
        "timeseries_merge_patch_applied": bool(getattr(model, "_stage2_merge_patch", False)),
        "load_after_memory": gpu_memory_snapshot(),
    }
    logger.log(f"model_load_time_sec: {load_time:.3f}")
    logger.log(f"load_info: {json.dumps(json_safe(load_info), ensure_ascii=False)}")
    return model, processor, tokenizer, load_time, load_info


def patch_timeseries_merge_device(model: Any, logger: TeeLogger) -> None:
    """Copied minimally from repro_kaggle/00_smoke_test_scripts/05_eval_sttest_tiny.py.

    The patch keeps remote-code time-series merge tensors on the embedding
    device. It is harmless for single GPU and avoids known device placement
    failures without introducing offload or multi-GPU behavior.
    """

    if not hasattr(model, "_merge_input_ids_with_time_series_features") or getattr(model, "_stage2_merge_patch", False):
        return

    def patched_merge(time_series_features, inputs_embeds, input_ids, attention_mask, labels, patch_cnt):
        import torch

        target_device = inputs_embeds.device
        input_ids = input_ids.to(target_device) if hasattr(input_ids, "to") else input_ids
        attention_mask = attention_mask.to(target_device) if hasattr(attention_mask, "to") else attention_mask
        labels = labels.to(target_device) if labels is not None and hasattr(labels, "to") else labels
        patch_cnt = patch_cnt.to(target_device) if patch_cnt is not None and hasattr(patch_cnt, "to") else patch_cnt

        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = False
            else:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
        else:
            left_padding = bool(_left_padding and not _right_padding)

        special_ts_token_mask_start = input_ids == model.config.ts_token_start_index
        special_ts_token_mask_end = input_ids == model.config.ts_token_end_index
        special_ts_token_mask = special_ts_token_mask_start | special_ts_token_mask_end
        num_special_ts_tokens = torch.sum(special_ts_token_mask_start, dim=-1)
        _, embed_dim = time_series_features.shape

        patch_index = 0
        num_total_patches = torch.zeros(batch_size, dtype=patch_cnt.dtype, device=target_device)
        special_ts_token_mask_start_nonzero = special_ts_token_mask_start.nonzero()
        special_ts_token_mask_start_with_size = special_ts_token_mask_start.clone().long()

        attn_mask_cnt = attention_mask.sum(dim=-1).to(target_device)
        for i in range(batch_size):
            num_ts_in_batch = int(num_special_ts_tokens[i].item())
            num_total_patches[i] = patch_cnt[patch_index : patch_index + num_ts_in_batch].sum() - 2 * num_ts_in_batch
            for idx in range(patch_index, patch_index + num_ts_in_batch):
                b_idx, pos = special_ts_token_mask_start_nonzero[idx]
                special_ts_token_mask_start_with_size[b_idx, pos] *= int(patch_cnt[idx].item() - 2)
            patch_index += num_ts_in_batch
            attn_mask_cnt[i] += int(num_total_patches[i].item())

        max_embed_dim = int((sequence_length + num_total_patches.max()).item())
        batch_indices, non_ts_indices = torch.where(~special_ts_token_mask)
        new_token_positions = torch.cumsum((special_ts_token_mask_start_with_size + 1), dim=-1) - 1

        nb_ts_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_ts_pad[:, None]

        text_to_overwrite = new_token_positions[batch_indices, non_ts_indices]
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=target_device
        )
        final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=target_device)
        for i in range(attention_mask.size(0)):
            count = int(attn_mask_cnt[i].item())
            if left_padding:
                final_attention_mask[i, max_embed_dim - count :] = 1
            else:
                final_attention_mask[i, :count] = 1

        final_labels = None
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), model.config.ignore_index, dtype=input_ids.dtype, device=target_device
            )

        batch_indices = batch_indices.to(target_device)
        non_ts_indices = non_ts_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_ts_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_ts_indices]

        ts_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=target_device)
        ts_to_overwrite[batch_indices, text_to_overwrite] = False

        reversed_cumsum = ts_to_overwrite.flip(dims=[-1]).cumsum(-1).flip(dims=[-1]) - 1
        ts_to_overwrite &= reversed_cumsum >= nb_ts_pad[:, None].to(target_device)

        expected_ts_values = time_series_features.shape[:-1].numel()
        actual_ts_values = int(ts_to_overwrite.sum().item())
        if actual_ts_values != expected_ts_values:
            raise ValueError(
                "The input provided to the model is wrong. "
                f"time series slots={actual_ts_values}, time series features={expected_ts_values}, "
                f"special ts tokens={int(torch.sum(special_ts_token_mask_start).item())}, patch_cnt={len(patch_cnt)}."
            )

        final_embedding[ts_to_overwrite] = time_series_features.contiguous().reshape(-1, embed_dim).to(target_device)
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        if position_ids.size(-1) < input_ids.size(-1):
            position_ids = position_ids[:, -input_ids.size(-1) :]

        pad_batch_indices, pad_indices = torch.where(input_ids == model.config.pad_token_id)
        if len(pad_batch_indices) > 0:
            indices_to_mask = new_token_positions[pad_batch_indices, pad_indices]
            final_embedding[pad_batch_indices, indices_to_mask] = 0

        if new_token_positions.shape == attention_mask.shape:
            new_token_positions = new_token_positions.masked_fill(attention_mask == 0, -1)

        return final_embedding, final_attention_mask, position_ids, final_labels, new_token_positions

    setattr(model, "_merge_input_ids_with_time_series_features", patched_merge)
    setattr(model, "_stage2_merge_patch", True)
    logger.log(f"Applied time-series merge device patch from {rel(SMOKE_PATCH_SOURCE)}")


def estimate_timeseries_patch_tokens(timeseries: Any) -> int:
    if not isinstance(timeseries, list):
        return 0
    total = 0
    for series in timeseries:
        if isinstance(series, list):
            total += math.ceil(len(series) / PATCH_SIZE)
    return total


def build_inputs(processor: Any, tokenizer: Any, sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if processor is None:
        raise RuntimeError("AutoProcessor is unavailable; cannot build text + timeseries inputs.")

    prompt = sample.get("input")
    timeseries = sample.get("timeseries")
    if not isinstance(prompt, str):
        raise TypeError("sample['input'] must be a string")
    if not isinstance(timeseries, list):
        raise TypeError("sample['timeseries'] must be a list")
    placeholder_count = prompt.count("<ts><ts/>")
    if placeholder_count != len(timeseries):
        raise ValueError(
            f"placeholder/timeseries mismatch: placeholders={placeholder_count}, timeseries_nodes={len(timeseries)}"
        )

    inputs = processor(text=prompt, timeseries=timeseries, return_tensors="pt")
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    processor_input_tokens = int(input_ids.shape[-1]) if input_ids is not None and hasattr(input_ids, "shape") else None
    tokenizer_prompt_tokens = None
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokenizer_prompt_tokens = len(tokenizer.encode(prompt))
    ts_tokens = estimate_timeseries_patch_tokens(timeseries)
    input_token_metric = (processor_input_tokens or tokenizer_prompt_tokens or 0) + ts_tokens
    return inputs, {
        "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
        "processor_input_ids_length": processor_input_tokens,
        "timeseries_patch_tokens_estimate": ts_tokens,
        "input_tokens_metric": input_token_metric,
    }


def generated_outputs_to_response(outputs: Any, inputs: Any, processor: Any, tokenizer: Any) -> tuple[str | None, int | None, str | None]:
    text_converter = tokenizer if tokenizer is not None else processor
    if text_converter is None or not hasattr(text_converter, "decode"):
        return None, None, "no tokenizer/processor method available to convert generated ids to text"

    try:
        generated_ids = outputs[0]
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        actual_new_tokens = None
        if input_ids is not None and hasattr(input_ids, "shape"):
            actual_new_tokens = int(generated_ids.shape[-1] - input_ids.shape[-1])
            generated_ids = generated_ids[input_ids.shape[-1] :]
        text = text_converter.decode(generated_ids, skip_special_tokens=True)
        return text, actual_new_tokens, None
    except Exception as exc:
        return None, None, f"{exc.__class__.__name__}: {exc}"


def parse_model_answer(task: str, response: str | None) -> tuple[Any, bool, str | None]:
    """Parse with the same permissive helpers used by evaluation/evaluate_qa.py."""

    if response is None:
        return None, False, "no_response"

    evaluate_qa = evaluate_qa_module()
    extracted = evaluate_qa._extract_tag_content(str(response))
    task_key = task.lower()

    if task_key == "forecasting":
        parsed = evaluate_qa._parse_series(extracted)
        if parsed:
            return parsed, True, None
        return None, False, f"evaluate_qa_parse_series_empty: {preview(extracted, 120)}"

    if task_key in {"entity", "etiological", "correlation", "causal"}:
        parsed = evaluate_qa._normalize_choice(extracted)
        if str(parsed).upper() in {"A", "B", "C", "D"}:
            return str(parsed).upper(), True, None
        return parsed, False, f"evaluate_qa_normalize_choice_not_single_choice: {preview(extracted, 120)}"

    return extracted, bool(str(extracted).strip()), None if str(extracted).strip() else "empty_extracted_answer"


def canonical_formatted_answer(task: str, parsed_answer: Any, parse_success: bool) -> str | None:
    if not parse_success:
        return None
    task_key = task.lower()
    if task_key == "forecasting":
        if not isinstance(parsed_answer, list):
            return None
        values = [round(float(value), 6) for value in parsed_answer]
        return f"<answer>{json.dumps(values, ensure_ascii=False)}</answer>"
    if task_key in {"entity", "etiological", "correlation", "causal"}:
        value = str(parsed_answer).strip().upper()
        if value in {"A", "B", "C", "D"}:
            return f"<answer>{value}</answer>"
    if parsed_answer is not None:
        return f"<answer>{str(parsed_answer).strip()}</answer>"
    return None


def answer_format_diagnostics(task: str, response: str | None, parsed_answer: Any, parse_success: bool) -> dict[str, Any]:
    text = "" if response is None else str(response)
    answer_open_count = len(re.findall(r"<answer\b[^>]*>", text, flags=re.IGNORECASE))
    answer_close_count = len(re.findall(r"</answer>", text, flags=re.IGNORECASE))
    final_answer_open_count = len(re.findall(r"<final_answer\b[^>]*>", text, flags=re.IGNORECASE))
    final_answer_close_count = len(re.findall(r"</final_answer>", text, flags=re.IGNORECASE))
    has_exact_answer_tag = answer_open_count == 1 and answer_close_count == 1
    formatted_answer = canonical_formatted_answer(task, parsed_answer, parse_success)
    if has_exact_answer_tag:
        format_error = None
    elif not text.strip():
        format_error = "empty_response"
    elif answer_open_count or answer_close_count:
        format_error = f"answer_tag_count_mismatch: open={answer_open_count}, close={answer_close_count}"
    elif final_answer_open_count or final_answer_close_count:
        format_error = (
            "uses_final_answer_tag_instead_of_answer_tag: "
            f"open={final_answer_open_count}, close={final_answer_close_count}"
        )
    else:
        format_error = "missing_answer_tag"
    return {
        "format_success": has_exact_answer_tag,
        "format_error": format_error,
        "raw_answer_tag_open_count": answer_open_count,
        "raw_answer_tag_close_count": answer_close_count,
        "raw_final_answer_tag_open_count": final_answer_open_count,
        "raw_final_answer_tag_close_count": final_answer_close_count,
        "formatted_answer": formatted_answer,
    }


def failure_type_from(stage: str, generate_success: bool, parse_success: bool) -> str:
    if stage == "model_loading":
        return "load_failed"
    if not generate_success:
        return "generate_failed"
    if not parse_success:
        return "parse_failed"
    return "no_failure"


def output_paths(output_root: Path) -> dict[str, Path]:
    prefix = "paper_cases"
    return {
        "prediction": output_root / f"{prefix}_prediction.jsonl",
        "summary": output_root / f"{prefix}_summary.json",
        "log": output_root / f"{prefix}_run.log",
    }


def load_stage22_rows() -> list[dict[str, Any]]:
    if not STAGE22_DATA_PATH.exists():
        if not PAPER_CASE_SOURCE_PATH.exists():
            raise FileNotFoundError(
                f"stage 2.2 data not found: {STAGE22_DATA_PATH}; source also missing: {PAPER_CASE_SOURCE_PATH}"
            )
        rows = load_jsonl(PAPER_CASE_SOURCE_PATH)
        validate_paper_case_rows(rows)
        write_jsonl(STAGE22_DATA_PATH, rows)
    rows = load_jsonl(STAGE22_DATA_PATH)
    validate_paper_case_rows(rows)
    return rows


def list_sample_choices(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{idx}: sample_id={row.get('sample_id')} task={sample_task(row)} "
        f"paper_case_id={row.get('paper_case_id')}"
        for idx, row in enumerate(rows)
    )


def select_paper_case_sample(sample_index: int | None, sample_id: str | None) -> tuple[dict[str, Any], int]:
    rows = load_stage22_rows()
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(rows):
            raise IndexError(f"sample_index out of range: {sample_index}\n{list_sample_choices(rows)}")
        return rows[sample_index], sample_index
    if sample_id is not None:
        matches = [(idx, row) for idx, row in enumerate(rows) if row.get("sample_id") == sample_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one sample_id={sample_id}, got {len(matches)}\n{list_sample_choices(rows)}")
        return matches[0][1], matches[0][0]
    raise ValueError(f"Provide --sample-index or --sample-id.\n{list_sample_choices(rows)}")


def base_prediction_record(sample: dict[str, Any], sample_index: int, max_new_tokens: int) -> dict[str, Any]:
    env = gpu_environment()
    gpu_name = env.get("gpu_names", [None])[0] if env.get("gpu_names") else None
    gpu_total_memory = env.get("gpu_total_memory", {}).get("gpu0")
    return {
        "experiment": "stage 2.2",
        "config": CONFIG_NAME,
        "sample_id": sample.get("sample_id"),
        "original_index": sample.get("original_line_index"),
        "sample_index": sample_index,
        "paper_case_id": sample.get("paper_case_id"),
        "paper_location": sample.get("paper_location"),
        "task": sample_task(sample),
        "category": sample.get("category"),
        "source_file": sample.get("source_file"),
        "input_preview": preview(sample.get("input")),
        "input_tokens": None,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": None,
        "response": None,
        "generate_success": False,
        "generate_error": None,
        "gpu_name": gpu_name,
        "gpu_total_memory": gpu_total_memory,
        "gpu_memory_before_generate": None,
        "gpu_memory_after_generate": None,
        "gpu_peak_memory": None,
        "latency_sec": None,
        "tokens_per_sec": None,
        "parsed_answer": None,
        "gold_answer": sample.get("output"),
        "parse_success": False,
        "parse_error": None,
        "format_success": False,
        "format_error": None,
        "raw_answer_tag_open_count": None,
        "raw_answer_tag_close_count": None,
        "raw_final_answer_tag_open_count": None,
        "raw_final_answer_tag_close_count": None,
        "formatted_answer": None,
        "failure_type": "unknown",
    }


def official_metrics_for_record(sample: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    if record.get("response") is None:
        return {
            "official_task": official_task_name(sample_task(sample)),
            "skipped": True,
            "skip_reason": "missing_prediction",
        }

    evaluate_qa = evaluate_qa_module()
    dataset = [dict(sample, idx=0)]
    predictions = {0: record["response"]}
    metrics = evaluate_qa.evaluate_predictions_for_task(dataset, predictions, official_task_name(sample_task(sample)))
    metrics["author_evaluate_qa"] = rel(AUTHOR_EVALUATE_QA)
    return metrics


def explain_failure(first_error: str | None) -> dict[str, Any]:
    if not first_error:
        return {
            "plain_explanation": "No failure was recorded.",
            "likely_stage": "none",
            "next_step": None,
        }

    lower = first_error.lower()
    if "flash_attn_import_failed" in lower or "flash_attn_2_cuda" in lower or "undefined symbol" in lower:
        return {
            "plain_explanation": (
                "Model loading stopped before weights/generation because AutoDL flash_attn is installed, "
                "but its CUDA extension cannot be imported by the active Python/PyTorch/CUDA environment."
            ),
            "likely_stage": "flash_attn_import_before_model_load",
            "next_step": (
                "Check the installed flash_attn package against torch/Python/CUDA, then reinstall or rebuild "
                "a flash_attn version compatible with this AutoDL environment."
            ),
        }

    if "out of memory" in lower or "cuda oom" in lower or "outofmemoryerror" in lower:
        return {
            "plain_explanation": "CUDA ran out of GPU memory during model loading or generation.",
            "likely_stage": "cuda_memory",
            "next_step": "Check nvidia-smi and the run log memory snapshots before changing model settings.",
        }

    return {
        "plain_explanation": "The failure was recorded, but it does not match a known stage-specific pattern yet.",
        "likely_stage": "unknown",
        "next_step": "Send the full *_run.log for diagnosis.",
    }


def summarize_sample(
    sample: dict[str, Any],
    record: dict[str, Any],
    load_success: bool,
    load_error: str | None,
    load_time_sec: float | None,
    load_info: dict[str, Any] | None,
    official_metrics: dict[str, Any],
    output_root: Path,
    attn_backend: str,
) -> dict[str, Any]:
    first_error = load_error or record.get("generate_error") or record.get("parse_error")
    return {
        "experiment": "stage 2.2",
        "config": CONFIG_NAME,
        "model": MODEL_NAME,
        "batch_size": 1,
        "max_new_tokens": record.get("max_new_tokens"),
        "precision": "fp16",
        "attn_backend": attn_backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "paper_cases_path": rel(STAGE22_DATA_PATH),
        "source_paper_cases_path": rel(PAPER_CASE_SOURCE_PATH),
        "result_root": rel(output_root),
        "sample": {
            "sample_id": sample.get("sample_id"),
            "sample_index": record.get("sample_index"),
            "original_index": sample.get("original_line_index"),
            "task": sample_task(sample),
            "category": sample.get("category"),
            "source_file": sample.get("source_file"),
            "paper_case_id": sample.get("paper_case_id"),
            "paper_location": sample.get("paper_location"),
            "gold_answer": sample.get("output"),
        },
        "load": {
            "success": load_success,
            "error": load_error,
            "time_sec": round(load_time_sec, 3) if load_time_sec is not None else None,
            "info": load_info or {},
        },
        "run_metrics": {
            "generate_success": record.get("generate_success"),
            "parse_success": record.get("parse_success"),
            "format_success": record.get("format_success"),
            "format_error": record.get("format_error"),
            "input_tokens": record.get("input_tokens"),
            "actual_new_tokens": record.get("actual_new_tokens"),
            "latency_sec": record.get("latency_sec"),
            "tokens_per_sec": record.get("tokens_per_sec"),
            "gpu_memory_before_generate": record.get("gpu_memory_before_generate"),
            "gpu_memory_after_generate": record.get("gpu_memory_after_generate"),
            "gpu_peak_memory": record.get("gpu_peak_memory"),
        },
        "official_metrics": official_metrics,
        "failure_type": record.get("failure_type"),
        "first_error": first_error,
        "failure_explanation": explain_failure(first_error),
    }


def existing_completed(path: Path, sample: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    rows = load_jsonl(path)
    for row in rows:
        if row.get("sample_id") == sample.get("sample_id"):
            return True
    return False


def run_one_sample(
    sample_index: int | None,
    sample_id: str | None,
    max_new_tokens: int,
    overwrite: bool,
    resume: bool,
    output_root: Path,
    logger: TeeLogger,
    attn_backend: str = DEFAULT_ATTN_BACKEND,
    preloaded_model: Any | None = None,
    preloaded_processor: Any | None = None,
    preloaded_tokenizer: Any | None = None,
    preloaded_load_time_sec: float | None = None,
    preloaded_load_info: dict[str, Any] | None = None,
    log_env: bool = True,
) -> dict[str, Any]:
    ensure_autodl_cache_env()
    ensure_single_gpu_env()
    if log_env:
        log_environment(logger)
    logger.log("=== Stage 2.2 Paper Cases Run ===")
    logger.log(f"sample_index: {sample_index}")
    logger.log(f"sample_id: {sample_id}")
    logger.log(f"max_new_tokens: {max_new_tokens}")
    logger.log(f"attn_backend: {attn_backend}")
    logger.log(f"output_root: {rel(output_root)}")

    sample, selected_index = select_paper_case_sample(sample_index, sample_id)
    validate_sample(sample, f"paper_cases sample {selected_index}")
    paths = output_paths(output_root)

    if paths["prediction"].exists() and not overwrite:
        if resume and existing_completed(paths["prediction"], sample):
            logger.log(f"Resume enabled; existing prediction found: {rel(paths['prediction'])}")
            return {"status": "skipped_existing", "prediction_path": rel(paths["prediction"])}
        if not resume:
            raise FileExistsError(
                f"Output exists: {paths['prediction']}. Use --overwrite true or --resume true."
            )

    if overwrite:
        for key in ("prediction", "summary"):
            if paths[key].exists():
                paths[key].unlink()

    record = base_prediction_record(sample, selected_index, max_new_tokens)
    load_success = False
    load_error = None
    load_time_sec = None
    load_info: dict[str, Any] | None = None
    stage = "model_loading"

    if preloaded_model is not None:
        if preloaded_processor is None or preloaded_tokenizer is None:
            raise RuntimeError("preloaded model requires preloaded processor and tokenizer")
        model = preloaded_model
        processor = preloaded_processor
        tokenizer = preloaded_tokenizer
        load_time_sec = preloaded_load_time_sec
        load_info = preloaded_load_info
        load_success = True
        logger.log("Using preloaded model for this sample.")
    else:
        try:
            model, processor, tokenizer, load_time_sec, load_info = load_model_and_processors(logger, attn_backend)
            load_success = True
        except Exception as exc:
            load_error = f"{exc.__class__.__name__}: {short_error(exc)}"
            logger.exception("model loading failed", exc)
            record["generate_error"] = load_error
            record["failure_type"] = failure_type_from(stage, False, False)
            official_metrics = official_metrics_for_record(sample, record)
            append_jsonl(paths["prediction"], record)
            summary = summarize_sample(
                sample,
                record,
                load_success,
                load_error,
                load_time_sec,
                load_info,
                official_metrics,
                output_root,
                attn_backend,
            )
            write_json(paths["summary"], summary)
            return summary

    started = time.perf_counter()
    try:
        import torch

        stage = "processor"
        inputs, token_info = build_inputs(processor, tokenizer, sample)
        record["input_tokens"] = token_info["input_tokens_metric"]
        logger.log(f"token_info: {json.dumps(json_safe(token_info), ensure_ascii=False)}")

        inputs = move_inputs_to_device(inputs, first_model_device(model))
        record["gpu_memory_before_generate"] = gpu_memory_snapshot()
        sync_cuda()
        reset_gpu_peak_stats()

        stage = "generate"
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        sync_cuda()

        latency = time.perf_counter() - started
        record["latency_sec"] = round(latency, 3)
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        record["gpu_memory_after_generate"] = gpu_memory_snapshot()
        response_text, actual_new_tokens, response_error = generated_outputs_to_response(outputs, inputs, processor, tokenizer)
        record["actual_new_tokens"] = actual_new_tokens
        if actual_new_tokens is not None and latency > 0:
            record["tokens_per_sec"] = round(actual_new_tokens / latency, 3)

        if response_error is not None:
            raise RuntimeError(response_error)
        record["generate_success"] = True
        record["response"] = response_text

        parsed_answer, parse_success, parse_error = parse_model_answer(sample_task(sample), response_text)
        record["parsed_answer"] = parsed_answer
        record["parse_success"] = parse_success
        record["parse_error"] = parse_error
        record.update(answer_format_diagnostics(sample_task(sample), response_text, parsed_answer, parse_success))
        record["failure_type"] = failure_type_from(
            stage,
            bool(record["generate_success"]),
            bool(record["parse_success"]),
        )
    except Exception as exc:
        try:
            sync_cuda()
        except Exception:
            pass
        latency = time.perf_counter() - started
        message = f"{exc.__class__.__name__}: {short_error(exc)}"
        logger.exception(f"{stage} failed", exc)
        record["latency_sec"] = round(latency, 3)
        record["gpu_memory_after_generate"] = gpu_memory_snapshot()
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        record["generate_error"] = message
        record["failure_type"] = failure_type_from(stage, bool(record["generate_success"]), False)

    official_metrics = official_metrics_for_record(sample, record)
    append_jsonl(paths["prediction"], record)
    summary = summarize_sample(
        sample,
        record,
        load_success,
        load_error,
        load_time_sec,
        load_info,
        official_metrics,
        output_root,
        attn_backend,
    )
    write_json(paths["summary"], summary)
    logger.log(f"Wrote prediction: {rel(paths['prediction'])}")
    logger.log(f"Wrote summary: {rel(paths['summary'])}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2.2 paper-cases fp16 A100 single-GPU runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Copy and validate the Stage 1 paper_cases dataset.")
    prepare.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False)

    subparsers.add_parser("list", help="List available Stage 2.2 paper-case samples.")

    run = subparsers.add_parser("run", help="Run exactly one paper-case sample.")
    selector = run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--sample-index", type=int, default=None)
    selector.add_argument("--sample-id", type=str, default=None)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False)
    run.add_argument("--resume", type=str2bool, nargs="?", const=True, default=False)
    run.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    run.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default=DEFAULT_ATTN_BACKEND,
    )

    run_all = subparsers.add_parser("run-all", help="Run all Stage 2.2 paper-case samples sequentially.")
    run_all.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run_all.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False)
    run_all.add_argument("--resume", type=str2bool, nargs="?", const=True, default=False)
    run_all.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    run_all.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default=DEFAULT_ATTN_BACKEND,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        logger = TeeLogger(STAGE22_DATA_DIR / "prepare.log")
        try:
            payload = prepare_paper_cases(overwrite=args.overwrite, logger=logger)
            write_json(STAGE22_DATA_DIR / "prepare_summary.json", payload)
            logger.log(f"prepare_summary: {json.dumps(json_safe(payload), ensure_ascii=False)}")
            return 0
        finally:
            logger.close()

    if args.command == "list":
        rows = load_stage22_rows()
        print(list_sample_choices(rows))
        return 0

    if args.command == "run":
        paths = output_paths(args.output_root)
        logger = TeeLogger(paths["log"])
        try:
            summary = run_one_sample(
                sample_index=args.sample_index,
                sample_id=args.sample_id,
                max_new_tokens=args.max_new_tokens,
                overwrite=args.overwrite,
                resume=args.resume,
                output_root=args.output_root,
                logger=logger,
                attn_backend=args.attn_backend,
            )
            logger.log(f"run_summary: {json.dumps(json_safe(summary), ensure_ascii=False)}")
            return 0
        finally:
            logger.close()

    if args.command == "run-all":
        paths = output_paths(args.output_root)
        if paths["prediction"].exists() and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"Output exists: {paths['prediction']}. Use --overwrite true or --resume true."
            )
        if args.overwrite:
            for key in ("prediction", "summary"):
                if paths[key].exists():
                    paths[key].unlink()

        rows = load_stage22_rows()
        logger = TeeLogger(paths["log"])
        try:
            ensure_autodl_cache_env()
            ensure_single_gpu_env()
            log_environment(logger)
            logger.log("=== Stage 2.2 Paper Cases Run-All ===")
            logger.log(f"rows: {len(rows)}")
            logger.log(f"max_new_tokens: {args.max_new_tokens}")
            logger.log(f"attn_backend: {args.attn_backend}")
            logger.log(f"output_root: {rel(args.output_root)}")
            model, processor, tokenizer, load_time_sec, load_info = load_model_and_processors(logger, args.attn_backend)

            summaries = []
            for index, row in enumerate(rows):
                logger.log(f"=== Stage 2.2 run-all sample {index + 1}/{len(rows)}: {row.get('sample_id')} ===")
                summary = run_one_sample(
                    sample_index=index,
                    sample_id=None,
                    max_new_tokens=args.max_new_tokens,
                    overwrite=False,
                    resume=True,
                    output_root=args.output_root,
                    logger=logger,
                    attn_backend=args.attn_backend,
                    preloaded_model=model,
                    preloaded_processor=processor,
                    preloaded_tokenizer=tokenizer,
                    preloaded_load_time_sec=load_time_sec,
                    preloaded_load_info=load_info,
                    log_env=False,
                )
                summaries.append(summary)
            aggregate = {
                "experiment": "stage 2.2",
                "rows": len(rows),
                "prediction_path": rel(paths["prediction"]),
                "summaries": summaries,
            }
            write_json(paths["summary"], aggregate)
            logger.log(f"run_all_summary: {json.dumps(json_safe(aggregate), ensure_ascii=False)}")
            return 0
        finally:
            logger.close()

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
