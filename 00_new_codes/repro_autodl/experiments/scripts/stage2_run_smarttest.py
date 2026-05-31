#!/usr/bin/env python3
"""Stage2 SmartTest runner for AutoDL A100 fp16 single-GPU inference.

This script is intentionally minimal:
- prepare a 2-row stage2 SmartTest file;
- run exactly one requested SmartTest case at a time;
- record run, resource, parser, and official single-sample metrics.

No stage1 outputs, tiny20 full run, paper cases, stress cases, or run-all entry
points are used here.
"""

from __future__ import annotations

"""
    base_prediction_record()
    先建一个 record，里面字段默认 None
    ↓
    run_one_case()
    生成 decoded，并填入 decoded_text / raw_response
    ↓
    output_paths()
    统一写到 predictions.jsonl / summary.json / run.log
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPRO_KAGGLE_ROOT = PROJECT_ROOT / "repro_kaggle"
REPRO_AUTODL_ROOT = PROJECT_ROOT / "repro_autodl"
SCRIPT_PATH = Path(__file__).resolve()

SOURCE_TINY20 = (
    REPRO_KAGGLE_ROOT
    / "experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl"
)
LEGACY_SMARTTEST = (
    REPRO_KAGGLE_ROOT
    / "experiments/stage1_subsets/exp1_resource_tiny20/smart_test/SmartTest.jsonl"
)
SMARTTEST_DIR = REPRO_AUTODL_ROOT / "experiments/stage2_subsets/experiment1_smart_test"
SMARTTEST_PATH = SMARTTEST_DIR / "SmartTest.jsonl"
RESULT_ROOT = REPRO_AUTODL_ROOT / "experiments/stage2_results/experiment1_smarttest"
AUTHOR_EVALUATE_QA = PROJECT_ROOT / "evaluation/evaluate_qa.py"
SMOKE_PATCH_SOURCE = REPRO_KAGGLE_ROOT / "00_smoke_test_scripts/05_eval_sttest_tiny.py"

MODEL_NAME = "Time-HD-Anonymous/STReasoner-8B"
CONFIG_NAME = "fp16_a100_single"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_AUTODL_CACHE = "/cloud/cloud-ssd1/hf_cache"
DEFAULT_ATTN_BACKEND = "flash_attention_2"
PATCH_SIZE = 8

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STRICT_CHOICE_RE = re.compile(r"^[A-Da-d]$")

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


def upsert_jsonl(path: Path, payload: dict[str, Any], identity_keys: tuple[str, ...]) -> None:
    rows = load_jsonl(path) if path.exists() else []
    rows = [
        row
        for row in rows
        if any(row.get(key) != payload.get(key) for key in identity_keys)
    ]
    rows.append(payload)
    write_jsonl(path, rows)


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
    logger.log("=== Stage2 Environment Check ===")
    logger.log(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"pwd: {Path.cwd()}")
    logger.log(f"script: {rel(SCRIPT_PATH)}")
    logger.log(f"project_root: {PROJECT_ROOT}")
    logger.log(f"git branch: {run_command_text(['git', 'branch', '--show-current'], PROJECT_ROOT)}")
    logger.log("git status --short:")
    logger.log(run_command_text(["git", "status", "--short"], PROJECT_ROOT))
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
    logger.log("df -h /cloud/cloud-ssd1:")
    logger.log(run_command_text(["df", "-h", "/cloud/cloud-ssd1"]))
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
        logger.log(f"{key}={os.environ.get(key)}")
    logger.log(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
    logger.log(f"TORCH_HOME={os.environ.get('TORCH_HOME')}")


def sample_task(sample: dict[str, Any]) -> str:
    return str(sample.get("task") or sample.get("category") or "unknown").lower()


def case_for_sample(sample: dict[str, Any]) -> str:
    return "forecasting" if sample_task(sample) == "forecasting" else "non_forecasting"


def sample_identity(sample: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (sample.get("sample_id"), sample.get("source_file"), sample.get("original_line_index"))


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


def validate_smarttest_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 2:
        raise ValueError(f"SmartTest expected 2 rows, got {len(rows)}")
    for idx, sample in enumerate(rows):
        validate_sample(sample, f"SmartTest row {idx}")
    counts = Counter(case_for_sample(row) for row in rows)
    if counts.get("forecasting") != 1 or counts.get("non_forecasting") != 1:
        raise ValueError(f"SmartTest expected 1 forecasting and 1 non_forecasting, got {dict(counts)}")


def stage2_smarttest_exists_and_valid(logger: TeeLogger) -> bool:
    if not SMARTTEST_PATH.exists():
        return False
    rows = load_jsonl(SMARTTEST_PATH)
    validate_smarttest_rows(rows)
    logger.log(f"Existing stage2 SmartTest is valid: {rel(SMARTTEST_PATH)}")
    return True


def prepare_smarttest(overwrite: bool, logger: TeeLogger) -> dict[str, Any]:
    ensure_autodl_cache_env()
    log_environment(logger)
    logger.log("=== Prepare SmartTest ===")
    logger.log(f"source tiny20: {rel(SOURCE_TINY20)}")
    logger.log(f"legacy SmartTest: {rel(LEGACY_SMARTTEST)}")
    logger.log(f"stage2 SmartTest: {rel(SMARTTEST_PATH)}")

    if SMARTTEST_PATH.exists() and not overwrite:
        rows = load_jsonl(SMARTTEST_PATH)
        validate_smarttest_rows(rows)
        return {
            "smarttest_path": rel(SMARTTEST_PATH),
            "status": "existing_valid",
            "rows": len(rows),
            "case_counts": dict(Counter(case_for_sample(row) for row in rows)),
            "sample_ids": [row.get("sample_id") for row in rows],
        }

    if not SOURCE_TINY20.exists():
        raise FileNotFoundError(f"tiny20 source file not found: {SOURCE_TINY20}")
    source_rows = load_jsonl(SOURCE_TINY20)
    source_keys = {sample_identity(row) for row in source_rows}

    selected: list[dict[str, Any]] | None = None
    selection_method = "legacy_smarttest"
    if LEGACY_SMARTTEST.exists():
        legacy_rows = load_jsonl(LEGACY_SMARTTEST)
        validate_smarttest_rows(legacy_rows)
        missing = [sample_identity(row) for row in legacy_rows if sample_identity(row) not in source_keys]
        if missing:
            raise ValueError(f"legacy SmartTest rows not found in tiny20 source: {missing}")
        selected = legacy_rows
    else:
        selection_method = "fallback_first_case_by_tiny20_order"
        forecasting = next((row for row in source_rows if case_for_sample(row) == "forecasting"), None)
        non_forecasting = next((row for row in source_rows if case_for_sample(row) == "non_forecasting"), None)
        if forecasting is None or non_forecasting is None:
            raise ValueError("Cannot find both forecasting and non_forecasting rows in tiny20 source.")
        selected = [forecasting, non_forecasting]

    validate_smarttest_rows(selected)
    write_jsonl(SMARTTEST_PATH, selected)
    logger.log(f"Wrote stage2 SmartTest: {rel(SMARTTEST_PATH)}")
    return {
        "smarttest_path": rel(SMARTTEST_PATH),
        "status": "written",
        "selection_method": selection_method,
        "rows": len(selected),
        "case_counts": dict(Counter(case_for_sample(row) for row in selected)),
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
    return load_module(AUTHOR_EVALUATE_QA, "stage2_author_evaluate_qa")


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


def decode_outputs(outputs: Any, inputs: Any, processor: Any, tokenizer: Any) -> tuple[str | None, int | None, str | None]:
    decoder = tokenizer if tokenizer is not None else processor
    if decoder is None or not hasattr(decoder, "decode"):
        return None, None, "no decoder with decode(...)"

    try:
        generated_ids = outputs[0]
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        actual_new_tokens = None
        if input_ids is not None and hasattr(input_ids, "shape"):
            actual_new_tokens = int(generated_ids.shape[-1] - input_ids.shape[-1])
            generated_ids = generated_ids[input_ids.shape[-1] :]
        text = decoder.decode(generated_ids, skip_special_tokens=True)
        return text, actual_new_tokens, None
    except Exception as exc:
        return None, None, f"{exc.__class__.__name__}: {exc}"


# Parser copied minimally from
# repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py.
def strict_parse_forecasting(content: str) -> tuple[list[float] | None, str | None]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"answer_tag_content_not_json: {exc}"
    if not isinstance(parsed, list):
        return None, "answer_tag_content_not_list"
    values: list[float] = []
    for item in parsed:
        if not isinstance(item, (int, float)):
            return None, "forecast_list_contains_non_numeric_value"
        values.append(float(item))
    return values, None


def parse_model_answer(task: str, decoded_text: str | None) -> tuple[Any, bool, str | None]:
    if decoded_text is None:
        return None, False, "no_decoded_text"

    matches = list(ANSWER_TAG_RE.finditer(decoded_text))
    if len(matches) != 1:
        return None, False, f"expected_exactly_one_answer_tag_got_{len(matches)}"

    content = matches[0].group(1).strip()
    if task.lower() == "forecasting":
        parsed, error = strict_parse_forecasting(content)
        return parsed, error is None, error

    if STRICT_CHOICE_RE.match(content):
        return content.upper(), True, None
    return None, False, f"answer_tag_content_not_single_choice: {preview(content, 120)}"


def failure_type_from(stage: str, generate_success: bool, decode_success: bool, parse_success: bool) -> str:
    if stage == "model_loading":
        return "load_failed"
    if not generate_success:
        return "generate_failed"
    if not decode_success:
        return "decode_failed"
    if not parse_success:
        return "parse_failed"
    return "no_failure"


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "prediction": output_root / "predictions.jsonl",
        "summary": output_root / "summary.json",
        "log": output_root / "run.log",
    }


def select_case_sample(case: str) -> tuple[dict[str, Any], int]:
    if not SMARTTEST_PATH.exists():
        raise FileNotFoundError(f"SmartTest not found; run prepare first: {SMARTTEST_PATH}")
    rows = load_jsonl(SMARTTEST_PATH)
    validate_smarttest_rows(rows)
    matches = [(idx, row) for idx, row in enumerate(rows) if case_for_sample(row) == case]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {case} row, got {len(matches)}")
    idx, sample = matches[0]
    return sample, idx


def base_prediction_record(sample: dict[str, Any], case: str, smarttest_index: int, max_new_tokens: int) -> dict[str, Any]:
    env = gpu_environment()
    gpu_name = env.get("gpu_names", [None])[0] if env.get("gpu_names") else None
    gpu_total_memory = env.get("gpu_total_memory", {}).get("gpu0")
    return {
        "case": case,
        "config": CONFIG_NAME,
        "sample_id": sample.get("sample_id"),
        "original_index": sample.get("original_line_index"),
        "smarttest_index": smarttest_index,
        "task": sample_task(sample),
        "category": sample.get("category"),
        "source_file": sample.get("source_file"),
        "input_preview": preview(sample.get("input")),
        "input_tokens": None,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": None,
        "raw_response": None,
        "decoded_text": None,
        "generate_success": False,
        "decode_success": False,
        "generate_error": None,
        "gpu_name": gpu_name,
        "gpu_total_memory": gpu_total_memory,
        "gpu_peak_memory": None,
        "latency_sec": None,
        "tokens_per_sec": None,
        "parsed_answer": None,
        "gold_answer": sample.get("output"),
        "parse_success": False,
        "parse_error": None,
        "failure_type": "unknown",
    }


def official_metrics_for_record(sample: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    if not record.get("decode_success") or record.get("decoded_text") is None:
        return {
            "official_task": official_task_name(sample_task(sample)),
            "skipped": True,
            "skip_reason": "decode_failed_or_missing_prediction",
        }

    evaluate_qa = evaluate_qa_module()
    dataset = [dict(sample, idx=0)]
    predictions = {0: record["decoded_text"]}
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

    if "decode" in lower:
        return {
            "plain_explanation": "Generation may have completed, but decoding the generated token ids failed.",
            "likely_stage": "decode",
            "next_step": "Send the run log and prediction jsonl so the decoder/output tensor path can be checked.",
        }

    return {
        "plain_explanation": "The failure was recorded, but it does not match a known stage-specific pattern yet.",
        "likely_stage": "unknown",
        "next_step": "Send the full *_run.log for diagnosis.",
    }


def summarize_case(
    case: str,
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
        "case": case,
        "config": CONFIG_NAME,
        "model": MODEL_NAME,
        "batch_size": 1,
        "max_new_tokens": record.get("max_new_tokens"),
        "precision": "fp16",
        "attn_backend": attn_backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "smarttest_path": rel(SMARTTEST_PATH),
        "result_root": rel(output_root),
        "sample": {
            "sample_id": sample.get("sample_id"),
            "original_index": sample.get("original_line_index"),
            "task": sample_task(sample),
            "category": sample.get("category"),
            "source_file": sample.get("source_file"),
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
            "decode_success": record.get("decode_success"),
            "parse_success": record.get("parse_success"),
            "input_tokens": record.get("input_tokens"),
            "actual_new_tokens": record.get("actual_new_tokens"),
            "latency_sec": record.get("latency_sec"),
            "tokens_per_sec": record.get("tokens_per_sec"),
            "gpu_peak_memory": record.get("gpu_peak_memory"),
        },
        "official_metrics": official_metrics,
        "failure_type": record.get("failure_type"),
        "first_error": first_error,
        "failure_explanation": explain_failure(first_error),
    }


def existing_completed(path: Path, sample: dict[str, Any], case: str) -> bool:
    if not path.exists():
        return False
    rows = load_jsonl(path)
    for row in rows:
        if row.get("case") == case and row.get("sample_id") == sample.get("sample_id"):
            return True
    return False


def run_one_case(
    case: str,
    max_new_tokens: int,
    overwrite: bool,
    resume: bool,
    output_root: Path,
    logger: TeeLogger,
    attn_backend: str = DEFAULT_ATTN_BACKEND,
) -> dict[str, Any]:
    ensure_autodl_cache_env()
    ensure_single_gpu_env()
    log_environment(logger)
    logger.log("=== Stage2 SmartTest Run ===")
    logger.log(f"case: {case}")
    logger.log(f"max_new_tokens: {max_new_tokens}")
    logger.log(f"attn_backend: {attn_backend}")
    logger.log(f"output_root: {rel(output_root)}")

    sample, smarttest_index = select_case_sample(case)
    validate_sample(sample, f"{case} sample")
    paths = output_paths(output_root)

    if paths["prediction"].exists() and not overwrite:
        if resume and existing_completed(paths["prediction"], sample, case):
            logger.log(f"Resume enabled; existing prediction found: {rel(paths['prediction'])}")
            return {"status": "skipped_existing", "prediction_path": rel(paths["prediction"])}

    if overwrite:
        logger.log("Overwrite enabled; replacing only this case/sample in the unified prediction file.")

    record = base_prediction_record(sample, case, smarttest_index, max_new_tokens)
    load_success = False
    load_error = None
    load_time_sec = None
    load_info: dict[str, Any] | None = None
    stage = "model_loading"

    try:
        model, processor, tokenizer, load_time_sec, load_info = load_model_and_processors(logger, attn_backend)
        load_success = True
    except Exception as exc:
        load_error = f"{exc.__class__.__name__}: {short_error(exc)}"
        logger.exception("model loading failed", exc)
        record["generate_error"] = load_error
        record["failure_type"] = failure_type_from(stage, False, False, False)
        official_metrics = official_metrics_for_record(sample, record)
        upsert_jsonl(paths["prediction"], record, ("case", "sample_id"))
        summary = summarize_case(
            case,
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
        sync_cuda()
        reset_gpu_peak_stats()

        stage = "generate"
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        sync_cuda()

        latency = time.perf_counter() - started
        record["generate_success"] = True
        record["latency_sec"] = round(latency, 3)
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        stage = "decode"
        decoded, actual_new_tokens, decode_error = decode_outputs(outputs, inputs, processor, tokenizer)
        record["actual_new_tokens"] = actual_new_tokens
        if actual_new_tokens is not None and latency > 0:
            record["tokens_per_sec"] = round(actual_new_tokens / latency, 3)

        if decode_error is not None:
            record["decode_success"] = False
            record["generate_error"] = decode_error
        else:
            record["decode_success"] = True
            record["decoded_text"] = decoded
            record["raw_response"] = decoded

        parsed_answer, parse_success, parse_error = parse_model_answer(sample_task(sample), decoded)
        record["parsed_answer"] = parsed_answer
        record["parse_success"] = parse_success
        record["parse_error"] = parse_error
        record["failure_type"] = failure_type_from(
            stage,
            bool(record["generate_success"]),
            bool(record["decode_success"]),
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
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        record["generate_error"] = message
        record["failure_type"] = failure_type_from(stage, bool(record["generate_success"]), False, False)

    official_metrics = official_metrics_for_record(sample, record)
    upsert_jsonl(paths["prediction"], record, ("case", "sample_id"))
    summary = summarize_case(
        case,
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
    parser = argparse.ArgumentParser(description="Stage2 SmartTest fp16 A100 single-GPU runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Generate or validate the stage2 SmartTest.jsonl.")
    prepare.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False)

    run = subparsers.add_parser("run", help="Run exactly one SmartTest case.")
    run.add_argument("--case", choices=["non_forecasting", "forecasting"], required=True)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False)
    run.add_argument("--resume", type=str2bool, nargs="?", const=True, default=False)
    run.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    run.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default=DEFAULT_ATTN_BACKEND,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        logger = TeeLogger(SMARTTEST_DIR / "prepare.log")
        try:
            payload = prepare_smarttest(overwrite=args.overwrite, logger=logger)
            logger.log(f"prepare_summary: {json.dumps(json_safe(payload), ensure_ascii=False)}")
            return 0
        finally:
            logger.close()

    if args.command == "run":
        paths = output_paths(args.output_root)
        logger = TeeLogger(paths["log"])
        try:
            summary = run_one_case(
                case=args.case,
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

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
