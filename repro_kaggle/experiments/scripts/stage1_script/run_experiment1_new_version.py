#!/usr/bin/env python3
"""Experiment 1 new runner: precision/resource inference with official eval.

This script intentionally starts from a clean structure instead of patching the
old experiment runner.

Layers:
  A. Run Layer: model loading, input construction, generate/decode, resources.
  B. Strict Diagnostic Layer: strict machine-format checks for debugging only.
  C. Official Eval Layer: author evaluate_qa.py-compatible files and metrics.
"""

from __future__ import annotations

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "repro_kaggle"
RESULT_ROOT = REPRO_ROOT / "experiments/stage1_results/experiment1_precision_resource"
DOC_ROOT = REPRO_ROOT / "experiments/stage1_docs"
SCRIPT_PATH = Path(__file__).resolve()
AUTHOR_EVALUATE_QA = PROJECT_ROOT / "evaluation/evaluate_qa.py"

MODEL_NAME = "Time-HD-Anonymous/STReasoner-8B"
DEFAULT_MAX_NEW_TOKENS = 512
PATCH_SIZE = 8

MAIN_DATA = (
    REPRO_ROOT
    / "experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl"
)
PAPER_DATA = REPRO_ROOT / "experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl"
STRESS_DATA = REPRO_ROOT / "experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl"

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STRICT_CHOICE_RE = re.compile(r"^[A-Da-d]$")


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    label: str
    precision: str
    cuda_visible_devices: str
    device_map_kind: str


CONFIGS: dict[str, ConfigSpec] = {
    "4bit_single": ConfigSpec("4bit_single", "4bit单卡", "4bit", "0", "single"),
    "8bit_single": ConfigSpec("8bit_single", "8bit单卡", "8bit", "0", "single"),
    "fp16_single": ConfigSpec("fp16_single", "fp16单卡", "fp16", "0", "single"),
    "fp16_dual": ConfigSpec("fp16_dual", "fp16双卡", "fp16", "0,1", "balanced"),
}

DATA_GROUPS = {
    "main": MAIN_DATA,
    "paper": PAPER_DATA,
    "stress": STRESS_DATA,
}

TASK_TO_OFFICIAL = {
    "forecasting": "reasoning_forecasting",
    "entity": "reasoning_entity",
    "etiological": "reasoning_etiological",
    "correlation": "reasoning_correlation",
    "causal": "reasoning_causal",
    "alignment": "alignment",
}


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


def append_jsonl(path: Path, payload: Any) -> None:
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
    return str(path.relative_to(PROJECT_ROOT))


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

    def log_exception(self, title: str, exc: BaseException) -> None:
        self.exception(title, exc)

    def close(self) -> None:
        self._fh.close()


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_task(sample: dict[str, Any]) -> str:
    return str(sample.get("task") or sample.get("category") or "unknown").lower()


def official_task_name(task: str) -> str:
    task_key = task.lower()
    if task_key not in TASK_TO_OFFICIAL:
        raise ValueError(f"Unsupported task for official evaluation: {task}")
    return TASK_TO_OFFICIAL[task_key]


def validate_sample_file(path: Path, expected_count: int, group_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{group_name} sample file not found: {path}")
    rows = load_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"{group_name} expected {expected_count} rows, got {len(rows)}: {path}")

    for idx, sample in enumerate(rows):
        for key in ("input", "timeseries", "output"):
            if key not in sample:
                raise ValueError(f"{group_name} row {idx} missing field: {key}")
        prompt = sample["input"]
        timeseries = sample["timeseries"]
        if not isinstance(prompt, str) or not isinstance(timeseries, list):
            raise TypeError(f"{group_name} row {idx} has invalid input/timeseries type")
        placeholder_count = prompt.count("<ts><ts/>")
        if placeholder_count != len(timeseries):
            raise ValueError(
                f"{group_name} row {idx} placeholder/timeseries mismatch: "
                f"{placeholder_count} vs {len(timeseries)}"
            )
        official_task_name(sample_task(sample))
    return rows


def validate_inputs() -> dict[str, Any]:
    main = validate_sample_file(MAIN_DATA, 20, "main")
    paper = validate_sample_file(PAPER_DATA, 4, "paper")
    stress = validate_sample_file(STRESS_DATA, 1, "stress")
    counts = Counter(sample_task(row) for row in main)
    expected = {"forecasting": 5, "entity": 5, "etiological": 5, "correlation": 5}
    if dict(counts) != expected:
        raise ValueError(f"main task counts mismatch: expected {expected}, got {dict(counts)}")
    return {
        "main": main,
        "paper": paper,
        "stress": stress,
        "counts": {
            "main": len(main),
            "paper": len(paper),
            "stress": len(stress),
            "main_task_counts": dict(counts),
        },
    }


def evidence_section(max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    return f"""## 证据查找

### 作者代码中的精度设置

- 训练脚本使用 fp16：
  - `scripts/qwen3-8b/train_stage1.sh:24`：`--fp16`
  - `scripts/qwen3-8b/train_stage2.sh:24`：`--fp16`
- 推理脚本中的 dtype：
  - `inference/llm_utils.py:97-98`：普通 vLLM worker 使用 `LLM(..., dtype='half')`，对应 fp16/half。
  - `inference/llm_utils.py:142-149`：time-series vLLM worker 未显式设置 dtype；实际 dtype 需要通过本实验记录加载后模型参数 dtype、量化配置和日志确认。

### 作者代码中的 max_new_tokens / max_tokens 设置

- 作者 vLLM 推理使用 `max_tokens=512`：
  - `inference/inference_tsmllm_vllm.py:64-68`：`SamplingParams(max_tokens=512, temperature=0.2)`
- 本实验使用 Hugging Face `generate`，对应参数记录为 `max_new_tokens={max_new_tokens}`。
- Hugging Face 的 `max_new_tokens` 表示最多生成的新 token 数，不包含 prompt tokens。
"""


# ---------------------------------------------------------------------------
# A. Run Layer
# ---------------------------------------------------------------------------


def set_hf_cache_env() -> None:
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/working/hf_cache/datasets")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def import_repro_loader() -> Any:
    return load_module(REPRO_ROOT / "scripts/03_load_streasoner_smoke.py", "experiment1_new_repro_loader")


def import_timeseries_patch_module() -> Any:
    return load_module(REPRO_ROOT / "scripts/05_eval_sttest_tiny.py", "experiment1_new_timeseries_patch")


def gpu_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"visible_gpu_count": 0, "gpu_names": [], "gpu_total_gib": {}}

    names = []
    totals: dict[str, float] = {}
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        names.append(props.name)
        totals[f"gpu{idx}"] = round(props.total_memory / 1024**3, 3)
    return {
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": names,
        "gpu_total_gib": totals,
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
            # Some CUDA allocator states cannot be reset before first allocation.
            continue


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_map_for(spec: ConfigSpec) -> Any:
    if spec.device_map_kind == "single":
        return {"": 0}
    if spec.device_map_kind == "balanced":
        return "balanced"
    raise ValueError(f"unknown device_map_kind: {spec.device_map_kind}")


def quantization_config_for(spec: ConfigSpec) -> Any | None:
    import torch
    from transformers import BitsAndBytesConfig

    if spec.precision == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if spec.precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


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


def load_model_and_processors(spec: ConfigSpec, logger: TeeLogger) -> tuple[Any, Any, Any, float, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM

    repro_loader = import_repro_loader()
    start = time.perf_counter()
    processor, tokenizer = repro_loader.load_processor_and_tokenizer(MODEL_NAME, logger)
    config = repro_loader.load_config(MODEL_NAME, "sdpa", logger)
    qconfig = quantization_config_for(spec)

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map_for(spec),
        "torch_dtype": torch.float16,
        "config": config,
    }
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig

    logger.log("=== Run Layer: Model Load ===")
    logger.log(f"model: {MODEL_NAME}")
    logger.log(f"precision: {spec.precision}")
    logger.log(f"requested device_map: {kwargs['device_map']}")
    logger.log(f"quantization_config: {qconfig}")

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    except Exception as first_exc:
        logger.exception("AutoModelForCausalLM load failed; trying AutoModel", first_exc)
        model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)

    model.eval()
    load_time = time.perf_counter() - start
    patch_module = import_timeseries_patch_module()
    patch_module.patch_timeseries_merge_device(model, logger)

    actual_map = getattr(model, "hf_device_map", None)
    distribution = model_distribution(actual_map)
    load_info = {
        "processor_class": processor.__class__.__name__ if processor is not None else None,
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else None,
        "model_class": model.__class__.__name__,
        "requested_device_map": json_safe(kwargs["device_map"]),
        "actual_device_map": json_safe(actual_map),
        "model_distribution": distribution,
        "use_cache": getattr(getattr(model, "config", None), "use_cache", None),
        "first_parameter_dtype": str(next(model.parameters()).dtype),
        "timeseries_merge_patch_applied": bool(getattr(model, "_repro_kaggle_merge_patch", False)),
        "load_after_memory": gpu_memory_snapshot(),
    }
    logger.log(f"model_load_time_sec: {load_time:.3f}")
    logger.log(f"load_info: {json.dumps(json_safe(load_info), ensure_ascii=False)}")
    return model, processor, tokenizer, load_time, load_info


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


def classify_bottleneck(stage: str, message: str | None, actual_new_tokens: int | None, max_new_tokens: int) -> str | None:
    lower = (message or "").lower()
    if stage in {"model_loading", "processor", "generate"}:
        if "out of memory" in lower or "cuda oom" in lower or "outofmemoryerror" in lower:
            return "资源瓶颈"
        if "offload" in lower or "device" in lower:
            return "资源瓶颈"
        if "size of tensor" in lower or "must match the size" in lower or "placeholder" in lower:
            return "输入/生成瓶颈"
    if stage in {"decode", "strict_diagnostic"}:
        return "输出与评测瓶颈"
    if stage == "none" and actual_new_tokens is not None and actual_new_tokens < max_new_tokens:
        return "输入/生成瓶颈"
    return None


def failure_reason(stage: str, message: str | None, actual_new_tokens: int | None, max_new_tokens: int) -> str | None:
    if stage == "none":
        if actual_new_tokens is not None and actual_new_tokens < max_new_tokens:
            return (
                f"生成提前结束：actual_new_tokens={actual_new_tokens}，"
                f"小于 max_new_tokens={max_new_tokens}；可能是 EOS 或模型自然停止。"
            )
        return None

    text = (message or "").strip()
    lower = text.lower()
    if "out of memory" in lower or "cuda oom" in lower or "outofmemoryerror" in lower:
        return f"显存不足导致失败：{text}"
    if "expected all tensors to be on the same device" in lower or "device mismatch" in lower:
        return f"设备不一致导致失败：{text}"
    if "offload" in lower:
        return f"offload 相关失败：{text}"
    if "size of tensor" in lower or "must match the size" in lower or "placeholder" in lower:
        return f"输入与 time-series 特征合并或构造时失败：{text}"
    return f"{stage} 阶段失败：{text}" if text else f"{stage} 阶段失败。"


def base_record(sample: dict[str, Any], group: str, local_index: int) -> dict[str, Any]:
    return {
        "sample": {
            "group": group,
            "local_index": local_index,
            "sample_id": sample.get("sample_id"),
            "task": sample_task(sample),
            "category": sample.get("category"),
            "source_file": sample.get("source_file"),
            "original_line_index": sample.get("original_line_index"),
            "gold_output": sample.get("output"),
        },
        "run": {
            "generate_success": False,
            "decode_success": False,
            "decoded_text": None,
            "latency_sec": None,
            "tokens_per_sec": None,
            "input_tokens": None,
            "actual_new_tokens": None,
            "token_details": None,
            "gpu_memory_before_generate": None,
            "gpu_memory_after_generate": None,
            "gpu_peak_memory_during_generate": None,
            "error_stage": "none",
            "error_type": None,
            "error_message": None,
            "bottleneck_type": None,
            "failure_reason": None,
        },
        "strict_diagnostic": None,
        "official_eval": {
            "official_task": official_task_name(sample_task(sample)),
            "included_in_prediction_file": False,
        },
    }


def load_failure_record(
    sample: dict[str, Any],
    group: str,
    local_index: int,
    error_message: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    record = base_record(sample, group, local_index)
    record["run"].update(
        {
            "error_stage": "model_loading",
            "error_type": "ModelLoadError",
            "error_message": error_message,
            "bottleneck_type": classify_bottleneck("model_loading", error_message, None, max_new_tokens),
            "failure_reason": failure_reason("model_loading", error_message, None, max_new_tokens),
        }
    )
    return record


def run_one_sample(
    model: Any,
    processor: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    group: str,
    local_index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    record = base_record(sample, group, local_index)
    stage = "processor"
    started = time.perf_counter()
    try:
        inputs, token_info = build_inputs(processor, tokenizer, sample)
        record["run"]["token_details"] = token_info
        record["run"]["input_tokens"] = token_info["input_tokens_metric"]
        inputs = move_inputs_to_device(inputs, first_model_device(model))

        record["run"]["gpu_memory_before_generate"] = gpu_memory_snapshot()
        sync_cuda()
        reset_gpu_peak_stats()

        stage = "generate"
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        sync_cuda()

        latency = time.perf_counter() - started
        record["run"]["generate_success"] = True
        record["run"]["latency_sec"] = round(latency, 3)
        record["run"]["gpu_peak_memory_during_generate"] = gpu_peak_snapshot()
        record["run"]["gpu_memory_after_generate"] = gpu_memory_snapshot()

        stage = "decode"
        decoded, actual_new_tokens, decode_error = decode_outputs(outputs, inputs, processor, tokenizer)
        record["run"]["actual_new_tokens"] = actual_new_tokens
        if actual_new_tokens is not None and latency > 0:
            record["run"]["tokens_per_sec"] = round(actual_new_tokens / latency, 3)

        if decode_error is not None:
            record["run"].update(
                {
                    "error_stage": "decode",
                    "error_type": "DecodeError",
                    "error_message": decode_error,
                    "bottleneck_type": classify_bottleneck("decode", decode_error, actual_new_tokens, max_new_tokens),
                    "failure_reason": failure_reason("decode", decode_error, actual_new_tokens, max_new_tokens),
                }
            )
            return record

        record["run"]["decode_success"] = True
        record["run"]["decoded_text"] = decoded
        record["official_eval"]["included_in_prediction_file"] = True

        early_reason = failure_reason("none", None, actual_new_tokens, max_new_tokens)
        if early_reason is not None:
            record["run"]["bottleneck_type"] = "输入/生成瓶颈"
            record["run"]["failure_reason"] = early_reason
        return record
    except Exception as exc:
        try:
            sync_cuda()
        except Exception:
            pass
        latency = time.perf_counter() - started
        message = f"{exc.__class__.__name__}: {short_error(exc)}"
        record["run"].update(
            {
                "latency_sec": round(latency, 3),
                "gpu_memory_after_generate": gpu_memory_snapshot(),
                "gpu_peak_memory_during_generate": gpu_peak_snapshot(),
                "error_stage": stage,
                "error_type": exc.__class__.__name__,
                "error_message": message,
                "bottleneck_type": classify_bottleneck(stage, message, None, max_new_tokens),
                "failure_reason": failure_reason(stage, message, None, max_new_tokens),
            }
        )
        return record


# ---------------------------------------------------------------------------
# B. Strict Diagnostic Layer
# ---------------------------------------------------------------------------


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


def strict_diagnostic_for_output(task: str, decoded_text: str | None) -> dict[str, Any]:
    if decoded_text is None:
        return {
            "strict_format_success": False,
            "strict_error": "no_decoded_text",
            "parsed_value": None,
            "answer_tag_count": 0,
            "note": "Diagnostic only; not used for official metrics.",
        }

    matches = list(ANSWER_TAG_RE.finditer(decoded_text))
    if len(matches) != 1:
        return {
            "strict_format_success": False,
            "strict_error": f"expected_exactly_one_answer_tag_got_{len(matches)}",
            "parsed_value": None,
            "answer_tag_count": len(matches),
            "note": "Diagnostic only; official evaluate_qa.py may be more permissive.",
        }

    content = matches[0].group(1).strip()
    task_key = task.lower()
    if task_key == "forecasting":
        parsed, error = strict_parse_forecasting(content)
        return {
            "strict_format_success": error is None,
            "strict_error": error,
            "parsed_value": parsed,
            "answer_tag_count": 1,
            "required_format": "<answer>[1.0, 2.0, ...]</answer>",
            "note": "Diagnostic only; official forecasting eval uses MAE/MAPE/coverage.",
        }

    if STRICT_CHOICE_RE.match(content):
        return {
            "strict_format_success": True,
            "strict_error": None,
            "parsed_value": content.upper(),
            "answer_tag_count": 1,
            "required_format": "<answer>A</answer> / <answer>B</answer> / <answer>C</answer> / <answer>D</answer>",
            "note": "Diagnostic only; official evaluate_qa.py normalizes choices separately.",
        }

    return {
        "strict_format_success": False,
        "strict_error": f"answer_tag_content_not_single_choice: {preview(content, 120)}",
        "parsed_value": None,
        "answer_tag_count": 1,
        "required_format": "<answer>A</answer> / <answer>B</answer> / <answer>C</answer> / <answer>D</answer>",
        "note": "Diagnostic only; official evaluate_qa.py may still parse this differently.",
    }


def apply_strict_diagnostics(records_by_group: dict[str, list[dict[str, Any]]]) -> None:
    for records in records_by_group.values():
        for record in records:
            task = record["sample"]["task"]
            decoded = record["run"].get("decoded_text")
            record["strict_diagnostic"] = strict_diagnostic_for_output(task, decoded)
            if record["run"]["decode_success"] and not record["strict_diagnostic"]["strict_format_success"]:
                record["run"].setdefault("diagnostic_error_stage", "strict_diagnostic")


# ---------------------------------------------------------------------------
# C. Official Eval Layer
# ---------------------------------------------------------------------------


def official_eval_module() -> Any:
    return load_module(AUTHOR_EVALUATE_QA, "experiment1_new_author_evaluate_qa")


def official_dataset_record(sample: dict[str, Any], idx: int) -> dict[str, Any]:
    obj = dict(sample)
    obj["idx"] = idx
    return obj


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def grouped_samples_for_official_eval(
    samples_by_group: dict[str, list[dict[str, Any]]],
    records_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]]]:
    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    for group, samples in samples_by_group.items():
        records = records_by_group[group]
        for original_idx, (sample, record) in enumerate(zip(samples, records)):
            official_task = official_task_name(sample_task(sample))
            grouped[group][official_task].append((original_idx, sample, record))
            grouped["all"][official_task].append((original_idx, sample, record))
    return grouped


def run_official_eval(
    out_dir: Path,
    samples_by_group: dict[str, list[dict[str, Any]]],
    records_by_group: dict[str, list[dict[str, Any]]],
    logger: TeeLogger,
) -> dict[str, Any]:
    evaluate_qa = official_eval_module()
    grouped = grouped_samples_for_official_eval(samples_by_group, records_by_group)
    official_root = out_dir / "official_eval"
    metrics_by_scope: dict[str, dict[str, Any]] = {}

    logger.log("=== Official Eval Layer ===")
    logger.log(f"author evaluate_qa.py: {rel(AUTHOR_EVALUATE_QA)}")

    for scope in sorted(grouped.keys()):
        metrics_by_scope[scope] = {}
        for official_task in sorted(grouped[scope].keys()):
            entries = grouped[scope][official_task]
            task_dir = official_root / scope / official_task
            dataset_rows: list[dict[str, Any]] = []
            prediction_rows: list[dict[str, Any]] = []

            for local_idx, (_original_idx, sample, record) in enumerate(entries):
                dataset_rows.append(official_dataset_record(sample, local_idx))
                decoded = record["run"].get("decoded_text")
                if record["run"].get("decode_success") and decoded is not None:
                    prediction_rows.append(
                        {
                            "idx": local_idx,
                            "response": decoded,
                            "num_tokens": record["run"].get("input_tokens"),
                            "sample_id": record["sample"].get("sample_id"),
                        }
                    )

            dataset_path = task_dir / "dataset.jsonl"
            prediction_path = task_dir / "generated_answer_new.json"
            write_jsonl(dataset_path, dataset_rows)
            write_json(prediction_path, prediction_rows)

            dataset = evaluate_qa.load_jsonl_dataset(str(dataset_path))
            predictions = evaluate_qa.load_prediction_files(str(task_dir), pattern="generated_answer")
            metrics = evaluate_qa.evaluate_predictions_for_task(dataset, predictions, official_task)
            metrics["dataset_path"] = rel(dataset_path)
            metrics["prediction_path"] = rel(prediction_path)
            metrics["author_evaluate_qa"] = rel(AUTHOR_EVALUATE_QA)
            write_json(task_dir / "evaluation_metrics.json", metrics)
            metrics_by_scope[scope][official_task] = metrics
            logger.log(f"{scope}/{official_task}: {json.dumps(json_safe(metrics), ensure_ascii=False)}")

    write_json(official_root / "official_metrics.json", metrics_by_scope)
    return metrics_by_scope


def summarize_official_metrics(metrics_by_scope: dict[str, Any]) -> dict[str, Any]:
    all_metrics = metrics_by_scope.get("all", {})
    choice_tasks = [
        "reasoning_entity",
        "reasoning_etiological",
        "reasoning_correlation",
        "reasoning_causal",
    ]
    weighted_correct = 0.0
    weighted_total = 0
    for task in choice_tasks:
        metrics = all_metrics.get(task)
        if not metrics or metrics.get("accuracy") is None:
            continue
        evaluated = int(metrics.get("evaluated_samples") or 0)
        weighted_correct += float(metrics["accuracy"]) * evaluated
        weighted_total += evaluated

    forecasting = all_metrics.get("reasoning_forecasting", {})
    return {
        "official_choice_accuracy_micro": round(weighted_correct / weighted_total, 6) if weighted_total else None,
        "official_choice_evaluated_samples": weighted_total,
        "official_forecasting_mae": forecasting.get("mae"),
        "official_forecasting_mape": forecasting.get("mape"),
        "official_forecasting_coverage": forecasting.get("coverage"),
        "official_metrics_note": (
            "Official metrics are produced by evaluation/evaluate_qa.py. "
            "Forecasting is reported as MAE/MAPE/coverage, not exact accuracy."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting and orchestration
# ---------------------------------------------------------------------------


def max_peak_memory(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    peak: dict[str, dict[str, float]] = {}
    for record in records:
        sample_peak = record["run"].get("gpu_peak_memory_during_generate") or {}
        for gpu, stats in sample_peak.items():
            peak.setdefault(gpu, {"max_allocated_gib": 0.0, "max_reserved_gib": 0.0})
            for key in ("max_allocated_gib", "max_reserved_gib"):
                peak[gpu][key] = max(peak[gpu][key], float(stats.get(key) or 0.0))
    return peak


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def summarize_records(
    spec: ConfigSpec,
    env: dict[str, Any],
    load_success: bool,
    load_error: str | None,
    load_time_sec: float | None,
    load_info: dict[str, Any],
    records_by_group: dict[str, list[dict[str, Any]]],
    official_metrics: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    all_records = [record for records in records_by_group.values() for record in records]
    generated = [record for record in all_records if record["run"].get("generate_success")]
    decoded = [record for record in all_records if record["run"].get("decode_success")]
    strict_ok = [
        record
        for record in all_records
        if record.get("strict_diagnostic") and record["strict_diagnostic"].get("strict_format_success")
    ]
    latencies = [float(record["run"]["latency_sec"]) for record in generated if isinstance(record["run"].get("latency_sec"), (int, float))]
    tokens_per_sec = [
        float(record["run"]["tokens_per_sec"])
        for record in generated
        if isinstance(record["run"].get("tokens_per_sec"), (int, float))
    ]
    input_tokens = [
        int(record["run"]["input_tokens"])
        for record in all_records
        if isinstance(record["run"].get("input_tokens"), int)
    ]
    actual_new_tokens = [
        int(record["run"]["actual_new_tokens"])
        for record in generated
        if isinstance(record["run"].get("actual_new_tokens"), int)
    ]

    stages = Counter(record["run"].get("error_stage") for record in all_records if record["run"].get("error_stage") != "none")
    bottlenecks = Counter(record["run"].get("bottleneck_type") for record in all_records if record["run"].get("bottleneck_type"))
    strict_errors = Counter(
        record["strict_diagnostic"].get("strict_error")
        for record in all_records
        if record.get("strict_diagnostic") and record["strict_diagnostic"].get("strict_error")
    )
    first_error = next(
        (
            record["run"].get("failure_reason") or record["run"].get("error_message")
            for record in all_records
            if record["run"].get("failure_reason") or record["run"].get("error_message")
        ),
        load_error,
    )

    official_summary = summarize_official_metrics(official_metrics)

    return {
        "config": spec.name,
        "label": spec.label,
        "model": MODEL_NAME,
        "batch_size": 1,
        "max_new_tokens": max_new_tokens,
        "precision": spec.precision,
        "cuda_visible_devices_requested_by_run_all": spec.cuda_visible_devices,
        "environment": env,
        "load": {
            "success": load_success,
            "error": load_error,
            "time_sec": round(load_time_sec, 3) if load_time_sec is not None else None,
            "info": load_info,
        },
        "counts": {group: len(records) for group, records in records_by_group.items()},
        "run_layer_metrics": {
            "generate_success_rate": rate(len(generated), len(all_records)),
            "decode_success_rate": rate(len(decoded), len(all_records)),
            "avg_input_tokens": round(sum(input_tokens) / len(input_tokens), 3) if input_tokens else None,
            "avg_actual_new_tokens": round(sum(actual_new_tokens) / len(actual_new_tokens), 3) if actual_new_tokens else None,
            "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max_latency_sec": round(max(latencies), 3) if latencies else None,
            "avg_tokens_per_sec": round(sum(tokens_per_sec) / len(tokens_per_sec), 3) if tokens_per_sec else None,
            "config_level_generate_peak_memory": max_peak_memory(all_records),
            "final_memory": gpu_memory_snapshot(),
        },
        "strict_diagnostic_metrics": {
            "strict_format_success_rate": rate(len(strict_ok), len(all_records)),
            "strict_error_counts": dict(strict_errors),
            "note": "Strict diagnostic is for output-format debugging only; it is not author official evaluation.",
        },
        "official_eval_summary": official_summary,
        "official_eval_by_scope_and_task": official_metrics,
        "failure_count_by_stage": dict(stages),
        "bottleneck_counts": dict(bottlenecks),
        "first_error": first_error,
    }


def record_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "main": out_dir / "main_predictions_new.jsonl",
        "paper": out_dir / "paper_predictions_new.jsonl",
        "stress": out_dir / "stress_predictions_new.jsonl",
    }


def write_records(records_by_group: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    for group, path in record_paths(out_dir).items():
        path.write_text("", encoding="utf-8")
        for record in records_by_group[group]:
            append_jsonl(path, record)


def write_config_report(spec: ConfigSpec, summary: dict[str, Any]) -> None:
    run_metrics = summary["run_layer_metrics"]
    strict_metrics = summary["strict_diagnostic_metrics"]
    official = summary["official_eval_summary"]
    report = f"""# 实验一：{spec.label}

## 固定配置

- 模型：`STReasoner_8B` / `{MODEL_NAME}`
- batch size：1
- max_new_tokens：{summary["max_new_tokens"]}
- precision：`{spec.precision}`
- requested device_map：`{summary["load"]["info"].get("requested_device_map")}`
- actual device_map：`{summary["load"]["info"].get("actual_device_map")}`
- model distribution：`{summary["load"]["info"].get("model_distribution")}`

## A. Run Layer

- load 成功：{summary["load"]["success"]}
- load 错误：{summary["load"]["error"]}
- generate 成功率：{run_metrics["generate_success_rate"]}
- decode 成功率：{run_metrics["decode_success_rate"]}
- 平均 input tokens：{run_metrics["avg_input_tokens"]}
- 平均 actual new tokens：{run_metrics["avg_actual_new_tokens"]}
- 平均/最高延迟：{run_metrics["avg_latency_sec"]} / {run_metrics["max_latency_sec"]} 秒
- 平均 tokens/s：{run_metrics["avg_tokens_per_sec"]}
- config-level generate 峰值显存：`{json.dumps(run_metrics["config_level_generate_peak_memory"], ensure_ascii=False)}`

## B. Strict Diagnostic Layer

- strict format 成功率：{strict_metrics["strict_format_success_rate"]}
- strict error counts：`{json.dumps(strict_metrics["strict_error_counts"], ensure_ascii=False)}`
- 说明：该层只诊断输出格式，不作为作者官方评测口径。

## C. Official Eval Layer

- official choice accuracy micro：{official["official_choice_accuracy_micro"]}
- official forecasting MAE：{official["official_forecasting_mae"]}
- official forecasting MAPE：{official["official_forecasting_mape"]}
- official forecasting coverage：{official["official_forecasting_coverage"]}
- 说明：官方指标复用 `evaluation/evaluate_qa.py`；forecasting 使用 MAE/MAPE/coverage，不计算 exact accuracy。

## 产物

- run records：`{rel(RESULT_ROOT / spec.name / "main_predictions_new.jsonl")}`，`{rel(RESULT_ROOT / spec.name / "paper_predictions_new.jsonl")}`，`{rel(RESULT_ROOT / spec.name / "stress_predictions_new.jsonl")}`
- official eval：`{rel(RESULT_ROOT / spec.name / "official_eval")}`
- summary：`{rel(RESULT_ROOT / spec.name / "summary_new.json")}`
- log：`{rel(RESULT_ROOT / spec.name / "run_new.log")}`

## 瓶颈

- failure_count_by_stage：`{json.dumps(summary["failure_count_by_stage"], ensure_ascii=False)}`
- bottleneck_counts：`{json.dumps(summary["bottleneck_counts"], ensure_ascii=False)}`
- first_error：{summary["first_error"]}
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / f"experiment1_{spec.name}.md").write_text(report, encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_experiment_summary() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    for config_name in CONFIGS:
        path = RESULT_ROOT / config_name / "summary_new.json"
        if path.exists():
            summaries[config_name] = json.loads(path.read_text(encoding="utf-8"))

    validation = validate_inputs()
    cols = ["4bit_single", "8bit_single", "fp16_single", "fp16_dual"]

    def cell(config_name: str, path: list[str]) -> str:
        current: Any = summaries.get(config_name, {})
        for key in path:
            if not isinstance(current, dict):
                return ""
            current = current.get(key)
        return fmt(current)

    rows = [
        ("配置证据", "加载方式", [[ "precision" ] for _ in cols]),
        ("", "device_map", [["load", "info", "requested_device_map"] for _ in cols]),
        ("", "实际模型分布", [["load", "info", "model_distribution"] for _ in cols]),
        ("", "is_cpu_offload", [["load", "info", "model_distribution", "has_cpu_offload"] for _ in cols]),
        ("", "use_cache", [["load", "info", "use_cache"] for _ in cols]),
        ("可运行证据", "input tokens（平均值）", [["run_layer_metrics", "avg_input_tokens"] for _ in cols]),
        ("", "actual new tokens（平均值）", [["run_layer_metrics", "avg_actual_new_tokens"] for _ in cols]),
        ("", "load 成功", [["load", "success"] for _ in cols]),
        ("", "generate 成功率", [["run_layer_metrics", "generate_success_rate"] for _ in cols]),
        ("资源与速度", "GPU 总显存", [["environment", "gpu_total_gib"] for _ in cols]),
        ("", "load 后显存", [["load", "info", "load_after_memory"] for _ in cols]),
        ("", "generate 峰值显存", [["run_layer_metrics", "config_level_generate_peak_memory"] for _ in cols]),
        ("", "平均延迟与最高延迟", [["run_layer_metrics", "avg_latency_sec"] for _ in cols]),
        ("", "tokens/s", [["run_layer_metrics", "avg_tokens_per_sec"] for _ in cols]),
        ("输出与评测", "decode 成功率", [["run_layer_metrics", "decode_success_rate"] for _ in cols]),
        ("", "strict diagnostic 成功率", [["strict_diagnostic_metrics", "strict_format_success_rate"] for _ in cols]),
        ("", "official choice accuracy", [["official_eval_summary", "official_choice_accuracy_micro"] for _ in cols]),
        ("", "official forecasting MAE", [["official_eval_summary", "official_forecasting_mae"] for _ in cols]),
        ("", "official forecasting MAPE", [["official_eval_summary", "official_forecasting_mape"] for _ in cols]),
        ("失败阶段、失败原因", "失败阶段、详细失败原因", [["first_error"] for _ in cols]),
    ]

    table = [
        "|           | 指标 | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |",
        "| --------- | ---- | -------: | -------: | -------: | ------- |",
    ]
    for section, metric, paths in rows:
        values = [cell(config_name, path) for config_name, path in zip(cols, paths)]
        table.append("| " + " | ".join([section, metric, *values]) + " |")

    doc = f"""# 实验一：不同精度推理资源测试

## 样本与目录

- 主测试样例：`{rel(MAIN_DATA)}`，共 {validation["counts"]["main"]} 条，四类任务各 5 条。
- 论文样例：`{rel(PAPER_DATA)}`，共 {validation["counts"]["paper"]} 条。
- 压力测试样例：`{rel(STRESS_DATA)}`，共 {validation["counts"]["stress"]} 条。
- 新运行脚本：`{rel(SCRIPT_PATH)}`
- 机器可读结果目录：`{rel(RESULT_ROOT)}`
- 官方评测逻辑：`{rel(AUTHOR_EVALUATE_QA)}`

{evidence_section(DEFAULT_MAX_NEW_TOKENS)}

## 分层口径

- A. Run Layer：只负责加载模型、构造输入、generate、decode、记录资源。
- B. Strict Diagnostic Layer：只诊断输出是否符合我们希望的机器可解析格式。
- C. Official Eval Layer：生成 `evaluate_qa.py` 可读取的 `generated_answer*.json`，并复用作者的解析和指标函数。

## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | 主测试 20 + 论文样例 4 + 压力测试 1 |
|   batch size   |                1                |
| max_new_tokens |               {DEFAULT_MAX_NEW_TOKENS}               |

{chr(10).join(table)}

## 说明

- 不再使用 `official_accuracy` / `official_parse_success_rate` 这类误导命名。
- forecasting 官方口径只报告 MAE / MAPE / coverage，不报告 exact accuracy。
- `strict diagnostic 成功率` 是我们自己的格式诊断，不等同于作者官方 parse 或 evaluation。
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "experiment_summary.md").write_text(doc, encoding="utf-8")


def write_prepare_docs(max_new_tokens: int) -> None:
    validation = validate_inputs()
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    summary = f"""# 实验一：不同精度推理资源测试

> 当前状态：新脚本已准备。运行后会按 A/B/C 三层分别输出 run records、strict diagnostics、official eval metrics。

## 样本与目录

- 主测试样例：`{rel(MAIN_DATA)}`，共 {validation["counts"]["main"]} 条，四类任务各 5 条。
- 论文样例：`{rel(PAPER_DATA)}`，共 {validation["counts"]["paper"]} 条。
- 压力测试样例：`{rel(STRESS_DATA)}`，共 {validation["counts"]["stress"]} 条。
- 新运行脚本：`{rel(SCRIPT_PATH)}`
- 机器可读结果目录：`{rel(RESULT_ROOT)}`
- 官方评测逻辑：`{rel(AUTHOR_EVALUATE_QA)}`

{evidence_section(max_new_tokens)}

## 运行方式

```bash
python {rel(SCRIPT_PATH)} run-config --config 4bit_single
python {rel(SCRIPT_PATH)} run-all
```

## 固定设置

- 模型：`{MODEL_NAME}`
- batch size：1
- max_new_tokens：{max_new_tokens}
- do_sample：False

## 口径修正

- Run Layer 只记录 generate/decode/resource，不做官方分数。
- Strict Diagnostic Layer 只检查 `<answer>...</answer>` 是否符合我们希望的机器格式。
- Official Eval Layer 复用 `evaluation/evaluate_qa.py`；forecasting 使用 MAE/MAPE/coverage。
"""
    (DOC_ROOT / "experiment_summary.md").write_text(summary, encoding="utf-8")


def run_config(config_name: str, max_new_tokens: int) -> int:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown config: {config_name}")

    spec = CONFIGS[config_name]
    validation = validate_inputs()
    samples_by_group: dict[str, list[dict[str, Any]]] = {
        "main": validation["main"],
        "paper": validation["paper"],
        "stress": validation["stress"],
    }

    out_dir = RESULT_ROOT / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in record_paths(out_dir).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    logger = TeeLogger(out_dir / "run_new.log")
    records_by_group: dict[str, list[dict[str, Any]]] = {"main": [], "paper": [], "stress": []}
    load_success = False
    load_error = None
    load_time_sec = None
    load_info: dict[str, Any] = {}

    try:
        set_hf_cache_env()
        env = gpu_environment()
        logger.log(f"=== Experiment 1 New Version: {spec.name} / {spec.label} ===")
        logger.log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.log(f"environment={json.dumps(json_safe(env), ensure_ascii=False)}")

        try:
            reset_gpu_peak_stats()
            model, processor, tokenizer, load_time_sec, load_info = load_model_and_processors(spec, logger)
            load_success = True
        except Exception as exc:
            load_error = f"{exc.__class__.__name__}: {short_error(exc)}"
            logger.exception("model loading failed", exc)
            for group, samples in samples_by_group.items():
                for local_index, sample in enumerate(samples):
                    records_by_group[group].append(load_failure_record(sample, group, local_index, load_error, max_new_tokens))
        else:
            for group, samples in samples_by_group.items():
                output_path = record_paths(out_dir)[group]
                for local_index, sample in enumerate(samples):
                    logger.log(
                        f"[{group} {local_index + 1}/{len(samples)}] "
                        f"sample_id={sample.get('sample_id')} task={sample_task(sample)}"
                    )
                    record = run_one_sample(model, processor, tokenizer, sample, group, local_index, max_new_tokens)
                    records_by_group[group].append(record)
                    append_jsonl(output_path, record)
                    logger.log(
                        "  "
                        f"generate={record['run']['generate_success']} "
                        f"decode={record['run']['decode_success']} "
                        f"new_tokens={record['run']['actual_new_tokens']} "
                        f"latency={record['run']['latency_sec']} "
                        f"stage={record['run']['error_stage']}"
                    )

        apply_strict_diagnostics(records_by_group)
        write_records(records_by_group, out_dir)
        official_metrics = run_official_eval(out_dir, samples_by_group, records_by_group, logger)
        summary = summarize_records(
            spec=spec,
            env=gpu_environment(),
            load_success=load_success,
            load_error=load_error,
            load_time_sec=load_time_sec,
            load_info=load_info,
            records_by_group=records_by_group,
            official_metrics=official_metrics,
            max_new_tokens=max_new_tokens,
        )
        write_json(out_dir / "summary_new.json", summary)
        write_config_report(spec, summary)
        logger.log("=== Summary ===")
        logger.log(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 0 if load_success else 1
    finally:
        logger.close()


def run_all(max_new_tokens: int) -> int:
    failures = 0
    for config_name, spec in CONFIGS.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "run-config",
            "--config",
            config_name,
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures += 1
    write_experiment_summary()
    return 0 if failures == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一新版本：不同精度推理资源测试")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    sub = parser.add_subparsers(dest="command")
    prepare_parser = sub.add_parser("prepare", help="验证样本并写入新实验说明，不运行模型")
    prepare_parser.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    run_config_parser = sub.add_parser("run-config", help="运行单个配置")
    run_config_parser.add_argument("--config", choices=CONFIGS.keys(), required=True)
    run_config_parser.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    run_all_parser = sub.add_parser("run-all", help="依次运行四组配置")
    run_all_parser.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    sub.add_parser("summarize", help="根据已有 summary_new.json 刷新总报告")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "prepare"
    if command == "prepare":
        write_prepare_docs(args.max_new_tokens)
        print(f"prepared: {DOC_ROOT / 'experiment_summary.md'}")
        return 0
    if command == "run-config":
        return run_config(args.config, args.max_new_tokens)
    if command == "run-all":
        return run_all(args.max_new_tokens)
    if command == "summarize":
        write_experiment_summary()
        print(f"wrote: {DOC_ROOT / 'experiment_summary.md'}")
        return 0
    raise ValueError(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
