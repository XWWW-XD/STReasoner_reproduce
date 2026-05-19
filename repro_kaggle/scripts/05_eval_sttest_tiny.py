#!/usr/bin/env python3
"""Tiny ST-Test evaluation for low-resource STReasoner comparisons.

This helper intentionally stays under repro_kaggle/. It does not import the
author's training or evaluation entrypoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script_module(filename: str, module_name: str) -> Any:
    module_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_smoke = load_script_module("03_load_streasoner_smoke.py", "repro_kaggle_03_load_streasoner_smoke")
_one_sample = load_script_module("04_run_one_sttest_sample.py", "repro_kaggle_04_run_one_sttest_sample")

DEFAULT_MODEL_NAME = _smoke.DEFAULT_MODEL_NAME
TeeLogger = _smoke.TeeLogger
class_name = _smoke.class_name
load_config = _smoke.load_config
load_processor_and_tokenizer = _smoke.load_processor_and_tokenizer
move_inputs_to_device = _smoke.move_inputs_to_device
set_hf_cache_env = _smoke.set_hf_cache_env
str2bool = _smoke.str2bool
decode_prediction = _one_sample.decode_prediction
json_safe = _one_sample.json_safe


DATASET_NAME = "Time-HD-Anonymous/ST-Bench"
DATASET_SUBSET = "ST-Test"
DEFAULT_OUTPUT_PATH = "repro_kaggle/outputs/sttest_tiny_predictions.jsonl"
DEFAULT_SUMMARY_PATH = "repro_kaggle/outputs/sttest_tiny_summary.json"
DEFAULT_LOG_PATH = "repro_kaggle/outputs/05_sttest_tiny_eval.log"
SELECTED_INDICES_PATH = "repro_kaggle/outputs/compare_single4bit_dualfp16_selected_indices.json"
ANSWER_FORMAT_INSTRUCTION = (
    "Please end your response with exactly one final answer tag in the format "
    "<answer>A</answer>, <answer>B</answer>, <answer>C</answer>, or <answer>D</answer>. "
    "Do not omit the final answer tag."
)
TARGET_CATEGORIES = ("correlation", "entity", "etiological", "forecasting")
VALID_CATEGORIES = ("all",) + TARGET_CATEGORIES
CHOICE_RE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)")
# 它只解析 A/B/C/D，没有单独处理 forecasting。
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny ST-Test evaluation.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--precision", choices=["4bit", "fp16"], default="4bit")
    parser.add_argument("--load_in_4bit", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--attn_backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument(
        "--device_strategy",
        choices=["single_gpu", "dual_auto", "dual_balanced"],
        default="single_gpu",
    )
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--samples_per_category", type=int, default=5)
    parser.add_argument("--category", choices=VALID_CATEGORIES, default="all")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--answer_format_prompt", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary_path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def resolve_precision(args: argparse.Namespace) -> str:
    if args.load_in_4bit is None:
        return args.precision
    return "4bit" if args.load_in_4bit else "fp16"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def preview_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def short_error_message(exc: BaseException | str, limit: int = 1000) -> str:
    message = str(exc)
    if len(message) > limit:
        return message[:limit] + "..."
    return message


def make_error(stage: str, exc: BaseException | str) -> dict[str, Any]:
    if isinstance(exc, BaseException):
        return {
            "error_stage": stage,
            "error_type": exc.__class__.__name__,
            "error_message": short_error_message(exc),
            "traceback": traceback.format_exc(),
        }
    return {
        "error_stage": stage,
        "error_type": "ParseError" if stage == "parse" else "Error",
        "error_message": short_error_message(exc),
        "traceback": None,
    }


def parse_choice(text: Any) -> tuple[str | None, str | None]:
    if text is None:
        return None, "empty"

    value = str(text).strip()
    if not value:
        return None, "empty"

    tag_match = ANSWER_TAG_RE.search(value)
    if tag_match:
        tagged = tag_match.group(1).strip()
        if not tagged:
            return None, "empty_answer_tag"
        if tagged in {"A", "B", "C", "D"}:
            return tagged, None
        choice_match = CHOICE_RE.search(tagged)
        if choice_match:
            return choice_match.group(1), None
        return None, f"answer_tag_not_choice: {preview_text(tagged, 120)}"

    if value in {"A", "B", "C", "D"}:
        return value, None

    choice_match = CHOICE_RE.search(value)
    if choice_match:
        return choice_match.group(1), None

    return None, "no_answer_tag_or_standalone_choice"


def gpu_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"visible_gpu_count": 0, "gpu_names": []}

    names = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        names.append(f"{props.name} ({props.total_memory / 1024**3:.2f} GiB)")
    return {"visible_gpu_count": torch.cuda.device_count(), "gpu_names": names}


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


def gpu_peak_snapshot() -> tuple[dict[str, float], dict[str, float]]:
    import torch

    max_allocated: dict[str, float] = {}
    max_reserved: dict[str, float] = {}
    if not torch.cuda.is_available():
        return max_allocated, max_reserved

    for idx in range(torch.cuda.device_count()):
        key = f"gpu{idx}"
        max_allocated[key] = round(torch.cuda.max_memory_allocated(idx) / 1024**3, 3)
        max_reserved[key] = round(torch.cuda.max_memory_reserved(idx) / 1024**3, 3)
    return max_allocated, max_reserved


def reset_cuda_peak_stats() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def target_categories(category: str) -> tuple[str, ...]:
    if category == "all":
        return TARGET_CATEGORIES
    return (category,)


def selection_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "subset": DATASET_SUBSET,
        "split": "train",
        "max_samples": args.max_samples,
        "samples_per_category": args.samples_per_category,
        "category": args.category,
        "target_categories": list(target_categories(args.category)),
    }


def load_or_select_samples(args: argparse.Namespace, logger: TeeLogger) -> list[tuple[int, dict[str, Any]]]:
    from datasets import load_dataset

    logger.log("=== Data Loading ===")
    logger.log(f"dataset: {DATASET_NAME}")
    logger.log(f"subset: {DATASET_SUBSET}")
    logger.log("split: train")

    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train")
    logger.log(f"dataset rows: {dataset.num_rows}")

    selected_path = Path(SELECTED_INDICES_PATH)
    expected_metadata = selection_metadata(args)
    selected_entries: list[dict[str, Any]] | None = None

    if selected_path.exists():
        try:
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
            if payload.get("metadata") == expected_metadata:
                selected_entries = list(payload.get("samples", []))
                logger.log(f"Reusing selected sample indices: {selected_path}")
            else:
                logger.log("Existing selected indices metadata differs; regenerating.")
        except Exception as exc:  # noqa: BLE001 - regenerate if the cache is stale.
            logger.log(f"Could not read selected indices file; regenerating. {exc.__class__.__name__}: {exc}")

    if selected_entries is None:
        counts: Counter[str] = Counter()
        selected_entries = []
        categories = set(target_categories(args.category))

        for index, row in enumerate(dataset):
            sample = dict(row)
            category = sample.get("category")
            if category not in categories:
                continue
            if counts[category] >= args.samples_per_category:
                continue

            selected_entries.append({"index": index, "category": category})
            counts[category] += 1

            if len(selected_entries) >= args.max_samples:
                break
            if all(counts[category] >= args.samples_per_category for category in categories):
                break

        payload = {"metadata": expected_metadata, "samples": selected_entries}
        write_json(selected_path, payload)
        logger.log(f"Wrote selected sample indices: {selected_path}")

    samples: list[tuple[int, dict[str, Any]]] = []
    for entry in selected_entries[: args.max_samples]:
        index = int(entry["index"])
        if index < 0 or index >= dataset.num_rows:
            raise IndexError(f"selected index {index} is out of range for dataset with {dataset.num_rows} rows")
        samples.append((index, dict(dataset[index])))

    logger.log(f"selected rows: {len(samples)}")
    by_category = Counter(sample.get("category") for _, sample in samples)
    logger.log(f"selected rows by category: {dict(by_category)}")
    return samples


def build_quantization_config(precision: str, logger: TeeLogger) -> Any | None:
    if precision != "4bit":
        logger.log("precision=fp16: not using BitsAndBytesConfig.")
        return None

    import torch
    from transformers import BitsAndBytesConfig

    logger.log("precision=4bit: using BitsAndBytesConfig with torch.float16 compute dtype.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def requested_device_map(device_strategy: str, logger: TeeLogger) -> Any:
    import torch

    if device_strategy == "single_gpu":
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logger.log("device_strategy=single_gpu: using device_map={'': 0}; expect CUDA_VISIBLE_DEVICES=0 externally.")
        return {"": 0}
    if device_strategy == "dual_auto":
        logger.log("device_strategy=dual_auto: using device_map='auto'.")
        return "auto"
    logger.log("device_strategy=dual_balanced: using device_map='balanced'.")
    return "balanced"


def load_model_for_strategy(
    model_name: str,
    precision: str,
    attn_backend: str,
    device_strategy: str,
    logger: TeeLogger,
) -> tuple[Any, Any | None, Any | None, float]:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM

    logger.log("=== Model Loading ===")
    logger.log(f"model_name: {model_name}")
    logger.log(f"precision: {precision}")
    logger.log("torch_dtype: torch.float16")
    logger.log("bf16: disabled")
    logger.log("flash_attn: not requested")
    logger.log(f"attn_backend: {attn_backend}")

    started = time.perf_counter()
    processor, tokenizer = load_processor_and_tokenizer(model_name, logger)
    config = load_config(model_name, attn_backend, logger)
    quantization_config = build_quantization_config(precision, logger)
    device_map = requested_device_map(device_strategy, logger)

    common_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "config": config,
    }
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    logger.log("Trying AutoModelForCausalLM.from_pretrained(...)")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    except Exception as first_exc:
        logger.log_exception("AutoModelForCausalLM load failed", first_exc)
        logger.log("Trying AutoModel.from_pretrained(...)")
        try:
            model = AutoModel.from_pretrained(model_name, **common_kwargs)
        except Exception:
            raise

    load_time = time.perf_counter() - started
    model.eval()
    logger.log(f"model class: {class_name(model)}")
    logger.log(f"processor class: {class_name(processor)}")
    logger.log(f"tokenizer class: {class_name(tokenizer)}")
    logger.log(f"model_load_time_sec: {load_time:.2f}")
    logger.log("MODEL_LOAD_PASS")
    return model, processor, tokenizer, load_time


def collect_cuda_devices(value: Any) -> set[int]:
    devices: set[int] = set()
    if value is None:
        return devices
    if isinstance(value, int):
        devices.add(value)
        return devices
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped.isdigit():
            devices.add(int(stripped))
        elif stripped.startswith("cuda:"):
            suffix = stripped.split(":", 1)[1]
            if suffix.isdigit():
                devices.add(int(suffix))
        return devices
    if isinstance(value, dict):
        for item in value.values():
            devices.update(collect_cuda_devices(item))
        return devices
    if isinstance(value, (list, tuple, set)):
        for item in value:
            devices.update(collect_cuda_devices(item))
        return devices
    return devices


def actual_device_map(model: Any, fallback: Any) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        return device_map
    return fallback


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "outofmemoryerror" in exc.__class__.__name__.lower() or "out of memory" in message or "cuda oom" in message


def is_device_mismatch_error(exc_or_message: BaseException | str) -> bool:
    message = str(exc_or_message)
    return "expected all tensors to be on the same device" in message.lower() or "device mismatch" in message.lower()


def classify_load_exception(exc: BaseException, device_strategy: str) -> tuple[str, str]:
    if is_oom_error(exc):
        return "OOM", "OutOfMemoryError"
    if is_device_mismatch_error(exc):
        return "DEVICE_MISMATCH", exc.__class__.__name__
    if device_strategy == "dual_balanced" and "balanced" in str(exc).lower():
        return "BALANCED_NOT_SUPPORTED", exc.__class__.__name__
    return "LOAD_FAIL", exc.__class__.__name__


def patch_timeseries_merge_device(model: Any, logger: TeeLogger) -> None:
    """Patch a device mismatch in remote-code generation when device_map spans GPUs."""

    if not hasattr(model, "_merge_input_ids_with_time_series_features") or getattr(model, "_repro_kaggle_merge_patch", False):
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
    setattr(model, "_repro_kaggle_merge_patch", True)
    logger.log("Applied runtime patch for multi-GPU time-series merge device alignment.")


def build_inputs(processor: Any | None, sample: dict[str, Any], answer_format_prompt: bool) -> Any:
    if processor is None:
        raise RuntimeError("AutoProcessor is unavailable; cannot construct text + timeseries inputs.")

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

    if answer_format_prompt:
        prompt = prompt.rstrip() + "\n\n" + ANSWER_FORMAT_INSTRUCTION

    return processor(text=prompt, timeseries=timeseries, return_tensors="pt")


def first_model_device(model: Any) -> Any:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_prediction(
    model: Any,
    processor: Any | None,
    tokenizer: Any | None,
    sample: dict[str, Any],
    max_new_tokens: int,
    answer_format_prompt: bool,
) -> str:
    import torch

    inputs = build_inputs(processor, sample, answer_format_prompt)
    inputs = move_inputs_to_device(inputs, first_model_device(model))
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return decode_prediction(outputs, inputs, processor, tokenizer)


def base_record(
    index: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
    precision: str,
    device_map: Any,
    is_multi_gpu: bool,
) -> dict[str, Any]:
    return {
        "index": index,
        "category": sample.get("category"),
        "input_preview": preview_text(sample.get("input"), 500),
        "output": sample.get("output"),
        "prediction": None,
        "parsed_prediction": None,
        "is_correct": None,
        "latency_sec": None,
        "precision": precision,
        "device_strategy": args.device_strategy,
        "max_new_tokens": args.max_new_tokens,
        "answer_format_prompt": args.answer_format_prompt,
        "actual_device_map": json_safe(device_map),
        "is_actually_multi_gpu": is_multi_gpu,
        "gpu_memory_before_generate": None,
        "gpu_memory_after_generate": None,
        "parse_failed": None,
        "error_stage": "none",
        "error_type": None,
        "error_message": None,
    }


def apply_record_error(record: dict[str, Any], stage: str, exc: BaseException | str) -> None:
    error = make_error(stage, exc)
    record["error_stage"] = error["error_stage"]
    record["error_type"] = error["error_type"]
    record["error_message"] = error["error_message"]
    record["traceback"] = error["traceback"]


def latency_stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    ordered = sorted(values)
    avg = sum(ordered) / len(ordered)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return avg, median, ordered[p95_index]


def category_summaries(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    accuracy_by_category: dict[str, Any] = {}
    success_by_category: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("category"))].append(record)

    for category, items in grouped.items():
        generated = [item for item in items if item.get("prediction") is not None]
        parsed = [item for item in generated if not item.get("parse_failed")]
        scored = [item for item in parsed if isinstance(item.get("is_correct"), bool)]
        accuracy = None
        if scored:
            accuracy = sum(1 for item in scored if item.get("is_correct")) / len(scored)
        accuracy_by_category[category] = {
            "accuracy": accuracy,
            "num_scored": len(scored),
        }
        success_by_category[category] = {
            "num_total": len(items),
            "num_generate_success": len(generated),
            "num_generate_failed": len([item for item in items if item.get("error_stage") in {"processor", "generate"}]),
            "num_parse_success": len(parsed),
            "num_parse_failed": len([item for item in generated if item.get("parse_failed")]),
        }

    return accuracy_by_category, success_by_category


def build_summary(
    args: argparse.Namespace,
    precision: str,
    env: dict[str, Any],
    selected_samples: list[tuple[int, dict[str, Any]]],
    records: list[dict[str, Any]],
    model_load_pass: bool,
    model_load_time_sec: float | None,
    total_eval_time_sec: float,
    device_map: Any,
    is_multi_gpu: bool,
    first_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated = [record for record in records if record.get("prediction") is not None]
    generate_failed = [record for record in records if record.get("error_stage") in {"processor", "generate"}]
    parse_failed = [record for record in generated if record.get("parse_failed")]
    parse_success = [record for record in generated if not record.get("parse_failed")]
    scored = [record for record in parse_success if isinstance(record.get("is_correct"), bool)]
    latencies = [record["latency_sec"] for record in generated if isinstance(record.get("latency_sec"), (int, float))]
    avg_latency, median_latency, p95_latency = latency_stats([float(value) for value in latencies])

    accuracy_overall = None
    if scored:
        accuracy_overall = sum(1 for record in scored if record.get("is_correct")) / len(scored)

    failure_count_by_stage: Counter[str] = Counter()
    failure_count_by_error_type: Counter[str] = Counter()
    for record in records:
        stage = record.get("error_stage")
        if stage and stage != "none":
            failure_count_by_stage[str(stage)] += 1
            failure_count_by_error_type[str(record.get("error_type") or "unknown")] += 1
    if first_error is not None and not records:
        failure_count_by_stage[str(first_error.get("error_stage", "unknown"))] += 1
        failure_count_by_error_type[str(first_error.get("error_type", "unknown"))] += 1

    first_error_message = None
    if first_error is not None:
        first_error_message = first_error.get("error_message")
    else:
        for record in records:
            if record.get("error_stage") != "none":
                first_error_message = record.get("error_message")
                break

    max_allocated, max_reserved = gpu_peak_snapshot()
    final_memory = gpu_memory_snapshot()
    final_allocated = {gpu: stats["allocated_gib"] for gpu, stats in final_memory.items()}
    final_reserved = {gpu: stats["reserved_gib"] for gpu, stats in final_memory.items()}
    accuracy_by_category, success_by_category = category_summaries(records)

    if not model_load_pass:
        hint = "OOM" if first_error and first_error.get("error_type") == "OutOfMemoryError" else "LOAD_FAIL"
    elif any(is_device_mismatch_error(record.get("error_message") or "") for record in records):
        hint = "DEVICE_MISMATCH"
    elif generate_failed or parse_failed:
        hint = "PARTIAL"
    else:
        hint = "PASS"

    return {
        "model_name": args.model_name,
        "precision": precision,
        "device_strategy": args.device_strategy,
        "attn_backend": args.attn_backend,
        "max_new_tokens": args.max_new_tokens,
        "answer_format_prompt": args.answer_format_prompt,
        "visible_gpu_count": env["visible_gpu_count"],
        "gpu_names": env["gpu_names"],
        "actual_device_map": json_safe(device_map),
        "is_actually_multi_gpu": is_multi_gpu,
        "model_load_pass": model_load_pass,
        "model_load_time_sec": round(model_load_time_sec, 3) if model_load_time_sec is not None else None,
        "total_eval_time_sec": round(total_eval_time_sec, 3),
        "num_samples_requested": len(selected_samples) if selected_samples else args.max_samples,
        "num_samples_run": len(records),
        "num_generate_success": len(generated),
        "num_generate_failed": len(generate_failed),
        "num_parse_success": len(parse_success),
        "num_parse_failed": len(parse_failed),
        "parse_fail_rate": (len(parse_failed) / len(generated)) if generated else None,
        "accuracy_overall_if_applicable": accuracy_overall,
        "accuracy_by_category": accuracy_by_category,
        "success_by_category": success_by_category,
        "avg_latency_sec": avg_latency,
        "median_latency_sec": median_latency,
        "p95_latency_sec": p95_latency,
        "max_allocated_gib_by_gpu": max_allocated,
        "max_reserved_gib_by_gpu": max_reserved,
        "final_allocated_gib_by_gpu": final_allocated,
        "final_reserved_gib_by_gpu": final_reserved,
        "failure_count_by_stage": dict(failure_count_by_stage),
        "failure_count_by_error_type": dict(failure_count_by_error_type),
        "first_error_message": first_error_message,
        "conclusion_hint": hint,
    }


def write_load_failure_outputs(
    args: argparse.Namespace,
    precision: str,
    env: dict[str, Any],
    selected_samples: list[tuple[int, dict[str, Any]]],
    started: float,
    exc: BaseException,
    logger: TeeLogger,
) -> int:
    marker, error_type = classify_load_exception(exc, args.device_strategy)
    if args.device_strategy == "single_gpu" and marker == "OOM":
        logger.log("SINGLE_GPU_OOM")
    if marker == "BALANCED_NOT_SUPPORTED":
        logger.log("BALANCED_NOT_SUPPORTED")
    logger.log("LOAD_FAIL")
    first_error = {
        "error_stage": "model_loading",
        "error_type": error_type,
        "error_message": short_error_message(exc),
    }
    summary = build_summary(
        args=args,
        precision=precision,
        env=env,
        selected_samples=selected_samples,
        records=[],
        model_load_pass=False,
        model_load_time_sec=None,
        total_eval_time_sec=time.perf_counter() - started,
        device_map=requested_device_map(args.device_strategy, logger),
        is_multi_gpu=False,
        first_error=first_error,
    )
    summary["conclusion_hint"] = marker if marker in {"OOM", "DEVICE_MISMATCH"} else "LOAD_FAIL"
    write_json(args.summary_path, summary)
    logger.log("=== Summary ===")
    logger.log(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1


def main() -> int:
    args = parse_args()
    precision = resolve_precision(args)
    logger = TeeLogger(args.log_path)
    records: list[dict[str, Any]] = []
    selected_samples: list[tuple[int, dict[str, Any]]] = []
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    started = time.perf_counter()

    try:
        set_hf_cache_env()
        env = gpu_environment()
        reset_cuda_peak_stats()

        logger.log("=== Tiny ST-Test Evaluation ===")
        logger.log(f"model_name: {args.model_name}")
        logger.log(f"precision: {precision}")
        logger.log(f"attn_backend: {args.attn_backend}")
        logger.log(f"device_strategy: {args.device_strategy}")
        logger.log(f"max_samples: {args.max_samples}")
        logger.log(f"samples_per_category: {args.samples_per_category}")
        logger.log(f"max_new_tokens: {args.max_new_tokens}")
        logger.log(f"answer_format_prompt: {args.answer_format_prompt}")
        logger.log(f"category: {args.category}")
        logger.log(f"output_path: {args.output_path}")
        logger.log(f"summary_path: {args.summary_path}")
        logger.log(f"log_path: {args.log_path}")
        logger.log(f"visible_gpu_count: {env['visible_gpu_count']}")
        logger.log(f"gpu_names: {env['gpu_names']}")

        try:
            selected_samples = load_or_select_samples(args, logger)
        except Exception as exc:  # noqa: BLE001 - write summary and stop before model loading.
            logger.log_exception("data loading failed", exc)
            first_error = {
                "error_stage": "data_loading",
                "error_type": exc.__class__.__name__,
                "error_message": short_error_message(exc),
            }
            summary = build_summary(
                args=args,
                precision=precision,
                env=env,
                selected_samples=[],
                records=[],
                model_load_pass=False,
                model_load_time_sec=None,
                total_eval_time_sec=time.perf_counter() - started,
                device_map=None,
                is_multi_gpu=False,
                first_error=first_error,
            )
            summary["conclusion_hint"] = "LOAD_FAIL"
            write_json(args.summary_path, summary)
            logger.log("TINY_EVAL_PARTIAL")
            return 1

        if not selected_samples:
            first_error = {
                "error_stage": "data_loading",
                "error_type": "NoSamplesSelected",
                "error_message": "No ST-Test samples matched the requested category limits.",
            }
            summary = build_summary(
                args=args,
                precision=precision,
                env=env,
                selected_samples=[],
                records=[],
                model_load_pass=False,
                model_load_time_sec=None,
                total_eval_time_sec=time.perf_counter() - started,
                device_map=None,
                is_multi_gpu=False,
                first_error=first_error,
            )
            write_json(args.summary_path, summary)
            logger.log("TINY_EVAL_PARTIAL")
            return 1

        try:
            requested_map = requested_device_map(args.device_strategy, logger)
            model, processor, tokenizer, model_load_time_sec = load_model_for_strategy(
                args.model_name,
                precision,
                args.attn_backend,
                args.device_strategy,
                logger,
            )
            device_map = actual_device_map(model, requested_map)
            cuda_devices = collect_cuda_devices(device_map)
            is_multi_gpu = len(cuda_devices) >= 2
            logger.log(f"actual_device_map: {device_map}")
            logger.log(f"is_actually_multi_gpu: {is_multi_gpu}")
            if args.device_strategy == "dual_auto" and not is_multi_gpu:
                logger.log("DUAL_AUTO_NOT_ACTUALLY_MULTI_GPU")
            if args.device_strategy == "dual_balanced" and not is_multi_gpu:
                logger.log("DUAL_BALANCED_NOT_ACTUALLY_MULTI_GPU")
            patch_timeseries_merge_device(model, logger)
            logger.log(f"gpu_memory_after_model_load: {gpu_memory_snapshot()}")
        except Exception as exc:  # noqa: BLE001 - do not run inference if loading failed.
            logger.log_exception("model loading failed", exc)
            return write_load_failure_outputs(args, precision, env, selected_samples, started, exc, logger)

        for offset, (index, sample) in enumerate(selected_samples, start=1):
            logger.log("-" * 80)
            logger.log(f"[{offset}/{len(selected_samples)}] index={index}, category={sample.get('category')}")
            record = base_record(index, sample, args, precision, device_map, is_multi_gpu)
            record["gpu_memory_before_generate"] = gpu_memory_snapshot()
            sample_started = time.perf_counter()
            try:
                synchronize_cuda()
                prediction = generate_prediction(
                    model,
                    processor,
                    tokenizer,
                    sample,
                    args.max_new_tokens,
                    args.answer_format_prompt,
                )
                synchronize_cuda()
                latency = time.perf_counter() - sample_started
                record["prediction"] = prediction
                record["latency_sec"] = round(latency, 3)
                record["gpu_memory_after_generate"] = gpu_memory_snapshot()
                logger.log(f"GENERATE_PASS index={index} latency_sec={latency:.3f}")

                target_answer, target_error = parse_choice(sample.get("output"))
                parsed_prediction, prediction_error = parse_choice(prediction)
                record["parsed_prediction"] = parsed_prediction
                if target_error is not None:
                    record["target_parse_error"] = target_error
                if prediction_error is not None or target_error is not None:
                    record["parse_failed"] = True
                    apply_record_error(record, "parse", prediction_error or target_error or "parse_failed")
                    logger.log(f"PARSE_FAIL index={index}: {record['error_message']}")
                else:
                    record["parse_failed"] = False
                    record["is_correct"] = target_answer == parsed_prediction
                    logger.log(
                        "PARSE_PASS "
                        f"index={index} target={target_answer} parsed_prediction={parsed_prediction} "
                        f"is_correct={record['is_correct']}"
                    )
            except Exception as exc:  # noqa: BLE001 - keep evaluating remaining samples.
                synchronize_cuda()
                record["latency_sec"] = round(time.perf_counter() - sample_started, 3)
                record["gpu_memory_after_generate"] = gpu_memory_snapshot()
                stage = "generate"
                message = str(exc).lower()
                if "placeholder" in message or "timeseries" in message or "processor" in message:
                    stage = "processor"
                apply_record_error(record, stage, exc)
                logger.log_exception(f"GENERATE_FAIL index={index} stage={stage}", exc)

            append_jsonl(args.output_path, record)
            records.append(record)

        summary = build_summary(
            args=args,
            precision=precision,
            env=env,
            selected_samples=selected_samples,
            records=records,
            model_load_pass=True,
            model_load_time_sec=model_load_time_sec,
            total_eval_time_sec=time.perf_counter() - started,
            device_map=device_map,
            is_multi_gpu=is_multi_gpu,
        )
        write_json(args.summary_path, summary)
        logger.log("=== Summary ===")
        logger.log(json.dumps(summary, ensure_ascii=False, indent=2))

        if summary["num_generate_failed"] == 0:
            logger.log("TINY_EVAL_PASS")
            return 0

        logger.log("TINY_EVAL_PARTIAL")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
