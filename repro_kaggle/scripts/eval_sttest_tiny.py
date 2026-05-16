#!/usr/bin/env python3
"""Tiny ST-Test evaluation for validating the Kaggle reproduction chain."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_streasoner_smoke import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    TeeLogger,
    build_quantization_config,
    class_name,
    load_config,
    load_model,
    load_processor_and_tokenizer,
    log_gpu_memory,
    move_inputs_to_device,
    set_hf_cache_env,
)
from run_one_sttest_sample import decode_prediction, json_safe  # noqa: E402


DATASET_NAME = "Time-HD-Anonymous/ST-Bench"
DATASET_SUBSET = "ST-Test"
DEFAULT_OUTPUT_PATH = "repro_kaggle/outputs/sttest_tiny_predictions.jsonl"
DEFAULT_SUMMARY_PATH = "repro_kaggle/outputs/sttest_tiny_summary.json"
DEFAULT_LOG_PATH = "repro_kaggle/outputs/sttest_tiny_eval.log"
VALID_CATEGORIES = ("all", "correlation", "entity", "etiological", "forecasting")
CHOICE_RE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)")
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny ST-Test evaluation.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--category", choices=VALID_CATEGORIES, default="all")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary_path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def parse_answer(text: Any) -> tuple[str | None, str | None]:
    if text is None:
        return None, "empty"

    value = str(text).strip()
    if not value:
        return None, "empty"

    tag_match = ANSWER_TAG_RE.search(value)
    if tag_match:
        tagged = tag_match.group(1).strip()
        if tagged:
            choice_match = CHOICE_RE.search(tagged)
            if choice_match:
                return choice_match.group(1), None
            return tagged, None
        return None, "empty_answer_tag"

    choice_match = CHOICE_RE.search(value)
    if choice_match:
        return choice_match.group(1), None

    return None, "no_answer_tag_or_choice"


def base_record(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "category": sample.get("category"),
        "output": sample.get("output"),
        "prediction": None,
        "parsed_prediction": None,
        "is_correct": None,
        "latency": None,
        "error": None,
    }


def load_samples(max_samples: int, category: str, logger: TeeLogger) -> list[tuple[int, dict[str, Any]]]:
    from datasets import load_dataset

    logger.log("=== Data Loading ===")
    logger.log(f"dataset: {DATASET_NAME}")
    logger.log(f"subset: {DATASET_SUBSET}")
    logger.log("split: train")
    logger.log(f"category filter: {category}")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train")
    logger.log(f"dataset rows: {dataset.num_rows}")

    selected: list[tuple[int, dict[str, Any]]] = []
    for index, sample in enumerate(dataset):
        sample_category = sample.get("category")
        if category != "all" and sample_category != category:
            continue
        selected.append((index, dict(sample)))
        if len(selected) >= max_samples:
            break

    logger.log(f"selected rows: {len(selected)}")
    return selected


def load_streasoner(model_name: str, logger: TeeLogger) -> tuple[Any, Any | None, Any | None]:
    logger.log("=== Model Loading ===")
    processor, tokenizer = load_processor_and_tokenizer(model_name, logger)
    config = load_config(model_name, "sdpa", logger)
    quantization_config = build_quantization_config(True, logger)
    model = load_model(model_name, config, quantization_config, "sdpa", logger)
    model.eval()
    logger.log(f"model class: {class_name(model)}")
    logger.log(f"processor class: {class_name(processor)}")
    logger.log(f"tokenizer class: {class_name(tokenizer)}")
    device_map = getattr(model, "hf_device_map", None)
    logger.log(f"device map: {device_map if device_map is not None else 'unavailable'}")
    log_gpu_memory(logger)
    patch_timeseries_merge_device(model, logger)
    return model, processor, tokenizer


def patch_timeseries_merge_device(model: Any, logger: TeeLogger) -> None:
    """Patch a device mismatch in remote-code generation when device_map spans GPUs.

    The STReasoner remote model can place the text embeddings on cuda:0 and the
    time-series encoder on cuda:1. Its merge helper then mixes a patch-count
    tensor from cuda:1 with text masks from cuda:0. This runtime-only patch keeps
    the helper's inputs on the text embedding device without editing the cached
    Hugging Face module or original project files.
    """

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


def build_inputs(processor: Any | None, sample: dict[str, Any]) -> Any:
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

    return processor(text=prompt, timeseries=timeseries, return_tensors="pt")


def infer_one(
    model: Any,
    processor: Any | None,
    tokenizer: Any | None,
    sample: dict[str, Any],
) -> str:
    import torch

    inputs = build_inputs(processor, sample)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    return decode_prediction(outputs, inputs, processor, tokenizer)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_samples = len(records)
    failed_records = [record for record in records if record.get("error")]
    success_records = [record for record in records if not record.get("error")]
    parse_failed = [record for record in success_records if record.get("parse_failed")]
    latencies = [record["latency"] for record in success_records if isinstance(record.get("latency"), (int, float))]

    accuracy_records = [
        record for record in success_records if isinstance(record.get("is_correct"), bool)
    ]
    accuracy = None
    if accuracy_records:
        accuracy = sum(1 for record in accuracy_records if record["is_correct"]) / len(accuracy_records)

    failure_steps = Counter(record["error"].get("step", "unknown") for record in failed_records)
    summary = {
        "num_samples": num_samples,
        "num_success": len(success_records),
        "num_failed": len(failed_records),
        "parse_fail_rate": (len(parse_failed) / len(success_records)) if success_records else None,
        "accuracy_if_applicable": accuracy,
        "avg_latency": (sum(latencies) / len(latencies)) if latencies else None,
        "failure_steps": dict(failure_steps),
        "parse_fail_count": len(parse_failed),
        "accuracy_denominator": len(accuracy_records),
    }
    return summary


def make_error(step: str, exc: BaseException, next_step: str) -> dict[str, str]:
    return {
        "step": step,
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "next_step": next_step,
    }


def main() -> int:
    args = parse_args()
    logger = TeeLogger(DEFAULT_LOG_PATH)
    records: list[dict[str, Any]] = []
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    try:
        set_hf_cache_env()
        logger.log("=== Tiny ST-Test Evaluation ===")
        logger.log(f"model_name: {args.model_name}")
        logger.log(f"max_samples: {args.max_samples}")
        logger.log(f"category: {args.category}")
        logger.log(f"output_path: {args.output_path}")
        logger.log(f"summary_path: {args.summary_path}")

        try:
            samples = load_samples(args.max_samples, args.category, logger)
        except Exception as exc:  # noqa: BLE001 - write summary instead of crashing.
            logger.log_exception("data loading failed", exc)
            summary = {
                "num_samples": 0,
                "num_success": 0,
                "num_failed": 1,
                "parse_fail_rate": None,
                "accuracy_if_applicable": None,
                "avg_latency": None,
                "failure_steps": {"data loading": 1},
                "error": make_error(
                    "data loading",
                    exc,
                    "Check HF dataset availability and retry load_dataset for Time-HD-Anonymous/ST-Bench ST-Test.",
                ),
            }
            write_json(args.summary_path, summary)
            logger.log("TINY_EVAL_PARTIAL")
            return 1

        if not samples:
            summary = {
                "num_samples": 0,
                "num_success": 0,
                "num_failed": 0,
                "parse_fail_rate": None,
                "accuracy_if_applicable": None,
                "avg_latency": None,
                "failure_steps": {},
                "error": f"No samples matched category={args.category!r}.",
            }
            write_json(args.summary_path, summary)
            logger.log("TINY_EVAL_PARTIAL")
            return 1

        try:
            model, processor, tokenizer = load_streasoner(args.model_name, logger)
        except Exception as exc:  # noqa: BLE001 - mark all selected samples as failed.
            logger.log_exception("model loading failed", exc)
            for index, sample in samples:
                record = base_record(index, sample)
                record["latency"] = 0.0
                record["error"] = make_error(
                    "model loading",
                    exc,
                    "Review model loading traceback; if SDPA is implicated, try adapting this script to pass eager.",
                )
                append_jsonl(args.output_path, record)
                records.append(record)
            summary = summarize(records)
            write_json(args.summary_path, summary)
            logger.log("TINY_EVAL_PARTIAL")
            return 1

        for offset, (index, sample) in enumerate(samples, start=1):
            logger.log("-" * 80)
            logger.log(f"[{offset}/{len(samples)}] index={index}, category={sample.get('category')}")
            record = base_record(index, sample)
            started = time.perf_counter()
            try:
                prediction = infer_one(model, processor, tokenizer, sample)
                latency = time.perf_counter() - started
                target_answer, target_parse_error = parse_answer(sample.get("output"))
                parsed_prediction, prediction_parse_error = parse_answer(prediction)

                record["prediction"] = prediction
                record["parsed_prediction"] = parsed_prediction
                record["latency"] = latency
                record["parse_failed"] = prediction_parse_error is not None
                if record["parse_failed"]:
                    record["parse_error"] = prediction_parse_error
                if target_parse_error is not None:
                    record["target_parse_error"] = target_parse_error

                if target_answer in {"A", "B", "C", "D"} and parsed_prediction in {"A", "B", "C", "D"}:
                    record["is_correct"] = target_answer == parsed_prediction
                else:
                    record["is_correct"] = None

                logger.log(
                    "result: "
                    f"target={target_answer}, parsed_prediction={parsed_prediction}, "
                    f"is_correct={record['is_correct']}, latency={latency:.2f}s"
                )
            except Exception as exc:  # noqa: BLE001 - keep evaluating remaining samples.
                latency = time.perf_counter() - started
                record["latency"] = latency
                step = "processor/generate"
                message = str(exc)
                if "placeholder" in message or "timeseries" in message or "processor" in message:
                    step = "processor"
                record["error"] = make_error(
                    step,
                    exc,
                    (
                        "Check processor(text=input, timeseries=timeseries, return_tensors='pt') "
                        "and the model.generate timeseries kwarg for this sample."
                    ),
                )
                logger.log_exception(f"sample index={index} failed at {step}", exc)

            append_jsonl(args.output_path, record)
            records.append(record)

        summary = summarize(records)
        write_json(args.summary_path, summary)
        logger.log("=== Summary ===")
        logger.log(json.dumps(summary, ensure_ascii=False, indent=2))

        if summary["num_failed"] == 0:
            logger.log("TINY_EVAL_PASS")
            return 0

        logger.log("TINY_EVAL_PARTIAL")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
