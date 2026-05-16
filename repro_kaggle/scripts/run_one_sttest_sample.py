#!/usr/bin/env python3
"""Run one ST-Test sample through STReasoner as a minimal inference check."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    str2bool,
)


DATASET_NAME = "Time-HD-Anonymous/ST-Bench"
DATASET_SUBSET = "ST-Test"
DEFAULT_OUTPUT_JSON = "repro_kaggle/outputs/one_sttest_prediction.json"
DEFAULT_OUTPUT_LOG = "repro_kaggle/outputs/run_one_sttest_sample.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ST-Test sample through STReasoner.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--load_in_4bit", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--attn_backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_log", default=DEFAULT_OUTPUT_LOG)
    return parser.parse_args()


def preview_text(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def timeseries_lengths(timeseries: Any) -> list[int | None]:
    if not isinstance(timeseries, list):
        return []

    lengths: list[int | None] = []
    for node in timeseries:
        try:
            lengths.append(len(node))
        except TypeError:
            lengths.append(None)
    return lengths


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_result(path: str, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")


def base_result(index: int, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    sample = sample or {}
    return {
        "index": index,
        "category": sample.get("category"),
        "input": sample.get("input"),
        "output": sample.get("output"),
        "prediction": None,
        "error": None,
    }


def load_one_sample(index: int, logger: TeeLogger) -> dict[str, Any]:
    from datasets import load_dataset

    logger.log("=== Data Loading ===")
    logger.log(f"dataset: {DATASET_NAME}")
    logger.log(f"subset: {DATASET_SUBSET}")
    logger.log(f"split: train")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train")
    logger.log(f"num_rows: {dataset.num_rows}")

    if index < 0 or index >= dataset.num_rows:
        raise IndexError(f"--index {index} is out of range for dataset with {dataset.num_rows} rows")

    sample = dict(dataset[index])
    ts = sample.get("timeseries")
    logger.log("=== Sample Summary ===")
    logger.log(f"index: {index}")
    logger.log(f"category: {sample.get('category')}")
    logger.log(f"input preview: {preview_text(sample.get('input'))}")
    logger.log(f"output: {sample.get('output')}")
    logger.log(f"timeseries node count: {len(ts) if isinstance(ts, list) else 'not a list'}")
    logger.log(f"timeseries sequence lengths: {timeseries_lengths(ts)}")
    return sample


def build_processor_inputs(processor: Any, sample: dict[str, Any], logger: TeeLogger) -> Any:
    prompt = sample.get("input")
    timeseries = sample.get("timeseries")
    placeholder_count = prompt.count("<ts><ts/>") if isinstance(prompt, str) else None
    node_count = len(timeseries) if isinstance(timeseries, list) else None

    logger.log("=== Processor Input Construction ===")
    logger.log(f"processor class: {class_name(processor)}")
    logger.log(f"<ts><ts/> placeholder count: {placeholder_count}")
    logger.log(f"timeseries node count: {node_count}")

    if processor is None:
        raise RuntimeError(
            "AutoProcessor is unavailable. Need a processor that accepts text plus timeseries; "
            "next step: inspect processing_qwen3_ts.py and model repo processor files."
        )
    if not isinstance(prompt, str):
        raise TypeError("sample['input'] must be a string for processor(text=...)")
    if not isinstance(timeseries, list):
        raise TypeError("sample['timeseries'] must be a list of node sequences")
    if placeholder_count != node_count:
        raise ValueError(
            "The processor expects one timeseries sequence per <ts><ts/> placeholder; "
            f"found placeholders={placeholder_count}, timeseries nodes={node_count}."
        )

    inputs = processor(text=prompt, timeseries=timeseries, return_tensors="pt")
    logger.log(f"processor output keys: {list(inputs.keys()) if hasattr(inputs, 'keys') else type(inputs).__name__}")
    for key, value in inputs.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        dtype = getattr(value, "dtype", None)
        logger.log(f"  {key}: type={type(value).__name__}, shape={shape}, dtype={dtype}")
    return inputs


def decode_prediction(outputs: Any, inputs: Any, processor: Any | None, tokenizer: Any | None) -> str:
    decoder = tokenizer if tokenizer is not None else processor
    if decoder is None or not hasattr(decoder, "decode"):
        return str(outputs)

    generated_ids = outputs[0]
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if input_ids is not None and hasattr(input_ids, "shape"):
        generated_ids = generated_ids[input_ids.shape[-1] :]

    return decoder.decode(generated_ids, skip_special_tokens=True)


def generate_one(
    model: Any,
    processor: Any | None,
    tokenizer: Any | None,
    inputs: Any,
    max_new_tokens: int,
    logger: TeeLogger,
) -> str:
    import torch

    logger.log("=== Generate ===")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(f"moving inputs to device: {device}")
    inputs = move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    logger.log(f"generate output shape: {tuple(outputs.shape) if hasattr(outputs, 'shape') else type(outputs).__name__}")
    prediction = decode_prediction(outputs, inputs, processor, tokenizer)
    logger.log(f"prediction: {prediction}")
    return prediction


def main() -> int:
    args = parse_args()
    logger = TeeLogger(args.output_log)
    sample: dict[str, Any] | None = None
    result = base_result(args.index)

    try:
        set_hf_cache_env()

        try:
            sample = load_one_sample(args.index, logger)
            result = base_result(args.index, sample)
        except Exception as exc:  # noqa: BLE001 - write structured failure.
            result["error"] = {
                "step": "data loading",
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "next_step": "Check HF dataset availability and confirm the ST-Test train split can be loaded.",
            }
            logger.log_exception("data loading failed", exc)
            write_result(args.output_json, result)
            logger.log("ONE_SAMPLE_RUN_FAIL")
            return 1

        try:
            processor, tokenizer = load_processor_and_tokenizer(args.model_name, logger)
            config = load_config(args.model_name, args.attn_backend, logger)
            quantization_config = build_quantization_config(args.load_in_4bit, logger)
            model = load_model(args.model_name, config, quantization_config, args.attn_backend, logger)
            logger.log(f"model class: {class_name(model)}")
            logger.log(f"processor class: {class_name(processor)}")
            logger.log(f"tokenizer class: {class_name(tokenizer)}")
            device_map = getattr(model, "hf_device_map", None)
            logger.log(f"device map: {device_map if device_map is not None else 'unavailable'}")
            log_gpu_memory(logger)
        except Exception as exc:  # noqa: BLE001 - write structured failure.
            next_step = "Review the model loading traceback; if it is SDPA related, rerun with --attn_backend eager."
            result["error"] = {
                "step": "model loading",
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "next_step": next_step,
            }
            logger.log_exception("model loading failed", exc)
            write_result(args.output_json, result)
            logger.log("ONE_SAMPLE_RUN_FAIL")
            return 1

        try:
            inputs = build_processor_inputs(processor, sample, logger)
        except Exception as exc:  # noqa: BLE001 - write structured failure.
            result["error"] = {
                "step": "processor",
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "next_step": (
                    "Use processing_qwen3_ts.py: processor(text=input, timeseries=timeseries, "
                    "return_tensors='pt'); verify placeholder count equals timeseries node count."
                ),
            }
            logger.log_exception("processor input construction failed", exc)
            write_result(args.output_json, result)
            logger.log("ONE_SAMPLE_RUN_FAIL")
            return 1

        try:
            result["prediction"] = generate_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                logger=logger,
            )
            result["error"] = None
            write_result(args.output_json, result)
            logger.log(f"wrote prediction: {args.output_json}")
            logger.log("ONE_SAMPLE_RUN_PASS")
            return 0
        except Exception as exc:  # noqa: BLE001 - write structured failure.
            result["error"] = {
                "step": "generate",
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "next_step": (
                    "Inspect model.generate accepted kwargs and the Qwen3TS model forward path; "
                    "confirm the processor output key 'timeseries' is expected by the loaded model."
                ),
            }
            logger.log_exception("generate failed", exc)
            write_result(args.output_json, result)
            logger.log("ONE_SAMPLE_RUN_FAIL")
            return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
