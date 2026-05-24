#!/usr/bin/env python3
"""Minimal Hugging Face loading smoke test for STReasoner checkpoints.

This script intentionally does not import project training, inference, or
evaluation code. It only checks whether the requested HF checkpoint can be
loaded, preferably with 4-bit bitsandbytes quantization, on the current machine.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from datetime import datetime
from importlib import metadata, util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_config import TRANSFORMERS_CACHE_PATH, apply_cache_config

apply_cache_config()

DEFAULT_MODEL_NAME = "Time-HD-Anonymous/STReasoner-8B"
DEFAULT_OUTPUT_LOG = "repro_kaggle/outputs/early_smoke_tests/load_streasoner_smoke.log"


class TeeLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._fh.write(message + "\n")
        self._fh.flush()

    def log_exception(self, title: str, exc: BaseException) -> None:
        self.log(f"[ERROR] {title}: {exc.__class__.__name__}: {exc}")
        self.log(traceback.format_exc())

    def close(self) -> None:
        self._fh.close()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test loading STReasoner from Hugging Face with optional 4-bit quantization."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--load_in_4bit", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--attn_backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--output_log", default=DEFAULT_OUTPUT_LOG)
    return parser.parse_args()


def set_hf_cache_env() -> None:
    apply_cache_config()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def log_environment(logger: TeeLogger) -> None:
    import torch

    logger.log("=== Environment ===")
    logger.log(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"Python version: {sys.version.replace(os.linesep, ' ')}")
    logger.log(f"platform: {platform.platform()}")
    logger.log(f"torch version: {torch.__version__}")
    logger.log(f"torch.version.cuda: {torch.version.cuda}")
    logger.log(f"cuda available: {torch.cuda.is_available()}")
    logger.log(f"gpu count: {torch.cuda.device_count()}")

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / 1024**3
        logger.log(f"gpu {idx}: {props.name}, total memory: {total_gib:.2f} GiB")

    logger.log(f"transformers version: {package_version('transformers')}")
    logger.log(f"accelerate version: {package_version('accelerate')}")
    logger.log(f"bitsandbytes version: {package_version('bitsandbytes')}")

    if util.find_spec("flash_attn") is None:
        logger.log("[WARNING] flash_attn is not available. Continuing without it.")
    else:
        logger.log("flash_attn available: yes")

    logger.log("=== Hugging Face Cache ===")
    logger.log(f"HF_HOME={os.environ.get('HF_HOME')}")
    logger.log(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    logger.log(f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
    logger.log(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
    logger.log(f"TORCH_HOME={os.environ.get('TORCH_HOME')}")


def log_cpu_ram(logger: TeeLogger) -> None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        logger.log(
            "CPU RAM: "
            f"total={vm.total / 1024**3:.2f} GiB, "
            f"available={vm.available / 1024**3:.2f} GiB, "
            f"used={vm.used / 1024**3:.2f} GiB"
        )
        return
    except Exception:
        pass

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        interesting = [line for line in meminfo if line.startswith(("MemTotal:", "MemAvailable:"))]
        logger.log("CPU RAM: " + "; ".join(interesting))
    except Exception as exc:
        logger.log(f"CPU RAM: unavailable ({exc})")


def class_name(obj: Any | None) -> str:
    return "None" if obj is None else obj.__class__.__name__


def load_processor_and_tokenizer(model_name: str, logger: TeeLogger) -> tuple[Any | None, Any | None]:
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    tokenizer = None

    logger.log("=== Processor / Tokenizer ===")
    try:
        logger.log("Trying AutoProcessor.from_pretrained(...)")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=TRANSFORMERS_CACHE_PATH,
        )
        logger.log(f"AutoProcessor loaded: {class_name(processor)}")
    except Exception as exc:
        logger.log_exception("AutoProcessor load failed", exc)

    try:
        logger.log("Trying AutoTokenizer.from_pretrained(...)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=TRANSFORMERS_CACHE_PATH,
        )
        logger.log(f"AutoTokenizer loaded: {class_name(tokenizer)}")
    except Exception as exc:
        logger.log_exception("AutoTokenizer load failed", exc)

    return processor, tokenizer


def load_config(model_name: str, attn_backend: str, logger: TeeLogger) -> Any:
    from transformers import AutoConfig

    logger.log("=== Config ===")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=TRANSFORMERS_CACHE_PATH)
    logger.log(f"config class: {class_name(config)}")

    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(config, attr):
            current = getattr(config, attr)
            logger.log(f"config.{attr}: {current!r} -> {attn_backend!r}")
            setattr(config, attr, attn_backend)

    return config


def build_quantization_config(load_in_4bit: bool, logger: TeeLogger) -> Any | None:
    if not load_in_4bit:
        logger.log("4-bit loading disabled by --load_in_4bit false.")
        return None

    from transformers import BitsAndBytesConfig
    import torch

    logger.log("Using 4-bit BitsAndBytesConfig with torch.float16 compute dtype.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    model_name: str,
    config: Any,
    quantization_config: Any | None,
    attn_backend: str,
    logger: TeeLogger,
) -> Any:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM

    common_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "config": config,
        "cache_dir": TRANSFORMERS_CACHE_PATH,
    }
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    logger.log("=== Model Load ===")
    logger.log("Trying AutoModelForCausalLM.from_pretrained(...)")
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
        logger.log_exception("AutoModelForCausalLM load failed", exc)
        if attn_backend == "sdpa":
            logger.log("[HINT] If this traceback points to SDPA attention, rerun with --attn_backend eager.")

    logger.log("Trying AutoModel.from_pretrained(...)")
    try:
        return AutoModel.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
        logger.log_exception("AutoModel load failed", exc)
        if attn_backend == "sdpa":
            logger.log("[HINT] If this traceback points to SDPA attention, rerun with --attn_backend eager.")
        raise


def log_gpu_memory(logger: TeeLogger) -> None:
    import torch

    logger.log("=== GPU Memory ===")
    if not torch.cuda.is_available():
        logger.log("CUDA is not available.")
        return

    for idx in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        logger.log(f"gpu {idx}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")


def log_load_success(model: Any, processor: Any | None, tokenizer: Any | None, logger: TeeLogger) -> None:
    logger.log("=== Load Success ===")
    logger.log(f"model class: {class_name(model)}")
    logger.log(f"processor class: {class_name(processor)}")
    logger.log(f"tokenizer class: {class_name(tokenizer)}")

    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        logger.log(f"model device map: {device_map}")
    else:
        logger.log("model device map: unavailable")

    log_gpu_memory(logger)
    log_cpu_ram(logger)


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def build_text_inputs(prompt: str, processor: Any | None, tokenizer: Any | None, logger: TeeLogger) -> Any | None:
    if processor is not None:
        try:
            logger.log("Trying to tokenize prompt with processor.")
            return processor(text=prompt, return_tensors="pt")
        except Exception as exc:
            logger.log_exception("processor(text=...) failed", exc)
            try:
                logger.log("Trying to tokenize prompt with processor(prompt, ...).")
                return processor(prompt, return_tensors="pt")
            except Exception as inner_exc:
                logger.log_exception("processor(prompt, ...) failed", inner_exc)

    if tokenizer is not None:
        try:
            logger.log("Trying to tokenize prompt with tokenizer.")
            return tokenizer(prompt, return_tensors="pt")
        except Exception as exc:
            logger.log_exception("tokenizer(prompt, ...) failed", exc)

    return None


def try_generate(model: Any, processor: Any | None, tokenizer: Any | None, logger: TeeLogger) -> bool:
    import torch

    logger.log("=== Minimal Generate Test ===")
    if not hasattr(model, "generate"):
        logger.log("Model object has no generate method. Skipping generate test; model loading still succeeded.")
        logger.log("GENERATE_FAIL_BUT_MODEL_LOAD_PASS")
        return False

    prompt = "Hello."
    inputs = build_text_inputs(prompt, processor, tokenizer, logger)
    if inputs is None:
        logger.log("No processor/tokenizer could tokenize the prompt. Skipping generate test; model loading still succeeded.")
        logger.log("GENERATE_FAIL_BUT_MODEL_LOAD_PASS")
        return False

    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        inputs = move_inputs_to_device(inputs, device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        logger.log(f"generate output tensor shape: {tuple(outputs.shape)}")
        decoder = tokenizer if tokenizer is not None else processor
        if decoder is not None and hasattr(decoder, "batch_decode"):
            decoded = decoder.batch_decode(outputs, skip_special_tokens=True)
            logger.log(f"generate decoded output: {decoded}")
        logger.log("Minimal text generate succeeded.")
        logger.log("GENERATE_PASS")
        return True
    except Exception as exc:
        logger.log_exception("Minimal text generate failed", exc)
        logger.log(
            "Model loading succeeded, but minimal text generate failed. "
            "This may be expected if the model requires time-series inputs; "
            "next step is to inspect the processor and the official inference script."
        )
        logger.log("GENERATE_FAIL_BUT_MODEL_LOAD_PASS")
        return False


def main() -> int:
    args = parse_args()
    logger = TeeLogger(args.output_log)
    try:
        set_hf_cache_env()
        log_environment(logger)
        logger.log("=== Arguments ===")
        logger.log(f"model_name: {args.model_name}")
        logger.log(f"load_in_4bit: {args.load_in_4bit}")
        logger.log(f"attn_backend: {args.attn_backend}")
        logger.log(f"output_log: {args.output_log}")

        processor, tokenizer = load_processor_and_tokenizer(args.model_name, logger)
        config = load_config(args.model_name, args.attn_backend, logger)
        quantization_config = build_quantization_config(args.load_in_4bit, logger)
        model = load_model(args.model_name, config, quantization_config, args.attn_backend, logger)
        log_load_success(model, processor, tokenizer, logger)
        try_generate(model, processor, tokenizer, logger)

        logger.log("=== Result ===")
        logger.log("MODEL_LOAD_PASS")
        return 0
    except Exception as exc:
        logger.log_exception("MODEL_LOAD_FAIL", exc)
        logger.log("=== Result ===")
        logger.log("MODEL_LOAD_FAIL")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
