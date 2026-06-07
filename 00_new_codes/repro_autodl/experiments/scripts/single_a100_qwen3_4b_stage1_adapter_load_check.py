#!/usr/bin/env python3
"""Minimal load check for the Stage 1 Qwen3-4B LoRA adapter."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


REPO_ROOT = Path("/root/autodl-tmp/STReasoner_reproduce")
MODEL_DIR = REPO_ROOT / "base_model/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = Path(
    os.environ.get(
        "STAGE1_ADAPTER_DIR",
        str(REPO_ROOT / "output/single_a100_qwen3_4b_stage1_lora_100steps"),
    )
)


def count_nonzero_lora_b(adapter_path: Path) -> tuple[int, int]:
    state_path = adapter_path / "adapter_model.bin"
    state = torch.load(state_path, map_location="cpu")
    lora_b = [(name, tensor) for name, tensor in state.items() if "lora_B" in name]
    nonzero = sum(int(tensor.abs().sum().item() > 0) for _, tensor in lora_b)
    return nonzero, len(lora_b)


def main() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"model_dir={MODEL_DIR}")
    print(f"adapter_dir={ADAPTER_DIR}")
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"missing model dir: {MODEL_DIR}")
    if not ADAPTER_DIR.is_dir():
        raise SystemExit(f"missing adapter dir: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    print(f"tokenizer_class={tokenizer.__class__.__name__}")
    print(f"processor_class={processor.__class__.__name__}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    print(f"base_model_class={base_model.__class__.__name__}")
    print(f"base_model_type={getattr(base_model.config, 'model_type', None)}")
    print(f"has_ts_encoder={hasattr(base_model, 'ts_encoder')}")

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
    print(f"peft_model_class={model.__class__.__name__}")
    print(f"active_adapter={model.active_adapter}")
    print(f"peft_config_keys={list(model.peft_config.keys())}")
    print(f"adapter_loaded={ADAPTER_DIR}")

    nonzero, total = count_nonzero_lora_b(ADAPTER_DIR)
    print(f"nonzero_lora_B={nonzero}/{total}")
    if total == 0 or nonzero != total:
        raise SystemExit("adapter lora_B tensors are not fully nonzero")

    del model
    torch.cuda.empty_cache()
    print("adapter_load_check=ok")


if __name__ == "__main__":
    main()
