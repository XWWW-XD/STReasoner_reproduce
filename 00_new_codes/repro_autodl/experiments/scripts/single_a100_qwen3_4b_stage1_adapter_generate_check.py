#!/usr/bin/env python3
"""Minimal generation check for the Stage 1 Qwen3-4B LoRA adapter.

This is not an evaluation script. It only proves that the saved adapter can be
loaded with the base model and enter the model.generate path with time-series
inputs.
"""

from __future__ import annotations

import json
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
DATA_FILE = REPO_ROOT / "data/ST-Bench/ST-Align/alignment_train.jsonl"
SAMPLE_INDEX = int(os.environ.get("STAGE1_SAMPLE_INDEX", "0"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE1_MAX_NEW_TOKENS", "16"))


def load_sample(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample index out of range: {index}")


def main() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"model_dir={MODEL_DIR}")
    print(f"adapter_dir={ADAPTER_DIR}")
    print(f"data_file={DATA_FILE}")
    print(f"sample_index={SAMPLE_INDEX}")
    print(f"max_new_tokens={MAX_NEW_TOKENS}")

    sample = load_sample(DATA_FILE, SAMPLE_INDEX)
    print(f"category={sample.get('category')}")
    print(f"gold={sample.get('output')}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sample["input"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        timeseries=sample["timeseries"],
        return_tensors="pt",
        padding=False,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"input_ids_shape={tuple(inputs['input_ids'].shape)}")
    if "timeseries" in inputs:
        print(f"timeseries_shape={tuple(inputs['timeseries'].shape)}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
    model.eval()

    model_inputs = {
        key: value.to("cuda:0") if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "timeseries" in model_inputs:
        model_inputs["timeseries"] = model_inputs["timeseries"].to(dtype=torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"generated_new_tokens={new_tokens.numel()}")
    print(f"response={response!r}")
    print("adapter_generate_check=ok")


if __name__ == "__main__":
    main()
