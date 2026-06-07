#!/usr/bin/env python3
"""Small generation probe for a Stage 1 Qwen3-4B LoRA adapter.

This is intentionally not a formal ST-Bench evaluation. It loads the adapter
once, runs a few ST-Align training samples through the time-series generation
path, and writes compact JSONL records for report/debug use.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


REPO_ROOT = Path("/root/autodl-tmp/STReasoner_reproduce")
MODEL_DIR = REPO_ROOT / "base_model/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = Path(
    os.environ.get(
        "STAGE1_ADAPTER_DIR",
        str(REPO_ROOT / "output/single_a100_qwen3_4b_stage1_lora_500steps"),
    )
)
DATA_FILE = REPO_ROOT / "data/ST-Bench/ST-Align/alignment_train.jsonl"
DEFAULT_INDICES = "0,1,71,72,121,122"
SAMPLE_INDICES = [
    int(x.strip())
    for x in os.environ.get("STAGE1_SAMPLE_INDICES", DEFAULT_INDICES).split(",")
    if x.strip()
]
MAX_NEW_TOKENS = int(os.environ.get("STAGE1_MAX_NEW_TOKENS", "32"))
OUT_FILE = Path(
    os.environ.get(
        "STAGE1_PROBE_OUT",
        str(REPO_ROOT / "00_new_codes/repro_autodl/experiments/results/stage1_lora_500steps_probe.jsonl"),
    )
)


def normalize_answer(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_samples(path: Path, indices: list[int]) -> dict[int, dict]:
    wanted = set(indices)
    found: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                found[i] = json.loads(line)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise IndexError(f"sample index out of range: {missing}")
    return found


def build_inputs(tokenizer, processor, sample: dict) -> tuple[dict, int]:
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
    return inputs, inputs["input_ids"].shape[-1]


def main() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"model_dir={MODEL_DIR}")
    print(f"adapter_dir={ADAPTER_DIR}")
    print(f"data_file={DATA_FILE}")
    print(f"sample_indices={SAMPLE_INDICES}")
    print(f"max_new_tokens={MAX_NEW_TOKENS}")
    print(f"out_file={OUT_FILE}")

    samples = load_samples(DATA_FILE, SAMPLE_INDICES)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
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

    records = []
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for index in SAMPLE_INDICES:
            sample = samples[index]
            inputs, prompt_len = build_inputs(tokenizer, processor, sample)
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
            gold = sample.get("output")
            record = {
                "sample_index": index,
                "category": sample.get("category"),
                "gold": gold,
                "response": response,
                "generated_new_tokens": int(new_tokens.numel()),
                "prompt_tokens": int(prompt_len),
                "exact_match_simple": normalize_answer(response) == normalize_answer(gold),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))

    exact = sum(1 for item in records if item["exact_match_simple"])
    print(f"probe_exact_match_simple={exact}/{len(records)}")
    print(f"cuda_max_memory_allocated_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"cuda_max_memory_reserved_gb={torch.cuda.max_memory_reserved() / 1024**3:.2f}")
    print("adapter_probe=ok")


if __name__ == "__main__":
    main()
