#!/usr/bin/env python3
"""Phase3 Waves 15-21: code registry and trace artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "artifacts"

MODULES = {
    "inference_entry": ROOT / "inference/inference_tsmllm_vllm.py",
    "llm_utils": ROOT / "inference/llm_utils.py",
    "prompt_json": ROOT / "inference/prompt.json",
    "evaluate_qa": ROOT / "evaluation/evaluate_qa.py",
    "evaluate_main": ROOT / "evaluation/evaluate.py",
    "supervised_processor": ROOT / "src/llamafactory/data/processor/supervised.py",
    "chatts_vllm": ROOT / "inference/vllm/chatts_vllm.py",
}


def scan_symbols(path: Path, patterns: list[str]) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    hits = []
    for i, line in enumerate(lines, 1):
        for pat in patterns:
            if re.search(pat, line):
                hits.append({"line": i, "text": line.strip()[:120], "pattern": pat})
    return hits[:40]


def build_code_registry() -> dict:
    entries = []
    for name, path in MODULES.items():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        entries.append({"id": name, "path": rel, "exists": path.exists(), "size_lines": len(path.read_text(encoding="utf-8", errors="ignore").splitlines()) if path.exists() else 0})
    return {"modules": entries, "p0_count": sum(1 for e in entries if e["exists"])}


def build_inference_trace() -> dict:
    p = MODULES["inference_entry"]
    u = MODULES["llm_utils"]
    return {
        "generated_answer_fields": scan_symbols(p, [r"num_tokens", r"generated_answer", r"max_tokens", r"SamplingParams"]),
        "worker_sampling": scan_symbols(u, [r"SamplingParams", r"worker_vllm_ts", r"llm\.generate"]),
        "links_to_wave6": {
            "num_tokens_is_input": "inference_tsmllm_vllm.py writes input_token_counts[idx] as num_tokens",
            "max_tokens_default_cli": "6144 for official ST-Test per run.log",
        },
    }


def build_evaluation_trace() -> dict:
    p = MODULES["evaluate_qa"]
    return {
        "load_prediction_files": scan_symbols(p, [r"def load_prediction_files", r"_extract_tag_content", r"predictions\[idx\]"]),
        "normalize_choice": scan_symbols(p, [r"def _normalize_choice", r"def _parse_series"]),
        "links_to_wave4_6": {
            "no_tag_fallback": "_extract_tag_content returns full text.strip() when no <answer> tag",
            "loose_triple_diverse": "causes fake triple-diverse when thinking prefixes differ",
        },
    }


def build_training_trace() -> dict:
    p = MODULES["supervised_processor"]
    return {
        "sft_loss_scope": scan_symbols(p, [r"IGNORE_INDEX", r"response", r"labels"]),
        "links_to_phase1": {
            "thinking_in_loss": "full thinking+answer in CE; evaluate only scores answer tag",
        },
    }


def build_encoder_trace() -> dict:
    p = MODULES["chatts_vllm"]
    return {
        "ts_embedding": scan_symbols(p, [r"TimeSeriesEmbedding", r"patch", r"MLP", r"class"]),
        "links_to_report39": "MLP bottleneck; verbatim numeric restatement often fails strict mismatch",
    }


def build_pipeline_diff() -> dict:
    return {
        "official": {"inference": "inference/inference_tsmllm_vllm.py", "eval": "evaluation/evaluate.py", "dtype": "half/vLLM"},
        "repro_kaggle": {"inference": "repro_kaggle HF generate", "max_tokens": "2048 typical", "comparable": False},
        "repro_autodl_stage2": {"inference": "stage2 scripts HF/vLLM mix", "comparable": "partial"},
    }


def build_crosswalk() -> dict:
    return {
        "rows": [
            {"phenomenon": "no_answer_tag_unclosed_thinking", "code_anchor": "inference max_tokens + evaluate _extract_tag_content fallback", "tier_intervention": "T1-2, T1-3, E1"},
            {"phenomenon": "loose_triple_diverse", "code_anchor": "evaluate_qa load_prediction_files + _normalize_choice", "tier_intervention": "T0-1 strict parser"},
            {"phenomenon": "flip_independent_of_mismatch", "code_anchor": "evaluate ignores thinking; SFT trains full response", "tier_intervention": "T1-1, T2-3"},
            {"phenomenon": "graph_plus_6pp", "code_anchor": "graph text in prompt; stage2.4 regex remove", "tier_intervention": "T1-4, E4"},
            {"phenomenon": "kaggle_not_comparable", "code_anchor": "pipeline_diff repro_kaggle", "tier_intervention": "do not use for official optimization"},
        ]
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "code_registry.json": build_code_registry(),
        "inference_trace.json": build_inference_trace(),
        "evaluation_trace.json": build_evaluation_trace(),
        "training_trace.json": build_training_trace(),
        "encoder_trace.json": build_encoder_trace(),
        "pipeline_diff_ledger.json": build_pipeline_diff(),
        "code_experiment_crosswalk.json": build_crosswalk(),
    }
    for name, data in payloads.items():
        path = OUT / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
