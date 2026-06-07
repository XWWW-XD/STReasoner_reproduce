#!/usr/bin/env python3
"""Inspect STReasoner Qwen3 time-series encoder code/config without loading model weights."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/STReasoner_reproduce')
MODEL_DIR = ROOT / 'base_model/Qwen3-4B-Instruct-2507'
OUT_DIR = ROOT / '00_new_codes/reports/artifacts/mlp_encoder_focused_analysis'


def main() -> None:
    config = json.loads((MODEL_DIR / 'config.json').read_text())
    ts = config['ts']
    patch_size = ts['patch_size']
    pos_dim = ts.get('embedding_dim', 16)
    if ts.get('use_position_embedding'):
        mlp_input_dim = patch_size + patch_size * pos_dim
    elif ts.get('use_position_idx'):
        mlp_input_dim = patch_size * 2
    else:
        mlp_input_dim = patch_size
    facts = {
        'modeling_file': str((MODEL_DIR / 'modeling_qwen3_ts.py').relative_to(ROOT)),
        'processing_file': str((MODEL_DIR / 'processing_qwen3_ts.py').relative_to(ROOT)),
        'config_file': str((MODEL_DIR / 'config.json').relative_to(ROOT)),
        'class_name': 'TimeSeriesEmbedding',
        'model_class': 'Qwen3TSForCausalLM',
        'processor_class': 'Qwen3TSProcessor',
        'encoder_config': ts,
        'derived_mlp_input_dim': mlp_input_dim,
        'mlp_hidden_dim': ts['hidden_size'],
        'mlp_output_dim': ts['hidden_size'],
        'num_linear_layers': ts['num_layers'],
        'num_gelu_layers': max(0, ts['num_layers'] - 1),
        'activation': 'GELU after each non-final Linear',
        'layer_norm': bool(ts.get('use_layer_norm', False)),
        'dropout': 'not present in TimeSeriesEmbedding.mlp',
        'patch_embedding_count': 'one embedding vector per patch',
        'normalization': 'sp_encoding subtracts mean; if max abs centered value >= 3, divide by max_abs/3; prompt stores offset/scaling/length/max/min/left/right',
        'patchify': 'valid values selected by mask, pad to multiple of patch_size with last valid value, reshape to (num_patches, patch_size)',
        'llm_merge': 'Qwen3TSForCausalLM.forward calls ts_encoder(timeseries), then _merge_input_ids_with_time_series_features replaces <ts> placeholder span in inputs_embeds with patch embeddings',
        'cross_patch_dependency_in_encoder': 'No explicit cross-patch operation in TimeSeriesEmbedding; patches are concatenated into batch dimension and passed independently through the same MLP. Cross-patch reasoning can only occur later through LLM self-attention over patch embeddings.',
    }
    out = OUT_DIR / 'mlp_encoder_facts.json'
    out.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
