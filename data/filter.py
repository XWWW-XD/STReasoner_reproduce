import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_config import TRANSFORMERS_CACHE_PATH, apply_cache_config

apply_cache_config()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyze_and_filter_timeseries(input_directory, output_directory, min_len=32, max_len=512,
                                  max_prompt_length=16384, check_tokenized_length=True,
                                  model_path=None):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    # Load tokenizer if we need to check tokenized length
    tokenizer = None
    processor = None
    ts_token_start_index = 151665  # <ts> token id
    ts_token_end_index = 151666    # <ts/> token id
    
    if check_tokenized_length:
        try:
            from transformers import AutoProcessor
            print("Loading tokenizer for length checking...")
            if model_path is None:
                model_path = os.path.join(REPO_ROOT, "base_model", "Qwen2.5-7B")
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=TRANSFORMERS_CACHE_PATH,
            )
            print(f"✓ Tokenizer loaded from {model_path}")
            print(f"  TS token IDs: <ts>={ts_token_start_index}, <ts/>={ts_token_end_index}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load tokenizer: {e}")
            print("Skipping tokenized length check...")
            check_tokenized_length = False
            
    jsonl_files = glob.glob(os.path.join(input_directory, "*.jsonl"))
    
    # Statistics collectors
    stats_before = collections.defaultdict(int)
    stats_after = collections.defaultdict(int)
    total_lines_before = 0
    total_lines_after = 0
    
    # New: Track reasons for filtering
    filtered_reasons = {
        'ts_length_out_of_range': 0,
        'ts_length_inconsistent': 0,
        'ts_placeholder_mismatch': 0,
        'ts_token_pair_mismatch': 0,
        'prompt_too_long': 0,  # New reason for tokenized length > max_prompt_length
    }
    
    # Track tokenized length statistics
    tokenized_lengths = []
    
    print(f"Found {len(jsonl_files)} JSONL files to process in {input_directory}.")
    print(f"Filtered files will be saved to {output_directory}.")
    
    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_directory, filename)
        
        filtered_lines = []
        lines_before_file = 0
        lines_after_file = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    lines_before_file += 1
                    try:
                        data = json.loads(line)
                        keep_row = True
                        filter_reason = None
                        
                        # Check time series related constraints
                        if 'timeseries' in data and isinstance(data['timeseries'], list):
                            row_lengths = []
                            for ts in data['timeseries']:
                                if isinstance(ts, list):
                                    ts_len = len(ts)
                                    row_lengths.append(ts_len)
                                    stats_before[ts_len] += 1
                                    
                                    # Global length constraint
                                    if ts_len < min_len or ts_len > max_len:
                                        keep_row = False
                                        if filter_reason is None:
                                            filter_reason = 'ts_length_out_of_range'
                            
                            # Extra consistency checks only if still a candidate
                            if keep_row and row_lengths:
                                # 1) All time series in the same row should have identical length
                                if len(set(row_lengths)) > 1:
                                    keep_row = False
                                    filter_reason = 'ts_length_inconsistent'
                            
                            if keep_row:
                                # 2) Number of <ts><ts/> placeholders in "input" should
                                #    match the number of time series segments
                                input_field = data.get("input", "")
                                if isinstance(input_field, str):
                                    ts_placeholder_count = input_field.count("<ts><ts/>")
                                else:
                                    ts_placeholder_count = 0
                                
                                if ts_placeholder_count != len(data["timeseries"]):
                                    keep_row = False
                                    filter_reason = 'ts_placeholder_mismatch'
                            
                            if keep_row:
                                # 3) NEW: Check if <ts> and <ts/> token pairs match
                                input_field = data.get("input", "")
                                if isinstance(input_field, str):
                                    # 正确的格式是 <ts> 和 <ts/> (自闭合标签)
                                    ts_start_count = input_field.count("<ts>")
                                    ts_end_count = input_field.count("<ts/>")
                                    
                                    if ts_start_count != ts_end_count:
                                        keep_row = False
                                        filter_reason = 'ts_token_pair_mismatch'
                            
                            if keep_row and check_tokenized_length:
                                # 4) NEW: Check tokenized prompt length and token pairs
                                try:
                                    input_text = data.get("input", "")
                                    timeseries = data.get("timeseries", [])
                                    
                                    # Tokenize with timeseries (same as in GRPO trainer)
                                    tokenized = processor(
                                        text=[input_text],
                                        timeseries=timeseries,
                                        return_tensors="pt",
                                        padding=False,
                                        add_special_tokens=False,
                                    )
                                    
                                    input_ids = tokenized["input_ids"][0]  # Get first (and only) sample
                                    tokenized_length = len(input_ids)
                                    tokenized_lengths.append(tokenized_length)
                                    
                                    # Check if tokenized length exceeds limit
                                    if tokenized_length > max_prompt_length:
                                        keep_row = False
                                        filter_reason = 'prompt_too_long'
                                    else:
                                        # Check if <ts> and <ts/> token counts match in tokenized ids
                                        ts_start_count = (input_ids == ts_token_start_index).sum().item()
                                        ts_end_count = (input_ids == ts_token_end_index).sum().item()
                                        
                                        if ts_start_count != ts_end_count:
                                            keep_row = False
                                            filter_reason = 'ts_token_pair_mismatch'
                                            # Debug info for first few mismatches
                                            if filtered_reasons['ts_token_pair_mismatch'] < 5:
                                                print(f"    Line {lines_before_file}: <ts>={ts_start_count}, <ts/>={ts_end_count} (mismatch!)")
                                except Exception as e:
                                    # If tokenization fails, skip this sample
                                    print(f"  ⚠️  Tokenization error on line {lines_before_file}: {e}")
                                    keep_row = False
                                    filter_reason = 'prompt_too_long'
                            
                            if keep_row:
                                filtered_lines.append(line)
                                lines_after_file += 1
                                # Update after stats
                                for l in row_lengths:
                                    stats_after[l] += 1
                            else:
                                # Record the reason for filtering
                                if filter_reason:
                                    filtered_reasons[filter_reason] += 1
                        else:
                            # Keep rows without timeseries field
                            filtered_lines.append(line)
                            lines_after_file += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            total_lines_before += lines_before_file
            total_lines_after += lines_after_file
            
            # Write filtered content to new file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for line in filtered_lines:
                    f.write(line)
            
            print(f"File {filename}: {lines_before_file} -> {lines_after_file} lines (Removed {lines_before_file - lines_after_file})")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("\n" + "="*50)
    print("Filtering Statistics Report")
    print("="*50)
    print(f"Total Rows Processed: {total_lines_before}")
    print(f"Total Rows Remaining: {total_lines_after}")
    print(f"Total Rows Removed:   {total_lines_before - total_lines_after}")
    
    # Show detailed filtering reasons
    print("\n" + "-"*50)
    print("Filtering Reasons Breakdown:")
    print("-"*50)
    total_filtered = sum(filtered_reasons.values())
    for reason, count in sorted(filtered_reasons.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total_lines_before * 100) if total_lines_before > 0 else 0
            reason_display = reason.replace('_', ' ').title()
            print(f"  {reason_display:<35} {count:>6} ({percentage:>5.2f}%)")
    
    if total_filtered > 0:
        print(f"  {'Total Filtered':<35} {total_filtered:>6}")
    
    # Show tokenized length statistics
    if check_tokenized_length and tokenized_lengths:
        print("\n" + "-"*50)
        print("Tokenized Prompt Length Statistics:")
        print("-"*50)
        print(f"  Samples checked: {len(tokenized_lengths)}")
        print(f"  Min length: {min(tokenized_lengths)}")
        print(f"  Max length: {max(tokenized_lengths)}")
        print(f"  Mean length: {sum(tokenized_lengths)/len(tokenized_lengths):.1f}")
        print(f"  Max allowed: {max_prompt_length}")
        over_limit = sum(1 for l in tokenized_lengths if l > max_prompt_length)
        if over_limit > 0:
            print(f"  ⚠️  Samples over limit: {over_limit} (filtered out)")
    
    # Calculate TS segment stats
    total_ts_before = sum(stats_before.values())
    total_ts_after = sum(stats_after.values())
    
    print(f"\nTotal Time Series Segments: {total_ts_before} -> {total_ts_after}")
    
    # Distribution Comparison (Summary)
    print("\nLength Distribution Summary (Top 10):")
    print(f"{'Length':<10} {'Before':<10} {'After':<10} {'Diff':<10}")
    print("-" * 40)
    
    # Find top frequent lengths from 'before' to show impact
    top_lengths = sorted(stats_before.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for length, count_before in top_lengths:
        count_after = stats_after.get(length, 0)
        diff = count_before - count_after
        print(f"{length:<10} {count_before:<10} {count_after:<10} {diff:<10}")

    # Show what was removed (lengths outside range)
    print(f"\nRemoved Lengths (Examples outside [{min_len}, {max_len}]):")
    removed_counts = {k: v for k, v in stats_before.items() if k < min_len or k > max_len}
    if removed_counts:
        for length in sorted(removed_counts.keys()):
            print(f"Length {length}: {removed_counts[length]} segments removed (associated rows dropped)")
    else:
        print("No lengths found outside the range.")

def main():
    parser = argparse.ArgumentParser(description="Filter reasoning JSONL files by time-series length and tokenized prompt length.")
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "reasoning_before_filter"),
                        help="Directory containing input *.jsonl files (default: data/reasoning_before_filter).")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "reasoning"),
                        help="Directory to write filtered *.jsonl files (default: data/reasoning).")
    parser.add_argument("--min_len", type=int, default=32,
                        help="Minimum allowed time-series length (default: 32).")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum allowed time-series length (default: 512).")
    parser.add_argument("--max_prompt_length", type=int, default=16384,
                        help="Maximum allowed tokenized prompt length (default: 16384).")
    parser.add_argument("--no_token_check", action="store_true",
                        help="Skip the (slow) tokenized-prompt-length sanity check.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to tokenizer model (default: base_model/Qwen2.5-7B under repo root).")
    args = parser.parse_args()

    analyze_and_filter_timeseries(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        max_prompt_length=args.max_prompt_length,
        check_tokenized_length=not args.no_token_check,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
