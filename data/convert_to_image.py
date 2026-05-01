#!/usr/bin/env python3
"""
Convert time series data in JSONL files to images for vision-language model training.
Replaces "timeseries" field with "images" field containing image file paths.
"""

import argparse
import base64
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _create_timeseries_figure(
    timeseries: Sequence[Sequence[float]],
    cols: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create a matplotlib figure for time series visualization.
    
    Args:
        timeseries: List of time series arrays
        cols: Optional column names for legend
    
    Returns:
        matplotlib Figure object
    """
    if not timeseries:
        raise ValueError("No time series data")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(timeseries)))

    for idx, series in enumerate(timeseries):
        label = cols[idx] if cols and idx < len(cols) else f"Node {idx}"
        ax.plot(series, label=label, color=colors[idx], linewidth=1.5)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def generate_image_from_timeseries(
    fig_dir: str,
    case_idx: int,
    timeseries: List[List[float]],
    cols: Optional[List[str]] = None,
    return_base64: bool = False,
) -> str:
    """
    Generate a visualization of time series data.
    
    Args:
        fig_dir: Directory to save the figure
        case_idx: Index of the current case (used for naming)
        timeseries: List of time series arrays
        cols: Optional column names for legend
        return_base64: If True, return base64 string instead of file path
    
    Returns:
        File path or base64 encoded string of the image
    """
    os.makedirs(fig_dir, exist_ok=True)

    fig = _create_timeseries_figure(timeseries, cols)

    # Save to file (DPI=72 for faster generation, still readable)
    img_path = os.path.join(fig_dir, f"case_{case_idx}.png")
    plt.savefig(img_path, dpi=72, format='png', bbox_inches='tight')

    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=72, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_b64
    
    plt.close(fig)
    return img_path


def _process_single_sample(
    args: Tuple[int, Dict[str, Any], str, bool, str]
) -> Tuple[int, Dict[str, Any]]:
    """Worker function for parallel processing."""
    idx, sample, image_dir, use_base64, output_dir = args
    
    timeseries = sample.get("timeseries", [])
    
    if not timeseries:
        new_sample = {k: v for k, v in sample.items() if k != "timeseries"}
        new_sample["images"] = []
        return idx, new_sample
    
    # Generate image
    result = generate_image_from_timeseries(
        fig_dir=image_dir,
        case_idx=idx,
        timeseries=timeseries,
        return_base64=use_base64,
    )
    
    # Create new sample with image instead of timeseries
    new_sample = {k: v for k, v in sample.items() if k != "timeseries"}
    
    if use_base64:
        new_sample["images"] = [result]
    else:
        # Use path relative to project root (where training runs from)
        new_sample["images"] = [result]
    
    # Update input text
    if "input" in new_sample:
        input_text = new_sample["input"]
        while "<ts><ts/>" in input_text:
            input_text = input_text.replace("<ts><ts/>", "", 1)
        if not input_text.startswith("<image>"):
            input_text = "<image>\n" + input_text.strip()
        new_sample["input"] = input_text
    
    return idx, new_sample


def convert_jsonl_to_image(
    input_path: str,
    output_path: str,
    image_dir: str,
    use_base64: bool = False,
    max_samples: Optional[int] = None,
    num_workers: int = 32,
) -> None:
    """
    Convert a JSONL file with timeseries to one with images.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        image_dir: Directory to save generated images
        use_base64: If True, store base64 strings; otherwise store file paths
        max_samples: Maximum number of samples to process (for testing)
        num_workers: Number of parallel workers
    """
    # Read all samples
    samples: List[Dict[str, Any]] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples from {input_path} with {num_workers} workers")
    
    # Ensure image directory exists
    os.makedirs(image_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    task_args = [
        (idx, sample, image_dir, use_base64, output_dir)
        for idx, sample in enumerate(samples)
    ]
    
    # Process in parallel
    results: Dict[int, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_sample, args): args[0] for args in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            idx, new_sample = future.result()
            results[idx] = new_sample
    
    # Sort by index and write output
    converted_samples = [results[i] for i in range(len(results))]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(converted_samples)} samples to {output_path}")
    print(f"Images saved to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert timeseries JSONL files to image-based format for VLM training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/reasoning",
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reasoning_image",
        help="Output directory for converted JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/reasoning_image/figures",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["_cot.jsonl", "_rl.jsonl", "_rl_new.jsonl"],
        help="File patterns to convert",
    )
    parser.add_argument(
        "--use_base64",
        action="store_true",
        help="Store images as base64 strings instead of file paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per file (for testing)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    # Find matching files
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    files_to_convert = []
    for filename in os.listdir(input_dir):
        for pattern in args.patterns:
            if filename.endswith(pattern):
                files_to_convert.append(filename)
                break
    
    if not files_to_convert:
        print(f"No files matching patterns {args.patterns} found in {input_dir}")
        return
    
    print(f"Found {len(files_to_convert)} files to convert: {files_to_convert}")
    
    # Convert each file
    for filename in files_to_convert:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        # Create separate image directory for each file
        base_name = os.path.splitext(filename)[0]
        image_subdir = os.path.join(args.image_dir, base_name)
        
        print(f"\n{'='*60}")
        print(f"Converting: {filename}")
        print(f"{'='*60}")
        
        convert_jsonl_to_image(
            input_path=input_path,
            output_path=output_path,
            image_dir=image_subdir,
            use_base64=args.use_base64,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
