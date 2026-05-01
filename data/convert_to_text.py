#!/usr/bin/env python3
"""
Convert reasoning jsonl files by inlining time series into the `input` text.

For each JSONL line:
- Replace each occurrence of "<ts><ts/>" in `input` with the corresponding time series text
  rendered as "<ts>v1, v2, ..., vN</ts>".
- Keep ONLY `input` and `output` fields in the output JSONL.

Output files are written to: data/reasoning_text/ (by default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional


PLACEHOLDER = "<ts><ts/>"


def _format_number(x: Any, decimals: int) -> str:
    # Keep ints compact, floats rounded.
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if decimals < 0:
            return repr(x)
        return f"{x:.{decimals}f}"
    # Fall back (some datasets may store numbers as strings)
    return str(x)


def series_to_text(series: Iterable[Any], *, decimals: int, sep: str) -> str:
    return "<ts>" + sep.join(_format_number(v, decimals) for v in series) + "</ts>"


def inline_timeseries_into_input(
    input_text: str,
    timeseries: Optional[List[List[Any]]],
    *,
    decimals: int,
    sep: str,
) -> str:
    if not timeseries:
        return input_text

    placeholder_count = input_text.count(PLACEHOLDER)
    if placeholder_count == 0:
        return input_text

    # Replace sequentially to preserve the node ordering in the prompt.
    new_text = input_text
    n = min(placeholder_count, len(timeseries))
    for i in range(n):
        new_text = new_text.replace(
            PLACEHOLDER, series_to_text(timeseries[i], decimals=decimals, sep=sep), 1
        )
    return new_text


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    decimals: int,
    sep: str,
    verbose: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inlined = 0
    mismatched = 0
    total = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for obj in iter_jsonl(input_path):
            total += 1
            input_text = obj.get("input", "")
            ts = obj.get("timeseries")
            new_input = inline_timeseries_into_input(
                input_text, ts, decimals=decimals, sep=sep
            )

            if new_input != input_text:
                inlined += 1
                ph = input_text.count(PLACEHOLDER)
                ts_len = len(ts) if isinstance(ts, list) else 0
                if ph != ts_len:
                    mismatched += 1

            out_obj = {"input": new_input, "output": obj.get("output")}
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    if verbose:
        print(
            f"[convert] {input_path.name}: {total} lines, inlined={inlined}, mismatched={mismatched} -> {output_path}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path("data") / "reasoning"),
        help="Directory containing reasoning jsonl files (default: data/reasoning)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("data") / "reasoning_text"),
        help="Directory to write converted jsonl files (default: data/reasoning_text)",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=[],
        help="Optional list of specific .jsonl filenames to convert (within input_dir). If omitted, converts all .jsonl files in input_dir.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for floats (default: 2). Use -1 to keep Python repr.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=", ",
        help='Separator between values inside <ts>...</ts> (default: ", ").',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file stats to stderr.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input_dir not found or not a directory: {input_dir}", file=sys.stderr)
        return 2

    if args.files:
        input_files = [input_dir / f for f in args.files]
    else:
        input_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".jsonl")

    if not input_files:
        print(f"ERROR: no .jsonl files found in {input_dir}", file=sys.stderr)
        return 2

    for p in input_files:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2

    for input_path in input_files:
        output_path = output_dir / input_path.name
        convert_file(
            input_path,
            output_path,
            decimals=args.decimals,
            sep=args.sep,
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

