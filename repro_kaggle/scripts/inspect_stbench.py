#!/usr/bin/env python3
"""Inspect loadability and sample structure for Time-HD-Anonymous/ST-Bench."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any


DATASET_NAME = "Time-HD-Anonymous/ST-Bench"
SUBSETS = ("ST-Test", "ST-Test-Text", "ST-SFT", "ST-CoT", "ST-Align")
OUTPUT_LOG = "repro_kaggle/outputs/stbench_inspect.log"


class TeeLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._fh.write(message + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def preview_text(value: Any, limit: int = 300) -> str:
    if value is None:
        return "<missing>"
    text = str(value).replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def compact_type(value: Any) -> str:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:8]})"
    return type(value).__name__


def first_existing(sample: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in sample:
            return sample[name]
    return None


def preview_timeseries(timeseries: Any) -> str:
    if timeseries is None:
        return "<missing>"

    if isinstance(timeseries, dict):
        items = list(timeseries.items())
        lines = [f"dict nodes: {len(items)}"]
        for node, values in items[:5]:
            lines.append(f"  node {node!r}: {preview_values(values)}")
        return "\n".join(lines)

    if isinstance(timeseries, list):
        lines = [f"list nodes: {len(timeseries)}"]
        for index, values in enumerate(timeseries[:5]):
            lines.append(f"  node[{index}]: {preview_values(values)}")
        return "\n".join(lines)

    return f"{type(timeseries).__name__}: {preview_text(timeseries, 300)}"


def preview_values(values: Any) -> str:
    if isinstance(values, list):
        return preview_text(values[:8], 200)
    if isinstance(values, tuple):
        return preview_text(values[:8], 200)
    if isinstance(values, dict):
        return f"dict(keys={list(values.keys())[:8]})"
    return preview_text(values, 200)


def inspect_subset(subset: str, logger: TeeLogger) -> bool:
    from datasets import load_dataset

    logger.log("=" * 80)
    logger.log(f"subset: {subset}")

    try:
        dataset = load_dataset(DATASET_NAME, subset, split="train")
    except Exception as exc:  # noqa: BLE001 - keep inspecting remaining subsets.
        logger.log(f"LOAD_FAILED: {subset}")
        logger.log(f"{exc.__class__.__name__}: {exc}")
        logger.log(traceback.format_exc())
        return False

    logger.log(f"LOAD_OK: {subset}")
    logger.log(f"num_rows: {dataset.num_rows}")
    logger.log(f"column_names: {dataset.column_names}")

    if dataset.num_rows == 0:
        logger.log("dataset is empty; no sample preview available")
        return True

    sample = dataset[0]
    logger.log("first sample field types:")
    for key, value in sample.items():
        logger.log(f"  {key}: {compact_type(value)}")

    input_value = first_existing(sample, ("input", "instruction", "prompt", "question"))
    output_value = first_existing(sample, ("output", "answer", "response", "label"))
    category_value = first_existing(sample, ("category", "task", "type"))
    timeseries_value = first_existing(sample, ("timeseries", "time_series", "ts", "series"))

    logger.log(f"input preview: {preview_text(input_value)}")
    logger.log(f"output preview: {preview_text(output_value)}")
    logger.log(f"category: {preview_text(category_value, 120)}")
    logger.log("timeseries preview:")
    logger.log(preview_timeseries(timeseries_value))
    return True


def main() -> int:
    logger = TeeLogger(OUTPUT_LOG)
    successes: list[str] = []
    failures: list[str] = []

    try:
        logger.log(f"dataset: {DATASET_NAME}")
        logger.log(f"log: {OUTPUT_LOG}")
        for subset in SUBSETS:
            if inspect_subset(subset, logger):
                successes.append(subset)
            else:
                failures.append(subset)

        logger.log("=" * 80)
        logger.log("summary:")
        logger.log(f"direct_load_ok: {successes}")
        logger.log(f"direct_load_failed: {failures}")
        return 0 if successes else 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
