#!/usr/bin/env python3
"""Safe pickle inspection via pickletools + prompt text from HEARTS source."""
import argparse
import json
import pickle
import pickletools
import re
from pathlib import Path
from textwrap import dedent

PROMPT_TEMPLATE = dedent(
    """\
    The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
    {
        "below": [float, percentage of time CGM < 70 mg/dL],
        "above": [float, percentage of time CGM > 180 mg/dL]
    }"""
)

MAX_PROMPT = 2000


def enrich_cgm_from_pickle(path: Path, excerpt: dict) -> dict:
    if excerpt.get("task") != "cgm_stat_calculation" or excerpt.get("error"):
        return excerpt
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        gl = data["cgm"]["Libre GL"]
        n = len(gl)
        below_n = int((gl < 70).sum())
        above_n = int((gl > 180).sum())
        in_range = int(((gl >= 70) & (gl <= 180)).sum())
        excerpt["ground_truth"] = dict(data["GT"])
        excerpt["glucose_stats"] = {
            "n_readings": n,
            "min_mg_dl": round(float(gl.min()), 2),
            "max_mg_dl": round(float(gl.max()), 2),
            "mean_mg_dl": round(float(gl.mean()), 2),
            "std_mg_dl": round(float(gl.std()), 2),
            "minutes_below_70": below_n,
            "minutes_above_180": above_n,
            "minutes_in_range_70_180": in_range,
            "pct_in_range": round(in_range / n * 100, 2),
        }
        excerpt["gt_verified_against_series"] = True
        excerpt["ground_truth_source_keys"] = ["GT.below", "GT.above"]
    except Exception as exc:
        excerpt["pickle_load_error"] = str(exc)
    return excerpt


def is_lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:20].startswith(b"version https://git-lfs")


def extract_row_count(ops: list) -> int | None:
    for op, arg, _ in ops:
        if op.name == "BININT2" and isinstance(arg, int) and 100 < arg < 100000:
            return arg
    return None


def extract_gt(ops: list) -> dict | None:
    gt = {}
    keys = list(ops)
    for i, (op, arg, _) in enumerate(keys):
        if arg == "GT":
            j = i + 1
            while j < len(keys) and len(gt) < 4:
                o, a, _ = keys[j]
                if a == "below":
                    _, val, _ = keys[j + 2]
                    if keys[j + 2][0].name == "BINFLOAT":
                        gt["below"] = val
                if a == "above":
                    _, val, _ = keys[j + 2]
                    if keys[j + 2][0].name == "BINFLOAT":
                        gt["above"] = val
                j += 1
            if "below" in gt and "above" in gt:
                return gt
    return None


def summarize_schema(path: Path) -> dict:
    if is_lfs_pointer(path):
        return {"error": "lfs_pointer", "path": str(path)}
    ops = list(pickletools.genops(path.read_bytes()))
    top_keys = []
    for op, arg, _ in ops:
        if op.name == "SHORT_BINUNICODE" and arg in ("cgm", "GT", "subject_id", "hr"):
            top_keys.append(arg)
    rows = extract_row_count(ops)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "detected_keys": sorted(set(top_keys)),
        "cgm_rows_hint": rows,
        "has_dataframe": any(a == "DataFrame" for _, a, _ in ops),
    }


def build_excerpt(path: Path, taxonomy: dict, index: int) -> dict:
    rel = path.name
    parts = path.parts
    ds = parts[-3] if len(parts) >= 3 else "unknown"
    task = parts[-2] if len(parts) >= 2 else "unknown"
    task_key = f"{ds}/{task}"
    task_meta = taxonomy.get("tasks", {}).get(task_key, {})
    if not task_meta and ds == "cgmacros":
        task_meta = taxonomy.get("cgmacros_catalog", {}).get(task, {})

    if is_lfs_pointer(path):
        return {"source_path": str(path), "error": "lfs_pointer_not_downloaded"}

    ops = list(pickletools.genops(path.read_bytes()))
    gt = extract_gt(ops)
    rows = extract_row_count(ops)

    prompt = PROMPT_TEMPLATE
    if task == "cgm_stat_calculation":
        prompt_source = "HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)"
    else:
        prompt_source = f"HEARTS code task {task} (see source)"

    truncated = len(prompt) > MAX_PROMPT
    if truncated:
        prompt = prompt[:MAX_PROMPT]

    return {
        "index": index,
        "source_path": "/".join(path.parts[-3:]),
        "dataset": ds,
        "task": task,
        "capability": task_meta.get("capability", "Perception"),
        "subtask": task_meta.get("subtask", ""),
        "metric": task_meta.get("metric", "sMAPE"),
        "prompt_or_query": prompt.strip(),
        "prompt_source": prompt_source,
        "prompt_truncated": truncated,
        "input_summary": (
            f"CGM time series as pandas DataFrame in pickle key 'cgm'; "
            f"columns include timestamp + Libre GL (mg/dL); ~{rows or '?'} rows per minute sampling."
        ),
        "ground_truth": gt,
        "ground_truth_source_keys": ["GT.below", "GT.above"] if gt else [],
        "notes": f"testcase file index {path.stem}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--excerpts-out", type=Path, required=True)
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    schemas = []
    excerpts = {
        "entries": [],
        "disclaimer": "GT and glucose_stats from pickle.load when available; pickletools fallback. Prompt from official task source.",
    }

    pkl_files = sorted(args.root.rglob("*.pkl"))
    for p in pkl_files:
        if p.parent.name and p.stem == "0":
            schemas.append(summarize_schema(p))

    for i, p in enumerate(pkl_files):
        excerpt = build_excerpt(p, taxonomy, i)
        excerpts["entries"].append(enrich_cgm_from_pickle(p, excerpt))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"samples": schemas}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.excerpts_out.write_text(json.dumps(excerpts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"schemas": len(schemas), "excerpts": len(excerpts["entries"])}, indent=2))


if __name__ == "__main__":
    main()
