#!/usr/bin/env python3
"""Task 2: inventory HF pickles vs paper Table 3; build task taxonomy."""
import argparse
import json
from pathlib import Path

PAPER_DATASETS = [
    {"dataset": "capture24", "domain": "Motion", "modality": "IMU (100Hz)", "paper_test_samples": 1204, "access": "Open Source"},
    {"dataset": "pamap2", "domain": "Motion", "modality": "IMU (100Hz)", "paper_test_samples": 106, "access": "Open Source"},
    {"dataset": "shanghai_diabetes", "domain": "Metabolic", "modality": "CGM (per minute)", "paper_test_samples": 232, "access": "Open Source"},
    {"dataset": "cgmacros", "domain": "Metabolic", "modality": "CGM, HR, Annotation", "paper_test_samples": 2333, "access": "Open Source"},
    {"dataset": "vitaldb", "domain": "Surgery", "modality": "MBP, EEG, ECG, PPG", "paper_test_samples": 2000, "access": "Open Source"},
    {"dataset": "shhs", "domain": "Sleep", "modality": "ECG, EEG, EOG, etc.", "paper_test_samples": 4799, "access": "Restricted"},
    {"dataset": "harespod", "domain": "Respiration", "modality": "Respiration, SpO2, HR", "paper_test_samples": 632, "access": "Open Source"},
    {"dataset": "phymer", "domain": "Emotion", "modality": "BVP, EDA, TEMP, HR", "paper_test_samples": 1830, "access": "Restricted"},
    {"dataset": "perg_ioba", "domain": "Ophthalmology", "modality": "PERG (1700Hz)", "paper_test_samples": 870, "access": "Open Source"},
    {"dataset": "gazebase", "domain": "Eye Movement", "modality": "Eye Tracking (1000Hz)", "paper_test_samples": 988, "access": "Open Source"},
    {"dataset": "globem", "domain": "Behavior", "modality": "Aggregated (per day)", "paper_test_samples": 1140, "access": "Restricted"},
    {"dataset": "bridge2ai_voice", "domain": "Speech", "modality": "Audio (100Hz)", "paper_test_samples": 1600, "access": "Restricted"},
    {"dataset": "vctk", "domain": "Speech", "modality": "Audio (16kHz)", "paper_test_samples": 200, "access": "Open Source"},
    {"dataset": "grabmyo", "domain": "Gesture", "modality": "EMG (2048Hz)", "paper_test_samples": 400, "access": "Open Source"},
    {"dataset": "coughvid", "domain": "COVID Cough", "modality": "Audio (48kHz)", "paper_test_samples": 892, "access": "Open Source"},
    {"dataset": "coswara", "domain": "COVID Cough", "modality": "Audio (48kHz)", "paper_test_samples": 1000, "access": "Open Source"},
]

CGMACROS_TASKS = {
    "cgm_stat_calculation": {"capability": "Perception", "subtask": "Stat. Calculation", "metric": "sMAPE (1-0.5*sMAPE)"},
    "iauc_calculation": {"capability": "Perception", "subtask": "Stat. Calculation", "metric": "Accuracy"},
    "a1c_classification": {"capability": "Inference", "subtask": "Physiological Classification", "metric": "Accuracy"},
    "meal_img_classification": {"capability": "Inference", "subtask": "Physiological Classification", "metric": "Accuracy"},
    "meal_time_localization": {"capability": "Inference", "subtask": "Event Localization", "metric": "IoU"},
    "meal_react_comparison": {"capability": "Inference", "subtask": "Subject Profiling", "metric": "Accuracy"},
    "fasting_glu_prediction": {"capability": "Inference", "subtask": "Subject Profiling", "metric": "Accuracy"},
    "meal_forecasting": {"capability": "Generation", "subtask": "Future Forecasting", "metric": "sMAPE"},
    "meal_forecasting_meal_info": {"capability": "Generation", "subtask": "Future Forecasting", "metric": "sMAPE"},
    "meal_forecasting_no_ref": {"capability": "Generation", "subtask": "Future Forecasting", "metric": "sMAPE"},
    "meal_forecasting_no_ref_meal_info": {"capability": "Generation", "subtask": "Future Forecasting", "metric": "sMAPE"},
    "non_meal_imputation_cgm_only": {"capability": "Generation", "subtask": "Signal Imputation", "metric": "sMAPE"},
    "non_meal_imputation_hr": {"capability": "Generation", "subtask": "Signal Imputation", "metric": "sMAPE"},
    "non_meal_imputation_calories": {"capability": "Generation", "subtask": "Signal Imputation", "metric": "sMAPE"},
}


def scan_pkl(root: Path) -> dict:
    out = {"total_pkl": 0, "total_bytes": 0, "by_dataset": {}}
    for p in sorted(root.rglob("*.pkl")):
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        ds, task = parts[0], parts[1]
        out["total_pkl"] += 1
        out["total_bytes"] += p.stat().st_size
        out["by_dataset"].setdefault(ds, {}).setdefault(task, []).append(
            {"file": str(rel), "bytes": p.stat().st_size}
        )
    return out


def scan_code_tasks(code_root: Path) -> dict:
    tasks = {}
    exp_dir = code_root / "exp"
    if not exp_dir.exists():
        return tasks
    for ds_dir in sorted(exp_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in ("base", "utils", "templates"):
            continue
        for py in ds_dir.glob("*.py"):
            if py.name == "base.py":
                continue
            task_name = py.stem
            info = CGMACROS_TASKS.get(task_name, {})
            tasks[f"{ds_dir.name}/{task_name}"] = {
                "dataset": ds_dir.name,
                "task": task_name,
                "source_file": str(py.relative_to(code_root)),
                **info,
            }
    return tasks


def write_table_datasets(out_dir: Path) -> None:
    lines = [
        "# HeaRTS 论文 Table 3 数据集摘要",
        "",
        "| Dataset | Domain | Modality | Paper #Test | Access |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in PAPER_DATASETS:
        lines.append(
            f"| {row['dataset']} | {row['domain']} | {row['modality']} | {row['paper_test_samples']} | {row['access']} |"
        )
    lines.append("")
    lines.append(f"论文合计 test samples: {sum(r['paper_test_samples'] for r in PAPER_DATASETS)}")
    (out_dir / "table_datasets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leaderboard(out_dir: Path) -> None:
    text = """# HeaRTS 论文 Table 2 摘要（Overall Score，非本地复现）

| Model | Perception | Inference | Generation | Deduction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Baseline | — | — | — | — | 0.61 |
| GPT 4.1 Mini | 0.89/0.77 | — | — | — | 0.63 |
| Claude 4.5 Haiku | 0.93/0.79 | — | — | — | 0.63 |
| Grok 4.1 Fast | 0.93/0.78 | — | — | — | 0.65 |
| Kimi K2 Thinking | 0.91/0.80 | — | — | — | 0.65 |
| GLM 4.7 | 0.92/0.80 | — | — | — | 0.66 |

来源：arXiv:2603.06638 Table 2（Perception 两列为 Stat.Calc / Feat.Ext. 宏类子分数示例）。

完整 110 任务 × 14 模型分数见论文附录 Table 6。
"""
    (out_dir / "table_leaderboard.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    hf_scan = scan_pkl(args.root)
    inventory = {
        "hf_frozen": hf_scan,
        "paper_total_test_samples": sum(r["paper_test_samples"] for r in PAPER_DATASETS),
        "paper_datasets": PAPER_DATASETS,
        "hf_vs_paper_note": "HF 当前仅发布部分 frozen pickle；论文 20,226 为全 benchmark。",
    }
    (args.out / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    code_tasks = scan_code_tasks(args.code_root) if args.code_root else {}
    taxonomy = {"tasks": code_tasks, "cgmacros_catalog": CGMACROS_TASKS}
    (args.out / "task_taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_table_datasets(args.out)
    write_leaderboard(args.out)
    print(json.dumps({"pkl_count": hf_scan["total_pkl"], "code_tasks": len(code_tasks)}, indent=2))


if __name__ == "__main__":
    main()
