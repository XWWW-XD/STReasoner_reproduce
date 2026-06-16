#!/usr/bin/env python3
"""Generate figures for HeaRTS survey report."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    ad = args.artifacts_dir
    inv = json.loads((ad / "dataset_inventory.json").read_text(encoding="utf-8"))
    tax = json.loads((ad / "task_taxonomy.json").read_text(encoding="utf-8"))

    paper = inv["paper_datasets"]
    names = [r["dataset"] for r in paper]
    paper_counts = [r["paper_test_samples"] for r in paper]
    hf_counts = []
    hf_by = inv["hf_frozen"]["by_dataset"]
    for r in paper:
        ds = r["dataset"]
        n = sum(len(v) for v in hf_by.get(ds, {}).values())
        hf_counts.append(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))
    w = 0.35
    ax.bar([i - w/2 for i in x], paper_counts, w, label="Paper Table 3 #test")
    ax.bar([i + w/2 for i in x], hf_counts, w, label="HF frozen .pkl (local)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("HeaRTS: Paper test samples vs HF local pickles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ad / "fig_samples_per_dataset.png", dpi=150)
    plt.close(fig)

    caps = Counter()
    for t in tax.get("tasks", {}).values():
        caps[t.get("capability", "Unknown")] += 1
    cap_order = ["Perception", "Inference", "Generation", "Deduction"]
    vals = [caps.get(c, 0) for c in cap_order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cap_order, vals, color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
    ax.set_title("CGMacros tasks in cloned HEARTS code (by capability)")
    ax.set_ylabel("Task count")
    fig.tight_layout()
    fig.savefig(ad / "fig_capability_breakdown.png", dpi=150)
    plt.close(fig)

    access = Counter(r["access"] for r in paper)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(list(access.values()), labels=list(access.keys()), autopct="%1.0f%%")
    ax.set_title("Paper datasets: data accessibility")
    fig.savefig(ad / "fig_accessibility.png", dpi=150)
    plt.close(fig)

  # HF publish status bar
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Paper total", "HF local pkl"], [inv["paper_total_test_samples"], inv["hf_frozen"]["total_pkl"]], color=["#888", "#4C78A8"])
    ax.set_title("Benchmark scale: paper vs HF release")
    fig.tight_layout()
    fig.savefig(ad / "fig_hf_vs_paper_total.png", dpi=150)
    plt.close(fig)

    print("figures written to", ad)


if __name__ == "__main__":
    main()
