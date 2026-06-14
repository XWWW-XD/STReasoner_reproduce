#!/usr/bin/env python3
"""Plot Stage1 A800 ckpt-500 training curves and ST-Align eval comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO = Path(__file__).resolve().parents[4]
TRAINER_STATE = REPO / (
    "00_new_codes/repro_autodl/experiments/checkpoints/"
    "Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused/trainer_state.json"
)
OUT_DIR = REPO / (
    "00_new_codes/reports/t3-autodl2-三阶段训练复现/artifacts/stage1_a800_ckpt500"
)
METRICS_128 = REPO / "exp/qwen3_4b_stage1_ckpt500_alignment_128"
METRICS_FULL = REPO / "exp/qwen3_4b_stage1_ckpt500_alignment_full"


def load_log_history() -> list[dict]:
    with TRAINER_STATE.open(encoding="utf-8") as fh:
        state = json.load(fh)
    return state["log_history"]


def plot_training_curves(history: list[dict], out_dir: Path) -> None:
    steps = [row["step"] for row in history]
    loss = [row["loss"] for row in history]
    grad_norm = [row["grad_norm"] for row in history]
    lr = [row["learning_rate"] for row in history]
    ts_lr = [row["ts_encoder_learning_rate"] for row in history]
    epoch = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "Qwen3-4B Stage1 ST-Align (A800, ZeRO-3 opt offload, steps 1-500)",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(steps, loss, color="#2563eb", linewidth=1.2)
    ax.set_title("Training loss")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, grad_norm, color="#dc2626", linewidth=1.0)
    ax.set_title("Gradient norm")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Grad norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, lr, label="LLM LR", color="#059669", linewidth=1.2)
    ax.plot(steps, ts_lr, label="TS encoder LR", color="#d97706", linewidth=1.0, alpha=0.9)
    ax.set_title("Learning rate (cosine + warmup)")
    ax.set_xlabel("Global step")
    ax.set_ylabel("LR")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, epoch, color="#7c3aed", linewidth=1.2)
    ax.set_title("Epoch progress")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Epoch")
    ax.grid(True, alpha=0.3)

    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(steps, loss, color="#2563eb", linewidth=1.4)
    ax.set_title("Training loss (step 1-500)")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.axvline(500, color="#94a3b8", linestyle="--", linewidth=1, label="checkpoint-500")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "training_loss.png", dpi=160)
    plt.close(fig)


def load_metrics(path: Path, backup_name: str) -> tuple[dict, dict]:
    current = json.loads((path / "evaluation_metrics.json").read_text(encoding="utf-8"))
    backup = json.loads((path / backup_name).read_text(encoding="utf-8"))
    return backup, current


def plot_eval_comparison(out_dir: Path) -> None:
    r28_128, r29_128 = load_metrics(METRICS_128, "evaluation_metrics_report28.json")
    r28_full, r29_full = load_metrics(METRICS_FULL, "evaluation_metrics_report28.json")

    metrics = ["overall_score", "relative_accuracy", "exact_match"]
    labels = ["Overall", "Rel. acc.", "Exact match"]
    x = range(len(metrics))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle("ST-Align eval: report 28 (buggy) vs report 29 (fixed)", fontsize=12)

    for ax, title, old_m, new_m in (
        (axes[0], "Head-128 subset", r28_128, r29_128),
        (axes[1], "Full 40512", r28_full, r29_full),
    ):
        old_vals = [old_m.get(k) or 0.0 for k in metrics]
        new_vals = [new_m.get(k) or 0.0 for k in metrics]
        ax.bar([i - width / 2 for i in x], old_vals, width, label="Report 28", color="#94a3b8")
        ax.bar([i + width / 2 for i in x], new_vals, width, label="Report 29", color="#2563eb")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_dir / "eval_metrics_report28_vs_29.png", dpi=160)
    plt.close(fig)


def write_summary(history: list[dict], out_dir: Path) -> None:
    first, last = history[0], history[-1]
    summary = {
        "global_steps": len(history),
        "loss_start": first["loss"],
        "loss_end": last["loss"],
        "grad_norm_start": first["grad_norm"],
        "grad_norm_end": last["grad_norm"],
        "learning_rate_end": last["learning_rate"],
        "ts_encoder_learning_rate_end": last["ts_encoder_learning_rate"],
        "epoch_end": last["epoch"],
        "source": str(TRAINER_STATE.relative_to(REPO)),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_log_history()
    plot_training_curves(history, OUT_DIR)
    plot_eval_comparison(OUT_DIR)
    write_summary(history, OUT_DIR)
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
