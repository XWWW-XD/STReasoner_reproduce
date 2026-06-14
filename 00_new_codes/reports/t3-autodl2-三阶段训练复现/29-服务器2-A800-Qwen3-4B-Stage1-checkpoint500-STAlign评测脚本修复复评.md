# 服务器2 A800 Qwen3-4B Stage1 checkpoint-500 ST-Align 评测脚本修复说明

日期：2026-06-14

## 1. 修复内容

`evaluation/evaluate_qa.py` 中 alignment 数值题评分：非零 target 原先未把 `rel_score` 计入 `overall_sum`，导致 `overall_score`/`relative_accuracy` 被低估。修复后对同一批 `generated_answer.json` 重评，**结果已写回报告 28**，不再单独保留修复前指标。

## 2. 复评命令

```bash
# head-128
PYTHONPATH=. python evaluation/evaluate.py \
  --task alignment \
  --dataset 00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl \
  --exp_path exp/qwen3_4b_stage1_ckpt500_alignment_128

# 全量 40512：用 evaluate.py 时需流式读 gold（完整 jsonl 会 OOM），或直接读取已写入的 evaluation_metrics.json
```

## 3. 最终指标（与报告 28 一致）

| 子集 | overall_score | relative_accuracy | exact_match |
|---|---:|---:|---:|
| head-128 | 0.7029 | 0.6433 | 0.9259 |
| 全量 40512 | 0.8405 | 0.7031 | 0.9490 |

产物：

```text
evaluation/evaluate_qa.py
exp/qwen3_4b_stage1_ckpt500_alignment_128/evaluation_metrics.json
exp/qwen3_4b_stage1_ckpt500_alignment_full/evaluation_metrics.json
```

训练曲线见报告 28 第 6 节；复现脚本：

```bash
/root/autodl-tmp/conda/envs/str-py310/bin/python \
  00_new_codes/repro_autodl/experiments/scripts/plot_stage1_a800_ckpt500_metrics.py
```
