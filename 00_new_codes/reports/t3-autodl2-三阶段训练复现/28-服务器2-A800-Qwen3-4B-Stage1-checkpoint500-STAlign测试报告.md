# 服务器2 A800 Qwen3-4B Stage1 checkpoint-500 ST-Align 测试报告

日期：2026-06-13

## 1. 背景

用户提醒：Stage1 训练完成或暂停后，必须在 ST-Align 测试集上跑效果，否则无法判断训练产物是否可用。

本次测试对象不是最终 1000 step 模型，而是用户要求暂停后保存的 Stage1 `checkpoint-500`：

```text
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
```

训练状态说明：

- 原训练日志最后已经到 step 509 左右。
- 可复用、已归档、已校验的模型断点是 `checkpoint-500`。
- 因训练使用 `--save_only_model`，该断点是模型权重断点，不包含完整 optimizer/scheduler/RNG Trainer 状态。

## 2. 测试集与推理入口

测试集：

```text
data/ST-Bench/ST-Align/alignment_test.jsonl
```

总样本数：

```text
40512
```

推理入口：

```text
inference/inference_tsmllm_vllm.py
```

评测入口：

```text
evaluation/evaluate.py
```

注意：`evaluation/evaluate.py` 的 alignment 默认路径是旧的 `data/alignment/alignment_test.jsonl`，所以本次必须显式传入 `data/ST-Bench/ST-Align/alignment_test.jsonl` 或对应子集路径。

## 3. 128 条 sanity 推理

先按用户要求跑 128 条：

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
HF_HOME=/root/autodl-tmp/cache \
TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/conda/envs/str-py310/bin/python inference/inference_tsmllm_vllm.py \
  --task alignment \
  --dataset data/ST-Bench/ST-Align/alignment_test.jsonl \
  --model_path 00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_samples 128 \
  --exp qwen3_4b_stage1_ckpt500_alignment_128 \
  --output_name generated_answer.json
```

第一次失败日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_inference_20260613_0948.log
```

失败原因：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

定位结果：

- 训练侧 `base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py` 已有 bf16 dtype 对齐补丁。
- GRPO/EasyR1 的 `src/EasyR1/verl/utils/chatts_vllm.py` 也已有同类补丁。
- 但当前 ST-Align 推理使用的 `inference/vllm/chatts_vllm.py` 没有同步补丁。
- 报错发生在 vLLM engine profile 阶段，dummy time-series 输入为 Float32，而 TS encoder MLP 权重已按 checkpoint dtype 加载成 BFloat16。

已做最小补丁：

```python
# inference/vllm/chatts_vllm.py
x_patches = torch.cat(patches_list, dim=0)
# Align input dtype with MLP weights to avoid matmul dtype mismatch.
target_dtype = self.mlp[0].weight.dtype
x_patches = x_patches.to(dtype=target_dtype)
x = self.mlp(x_patches)
```

验证：

```bash
/root/autodl-tmp/conda/envs/str-py310/bin/python -m py_compile \
  inference/vllm/chatts_vllm.py \
  inference/inference_tsmllm_vllm.py \
  evaluation/evaluate.py
```

结果：语法检查通过。

补丁后成功推理日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_inference_20260613_095619_patched.log
```

输出文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_128/generated_answer.json
```

生成结果检查：

```text
样本数：128
idx 范围：0-127
空响应：0
最短响应字符数：1
最长响应字符数：14
平均响应字符数：5.0
前 128 条总输入 tokens：32666
平均输入 tokens：255.2
```

样例输出：

```text
idx=0 response='150'
idx=1 response='0.0174533'
idx=2 response='-4.712'
idx=3 response='sinusoidal'
idx=4 response='200'
```

结论：128 条推理输出形态正常，能进入 vLLM、能处理 TS 输入、能生成短答案，没有空输出或卡死。

## 4. 128 条官方评分

为了避免用全量 40512 条测试集评价 128 条预测导致 coverage 被稀释，先生成严格对应的 head-128 子集：

```bash
mkdir -p 00_new_codes/repro_autodl/experiments/eval_subsets
head -n 128 data/ST-Bench/ST-Align/alignment_test.jsonl \
  > 00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl
```

评分命令：

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/str-py310/bin/python evaluation/evaluate.py \
  --task alignment \
  --dataset 00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl \
  --exp_path exp/qwen3_4b_stage1_ckpt500_alignment_128
```

注意：直接执行 `python evaluation/evaluate.py` 会遇到：

```text
ModuleNotFoundError: No module named 'evaluation'
```

原因是脚本内部按包路径 `from evaluation.evaluate_qa import ...` 导入，需要在 repo root 下加 `PYTHONPATH=.`。

评分日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_evaluate_20260613_095940.log
```

指标文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_128/evaluation_metrics.json
```

官方脚本输出：

```json
{
  "task": "alignment",
  "total_samples": 128,
  "evaluated_samples": 128,
  "missing_predictions": 0,
  "coverage": 1.0,
  "overall_score": 0.1953125,
  "exact_match": 0.9259259259259259,
  "relative_accuracy": null,
  "total_input_tokens": 32666,
  "avg_input_tokens": 255.2,
  "samples_with_token_info": 128
}
```

## 5. 指标异常说明

`overall_score=0.1953125` 和 `exact_match=0.9259259259259259` 同时出现，看起来不一致。检查 `evaluation/evaluate_qa.py` 后发现 alignment 数值题评分逻辑存在问题：

```python
if target_float is not None and pred_float is not None:
    if abs(target_float) > 1e-6:
        rel_error = abs(pred_float - target_float) / abs(target_float)
    else:
        rel_error = abs(pred_float - target_float)
        rel_score = max(0.0, 1.0 - rel_error)
        rel_sum += rel_score
        rel_total += 1
        overall_sum += rel_score
```

对于非零数值答案，代码只计算 `rel_error`，但没有把相对分数加入 `rel_sum` 或 `overall_sum`。因此：

- `coverage=1.0` 和 `missing_predictions=0` 可以信。
- 推理是否能正常生成可以信。
- 但 `overall_score` / `relative_accuracy` 不能完整代表所有数值题效果。
- 如果后续要严肃比较 Stage1/Stage2/Stage3，需要修正或另写一个 ST-Align 数值评测脚本，并记录与官方 evaluator 的差异。

## 6. 当前判断

128 条 sanity 的主要目标是判断链路能不能跑通，而不是给最终论文级指标。

本次结果：

- vLLM 推理链路已打通。
- checkpoint-500 能被加载。
- time-series 输入能进入模型。
- 128/128 有输出。
- 官方 evaluator 能读取预测并评分。
- 唯一明确异常是官方 alignment 数值评分代码本身不完整，不是 checkpoint 生成失败。

因此可以继续跑全量 ST-Align 测试。

## 7. 全量测试操作与结果

全量推理命令：

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
HF_HOME=/root/autodl-tmp/cache \
TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/conda/envs/str-py310/bin/python inference/inference_tsmllm_vllm.py \
  --task alignment \
  --dataset data/ST-Bench/ST-Align/alignment_test.jsonl \
  --model_path 00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --exp qwen3_4b_stage1_ckpt500_alignment_full \
  --output_name generated_answer.json
```

全量评分命令：

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/str-py310/bin/python evaluation/evaluate.py \
  --task alignment \
  --dataset data/ST-Bench/ST-Align/alignment_test.jsonl \
  --exp_path exp/qwen3_4b_stage1_ckpt500_alignment_full
```

全量输出预期：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_full/generated_answer.json
exp/qwen3_4b_stage1_ckpt500_alignment_full/evaluation_metrics.json
```

实际全量推理日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_inference_20260613_100107.log
```

实际全量推理结果：

```text
开始时间：2026-06-13 10:01 左右
结束时间：2026-06-13 10:43:46
生成耗时：约 40 分钟
样本数：40512 / 40512
idx 范围：0-40511
空响应：0
输出文件大小：36M
总输入 tokens：16394615
平均输入 tokens：404.69
GPU 峰值观察：约 77980 / 81920 MiB，utilization 约 91%
```

输出文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_full/generated_answer.json
```

生成样例：

```text
first3:
idx=0 response='40'
idx=1 response='0.0174'
idx=2 response='-2.0944'

last3:
idx=40509 response='0.7'
idx=40510 response='7.5'
idx=40511 response='0.7'
```

完整性检查：

```text
n = 40512
idx_minmax = 0, 40511
empty = 0
min_len = 1
max_len = 14
avg_len = 3.61
```

实际全量评分日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_evaluate_20260613_104412.log
```

官方全量评分文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_full/evaluation_metrics.json
```

官方全量评分输出：

```json
{
  "task": "alignment",
  "total_samples": 40512,
  "evaluated_samples": 40512,
  "missing_predictions": 0,
  "coverage": 1.0,
  "overall_score": 0.5304854117288709,
  "exact_match": 0.9489557115732768,
  "relative_accuracy": 0.0,
  "total_input_tokens": 16394615,
  "avg_input_tokens": 404.69,
  "samples_with_token_info": 40512
}
```

解释：

- `coverage=1.0`、`missing_predictions=0` 是关键健康信号：全量没有漏样本。
- `exact_match=0.9489557` 对应非数值类目标的 exact match。
- `relative_accuracy=0.0` 和 `overall_score=0.5304854` 受第 5 节提到的官方数值评分代码缺陷影响，不能直接当作完整 ST-Align 数值能力指标。

## 8. 非官方诊断指标

为避免只看到官方 evaluator 的缺陷，本次额外计算了一个诊断文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_full/diagnostic_alignment_metrics_fixed_logic.json
```

这个诊断没有替代官方结果，只用于判断 checkpoint 是否明显坏掉，以及数值/非数值输出是否可解析。

诊断结果：

```json
{
  "total": 40512,
  "missing": 0,
  "numeric_targets": 17865,
  "numeric_pred_parseable": 17865,
  "numeric_parse_rate": 1.0,
  "numeric_exact_abs_1e-6": 0.4469633361321019,
  "numeric_exact_abs_1e-3": 0.46397984886649873,
  "numeric_mae": 8.694733430937674,
  "numeric_median_abs_error": 0.04999999999999999,
  "numeric_mean_relative_error_nonzero": 1.1668330577047046,
  "numeric_median_relative_error_nonzero": 0.07692307692307682,
  "nonnumeric_targets": 22647,
  "nonnumeric_exact": 0.9489557115732768,
  "diagnostic_overall_score_fixed_logic": 0.840536675448961
}
```

诊断解读：

- 17865 条数值目标的预测全部可解析为数字，说明模型没有出现大面积格式崩坏。
- 非数值目标 exact match 约 94.90%，与官方 `exact_match` 一致。
- 数值目标 `abs <= 1e-3` 的比例约 46.40%，中位绝对误差约 0.05。
- 按补齐后的相对分数逻辑，诊断 overall 约 0.8405；这不是官方指标，不能直接和论文表格混比。

## 9. 最终判断

Stage1 checkpoint-500 在 ST-Align 全量测试上的链路结果是健康的：

- checkpoint 能加载。
- vLLM TS 推理能跑完整 40512 条。
- 输出没有空响应。
- 官方 evaluator 能完成全量评分。
- 格式层面没有明显坏掉。

需要注意的异常：

- `inference/vllm/chatts_vllm.py` 需要同步 bf16 dtype 对齐补丁，否则 bf16 checkpoint 在 vLLM 初始化阶段会失败。
- `evaluation/evaluate_qa.py` 的 alignment 数值评分逻辑不完整，导致官方 `relative_accuracy`/`overall_score` 对数值题解释困难。
- 当前测试对象是 `checkpoint-500`，不是 1000 step 完整 Stage1 终点；因为用户要求暂停到 500，这次报告只评价 checkpoint-500。

## 10. 关服务器前必须保留

至少保留：

```text
inference/vllm/chatts_vllm.py
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused/
exp/qwen3_4b_stage1_ckpt500_alignment_128/
00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_inference_20260613_095619_patched.log
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_evaluate_20260613_095940.log
```

全量跑完后还要保留：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_full/
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_inference_20260613_100107.log
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_evaluate_20260613_104412.log
exp/qwen3_4b_stage1_ckpt500_alignment_full/evaluation_metrics.json
exp/qwen3_4b_stage1_ckpt500_alignment_full/diagnostic_alignment_metrics_fixed_logic.json
```
