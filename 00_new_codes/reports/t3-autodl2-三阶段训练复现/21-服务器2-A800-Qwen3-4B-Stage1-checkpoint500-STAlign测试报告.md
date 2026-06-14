# 服务器2 A800 Qwen3-4B Stage1 checkpoint-500 ST-Align 测试报告

日期：2026-06-13

## 2. 测试集入口

测试集：

```text
data/ST-Bench/ST-Align/alignment_test.jsonl
```

## 3. 问题记录

### 3.1 问题1：当前 ST-Align 推理使用的 `inference/vllm/chatts_vllm.py` 没有同步bf16补丁

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

### 3.2 问题2：`evaluation/evaluate_qa.py` 数值题评分 bug

初版评分时 `overall_score=0.1953` 与 `exact_match=0.9259` 看起来不一致。原因是非零数值 target 只算了 `rel_error`，未计入 `rel_sum`/`overall_sum`：

```python
# 修复前（错误）
if abs(target_float) > 1e-6:
    rel_error = abs(pred_float - target_float) / abs(target_float)
else:
    rel_error = abs(pred_float - target_float)
    rel_score = max(0.0, 1.0 - rel_error)
    rel_sum += rel_score  # 仅 zero target 会累加
```

已修复为对所有数值题统一累加：

```python
# 修复后
if abs(target_float) > 1e-6:
    rel_error = abs(pred_float - target_float) / abs(target_float)
else:
    rel_error = abs(pred_float - target_float)
rel_score = max(0.0, 1.0 - rel_error)
rel_sum += rel_score
rel_total += 1
overall_sum += rel_score
```

下文评分结果均使用修复后的 `evaluation/evaluate_qa.py`，对同一批 `generated_answer.json` 重评得到。

## 4. 评分结果

评分命令：

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/str-py310/bin/python evaluation/evaluate.py \
  --task alignment \
  --dataset 00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl \
  --exp_path exp/qwen3_4b_stage1_ckpt500_alignment_128
```

评分日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_evaluate_report29_20260614_120007.log
```

指标文件：

```text
exp/qwen3_4b_stage1_ckpt500_alignment_128/evaluation_metrics.json
```

官方脚本输出（修复后）：

```json
{
  "task": "alignment",
  "total_samples": 128,
  "evaluated_samples": 128,
  "missing_predictions": 0,
  "coverage": 1.0,
  "overall_score": 0.7029,
  "exact_match": 0.9259,
  "relative_accuracy": 0.6433,
  "total_input_tokens": 32666,
  "avg_input_tokens": 255.2,
  "samples_with_token_info": 128
}
```

## 5. 全量测试操作与结果

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

官方全量评分输出（修复后）：

```json
{
  "task": "alignment",
  "total_samples": 40512,
  "evaluated_samples": 40512,
  "missing_predictions": 0,
  "coverage": 1.0,
  "overall_score": 0.8405,
  "exact_match": 0.9490,
  "relative_accuracy": 0.7031,
  "total_input_tokens": 16394615,
  "avg_input_tokens": 404.69,
  "samples_with_token_info": 40512
}
```

解释：

- `coverage=1.0`、`missing_predictions=0`：全量没有漏样本。
- `exact_match≈0.949`：非数值类目标的 exact match。
- `relative_accuracy≈0.703`：数值类目标的相对准确度。
- `overall_score≈0.841`：数值 + 非数值混合总分。

## 8. 长期保留与对比口径

本报告对应的模型是 Stage1 full SFT 的 `checkpoint-500`，后续不再继续补跑 full SFT；这些文件主要用于和后续 LoRA/Stage2 结果对比。

需要长期保留：

```text
inference/vllm/chatts_vllm.py
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused/
exp/qwen3_4b_stage1_ckpt500_alignment_128/
exp/qwen3_4b_stage1_ckpt500_alignment_full/
00_new_codes/repro_autodl/experiments/eval_subsets/alignment_test_head128.jsonl
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_inference_20260613_095619_patched.log
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_128_evaluate_report29_20260614_120007.log
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_inference_20260613_100107.log
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_ckpt500_alignment_full_evaluate_20260613_104412.log
exp/qwen3_4b_stage1_ckpt500_alignment_full/evaluation_metrics.json
00_new_codes/reports/t3-autodl2-三阶段训练复现/artifacts/stage1_a800_ckpt500/
```

使用口径：

- `checkpoint-500` 是模型权重断点，不是 optimizer/scheduler/RNG 完整续训断点。
- 报告 28 的全量 ST-Align 指标可作为后续 LoRA 或 Stage2 前后对比基线。
- `inference/vllm/chatts_vllm.py` 的 bf16 dtype 补丁需要保留，否则该 checkpoint 在 ST-Align vLLM 推理时会复现 dtype mismatch。
