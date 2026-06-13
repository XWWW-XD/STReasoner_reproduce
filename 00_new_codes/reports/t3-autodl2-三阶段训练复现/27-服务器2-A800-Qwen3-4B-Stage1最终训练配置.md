# 服务器2 A800 Qwen3-4B Stage1 ST-Align 最终训练配置

时间：2026-06-12  
任务：单张 A800 上运行 Qwen3-4B-Instruct-2507 的 Stage1 ST-Align 全参 SFT。

## 1. 最终结论

最终采用：

```text
Qwen3-4B-Instruct-2507
Stage1 ST-Align
full fine-tuning
bf16
ZeRO-3
optimizer CPU offload
parameter 不 offload
micro batch = 2
gradient accumulation = 32
global batch = 64
cutoff_len = 10000
max_steps = 1000
```

这套配置保留 ZeRO-3 和全参训练，没有改成 ZeRO-2，也没有使用 LoRA、QLoRA、量化或低配训练逻辑。

## 2. 为什么不是 offload-all

之前尝试过：

```text
ZeRO-3
offload_param = cpu
offload_optimizer = cpu
```

但 batch1、batch4、batch8 都出现 `return code = -9`。复查后确认根因不是 GPU OOM，而是容器 cgroup 内存限制。

关键证据：

```text
/sys/fs/cgroup/memory/memory.limit_in_bytes      = 128849018880
/sys/fs/cgroup/memory/memory.max_usage_in_bytes  = 128849076224
/sys/fs/cgroup/memory/memory.failcnt             = 15
/sys/fs/cgroup/memory/memory.oom_control         = oom_kill 4
```

当前容器实际内存上限约 120GiB。ZeRO-3 offload-all 初始化与训练时 CPU 内存会冲到 cgroup 上限，触发系统 OOM kill，因此日志里只看到 launcher 报 `return code = -9`，没有 Python traceback，也没有 CUDA OOM traceback。

因此最终不是降低 ZeRO 级别，而是保留 ZeRO-3，只去掉 parameter CPU offload：

```text
offload_optimizer = cpu
offload_param = none
```

参数留在 A800 显存，CPU 只承担 optimizer offload，避开 120GiB cgroup 内存上限。

## 3. DeepSpeed 配置

配置文件：

```text
00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json
```

内容：

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": false
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "steps_per_print": "inf"
}
```

说明：

- `stage=3`：保留 ZeRO-3。
- `offload_optimizer.device=cpu`：optimizer state 放 CPU。
- 不设置 `offload_param`：参数不 offload 到 CPU。
- `pin_memory=false`：减少 pinned CPU memory 压力。
- `bf16.enabled=auto`，训练命令使用 `--bf16`。

## 4. 正式训练命令

当前正式训练使用以下命令启动：

```bash
cd /root/autodl-tmp/STReasoner_reproduce

WANDB_DISABLED=true \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
HF_HOME=/root/autodl-tmp/cache \
TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/conda/envs/str-py310/bin/deepspeed \
  --num_gpus 1 \
  --master_port=19901 \
  src/train.py \
  --deepspeed 00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json \
  --stage sft \
  --model_name_or_path ./base_model/Qwen3-4B-Instruct-2507 \
  --dataset alignment \
  --interleave_probs 1 \
  --do_train \
  --mix_strategy interleave_over \
  --template STReasoner-Align \
  --finetuning_type full \
  --output_dir ./output/Qwen3-4B-Instruct-2507-stage1 \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --timeseries_sft_lr 1e-5 \
  --warmup_ratio 0.02 \
  --num_train_epochs 0 \
  --max_steps 1000 \
  --plot_loss \
  --bf16 \
  --save_only_model \
  --save_safetensors False \
  --preprocessing_num_workers 96 \
  --trust_remote_code True \
  --cutoff_len 10000
```

后台运行时使用 `setsid`，避免当前 shell 退出影响训练。

## 5. 训练参数口径

关键训练参数：

```text
model_name_or_path = ./base_model/Qwen3-4B-Instruct-2507
dataset = alignment
template = STReasoner-Align
stage = sft
finetuning_type = full
precision = bf16
cutoff_len = 10000
learning_rate = 1e-5
timeseries_sft_lr = 1e-5
lr_scheduler_type = cosine
warmup_ratio = 0.02
max_steps = 1000
per_device_train_batch_size = 2
gradient_accumulation_steps = 32
global batch = 64
save_steps = 100
save_total_limit = 2
save_only_model = true
```

`save_total_limit=2` 是存储稳定性设置。当前 `/root/autodl-tmp` 可用空间有限，而每个 4B full checkpoint 约 8GB 以上；如果每 100 step 保存且不限制保留数，1000 step 会保留 10 个 checkpoint，容易撑爆磁盘。该参数不改变训练样本、模型、loss、学习率或有效 batch。

## 6. bf16 必要源码补丁

bf16 初次尝试时失败：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

失败位置：

```text
modeling_qwen3_ts.py
x = self.mlp(x_patches)
```

原因：bf16 下 MLP 权重已是 BFloat16，但 TS encoder 构造出的 `x_patches` 仍是 Float32。

补丁位置：

```text
base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
/root/autodl-tmp/cache/huggingface/modules/transformers_modules/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
```

补丁内容：

```python
# Align input dtype with MLP weights to avoid matmul dtype mismatch.
target_dtype = self.mlp[0].weight.dtype
x_patches = x_patches.to(dtype=target_dtype)
x = self.mlp(x_patches)
```

这不是训练逻辑改动，只是让自定义 TS encoder 输入遵守当前模型精度。代码库已有同类写法：

```text
src/EasyR1/verl/utils/chatts_vllm.py
```

## 7. Smoke 验证

验证命令使用同一套最终配置，只把 `max_steps` 改成 20，输出到 smoke 目录。

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_211814_bf16_zero3opt_batch2_ga32_cgroup_smoke.log
```

结果：

```text
20/20 steps 完成
train_runtime = 0:27:28.67
train_loss = 5.3613
train_steps_per_second = 0.012
checkpoint-20 保存成功
最终模型保存成功
cgroup oom_kill 未增加
```

输出：

```text
output/Qwen3-4B-Instruct-2507-stage1-bf16-zero3opt-smoke-211814
output/Qwen3-4B-Instruct-2507-stage1-bf16-zero3opt-smoke-211814/checkpoint-20
```

## 8. 当前正式训练状态

正式训练进程：

```text
已按用户要求暂停，无 deepspeed / src/train.py 残留进程
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log
```

输出目录：

```text
output/Qwen3-4B-Instruct-2507-stage1
```

截至本次巡检（本地时间 2026-06-13 09:33:50 CST；UTC 2026-06-13 01:33:50），训练已按用户要求停在 checkpoint-500 之后，不再继续跑 checkpoint-600。

说明：仓库内官方 4B Stage1 脚本 `scripts/qwen3-4b-instruct/train_stage1.sh` 当前写的是 `--max_steps 1000`，不是 500。本次暂停是根据用户“暂停到500步”的新指令执行。因为发出中断时训练已从 checkpoint-500 继续推进，trainer log 最后一条写到 step 509；但可恢复/可接 Stage2 的断点按已保存的 checkpoint-500 口径使用。

```text
step 1: loss=16.5176, grad_norm=595.2406
step 2: loss=16.4525, grad_norm=327.4049
step 3: loss=16.4833, grad_norm=453.6838
step 4: loss=16.4049, grad_norm=883.2960
step 5: loss=16.2219, grad_norm=1815.4838
step 6: loss=14.9675, grad_norm=1014.6637
step 7: loss=14.6346, grad_norm=1246.8363
step 8: loss=11.9948, grad_norm=1307.9305
step 20: loss=3.3302, grad_norm=2521.9011
step 30: loss=0.9146, grad_norm=292.6710
step 50: loss=0.4961, grad_norm=48.2601
step 74: loss=0.4977, grad_norm=28.1493
step 100: loss=0.5204, grad_norm=26.3501
step 104: loss=0.4635, grad_norm=55.3290
step 115: loss=0.5272
step 116: loss=0.3866
step 168: loss=0.4344
step 180: loss=0.4189
step 190: loss=0.3123
step 200: loss=0.2398
step 206: loss=0.3513
step 207: loss=0.4936
step 208: loss=0.3253
step 209: loss=0.3871
step 232: loss=0.3554
step 235: loss=0.3350
step 239: loss=0.3869
step 289: loss=0.2683
step 300: loss=0.2500
step 303: loss=0.2556
step 386: loss=0.1226
step 390: loss=0.2539
step 400: loss=0.2575
step 485: loss=0.2522
step 500: loss=0.2063
step 504: loss=0.1777
step 509: loss=0.1268
```

checkpoint-500 已保存成功，日志显示配置、tokenizer、processor 与 2 个模型 shard 均已写入；`save_total_limit=2` 已按预期删除旧的 checkpoint-300：

```text
Saving model checkpoint to ./output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500
Configuration saved in .../checkpoint-500/config.json
The model is bigger than the maximum size per checkpoint (5GB) and is going to be split in 2 checkpoint shards.
Deleting older checkpoint [output/Qwen3-4B-Instruct-2507-stage1/checkpoint-300] due to args.save_total_limit
processor saved in .../checkpoint-500/processor_config.json
```

暂停后核验：

```text
checkpoint-500 核心文件存在且非空：
  config.json
  generation_config.json
  pytorch_model.bin.index.json
  pytorch_model-00001-of-00002.bin
  pytorch_model-00002-of-00002.bin
  tokenizer_config.json
  processor_config.json
checkpoint-500 大小约 8.3GB
当前保留 checkpoint-400 与 checkpoint-500
```

资源读数：

```text
GPU memory = 0MB used / 81221MB free
cgroup usage ~= 31.2GB / 128.85GB limit bytes
cgroup oom_kill = 4，未增加
disk /root/autodl-tmp ~= 40GB free
checkpoint-400 与 checkpoint-500 保存成功，checkpoint-300 已按 save_total_limit 删除，输出目录约 17GB
```

如果后续要从 500 继续训练，可从 `output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500` 恢复；如果进入 Stage2，则优先使用该 checkpoint-500 作为 Stage1 产物。

## 9. 后续监控命令

查看训练日志：

```bash
tail -f /root/autodl-tmp/STReasoner_reproduce/00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log
```

查看进程：

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E 'deepspeed|src/train.py' | grep -v grep
```

查看 GPU：

```bash
nvidia-smi
```

查看 cgroup 内存：

```bash
printf 'usage=%s max=%s failcnt=%s oom_kill=%s\n' \
  "$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)" \
  "$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes)" \
  "$(cat /sys/fs/cgroup/memory/memory.failcnt)" \
  "$(awk '$1==\"oom_kill\" {print $2}' /sys/fs/cgroup/memory/memory.oom_control)"
```

查看磁盘：

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/STReasoner_reproduce/output/Qwen3-4B-Instruct-2507-stage1
```

## 10. 最终口径

推荐正式配置：

```text
Qwen3-4B-Instruct-2507
Stage1 ST-Align
full SFT
bf16
ZeRO-3
optimizer-only CPU offload
no parameter offload
micro batch 2
gradient accumulation 32
global batch 64
cutoff_len 10000
max_steps 1000
save_steps 100
save_total_limit 2
```

不推荐再使用：

```text
ZeRO-3 offload_param + offload_optimizer
```

原因是它会撞当前容器 120GiB cgroup 内存上限，导致 `return code = -9`。

## 11. 关机前断点归档

用户要求暂停到 500 步后，已把 `checkpoint-500` 另存为 trainer 管理目录外的保护快照：

```text
源断点：
output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500

保护快照：
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
```

保存方式：

```text
cp -al
```

这是硬链接快照，不额外复制 8.3GB 模型权重数据；即使以后继续训练时 `save_total_limit=2` 删除 `output/.../checkpoint-500`，保护快照目录仍可保留 checkpoint-500 的文件内容。

保护快照内新增文件：

```text
FILES.txt
SHA256SUMS
SNAPSHOT_INFO.txt
```

`SNAPSHOT_INFO.txt` 记录：

```text
saved_at_local=2026-06-13_09:37:05_CST
saved_at_utc=2026-06-13_01:37:05_UTC
source=output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500
method=cp -al hardlink snapshot plus SHA256SUMS
trainer_last_logged_step=509
usable_checkpoint_step=500
official_script_max_steps=1000
```

已执行完整校验：

```bash
cd /root/autodl-tmp/STReasoner_reproduce/00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
sha256sum -c SHA256SUMS
```

结果：所有 checkpoint 文件均为 `OK`，包括：

```text
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
pytorch_model.bin.index.json
config.json
tokenizer_config.json
processor_config.json
trainer_state.json
training_args.bin
```

注意：本次训练命令使用了 `--save_only_model`。因此 checkpoint-500 是完整模型权重断点，适合接 Stage2 或作为继续微调的起点；但它不是包含 optimizer、scheduler、rng state 的完整 Trainer 状态，不能保证逐 bit 等价于不中断地从 step 500 继续跑到 step 1000。

## 12. 以后复跑如何操作

### 12.1 关机回来先做核验

进入仓库：

```bash
cd /root/autodl-tmp/STReasoner_reproduce
```

确认保护快照还在：

```bash
test -d 00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
```

校验 checkpoint 文件：

```bash
cd /root/autodl-tmp/STReasoner_reproduce/00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
sha256sum -c SHA256SUMS
```

确认 GPU 空闲：

```bash
nvidia-smi
```

### 12.2 作为 Stage2 输入

如果当前策略是“Stage1 到 checkpoint-500 就结束”，Stage2 的 `model_name_or_path` 应优先使用保护快照：

```text
./00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
```

不要依赖 `output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500` 一定存在，因为以后如果继续训练，`save_total_limit=2` 可能轮转删除 output 目录下旧 checkpoint。

### 12.3 从 checkpoint-500 继续 Stage1

已经写好脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh
```

语法检查已通过：

```bash
bash -n 00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh
```

默认行为：

```text
MODEL_CHECKPOINT=./00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
OUTPUT_DIR=./output/Qwen3-4B-Instruct-2507-stage1-continue-from500
MAX_STEPS=500
per_device_train_batch_size=2
gradient_accumulation_steps=32
bf16
ZeRO-3 optimizer-only CPU offload
```

运行方式：

```bash
cd /root/autodl-tmp/STReasoner_reproduce
bash 00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh
```

如果想指定新输出目录：

```bash
OUTPUT_DIR=./output/Qwen3-4B-Instruct-2507-stage1-continue-from500-rerun \
bash 00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh
```

重要限制：

```text
这不是 optimizer-state exact resume。
原因：正式训练保存时用了 --save_only_model，没有保存 optimizer.pt / scheduler.pt / rng_state。
该脚本会从 checkpoint-500 的模型权重继续训练 500 个新 step。
```

如果以后需要严格从某个 step 继续完整 Trainer 状态，下一轮训练不要使用 `--save_only_model`，并确认 checkpoint 中包含 optimizer/scheduler/rng 文件。

## 13. 本次实验日志索引

主要正式训练日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log
```

正式 trainer JSONL：

```text
output/Qwen3-4B-Instruct-2507-stage1/trainer_log.jsonl
```

成功 smoke 日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_211814_bf16_zero3opt_batch2_ga32_cgroup_smoke.log
```

关键失败/排查日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_184603.log
qwen3_4b_stage1_align_single_a800_20260612_184825_wandb_disabled.log
qwen3_4b_stage1_align_single_a800_20260612_185459_batch1.log
qwen3_4b_stage1_align_single_a800_20260612_190259_fused_adamw_batch2.log
qwen3_4b_stage1_align_single_a800_20260612_191030_fused_adamw_batch1.log
qwen3_4b_stage1_align_single_a800_20260612_193522_fp16_offload_batch1_ga64.log
qwen3_4b_stage1_align_single_a800_20260612_203455_fp16_offload_batch8_ga8_train_setsid.log
qwen3_4b_stage1_align_single_a800_20260612_204515_fp16_offload_batch4_ga16_train_setsid.log
qwen3_4b_stage1_align_single_a800_20260612_210356_bf16_offload_batch4_ga16_train_setsid.log
qwen3_4b_stage1_align_single_a800_20260612_210951_bf16_offload_batch4_ga16_patch_smoke.log
```

实验配置：

```text
00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json
```

复跑脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh
```

## 14. 正常情况记录

本次最终稳定配置：

```text
bf16
ZeRO-3
optimizer-only CPU offload
parameter 不 offload
micro batch 2
gradient accumulation 32
global batch 64
cutoff_len 10000
save_steps 100
save_total_limit 2
```

正常现象：

```text
20-step smoke 成功完成并保存。
正式训练成功保存 checkpoint-100 / 200 / 300 / 400 / 500。
save_total_limit=2 正常轮转旧 checkpoint。
checkpoint-500 保存后继续到了 step 509，随后按用户要求中断。
中断后无 deepspeed / src/train.py 残留进程。
GPU 显存释放为 0MB used。
cgroup oom_kill 没有继续增加，仍为 4。
保护快照 sha256sum -c 全部 OK。
```

checkpoint 轮转记录：

```text
checkpoint-300 保存后删除 checkpoint-100。
checkpoint-400 保存后删除 checkpoint-200。
checkpoint-500 保存后删除 checkpoint-300。
当前 output 目录保留 checkpoint-400 与 checkpoint-500。
保护快照额外保留 checkpoint-500。
```

## 15. 异常情况记录

### 15.1 W&B 未禁用导致启动失败

现象：

```text
wandb.errors.errors.UsageError: No API key configured.
```

处理：

```text
设置 WANDB_DISABLED=true。
```

### 15.2 GPU-only ZeRO-3 OOM

现象：

```text
fp16 + GPU-only ZeRO-3 + batch=2 在 optimizer step 附近 OOM。
batch=1 仍 OOM。
fused AdamW 避开 foreach_sqrt 后，后续又在 logits.float() loss 位置 OOM。
```

结论：

```text
单卡 A800 80G 上，Qwen3-4B full SFT + cutoff_len=10000 的 GPU-only ZeRO-3 余量不够。
```

### 15.3 offload-all 被 cgroup CPU 内存杀掉

现象：

```text
ZeRO-3 offload_param + offload_optimizer 多次 return code = -9。
没有 Python traceback，也不是 CUDA OOM traceback。
```

根因：

```text
容器 cgroup 内存上限约 120GiB。
offload-all 会把 CPU 内存推到 cgroup 上限，触发 OOM kill。
```

最终处理：

```text
保留 ZeRO-3。
只 offload optimizer。
parameter 不 offload。
pin_memory=false。
```

### 15.4 bf16 dtype mismatch

现象：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

补丁：

```python
target_dtype = self.mlp[0].weight.dtype
x_patches = x_patches.to(dtype=target_dtype)
x = self.mlp(x_patches)
```

补丁位置：

```text
base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
/root/autodl-tmp/cache/huggingface/modules/transformers_modules/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
```

### 15.5 ZeRO-2 没有作为最终路线

中间曾短暂验证过 ZeRO-2 方向，但用户明确要求不要“一位地换配置降方法”。最终没有采用 ZeRO-2，而是继续排查不稳定原因，定位到 cgroup 内存与 bf16 dtype 问题后，回到 ZeRO-3 optimizer-only offload。

## 16. 关服务器前清单

关机前已经保存/确认：

```text
checkpoint-500 原始目录存在：
output/Qwen3-4B-Instruct-2507-stage1/checkpoint-500

checkpoint-500 保护快照存在：
00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused

保护快照 SHA256SUMS 校验通过。

正式训练日志存在：
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log

最终 DeepSpeed 配置存在：
00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json

复跑脚本存在且 bash -n 通过：
00_new_codes/repro_autodl/experiments/scripts/resume_qwen3_4b_stage1_from_checkpoint500_model_continue.sh

本报告已更新：
00_new_codes/reports/t3-autodl2-三阶段训练复现/27-服务器2-A800-Qwen3-4B-Stage1最终训练配置.md
```

关机后回来，最先执行：

```bash
cd /root/autodl-tmp/STReasoner_reproduce/00_new_codes/repro_autodl/experiments/checkpoints/Qwen3-4B-Instruct-2507-stage1-checkpoint-500-paused
sha256sum -c SHA256SUMS
```

如果校验通过，再决定接 Stage2 或继续 Stage1。
