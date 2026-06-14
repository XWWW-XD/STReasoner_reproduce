# 26-服务器2 A800 Qwen3-4B Stage1 全参训练排查与 batch=8 操作日志

> 后续更正：本报告记录的是中间排查过程，其中 batch=8 + ZeRO-3 offload-all 的判断已被后续 cgroup 证据推翻。最终可执行配置见 `27-服务器2-A800-Qwen3-4B-Stage1最终训练配置.md`：bf16、ZeRO-3、仅 optimizer CPU offload、parameter 不 offload、micro batch=2、gradient accumulation=32。不要按本报告早期的 batch=8/offload-all 结论启动正式训练。

日期：2026-06-12  
机器：AutoDL 服务器2，单卡 NVIDIA A800 80GB PCIe  
代码根目录：`/root/autodl-tmp/STReasoner_reproduce`  
模型：`base_model/Qwen3-4B-Instruct-2507`  
数据：`data/ST-Bench/ST-Align/alignment_train.jsonl`  
入口：`src/train.py`  
模板：`STReasoner-Align`  
数据集名：`alignment`

## 0. 后续更正后的阅读说明

Stage1 可以继续走官方源码入口的全参 SFT，但在单卡 A800 80G 上，**GPU-only ZeRO-3 不稳定**：无论 batch=2 还是 batch=1，都会在 optimizer/loss/overflow 检查附近顶到 76-78GB 后 OOM。  

下面这组配置是当时基于 GPU 显存读数得到的中间判断，后续已确认不可作为最终正式训练配置，因为 offload-all 会触发容器 cgroup CPU 内存 OOM kill：

```text
full finetuning
fp16
ZeRO-3
CPU param offload + CPU optimizer offload
per_device_train_batch_size=8
gradient_accumulation_steps=8
global batch=64
max_steps=1000
cutoff_len=10000
```

该 batch=8 试验当时训练中显存约 **42.9GB**，还剩约 **38.3GB**；step 时间约 **38-39s/step**。但这个读数没有覆盖 cgroup CPU 内存风险。后续 batch1、batch4、batch8 的 ZeRO-3 offload-all 均出现 `return code = -9`，并确认撞到约 120GiB cgroup 内存上限。因此最终训练不采用 batch=8/offload-all，最终配置见第 27 号报告。

## 1. 与官方 Stage1 的对照

官方 4B Stage1 脚本：

```text
scripts/qwen3-4b-instruct/train_stage1.sh
```

关键参数：

```text
deepspeed --num_gpus 8
--deepspeed ds_config/ds_config_3.json
--stage sft
--model_name_or_path ./base_model/Qwen3-4B-Instruct-2507
--dataset alignment
--template STReasoner-Align
--finetuning_type full
--per_device_train_batch_size 2
--gradient_accumulation_steps 32
--learning_rate 1e-5
--timeseries_sft_lr 1e-5
--max_steps 1000
--fp16
--cutoff_len 10000
```

官方脚本是 **fp16**，不是 bf16。官方 `ds_config/ds_config_3.json` 是 **ZeRO-3，无 CPU offload**。今天最终切到 `ds_config/ds_config_3_offload_all.json`，是因为单卡 GPU-only OOM；这改变的是显存/CPU 内存放置，不改变 full finetuning、数据、模板、学习率、步数、cutoff_len。

官方脚本的 `2 * 32` 在单进程视角 global batch 为 64；batch=8 时用 `8 * 8 = 64`，保持同一 global batch，只减少 gradient accumulation 次数。

注意：当前按官方脚本形状没有显式加 `--flash_attn fa2`，日志显示 `Using torch SDPA for faster training and inference.`。这是和 report 25 中“优先 fa2”的预案不同的地方；本轮为了贴官方脚本，没有把 attention 后端作为新变量继续混入。

## 2. 读取过的 00_new_codes 依据

本次重点读取：

- `00_new_codes/repro_autodl/experiments/scripts/README.md`
- `00_new_codes/repro_autodl/experiments/logs/*.log`
- `00_new_codes/reports/t3-autodl2-三阶段训练复现/16-服务器1旧策略-单卡A100stage1情况报告.md`
- `00_new_codes/reports/t3-autodl2-三阶段训练复现/17-服务器1旧策略-Stage 1 LoRA 分段训练经验（500→2000）.md`
- `00_new_codes/reports/t3-autodl2-三阶段训练复现/25-服务器2新策略-单卡A800-80G-4B全参训练方案.md`
- `00_new_codes/reports/t3-autodl2-三阶段训练复现/ST-Align详细分析/05-Stage1含义与建议.md`
- `00_new_codes/reports/t3-autodl2-三阶段训练复现/ST-Align详细分析/07-通读alignment_train读出来的发现.md`
- `00_new_codes/reports/t3-三阶段训练复现/15-Qwen3-4B三阶段Baseline实施计划.md`

历史 A100 LoRA 报告只作为经验：环境变量、PATH/ninja、W&B、cache 路径、Stage1 数据结构风险。它不作为本次 A800 全参训练默认路线。

## 3. 环境与门禁状态

已确认：

```text
GPU: NVIDIA A800 80GB PCIe
PyTorch: 2.6.0+cu124
Transformers: 4.52.4
DeepSpeed: 0.16.4
Python env: /root/autodl-tmp/conda/envs/str-py310
模型参数量: 4,438,523,920 trainable params
训练样本数: 194,212
```

当前资源状态：

```text
/                 30G total, 23G free
/root/autodl-tmp 100G total, 74G free
CPU memory       1.0Ti total, 913Gi available
GPU              当前空闲
```

注意事项：

- `WANDB_DISABLED=true` 必须设置；否则无人值守会因为 W&B API key 失败。
- `PATH` 必须包含 `/root/autodl-tmp/conda/envs/str-py310/bin`；否则 CPUAdam JIT 编译会找不到 `ninja`。
- 使用绝对路径 `/root/autodl-tmp/conda/envs/str-py310/bin/deepspeed` 比直接 `deepspeed` 更稳。
- `TORCH_EXTENSIONS_DIR` 要指向数据盘 cache；CPUAdam 已在该目录编译成功。

## 4. 今天的训练尝试日志汇总

日志目录：

```text
00_new_codes/repro_autodl/experiments/logs/
```

### 4.1 W&B 启动失败

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_184603.log
```

配置：

```text
ds_config/ds_config_3.json
fp16
batch=2
grad_acc=32
global batch=64
no CPU offload
```

失败点：

```text
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

结论：训练入口本身没坏；必须禁用 W&B。

### 4.2 GPU-only ZeRO-3 + fp16 + batch=2

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_184825_wandb_disabled.log
```

配置：

```text
ds_config/ds_config_3.json
fp16
batch=2
grad_acc=32
global batch=64
no CPU offload
```

跑到 step 11，前 11 步是 fp16 dynamic loss scale overflow skip，随后在 optimizer step OOM：

```text
torch._foreach_sqrt(device_exp_avg_sqs)
Tried to allocate 5.17 GiB
GPU total 79.32 GiB, free 3.15 GiB
process memory 76.16 GiB
```

结论：GPU-only batch=2 不够余量，AdamW multi-tensor 临时张量触发 OOM。

### 4.3 GPU-only ZeRO-3 + fp16 + batch=1

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_185459_batch1.log
```

配置：

```text
ds_config/ds_config_3.json
fp16
batch=1
grad_acc=32
global batch=32
no CPU offload
```

仍在 optimizer step OOM：

```text
torch._foreach_sqrt(device_exp_avg_sqs)
Tried to allocate 5.17 GiB
GPU total 79.32 GiB, free 3.15 GiB
process memory 76.16 GiB
```

结论：只降 micro batch 不能解决 optimizer-state 峰值；瓶颈不是单纯 activation。

### 4.4 GPU-only ZeRO-3 + fp16 + fused AdamW + batch=2

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_190259_fused_adamw_batch2.log
```

配置：

```text
ds_config/ds_config_3.json
fp16
batch=2
grad_acc=32
--optim adamw_torch_fused
no CPU offload
```

fused AdamW 解决了 `foreach_sqrt` 这个具体 OOM，step 12 出现真实更新：

```text
loss=16.2878
grad_norm=748.4283508518913
ts_encoder_learning_rate=5e-7
```

随后在 CausalLM loss 中 OOM：

```text
loss_utils.py: logits = logits.float()
Tried to allocate 7.34 GiB
GPU free 2.67 GiB
process memory 76.64 GiB
```

结论：GPU-only 即使用 fused optimizer，loss/logits float cast 仍无余量。

### 4.5 GPU-only ZeRO-3 + fp16 + fused AdamW + batch=1

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_191030_fused_adamw_batch1.log
```

配置：

```text
ds_config/ds_config_3.json
fp16
batch=1
grad_acc=32
--optim adamw_torch_fused
no CPU offload
```

step 11 出现真实更新：

```text
loss=16.4644
grad_norm=403.3739081018044
```

随后在 ZeRO-3 overflow check OOM：

```text
torch.isinf(self.grad_partitions_flat_buffer).any()
Tried to allocate 4.13 GiB
GPU free 1.46 GiB
process memory 77.85 GiB
```

结论：GPU-only batch=1 仍然太贴边，不适合正式长跑。

### 4.6 GPU-only ZeRO-3 + bf16 + fused AdamW + batch=1

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_192327_bf16_fused_batch1_ga64.log
```

配置：

```text
ds_config/ds_config_3.json
bf16
batch=1
grad_acc=64
--optim adamw_torch_fused
no CPU offload
```

失败点：

```text
state["exp_avg_sq"] = torch.zeros_like(...)
Tried to allocate 5.17 GiB
GPU free 2.59 GiB
process memory 76.72 GiB
```

结论：bf16 消除了 fp16 loss scale 问题，但不能消除 AdamW state 显存压力。

### 4.7 ZeRO-3 CPU offload + fp16 + batch=1

日志：

```text
qwen3_4b_stage1_align_single_a800_20260612_193522_fp16_offload_batch1_ga64.log
```

配置：

```text
ds_config/ds_config_3_offload_all.json
fp16
batch=1
grad_acc=64
global batch=64
CPU param offload
CPU optimizer offload
```

结果：

```text
step 12 出现真实更新
loss=16.2312
grad_norm=518.4801025390625
ts_encoder_learning_rate=5e-7
```

速度：

```text
约 121s/step
1000 steps 粗估约 33-34 小时
```

后续：

```text
return code = -9
```

该日志没有 Python traceback，也没有 CUDA OOM 文本；不能从日志确认是系统 kill、外部终止、还是其他运行时中断。由于 step 12 已过真实更新，说明 offload 路线能绕过 GPU OOM，但 batch=1 太慢。

### 4.8 batch=3 / batch=4 / batch=8 前台短验证

这几次是前台 `timeout` 验证，没有写入 `experiments/logs/*.log`；结果保存在终端输出与 `output/Qwen3-4B-Instruct-2507-stage1/trainer_log.jsonl` 的最后一次记录中。

batch=3：

```text
batch=3
grad_acc=21
global batch=63
显存采样约 12.9GB
step 时间约 55s
```

batch=4：

```text
batch=4
grad_acc=16
global batch=64
显存采样约 25.5GB
step 时间约 48s
```

batch=8：

```text
batch=8
grad_acc=8
global batch=64
显存采样约 42.9GB used / 38.3GB free
step 1: 39.53s
step 2: 38.93s
step 3: 39.05s
step 4: 38.01s
step 5: 38.67s
```

batch=8 的 `trainer_log.jsonl`：

```json
{"current_steps": 1, "total_steps": 1000, "loss": 16.5708, "elapsed_time": "0:00:39", "remaining_time": "10:58:08"}
{"current_steps": 2, "total_steps": 1000, "loss": 16.5045, "elapsed_time": "0:01:18", "remaining_time": "10:49:04"}
{"current_steps": 3, "total_steps": 1000, "loss": 16.598, "elapsed_time": "0:01:57", "remaining_time": "10:49:18"}
{"current_steps": 4, "total_steps": 1000, "loss": 16.5758, "elapsed_time": "0:02:33", "remaining_time": "10:37:40"}
{"current_steps": 5, "total_steps": 1000, "loss": 16.7774, "elapsed_time": "0:03:13", "remaining_time": "10:41:47"}
```

这 5 步仍是 fp16 dynamic loss scale 初期 overflow skip，`grad_norm=0.0` 属正常下调过程。按 batch=1/offload 的规律，约 step 12 才会进入首个真实 optimizer update。正式训练不应因前 10-11 个 overflow skip 手动中断。

## 5. batch=8 结论的后续更正

本报告早期根据 GPU 显存读数推荐过 `batch=8 + fp16 + ZeRO-3 offload-all`。这个结论后来被正式训练和 cgroup 证据推翻：batch1、batch4、batch8 在 offload-all 路线下都出现 `return code = -9`，并确认容器 CPU 内存上限约 120GiB，ZeRO-3 parameter+optimizer offload 会撞到 cgroup OOM kill。

因此不要再按本报告早期的 batch=8 操作命令启动训练。保留这些记录的价值，是说明当时“GPU 显存看起来够”并不等于系统整体可长跑；真正不稳定点在 CPU offload-all 与容器 cgroup 内存限制。

最终可对比产物不是 batch=8/offload-all，而是报告 27 记录的：

```text
bf16
ZeRO-3
optimizer-only CPU offload
parameter 不 offload
micro batch = 2
gradient accumulation = 32
cutoff_len = 10000
checkpoint-500
```

## 6. 本报告保留价值

本报告作为历史排查流水保留，主要用于回答三件事：

- 为什么官方 GPU-only ZeRO-3 在单卡 A800 上不够稳：optimizer/loss/overflow check 相关峰值会顶到 76-78GB 后 OOM。
- 为什么 batch=8/offload-all 最终作废：短跑显存读数正常，但正式训练被 cgroup CPU OOM kill。
- 为什么最终没有走 ZeRO-2：用户明确要求不要简单降 ZeRO 等级，后续定位到 cgroup 与 dtype mismatch 后，回到 ZeRO-3 optimizer-only offload。

如果以后改 LoRA，本报告不再作为训练入口，只作为 full SFT 排查参考。

## 7. 对 Stage1 效果的预期

ST-Align 数据结构本身有明显偏差：大量样本不需要读 TS，仅凭 graph text 或 metadata 可答；真正 temporal 数值反演又很难。历史 LoRA 探针显示 loss 可以下降，但 temporal balanced 面板长期不升。  

因此正式 Stage1 的验收不要只看 train loss。更合理的口径：

```text
1. 训练流程完整跑完；
2. checkpoint 能加载；
3. Stage2 能从 Stage1 输出继续；
4. 用固定小面板看格式、evolution/kappa、spatial 行为；
5. 不用 Stage1 spatial 成功宣称模型已经学会时序读取。
```

## 8. 不再建议做的事

- 不建议回到 GPU-only ZeRO-3 正式长跑；已多次在 76-78GB 附近 OOM。
- 不建议继续使用 ZeRO-3 offload-all；它会触发容器 cgroup CPU OOM kill。
- 不建议沿用 batch=8/offload-all 操作命令；该命令只保留为失败路径证据。
- 不建议继续补跑 full SFT 到 1000 step；当前 full SFT 已停在 checkpoint-500，后续主线若改 LoRA，应以该产物和评测作为对比基线。
- 不建议把本报告里的早期“推荐 batch=8”当成最终建议；最终口径以报告 27 为准。

## 9. 2026-06-12 晚间正式尝试补充日志

### 9.1 batch=8 offload fp16 正式训练

正式启动命令使用 `setsid` 后台运行，核心参数：

```text
ds_config/ds_config_3_offload_all.json
per_device_train_batch_size=8
gradient_accumulation_steps=8
fp16
cutoff_len=10000
max_steps=1000
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_203455_fp16_offload_batch8_ga8_train_setsid.log
```

观察结果：

- 配置确认进入训练：`train_micro_batch_size_per_gpu=8`，`train_batch_size=64`。
- 前 10 个 step 仍为 fp16 overflow skip。
- step 11 出现第一次真实更新：

```text
loss=16.6593
grad_norm=1106.6802978515625
ts_encoder_learning_rate=5e-7
```

- 随后 DeepSpeed launcher 报 `Killing subprocess ... exits with return code = -9`。
- 没有 Python traceback，也没有 CUDA OOM traceback。
- 训练退出后 GPU 已释放。

结论：batch=8 的显存前向/反向短跑可以进入训练，但正式训练在第一次真实 optimizer/update 附近被系统杀掉；这更像进程/CPU/offload/系统层面的 kill，而不是常规 CUDA OOM。

### 9.2 batch=4 offload fp16 正式训练

为排除 batch=8 过大，回退到：

```text
per_device_train_batch_size=4
gradient_accumulation_steps=16
fp16
global batch = 64
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_204515_fp16_offload_batch4_ga16_train_setsid.log
```

观察结果：

- 配置确认进入训练：`train_micro_batch_size_per_gpu=4`，`train_batch_size=64`。
- step 11 出现第一次真实更新：

```text
loss=16.6395
grad_norm=1228.1307373046875
ts_encoder_learning_rate=5e-7
```

- 随后同样被 launcher kill，返回码 `-9`。
- 过程中采样到 GPU 约 61.8GB used / 19.5GB free。
- CPU 内存日志显示 ZeRO/offload 初始化后曾到约 176.89GB virtual memory。

结论：把 micro batch 从 8 降到 4 没有解决 `-9`；问题不只是 batch=8 的 GPU 显存占用。

### 9.3 offload bf16 batch=4 尝试

按“改 bf”的要求，把 fp16 改为 bf16，其他保持 batch=4/offload：

```text
ds_config/ds_config_3_offload_all.json
per_device_train_batch_size=4
gradient_accumulation_steps=16
bf16
cutoff_len=10000
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_210356_bf16_offload_batch4_ga16_train_setsid.log
```

DeepSpeed 配置确认：

```text
bfloat16_enabled = True
fp16_enabled = False
train_micro_batch_size_per_gpu = 4
train_batch_size = 64
```

失败位置：

```text
modeling_qwen3_ts.py line 583: ts_features, patch_cnt = self.ts_encoder(timeseries)
modeling_qwen3_ts.py line 174: x = self.mlp(x_patches)
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

结论：这次不是显存问题。bf16 下模型权重已转成 BFloat16，但 time-series encoder 的 `x_patches` 仍是 Float32，进入 `self.mlp` 时触发 dtype mismatch。若继续走 bf16，需要在 Qwen3TS 的 time-series encoder 源码里做最小 dtype 对齐，例如在 `self.mlp(x_patches)` 前把 `x_patches` 转到 MLP 权重 dtype/device。这个改动不改变训练入口、数据、模板、loss 或 batch 语义，只是让自定义 TS encoder 遵守当前精度模式。

### 9.4 bf16 dtype 补丁与短跑结果

代码库已有同类实现可参考：

```text
src/EasyR1/verl/utils/chatts_vllm.py
```

其中在 TS MLP 前已有：

```python
target_dtype = self.mlp[0].weight.dtype
x_patches = x_patches.to(dtype=target_dtype)
```

因此同步补丁到训练加载的 Qwen3TS 代码：

```text
base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
/root/autodl-tmp/cache/huggingface/modules/transformers_modules/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
```

补丁后已通过：

```bash
python -m py_compile base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
python -m py_compile /root/autodl-tmp/cache/huggingface/modules/transformers_modules/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py
```

随后启动 bf16/offload/batch4/ga16 smoke：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_210951_bf16_offload_batch4_ga16_patch_smoke.log
```

结果：

- dtype mismatch 消失，训练进入 `***** Running training *****`。
- DeepSpeed 确认 `bfloat16_enabled=True`、`fp16_enabled=False`、ZeRO-3 CPU param/optimizer offload。
- 但约 1 分钟后仍出现 `Killing subprocess ... return code = -9`，没有 Python traceback，也没有 CUDA OOM traceback。
- 退出后 GPU 为空。

判断：bf16 兼容性问题已被最小补丁修掉；当前阻塞点回到 ZeRO-3 CPU offload 相关的系统级 kill。单卡 ZeRO-3 参数 offload 没有多卡分片收益，下一步更合理的实验是参数留 GPU、只把 optimizer offload 到 CPU，即 ZeRO-2 optimizer offload + bf16。

当时曾短暂新增实验配置：

```text
00_new_codes/repro_autodl/experiments/ds_config_zero2_optimizer_offload.json
```

该方向随后停止，配置文件已删除；不要按 ZeRO-2 路线继续。

### 9.5 对 `-9` 的根因复查与纠正

上面的 ZeRO-2 方向后来被停止。原因是这会引入过大的训练配置变量，不适合作为优先路线。真正应该先查 `-9` 的系统原因。

复查 cgroup 后发现：

```text
/sys/fs/cgroup/memory/memory.limit_in_bytes      = 128849018880
/sys/fs/cgroup/memory/memory.max_usage_in_bytes  = 128849076224
/sys/fs/cgroup/memory/memory.failcnt             = 15
/sys/fs/cgroup/memory/memory.oom_control         = oom_kill 4
```

也就是说当前容器内存上限约 120GiB，并不是 `free -h` 看到的 1TiB 宿主内存。此前 ZeRO-3 CPU offload-all 初始化时 DeepSpeed/psutil 报过 176-190GB CPU virtual memory，已经超过容器 cgroup 上限。因此 batch1、batch4、batch8 都在 offload-all 路线下出现 `return code = -9`，根因高度一致：容器 cgroup OOM kill。

更准确的修复方向不是降 ZeRO 等级，而是继续保留 ZeRO-3，只去掉参数 CPU offload：

```text
ZeRO-3
offload_optimizer = cpu
offload_param = none
bf16
```

这样参数留在 A800，CPU 只承担 optimizer offload，避免 offload-all 撞 120GiB cgroup 上限。

新增实验配置：

```text
00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json
```

短跑命令核心参数：

```text
--deepspeed 00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json
--bf16
--per_device_train_batch_size 2
--gradient_accumulation_steps 32
--max_steps 20
--cutoff_len 10000
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_211814_bf16_zero3opt_batch2_ga32_cgroup_smoke.log
```

截至观察到 step 6：

```text
step 1: loss=16.5176, grad_norm=596.5061
step 2: loss=16.4525, grad_norm=327.3342
step 3: loss=12.5718, grad_norm=1328.7196
step 4: loss=8.9475,  grad_norm=1377.3773
step 5: loss=7.3587,  grad_norm=1041.3975
step 6: loss=6.0209,  grad_norm=695.1506
```

资源读数：

```text
cgroup memory usage ~= 89.8GB / 120GB
cgroup oom_kill 仍为 4，未增加
GPU memory ~= 42.6GB used / 38.7GB free
```

当前判断：`offload_param + offload_optimizer` 是导致 `-9` 的主要不稳定点；`ZeRO-3 + optimizer-only CPU offload + bf16` 是更贴近原目标的单卡 A800 可行路线。

### 9.6 bf16 ZeRO-3 optimizer-only smoke 完成

上述 smoke 最终完整跑完 20/20，并成功保存 checkpoint 与最终模型：

```text
output/Qwen3-4B-Instruct-2507-stage1-bf16-zero3opt-smoke-211814
output/Qwen3-4B-Instruct-2507-stage1-bf16-zero3opt-smoke-211814/checkpoint-20
```

最终指标：

```text
train_runtime = 0:27:28.67
train_loss = 5.3613
train_steps_per_second = 0.012
```

资源状态：

```text
训练结束后 GPU 释放
cgroup oom_kill 仍为 4，未增加
```

说明 `ZeRO-3 + optimizer-only CPU offload + bf16 + batch2/ga32` 能越过此前 offload-all 的 cgroup OOM kill，并能正常保存模型。

### 9.7 正式 Stage1 启动与后续状态

正式训练启动时间：2026-06-12 21:49 UTC。

核心参数：

```text
deepspeed --num_gpus 1
--deepspeed 00_new_codes/repro_autodl/experiments/ds_config_zero3_optimizer_offload.json
--dataset alignment
--template STReasoner-Align
--finetuning_type full
--output_dir ./output/Qwen3-4B-Instruct-2507-stage1
--per_device_train_batch_size 2
--gradient_accumulation_steps 32
--learning_rate 1e-5
--timeseries_sft_lr 1e-5
--max_steps 1000
--bf16
--cutoff_len 10000
--save_steps 100
--save_total_limit 2
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log
```

`save_total_limit=2` 是为了避免当前 `/root/autodl-tmp` 只剩约 57GB 时被 10 个 checkpoint 撑爆磁盘；它只限制 checkpoint 保留数量，不改变训练样本、模型、loss、学习率或有效 batch。

启动后观察到 step 3：

```text
step 1: loss=16.5176, grad_norm=595.2406
step 2: loss=16.4525, grad_norm=327.4049
step 3: loss=16.4833, grad_norm=453.6838
```

资源读数：

```text
cgroup usage ~= 107.8GB / 120GB
cgroup oom_kill = 4，未增加
GPU memory ~= 36.0GB used / 45.2GB free
```

按当时约 80-85 秒/step 估算，1000 step 约 22-24 小时。

后续状态已更新在报告 27：正式 full SFT 最终按用户指令停在 `checkpoint-500`，不再继续补跑 1000 step；该 checkpoint 已做保护快照，并已在报告 28 中完成 ST-Align 全量测试。后续如果转 LoRA，本报告只作为 full SFT 排查历史，不作为新的训练操作手册。
