# Stage 1 当前输出索引与继续训练清单

## 当前结论

当前 Stage 1 单卡 4B 复现实验还没有完成到“可以声称 Stage 1 训练充分”的程度，但已经建立了可继续推进的稳定链路：

- full fine-tuning 在当前容器内存限制下不可行，主要卡在 CPU/cgroup 内存，而不是 GPU 显存。
- LoRA Stage 1 能训练、能保存、能标准 PEFT 加载、能做小探针。
- 最新可靠断点是 `2000 save_state`。
- 后续真正续训应从 `2000 save_state/checkpoint-500` 出发，输出到新的 `2500steps_from2000_save_state` 目录。
- 当前不要再把 `500 -> 1000 -> 1500` 称为严格连续训练；它们是“加载 adapter 后分段再训”。

## 输出目录索引

| 输出目录 | 训练含义 | train_loss | runtime | 大小 | 可标准加载 | 可严格 resume | 备注 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `output/single_a100_qwen3_4b_stage1_lora_smoke/` | LoRA smoke，10 steps | 16.3937 | 52.1041s | 159M | 是 | 否 | 只证明链路能跑，不看效果 |
| `output/single_a100_qwen3_4b_stage1_lora_100steps/` | LoRA 100 steps | 12.2186 | 482.5574s | 159M | 是 | 否 | 早期输出质量仍差 |
| `output/single_a100_qwen3_4b_stage1_lora_500steps/` | LoRA 500 steps | 3.8558 | 2475.1534s | 159M | 是 | 否 | 第一段较稳定 adapter |
| `output/single_a100_qwen3_4b_stage1_lora_1000steps_from500/` | 从 500 adapter 再训 500 steps | 0.8911 | 2434.5553s | 159M | 是 | 否 | adapter 分段接续，不是 Trainer resume |
| `output/single_a100_qwen3_4b_stage1_lora_1500steps_from1000/` | 从 1000 adapter 再训 500 steps | 0.5176 | 2391.6142s | 159M | 是 | 否 | adapter 分段接续，不是 Trainer resume |
| `output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/` | 从 1500 adapter 再训 500 steps，并保存 Trainer state | 0.4134 | 2509.1812s | 287M | 是 | 是，从 `checkpoint-500` 开始 | 当前最重要断点 |

解释：

- `可标准加载=是` 指 `PeftModel.from_pretrained(...)` 能加载，且 `nonzero_lora_B=257/257`。
- `可严格 resume=是` 指 checkpoint 中有 `optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`，可以用 `resume_from_checkpoint`。
- 前面几个 159M 目录主要只保存 adapter，没有 optimizer/scheduler/rng 状态，所以不能作为严格续训断点。
- `2000 save_state` 目录变成 287M 是合理的，因为多保存了 optimizer state，不是磁盘异常。

## 探针结果索引

这些探针不是官方 ST-Bench evaluate，只是 Stage 1 训练中间健康检查。作用是快速观察模型是否能沿 TS 生成链路输出短 answer，以及不同类别是否有明显变化。

| adapter | 6条小探针 | 30条健康探针 | 40条 temporal balanced | 推理显存峰值 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 500 steps | 2/6 | 未跑 | 未跑 | 8.50G allocated / 8.60G reserved | 初步能答，但很弱 |
| 1000 from 500 | 4/6 | 14/30 | 未跑 | 8.50G / 8.60G | spatial 类开始稳定 |
| 1500 from 1000 | 未单独记录 | 16/30 | 6/40 | 8.50G / 8.60G；temporal 8.67G / 8.77G | 总体小幅进步，temporal 数值仍弱 |
| 2000 save_state | 未单独记录 | 17/30 | 6/40 | 8.50G / 8.60G；temporal 8.67G / 8.77G | loss 继续降，但 temporal 数值未突破 |

2000 save_state 的 40 条 temporal balanced 分项：

```text
amplitude: 0/5
frequency: 0/5
phase: 0/5
evolution pattern: 4/5
long-term baseline: 0/5
kappa: 2/5
sigma: 0/5
lambda: 0/5
```

这个分项比总分更重要：模型对离散模式类问题已经有一点能力，但对连续数值参数仍常输出固定偏置或常见错误值。

## 关键日志索引

训练日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_loadable_fa2_20260604_174208.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_loadable_fa2_20260604_174411.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_loadable_fa2_20260604_180409.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_500to1000_fa2_20260604_185655.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_1000to1500_fa2_20260604_194332.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_1500to2000_save_state_fa2_20260604_203607.log
```

加载检查日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_adapter_load_check_20260604_184604.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1000steps_from500_adapter_load_check_20260604_193809.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1500steps_from1000_adapter_load_check_20260604_202407.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_2000steps_from1500_save_state_adapter_load_check_20260604_231307.log
```

探针结果文件：

```text
00_new_codes/repro_autodl/experiments/results/stage1_lora_1000steps_from500_probe30_20260604_194116.jsonl
00_new_codes/repro_autodl/experiments/results/stage1_lora_1500steps_from1000_probe30_20260604_202435.jsonl
00_new_codes/repro_autodl/experiments/results/stage1_lora_1500steps_from1000_temporal_balanced_probe40_20260604_202856.jsonl
00_new_codes/repro_autodl/experiments/results/stage1_lora_2000steps_from1500_save_state_probe30_20260604_231340.jsonl
00_new_codes/repro_autodl/experiments/results/stage1_lora_2000steps_from1500_save_state_temporal_balanced_probe40_20260604_231446.jsonl
```

## 下次继续训练前检查清单

不要直接开训，先确认：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/adapter_model.bin
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/optimizer.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/scheduler.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/rng_state.pth
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/trainer_state.json
```

还要确认：

```text
base_model/Qwen3-4B-Instruct-2507/ 存在
base_model/Qwen3-4B-Instruct-2507/ 里有 STReasoner TS config/code
/root/autodl-tmp/cache/huggingface 可用
HF_HUB_OFFLINE=1
GPU 空闲
/root/autodl-tmp 至少留出几个 GB 空间
```

## 下次继续命令

脚本已经修正为读取 2000 断点，输出到新 2500 目录：

```bash
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh
```

预期输出目录：

```text
output/single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state/
```

重要：不要把输出写回 `2000steps_from1500_save_state/`，否则可能覆盖或轮转掉干净断点。

## 继续训练后必须做的验证

训练完成后，依次做：

1. 检查 `train_results.json`、`trainer_state.json`、`checkpoint-*` 是否生成。
2. 检查新 checkpoint 是否有 `optimizer.pt`、`scheduler.pt`、`rng_state.pth`。
3. 跑标准 PEFT adapter load check。
4. 跑 30 条健康探针。
5. 跑 40 条 temporal balanced probe。
6. 将结果与 1500、2000 对比。
7. 只在 temporal balanced 或健康探针有实质改善时，才建议继续更长训练。

## 判断是否继续的标准

可以继续：

- train loss 仍下降，且没有明显输出退化。
- 30 条健康探针继续上升，或者保持稳定但 temporal balanced 分项改善。
- temporal 数值题不再只输出固定常见错误值。
- checkpoint state 保持完整，磁盘没有异常增长。

应该暂停分析：

- loss 下降但 30/40 探针不变或变差。
- temporal 数值题继续固定输出同一批错误值。
- checkpoint 保存异常，无法标准加载。
- 数据盘逼近满盘，或者输出目录出现非预期膨胀。

## 不要误解的点

- 当前还不能说“复现了论文 Stage 1 效果”。
- 当前是 Qwen3-4B-Instruct-2507 上的单卡 LoRA 链路复现，不是 STReasoner-8B 完整训练复现。
- 探针不是官方 evaluate，只用于训练中间决策。
- `2000 save_state` 是目前最适合继续的断点，但它之前的 1500 来源仍是 adapter 分段接续，不是从头严格连续 2000 steps。

## 2026-06-04 追加：preflight 与 postcheck 脚本

为了减少下次继续训练时的临时拼命令，已新增两个辅助脚本。本轮没有启动训练。

### 训前 preflight

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_preflight_2000to2500.sh
```

作用：

- 检查 `base_model/Qwen3-4B-Instruct-2507` 和 STReasoner TS config/code 是否存在。
- 检查 `data/ST-Bench/ST-Align/alignment_train.jsonl` 是否存在。
- 检查 `2000 save_state/checkpoint-500` 的 `adapter_model.bin`、`adapter_config.json`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`。
- 检查 `2500steps_from2000_save_state` 输出目录是否已经存在；如果存在就停止，避免混合旧输出。
- 对真正续训脚本做 `bash -n`。
- 打印系统盘、数据盘、GPU 状态。

本轮已运行，结果：

```text
ready_to_train=ok
/ = 11%
/root/autodl-tmp = 66%
A100-PCIE-40GB: 1MiB / 40960MiB, GPU-Util 0%, no running processes
expected output: output/single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state
```

### 训后 postcheck

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_2500_postcheck.sh
```

作用：

- 检查 2500 输出目录、`train_results.json`、`trainer_state.json`、最新 `checkpoint-*` 的 state 文件。
- 跑标准 PEFT adapter load check。
- 跑 30 条健康探针。
- 跑 40 条 temporal balanced probe。
- 自动把日志和 jsonl 结果写到 `00_new_codes/repro_autodl/experiments/logs/` 和 `results/`。

注意：postcheck 只有在 2500 训练完成后才运行；当前 2500 输出目录还不存在，所以本轮只做了 `bash -n` 语法检查。

### 下次推荐顺序

```bash
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_preflight_2000to2500.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_2500_postcheck.sh
```

仍然强调：第二条是真训练，需要用户明确允许后再执行。

