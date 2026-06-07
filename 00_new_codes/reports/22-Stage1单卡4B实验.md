# Stage 1 单卡 4B 训练 smoke 与小步数实验报告

## 结论

本轮已经完成单卡 A100-40GB 上的 Stage 1 训练链路验证：使用 `Qwen3-4B-Instruct-2507`，走仓库官方 `src/train.py` 的 SFT 入口，训练 `ST-Align/alignment_train.jsonl`，最终得到两份有效 LoRA 产物：

- `10` step smoke：`output/single_a100_qwen3_4b_stage1_lora_smoke/`
- `100` step 小步数训练：`output/single_a100_qwen3_4b_stage1_lora_100steps/`

这不是论文规模的 Stage 1 完整复现。它的意义是确认：当前单卡、当前 4B 模型、当前 ST-Align 数据、当前 STReasoner TS config、当前 LLaMA-Factory 训练入口可以完成一次真实参数更新、保存 adapter，并且不会再把系统盘写爆。

最重要的判断依据不是“脚本结束了”，而是：

- 最终日志明确使用 `FlashAttention-2`，没有使用 SDPA。
- `trainer_state.json` 记录了完整 step、loss、runtime。
- `adapter_model.bin` 中所有 `lora_B` 张量都非零。
- `adapter_model.bin` 中 `ts_encoder` 相关张量也都非零。
- 日志显示 `Applied SFT learning rate 0.000010 to 10 time-series encoder parameter(s).`
- 标准 `PeftModel.from_pretrained(base_model, adapter_dir)` 加载检查通过。

因此这次不是“只保存空 adapter”或“loss 看起来动了但 optimizer 没有更新”的假成功。

本报告有一个重要修正：最早的 10 step / 100 step LoRA 产物虽然完成训练并保存了非零权重，但因为同时使用了 `--additional_target ts_encoder` 和 `ts_encoder.mlp.*` LoRA target，导致标准 PEFT 加载时报 `KeyError: base_model.model.ts_encoder.mlp.0.lora_A.default.weight`。因此旧产物只保留为排错证据，最终有效产物已重跑为“只对 `ts_encoder.mlp.*` 做 LoRA，不把整个 `ts_encoder` 当 modules_to_save 保存”的版本。

在 `100` step 通过后，继续扩展了一个 `500` step Stage 1 LoRA 小实验。`500` step adapter 也通过标准加载和最小生成检查。第 0 条 ST-Align 样例的 gold 是 `8.0`，`500` step adapter 生成 `1.00`，说明输出形态从 100 step 的乱码变成短数字，但不能说明答案正确，也不能当作正式评测。

## 前置知识

### 1. 这次训练的 Stage 1 是什么

STReasoner 的训练流程分多个阶段。今天只处理 Stage 1：

- 数据：`data/ST-Bench/ST-Align/alignment_train.jsonl`
- 训练类型：SFT
- 模板：`STReasoner-Align`
- 样例数：`153700`
- 目标：让接入 time-series encoder 的 Qwen 模型学习 ST-Bench 输入格式、时间序列 token、图结构描述和答案对齐。

今天不做 Stage 2、Stage 3，也不声称得到论文最终效果。Stage 1 先跑稳，后面阶段下次单独设计。

### 2. 4B 模型是什么

本地模型路径：

```text
base_model/Qwen3-4B-Instruct-2507/
```

它来自 `Qwen/Qwen3-4B-Instruct-2507`，是 Qwen 官方 post-trained / instruct 模型，不是 STReasoner 官方在 HF 上发布的 8B 微调成品。

为了让它能按 STReasoner 的 time-series 结构加载，已经做过：

- 将 `base_model/Config-Qwen3-4B-Instruct-2507/` 中的 TS config 复制到模型目录。
- 运行 `initial_model.py --model_path base_model/Qwen3-4B-Instruct-2507` 初始化 `ts_encoder`。
- 修正 `initial_model.py` 的保存 dtype，避免默认 fp32 保存导致模型目录膨胀到约 `17G`。

当前模型目录大小约 `8.3G`。

### 3. 为什么不继续 full fine-tuning

原本最像论文训练的路线是 4B full fine-tuning：

- `finetuning_type=full`
- DeepSpeed ZeRO-3
- CPU optimizer offload
- `cutoff_len=4096`
- `max_steps=10`

这条路线已经验证过能加载模型、读取数据、预处理数据、编译 CPUAdam，但会在初始化 optimizer states 前后被容器内存上限杀掉。

关键日志：

```text
Before initializing optimizer states
CPU Virtual Memory: used = 78.48 GB
subprocess exits with return code = -9
```

当前容器 cgroup 内存上限约 `77.3GB`。所以这是容器内存限制导致的失败，不是 GPU OOM，不是数据缺失，也不是系统盘爆满。

因此今天改做 LoRA。这个选择的目的不是伪装成论文设置，而是在单卡 A100-40GB 和有限 CPU memory 里，把 Stage 1 训练链路先真实跑通。

### 4. LoRA 和这次 adapter 的含义

LoRA 训练冻结大部分 base model 权重，只训练低秩 adapter。这样显著减少可训练参数和 optimizer state。

本次最终配置：

- `finetuning_type=lora`
- `lora_rank=8`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `lora_target=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 不使用 `additional_target=ts_encoder`
- `bf16`
- `flash_attn=fa2`

训练日志显示：

```text
trainable params: 16,700,480
all params: 4,455,224,400
trainable%: 0.3749
```

也就是说，本次训练更新约 `1670` 万参数，占总参数约 `0.37%`。这部分包含 Qwen decoder 线性层 LoRA，以及 `ts_encoder.mlp.0/2/4/6/8` 的 LoRA 参数。

## 代码与脚本修改

### 1. 最小源码修正：注册 qwen3 time-series 类型

修改文件：

```text
src/llamafactory/model/model_utils/timeseries.py
```

原因：本地 `Qwen3TSConfig` 的 `model_type` 是 `qwen3`：

```text
base_model/Qwen3-4B-Instruct-2507/configuration_qwen3_ts.py: model_type = "qwen3"
base_model/Qwen3-4B-Instruct-2507/config.json: "model_type": "qwen3"
```

但 `timeseries.py` 原先只注册了 `qwen3ts`。如果不注册 `qwen3`，`timeseries_sft_lr` 和 TS LoRA target 相关逻辑不会按预期识别当前模型。

本轮做的修正是给 `qwen3` 增加同样的 time-series model definition：

```python
_register_timeseries_model(
    model_type="qwen3",
    encoder_key="ts_encoder",
    lora_target_prefixes=["ts_encoder.mlp"],
    modules_to_save=[],
)
```

这属于必要的适配修正，不是改 prompt，也不是重写训练逻辑。

### 2. 新增训练脚本

新增脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_smoke.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_100steps.sh
```

两个脚本都调用官方训练入口：

```text
src/train.py
```

关键环境变量：

```bash
export PYTHONPATH=.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
```

这些变量的意义：

- 所有 HuggingFace / datasets / torch / triton 缓存都放到数据盘 `/root/autodl-tmp`。
- 避免写入系统盘 `/root/.cache`。
- 离线加载模型，确认命中本地模型目录。
- 关闭 W&B，避免无人值守训练时卡在 API key。

## 中途失败与修正

### 1. bitsandbytes 与 triton 不兼容

最早卡在：

```text
ModuleNotFoundError: No module named 'triton.ops'
```

当前环境：

```text
torch 2.6.0
triton 3.2.0
```

仓库 requirements 中的 `bitsandbytes==0.43.1` 对当前 triton 不兼容。已将当前环境中的 bitsandbytes 升到 `0.45.2`。这是对环境的必要修正，记录为偏离 requirements 的地方。

### 2. DeepSpeed CPUAdam 依赖 ninja

DeepSpeed 编译 CPUAdam 时需要 `ninja`。conda env 里有 `ninja`，但脚本如果用绝对路径调用 python，`PATH` 里不一定包含 env 的 `bin`。

处理方式：训练脚本开头加入：

```bash
export PATH="/root/autodl-tmp/conda/envs/str-py310/bin:${PATH}"
```

### 3. full fine-tuning 被内存上限杀掉

这部分已经在前面解释。结论是：不再重复 full fine-tuning smoke，改做 LoRA。

### 4. `lora_target=all` 不是好选择

第一次 LoRA smoke 用了 `lora_target=all`，失败日志：

```text
ValueError: Target module Qwen3DecoderLayer(...) is not supported
```

原因：time-series encoder 里有 `ts_encoder.mlp`，它是 `Sequential`，内部层名包含 `0,2,4,...` 这样的数字模块名。`find_all_linear_modules` 把这些数字也加入 target 后，PEFT 匹配范围过宽，最后误匹配到 `Qwen3DecoderLayer`，导致不支持的模块被当成 LoRA target。

修正方式：不用 `lora_target=all`，显式指定 Qwen 线性层：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

再由 time-series patch 自动追加完整的 `ts_encoder.mlp.*` target。

### 5. W&B 未关闭会卡住训练

一次 retry 已经进入训练开始阶段，但被 W&B API key 卡住：

```text
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

修正：

```bash
export WANDB_DISABLED=true
--report_to none
```

### 6. fp16 smoke 跑完但没有有效更新

有一次 fp16 + W&B disabled 的 LoRA smoke 完成了 10 steps，但日志里：

```text
grad_norm = nan
learning_rate = 0.0
ts_encoder_learning_rate = 0.0
```

并且检查 adapter 后发现 LoRA B 张量仍为零。判断：fp16 动态 loss scaling 可能因为 overflow 跳过了 optimizer step。这种情况不能算有效训练。

修正：A100 支持 bf16，最终脚本改用：

```text
--bf16
```

bf16 后 `grad_norm` 有限，adapter 权重也确实非零。

### 7. 避免 SDPA，最终显式使用 FlashAttention-2

中间有一次 bf16 smoke 成功，但日志显示：

```text
Using torch SDPA for faster training and inference.
```

由于本项目此前明确要求“不许用 sdpa”，最终脚本加入：

```text
--flash_attn fa2
```

最终两份有效产物的日志都显示：

```text
Using FlashAttention-2 for faster training and inference.
```

### 8. `additional_target=ts_encoder` 会造成 adapter 标准加载失败

继续做 Stage 1 最小可用性检查时，发现旧版 adapter 不能被标准 PEFT 加载：

```text
KeyError: 'base_model.model.ts_encoder.mlp.0.lora_A.default.weight'
```

原因：旧脚本同时做了两件事：

- 用 time-series patch 把 `ts_encoder.mlp.*` 加入 LoRA target。
- 又通过 `--additional_target ts_encoder` 把整个 `ts_encoder` 放进 `modules_to_save`。

这样训练时能跑，但保存出的 adapter 里同时混有 `ts_encoder.modules_to_save.*` 和 `ts_encoder.mlp.*` LoRA key，标准 `PeftModel.from_pretrained` 回读时 key 对不上。

最终修正：

- 删除脚本中的 `--additional_target ts_encoder`。
- 保留显式 Qwen LoRA target。
- 让 `patch_timeseries_modules_for_lora()` 自动追加 `ts_encoder.mlp.0/2/4/6/8`。

修正后 `adapter_config.json` 中：

```text
modules_to_save = null
target_modules includes ts_encoder.mlp.0/2/4/6/8
```

最终加载检查通过：

```text
peft_model_class=PeftModelForCausalLM
active_adapter=default
nonzero_lora_B=257/257
adapter_load_check=ok
```

## 最终实验配置

### 共同配置

| 项目 | 值 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| Python | `/root/autodl-tmp/conda/envs/str-py310/bin/python` |
| 训练入口 | `src/train.py` |
| 模型 | `base_model/Qwen3-4B-Instruct-2507` |
| 数据 | `data/ST-Bench/ST-Align/alignment_train.jsonl` |
| 样例数 | `153700` |
| 模板 | `STReasoner-Align` |
| 训练类型 | SFT |
| 微调方式 | LoRA |
| dtype | bf16 |
| attention | FlashAttention-2 |
| batch size | per device 1 |
| gradient accumulation | 8 |
| effective batch size | 8 |
| cutoff_len | 4096 |
| learning_rate | `1e-5` |
| timeseries_sft_lr | `1e-5` |
| trainable params | `16,700,480` |
| all params | `4,455,224,400` |
| trainable% | `0.3749` |

### 10 step smoke

命令脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_smoke.sh
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_loadable_fa2_20260604_174208.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_loadable_adapter_check_20260604_174342.log
```

输出：

```text
output/single_a100_qwen3_4b_stage1_lora_smoke/
```

结果：

| 指标 | 值 |
|---|---|
| global_step | 10 |
| max_steps | 10 |
| train_runtime | 52.1041 秒 |
| train_samples_per_second | 1.535 |
| train_steps_per_second | 0.192 |
| train_loss | 16.393671607971193 |
| first logged loss | 16.5882 |
| last logged loss | 16.6014 |
| min logged loss | 16.0539 |
| max logged loss | 16.8981 |
| 输出目录大小 | 159M |
| `lora_B` 非零张量 | 257 / 257 |
| `ts_encoder` 非零张量 | 10 / 10 |
| `adapter_model.bin` | 64M |
| 标准 PEFT 加载 | 通过 |

判断：smoke 通过。它完成了模型加载、数据读取、训练 step、参数更新、adapter 保存、trainer state 保存和标准 PEFT 回读。

### 100 step 小步数训练

命令脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_100steps.sh
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_loadable_fa2_20260604_174411.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_loadable_adapter_check_20260604_175255.log
```

输出：

```text
output/single_a100_qwen3_4b_stage1_lora_100steps/
```

结果：

| 指标 | 值 |
|---|---|
| global_step | 100 |
| max_steps | 100 |
| train_runtime | 482.5574 秒 |
| train_samples_per_second | 1.658 |
| train_steps_per_second | 0.207 |
| train_loss | 12.218571281433105 |
| first logged loss | 16.5882 |
| last logged loss | 10.1921 |
| min logged loss | 7.68 |
| max logged loss | 16.9052 |
| 输出目录大小 | 159M |
| `lora_B` 非零张量 | 257 / 257 |
| `ts_encoder` 非零张量 | 10 / 10 |
| `adapter_model.bin` | 64M |
| 标准 PEFT 加载 | 通过 |

判断：100 step 小步数训练通过。loss 从约 `16.59` 到最后约 `10.19`，中间最低到 `7.68`。这个趋势只能说明小步数训练链路有效，不能说明模型已经达到可用论文效果。

### 500 step 小步数训练

命令脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_500steps.sh
```

日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_loadable_fa2_20260604_180409.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_adapter_load_check_20260604_184604.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_adapter_generate_check_20260604_184630.log
```

输出：

```text
output/single_a100_qwen3_4b_stage1_lora_500steps/
```

结果：

| 指标 | 值 |
|---|---|
| global_step | 500 |
| max_steps | 500 |
| train_runtime | 2475.1534 秒 |
| train_samples_per_second | 1.616 |
| train_steps_per_second | 0.202 |
| train_loss | 3.8557742080688477 |
| first logged loss | 16.4267 |
| last logged loss | 1.5853 |
| min logged loss | 1.0787 |
| max logged loss | 16.4267 |
| 输出目录大小 | 159M |
| `lora_B` 非零张量 | 257 / 257 |
| `ts_encoder` 非零张量 | 10 / 10 |
| `adapter_model.bin` | 64M |
| 标准 PEFT 加载 | 通过 |
| 最小生成检查 | 通过 |

500 step 的最小生成检查使用 ST-Align 第 0 条样例：

```text
category=temporal
gold=8.0
input_ids_shape=(1, 492)
timeseries_shape=(5, 192, 1)
generated_new_tokens=5
response='1.00'
adapter_generate_check=ok
```

判断：500 step 小实验完成了更长一点的 Stage 1 LoRA 训练，loss 继续下降，adapter 可以标准加载，也可以进入带 time-series 输入的 `model.generate`。但是第 0 条样例输出 `1.00` 不等于 gold `8.0`，不能把它写成效果复现，只能写成训练链路和输出格式的进一步验证。

## 输出文件说明

两个输出目录结构基本一致。主要文件含义如下：

- `adapter_model.bin`：LoRA adapter 和 `ts_encoder` 相关保存权重，是最重要的训练产物。
- `adapter_config.json`：PEFT adapter 配置，记录 LoRA rank、target modules 等。
- `trainer_state.json`：训练步数、loss、runtime、学习率等状态。
- `trainer_log.jsonl`：逐步日志的 jsonl 版本，适合后续画图或分析。
- `train_results.json` / `all_results.json`：训练结束汇总指标。
- `training_loss.png`：LLaMA-Factory 自动生成的 loss 图。
- `checkpoint-10/` 或 `checkpoint-100/`：训练中保存的 checkpoint。
- tokenizer / processor 文件：为后续加载 adapter 时保留配套 tokenizer 和 processor。

目前每个输出目录约 `159M`，比旧版 `additional_target=ts_encoder` 的输出更小，也比 full checkpoint 小很多，数据盘压力可控。

## 资源占用

训练后检查：

```text
系统盘 /：30G 总量，约 3.1G 已用，使用率 11%
数据盘 /root/autodl-tmp：50G 总量，约 32G 已用，剩余约 19G，使用率 64%
GPU：A100-PCIE-40GB，训练结束后显存约 1MiB 使用，空闲
```

本轮没有把 torch extension、HF cache、triton cache 写到系统盘。系统盘没有复现之前 checkpoint 实验导致的 100% 问题。

需要注意：数据盘剩余约 `19G`，足够继续做短训练和少量 adapter 保存，但不适合无限保留多份 full checkpoint。后续如果做更长 Stage 1，需要设置 `save_total_limit=1` 或按阶段清理旧输出。

## 当前有效产物与无效产物

### 有效产物

建议保留：

```text
output/single_a100_qwen3_4b_stage1_lora_smoke/
output/single_a100_qwen3_4b_stage1_lora_100steps/
output/single_a100_qwen3_4b_stage1_lora_500steps/
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_loadable_fa2_20260604_174208.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_loadable_fa2_20260604_174411.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_loadable_fa2_20260604_180409.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_loadable_adapter_check_20260604_174342.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_loadable_adapter_check_20260604_175255.log
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_smoke.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_100steps.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_500steps.sh
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_adapter_load_check.py
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_adapter_generate_check.py
```

### 只作为排错证据的日志

这些日志记录了失败原因，不建议当作正式结果：

```text
00_new_codes/repro_autodl/experiments/logs/single_a100_qwen3_4b_stage1_smoke_20260604_103300.log
00_new_codes/repro_autodl/experiments/logs/single_a100_qwen3_4b_stage1_smoke_retry_bnb0452_20260604_112548.log
00_new_codes/repro_autodl/experiments/logs/single_a100_qwen3_4b_stage1_smoke_retry_ninja_path_20260604_112758.log
00_new_codes/repro_autodl/experiments/logs/single_a100_qwen3_4b_stage1_smoke_retry_timeout1800_20260604_113051.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_20260604_115034.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_retry_20260604_115146.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_wandb_off_20260604_115244.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_bf16_20260604_115622.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_20260604_115852.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_fa2_20260604_121653.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_100steps_fa2_20260604_120810.log
00_new_codes/repro_autodl/experiments/logs/stage1_lora_adapter_load_check_20260604_174022.log
```

其中 `stage1_lora_smoke_bf16_20260604_115622.log` 和 `stage1_lora_100steps_20260604_115852.log` 虽然能跑，但使用了 SDPA，不作为最终采用结果。`stage1_lora_smoke_fa2_20260604_121653.log` 和 `stage1_lora_100steps_fa2_20260604_120810.log` 虽然用了 FlashAttention-2 且训练有效，但旧脚本含 `--additional_target ts_encoder`，标准 PEFT 加载不合格，因此也只作为排错证据。

## 后续建议

如果继续只做 Stage 1，建议下一步不是马上进入 Stage 2，而是：

1. 后续如果继续 Stage 1，可从当前 `500` step adapter 继续训练到 `1000` 或更多 steps，但要先明确“继续训练”脚本如何 resume adapter。
2. 继续使用 `bf16 + flash_attn fa2`。
3. 继续保存到独立 output 目录，保留 `save_total_limit=1`。
4. 每次训练后检查 adapter 是否真的更新，至少检查 `lora_B` 非零、`ts_encoder` 非零、标准 PEFT 加载和最小生成。
5. 如果想判断效果，不能只看第 0 条训练样例，需要另写小规模 held-out / ST-Test 检查计划。

不建议现在承诺“单卡完整复现论文 Stage 1 full fine-tuning”。当前最稳妥的说法是：单卡 A100-40GB 可以完成 4B 模型的 Stage 1 LoRA 训练链路和小步数训练；论文规模 full fine-tuning 受当前容器内存限制，不适合作为今天路线。

## 500 step 多样例生成探针

在 `500` step adapter 通过标准加载和第 0 条生成检查后，又补做了一个很小的多样例探针。目的不是正式评测，而是避免只看第 0 条样例就误判模型状态。

新增脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_adapter_probe.py
```

运行设置：

```text
adapter_dir=output/single_a100_qwen3_4b_stage1_lora_500steps
data_file=data/ST-Bench/ST-Align/alignment_train.jsonl
sample_indices=0,1,71,72,121,122
max_new_tokens=32
```

样例选择方式：

- `0,1`：`temporal` 类样例。
- `71,72`：`spatial` 类样例。
- `121,122`：`spatial_temporal` 类样例。

探针产物：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_adapter_probe_20260604_185255.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_500steps_probe_20260604_185255.jsonl
```

结果如下：

| sample_index | category | gold | response | generated_new_tokens | simple exact match |
|---:|---|---|---|---:|---|
| 0 | temporal | 8.0 | 1.00 | 5 | false |
| 1 | temporal | 0.0654 | 1.0 | 4 | false |
| 71 | spatial | no | no | 2 | true |
| 72 | spatial | no | no | 2 | true |
| 121 | spatial_temporal | demand_source | 0.00 | 5 | false |
| 122 | spatial_temporal | demand_source | 0.00 | 5 | false |

汇总：

```text
probe_exact_match_simple=2/6
cuda_max_memory_allocated_gb=8.50
cuda_max_memory_reserved_gb=8.60
adapter_probe=ok
```

这个结果说明：

- `500` step adapter 的生成链路是稳定的，模型不会再像 `100` step 时输出乱码。
- 输出已经偏向短答案格式，生成 token 数很少，没有出现长篇重复。
- 但 `temporal` 与 `spatial_temporal` 仍明显没有学会，`spatial` 的两个 `no` 只能说明这几个样例上匹配，不能当成正式效果。
- 当前阶段只能写成“训练链路和 adapter 可用，小样例探针部分匹配”，不能写成“Stage 1 效果复现”。

如果明天继续，建议先不要进入 Stage 2/3。更合理的路线是基于当前 `500` step adapter 做接续训练，例如扩到 `1000` steps，然后复用同一个探针比较输出是否继续改善。

## 500 到 1000 step 接续训练

在 `500` step 探针之后，又做了一次 Stage 1 LoRA 接续训练。这里要特别说明：这不是 Trainer optimizer state 级别的严格 resume，因为前面的脚本使用了 `save_only_model`，checkpoint 中没有保留 optimizer state。实际做法是：

1. 加载 base model：`base_model/Qwen3-4B-Instruct-2507/`
2. 通过 `--adapter_name_or_path ./output/single_a100_qwen3_4b_stage1_lora_500steps` 加载已有 `500` step adapter。
3. 在这个 adapter 基础上再训练 `500` steps。
4. 保存到新目录：`output/single_a100_qwen3_4b_stage1_lora_1000steps_from500/`

因此报告里把它称为 `1000steps_from500` 或“500+500 adapter 接续训练”，不把它写成严格意义的 optimizer-resume。

新增脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_continue_500to1000.sh
```

训练日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_500to1000_fa2_20260604_185655.log
```

关键日志证据：

```text
Loaded adapter(s): ./output/single_a100_qwen3_4b_stage1_lora_500steps
Using FlashAttention-2 for faster training and inference.
Applied SFT learning rate 0.000010 to 10 time-series encoder parameter(s).
```

训练指标：

| 项目 | 数值 |
|---|---:|
| 额外训练步数 | 500 |
| 概念累计步数 | 约 1000 |
| train_runtime | 2434.5553 秒，约 40 分 35 秒 |
| train_loss | 0.8910863523483277 |
| loss first logged | 1.4257 |
| loss last logged | 0.6153 |
| loss min | 0.4825 |
| loss max | 1.7625 |
| train_samples_per_second | 1.643 |
| train_steps_per_second | 0.205 |

输出目录：

```text
output/single_a100_qwen3_4b_stage1_lora_1000steps_from500/
```

目录大小仍约 `159M`，没有出现 checkpoint 目录异常膨胀。

标准 adapter 加载检查通过：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1000steps_from500_adapter_load_check_20260604_193809.log
nonzero_lora_B=257/257
adapter_load_check=ok
```

### 1000 step 多样例生成探针

探针仍使用同一组 6 条 ST-Align 训练样例：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1000steps_from500_adapter_probe_20260604_193836.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_1000steps_from500_probe_20260604_193836.jsonl
```

结果如下：

| sample_index | category | gold | 500-step response | 1000-step response | simple exact match |
|---:|---|---|---|---|---|
| 0 | temporal | 8.0 | 1.00 | 200 | false |
| 1 | temporal | 0.0654 | 1.0 | 0.036 | false |
| 71 | spatial | no | no | no | true |
| 72 | spatial | no | no | no | true |
| 121 | spatial_temporal | demand_source | 0.00 | demand_source | true |
| 122 | spatial_temporal | demand_source | 0.00 | demand_source | true |

汇总：

```text
500-step probe_exact_match_simple=2/6
1000-step probe_exact_match_simple=4/6
cuda_max_memory_allocated_gb=8.50
cuda_max_memory_reserved_gb=8.60
```

解释：

- 接续训练后，`spatial_temporal` 两条样例从错误的 `0.00` 变为正确的 `demand_source`，说明继续训练确实改变了输出能力。
- `spatial` 两条样例继续保持正确的 `no`。
- `temporal` 两条仍不正确，尤其第 0 条从 `1.00` 变成 `200`，不能写成已经学会数值型 temporal 任务。
- 这仍然只是训练集小样例探针，不是 held-out evaluation，更不是论文效果复现。

当前最稳妥结论：单卡 A100-40GB 上，4B LoRA Stage 1 已经能稳定完成 adapter 接续训练，且 500 到 1000 step 的小样例探针有可见改善；但 temporal 数值类任务仍不稳定，后续如果继续 Stage 1，应继续增加步数或设计更系统的小规模验证。

### 1000 step 30 条训练样例健康检查

为了避免 6 条样例过小，又补做了一个稍宽的训练集健康检查：每个类别取最前面的 10 条，共 30 条。它仍然不是 held-out evaluation，只用于观察当前 adapter 是否已经开始掌握不同类别的输出形式。

探针产物：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1000steps_from500_adapter_probe30_20260604_194116.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_1000steps_from500_probe30_20260604_194116.jsonl
```

样例范围：

```text
temporal: 0,1,2,3,4,5,6,7,8,9
spatial: 71,72,73,74,75,76,77,78,79,80
spatial_temporal: 121,122,123,124,125,126,127,128,129,130
```

类别汇总：

| category | correct | total | observation |
|---|---:|---:|---|
| temporal | 0 | 10 | 数值和类型答案仍不稳定，经常输出固定错误值，如 `200`、`0.036`、`0.25` |
| spatial | 10 | 10 | `yes/no` 类样例全部匹配 |
| spatial_temporal | 4 | 10 | `demand_source` 和部分数字类匹配，但 `propagation`、节点编号类仍有错 |
| total | 14 | 30 | 只能说明训练集小探针已有部分学习迹象 |

这个 30 条探针把当前状态说得更清楚：

- `spatial` 的 yes/no 短答案已经最容易学到。
- `spatial_temporal` 有部分进展，但还不稳。
- `temporal` 是明显短板，尤其连续数值或周期参数类问题没有学会。

因此后续如果继续 Stage 1，优先目标不是“进入下一阶段”，而是继续观察 temporal 是否会随着更多 step 改善。下一次可继续从 `1000steps_from500` adapter 接续训练到约 `1500`，然后复用同一组 30 条探针。

## 1000 到 1500 step 接续训练

基于 `1000steps_from500` 的 30 条健康检查，继续做了一段 `1000 -> 1500` adapter 接续训练。和上一段一样，这仍然是加载已有 adapter 后继续训练 `500` steps，不是带 optimizer state 的严格 resume。

新增脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_continue_1000to1500.sh
```

训练日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_1000to1500_fa2_20260604_194332.log
```

输出目录：

```text
output/single_a100_qwen3_4b_stage1_lora_1500steps_from1000/
```

训练指标：

| 项目 | 数值 |
|---|---:|
| 额外训练步数 | 500 |
| 概念累计步数 | 约 1500 |
| train_runtime | 2391.6142 秒，约 39 分 52 秒 |
| train_loss | 0.5175953221321106 |
| loss first logged | 0.6012 |
| loss last logged | 0.4339 |
| loss min | 0.2798 |
| loss max | 0.8553 |
| train_samples_per_second | 1.673 |
| train_steps_per_second | 0.209 |

标准 adapter 加载检查：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1500steps_from1000_adapter_load_check_20260604_202407.log
nonzero_lora_B=257/257
adapter_load_check=ok
```

30 条健康检查：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1500steps_from1000_adapter_probe30_20260604_202435.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_1500steps_from1000_probe30_20260604_202435.jsonl
```

类别汇总：

| stage | temporal | spatial | spatial_temporal | total |
|---|---:|---:|---:|---:|
| 1000steps_from500 | 0/10 | 10/10 | 4/10 | 14/30 |
| 1500steps_from1000 | 1/10 | 10/10 | 5/10 | 16/30 |

观察：

- `spatial` 仍保持 `10/10`。
- `spatial_temporal` 从 `4/10` 到 `5/10`，略有改善。
- `temporal` 从 `0/10` 到 `1/10`，其中 `sinusoidal` 类别题答对了一条，但数值参数仍大量错误。
- 训练 loss 继续下降，但健康检查只小幅提升，说明继续堆 step 有帮助但不够快。

当前不建议把这个结果写成 Stage 1 复现完成。更诚实的说法是：单卡 4B LoRA Stage 1 链路已经稳定，adapter 接续训练从 500 到 1500 steps 后训练集小探针从 `2/6`、`4/6`、`14/30` 推进到 `16/30`，但 temporal 数值任务仍未学稳。

## temporal 专项分析

`1500steps_from1000` 的 30 条健康检查显示 temporal 仍是主要短板。为了确认 temporal 到底弱在哪些题型，重新扫描了完整 `ST-Align/alignment_train.jsonl` 中的 temporal 样例。

temporal 总量：

```text
temporal_total=63943
```

主要题型分布：

| temporal question type | count | example gold |
|---|---:|---|
| sinusoidal amplitude | 12943 | 8.0 |
| sinusoidal frequency | 12943 | 0.0654 |
| sinusoidal phase | 12943 | 1.5708 |
| evolution pattern | 6227 | sinusoidal |
| long-term baseline | 6227 | 12.0 |
| mean reversion speed / kappa | 6216 | 0.3 |
| random fluctuation intensity / sigma | 4616 | 0.01 |
| coupling strength / lambda | 1827 | 1.4 |

然后构造了一个更均衡的 temporal 专项探针：每个主要题型取 5 条，共 40 条。

索引文件：

```text
00_new_codes/repro_autodl/experiments/results/stage1_temporal_balanced_indices_20260604_202843.json
```

探针产物：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_1500steps_from1000_temporal_balanced_probe40_20260604_202856.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_1500steps_from1000_temporal_balanced_probe40_20260604_202856.jsonl
```

专项探针结果：

| temporal question type | correct | total | common wrong response pattern |
|---|---:|---:|---|
| sinusoidal amplitude | 0 | 5 | gold `8.0`，response 固定为 `20.0` |
| sinusoidal frequency | 0 | 5 | gold `0.0654`，response 固定为 `0.0364` |
| sinusoidal phase | 0 | 5 | gold `1.5708`，response 固定为 `-1.5708` |
| evolution pattern | 4 | 5 | `sinusoidal` / `mean_reverting` 分类题基本能答 |
| long-term baseline | 0 | 5 | gold `12.0~15.0`，response 常为 `100` / `120` |
| kappa | 2 | 5 | gold `0.3` 可答，对 `0.2/0.15` 常答 `0.3` |
| sigma | 0 | 5 | response 常偏到 `0.03/0.004/0.008` |
| lambda | 0 | 5 | response 常偏到 `1.5/0.75/0.1` |
| total | 6 | 40 | 主要靠 evolution pattern 和少量 kappa 得分 |

这个专项探针说明 temporal 并不是“完全不会输出格式”，而是：

- 分类型 temporal 问题较容易学，例如 `evolution pattern`。
- 少数离散参数可被记住，例如 `kappa=0.3`。
- 需要从时间序列估计连续参数或周期参数的题型仍然很差。
- 模型经常输出某个常见但错误的候选值，像是在学答案先验，而不是稳定读取 time-series 数值。

### 分段接续训练的限制

还需要修正一个重要表述：当前 `500 -> 1000 -> 1500` 不是严格意义的 Trainer resume。

文件证据：

```text
output/*/checkpoint-500/ 中没有 optimizer.pt、scheduler.pt、rng_state.pth
training_args.bin: save_only_model=True
training_args.bin: resume_from_checkpoint=None
training_args.bin: seed=42
training_args.bin: max_steps=500
```

日志证据：

```text
Loaded adapter(s): ./output/single_a100_qwen3_4b_stage1_lora_500steps
Loaded adapter(s): ./output/single_a100_qwen3_4b_stage1_lora_1000steps_from500
Total optimization steps = 500
```

因此当前做法的真实含义是：

1. 加载上一个 adapter 权重。
2. 用新的 Trainer / optimizer / scheduler 再训练 500 steps。
3. 由于 `seed=42` 和 `data_seed=None`，每段训练的数据顺序可能高度相似，学习率 schedule 也每段重新 warmup/cosine。

这不影响“adapter 权重确实继续被更新”的判断，但会影响“等价于连续训练 1500 steps”的说法。后续如果要更接近正式 Stage 1，应改成保留 optimizer/scheduler/rng state 的 checkpoint，并用 `resume_from_checkpoint` 继续，而不是只用 `adapter_name_or_path` 分段接续。

下一步建议：

- 不再把当前 `1500steps_from1000` 直接称为严格 1500-step 连续训练。
- 后续如果继续训练，建议新建 checkpoint-preserving 路线，去掉 `--save_only_model`，保存 optimizer state。
- 后续固定使用两个探针：30 条三类别健康检查 + 40 条 temporal balanced probe。
- temporal 改善应重点看 amplitude/frequency/phase/baseline/sigma/lambda，而不是只看总分。

## 后续 checkpoint-preserving 脚本

为了修正“只加载 adapter 分段接续”的限制，已经准备了下一步脚本，但本轮没有启动长训练。

第一步：从当前 `1500steps_from1000` adapter 继续训练 500 steps，并保存 optimizer/scheduler/rng state。

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_continue_1500to2000_save_state.sh
```

输出目录：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/
```

关键区别：

- 不再使用 `--save_only_model`。
- 仍使用 `adapter_name_or_path=output/single_a100_qwen3_4b_stage1_lora_1500steps_from1000`。
- 训练结束后应检查 `checkpoint-500/optimizer.pt`、`scheduler.pt`、`rng_state.pth` 是否存在。

第二步：如果第一步完成，后续应使用真正的 checkpoint resume 模板：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh
```

它会检查：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/optimizer.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/scheduler.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/trainer_state.json
```

并使用：

```text
--adapter_name_or_path ./output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500
--resume_from_checkpoint ./output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500
--max_steps 1000
```

这里 `max_steps=1000` 的含义是：从 checkpoint 的 `global_step=500` 继续到该 run 的 `global_step=1000`，也就是再增加 500 steps。这个脚本是为了后续验证“真正连续训练”是否比当前 adapter 分段接续更稳定。

## 2026-06-04 夜间断点更新：先停在 Stage 1 2000 save_state

本轮已按“先把 Stage 1 做扎实”的要求推进到一个新的可靠断点。今晚不再启动新的训练。

当前最新有效输出：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/
```

这一步的真实含义：

- 从 `output/single_a100_qwen3_4b_stage1_lora_1500steps_from1000/` 的 adapter 权重继续训练 500 steps。
- 本段仍不是从旧 optimizer state 严格恢复，因为输入端还是上一个 adapter。
- 但本段训练结束时已经保留了 Trainer 断点状态，后续可以从本段的 `checkpoint-500` 做更严格的 `resume_from_checkpoint`。

关键训练结果：

```text
train_runtime = 2509.1812 秒，约 41 分 49 秒
train_loss = 0.4133506908416748
train_samples_per_second = 1.594
train_steps_per_second = 0.199
output dir size = 287M
```

checkpoint 状态已经确认存在：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/optimizer.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/scheduler.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/rng_state.pth
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/trainer_state.json
```

这说明 `save_state` 路线已经跑通。和之前只保存 adapter 的目录相比，这个目录会更大，但这是后续严格断点续训所必需的，不是异常膨胀。

adapter 标准加载检查也已通过：

```text
adapter_load_check=ok
nonzero_lora_B=257/257
```

对应日志：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_2000steps_from1500_save_state_adapter_load_check_20260604_231307.log
```

2000 save_state 的 30 条健康探针结果：

```text
probe_exact_match_simple = 17/30
cuda_max_memory_allocated_gb = 8.50
cuda_max_memory_reserved_gb = 8.60
```

对应文件：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_2000steps_from1500_save_state_probe30_20260604_231340.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_2000steps_from1500_save_state_probe30_20260604_231340.jsonl
```

2000 save_state 的 40 条 temporal balanced 探针结果：

```text
probe_exact_match_simple = 6/40
cuda_max_memory_allocated_gb = 8.67
cuda_max_memory_reserved_gb = 8.77
```

分项观察：

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

对应文件：

```text
00_new_codes/repro_autodl/experiments/logs/stage1_lora_2000steps_from1500_save_state_temporal_balanced_probe40_20260604_231446.log
00_new_codes/repro_autodl/experiments/results/stage1_lora_2000steps_from1500_save_state_temporal_balanced_probe40_20260604_231446.jsonl
```

阶段性判断：

- loss 从前几段继续下降，但 temporal balanced 探针总分仍是 `6/40`。
- 30 条健康探针从 `1500steps_from1000` 的 `16/30` 到 `2000 save_state` 的 `17/30`，只小幅变化。
- 目前模型已经能稳定输出简短 answer，不再是早期 100 steps 那种明显乱码状态。
- spatial 类和部分 spatial-temporal 类较稳。
- temporal 数值参数还没有学扎实，尤其 amplitude/frequency/phase/baseline/sigma/lambda 仍明显失败。
- 继续训练是否有收益，需要看后续“严格 Trainer resume”是否能让 temporal 细分项改善，而不是只看 train loss。

下次如果继续，优先不是重新跑已有 2000，而是从这里开始：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh
```

该脚本应从：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500
```

继续。启动前先检查：

```text
optimizer.pt
scheduler.pt
rng_state.pth
trainer_state.json
```

如果这几个文件还在，再跑 `resume_2000to2500_save_state` 才有意义。跑完以后仍然要做同样两组探针：

- 30 条三类别健康检查。
- 40 条 temporal balanced probe。

今晚停止点：

- 不再启动 `2000 -> 2500`。
- 不再改 Stage 2/Stage 3。
- 保留当前 `2000 save_state` 作为明天继续的断点。

## 2026-06-04 续训脚本小修正：保留 2000 断点，输出到 2500 新目录

本轮没有启动新训练，只做了继续训练前的小任务检查。

发现的问题：

- 原 `single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh` 把 `OUTPUT_DIR` 仍指向 `output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state`。
- 如果直接运行，Trainer 可能在同一目录写入 `checkpoint-1000`，并因为 `save_total_limit=1` 轮转掉原来的 `checkpoint-500`。
- 这会让 2000 save_state 断点不够干净，也会让目录名和实际训练步数不一致。

已做的修正：

```text
SOURCE_DIR=output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state
RESUME_DIR="${SOURCE_DIR}/checkpoint-500"
OUTPUT_DIR=output/single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state
```

并补充检查：

```text
test -f "${RESUME_DIR}/rng_state.pth"
```

修正后的含义：

- 从 2000 断点目录读取 adapter、optimizer、scheduler、rng、trainer_state。
- 新训练输出写到 `2500steps_from2000_save_state`。
- 原 `2000steps_from1500_save_state/checkpoint-500` 会被保留，方便回滚和对照。

已完成轻量验证：

```text
bash -n 00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh  # 通过
find output -maxdepth 1 -name 'single_a100_qwen3_4b_stage1_lora_2500steps_from2000_save_state'  # 无输出，说明未误启动训练
```

下次如果用户明确允许继续训练，再运行该脚本。运行前仍应检查：

```text
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/adapter_model.bin
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/optimizer.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/scheduler.pt
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/rng_state.pth
output/single_a100_qwen3_4b_stage1_lora_2000steps_from1500_save_state/checkpoint-500/trainer_state.json
```

继续训练后固定做：

- 标准 PEFT adapter load check。
- 30 条三类别健康探针。
- 40 条 temporal balanced probe。
- 将结果与 1500、2000 save_state 对比，不只看 loss。

## Stage 1 当前输出索引

已新增专门索引报告：

```text
00_new_codes/reports/32-Stage1输出索引清单.md
```

该报告汇总了当前所有 Stage 1 LoRA 输出目录、训练方式、loss、runtime、目录大小、标准加载状态、是否可严格 resume、探针结果、关键日志和下次继续训练前检查清单。后续继续 Stage 1 前优先阅读这份索引。

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

