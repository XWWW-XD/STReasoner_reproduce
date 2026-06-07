# 单卡 A100 Stage 1 正式训练执行计划

## 结论先行

今天只做 Stage 1。当前最稳妥的路线是：

1. 承认并记录：4B full fine-tuning + ZeRO-3 CPU offload 已被当前容器内存上限卡住。
2. 不继续重复 full fine-tuning smoke。
3. 改做 Stage 1 LoRA smoke，目标是先让训练真正进入 step、出现 loss、正常保存。
4. LoRA smoke 通过后，再做 Stage 1 `100` steps 小实验。
5. 最后写 Stage 1 专门报告，把原因、命令、日志、loss、资源占用和输出文件都讲清楚。

今天的重点不是追论文指标，而是把“单卡怎么训练、怎么判断训练真的跑通、怎么记录失败和资源瓶颈”学明白。

## 当前状态

### 2026-06-04 最新断点

Stage 1 LoRA 已经从 smoke 推进到 `1500steps_from1000`：

- `500` step 输出：`output/single_a100_qwen3_4b_stage1_lora_500steps/`
- `500` step 训练日志：`00_new_codes/repro_autodl/experiments/logs/stage1_lora_500steps_loadable_fa2_20260604_180409.log`
- `500` step 探针：同一 6 条 ST-Align 训练样例简单精确匹配 `2/6`。
- `1000steps_from500` 输出：`output/single_a100_qwen3_4b_stage1_lora_1000steps_from500/`
- `1000steps_from500` 训练日志：`00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_500to1000_fa2_20260604_185655.log`
- `1000steps_from500` 标准加载检查：通过，`PeftModelForCausalLM`，`nonzero_lora_B=257/257`。
- `1000steps_from500` 探针：同一 6 条 ST-Align 训练样例简单精确匹配 `4/6`。
- `1000steps_from500` 30 条健康检查：每类取 10 条训练样例，整体 `14/30`；其中 `temporal=0/10`，`spatial=10/10`，`spatial_temporal=4/10`。
- `1500steps_from1000` 输出：`output/single_a100_qwen3_4b_stage1_lora_1500steps_from1000/`
- `1500steps_from1000` 训练日志：`00_new_codes/repro_autodl/experiments/logs/stage1_lora_continue_1000to1500_fa2_20260604_194332.log`
- `1500steps_from1000` 标准加载检查：通过，`PeftModelForCausalLM`，`nonzero_lora_B=257/257`。
- `1500steps_from1000` 30 条健康检查：整体 `16/30`；其中 `temporal=1/10`，`spatial=10/10`，`spatial_temporal=5/10`。

注意：`1000steps_from500` 是加载 `500` step adapter 后再训练 `500` steps，不是 optimizer state 级别的严格 resume。因为前面保存时使用 `save_only_model`，没有保留 optimizer state。

当前结果：

- `spatial` 两条样例保持正确。
- `spatial_temporal` 两条从 `500` step 的 `0.00` 变为正确的 `demand_source`。
- 30 条健康检查中，`spatial` 已经稳定，但 `temporal` 仍只有 `1/10`。

结论：Stage 1 单卡 4B LoRA 链路已经可训练、可保存、可标准加载、可生成，并且 500 到 1500 step 有可见改善；但 temporal 数值型任务仍是明显短板，不能写成 Stage 1 效果复现。暂时不要进入 Stage 2/3。

重要修正：当前 `500 -> 1000 -> 1500` 是 adapter 分段接续训练，不是严格 Trainer resume。因为脚本使用了 `save_only_model=True`，checkpoint 里没有 `optimizer.pt`、`scheduler.pt`、`rng_state.pth`，并且 `training_args.bin` 里 `resume_from_checkpoint=None`。后续如果继续 Stage 1，不应只机械接着堆 step，而应优先建立 checkpoint-preserving 训练方式，保留 optimizer/scheduler/rng state，再用 `resume_from_checkpoint` 做真正连续训练。

temporal 专项探针结果：`1500steps_from1000` 对 40 条均衡 temporal 样例只达到 `6/40`。其中 evolution pattern `4/5`、kappa `2/5`，但 amplitude/frequency/phase/baseline/sigma/lambda 基本为 `0/5`。下一步重点应分析和改善这些连续数值参数题，而不是只看总 probe 分。

下一步已准备但尚未运行的脚本：

- `00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_continue_1500to2000_save_state.sh`
- `00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh`

第一个脚本用于从 `1500steps_from1000` adapter 继续训练，并保存 optimizer/scheduler/rng state。第二个脚本用于后续基于 `checkpoint-500` 做真正 `resume_from_checkpoint`。这两个脚本的目的不是立即追分，而是修正训练连续性。

### 硬件与磁盘

- GPU：单张 NVIDIA A100-PCIE-40GB。
- 当前 GPU 状态：空闲，无训练、下载、vLLM、EasyR1 残留进程。
- 系统盘 `/`：约 `10%` 使用率
- 数据盘 `/root/autodl-tmp`：约 `64%` 使用率，剩余约 `19G`。可以做 smoke 和小规模训练，但不能随便保留多个完整 checkpoint。

### 模型

当前 4B 模型已经准备好：

- 路径：`base_model/Qwen3-4B-Instruct-2507/`
- 来源：`Qwen/Qwen3-4B-Instruct-2507`，这是 Qwen 官方 post-trained / instruct 模型，不是 STReasoner 微调成品。
- 状态：已下载、已复制 `base_model/Config-Qwen3-4B-Instruct-2507/` 的 STReasoner TS config。
- 状态：已运行 `initial_model.py --model_path base_model/Qwen3-4B-Instruct-2507`。
- 最终保存：2 个 fp16 safetensors shard。
- 大小：约 `8.3G`。
- 检查脚本：`00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_model_check.py` 已通过。

注意：昨晚发现 `initial_model.py` 默认会把模型保存成 fp32，导致目录膨胀到约 `17G`；现在已修正为 `torch_dtype=torch.float16` 加载，并在保存前 `model.half()`。

### 数据

今天只使用 Stage 1 数据：

- `data/ST-Bench/ST-Align/alignment_train.jsonl`

其他训练阶段和评测数据虽然本地已经补齐，但今天不纳入执行计划；下次需要时再单独写计划、单独启动。

ST-Bench 训练数据已经补齐。Stage 1 smoke 的依赖问题已经处理，但 full fine-tuning 又暴露出容器内存上限问题：

- 已解决问题 1：`bitsandbytes==0.43.1` 与 `triton==3.2.0` 不兼容，`import bitsandbytes` 会触发 `ModuleNotFoundError: No module named 'triton.ops'`。已将当前环境中的 `bitsandbytes` 升到 `0.45.2`，不改 torch/triton/deepspeed/transformers。
- 已解决问题 2：DeepSpeed CPUAdam 需要 `ninja` 可执行文件。`ninja` 在 conda env 里存在，但原脚本用绝对路径调用 deepspeed，env `bin` 不在 `PATH`。已在 smoke 脚本中加入 `PATH=/root/autodl-tmp/conda/envs/str-py310/bin:$PATH`。
- 已解决问题 3：PyTorch extension 默认写入 `/root/.cache/torch_extensions`。当前 Stage 1 smoke 脚本需要使用 `TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions`，CPUAdam 已成功编译到数据盘，约 `43M`。
- 当前新断点：4B full fine-tuning + ZeRO-3 CPU offload 在初始化 optimizer states 前后超过容器内存上限。cgroup `memory.max` 约 `77.3GB`，DeepSpeed 日志显示 CPU virtual memory 到约 `78.48GB` 后子进程被 `SIGKILL`，返回码 `-9`。
- 训练状态：尚未进入真实 backward/update step，也没有保存 checkpoint。

## 今天只做 Stage 1

Stage 1 还没有真正跑到训练 step，如果现在讨论别的阶段，很容易变成“计划很多、链路没学会”。今天的目标是把 Stage 1 这一个阶段拆明白、跑扎实、记录清楚。

Stage 1 今天的学习目标：

- 看懂一次 SFT 训练从命令到日志的完整链路。
- 分清 full fine-tuning、LoRA、QLoRA、只训 `ts_encoder` 的区别。
- 记录清楚为什么 full fine-tuning 失败，不把资源限制误判成代码错误。
- 做出一个能在当前单卡 A100 容器里真正进入 step 的 Stage 1 smoke。
- 在 smoke 通过后，跑一个小步数 Stage 1 实验，并写出可复盘报告。

今天明确不做：

- 不跑 full ST-Test。
- 不下载新大模型。
- 不重复 4B full fine-tuning smoke。
- 不为了“看起来更像论文”而忽略当前资源限制。

## Stage 1 已学到的事实

### Stage 1 的数据和任务

Stage 1 使用的是 ST-Align 数据：

- 数据文件：`data/ST-Bench/ST-Align/alignment_train.jsonl`
- 本地记录：约 `153700` 条训练样例
- 训练类型：SFT
- 模板：`STReasoner-Align`
- 目标：让带 time-series 结构的模型学会对齐 ST-Bench 的输入、时间序列 token、图结构和简短答案格式。

这一步不是论文最终效果复现。它是后续训练的地基：如果 Stage 1 都不能稳定训练和保存，后续任务都不应该开始。

### 现有 full fine-tuning 失败原因

原始 Stage 1 full fine-tuning 路线大致是：

- `finetuning_type=full`
- 4.44B 参数全量可训练
- DeepSpeed ZeRO-3
- CPU optimizer offload
- `max_steps=10`
- `cutoff_len=4096`

这条路线已经验证过：

- 模型能加载。
- processor 能加载。
- alignment 数据能读取。
- tokenizer 预处理能跑。
- DeepSpeed CPUAdam 能编译。
- 但没有进入 backward/update step。

最终失败点：

```text
Before initializing optimizer states
CPU Virtual Memory: used = 78.48 GB
subprocess exits with return code = -9
```

容器实际内存上限：

```text
cgroup memory.max ~= 77.3GB
```

判断：当前机器虽然宿主机看起来内存很大，但容器 cgroup 限制约 77GB。4B full fine-tuning 初始化 optimizer states 时超过这个上限，所以被系统杀掉。这个失败不是数据问题、不是模型缺文件、不是 GPU 显存爆、不是系统盘爆，也不是 `bitsandbytes` 依赖问题。

### 已处理的环境问题

已经解决：

- `bitsandbytes==0.43.1` 和 `triton==3.2.0` 不兼容，已将 `bitsandbytes` 升级到 `0.45.2`。
- `ninja` 可执行文件不在默认 `PATH`，已在 smoke 脚本中加入 conda env `bin`。
- PyTorch extension 默认写系统盘，已加入 `TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions`。
- DeepSpeed CPUAdam 已成功编译到数据盘，缓存约 `43M`。

后续报告必须把这几件事和 full fine-tuning 内存失败分开写，不要混成一个“依赖问题”。

## Stage 1 可选路线

### 路线 A：LoRA Stage 1，优先推荐

含义：冻结 4B 主体权重，只在指定线性层上训练低秩 adapter。

优点：

- 显著降低可训练参数和 optimizer states。
- 最容易在当前 77GB 容器内存限制下跑通。
- 仍然是在官方训练入口 `src/train.py` 和 LLaMA-Factory SFT 链路里训练，学习价值高。
- 输出 adapter 较小，数据盘压力低。

风险：

- 不是论文 full fine-tuning 设置。
- 如果没有把 time-series 相关模块纳入训练，可能只能调整语言层，不能充分学习 `ts_encoder`。

建议做法：

- `finetuning_type=lora`
- `lora_rank=8` 起步
- 不使用 `lora_target=all`；它会把 `ts_encoder.mlp` 的数字子层名扩大匹配，可能误命中不支持的模块。
- 显式使用 Qwen 线性层 target：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- 不使用 `additional_target=ts_encoder`；最终验证发现它会和 `ts_encoder.mlp.*` LoRA target 共同导致 adapter 标准 PEFT 加载失败。
- 让 time-series patch 自动追加 `ts_encoder.mlp.0/2/4/6/8`，这样能训练 TS encoder 的 LoRA 参数，也能被 `PeftModel.from_pretrained` 正常回读。
- `max_steps=10` smoke，通过后再 `max_steps=100`

### 路线 B：QLoRA Stage 1，第二选择

含义：用 4-bit 或 8-bit 量化加载 base model，再训练 LoRA adapter。

优点：

- 比普通 LoRA 进一步省显存和内存。
- 如果 LoRA 仍然资源紧张，可以尝试。

风险：

- 更依赖 bitsandbytes 量化路径。虽然当前 `bitsandbytes==0.45.2` 已能导入，但 QLoRA 会走更多 bnb 代码。
- 与当前仓库 `requirements.txt` 的 `bitsandbytes==0.43.1` 有偏离，需要明确记录。
- 如果只是为了先学会 Stage 1 训练，复杂度高于普通 LoRA。

建议：只有路线 A 仍然不稳定时，再尝试 QLoRA。

### 路线 C：只训 `ts_encoder` / 少量 time-series 参数，学习价值高但要先确认代码支持

含义：冻结语言模型大部分参数，只训练 time-series encoder 或少量额外模块。

优点：

- 最贴合“让 Qwen 4B 接入 STReasoner 时间序列结构”的核心问题。
- 可训练参数更少，理论上资源压力低。
- 有助于理解 STReasoner 的 `ts_encoder` 到底怎么接入。

风险：

- 需要确认 LLaMA-Factory 的 `freeze` 是否能准确选中 `ts_encoder`。
- 如果模块名写错，可能出现“看似训练，实际没训练目标模块”的假成功。
- 已验证 `additional_target=ts_encoder` 不适合与 `ts_encoder.mlp.*` LoRA target 同时使用；会造成 adapter 保存后标准加载失败。

建议：作为路线 A 的并行学习点。先用脚本打印 trainable parameters，确认 `ts_encoder` 是否真的参与训练，再决定是否正式跑。

### 路线 D：换更小模型，暂不作为今天第一选择

含义：改用 1.7B / 0.6B 等更小 Qwen。

优点：

- 资源压力最低。

风险：

- 当前仓库没有现成 STReasoner TS config/scripts。
- 会把今天任务从“学习 Stage 1 训练”变成“适配新模型结构”。

建议：如果 4B LoRA / QLoRA 仍跑不通，再考虑。

## 今天推荐执行路线

今天优先做路线 A：**4B LoRA Stage 1**。

执行原则：

1. 不再跑 full fine-tuning smoke。
2. 先写一个独立 LoRA smoke 脚本，不覆盖已有 full smoke 脚本。
3. smoke 只跑 `max_steps=10`。
4. smoke 通过后，跑一个 `max_steps=100` 的 Stage 1 小实验。
5. 每一步都写日志和报告，重点记录“为什么这样改”和“改完后学到了什么”。

## Stage 1 详细执行清单

### Step 1：训练前只读检查

目的：确认没有残留进程、磁盘干净、依赖版本明确。

必须记录：

```bash
ps -eo pid,etime,cmd | grep -E 'deepspeed|src/train.py|torchrun|llamafactory' | grep -v grep || true
nvidia-smi
df -h / /root/autodl-tmp
du -sh base_model/Qwen3-4B-Instruct-2507 data/ST-Bench output /root/autodl-tmp/cache/huggingface /root/autodl-tmp/cache/torch_extensions /root/.cache 2>/dev/null
/root/autodl-tmp/conda/envs/str-py310/bin/python - <<'PY'
import importlib.metadata as m
for p in ["torch", "triton", "bitsandbytes", "accelerate", "deepspeed", "transformers", "peft"]:
    print(p, m.version(p))
PY
```

通过标准：

- 没有训练残留进程。
- 系统盘 `/` 低于 `50%`。
- 数据盘至少剩 `10G`。
- `bitsandbytes` 版本记录清楚。
- `output/` 中没有即将被覆盖的重要结果。

### Step 2：写 LoRA smoke 脚本

建议新建脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_smoke.sh
```

脚本应该复用现有 Stage 1 full smoke 的结构，但改这些点：

- `--finetuning_type lora`
- `--lora_rank 8`
- `--lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 不加 `--additional_target ts_encoder`
- 输出目录改成 `output/single_a100_qwen3_4b_stage1_lora_smoke`
- `--max_steps 10`
- `--save_steps 10`
- 保留 `--save_only_model`
- 保留 cache 环境变量和 `TORCH_EXTENSIONS_DIR`
- 保留 `PATH=/root/autodl-tmp/conda/envs/str-py310/bin:$PATH`

注意：最终验证发现 `additional_target=ts_encoder` 会导致标准 PEFT 加载失败；不要再把整个 `ts_encoder` 作为 modules_to_save 保存。当前正确路线是显式写 Qwen LoRA target，并让 time-series patch 自动追加 `ts_encoder.mlp.*`。

### Step 3：先做 dry run 式检查

目的：尽量在正式训练前发现参数拼写错误。

检查项：

- `bash -n` 通过。
- 输出目录不存在或为空。
- 脚本中的 `max_steps` 是 `10`。
- 脚本没有使用 full fine-tuning。
- 脚本没有覆盖 `output/single_a100_qwen3_4b_stage1_smoke`。

### Step 4：运行 LoRA smoke

运行命令：

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_smoke.sh \
  2>&1 | tee 00_new_codes/repro_autodl/experiments/logs/stage1_lora_smoke_$(date +%Y%m%d_%H%M%S).log
```

观察重点：

- 是否加载 `Qwen3TSForCausalLM`。
- 是否读取 `ST-Bench/ST-Align/alignment_train.jsonl`。
- 是否打印 trainable params，trainable 参数量是否明显低于 full fine-tuning。
- 是否进入 step 进度条。
- loss 是否出现。
- 是否保存 adapter/model。
- 是否仍然触发 DeepSpeed CPUAdam 和大量 optimizer states。

通过标准：

- 至少完成 10 steps。
- 日志中出现 loss。
- 正常退出。
- 输出目录非空。
- 系统盘没有增长。
- 数据盘没有异常增长。

停止条件：

- 再次出现 `return code = -9`。
- CUDA OOM。
- 数据盘剩余小于 `8G`。
- 输出目录异常大，超过预期数 GB。
- trainable params 接近 4.44B，说明没有真正切到 LoRA。

### Step 5：分析 LoRA smoke

smoke 成功后，不要马上跑 100 steps。先分析：

- `trainable params` 是多少。
- `all params` 是多少。
- `trainable%` 是多少。
- 每 step 大概耗时多少。
- GPU 峰值显存是多少。
- CPU 内存是否稳定。
- 输出目录大小是多少。
- 保存的是 adapter 还是完整模型。
- `ts_encoder` 是否在 trainable 参数里。

如果日志没有清楚写出 `ts_encoder` 是否 trainable，需要额外写一个只读检查脚本，加载模型/adapter 后打印 trainable parameter names 的前后若干项，确认到底训练了哪些模块。

### Step 6：Stage 1 100 steps 小实验

只有 LoRA smoke 通过后才做。

建议新建脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_lora_100steps.sh
```

建议配置：

- 复用 LoRA smoke 成功配置。
- `max_steps=100`
- `save_steps=100`
- 输出目录：`output/single_a100_qwen3_4b_stage1_lora_100steps`
- 不覆盖 smoke 输出。

目的：

- 看 loss 是否可记录、是否有明显异常。
- 看训练速度是否适合继续扩大。
- 看 100 steps 输出大小和保存格式。
- 形成后续训练是否能接 adapter 的判断。

停止条件：

- 单步耗时远超预期，导致 100 steps 不适合今天完成。
- loss 为 NaN / inf。
- 输出目录异常膨胀。
- 系统盘或数据盘异常增长。

### Step 7：最小可用性检查

100 steps 完成后，先不跑 full ST-Test。只做最小检查：

- 输出目录是否存在 adapter/model 文件。
- tokenizer / processor 是否被保存。
- 是否能被 `from_pretrained` 或训练框架重新识别。
- 如果推理链路支持 adapter，则只做 1 条样例加载测试。

这里不要求准确率，不写论文效果结论。目标只是证明 Stage 1 输出可复用。

## Stage 1 报告要求

建议新写报告：

```text
00_new_codes/reports/22-Stage1单卡4B实验.md
```

报告必须写清：

- 这次为什么放弃 full fine-tuning。
- full fine-tuning 失败证据：`memory.max`、`78.48GB`、`return code=-9`。
- 为什么选择 LoRA。
- LoRA / QLoRA / full fine-tuning / `ts_encoder` 分别是什么，为什么今天先选 LoRA。
- Stage 1 的前置知识：base model、STReasoner TS config、`initial_model.py`、ST-Align 数据、SFT、adapter、checkpoint/save 的关系。
- 本次设计细节：为什么不覆盖旧脚本，为什么独立 output 目录，为什么先 10 steps 再 100 steps。
- 是否误用过 `additional_target=ts_encoder`，以及为什么最终删除它。
- 最终训练了哪些参数。
- smoke 命令、日志路径、输出路径。
- 100 steps 命令、日志路径、输出路径。
- loss 片段。
- 耗时。
- GPU 显存。
- CPU/cgroup 内存。
- 系统盘和数据盘前后变化。
- 输出目录大小。
- 是否生成可复用 adapter/model。
- 下一步是否适合继续后续训练。
- 本次 Stage 1 输出是否具备以后继续训练的基础。
- 所有关键命令和关键日志片段，后续阅读时不需要再到处翻文件才能理解过程。

报告里不要写：

- 不要说“复现了论文效果”。
- 不要把 LoRA 结果说成 full fine-tuning。
- 不要把 100 steps 说成完整 Stage 1。
- 不要展开其他阶段计划。

报告写法要求：

- 先写结论和当前断点，再写前置知识、设计理由、执行过程、日志证据、资源变化和下一步。
- 技术细节从浅到深展开，尽量让第一次读训练流程的人也能跟上。
- 报告要具体，不只写“成功/失败”，还要写为什么这样判断。
- 日志保留原始命令输出；报告中摘录关键片段并解释含义。

## Stage 1 学习笔记模板

为了后续自己能学会，报告中建议固定写这几段：

### 我这次训练的是什么

说明：

- base model 是什么。
- STReasoner TS config 做了什么。
- Stage 1 alignment 数据是什么。
- LoRA adapter 训练了哪些模块。

### 为什么原方案失败

说明：

- full fine-tuning optimizer states 为什么大。
- ZeRO-3 CPU offload 为什么会吃 CPU 内存。
- 宿主机 `free -h` 和 cgroup `memory.max` 为什么不是一回事。

### 我怎么改的

说明：

- 改了 bitsandbytes 版本。
- 改了 PATH。
- 改了 `TORCH_EXTENSIONS_DIR`。
- 改了训练方式为 LoRA。

### 我如何判断成功

说明：

- 进入 step。
- loss 出现。
- 正常保存。
- 输出可加载。
- 磁盘和内存没有失控。

### 我还不知道什么

说明：

- `ts_encoder` 是否确实训练。
- LoRA adapter 是否能被后续训练或推理脚本正确加载。
- 100 steps 是否足以让输出有任何可观察变化。

## 今天的成功标准

今天成功不是“跑完三阶段”，而是把 Stage 1 学扎实：

- 明确 full fine-tuning 在当前容器内存限制下不可行。
- 完成 LoRA Stage 1 smoke。
- 完成 LoRA Stage 1 100 steps，或者明确记录为什么 100 steps 还不能做。
- 写出 Stage 1 报告，让后续能根据报告复现同样步骤。
- 不污染系统盘。
- 不覆盖已有结果。

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

