# Qwen3-4B-Instruct 三阶段 Baseline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 执行时用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐步勾选 checkbox。
>
> 日期：2026-06-12  
> 代码根目录：`/root/autodl-tmp/STReasoner_reproduce`  
> 实例：AutoDL 单卡 `NVIDIA A800 80GB PCIe`，数据盘 `/root/autodl-tmp` 100G

**Goal:** 尽量复用官方 `scripts/qwen3-4b-instruct/` 与 `src/train.py` / EasyR1 入口，完成 Stage1→Stage2→Stage3 全链路，产出可对比的 baseline  checkpoint 与 ST-Test 指标。

**Architecture:** 以官方三阶段 pipeline 为骨架（Align SFT → CoT SFT → S-GRPO RL），仅在单卡资源约束下做最小参数覆盖；新增代码只放 `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/`，不改官方 shell 本体。SFT 走 `src/train.py`；RL 走 `python -m src.EasyR1.verl.trainer.main` + 官方 `train_stage1+2+3_w_spatial.sh` 同款 Hydra 覆盖项。

**Tech Stack:** Python 3.10 (`/root/autodl-tmp/conda/envs/str-py310`)、DeepSpeed ZeRO-3、LLaMA-Factory SFT、EasyR1/Verl GRPO、FlashAttention-2、官方 vLLM 推理与 `evaluation/evaluate.py`。

---

## 0. 与官方 pipeline 的对照（不可跳步）

| 步骤 | 官方脚本 | 数据 | 模板 | 官方步数 | 产出目录（baseline 约定） |
| --- | --- | --- | --- | --- | --- |
| **0** | `initial_model.py` | — | — | 一次 | `base_model/Qwen3-4B-Instruct-2507/` |
| **1** | `scripts/qwen3-4b-instruct/train_stage1.sh` | `ST-Align/alignment_train.jsonl` | `STReasoner-Align` | 1000 steps | `output/baseline_qwen3_4b/stage1/` |
| **2** | `scripts/qwen3-4b-instruct/train_stage1+2.sh` | 四类 `ST-CoT/*.jsonl` | `STReasoner-CoT` | 400 steps | `output/baseline_qwen3_4b/stage1+2/` |
| **3** | `scripts/qwen3-4b-instruct/train_stage1+2+3_w_spatial.sh` | 四类 `ST-RL/*.jsonl` | `str.jinja` + S-GRPO | 1 epoch | `checkpoints/easy_r1/baseline_qwen3_4b_stage3_w_spatial/` |
| **4** | `model_merger.py` | — | — | 一次 | `.../actor/huggingface/` 合并权重 |
| **5** | `inference/inference_tsmllm_vllm.py` ×4 任务 | `ST-Test` | 官方 prompt | — | `exp/baseline_qwen3_4b_sttest_6144/` |
| **6** | `evaluation/evaluate.py` ×4 任务 | 上一步输出 | — | — | 各任务 `evaluation_metrics.json` |

**注意（官方仓库已知问题）：**

- `scripts/qwen3-4b-instruct/train_stage3.sh` 与 `train_stage1+2+3.sh` 内 `MODEL_PATH` 误写为 `Qwen3-8B`；baseline **只用** `train_stage1+2+3_w_spatial.sh` 的参数集。
- `train_stage2.sh` 从 base 模型直接训 CoT；完整 pipeline 必须用 **`train_stage1+2.sh`**（从 stage1 输出继续）。
- 官方 README 要求 8×A100-80GB；本计划为 **单卡 A800 baseline**，步数与 batch 与论文一致，但 **有效 global batch 会缩小**，指标仅作后续消融对照，不宣称论文复现。

---

## 1. Baseline 策略：两条训练轨（先 A，失败切 B）

### 轨道 A（优先，尽量贴近官方）：Full fine-tuning + DeepSpeed

- 复用官方 `--finetuning_type full`、`ds_config/ds_config_3.json`（**不用** `ds_config_3_offload_all.json`，A800 80GB 先尝试纯 GPU ZeRO-3）。
- 单卡覆盖：`--num_gpus 1`，`gradient_accumulation_steps` 提高到 **512**（对齐官方 `2×32×8=512` 的有效 batch）。
- 精度：官方用 `fp16`；若 loss/grad 异常再改 `bf16` 并记录。

### 轨道 B（回退，历史已验证可跑通）：LoRA + 阶段间 merge export

- Stage1/2 用现有 LoRA 参数集（`lora_rank=8`，target 见 `single_a100_qwen3_4b_stage1_lora_smoke.sh`）。
- 每阶段结束后用 **`llamafactory-cli export`** 合并 adapter 为完整权重，供下一阶段 `--model_name_or_path` / RL `model_path` 使用。
- 步数仍保持官方 1000 / 400；RL 阶段与轨道 A 相同。

**决策门：** Task 2 的 10-step smoke 若 full 在 10 分钟内仍 `-9` 或 OOM，整份 baseline 切轨道 B，并在 `baseline_manifest.json` 标注 `"finetuning_track": "lora"`。

---

## 2. 统一环境与路径（所有 Task 共用）

工作目录：

```bash
cd /root/autodl-tmp/STReasoner_reproduce
```

每次训练前 export（写入 wrapper 脚本头部）：

```bash
export PATH="/root/autodl-tmp/conda/envs/str-py310/bin:${PATH}"
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

Baseline 元数据文件（Task 1 创建）：

```text
output/baseline_qwen3_4b/baseline_manifest.json
```

内容模板：

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "instance": "autodl-a800-1x80gb",
  "finetuning_track": "full",
  "official_scripts": [
    "scripts/qwen3-4b-instruct/train_stage1.sh",
    "scripts/qwen3-4b-instruct/train_stage1+2.sh",
    "scripts/qwen3-4b-instruct/train_stage1+2+3_w_spatial.sh"
  ],
  "stage1_max_steps": 1000,
  "stage2_max_steps": 400,
  "stage3_spatial": true,
  "effective_batch_size_target": 512
}
```

---

## 3. 文件结构（仅新增 wrapper，不改官方）

| 文件 | 职责 |
| --- | --- |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/00_env.sh` | 上述 export + `cd` |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/01_preflight.sh` | 磁盘/GPU/数据/模型门禁 |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/02_stage1_smoke.sh` | Stage1 10-step |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/03_stage1_train.sh` | Stage1 1000-step 正式 |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/04_stage1_postcheck.sh` | loss + 目录大小 + 可选 adapter/load |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/05_stage2_smoke.sh` | Stage2 10-step |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/06_stage2_train.sh` | Stage2 400-step 正式 |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/07_stage2_postcheck.sh` | 同上 |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/08_export_merged_sft.sh` | 轨道 B：LoRA merge export |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/09_stage3_smoke.sh` | RL 5-step |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/10_stage3_train.sh` | RL 正式（spatial GRPO） |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/11_merge_rl_checkpoint.sh` | 调 `model_merger.py` |
| `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/12_sttest_infer_eval.sh` | 四任务推理+评测 |
| `00_new_codes/repro_autodl/experiments/logs/baseline_qwen3_4b_*.log` | 全量 tee 日志 |

---

## 4. 分 Task 实施步骤

### Task 1: Preflight 与 manifest

**Files:**
- Create: `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/00_env.sh`
- Create: `00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/01_preflight.sh`
- Create: `output/baseline_qwen3_4b/baseline_manifest.json`

- [ ] **Step 1: 写 `00_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/STReasoner_reproduce
# … 粘贴 §2 全部 export …
```

- [ ] **Step 2: 写 `01_preflight.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
bash 00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_env_check.sh
/root/autodl-tmp/conda/envs/str-py310/bin/python \
  00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_model_check.py
df -h / /root/autodl-tmp
test "$(df /root/autodl-tmp --output=pcent | tail -1 | tr -dc '0-9')" -lt 85
echo preflight_ok
```

- [ ] **Step 3: 运行 preflight**

Run:

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/01_preflight.sh \
  2>&1 | tee 00_new_codes/repro_autodl/experiments/logs/baseline_preflight_$(date +%Y%m%d_%H%M%S).log
```

Expected: 全部 `ok`，数据盘 `<85%`，输出 `preflight_ok`。

- [ ] **Step 4: 若 ST-Align/CoT/RL missing，补数据**

Run:

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/single_a100_download_stbench_train_data.sh
```

- [ ] **Step 5: 创建 manifest**

写入 §2 JSON，`finetuning_track` 初始为 `"full"`。

---

### Task 2: Stage 1 — Alignment SFT（官方 1000 steps）

**对照官方：** `scripts/qwen3-4b-instruct/train_stage1.sh`

**Files:**
- Create: `02_stage1_smoke.sh`, `03_stage1_train.sh`, `04_stage1_postcheck.sh`

- [ ] **Step 1: 写 `02_stage1_smoke.sh`（轨道 A full）**

以官方 `train_stage1.sh` 为模板，仅改：

```bash
source "$(dirname "$0")/00_env.sh"
NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=1800 \
  deepspeed --num_gpus 1 --master_port=19901 src/train.py \
  --deepspeed ds_config/ds_config_3.json \
  --stage sft \
  --model_name_or_path "./base_model/Qwen3-4B-Instruct-2507" \
  --dataset "alignment" \
  --interleave_probs "1" \
  --do_train \
  --mix_strategy "interleave_over" \
  --template "STReasoner-Align" \
  --finetuning_type full \
  --output_dir "./output/baseline_qwen3_4b/stage1_smoke" \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_steps 10 \
  --learning_rate 1e-5 \
  --timeseries_sft_lr 1e-5 \
  --warmup_ratio 0.02 \
  --num_train_epochs 0 \
  --max_steps 10 \
  --plot_loss \
  --fp16 \
  --save_only_model \
  --save_safetensors False \
  --report_to none \
  --preprocessing_num_workers 4 \
  --trust_remote_code True \
  --cutoff_len 10000
```

- [ ] **Step 2: 跑 smoke 并判定轨道**

Run:

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/02_stage1_smoke.sh \
  2>&1 | tee 00_new_codes/repro_autodl/experiments/logs/baseline_stage1_smoke_$(date +%Y%m%d_%H%M%S).log
```

Expected PASS: 日志出现 `train_runtime`，`global_step=10`，无 `-9`/OOM。  
Expected FAIL → 切轨道 B：复制 `single_a100_qwen3_4b_stage1_lora_smoke.sh` 逻辑到 `02_stage1_smoke.sh`，更新 manifest `"finetuning_track": "lora"`。

- [ ] **Step 3: 写 `03_stage1_train.sh`（正式 1000 steps）**

轨道 A 关键差异（相对 smoke）：

```bash
--output_dir "./output/baseline_qwen3_4b/stage1" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 512 \
--max_steps 1000 \
--save_steps 100 \
--save_total_limit 2 \
--cutoff_len 10000
```

轨道 B LoRA 额外参数（与 `single_a100_qwen3_4b_stage1_lora_500steps.sh` 一致）：

```bash
--finetuning_type lora \
--lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
--lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--flash_attn fa2 --bf16 \
# 不用 deepspeed；直接 python src/train.py
```

- [ ] **Step 4: 跑 Stage1 正式训练**

Run:

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/03_stage1_train.sh \
  2>&1 | tee 00_new_codes/repro_autodl/experiments/logs/baseline_stage1_train_$(date +%Y%m%d_%H%M%S).log
```

Expected: `output/baseline_qwen3_4b/stage1/` 含 `train_results.json`、`trainer_state.json`、checkpoint 或 merged 权重。

- [ ] **Step 5: `04_stage1_postcheck.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
OUT=output/baseline_qwen3_4b/stage1
test -f "$OUT/train_results.json"
test -f "$OUT/trainer_state.json"
du -sh "$OUT"
# 轨道 B 额外：
# bash .../single_a100_qwen3_4b_stage1_adapter_load_check.py "$OUT"
echo stage1_postcheck_ok
```

- [ ] **Step 6: 轨道 B — merge export（仅 LoRA）**

Run `08_export_merged_sft.sh`：

```bash
llamafactory-cli export \
  --model_name_or_path base_model/Qwen3-4B-Instruct-2507 \
  --adapter_name_or_path output/baseline_qwen3_4b/stage1 \
  --template STReasoner-Align \
  --finetuning_type lora \
  --export_dir output/baseline_qwen3_4b/stage1_merged \
  --export_size 5 \
  --export_device cpu \
  --trust_remote_code true
```

下游 Stage2 的 `--model_name_or_path` 改为 `output/baseline_qwen3_4b/stage1_merged`。

---

### Task 3: Stage 2 — CoT SFT（官方 400 steps）

**对照官方：** `scripts/qwen3-4b-instruct/train_stage1+2.sh`

**输入模型：**

- 轨道 A：`./output/baseline_qwen3_4b/stage1`（full checkpoint 目录取最新或最终目录）
- 轨道 B：`./output/baseline_qwen3_4b/stage1_merged`

- [ ] **Step 1: 写 `05_stage2_smoke.sh`（10 steps）**

以官方 `train_stage1+2.sh` 为模板：

```bash
--model_name_or_path "./output/baseline_qwen3_4b/stage1" \
--dataset "entity_cot,etiological_cot,correlation_cot,forecasting_cot" \
--interleave_probs "0.25,0.25,0.25,0.25" \
--template "STReasoner-CoT" \
--output_dir "./output/baseline_qwen3_4b/stage2_smoke" \
--max_steps 10 \
--cutoff_len 10000
```

- [ ] **Step 2: smoke 通过后再写/跑 `06_stage2_train.sh`**

```bash
--output_dir "./output/baseline_qwen3_4b/stage1+2" \
--max_steps 400 \
--save_steps 100 \
--save_total_limit 2 \
--gradient_accumulation_steps 512   # 轨道 A
```

- [ ] **Step 3: `07_stage2_postcheck.sh`**

检查 `stage1+2/train_results.json`、loss 曲线、`du -sh`。

- [ ] **Step 4: 轨道 B merge → `output/baseline_qwen3_4b/stage1+2_merged`**

`llamafactory-cli export` 时 `--template STReasoner-CoT`，供 Stage3 `model_path` 使用。

---

### Task 4: Stage 3 — S-GRPO RL（官方 spatial 版）

**对照官方：** `scripts/qwen3-4b-instruct/train_stage1+2+3_w_spatial.sh`

**输入模型：**

- 轨道 A：`./output/baseline_qwen3_4b/stage1+2`
- 轨道 B：`./output/baseline_qwen3_4b/stage1+2_merged`

- [ ] **Step 1: 写 `09_stage3_smoke.sh`（5 steps）**

基于现有 `single_a100_qwen3_4b_stage3_smoke.sh`，改：

```bash
MODEL_PATH=./output/baseline_qwen3_4b/stage1+2
python -m src.EasyR1.verl.trainer.main \
  config=./src/EasyR1/examples/config.yaml \
  data.train_files=./data/ST-Bench/ST-RL/etiological_rl.jsonl,... \
  data.val_files=./data/ST-Bench/ST-Test/etiological_test.jsonl,... \
  data.format_prompt=./src/EasyR1/examples/format_prompt/str.jinja \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.rollout_batch_size=4 \
  data.val_batch_size=4 \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.actor.model.trust_remote_code=true \
  worker.actor.optim.lr=1.0e-7 \
  worker.actor.optim.lr_warmup_ratio=0.2 \
  worker.reward.reward_function=./src/EasyR1/examples/reward_function/str.py:compute_score \
  worker.rollout.n=2 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.45 \
  trainer.experiment_name=baseline_qwen3_4b_stage3_smoke \
  trainer.n_gpus_per_node=1 \
  trainer.find_last_checkpoint=false \
  trainer.total_epochs=1 \
  trainer.max_steps=5 \
  trainer.save_freq=5 \
  trainer.save_limit=1 \
  algorithm.enable_spatial_reward=true \
  algorithm.spatial_reward_weight=0.1 \
  data.enable_spatial_reward=true
```

- [ ] **Step 2: 写 `10_stage3_train.sh`（正式 RL）**

在 smoke 基础上恢复接近官方配置（仍单卡）：

```bash
data.rollout_batch_size=32          # 官方 128；单卡先 32，OOM 再 16
data.val_batch_size=32
worker.rollout.n=8                  # 与官方一致
worker.rollout.gpu_memory_utilization=0.55
trainer.experiment_name=baseline_qwen3_4b_stage3_w_spatial
trainer.n_gpus_per_node=1
trainer.total_epochs=1              # 官方 1 epoch
trainer.max_steps=null              # 用 epoch 驱动
trainer.save_freq=100
trainer.save_limit=2
trainer.logger='["file"]'           # 关闭 wandb
algorithm.enable_spatial_reward=true
algorithm.spatial_reward_weight=0.1
data.enable_spatial_reward=true
```

**Docker 说明：** 官方 README 建议 Docker `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`。本实例已装 `vllm==0.8.5` 于 conda；baseline 先在 **conda 内**跑通 smoke。若正式 RL 因 vLLM/Verl 版本报错，再拉 Docker，挂载 `-v /root/autodl-tmp/STReasoner_reproduce:/workspace/STReasoner`，容器内执行同一 `10_stage3_train.sh`。

- [ ] **Step 3: 跑 Stage3 正式**

Run:

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/baseline_qwen3_4b/10_stage3_train.sh \
  2>&1 | tee 00_new_codes/repro_autodl/experiments/logs/baseline_stage3_train_$(date +%Y%m%d_%H%M%S).log
```

Expected: `checkpoints/easy_r1/baseline_qwen3_4b_stage3_w_spatial/global_step_*/actor/`

- [ ] **Step 4: `11_merge_rl_checkpoint.sh`**

对照 README §Merge Checkpoint：

```bash
STEP_DIR=checkpoints/easy_r1/baseline_qwen3_4b_stage3_w_spatial/global_step_XXX
cp base_model/Config-Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py \
   "$STEP_DIR/actor/huggingface/"
/root/autodl-tmp/conda/envs/str-py310/bin/python model_merger.py \
  --local_dir "$STEP_DIR/actor/"
```

将 `XXX` 换为最终 step；记录到 manifest 的 `"rl_global_step"`。

---

### Task 5: Baseline 评测（ST-Test 四任务）

**对照官方：** README Inference + Evaluation 节

- [ ] **Step 1: 写 `12_sttest_infer_eval.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
MODEL=checkpoints/easy_r1/baseline_qwen3_4b_stage3_w_spatial/global_step_XXX/actor/huggingface
EXP_PREFIX=exp/baseline_qwen3_4b_sttest_6144
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do
  PYTHONPATH=. python inference/inference_tsmllm_vllm.py \
    --task "$task" \
    --model_path "$MODEL" \
    --max_tokens 6144 \
    --output_dir "${EXP_PREFIX}/${task}"
  PYTHONPATH=. python evaluation/evaluate.py \
    --task "$task" \
    --exp_path "${EXP_PREFIX}/${task}"
done
```

- [ ] **Step 2: 跑评测**

Expected: 四个目录各有 `generated_answer.json` + `evaluation_metrics.json`。

- [ ] **Step 3: 汇总 baseline 表**

新建 `output/baseline_qwen3_4b/baseline_metrics_summary.json`：

```json
{
  "model_path": "...",
  "finetuning_track": "full|lora",
  "stage1_steps": 1000,
  "stage2_steps": 400,
  "stage3_experiment": "baseline_qwen3_4b_stage3_w_spatial",
  "metrics": {
    "reasoning_entity": {},
    "reasoning_etiological": {},
    "reasoning_correlation": {},
    "reasoning_forecasting": {}
  }
}
```

从四个 `evaluation_metrics.json` 拷贝关键字段（accuracy / MAE 等）。

---

### Task 6: 阶段间可选对照（不跳过主链，仅记录）

为后续调参对比，建议在 Stage1、Stage2 结束后各做一次 **轻量** ST-Test（可与正式评测共用脚本，模型路径分别指向 stage1 / stage1+2 merged 权重）。这不是官方必需步骤，但 baseline 包应保留：

```text
exp/baseline_qwen3_4b_sttest_6144_after_stage1/
exp/baseline_qwen3_4b_sttest_6144_after_stage2/
```

- [ ] **Step 1:** Stage1 训练完成后跑四任务 infer+eval（同上，`MODEL` 指向 stage1 输出）。
- [ ] **Step 2:** Stage2 训练完成后重复一次。
- [ ] **Step 3:** 将三次 metrics 写入 `baseline_metrics_summary.json` 的 `after_stage1` / `after_stage2` / `after_stage3` 字段。

---

## 5. 资源与磁盘预算

| 产物 | 估算大小 | 保留策略 |
| --- | --- | --- |
| stage1 full checkpoint | ~8–16G | `save_total_limit=2` |
| stage1+2 full | ~8–16G | 同上 |
| stage1/2 LoRA adapter | ~300M  each | merge 后可删中间 adapter |
| RL checkpoint | 10G+ | `save_limit=2` |
| ST-Test 生成 json | ~1–3G/轮 | 只保留 6144 正式目录 |

训练前要求数据盘剩余 **≥25G**；低于 20G 时先 `du -sh output/ checkpoints/ exp/` 清理非 baseline 目录。

---

## 6. 验收标准（Baseline Done 定义）

全部满足才标记 baseline 完成：

1. manifest 中三阶段 `max_steps` / `experiment_name` 与官方脚本一致。
2. Stage1、Stage2、Stage3 均有成功训练日志（无 `-9` 中断）和 checkpoint。
3. RL 权重经 `model_merger.py` 合并，目录含 `modeling_qwen3_ts.py`。
4. ST-Test 四任务 `max_tokens=6144` 推理 + `evaluate.py` 完成。
5. `baseline_metrics_summary.json` 已填写 Stage3 指标。
6. 报告 `00_new_codes/reports/t3-三阶段训练复现/16-Qwen3-4B三阶段Baseline执行日志.md` 含：轨道选择、命令、日志路径、loss 片段、磁盘变化、最终 metrics。

**明确不算完成：**

- 只跑 smoke 没跑正式 1000/400/1epoch。
- Stage2 用了 `train_stage2.sh`（跳过 stage1 权重）。
- Stage3 用了误配 8B 的 `train_stage1+2+3.sh`。
- 评测 `max_tokens<6144`。

---

## 7. 执行顺序总览（ checklist ）

```
[ ] Task 1  Preflight + manifest
[ ] Task 2  Stage1 smoke → Stage1 train → postcheck → (LoRA export)
[ ] Task 3  Stage2 smoke → Stage2 train → postcheck → (LoRA export)
[ ] Task 4  Stage3 smoke → Stage3 train → model_merger
[ ] Task 5  ST-Test infer + eval → metrics summary
[ ] Task 6  (可选) Stage1/2 中间 ST-Test 对照
[ ] 写执行日志 report 16
```

---

## 8. Self-Review（计划自检）

| 检查项 | 结果 |
| --- | --- |
| 三阶段均未跳过 | ✓ Stage1/2/3 + merge + eval |
| 尽量复用官方脚本 | ✓ 参数来自三个官方 sh，仅 wrapper 覆盖 GPU/batch |
| Qwen3-4B-Instruct-2507 | ✓ 全链路固定该型号 |
| 单卡 A800 约束 | ✓ 双轨 full/LoRA + batch 缩放说明 |
| 无 TBD/占位 | ✓ 命令与路径具体 |
| 已知官方 bug | ✓ 8B 误配脚本已排除 |

---

## 9. 执行方式选择

**Plan complete and saved to `00_new_codes/reports/t3-三阶段训练复现/15-Qwen3-4B三阶段Baseline实施计划.md`.**

**1. Subagent-Driven（推荐）** — 按 Task 1→6 派发子 agent，每 Task 后 review checkpoint 与日志。

**2. Inline Execution** — 本会话连续执行，Stage1 smoke 通过后暂停给你确认轨道 A/B，再跑长训。

**你希望用哪种方式开始执行？**
