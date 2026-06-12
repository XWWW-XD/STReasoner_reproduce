# AutoDL A800 环境配置与下载加速

> 日期：2026-06-12  
> 实例：AutoDL，`NVIDIA A800 80GB PCIe`，数据盘 `/root/autodl-tmp` 100G  
> 分支：`autodl-A800`  
> 目的：通读代码库后完成 SFT 环境配置，解决 AutoDL 默认下载慢的问题，并留下可复用脚本。

---

## 1. 结论

**Python 环境与核心依赖已就绪**，可在数据盘直接跑推理/训练 smoke：

| 组件 | 版本 | 状态 |
| --- | --- | --- |
| Python | 3.10.20 | `/root/autodl-tmp/conda/envs/str-py310` |
| torch | 2.6.0+cu124 | CUDA 可用 |
| flash-attn | 2.7.2.post1 | 本地 wheel 安装 |
| transformers | 4.52.4 | OK |
| vllm | 0.8.5 | OK |
| deepspeed | 0.16.4 | OK |
| bitsandbytes | **0.45.2** | 已从 requirements 0.43.1 升级（见 §5） |

**尚未就绪（需另行下载，不占 git）：**

- 训练集：`ST-Align` / `ST-CoT` / `ST-RL`（`env_check` 报 missing）
- 模型权重：`base_model/Qwen3-4B-Instruct-2507`、`STReasoner-8B` 等（仅 Config 在仓库内）
- ST-Test 四类测试 jsonl **已在仓库**（可先做评测链路检查）

一键重装脚本：`00_new_codes/repro_autodl/setup_str_env.sh`

---

## 2. 代码库通读摘要

本仓库 = **官方 STReasoner 实现** + **`00_new_codes` 复现层**（详见 `00_new_codes/reports/t0-阅读材料/01-pipeline数据通路.md`）。

### 2.1 官方三阶段 pipeline

| 阶段 | 方式 | 入口 |
| --- | --- | --- |
| Stage 1 | 对齐 SFT | `scripts/qwen3-8b/train_stage1.sh` → `src/train.py` |
| Stage 2 | CoT SFT | `scripts/qwen3-8b/train_stage1+2.sh` |
| Stage 3 | S-GRPO RL | Docker + `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh` |

### 2.2 环境与缓存（必读）

- `requirements.txt`：SFT 依赖（torch 2.6、vllm 0.8.5、deepspeed、flash-attn wheel）
- `cache_config.py`：AutoDL 下 HF 缓存固定为 `/root/autodl-tmp/cache/huggingface`
- `download_model.py` / `download_dataset.py`：HF Hub 拉模型与 ST-Bench
- 推理：`inference/inference_tsmllm_vllm.py` + `evaluation/evaluate.py`
- RL 阶段需 Docker `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`（本次未配置）

### 2.3 本项目约定

- **不要用** `/root/miniconda3/bin/python` 跑正式实验
- 训练脚本 PATH 需包含 conda env `bin`（否则 `ninja` 找不到）
- 大文件、缓存、conda env 一律放 **数据盘** `/root/autodl-tmp`

---

## 3. 下载慢的原因与对策

AutoDL 本区 **学术加速未开通**（`/etc/network_turbo` 仅提示用第三方镜像）。慢主要来自：

| 来源 | 问题 | 对策 |
| --- | --- | --- |
| pip 默认源 | 国外 PyPI 慢 | **清华源** + 缓存到数据盘 |
| PyTorch | 体积大（~2GB+） | 官方 `cu124` 索引直连（本实例实测 ~60MB/s，可接受） |
| flash-attn | GitHub Release wheel | **ghproxy** 预下载到 `cache/wheels/` 再本地 pip |
| HuggingFace | 模型/数据集 | `HF_ENDPOINT=https://hf-mirror.com` |
| conda | 基础包 | 已有 `~/.condarc` 清华 anaconda 镜像 |
| 系统盘 | 30G 易满 | conda env、pip cache、HF cache、torch extensions 全部放数据盘 |

### 3.1 已写入的配置

**`/root/.pip/pip.conf`**

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
cache-dir = /root/autodl-tmp/cache/pip
```

**`/root/.bashrc`（新开 terminal 自动生效）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/huggingface/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch_extensions
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export PATH="/root/autodl-tmp/conda/envs/str-py310/bin:$PATH"
```

### 3.2 推荐安装顺序（避免重复踩坑）

```bash
# 1. conda 环境（数据盘）
conda create -p /root/autodl-tmp/conda/envs/str-py310 python=3.10 -y

# 2. torch（单独装，用官方 CUDA wheel 索引）
/root/autodl-tmp/conda/envs/str-py310/bin/pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. flash-attn（ghproxy 下载 wheel 后本地装）
wget -O /root/autodl-tmp/cache/wheels/flash_attn-....whl \
  "https://ghproxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/..."
/root/autodl-tmp/conda/envs/str-py310/bin/pip install /root/autodl-tmp/cache/wheels/flash_attn-....whl

# 4. 其余 requirements（跳过 torch 与 flash 行）
grep -vE '^(torch==|https://github.com/Dao-AILab|#)' requirements.txt | grep -v '^$' \
  > /tmp/str_requirements_rest.txt
/root/autodl-tmp/conda/envs/str-py310/bin/pip install -r /tmp/str_requirements_rest.txt

# 5. bitsandbytes 兼容 triton 3.2
/root/autodl-tmp/conda/envs/str-py310/bin/pip install bitsandbytes==0.45.2
```

或直接：`bash 00_new_codes/repro_autodl/setup_str_env.sh`

---

## 4. 验证记录

### 4.1 包导入

```
python 3.10.20
torch 2.6.0+cu124 cuda True 12.4
flash_attn 2.7.2.post1
transformers 4.52.4
vllm 0.8.5
deepspeed 0.16.4
bitsandbytes 0.45.2
HF_HOME /root/autodl-tmp/cache/huggingface
```

### 4.2 项目 env_check

```bash
bash 00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_env_check.sh
```

- GPU / Python / Config：通过
- ST-Test 四类：ok
- ST-Align / CoT / RL：missing（未下载完整 ST-Bench）
- Qwen3-4B 权重：missing

### 4.3 磁盘占用（配置后）

| 路径 | 大小 |
| --- | --- |
| `/root/autodl-tmp/conda/envs/str-py310` | ~8.9G |
| `/root/autodl-tmp/cache` | ~182M（wheel + 小缓存） |
| 系统盘 `/` | 20%（5.9G/30G） |
| 数据盘 `/root/autodl-tmp` | 10%（9.3G/100G） |

---

## 5. 已知偏离 requirements 的修正

| 项 | requirements | 实际 | 原因 |
| --- | --- | --- | --- |
| bitsandbytes | 0.43.1 | **0.45.2** | 0.43.1 + triton 3.2.0 → `No module named 'triton.ops'`（见 t3 报告 §8.1） |

其余版本与 `requirements.txt` 一致。

---

## 6. 下一步：数据与模型下载（HF 镜像）

```bash
cd /root/autodl-tmp/STReasoner_reproduce
export HF_ENDPOINT=https://hf-mirror.com

# 完整 ST-Bench（含训练集）
/root/autodl-tmp/conda/envs/str-py310/bin/python download_dataset.py

# 基座 / 复现用模型（按需）
/root/autodl-tmp/conda/envs/str-py310/bin/python download_model.py --repo_id Qwen/Qwen3-4B-Instruct-2507
cp -rf base_model/Config-Qwen3-4B-Instruct-2507/* base_model/Qwen3-4B-Instruct-2507/
/root/autodl-tmp/conda/envs/str-py310/bin/python initial_model.py --model_path base_model/Qwen3-4B-Instruct-2507
```

下载完成后重跑 `single_a100_qwen3_4b_env_check.sh` 确认全部 ok。

---

## 7. 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-06-12 | 初稿：环境配置、镜像加速、验证与脚本 |
