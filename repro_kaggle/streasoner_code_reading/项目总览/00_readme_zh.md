# 00 README 中文版

> 本文是仓库根目录 `README.md` 的中文翻译版，便于阅读和组会准备。命令、路径、模型名、数据集名和链接保持原样。

<div align="center">

# (ACL 2026 Main) STReasoner：面向时空推理的时间序列 LLM

**通过空间感知强化学习，增强 LLM 在时间序列中的时空推理能力**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ST--Bench-orange)](https://huggingface.co/datasets/Time-HD-Anonymous/ST-Bench)
[![Model](https://img.shields.io/badge/🤗%20Model-STReasoner--8B-purple)](https://huggingface.co/Time-HD-Anonymous/STReasoner-8B)
[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv:2601.03248-red)](https://arxiv.org/pdf/2601.03248)

<img src="../../figures/streasoner.png" width="100">

</div>

---

## ✨ 亮点

- 📊 **多模态基准**：ST-Bench，用于时间序列中的时空推理任务。

- 🔗 **集成时间序列编码器**：在 LLM 中加入专用的 time series encoder。  
  *支持：Qwen3-8B、Qwen3-4B-Instruct、Qwen2.5-14B-Instruct*

- 🚀 **完整训练流程**：首个面向带时间序列编码器 LLM 的 SFT + RL 完整训练 pipeline。

---

## 📰 新闻

- **[2026/04/06]** STReasoner 已被 ACL 2026 Main 接收。
- **[2026/02/19]** 现在支持 Qwen3-4B、Qwen3-8B 和 Qwen3-14B，并在评估中加入 MAPE 指标。
- **[2026/01/06]** 发布 STReasoner 的完整 pipeline 训练代码。

---

## 📖 概览

<img src="../../figures/method.png" width="800">

**STReasoner** 是一个面向时间序列时空推理的框架，通过精心设计的 **三阶段 pipeline** 进行训练：

| 阶段 | 方法 | 说明 |
|:---:|:---:|:---|
| 1 | SFT | 用于时间序列对齐的监督微调 |
| 2 | SFT | 用于 reasoning cold start 的监督微调 |
| 3 | RL | 使用 **S-GRPO** 的强化学习，即 Spatial-aware Group Relative Policy Optimization |

---

## ⚙️ 环境要求

**硬件：** 8 × NVIDIA A100-SXM4-80GB，或同等级硬件。

**软件：** CUDA 12.8。

### SFT 环境：Stage 1 和 Stage 2

```bash
conda create --name str python==3.10
conda activate str
pip install -r requirements.txt
```

### RL 环境：Stage 3

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

---

## 📦 数据准备

从 [🤗 HuggingFace](https://huggingface.co/datasets/Time-HD-Anonymous/ST-Bench) 下载 **ST-Bench** 数据集：

```bash
python download_dataset.py
```

---

## 🚀 训练

### 1. 准备基础模型

```bash
python download_model.py --repo_id Qwen/Qwen3-8B
cp -rf base_model/Config-Qwen3-8B/* base_model/Qwen3-8B/
python initial_model.py --model_path base_model/Qwen3-8B
```

### 2. Stage 1 & 2：SFT

```bash
bash scripts/qwen3-8b/train_stage1.sh      # -> STReasoner-8B-Align
bash scripts/qwen3-8b/train_stage1+2.sh    # -> STReasoner-8B-CoT
```

> 📦 **SFT Checkpoints：**  
> [STReasoner-8B-Align](https://huggingface.co/Time-HD-Anonymous/STReasoner-8B-Align) · [STReasoner-8B-CoT](https://huggingface.co/Time-HD-Anonymous/STReasoner-8B-CoT)

### 3. Stage 3：RL

启动 Docker 容器：

```bash
docker run -it --gpus all \
  --name verl_env \
  --shm-size=40g \
  -v .:/workspace/STReasoner \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 bash
```

进入容器后：

```bash
cd STReasoner

# 使用 Spatial-aware GRPO
bash scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh

# 或者使用 vanilla GRPO
bash scripts/qwen3-8b/train_stage1+2+3.sh
```

### 4. 合并 checkpoint

```bash
cp base_model/Config-Qwen3-8B/modeling_qwen3_ts.py \
   checkpoints/easy_r1/qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/huggingface

python model_merger.py \
   --local_dir checkpoints/easy_r1/qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/
```

---

## 🔮 推理

在所有 reasoning 任务上运行推理。

先退出 Docker 容器，然后激活 `str` 环境：

```bash
conda activate str
```

然后运行：

```bash
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do 
    python inference/inference_tsmllm_vllm.py \
        --task $task \
        --model_path checkpoints/easy_r1/qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/huggingface
done
```

---

## 📊 评估

在每个任务上评估模型表现：

```bash
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do 
    python evaluation/evaluate.py \
        --task $task \
        --exp_path exp/$task-qwen3_8b_grpo_stage1+2+3_w_spatial
done
```

---

## 🎨 替代训练方式：文本或图像 Prompting

| 模态 | Stage 2 脚本 | Stage 2+3 脚本 |
|:---:|:---|:---|
| **Text** | `scripts/qwen3-8b/train_stage2_only_text.sh` | `scripts/qwen3-8b/train_stage2+3_w_spatial_only_text.sh` |
| **Image** | `scripts/qwen3-vl-8b-instruct/train_stage2_only_image.sh` | `scripts/qwen3-vl-8b-instruct/train_stage2+3_w_spatial_only_image.sh` |

---

## 🙏 致谢

感谢以下项目的重要贡献：

- [EasyR1](https://github.com/hiyouga/EasyR1)：为 RL 训练配置提供强化学习框架。
- [Verl](https://github.com/volcengine/verl)：为 Stage 3 训练提供强化学习框架和环境。
- [ChatTS](https://github.com/NetManAIOps/ChatTS)：提供 temporal-spatial encoder 以及 HuggingFace / vLLM 实现参考。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：为 SFT 阶段提供监督微调框架。
- [vLLM](https://github.com/vllm-project/vllm)：快速模型推理引擎。

---

<div align="center">

**如果你觉得这个工作有帮助，可以考虑给它一个 star。**

</div>
