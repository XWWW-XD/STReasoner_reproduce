# 15 环境与运行风险

本文只做静态环境分析，不运行训练、推理、DeepSpeed、RL，也不下载模型或数据。结论分为：

- `已从代码确认`：来自 README、requirements、脚本或配置文件。
- `根据代码推断，未由真实运行验证`：来自脚本参数和常见运行行为，但本地没有实际跑。
- `尚未确认`：缺少真实运行、真实硬件或完整数据/模型验证。

## 1. 依赖总表

| 依赖 | 已从代码确认的位置 | 版本 / 用途 | 风险等级 |
|---|---|---|---|
| Python | `README.md:7`, `README.md:62-64`; `src/EasyR1/README.md:41` | SFT README 明确 Python 3.10；EasyR1 要求 Python 3.9+ | 中 |
| CUDA | `README.md:8`, `README.md:56-58`; `README.md:71`; `src/EasyR1/README.md:51-52` | 主 README 标 CUDA 12.8；RL Docker 镜像名是 `cu12.9-vllm0.11.0` | 高 |
| GPU | `README.md:56`; `scripts/qwen3-8b/train_stage1.sh:1`; `scripts/qwen3-8b/train_stage1+2+3.sh:24` | README 推荐 8 x NVIDIA A100-SXM4-80GB；SFT 和 RL 脚本都按 8 卡写 | 高 |
| torch | `requirements.txt:9`; `base_model/Config-Qwen3-8B/config.json:71` | 主环境固定 `torch==2.6.0`，模型配置为 `float16` | 高 |
| flash-attn | `requirements.txt:15`; `src/EasyR1/README.md:43` | 主环境用 `flash_attn-2.7.2.post1+cu12torch2.6...cp310` wheel；EasyR1 要求 `flash-attn>=2.4.3` | 高 |
| transformers | `requirements.txt:11`; `base_model/Config-Qwen3-8B/config.json:72`; `src/EasyR1/requirements.txt:18` | 主环境固定 `4.52.4`；EasyR1 侧要求 `>=4.54.0,<=4.57.0` | 高 |
| vLLM | `requirements.txt:13`; `inference/inference_tsmllm_vllm.py:15`; `README.md:71`; `src/EasyR1/requirements.txt:19` | 推理脚本注释要求 `vllm==0.8.5`；RL Docker 镜像名是 `vllm0.11.0`；EasyR1 要求 `vllm>=0.8.0` | 高 |
| DeepSpeed | `requirements.txt:14`; `scripts/qwen3-8b/train_stage1.sh:1-2`; `ds_config/ds_config_3.json:1-24` | 主环境固定 `deepspeed==0.16.4`；SFT 脚本用 ZeRO-3 配置 | 高 |
| LLaMA-Factory | `src/train.py:15`; `src/llamafactory/extras/env.py:30`; `src/llamafactory/extras/misc.py:95-101` | SFT 入口调用内置 LLaMA-Factory，版本字符串 `0.9.4.dev0`，会检查 datasets / accelerate / peft 版本 | 中 |
| EasyR1 / verl | `src/EasyR1/README.md:1-13`; `src/EasyR1/setup.py:54-55`; `scripts/qwen3-8b/train_stage1+2+3.sh:7-8` | Stage 3 通过 `python3 -m src.EasyR1.verl.trainer.main` 运行，EasyR1 setup 要求 Python >=3.9 | 高 |
| Ray | `src/EasyR1/requirements.txt:15`; `src/EasyR1/verl/trainer/main.py:17-32`; `src/EasyR1/README.md:127-139` | RL 训练依赖 Ray worker / resource pool / 多节点接口 | 高 |
| datasets / pandas / peft / trl | `requirements.txt:2`, `requirements.txt:7-8`, `requirements.txt:12`; `src/llamafactory/extras/misc.py:95-101` | 数据加载、SFT、PEFT/TRL 相关依赖 | 中 |
| wandb / omegaconf | `requirements.txt:22-23`; `src/EasyR1/examples/config.yaml:87-105` | RL config 使用 OmegaConf 风格覆盖；默认 logger 包含 `wandb` | 中 |
| HuggingFace Hub | `requirements.txt:4`; `download_dataset.py:17-21`; `download_model.py:12` | 数据和模型下载依赖 HuggingFace Hub | 高 |

`已从代码确认`：主仓库没有单一统一环境文件能同时覆盖 SFT、vLLM 推理和 RL。SFT 主环境来自 `requirements.txt`，RL 推荐直接用 Docker 镜像，见 `README.md:68-71`、`README.md:111-115`。

`已从代码确认`：本地静态搜索没有定位到 `Dockerfile`；但 `src/EasyR1/README.md:46` 提到 EasyR1 上游提供 Dockerfile，当前仓库主线使用的是预构建镜像 `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`，见 `README.md:71`。

## 2. 高风险点

### CUDA / torch / flash-attn 版本耦合

`已从代码确认`：主环境固定 `torch==2.6.0`，并直接安装一个针对 `cu12 + torch2.6 + cp310 + linux_x86_64` 的 flash-attn wheel，见 `requirements.txt:9`、`requirements.txt:15`。

`风险解释`：这个 wheel 对 Python 版本、CUDA ABI、PyTorch 版本和操作系统都很敏感。README 说用 Python 3.10 和 CUDA 12.8，见 `README.md:7-8`、`README.md:58`；如果本地是 Windows、CUDA 版本不同、Python 不是 3.10，直接 `pip install -r requirements.txt` 很容易在 flash-attn 或 torch CUDA 扩展处失败。

### GPU 显存和卡数

`已从代码确认`：README 明确推荐 `8 x NVIDIA A100-SXM4-80GB`，见 `README.md:56`。Qwen3-8B SFT 脚本写死 `deepspeed --num_gpus 8`，见 `scripts/qwen3-8b/train_stage1.sh:1`。RL 脚本写 `trainer.n_gpus_per_node=8`，见 `scripts/qwen3-8b/train_stage1+2+3.sh:24`。

`根据代码推断，未由真实运行验证`：在非 8 卡环境上直接跑官方脚本，大概率需要改 `--num_gpus`、DeepSpeed batch、RL `trainer.n_gpus_per_node`、rollout batch 和 tensor parallel 参数，否则会失败或 OOM。

### vLLM 自定义 patch / 注册路径

`已从代码确认`：推理脚本注释写明需要 `vllm==0.8.5`，见 `inference/inference_tsmllm_vllm.py:15`。同一脚本必须先 `import inference.vllm.chatts_vllm`，用于注册自定义 TS 模型和 multimodal processor，见 `inference/inference_tsmllm_vllm.py:17`、`inference/inference_tsmllm_vllm.py:23`。注册发生在 `inference/vllm/chatts_vllm.py:404`、`inference/vllm/chatts_vllm.py:582`、`inference/vllm/chatts_vllm.py:761-765`。

`高风险原因`：这不是普通 vLLM 文本模型调用，而是依赖本仓库 `inference/vllm/chatts_vllm.py` 对 vLLM 内部 multimodal API 的适配。vLLM API 在版本间变化较快；主推理要求 `0.8.5`，RL Docker 镜像名却是 `vllm0.11.0`，见 `README.md:71`，两套环境不能简单混用。

### DeepSpeed / LLaMA-Factory

`已从代码确认`：SFT 脚本通过 `deepspeed --num_gpus 8 --master_port=19901 src/train.py` 进入训练，见 `scripts/qwen3-8b/train_stage1.sh:1`。`src/train.py:15` 调用 `llamafactory.train.tuner.run_exp`。Stage 1 使用 `ds_config/ds_config_3.json`，该配置是 ZeRO stage 3，并设置 `stage3_gather_16bit_weights_on_model_save=true`，见 `ds_config/ds_config_3.json:15-23`。

`已从代码确认`：LLaMA-Factory parser 对 DeepSpeed 有额外版本和启动方式要求：`deepspeed>=0.10.0,<=0.16.9`，并要求 DeepSpeed 分布式训练用合适的 distributed launch，见 `src/llamafactory/hparams/parser.py:173-176`、`src/llamafactory/hparams/parser.py:248-249`。

`高风险原因`：DeepSpeed 和 Transformers / Accelerate 的版本组合、Windows 支持、CUDA 扩展编译、NCCL、多卡通信都容易成为阻塞点。

### RL 框架 / Ray / EasyR1

`已从代码确认`：Stage 3 脚本调用 `python3 -m src.EasyR1.verl.trainer.main`，见 `scripts/qwen3-8b/train_stage1+2+3.sh:7`。配置继承 `src/EasyR1/examples/config.yaml`，其中默认 `algorithm.adv_estimator=grpo`，见 `src/EasyR1/examples/config.yaml:23-28`。RL 脚本设置 `data.rollout_batch_size=128`、`worker.rollout.n=8`、`trainer.n_gpus_per_node=8`，见 `scripts/qwen3-8b/train_stage1+2+3.sh:15`、`scripts/qwen3-8b/train_stage1+2+3.sh:22`、`scripts/qwen3-8b/train_stage1+2+3.sh:24`。

`已从代码确认`：EasyR1 README 明确给出 Ray 多节点启动方式，见 `src/EasyR1/README.md:127-139`，并在 FAQ 里把 CUDA OOM 和 DeepSpeed 冲突列为问题，见 `src/EasyR1/README.md:209`、`src/EasyR1/README.md:215`。

`高风险原因`：Stage 3 同时涉及 Ray、FSDP/rollout、vLLM、reward function、多进程 GPU 调度和 checkpoint merge。它是本仓库最不适合本周临时硬跑的部分。

### 模型下载和初始化

`已从代码确认`：README 的基础模型准备需要下载 `Qwen/Qwen3-8B`，复制 `base_model/Config-Qwen3-8B/*` 到模型目录，再运行 `initial_model.py`，见 `README.md:91-93`。下载脚本调用 `huggingface_hub.snapshot_download()`，见 `download_model.py:12`。

`高风险原因`：Qwen3-8B 权重体积大，下载耗时、断点、磁盘空间、网络访问和 `trust_remote_code=True` 都是风险点。推理脚本也支持 HuggingFace model ID，但会触发模型下载，见 `inference/inference_tsmllm_vllm.py:232-238`。

### 数据下载

`已从代码确认`：数据下载脚本从 `Time-HD-Anonymous/ST-Bench` 下载到 `data/ST-Bench`，见 `download_dataset.py:17-21`。当前本地 `data/` 下只看到 `data/dataset_info.json`，未看到真实 ST-Bench jsonl 文件。

`高风险原因`：没有真实数据时不能做真实 inference/evaluation；如果下载不完整，默认路径会触发 `FileNotFoundError`，推理脚本检查见 `inference/inference_tsmllm_vllm.py:225-230`，评估脚本检查见 `evaluation/evaluate.py:153-158`。

## 3. 轻量执行路径

### 路径 A：只做静态阅读

`建议本周优先级最高`。继续阅读并完善 `docs/streasoner_code_reading/`，只用 `rg`、`Get-Content`、`git status`、`python -m py_compile` 这类静态或轻量命令。

适合回答组会问题：

- 模型结构：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`
- Prompt / inference：`inference/prompt_utils.py`、`inference/inference_tsmllm_vllm.py`
- vLLM TS patch：`inference/vllm/chatts_vllm.py`
- SFT：`scripts/qwen3-8b/train_stage1.sh`、`src/train.py`
- RL：`scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh`、`src/EasyR1/verl/trainer/ray_trainer.py`
- Evaluation：`evaluation/evaluate.py`、`evaluation/evaluate_qa.py`

### 路径 B：只看已有结果

`已从本地文件确认`：仓库中已有 `exp_STReasoner-8B` 结果目录，包含四个任务的 `generated_answer.json` 和 `evaluation_metrics.json`：

- `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/`
- `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/`
- `exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/`
- `exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/`

这是本周汇报最稳的实证材料：可以展示已有输出格式、已有指标文件和不同任务的 evaluation 结果，但要明确说“这是仓库已有结果，不是本周重跑结果”。

### 路径 C：只跑 evaluation

`根据代码推断，未由真实运行验证`：如果真实测试集已经在本地，且已有 `generated_answer.json` 可用，可以只跑 evaluation，不需要加载大模型。入口参数见 `evaluation/evaluate.py:102-127`，主流程是读取数据、读取 prediction、计算 metrics、写 `evaluation_metrics.json`，见 `evaluation/evaluate.py:160-174`。

推荐命令结构：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp_STReasoner-8B/reasoning_entity-STReasoner-8B
```

注意：README 的 evaluation 示例没有显式传 `--dataset`，见 `README.md:161-170`；而 `evaluation/evaluate.py` 默认路径是 `data/reasoning/*.jsonl`，见 `evaluation/evaluate.py:58-79`。因此如果只有 `data/ST-Bench/ST-Test/*.jsonl`，建议显式传 `--dataset`。

### 路径 D：只跑 `max_samples=1` inference sanity check

`前提条件`：模型已经本地就绪，vLLM 0.8.5 环境已经正确安装，GPU 显存足够，真实数据已下载。

推理脚本提供 `--max_samples`，见 `inference/inference_tsmllm_vllm.py:203-208`。GPU 参数是 `--num_gpus` 和 `--num_gpus_per_process`，见 `inference/inference_tsmllm_vllm.py:179-190`。实际 vLLM worker 用 `tensor_parallel_size=len(gpu_id.split(','))` 和 `gpu_memory_utilization=0.95` 初始化模型，见 `inference/llm_utils.py:142-149`。

最低风险 sanity check 命令结构：

```bash
python inference/inference_tsmllm_vllm.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --model_path /path/to/local/STReasoner-8B \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_samples 1 \
  --output_name generated_answer_sanity.json
```

`尚未确认`：单卡是否能装下 STReasoner-8B + vLLM TS 路径，取决于 GPU 显存、上下文长度、TS token 数和 vLLM KV cache；本周没有实际运行验证。

## 4. 不建议本周做的事

1. 不建议从零跑 Stage 1 / Stage 2 SFT。理由：脚本默认 8 卡 DeepSpeed ZeRO-3，见 `scripts/qwen3-8b/train_stage1.sh:1-2`，且 `cutoff_len=10000`、`per_device_train_batch_size=2`、`gradient_accumulation_steps=32`，见 `scripts/qwen3-8b/train_stage1.sh:13-14`、`scripts/qwen3-8b/train_stage1.sh:29`。
2. 不建议跑 Stage 3 RL / S-GRPO。理由：需要 Docker/Ray/EasyR1/vLLM/RL rollout，多卡配置写死 8 卡，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:7-31`。
3. 不建议临时混用 SFT 环境和 RL Docker 环境。理由：主环境固定 `vllm==0.8.5`，见 `requirements.txt:13`，RL Docker 镜像名是 `vllm0.11.0`，见 `README.md:71`。
4. 不建议在 Windows 原生环境直接安装 flash-attn wheel。理由：`requirements.txt:15` 指向 Linux x86_64、cp310、cu12、torch2.6 wheel。
5. 不建议临时下载大模型后直接推理全量任务。理由：模型下载、custom code 初始化、vLLM TS 注册、GPU 显存和数据路径都是独立风险点，见 `download_model.py:12`、`inference/inference_tsmllm_vllm.py:15-23`、`inference/llm_utils.py:149`。
6. 不建议修改核心代码“抢救环境”。本阶段目标是代码阅读和组会汇报；任何临时 patch 都会增加后续复现的不确定性。
7. 不建议把已有 `exp_STReasoner-8B` 指标当成本周复现结果。它可以作为仓库自带结果分析，但应明确来源。

## 5. 下周复现建议

1. 先确定硬件和系统：优先 Linux + CUDA 12.x + Python 3.10；如果没有 8 x 80GB GPU，不要承诺完整 SFT/RL。
2. 分环境管理：SFT / inference 用主 `requirements.txt` 环境；RL 用 README 指定 Docker，不混装。
3. 先下载数据，不下载模型：用 `download_dataset.py` 获取 ST-Bench 后，先跑 dataset 文件存在性、字段统计和 evaluation dry check。
4. 再做 evaluation-only：用已有 `exp_STReasoner-8B` + 真实 test jsonl 复算指标，确认评价链路。
5. 再做 inference sanity：只用 `--max_samples 1`，先单任务、单样本、单输出文件，不覆盖已有结果。
6. 最后才考虑训练：先小模型或官方 checkpoint sanity，再决定是否动 Stage 1/2；Stage 3 RL 放到专门复现周。

### 本文档核心结论

1. `已从代码确认`：官方完整路线假设很重，README 写明 8 x A100-SXM4-80GB 和 CUDA 12.8。
2. `已从代码确认`：SFT/inference 主环境和 RL Docker 环境的 vLLM / transformers 版本不一致，不能简单合并成一个环境。
3. `已从代码确认`：当前仓库已有结果目录可用于汇报，但本地没有真实 ST-Bench jsonl。
4. `根据代码推断，未由真实运行验证`：本周最低风险路径是静态阅读、已有结果分析、evaluation-only；inference 只在模型和环境已就绪时做 `max_samples=1`。

### 组会讲法

这套代码不是“pip install 后直接跑”的轻量项目，而是分三套风险：SFT 依赖 DeepSpeed + LLaMA-Factory，推理依赖 vLLM 0.8.5 的自定义 time series multimodal patch，RL 依赖 Docker + EasyR1/verl + Ray + vLLM rollout。官方资源假设是 8 张 A100 80GB。基于本周时间，我建议把汇报定位为代码结构和流程复现准备：静态读懂模型、prompt、inference、evaluation 和 S-GRPO；结果展示使用仓库已有 `exp_STReasoner-8B`，不声称已经重跑完整训练。

### 后续需要验证的问题

1. `尚未确认`：真实机器上 Python 3.10 + CUDA 12.8 + torch 2.6 + flash-attn wheel 是否能一次安装成功。
2. `尚未确认`：vLLM 0.8.5 与当前 `inference/vllm/chatts_vllm.py` 在本地 GPU 上是否能正常注册并加载 STReasoner-8B。
3. `尚未确认`：单卡 `max_samples=1` 推理的最低显存需求。
4. `尚未确认`：已有 `exp_STReasoner-8B` 的 `generated_answer.json` 与当前 evaluation 脚本重新计算出的指标是否完全一致。
5. `尚未确认`：RL Docker 中 vLLM 0.11.0 与本仓库 time series 模型注册逻辑是否完全兼容。
