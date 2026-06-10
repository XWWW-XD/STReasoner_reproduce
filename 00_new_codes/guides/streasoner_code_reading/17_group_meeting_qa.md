# 17 组会问答准备

使用原则：回答时先给结论，再补代码证据；没有运行验证的地方要主动说明。不要把“静态代码阅读”说成“完整复现实验”。

## Q1 你代码跑通了吗？

**简短回答**

没有跑通完整训练，也没有跑大模型推理。本周完成的是静态代码阅读、已有结果目录分析和复现风险评估。

**更完整回答**

我没有启动 Stage 1/2 SFT、Stage 3 RL、DeepSpeed 或 vLLM 大模型推理。主要原因是官方 README 的完整路线资源要求很高，写明硬件是 `8 x NVIDIA A100-SXM4-80GB` 和 CUDA 12.8，见 `README.md:56-58`。我本周做的是把数据、prompt、模型结构、time series encoder、embedding merge、inference、evaluation、SFT 和 S-GRPO 的代码链路逐步读清楚，并整理到 `docs/streasoner_code_reading/`。

**如果我还没验证，应如何诚实表达**

可以说：“我目前没有完整跑通训练和推理，只完成了静态代码级追踪。已有 `exp_STReasoner-8B/` 可以作为仓库自带结果分析，但不是我本周重跑出来的结果。”

**不要吹过头的边界**

不要说“我已经复现了论文结果”。不要把已有 `evaluation_metrics.json` 当成自己的复现实验结果。

## Q2 为什么这周没有跑完整训练？

**简短回答**

因为完整训练风险很高，资源和环境都不轻量；本周更合理的目标是先读懂代码结构和复现链路。

**更完整回答**

Stage 1/2 SFT 脚本默认用 `deepspeed --num_gpus 8`，见 `scripts/qwen3-8b/train_stage1.sh:1`；Stage 3 RL 脚本默认 `trainer.n_gpus_per_node=8`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:24`。环境上，主 `requirements.txt` 固定 `torch==2.6.0`、`transformers==4.52.4`、`vllm==0.8.5`，见 `requirements.txt:9-13`；RL README 推荐 Docker 镜像是 `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`，见 `README.md:71`。这些组合说明直接完整跑训练很容易被显存、CUDA、vLLM API、DeepSpeed 和 Ray 卡住。

**如果我还没验证，应如何诚实表达**

可以说：“我不是因为忽略实验，而是先规避高风险盲跑。当前阶段先保证代码理解可解释，下周再做低风险 evaluation-only 或单样本 sanity check。”

**不要吹过头的边界**

不要说“训练一定能跑，只是我没时间”。实际能否跑通取决于硬件、驱动、CUDA、模型和数据是否就绪。

## Q3 你现在真正理解了代码哪部分？

**简短回答**

我比较确定理解了从样本到 prompt、time series 编码、embedding 合并、vLLM 推理输出、evaluation 评分的主链路；SFT/RL 部分理解到脚本和关键函数层面，但没有运行验证。

**更完整回答**

我能完整讲清一条样本的流动：JSONL 通过 `load_dataset()` 读成 dict，`prepare_batches()` 把 `sample["input"]` 放进 `question_list`，把 `sample["timeseries"]` 放进 `ts_list`，见 `inference/inference_tsmllm_vllm.py:98-149`。prompt 侧通过 `get_prompt_suffix(task)` 追加输出格式约束，见 `inference/prompt_utils.py:27-38`。time series 侧由 `sp_encoding()` 和 `TimeSeriesEmbedding` 处理，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`、`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。embedding 合并由 HF 的 `_merge_input_ids_with_time_series_features()` 或 vLLM 的 `merge_multimodal_embeddings()` 路径完成，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`、`inference/vllm/chatts_vllm.py:699-709`。evaluation 侧按任务算 accuracy 或 MAE/MAPE，见 `evaluation/evaluate_qa.py:198-330`。

**如果我还没验证，应如何诚实表达**

可以说：“这些是静态代码确认的调用链；真实 batch 的 shape、vLLM cache 行为和运行时显存还没有验证。”

**不要吹过头的边界**

不要说“我理解了所有训练细节”。RL 训练日志、实际 reward 曲线、checkpoint 合并结果还没有运行验证。

## Q4 STReasoner 和普通 LLM 有什么区别？

**简短回答**

普通 LLM 只吃文本 token；STReasoner 额外把真实时间序列编码成 embedding token，并插入到文本 embedding 序列里。

**更完整回答**

普通 Qwen3 的输入主要是 `input_ids` 对应的文本 embedding。STReasoner 在 Qwen3 上新增了 time series processor、`TimeSeriesEmbedding` 和 embedding merge 逻辑。文本 prompt 中的 `<ts><ts/>` 是占位符，真实数值序列来自 `timeseries` 字段。`TimeSeriesEmbedding` 把数值序列 patchify 后映射到 Qwen hidden size，然后 `_merge_input_ids_with_time_series_features()` 把 TS embedding 和文本 embedding 合成一个序列，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。

**如果我还没验证，应如何诚实表达**

可以说：“我确认了模型代码结构和静态 forward 路径，但没有实际跑 forward 验证每个 batch 的 tensor shape。”

**不要吹过头的边界**

不要说 STReasoner 有独立图神经网络模块。当前代码中可确认的是 time series encoder 和 graph text prompt，不是 GNN。

## Q5 time series encoder 是怎么接入 LLM 的？

**简短回答**

时间序列先被编码成 value/mask 特征，再按 patch 切分，通过 MLP 投影到 LLM hidden size，最后替换 prompt 中 `<ts><ts/>` 对应的位置。

**更完整回答**

接入分三步。第一步，processor 的 `sp_encoding()` 对原始序列做归一化，并生成数值和 mask 交错的结构，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`。第二步，`TimeSeriesEmbedding.forward()` 根据 mask 计算 `valid_lengths`，再用 `(valid_lengths + patch_size - 1) // patch_size` 得到 patch 数，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:80-87`。第三步，patch 经过 MLP 投影成 hidden size embedding，之后 forward 里调用 merge 函数把它插进文本 embedding，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:171-179`、`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。

**如果我还没验证，应如何诚实表达**

可以说：“我用代码公式推导过，例如长度 16、patch_size 8 会得到 2 个 TS embedding token；但还没有用真实样本 forward 验证所有 shape。”

**不要吹过头的边界**

不要说 encoder 理解了完整时空拓扑。它主要处理数值时间序列；空间结构主要通过 prompt 文本给 LLM。

## Q6 graph prompting 有什么作用？

**简短回答**

它把节点关系用自然语言写进 prompt，让 LLM 在推理时看到空间依赖关系。

**更完整回答**

从代码看，graph prompting 不是一个独立类或 graph encoder，而是数据样本 `input` 里的一段 `Graph Structure: ...` 文本。推理代码没有重新生成 graph prompt，只是直接读取 `sample["input"]`，见 `inference/inference_tsmllm_vllm.py:136-140`。S-GRPO 里 `remove_graph_structure()` 会删除这段图结构文本来构造 no-graph 对照，见 `src/EasyR1/verl/utils/dataset.py:35-56`。这说明 graph text 是 spatial-aware 对照训练的关键输入片段。

**如果我还没验证，应如何诚实表达**

可以说：“我确认已有输出和 RL 代码都围绕 `Graph Structure` 文本工作；但真实 ST-Bench 中每条样本的 graph prompt 格式还需要下载数据后逐行验证。”

**不要吹过头的边界**

不要说 graph prompting 一定来自仓库中的生成脚本；当前没有定位到 graph prompt 生成模块。

## Q7 w/ spatial structure 和 w/o spatial structure 是什么？

**简短回答**

`w/ spatial structure` 是保留 `Graph Structure` 的原始 prompt；`w/o spatial structure` 是删除图结构文本后的对照 prompt，time series 仍然保留。

**更完整回答**

在 S-GRPO 数据侧，如果 `data.enable_spatial_reward=true`，`RLHFDataset.__getitem__()` 会同时构造原 prompt 和 no-graph prompt，见 `src/EasyR1/verl/utils/dataset.py:313-334`。no-graph prompt 由 `remove_graph_structure()` 删除从 `Graph Structure:` 到 `please analyze` 前的片段，见 `src/EasyR1/verl/utils/dataset.py:35-56`。trainer 侧再分别对原 prompt 和 no-graph prompt rollout，并比较两者 reward，见 `src/EasyR1/verl/trainer/ray_trainer.py:526-550`。

**如果我还没验证，应如何诚实表达**

可以说：“我确认了代码中构造 no-graph 分支的逻辑，但没有用真实 RL batch 验证删除规则在所有样本上都完全匹配。”

**不要吹过头的边界**

不要说 w/o spatial structure 是换了一个数据集。主线代码里它是在同一条样本上删除 graph text。

## Q8 S-GRPO 和普通 GRPO 差异是什么？

**简短回答**

S-GRPO 在普通 GRPO 上加了 spatial 对照奖励：比较原始 prompt 和 no-graph prompt 的 reward，把空间结构带来的收益作为 bonus 加入训练信号。

**更完整回答**

脚本层面，vanilla GRPO 和 S-GRPO 都读同一组 `ST-RL/*.jsonl`，reward function 都是 `src/EasyR1/examples/reward_function/str.py:compute_score`。S-GRPO 脚本额外设置 `algorithm.enable_spatial_reward=true`、`algorithm.spatial_reward_weight=0.1`、`data.enable_spatial_reward=true`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`。trainer 侧 `_compute_spatial_reward()` 比较 original reward 和 no-graph reward，如果 `original_r > no_graph_r * 0.8` 就给 spatial reward，见 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`。进入 advantage 前，代码把 `spatial_reward_weight * spatial_reward` 加到 `token_level_scores`，见 `src/EasyR1/verl/trainer/ray_trainer.py:724-729`。

**如果我还没验证，应如何诚实表达**

可以说：“我确认了代码实现的差异，但还没有把论文 S-GRPO 数学公式逐项对照到代码，特别是 `0.8` 阈值和 `spatial_reward_weight` 的论文来源。”

**不要吹过头的边界**

不要说 S-GRPO 是训练了一个新的 spatial reward model。代码中 spatial reward 是 trainer 里计算的 rule-based 对照 bonus。

## Q9 这篇论文最核心的技术贡献是什么？

**简短回答**

从代码角度看，核心贡献是把时间序列数值模态接入 LLM，并用 spatial-aware RL 强化模型利用图结构做时空推理。

**更完整回答**

我会把贡献讲成三层。第一，数据层有 ST-Bench，覆盖 alignment、CoT/SFT、RL、test 等 split，见 `data/dataset_info.json` 和 `README.md:78-81`。第二，模型层把 Qwen3 扩展成 Qwen3TS：新增 `TimeSeriesEmbedding`，把数值序列变成可插入 LLM 的 embedding token，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。第三，训练层用 S-GRPO 构造 original/no-graph 对照，通过 spatial reward 鼓励模型利用空间结构，见 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`、`:724-729`。

**如果我还没验证，应如何诚实表达**

可以说：“这是我基于 README、脚本和代码映射得到的理解；论文原文中的贡献表述和公式还需要逐项核对。”

**不要吹过头的边界**

不要说所有贡献都已经由本地实验验证。当前只确认了代码落点和实现路线。

## Q10 你下周准备做什么最小复现？

**简短回答**

先下载真实 ST-Bench，用已有 `exp_STReasoner-8B` 跑 evaluation-only；如果模型和 vLLM 环境就绪，再做 `max_samples=1` inference sanity check。

**更完整回答**

最小复现路径分两步。第一步不加载模型，只下载数据后复算已有结果：`evaluation/evaluate.py` 会读取 dataset 和 `generated_answer.json`，再写 `evaluation_metrics.json`，见 `evaluation/evaluate.py:160-174`。这能验证数据路径、prediction 格式和指标脚本。第二步才是单样本 inference sanity check，使用 `inference/inference_tsmllm_vllm.py --max_samples 1`，见 `inference/inference_tsmllm_vllm.py:203-208`。这一步必须等本地模型、vLLM 0.8.5 和 GPU 显存都准备好。

**如果我还没验证，应如何诚实表达**

可以说：“下周的最小目标不是训练，而是先复算已有结果和跑通单样本推理链路。”

**不要吹过头的边界**

不要承诺下周完整重训 Stage 1/2/3。那需要多卡、DeepSpeed、Ray、vLLM 和较长调试周期。

## Q11 如果显存不够怎么办？

**简短回答**

先不要跑训练；优先做不需要 GPU 的 evaluation-only。推理 sanity check 也只做单样本，训练要降模型规模或改分布式配置。

**更完整回答**

显存不足时，优先级应该是：第一，继续静态阅读和已有结果分析；第二，跑 evaluation-only，因为它不加载模型；第三，如果必须推理，只跑 `--max_samples 1`，并设置较小的任务范围和单独输出文件。训练侧不要直接跑官方脚本，因为它默认 8 卡。若后续确实要训练，需要系统性调整 `--num_gpus`、`per_device_train_batch_size`、`gradient_accumulation_steps`、DeepSpeed 配置、RL 的 `trainer.n_gpus_per_node`、`data.rollout_batch_size` 和 `worker.rollout.n`，这些在 `scripts/qwen3-8b/train_stage1.sh` 和 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh` 中都有硬配置。

**如果我还没验证，应如何诚实表达**

可以说：“我现在只能根据脚本判断显存风险，没有实测最低显存。单卡能否装下 STReasoner-8B + vLLM TS 路径还需要跑 sanity check。”

**不要吹过头的边界**

不要说“把 batch 调小就一定能跑”。vLLM KV cache、模型权重、TS token 数、上下文长度和分布式配置都会影响显存。

## Q12 这个项目和你的研究方向有什么关系？

**简短回答**

它提供了一个把时间序列、图结构和 LLM 推理结合起来的完整工程范例，可以作为我后续研究时空数据理解和多模态推理的代码参照。

**更完整回答**

如果我的方向涉及时间序列、时空数据、图结构推理或 LLM for scientific / structured data，这个项目很有参考价值。它不是简单把数值序列转成文本，而是通过 `TimeSeriesEmbedding` 把数值模态接入 LLM embedding 空间；同时通过 graph prompt 和 S-GRPO 对照奖励，让模型学习利用空间结构。这对后续研究有两点启发：一是数值模态可以通过专门 encoder 接入 LLM；二是结构信息不一定必须先做 GNN，也可以通过 prompt 和 RL 对照信号来强化利用。

**如果我还没验证，应如何诚实表达**

可以说：“目前关系主要体现在方法和工程参考层面。我还没有验证它在我的具体数据或任务上是否有效。”

**不要吹过头的边界**

不要说它已经证明能解决我的研究问题。更稳妥的说法是：它提供了一个可迁移的设计思路和代码框架。

## Q13 你如何确认 evaluation 没有加载模型？

**简短回答**

evaluation 只读取 dataset 和 prediction 文件，然后计算指标并写 JSON，不调用模型加载。

**更完整回答**

`evaluation/evaluate.py` 解析 `--exp_path`、`--dataset`、`--task` 等参数，检查文件存在后调用 `load_jsonl_dataset()` 和 `load_prediction_files()`，再调用 `evaluate_predictions_for_task()`，见 `evaluation/evaluate.py:102-174`。具体指标在 `evaluation/evaluate_qa.py`：forecasting 解析数值序列算误差，多选任务抽 `<answer>` 中的 A-D 算 accuracy，见 `evaluation/evaluate_qa.py:198-330`。

**如果我还没验证，应如何诚实表达**

可以说：“我确认了静态代码路径不加载模型，但还没有用真实 ST-Bench 文件重新跑一遍 evaluation。”

**不要吹过头的边界**

不要说当前已有 metrics 一定由这个版本的 evaluation 脚本生成；已有结果可能来自旧脚本，因为字段和当前代码略有差异。

## Q14 你最担心哪一个代码风险点？

**简短回答**

最担心 time series placeholder、真实 `timeseries` 列表、patch count 和 embedding merge 之间的对齐问题。

**更完整回答**

整个模型链路依赖 `<ts><ts/>` 数量、`timeseries` 条数、`patch_cnt` 顺序和 TS embedding 顺序一致。HF merge 函数里有校验，不一致会抛错，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:480-493`。vLLM 路径则先做 prompt replacement，再调用 `merge_multimodal_embeddings()`，见 `inference/vllm/chatts_vllm.py:352-401`、`:699-709`。这部分是核心，也是最需要用真实样本 sanity check 的地方。

**如果我还没验证，应如何诚实表达**

可以说：“静态代码显示有校验和处理逻辑，但我还没有用真实样本比较 HF 和 vLLM 两条路径的最终 embedding 对齐结果。”

**不要吹过头的边界**

不要说 merge 逻辑已经完全正确。没有真实 forward 或 vLLM 推理验证前，只能说代码意图和静态机制是清楚的。

## Q15 你会怎么把这篇论文讲成一句话？

**简短回答**

STReasoner 是一个把时间序列数值编码进 LLM，并通过空间结构 prompt 和 S-GRPO 强化时空推理能力的框架。

**更完整回答**

一句话展开就是：它用专门的 Time Series Encoder 把连续时间序列变成 LLM 可处理的 embedding token，用 prompt 中的 `Graph Structure` 提供空间关系，再用三阶段训练从 alignment、CoT SFT 走到 spatial-aware GRPO，让模型在四类时空 reasoning 任务上生成结构化答案。

**如果我还没验证，应如何诚实表达**

可以说：“这是基于代码阅读和 README 的概括，不代表我已经复现了论文指标。”

**不要吹过头的边界**

不要说它完全解决了 LLM 时空推理问题。更准确的是：它提供了一套面向时空时间序列推理的模型和训练框架。
