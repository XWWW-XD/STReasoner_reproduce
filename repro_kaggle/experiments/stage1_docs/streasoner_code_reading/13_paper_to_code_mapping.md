# 13 论文概念到代码映射

本文把论文和 README 中的关键概念映射到仓库代码。注意：本文没有逐字核对论文 PDF 原文；“论文中的含义”主要来自论文标题、README 三阶段描述、脚本命名和代码行为。因此，涉及论文精确定义的地方会明确标注为 `根据代码推断，未由论文原文逐句验证`。

状态标记：

- `已从代码确认`：仓库中有明确文件、脚本、类或函数证据。
- `合理推测`：代码实现支持该解释，但概念名称不是代码中的显式类名或函数名。
- `论文概念暂未定位到代码`：没有找到独立代码模块、脚本入口或明确调用链。

## 1. 映射总表

| 论文概念 | 论文中的含义 | 代码中的位置 | 是否已确认 | 相关文件 / 函数 | 组会讲法 |
|---|---|---|---|---|---|
| ST-Bench | 时空推理 benchmark 数据集总称，包含训练、RL、测试等 split | 下载到 `data/ST-Bench/`；数据注册在 `data/dataset_info.json` | 已从代码确认 | `README.md:78-81`, `download_dataset.py:12-24`, `data/dataset_info.json:1-145` | ST-Bench 是整个数据根目录，不是单个 JSONL；代码按阶段拆成 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test。 |
| ST-Align | Stage 1 用于 time-series alignment 的监督微调数据 | `alignment`、`alignment_test` 注册到 `ST-Bench/ST-Align/*.jsonl`；Stage 1 脚本使用 `alignment` | 已从代码确认 | `data/dataset_info.json:2-17`, `scripts/qwen3-8b/train_stage1.sh:5-11`, `README.md:48` | 第一阶段让模型先学会文本 prompt、`<ts>` 占位符和真实 time series 输入之间的基本对齐。 |
| ST-SFT | 论文中可能指普通 SFT / finetune split | 注册到 `ST-Bench/ST-SFT/*_finetune.jsonl`，但未找到官方 8B 主线脚本直接使用 | 部分确认 | `data/dataset_info.json:50-81`; 未找到 `scripts/qwen3-8b` 使用 `entity_sft` 等名称 | ST-SFT 文件名存在，但这套仓库主线 Stage 2 使用的是 ST-CoT，不是这些 `*_sft` dataset 名称。 |
| ST-RL | Stage 3 RL / GRPO / S-GRPO 使用的数据 | 四类 RL JSONL 作为 `data.train_files` 传入 EasyR1 | 已从代码确认 | `data/dataset_info.json:82-113`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:9-14` | 第三阶段不是 SFT，而是用 ST-RL prompt 让模型 rollout 多个回答，再由 reward 计算 GRPO 更新。 |
| Time Series Encoder | 把数值时间序列切 patch 并投影到 LLM hidden size 的模块 | `TimeSeriesEmbedding`，被 `Qwen3TSForCausalLM` 实例化为 `self.ts_encoder` | 已从代码确认 | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`, `:352-364`, `:580-588` | 它是 STReasoner 相比普通 Qwen3 的关键新增模块：把连续时序转成可插入 token 序列的 embedding。 |
| Graph Prompting | 在 prompt 中显式写入图结构，让 LLM 读到节点依赖关系 | 样本 `input/question_text` 中包含 `Graph Structure: ...`；没有独立 `GraphPrompting` 类 | 合理推测 | `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:4`, `src/EasyR1/verl/utils/dataset.py:35-56` | 图结构不是图神经网络模块，而是自然语言 prompt 里的 `Graph Structure` 文本。 |
| spatial structure | 样本中的空间依赖 / 节点连接结构 | 主要体现为 `Graph Structure: Node a->Node b` 文本；S-GRPO 用它构造 original/no-graph 对照 | 已从代码确认 | `exp_STReasoner-8B/*/generated_answer.json`, `src/EasyR1/verl/utils/dataset.py:35-56` | 代码里的 spatial structure 主要是 prompt 片段，不是显式邻接矩阵张量。 |
| w/ spatial structure | 带图结构的原始 prompt 分支 | 正常 `input` / `messages` 不删除 `Graph Structure`，直接生成 rollout | 已从代码确认 | `src/EasyR1/verl/utils/dataset.py:313-319`, `src/EasyR1/verl/trainer/ray_trainer.py:520-545` | 原 prompt 保留图结构，用作 S-GRPO 的“带空间结构”回答。 |
| w/o spatial structure | 去掉图结构的对照 prompt 分支 | `remove_graph_structure()` 删除 `Graph Structure: ... please analyze` 之前的片段；仍保留同一条 timeseries | 已从代码确认 | `src/EasyR1/verl/utils/dataset.py:35-56`, `:321-334`, `src/EasyR1/verl/trainer/ray_trainer.py:526-550` | no-graph 不是换数据集，而是在同一条样本上删除图结构文本，构造对照回答。 |
| S-GRPO | Spatial-aware Group Relative Policy Optimization | `train_stage1+2+3_w_spatial.sh` 打开 spatial 开关；trainer 比较 original/no-graph reward 并加权注入 token score | 已从代码确认 | `README.md:50`, `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`, `src/EasyR1/verl/trainer/ray_trainer.py:466-494`, `:724-729` | S-GRPO = GRPO + 空间结构对照奖励；不是单独训练 reward model。 |
| reward | RL 中对模型回答打分的 rule-based reward | `str.py` 实现 format 和 accuracy；spatial reward 在 trainer 中实现 | 已从代码确认 | `src/EasyR1/examples/reward_function/str.py:12-15`, `:71-108`, `src/EasyR1/verl/trainer/ray_trainer.py:466-494` | 基础 reward 看格式和答案正确性；S-GRPO 再看带图结构回答是否优于去图结构回答。 |
| inference | 用训练/合并后的模型在 ST-Test 上生成回答 | `inference/inference_tsmllm_vllm.py` 读取 dataset、构造 prompt、调用 vLLM、写 `generated_answer.json` | 已从代码确认 | `README.md:144-156`, `inference/inference_tsmllm_vllm.py:41-61`, `:218-238`, `:296-322` | 推理链路是 ST-Test JSONL + timeseries -> vLLM TS model -> `generated_answer.json`。 |
| evaluation | 读取预测和测试集，计算 accuracy / MAE / MAPE 等指标 | `evaluation/evaluate.py` CLI，`evaluation/evaluate_qa.py` 指标函数 | 已从代码确认 | `README.md:161-170`, `evaluation/evaluate.py:100-174`, `evaluation/evaluate_qa.py:198-330` | 评估不加载模型，只按 `idx` 对齐预测和标准答案，多选算 accuracy，forecasting 算 MAE/MAPE。 |

## 2. 已确认概念

### ST-Bench

`已从代码确认`：README 要求通过 `python download_dataset.py` 下载 ST-Bench，见 `README.md:78-81`。下载脚本把 HuggingFace dataset `Time-HD-Anonymous/ST-Bench` 下载到本仓库的 `data/ST-Bench/`，见 `download_dataset.py:12-24`。

`已从代码确认`：`data/dataset_info.json` 把 ST-Bench 拆成多个 split：ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test 等。普通 time-series 版本统一使用 `input/output/timeseries` 三列映射，见 `data/dataset_info.json:1-145`。

组会讲法：ST-Bench 在代码里不是一个 dataloader 类，而是一个目录协议：不同阶段的脚本通过路径或 dataset registry 读取不同 split。

### ST-Align

`已从代码确认`：`alignment` 数据项指向 `ST-Bench/ST-Align/alignment_train.jsonl`，`alignment_test` 指向 `ST-Bench/ST-Align/alignment_test.jsonl`，列映射都是 `input/output/timeseries`，见 `data/dataset_info.json:2-17`。

`已从代码确认`：Stage 1 脚本使用 `--dataset "alignment"` 和 `--template "STReasoner-Align"`，见 `scripts/qwen3-8b/train_stage1.sh:5-10`。README 把 Stage 1 描述为 supervised fine-tuning for time series alignment，见 `README.md:48`。

组会讲法：Stage 1 是把 Qwen3 改造成能吃 STReasoner 输入格式的对齐阶段，重点不是复杂推理，而是让模型适配 time series 模态和 prompt 结构。

### ST-RL

`已从代码确认`：ST-RL 四类任务注册在 `data/dataset_info.json:82-113`。S-GRPO 脚本把四个文件作为 `data.train_files` 传给 EasyR1，并指定 `data.prompt_key=input`、`data.ts_key=timeseries`、`data.answer_key=output`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:9-14`。

组会讲法：Stage 3 的训练样本仍然是 `input/output/timeseries`，但训练目标从“模仿标准答案”变成“采样多个回答，用 reward 和 GRPO 更新策略”。

### Time Series Encoder

`已从代码确认`：`TimeSeriesEmbedding` 定义在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。`Qwen3TSForCausalLM.__init__()` 中创建 `self.ts_encoder = TimeSeriesEmbedding(config.ts)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:352-364`。

`已从代码确认`：`Qwen3TSForCausalLM.forward()` 在有 `timeseries` 输入时调用 `self.ts_encoder(timeseries)`，并把 `ts_features, patch_cnt` 交给 embedding merge 函数，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。

组会讲法：论文里的 time series encoder 对应代码中的 `TimeSeriesEmbedding`。它把时间序列切成 patch，再投影到 Qwen hidden size，使 TS embedding 可以和文本 token embedding 一起进入 LLM。

### spatial structure / w/ spatial structure / w/o spatial structure

`已从代码确认`：本地已有推理输出的 `question_text` 直接包含 `Graph Structure: Node ...`，例如 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:4`。

`已从代码确认`：`remove_graph_structure()` 会删除从 `Graph Structure:` 到 `please analyze` 之前的文本，见 `src/EasyR1/verl/utils/dataset.py:35-56`。当 `data.enable_spatial_reward=true` 且样本含 `timeseries` 时，`RLHFDataset.__getitem__()` 会同时构造原 prompt 和 no-graph prompt，见 `src/EasyR1/verl/utils/dataset.py:313-334`。

`已从代码确认`：trainer 中 spatial 分支会生成原 prompt rollout 和 no-graph rollout，见 `src/EasyR1/verl/trainer/ray_trainer.py:526-550`。

组会讲法：代码里的“空间结构”就是题面中的图结构文本。with spatial 是保留这段文本，without spatial 是删除这段文本，但 time series 仍然保留。

### S-GRPO

`已从代码确认`：README 把 Stage 3 描述为 RL with S-GRPO，见 `README.md:50`。官方 S-GRPO 入口是 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh`，它比 vanilla GRPO 多出 `algorithm.enable_spatial_reward=true`、`algorithm.spatial_reward_weight=0.1`、`data.enable_spatial_reward=true`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`。

`已从代码确认`：`RayPPOTrainer._compute_spatial_reward()` 比较 original reward 和 no-graph reward。如果 `original_r > no_graph_r * 0.8`，就把 spatial reward 置 1，否则置 0，见 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`。进入 advantage 前，代码把 `spatial_reward_weight * spatial_reward` 加到 `token_level_scores`，见 `src/EasyR1/verl/trainer/ray_trainer.py:724-729`。

组会讲法：S-GRPO 的代码实现可以讲成“用去掉图结构的回答作为反事实对照，奖励那些确实从图结构中受益的回答”。

### reward

`已从代码确认`：Stage 3 脚本指定 reward function 为 `./src/EasyR1/examples/reward_function/str.py:compute_score`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:21`。

`已从代码确认`：基础 reward 由两个部分组成：`format_reward()` 检查 `<think>...</think><answer>...</answer>` 格式，见 `src/EasyR1/examples/reward_function/str.py:12-15`；`accuracy_reward()` 对多选字母、数值序列或其他答案计算准确性，见 `src/EasyR1/examples/reward_function/str.py:71-99`；`compute_score()` 合成 `overall/format/accuracy`，见 `src/EasyR1/examples/reward_function/str.py:101-108`。

组会讲法：基础 reward 看“格式是否像推理答案”和“最终答案是否对”；S-GRPO 的 spatial reward 是在 trainer 里额外加的，不在 `str.py` 里。

### inference

`已从代码确认`：README 推理命令调用 `inference/inference_tsmllm_vllm.py`，见 `README.md:144-156`。推理脚本的 `DEFAULT_TASK_CONFIG` 把 reasoning 任务映射到 `data/ST-Bench/ST-Test/*.jsonl`，见 `inference/inference_tsmllm_vllm.py:41-61`。

`已从代码确认`：主函数检查 dataset 和 model path，调用 vLLM 生成，然后写出 `generated_answer.json`。输出条目包含 `idx/question_text/response/num_tokens`，见 `inference/inference_tsmllm_vllm.py:218-238`、`inference/inference_tsmllm_vllm.py:296-322`。

组会讲法：推理阶段不会再训练，它只是把 ST-Test 的 `input` 和 `timeseries` 喂给 vLLM TS model，生成可评估的 JSON。

### evaluation

`已从代码确认`：README evaluation 命令调用 `evaluation/evaluate.py`，见 `README.md:161-170`。`evaluate.py` 解析 `--exp_path/--dataset/--task/--pred_pattern/--repo_root`，读取 dataset 和 prediction，再写 `evaluation_metrics.json`，见 `evaluation/evaluate.py:100-174`。

`已从代码确认`：多选任务走 `evaluate_multiple_choice_predictions()`，输出 `accuracy`，见 `evaluation/evaluate_qa.py:278-330`；forecasting 走 `evaluate_forecasting_predictions()`，当前代码输出 `mae/mape/target_stats`，见 `evaluation/evaluate_qa.py:198-275`。

组会讲法：评估脚本不加载模型，只做文件级指标计算。多选看 `<answer>` 里的 A-D，forecasting 解析数字序列算误差。

## 3. 合理推测概念

### Graph Prompting

`合理推测`：论文里的 Graph Prompting 在代码里没有一个独立类名或函数名。它主要由数据样本的 `input` 字段提供：题面中包含节点时间序列占位符和 `Graph Structure: ...` 文本。推理脚本只是读取 `sample["input"]`，不会重新构造图结构，见 `inference/inference_tsmllm_vllm.py:41-61` 和前序分析中的 `prepare_batches()` 读取逻辑。

代码证据是：本地已有输出的 `question_text` 包含 `Graph Structure`，见 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:4`；S-GRPO 的 no-graph 对照明确删除这一段，见 `src/EasyR1/verl/utils/dataset.py:35-56`。

组会讲法：可以说“Graph Prompting 在代码中是数据驱动的 prompt 设计，而不是一个单独的 graph encoder 模块”。如果要讲得更严谨，应补充“未在代码中找到名为 GraphPrompting 的模块”。

### ST-SFT

`合理推测`：`data/dataset_info.json` 注册了 `ST-Bench/ST-SFT/{entity,etiological,correlation,forecasting}_finetune.jsonl`，见 `data/dataset_info.json:50-81`。这些文件名说明仓库支持一个 ST-SFT split。

`尚未确认`：本次静态搜索没有发现官方 Qwen3-8B 主线脚本直接使用 `entity_sft`、`etiological_sft`、`correlation_sft` 或 `forecasting_sft`。README 三阶段表中 Stage 2 写的是 cold-start reasoning SFT，代码主线实际用的是 `ST-CoT/*_cot.jsonl`，见 `data/dataset_info.json:18-49` 和 `scripts/qwen3-8b/train_stage1+2.sh:5-9`。

组会讲法：ST-SFT 在 registry 里存在，但主线脚本没有用它。我会把它作为“已注册但用途未确认的数据 split”，不把它强行讲成官方 Stage 2 主线。

### spatial structure

`合理推测`：论文中的 spatial structure 可能有更广泛含义，例如真实空间拓扑、节点依赖、传播方向等。代码中可直接确认的落点是 prompt 里的 `Graph Structure` 文本和 S-GRPO 的 no-graph 对照。没有发现单独的 adjacency matrix 输入给模型 forward。

组会讲法：就代码而言，空间结构主要通过语言 prompt 注入；模型没有显式 GNN 层，LLM 依靠注意力在文本 token 和 TS embedding 上综合推理。

## 4. 暂未定位概念

### 独立的 Graph Prompting 模块

`论文概念暂未定位到代码`：没有找到 `GraphPrompting`、`GraphPrompt`、`graph_encoder` 之类独立模块来生成或编码图结构。当前可见实现是数据样本中已有 `Graph Structure` 文本，推理和训练代码负责读取和传递。

需要后续验证：下载真实 ST-Bench 后检查 `input` 是否都已经预先写好 graph prompt；如果数据生成脚本不在仓库中，则 graph prompt 的构造过程可能不在本仓库。

### ST-SFT 的官方训练入口

`论文概念暂未定位到代码`：虽然 `ST-SFT` split 在 `data/dataset_info.json:50-81` 注册，但没有在官方 Qwen3-8B 主线训练脚本中定位到使用这些 dataset 名称的入口。

需要后续验证：结合论文实验设置或真实数据说明，确认 ST-SFT 是早期命名、备用 split、ablation split，还是论文中另有含义。

### 论文公式级 S-GRPO 对照

`论文概念暂未定位到代码`：代码中已定位到 S-GRPO 的实现路径，但尚未逐项对照论文数学公式。特别是 `original_r > no_graph_r * 0.8` 和 `spatial_reward_weight=0.1` 是否完全对应论文公式，需要结合论文原文验证。

## 5. 组会讲法

### 本文档核心结论

1. `已从代码确认`：论文 pipeline 可以在代码中落成三条主线：ST-Bench 数据 split、Qwen3TS 模型改造、EasyR1 S-GRPO 训练。
2. `已从代码确认`：Time Series Encoder 对应 `TimeSeriesEmbedding`，不是论文里的抽象黑箱；它在 `Qwen3TSForCausalLM.forward()` 中被调用。
3. `已从代码确认`：S-GRPO 的核心代码不是 reward function 文件，而是 `RayPPOTrainer` 中 original/no-graph reward 对比和 spatial reward 加权注入。
4. `合理推测`：Graph Prompting 在本仓库中主要体现为数据样本的 `Graph Structure` 文本，而不是独立模块。
5. `论文概念暂未定位到代码`：ST-SFT 的官方主线训练入口和独立 Graph Prompting 模块尚未找到。

### 组会可讲版本

我把论文里的概念逐个对到了代码。ST-Bench 对应 `data/ST-Bench/` 这个数据根目录，里面按阶段拆成 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test。Stage 1 用 ST-Align 做 time series alignment；Stage 2 主线代码用 ST-CoT 做 reasoning cold start；Stage 3 用 ST-RL 进入 EasyR1 的 GRPO / S-GRPO。

模型侧，论文里的 Time Series Encoder 在代码里就是 `TimeSeriesEmbedding`。它把数值时间序列切 patch、过 MLP，得到和 Qwen hidden size 对齐的 embedding，再合并到文本 token 序列里。

空间结构侧，代码里没有单独 GNN 或 graph encoder。图结构主要写在 prompt 的 `Graph Structure` 文本中。S-GRPO 的关键做法是构造一个 no-graph 对照：删除 `Graph Structure`，但保留同一条 time series。训练时比较原 prompt 和 no-graph prompt 的 reward，如果带图结构的回答更好，就给 spatial reward bonus，再进入 GRPO 的组内 advantage 和 policy update。

最后，推理和评估是文件级链路：推理脚本生成 `generated_answer.json`，评估脚本读取它和测试集，按任务算 accuracy 或 MAE/MAPE。这里不需要重新训练也能讲清代码结构，但不能声称已经复现训练曲线或论文数值。

### 后续需要验证的问题

1. 下载真实 ST-Bench，核对 `input` 中的 `Graph Structure` 是否全部由数据预生成。
2. 对照论文原文，确认 ST-SFT 与 ST-CoT、ST-RL 的定义差异。
3. 对照论文公式，确认代码中的 `original_r > no_graph_r * 0.8` 和 `spatial_reward_weight` 是否等同于论文 S-GRPO 设置。
4. 检查是否有未纳入主线的脚本或分支使用 `ST-SFT` split。
