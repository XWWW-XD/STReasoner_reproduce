# 16 组会 PPT 大纲

说明：本 PPT 大纲基于 `docs/streasoner_code_reading/01_repo_map.md` 到 `15_environment_risks.md` 的静态代码阅读结果。没有运行训练、RL、DeepSpeed 或大模型推理；已有结果只来自仓库中的 `exp_STReasoner-8B/`。

## Slide 1 本周目标与约束

**要点**

1. 本周目标不是完整复现训练，而是建立可汇报的代码理解框架。
2. 阅读范围覆盖：数据格式、prompt、inference、模型结构、TS encoder、embedding merge、generation、SFT、S-GRPO、evaluation、环境风险。
3. 严格不跑大规模训练、不跑 RL、不下载大模型、不改核心代码。
4. 输出成果是 `docs/streasoner_code_reading/` 下 16 份 Markdown 文档。

**建议放什么图/表/代码截图**

- 表格：`任务编号 -> 文档名 -> 解决的问题`。
- 截图：`docs/streasoner_code_reading/` 文件列表。
- 小标注：`已从代码确认 / 根据代码推断，未由真实数据验证 / 尚未确认` 三类证据标签。

**我口头应该怎么讲**

这周我的定位是代码阅读和复现准备，不是宣称已经跑通完整训练。我把论文里的关键概念拆成代码问题：数据怎么进来，prompt 怎么构造，time series 怎么编码，embedding 怎么合并，推理和评估怎么走，SFT/RL 脚本分别做什么。所有结论都尽量绑定到具体文件、类和函数；没有真实数据或没有运行验证的地方，我单独放进 `uncertainty_log.md`。

## Slide 2 论文问题定义：为什么是 spatio-temporal reasoning

**要点**

1. 任务同时涉及时间序列数值变化和节点之间的空间/图结构关系。
2. 普通文本 LLM 只能读自然语言 token；STReasoner 额外接收真实 time series 数值输入。
3. 代码中的 spatial structure 主要表现为 prompt 里的 `Graph Structure` 文本，而不是独立 GNN 模块。
4. 四类 reasoning 任务覆盖 etiological、entity、correlation、forecasting，见数据和 evaluation 任务分支。

**建议放什么图/表/代码截图**

- 画一张概念图：`Node time series + Graph Structure + Question -> Reasoning Answer`。
- 截图建议：
  - `docs/streasoner_code_reading/03_data_format.md`
  - `inference/inference_tsmllm_vllm.py:41-61` 的任务到数据路径映射。
  - `evaluation/evaluate_qa.py:198-330` 的 forecasting / multiple-choice 指标分支。

**我口头应该怎么讲**

我理解这篇工作的任务不是单纯时间序列预测，也不是普通图问答。每条样本里既有多个节点的历史序列，也有节点之间的结构关系，还要回答因果、实体、相关性或预测类问题。从代码看，图结构不是作为邻接矩阵送入模型，而是写在 prompt 里的 `Graph Structure` 段落；真实数值时间序列则通过 `<ts><ts/>` 占位符和 `timeseries` 字段进入模型。

## Slide 3 方法总览：STReasoner 三阶段

**要点**

1. README 把训练流程分为三阶段：Stage 1 SFT alignment、Stage 2 SFT cold-start reasoning、Stage 3 RL with S-GRPO。
2. Stage 1 使用 `alignment` 数据和 `STReasoner-Align` 模板。
3. Stage 2 使用四类 CoT 数据和 `STReasoner-CoT` 模板。
4. Stage 3 使用 EasyR1 / verl 做 GRPO 或 S-GRPO，spatial 版本额外打开 spatial reward。

**建议放什么图/表/代码截图**

- 流程图：
  `Qwen3 base + TS config -> Stage 1 Align -> Stage 2 CoT -> Stage 3 S-GRPO -> STReasoner-8B`
- 截图建议：
  - `README.md:44-50` 三阶段表。
  - `scripts/qwen3-8b/train_stage1.sh:1-29`
  - `scripts/qwen3-8b/train_stage1+2.sh:1-29`
  - `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:7-31`

**我口头应该怎么讲**

三阶段可以讲成“先对齐输入模态，再冷启动推理，再用 RL 强化空间结构利用”。第一阶段让模型学会处理 time series 占位符和真实数值序列；第二阶段用 CoT 数据训练 reasoning 格式；第三阶段进入 EasyR1 的 GRPO，其中 S-GRPO 和 vanilla GRPO 的主要脚本差异是打开 `algorithm.enable_spatial_reward=true`、`data.enable_spatial_reward=true`，并设置 `spatial_reward_weight`。

## Slide 4 代码仓库地图

**要点**

1. `base_model/Config-Qwen3-8B/` 是模型结构核心：processor、config、modeling。
2. `inference/` 是推理主链路和 vLLM TS patch。
3. `evaluation/` 是指标计算，不加载模型。
4. `src/llamafactory/` 支撑 SFT；`src/EasyR1/verl/` 支撑 RL / S-GRPO。
5. `scripts/qwen3-8b/` 把论文三阶段落成可执行 shell 入口。

**建议放什么图/表/代码截图**

- 仓库模块地图表：
  `目录 -> 作用 -> 关键文件 -> 对应论文概念`
- 截图建议：
  - `docs/streasoner_code_reading/01_repo_map.md`
  - `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`
  - `inference/vllm/chatts_vllm.py`
  - `src/EasyR1/verl/trainer/ray_trainer.py`

**我口头应该怎么讲**

我把仓库分成五块：第一块是模型改造，负责把 Qwen3 变成 Qwen3TS；第二块是 inference，负责读 ST-Test、构造 prompt、调用 vLLM；第三块是 evaluation，只根据 `generated_answer.json` 算 accuracy 或 MAE/MAPE；第四块是 SFT，走 LLaMA-Factory；第五块是 RL，走 EasyR1/verl。这样看代码时不会迷路。

## Slide 5 数据与任务格式

**要点**

1. ST-Bench 数据统一围绕 `input / output / timeseries` 三个核心字段。
2. 推理阶段 `prepare_batches()` 直接读取 `sample["input"]` 作为 question，把 `sample["timeseries"]` 放进 `ts_list`。
3. `input` 里通常已经包含节点描述、`<ts><ts/>`、`Graph Structure` 和具体问题。
4. 多选任务输出 A-D；forecasting 输出数值序列。

**建议放什么图/表/代码截图**

- 一条伪样本结构图：
  `jsonl line -> dict(input/output/timeseries) -> question_list + ts_list`
- 四类任务对比表：`task -> input emphasis -> expected answer -> metric`。
- 截图建议：
  - `docs/streasoner_code_reading/03_data_format.md`
  - `docs/streasoner_code_reading/14_single_sample_trace.md`
  - `inference/inference_tsmllm_vllm.py:98-149`

**我口头应该怎么讲**

数据层最重要的结论是：推理代码没有重新拼 graph、question 和 node 描述，而是把样本里的 `input` 当作完整题面。时间序列不展开成普通文本，而是保存在 `timeseries` 字段，并通过题面中的 `<ts><ts/>` 占位符和真实数值列表对齐。evaluation 再用 `idx` 把 prediction 对回原始测试集。

## Slide 6 Inference 主链路

**要点**

1. 入口是 `inference/inference_tsmllm_vllm.py`，核心参数包括 `task / dataset / model_path / exp / num_gpus / num_gpus_per_process / max_samples`。
2. 主流程：load dataset -> prepare batches -> append prompt_suffix -> tokenizer stats -> LLMClient -> write `generated_answer.json`。
3. `LLMClient(engine="vllm-ts")` 会启动 vLLM worker，并把 `timeseries` 放入 `multi_modal_data`。
4. vLLM 侧必须先 import `inference.vllm.chatts_vllm` 注册自定义 TS 模型。

**建议放什么图/表/代码截图**

- 函数调用链图：
  `main() -> load_dataset() -> prepare_batches() -> answer_question_list() -> LLMClient.llm_batch_generate() -> worker_vllm_ts() -> llm.generate()`
- 截图建议：
  - `inference/inference_tsmllm_vllm.py:153-214` 参数定义。
  - `inference/inference_tsmllm_vllm.py:269-322` 主流程和输出。
  - `inference/llm_utils.py:142-149` vLLM TS worker 初始化。
  - `inference/llm_utils.py:311-336` multi-modal 输入包装。

**我口头应该怎么讲**

推理链路可以讲成文件级 pipeline。脚本先根据任务找到 JSONL，读取每条样本的 `input` 和 `timeseries`；然后给题面追加任务级输出格式约束，也就是 prompt suffix；之后 `LLMClient` 把文本 prompt 和 timeseries 一起包装给 vLLM。最后输出是 `generated_answer.json`，每条包含 `idx`、`question_text`、`response`，当前脚本还会写 `num_tokens`。

## Slide 7 Time Series Encoder

**要点**

1. 核心类是 `TimeSeriesEmbedding`，位于 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。
2. processor 先用 `sp_encoding()` 对序列做归一化和 value/mask 编码。
3. encoder 根据 mask 计算 `valid_lengths`，再用 `patch_cnt = ceil(valid_lengths / patch_size)` 切 patch。
4. 每个 patch 经过 MLP 投影到 LLM hidden size，使 TS embedding 能进入同一 attention 序列。
5. 配置中 Qwen3-8B 的 `patch_size=8`、`hidden_size=4096`，见 `base_model/Config-Qwen3-8B/config.json:73-83`。

**建议放什么图/表/代码截图**

- 示意图：
  `timeseries length 16 -> patch_size 8 -> 2 TS embedding tokens`
- 类似 ViT 但不过度类比的图：
  `连续序列 patch -> MLP projection -> LLM embedding space`
- 截图建议：
  - `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`
  - `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:80-87`
  - `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:171-179`

**我口头应该怎么讲**

Time Series Encoder 做的事是把连续数值序列变成少量 embedding token。它先根据 mask 判断有效长度，然后按 `patch_size` 切成 patch，比如长度 16、patch size 8 就得到 2 个 TS token。这样做的直观好处是减少 token 数量，同时把局部连续数值片段投影到和 Qwen token embedding 相同的 hidden size，后面就可以统一走 Transformer attention。

## Slide 8 Embedding 合并机制

**要点**

1. HF 路径核心函数是 `_merge_input_ids_with_time_series_features()`，位于 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`。
2. 特殊 token `<ts>` 和 `<ts/>` 标记 TS 占位范围，真实 token embedding 会被 TS patch embedding 替换/扩展。
3. 多个节点按 prompt 中 `<ts>` 出现顺序，顺序消费 `patch_cnt` 和 `time_series_features`。
4. 合并后重建 `inputs_embeds / attention_mask / position_ids / labels`。
5. vLLM 路径不直接走 HF merge，而是 prompt replacement + `merge_multimodal_embeddings()`。

**建议放什么图/表/代码截图**

- 合并前后示意图：
  `A <ts><ts/> B <ts><ts/> C`
  -> `A ts0_0 ts0_1 B ts1_0 ts1_1 ts1_2 C`
- 截图建议：
  - `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`
  - `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`
  - `inference/vllm/chatts_vllm.py:352-401`
  - `inference/vllm/chatts_vllm.py:699-709`

**我口头应该怎么讲**

这里是模型结构里最关键的“接缝”。文本 prompt 里有 `<ts><ts/>`，但模型不能只看这两个文本 token。代码会用 TS encoder 输出的 patch embedding 替换这些位置，并根据每条时间序列的 patch 数扩展序列长度。合并后，文本 token embedding 和 TS embedding 在同一个序列里被 Qwen3Model 处理。风险点也在这里：placeholder 数、timeseries 数、patch 数必须严格对齐。

## Slide 9 SFT/RL 训练流程

**要点**

1. SFT 入口：`deepspeed ... src/train.py`，再进入 LLaMA-Factory 的 `run_exp()`。
2. Stage 1 使用 `alignment`；Stage 2 使用 `entity_cot, etiological_cot, correlation_cot, forecasting_cot`。
3. RL 入口：`python3 -m src.EasyR1.verl.trainer.main`，配置来自 `src/EasyR1/examples/config.yaml` 和脚本覆盖参数。
4. vanilla GRPO 和 S-GRPO 的脚本差异主要是 spatial reward 开关和权重。
5. reward 分两层：`str.py` 计算 format / accuracy，trainer 里额外计算 spatial reward。

**建议放什么图/表/代码截图**

- 训练流程图：
  `Stage 1 Align SFT -> Stage 2 CoT SFT -> Stage 3 GRPO/S-GRPO -> checkpoint merge`
- 对比表：
  `train_stage1+2+3.sh` vs `train_stage1+2+3_w_spatial.sh`
- 截图建议：
  - `scripts/qwen3-8b/train_stage1.sh:1-29`
  - `scripts/qwen3-8b/train_stage1+2.sh:1-29`
  - `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:7-31`
  - `src/EasyR1/examples/reward_function/str.py:12-108`
  - `src/EasyR1/verl/trainer/ray_trainer.py:466-494`

**我口头应该怎么讲**

SFT 和 RL 要分开讲。SFT 是监督学习，脚本通过 DeepSpeed 启动 LLaMA-Factory，Stage 1 做 alignment，Stage 2 做 CoT reasoning cold start。RL 是 EasyR1/verl，模型对同一 prompt rollout 多个回答，然后 rule-based reward 打分，GRPO 做组内 advantage。S-GRPO 的额外点是构造 no-graph prompt，比较带图和去图的 reward，再把 spatial bonus 加到 token-level score 里。

## Slide 10 当前进展、未确认问题、下周计划

**要点**

1. 当前已完成静态代码链路：数据、prompt、模型、推理、评估、SFT、RL、环境风险。
2. 已有 `exp_STReasoner-8B/` 可用于展示仓库自带结果，但不能说成本周复现结果。
3. 最大未确认项：真实 ST-Bench 数据未下载、vLLM TS 路径未运行、S-GRPO 公式未和论文逐项核对、环境未安装验证。
4. 本周最低风险路径：静态阅读 + 已有结果分析 + evaluation-only；只有模型环境就绪时才做 `max_samples=1` inference sanity check。
5. 下周建议：先下载数据、复算已有结果指标，再做单样本 inference，最后再评估是否启动训练。

**建议放什么图/表/代码截图**

- 风险矩阵：
  `任务 -> 是否需要模型 -> 是否需要 GPU -> 风险等级 -> 本周是否建议`
- 截图建议：
  - `docs/streasoner_code_reading/15_environment_risks.md`
  - `docs/streasoner_code_reading/uncertainty_log.md`
  - `exp_STReasoner-8B/*/evaluation_metrics.json`

**我口头应该怎么讲**

我现在能比较完整地讲清“代码是怎么设计的”，但还不能说“训练已经复现”。高风险主要在环境：README 假设 8 张 A100 80GB，SFT/inference 主环境和 RL Docker 的 vLLM/transformers 版本也不一致。下周我建议先补真实数据，把已有结果重新用 evaluation 脚本复算；如果模型和 vLLM 环境就绪，再只跑一条 `max_samples=1` 的 inference sanity check。

## 2 分钟口头总述

这周我没有尝试硬跑完整训练，而是把 STReasoner 的代码结构系统读了一遍，并整理成文档。我的理解是，这篇工作的核心问题是让 LLM 同时处理时间序列数值和空间结构。代码里，空间结构主要以 prompt 中的 `Graph Structure` 文本出现；真实时间序列则通过 `<ts><ts/>` 占位符和 `timeseries` 数值字段进入模型。

模型侧，普通 Qwen3 被扩展成 Qwen3TS。关键新增模块是 `TimeSeriesEmbedding`：它把时间序列按 `patch_size` 切成 patch，再用 MLP 投影到 Qwen hidden size。之后 `_merge_input_ids_with_time_series_features()` 或 vLLM 侧的 `merge_multimodal_embeddings()` 把 TS embedding 插入文本 embedding 序列，使文本 token 和时间序列 token 一起进入 attention。

流程侧，推理脚本从 ST-Test JSONL 读取 `input` 和 `timeseries`，追加任务级 prompt suffix，经 `LLMClient(engine="vllm-ts")` 调用 vLLM，最后写 `generated_answer.json`。评估脚本不加载模型，只按 `idx` 对齐 prediction 和 dataset：多选任务算 accuracy，forecasting 任务算 MAE/MAPE。

训练侧，README 和脚本对应三阶段：Stage 1 是 alignment SFT，Stage 2 是 CoT reasoning cold start，Stage 3 是 GRPO 或 S-GRPO。S-GRPO 的关键代码不是单独 reward 文件，而是在 EasyR1 trainer 里构造 original prompt 和 no-graph prompt，对比 reward 后给 spatial bonus。

目前的结论主要来自静态代码阅读。还没确认的包括真实 ST-Bench 数据内容、vLLM TS 路径实际能否跑通、S-GRPO 公式和论文逐项对应关系，以及环境安装。下周我建议先下载数据并复算已有 `exp_STReasoner-8B` 的 evaluation，再做单样本 inference sanity check。

## 老师可能追问与回答

**Q1：你这周没有跑训练，那这份汇报的可信度在哪里？**

答：我不会把它说成复现实验结果。可信度来自静态代码证据：每个结论都绑定到文件、类、函数或脚本参数。例如 TS encoder 对应 `TimeSeriesEmbedding`，embedding merge 对应 `_merge_input_ids_with_time_series_features()`，S-GRPO 对应 EasyR1 trainer 里的 original/no-graph reward 对比。没有运行验证的内容都放进了 `uncertainty_log.md`。

**Q2：STReasoner 和普通 Qwen3 最大区别是什么？**

答：普通 Qwen3 只有文本 token embedding；STReasoner 新增了 time series processor、`TimeSeriesEmbedding` 和 embedding merge 逻辑。真实时间序列不是变成一长串文本数字，而是 patchify 后映射成和文本 token 同维度的 embedding，再插入同一个 attention 序列。

**Q3：图结构是怎么进模型的？有没有 GNN？**

答：目前代码中没有定位到独立 GNN 或 graph encoder。可确认的是，图结构主要以 `Graph Structure` 文本写在样本 `input` 里，由 LLM 作为自然语言 prompt 读取。S-GRPO 也是通过删除这段 graph text 构造 no-graph 对照，而不是改变一个显式邻接矩阵张量。

**Q4：S-GRPO 到底比 vanilla GRPO 多了什么？**

答：脚本层面多了 `algorithm.enable_spatial_reward=true`、`data.enable_spatial_reward=true` 和 `spatial_reward_weight`。数据层面会额外构造 no-graph prompt。trainer 层面对 original prompt 和 no-graph prompt 分别 rollout / reward，然后计算 spatial reward，把 bonus 加到 token-level score 中，再走 GRPO advantage 和 policy update。

**Q5：为什么要 patchify time series，而不是直接把每个时间点当 token？**

答：代码层面 patchify 有两个作用：减少 TS token 数量；把一个局部连续时间片段通过 MLP 投影到 LLM hidden size。这样既控制序列长度，又让数值模态能和文本 token embedding 对齐。这个类比有点像 ViT 的 patch embedding，但这里只是一维时间序列 patch，不应该过度类比到二维图像空间结构。

**Q6：推理和训练走的是同一套 embedding merge 吗？**

答：概念相同，但实现路径不同。HF 模型里有 `_merge_input_ids_with_time_series_features()` 手工重排 embedding、mask 和 position ids；官方推理脚本走 vLLM，自定义 `chatts_vllm.py` 先做 prompt replacement，再调用 vLLM 的 `merge_multimodal_embeddings()`。两条路径最终目标一致，但我还没有用同一条样本运行比较最终 embedding 对齐结果。

**Q7：如果下周只能做一个最稳的实验，你会做什么？**

答：先下载真实 ST-Bench，不加载模型，复用仓库已有 `exp_STReasoner-8B` 的 `generated_answer.json` 跑 evaluation-only。这个路径不需要 GPU，也不涉及 vLLM 和大模型加载，风险最低，还能验证数据路径、prediction 格式和指标脚本是否一致。

**Q8：环境最大风险是什么？**

答：主要是三套环境风险叠加。README 假设 8 张 A100 80GB；主 requirements 固定 `torch==2.6.0`、`transformers==4.52.4`、`vllm==0.8.5` 和 Linux flash-attn wheel；RL Docker 镜像则是 `cu12.9-vllm0.11.0`，EasyR1 侧 transformers 版本也更高。所以 SFT/inference 环境和 RL 环境不能随便混用。

**Q9：你现在最不确定的技术点是什么？**

答：第一是真实数据中 `<ts><ts/>`、`timeseries`、`patch_cnt` 是否在所有任务上完全对齐；第二是 vLLM 路径实际运行时的 multimodal merge 和 cache 行为；第三是论文公式中的 S-GRPO 和代码里 `original_r > no_graph_r * 0.8`、`spatial_reward_weight` 的逐项对应关系。这些都已经列入后续验证计划。

**Q10：这套 PPT 里能展示什么“结果”？**

答：可以展示仓库自带 `exp_STReasoner-8B/` 下的 `generated_answer.json` 和 `evaluation_metrics.json`，但必须说明这是已有结果，不是本周重跑结果。本周的主要结果是代码阅读产物和复现风险评估。
