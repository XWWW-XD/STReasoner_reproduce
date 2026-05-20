# 19 自检与一致性检查

本文档对 `docs/streasoner_code_reading/` 下已经生成的代码阅读文档做一次静态自检。自检范围是 Markdown 文档、README、scripts 和关键源码路径；没有运行训练、推理、deepspeed、RL，也没有下载模型或数据。

结论先行：目前没有发现需要立即推翻前面文档的硬矛盾；但存在若干必须在组会中明确边界的问题，尤其是 ST-SFT 与官方 Stage 2 脚本的关系、evaluation 默认路径与 README 示例的差异、现有结果文件与当前 inference 输出格式差异、S-GRPO 论文公式是否完全对应代码、以及 `timeseries_sft_lr` 是否实际命中 TS 参数组。

## 1. 文档完整性检查

### 1.1 已生成文档

已从文件系统确认：`docs/streasoner_code_reading/` 下已经覆盖任务 01-18，并包含统一的 `uncertainty_log.md`。本次新增本文档作为任务 20 输出。

| 文档 | 状态 | 备注 |
|---|---:|---|
| `01_repo_map.md` | 已生成 | 仓库地图与模块入口 |
| `02_official_pipeline.md` | 已生成 | README 与官方脚本流程 |
| `03_data_format.md` | 已生成 | 数据格式、任务类型、样本字段 |
| `04_prompt_construction.md` | 已生成 | prompt 文件与构造逻辑 |
| `05_inference_flow.md` | 已生成 | inference 主链路 |
| `06_model_overview.md` | 已生成 | 模型结构总览 |
| `07_time_series_encoder.md` | 已生成 | `TimeSeriesEmbedding` 深读 |
| `08_embedding_merge.md` | 已生成 | TS embedding 合并机制 |
| `09_generation_logic.md` | 已生成 | generation 与 cache 逻辑 |
| `10_sft_training_flow.md` | 已生成 | SFT 训练入口 |
| `11_sgrpo_rl.md` | 已生成 | RL / S-GRPO |
| `12_evaluation_flow.md` | 已生成 | evaluation 流程 |
| `13_paper_to_code_mapping.md` | 已生成 | 论文概念到代码映射 |
| `14_single_sample_trace.md` | 已生成 | 单样本端到端追踪 |
| `15_environment_risks.md` | 已生成 | 环境与运行风险 |
| `16_group_meeting_ppt_outline.md` | 已生成 | 组会 PPT 大纲 |
| `17_group_meeting_qa.md` | 已生成 | 组会问答准备 |
| `18_final_report.md` | 已生成 | 最终综合报告 |
| `uncertainty_log.md` | 已生成 | 持续记录未确认事项 |

### 1.2 结构完整性

已从文档标题检查确认：每个任务文档均包含用户要求的一级标题和核心二级标题。综合类文档 `16_group_meeting_ppt_outline.md`、`17_group_meeting_qa.md`、`18_final_report.md` 以汇报为目标，结构比源码专题文档更概括；这类文档中的高层说法应追溯到前面的专题文档，而不应被当作新的源码证据。

### 1.3 自检结论

已从文档确认：

- 任务 03-18 均已有对应 Markdown 输出。
- `uncertainty_log.md` 已记录多个“未运行验证 / 未下载真实数据 / 未核对论文公式”的边界。
- 前面文档基本遵守了“已从代码确认 / 根据代码推断 / 尚未确认”的表达方式。

尚未确认：

- 没有逐句机器验证所有自然语言结论是否都有行号支撑。
- 没有通过真实 ST-Bench 数据、真实模型 checkpoint、真实 vLLM 推理或真实 evaluation 运行来验证文档链路。

## 2. 证据链检查

### 2.1 证据链较强的结论

以下内容可以认为证据链较强，因为前面文档绑定了具体文件、函数或脚本。

| 结论 | 当前证据 | 自检判断 |
|---|---|---|
| 仓库包含 base model、SFT、RL、inference、evaluation 五类主线模块 | `README.md:44-50`，`base_model/Config-Qwen3-8B/`，`src/`，`inference/`，`evaluation/` | 可以放心讲 |
| 数据注册中存在 ST-Align、ST-SFT、ST-CoT、ST-RL、ST-Test 等命名 | `data/dataset_info.json:82-145` | 可以放心讲 |
| inference 读取 `input` 和 `timeseries` 字段，并构造 `question_list`、`ts_list` | `inference/inference_tsmllm_vllm.py:98-149`，`:272-277` | 可以放心讲 |
| prompt 构造依赖 `inference/prompt_utils.py` 和 `inference/prompt.json` | `inference/prompt_utils.py:27-38`，`inference/prompt.json` | 可以放心讲 |
| TS encoder 的核心类是 `TimeSeriesEmbedding` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179` | 可以放心讲 |
| TS 配置包含 `patch_size`、`num_features`、`hidden_size` 等字段 | `base_model/Config-Qwen3-8B/config.json:73-83` | 可以放心讲 |
| HF 路径存在 `_merge_input_ids_with_time_series_features()` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513` | 可以放心讲 |
| generation 中有避免重复注入 timeseries 的逻辑 | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-305` | 可以谨慎讲，需注明未运行验证 |
| vLLM 路径使用 prompt replacement 和 multimodal merge，而不是直接走 HF merge 函数 | `inference/vllm/chatts_vllm.py:352-401`，`:699-709` | 可以放心讲 |
| SFT 入口从 `src/train.py` 进入 LLaMA-Factory 风格训练链路 | `src/train.py:15`，`src/llamafactory/` | 可以放心讲 |
| RL 数据集类会构造 graph / no-graph 两套 prompt | `src/EasyR1/verl/utils/dataset.py:313-334` | 可以放心讲 |
| S-GRPO 代码中存在基于 no-graph reward 的 spatial bonus | `src/EasyR1/verl/trainer/ray_trainer.py:526-550` | 可以放心讲代码现象，谨慎讲论文等价性 |
| reward function 中能确认 accuracy reward 和 format reward | `src/EasyR1/examples/reward_function/str.py:12-108` | 可以放心讲 |
| evaluation 读取 `generated_answer.json` 并按任务计算指标 | `evaluation/evaluate.py:102-174`，`evaluation/evaluate_qa.py:198-330` | 可以放心讲 |

### 2.2 证据链较弱或必须标注边界的结论

| 说法 | 风险 | 正确表达 |
|---|---|---|
| “论文的完整公式已经和代码逐项对应” | 没有全文逐式核对论文公式 | 只能说“已定位到代码中的 S-GRPO 实现入口和 spatial bonus 逻辑，尚未逐项核对论文公式” |
| “Graph Prompting 是一个独立模块” | 代码中没有找到独立 graph prompting 模块 | 应说“graph/spatial 信息主要体现为 prompt 文本和 no-graph 对照数据处理” |
| “ST-SFT 就是官方 Stage 2 数据” | `data/dataset_info.json` 有 ST-SFT，但 qwen3-8b 官方主脚本 Stage 2 使用 `*_cot` | 应说“ST-SFT 在数据注册中存在；官方 qwen3-8b 脚本的 reasoning cold start 主线使用 ST-CoT 命名” |
| “当前 inference 输出一定等同于已有结果目录格式” | 现有 `generated_answer.json` 可能是旧版本输出，当前脚本写入字段更多 | 应说“当前脚本输出格式与已有结果目录需要区分，已有结果可作为历史产物分析” |
| “`timeseries_sft_lr` 一定生效” | 静态阅读发现 `model_type` 与 `TIMESERIES_MODELS` 注册名可能不完全一致 | 应说“代码提供 TS 学习率机制，但是否在 qwen3 配置下命中参数组需要运行或进一步断点确认” |
| “max_samples=1 inference 一定能跑” | 依赖模型、vLLM patch、GPU、checkpoint 是否就绪 | 应说“模型和环境已就绪时可作为最低风险 sanity check，不保证当前机器可直接运行” |

### 2.3 需要写入 uncertainty 的证据缺口

已在 `uncertainty_log.md` 中持续记录的核心缺口包括：

- 未下载和抽查真实 ST-Bench JSONL 全量数据。
- 未运行 vLLM inference。
- 未运行 evaluation sanity check。
- 未运行 SFT / RL。
- 未核对 S-GRPO 代码与论文公式的一一对应。
- 未验证 `timeseries_sft_lr` 在当前 qwen3 配置下是否实际命中 TS 参数。

## 3. 前后矛盾检查

### 3.1 未发现需要推翻的硬矛盾

自检没有发现“同一件事在不同文档中被写成相互排斥结论”的严重问题。主要问题不是硬矛盾，而是有些地方需要在组会中说明“代码注册、官方脚本、README 示例、已有结果目录”分别代表不同层面的证据。

### 3.2 需要解释的表面不一致

| 表面不一致 | 涉及文档 | 当前判断 | 组会表达建议 |
|---|---|---|---|
| `ST-SFT` 存在，但 Stage 2 主脚本用 `ST-CoT` | `03_data_format.md`，`10_sft_training_flow.md`，`13_paper_to_code_mapping.md` | 不是硬矛盾；一个是数据注册名，一个是官方脚本选择 | “代码里注册了 ST-SFT，但 qwen3-8b 官方 Stage 2 脚本采用 ST-CoT 数据配置” |
| README evaluation 示例与 `evaluate.py` 默认数据路径可能不一致 | `02_official_pipeline.md`，`12_evaluation_flow.md` | 需要显式传 `--dataset` 避免默认路径误用 | “复用 ST-Bench/ST-Test 时建议显式指定 dataset 路径” |
| 当前 inference 脚本输出字段与已有 `exp_STReasoner-8B` 结果不完全一致 | `05_inference_flow.md`，`12_evaluation_flow.md` | 可能是版本差异或历史输出 | “已有结果可用于观察结构，但不能反推当前脚本所有字段都一致” |
| inference vLLM 版本与 RL Docker vLLM 版本不同 | `15_environment_risks.md` | 说明推理与 RL 环境可能分离 | “不要把 inference 环境和 EasyR1/RL 环境混成一个统一环境” |
| HF merge 和 vLLM merge 逻辑相似但实现不同 | `08_embedding_merge.md`，`05_inference_flow.md` | 不是矛盾，是两条执行路径 | “HF 路径展示完整重排算法，vLLM 路径走 vLLM 多模态合并接口” |
| Graph Prompting 是论文概念，但代码中没有独立同名模块 | `04_prompt_construction.md`，`11_sgrpo_rl.md`，`13_paper_to_code_mapping.md` | 代码中体现为 prompt 文本和 graph/no-graph 对照 | “graph prompting 在代码里更像数据/prompt 组织方式，而不是单独模型层” |

### 3.3 关键函数漏读检查

已较充分阅读：

- `TimeSeriesEmbedding`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`
- `_merge_input_ids_with_time_series_features()`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`
- `prepare_inputs_for_generation()`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-305`
- inference batch 构造：`inference/inference_tsmllm_vllm.py:98-149`
- vLLM TS 处理：`inference/vllm/chatts_vllm.py:352-401`，`:628-709`
- RL reward 与 spatial bonus：`src/EasyR1/examples/reward_function/str.py:12-108`，`src/EasyR1/verl/trainer/ray_trainer.py:526-550`
- evaluation 指标：`evaluation/evaluate.py:102-174`，`evaluation/evaluate_qa.py:198-330`

尚未充分验证：

- vLLM 外部库 `merge_multimodal_embeddings()` 的内部实现，不在本仓库源码内。
- 论文中 S-GRPO 数学公式与 `src/EasyR1/verl/trainer/ray_trainer.py` 的逐项对应。
- 数据集生成脚本或 graph prompt 生成源头，如果存在，尚未在仓库中定位到完整链路。
- DeepSpeed / transformers / vLLM 在当前机器的实际运行行为。
- checkpoint merge 细节只做了路径级阅读，尚不足以回答所有权重合并问题。

## 4. 风险表

| 风险点 | 严重程度 | 当前证据 | 补救动作 |
|---|---:|---|---|
| 没有完整跑通训练、推理、RL | 高 | 用户约束明确禁止重型运行；文档均为静态阅读 | 组会开头主动说明“本周目标是代码阅读，不是复现指标”；下周先做 `max_samples=1` 或 evaluation-only |
| 真实 ST-Bench 数据未下载或未全量抽查 | 高 | `03_data_format.md` 基于代码读取逻辑和少量本地可见结构 | 下载官方数据后抽查每类任务 3 条样本，补齐真实字段示例 |
| S-GRPO 代码与论文公式未逐项核对 | 高 | 已定位 `ray_trainer.py:526-550` 的 spatial bonus，但未做公式级对照 | 组会谨慎说“代码实现观察”；下周对照论文公式逐项标注 |
| `timeseries_sft_lr` 是否实际生效未确认 | 高 | `10_sft_training_flow.md` 发现 `model_type` 与 TS 模型注册名可能存在命中风险 | 用轻量打印参数组或 dry-run 配置解析确认，不启动训练 |
| Graph Prompting 的生成源头未定位 | 中 | 已确认 prompt 中使用 graph text；未找到独立 graph prompting 生成模块 | 搜索数据预处理脚本和原始数据构造流程；组会不要说成独立模型模块 |
| README evaluation 示例与实际 dataset 路径存在差异 | 中 | README 示例省略 `--dataset`，`evaluate.py` 有默认 `data/reasoning/*.jsonl` | 实际复用时显式传 `--dataset data/ST-Bench/ST-Test/...jsonl` |
| 当前 inference 输出与已有结果目录格式不完全一致 | 中 | 当前脚本写 `num_tokens`，部分已有结果未见该字段 | 汇报时区分“当前脚本格式”和“已有历史结果格式” |
| vLLM inference 环境与 RL Docker 环境版本不同 | 中 | inference 注释要求 `vllm==0.8.5`；EasyR1 依赖出现 `vllm==0.11.0` | 分开建环境；不要在同一环境中混跑 inference 与 RL |
| HF merge 与 vLLM merge 未做运行等价验证 | 中 | HF 有完整 merge 函数；vLLM 走 prompt replacement + `merge_multimodal_embeddings()` | 只讲两条路径“设计上对应”；下周用单样本对齐 token 数和输出形状 |
| evaluation 指标尚未用本地结果复算 | 中 | 已读 `evaluate.py` / `evaluate_qa.py`，但未执行 | 用已有 `exp_STReasoner-8B` 结果做只读 evaluation sanity check |
| 单卡显存需求未知 | 中 | README 给出 A800/H100 多卡脚本；未做显存测量 | 组会不要承诺消费级 GPU 可跑；下周先评估模型加载需求 |
| 部分综合文档没有逐句引用源码行号 | 低 | `16_group_meeting_ppt_outline.md`、`17_group_meeting_qa.md`、`18_final_report.md` 是摘要型文档 | 把它们视为汇报材料；技术细节回到 03-15 专题文档查证 |

## 5. 组会表达边界

### 5.1 组会可以放心讲

这些内容有比较直接的代码证据，可以作为本周代码阅读的稳定结论。

1. 仓库主线可以分为数据、模型、SFT、RL、inference、evaluation 几块；对应目录分别是 `data/`、`base_model/Config-Qwen3-8B/`、`src/llamafactory/`、`src/EasyR1/`、`inference/`、`evaluation/`。
2. STReasoner 相比普通 Qwen3 增加了 time series 输入处理，核心 HF 类包括 `TimeSeriesEmbedding` 和 `Qwen3TSForCausalLM`，位置在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`。
3. Time series 会被 patchify，再经 MLP 投影到 LLM hidden size，然后作为 embedding token 进入语言模型序列。
4. HF 路径中 `_merge_input_ids_with_time_series_features()` 负责把 `<ts>` / `<ts/>` placeholder 替换为 TS embedding 片段。
5. vLLM 推理路径与 HF 路径不同：vLLM 侧先做 prompt replacement，再通过 vLLM 多模态合并接口处理 TS embedding。
6. RL 代码中可以确认有 graph/no-graph prompt 对照，并有基于 no-graph reward 的 spatial bonus 逻辑。
7. reward function 中可以确认 accuracy reward 和 format reward。
8. evaluation 代码会读取 `generated_answer.json`，并按任务类型计算分类准确率或数值误差类指标。

### 5.2 组会可以谨慎讲

这些内容可以讲，但必须带上“根据静态代码阅读”“尚未运行验证”或“需要下周验证”的边界。

1. ST-Bench / ST-Align / ST-SFT / ST-RL 的用途映射：代码注册和脚本能支撑大体对应关系，但真实数据内容还需要下载后抽查。
2. Stage 2 reasoning cold start 的训练目标：脚本和数据命名支持这个解释，但要说明官方 qwen3-8b 脚本使用的是 `*_cot` 数据，而不是直接用 `*_sft`。
3. S-GRPO 的直观解释：可以说“代码里用 with-graph 与 no-graph reward 比较形成 spatial bonus”，但不要说已经完成论文公式复现。
4. Graph Prompting 的作用：可以说“空间结构以文本 graph prompt 形式进入模型”，但不要说存在独立 graph neural network 或独立 graph prompting 模块。
5. 小规模 sanity check 建议：可以提出 `max_samples=1`、evaluation-only，但必须加前提“模型、数据、vLLM patch 和 GPU 环境已就绪”。
6. patchify 的设计动机：可以从代码结构解释“减少 token 数、对齐 LLM embedding 空间”，但若谈论文动机，需要回到论文原文核对。

### 5.3 组会不要讲成结论

以下内容目前证据不足，不应在组会中讲成已经完成或已经证明。

1. 不要说“我已经完整复现了训练 / 推理 / RL / 论文指标”。
2. 不要说“S-GRPO 代码与论文公式完全一致”，除非完成逐项公式核对。
3. 不要说“真实 ST-Bench 所有样本字段都已确认”，因为目前主要依据代码读取逻辑。
4. 不要说“vLLM 路径和 HF 路径运行结果完全一致”，目前只做了静态对应。
5. 不要说“`timeseries_sft_lr` 在 qwen3-8b 训练中一定生效”，当前只能说代码提供了机制但仍需验证命中。
6. 不要说“README 中所有命令可以在当前机器直接运行”，因为 CUDA、vLLM、deepspeed、GPU 显存和模型路径都可能阻塞。
7. 不要说“Graph Prompting 是额外模型结构”，目前代码证据更支持它是 prompt / 数据组织方式。
8. 不要承诺“单卡或本机一定能跑通 STReasoner-8B 推理”，显存和环境风险尚未实测。

### 5.4 自检后的汇报策略

建议组会采用三层表达：

1. 先讲“本周完成了静态代码阅读和文档化，没有声称复现指标”。
2. 再讲“我已经读清楚的主链路”：数据样本进入 prompt，TS 进入 encoder，embedding 合并到 LLM，vLLM 生成答案，evaluation 读取结果评分。
3. 最后主动列出“下周最小验证”：真实数据抽样、evaluation-only 复算、`max_samples=1` inference sanity check、S-GRPO 公式对照、TS 学习率参数组检查。

这样表达最稳：既能展示本周确实读了关键代码，也不会把尚未运行验证的内容包装成复现结论。
