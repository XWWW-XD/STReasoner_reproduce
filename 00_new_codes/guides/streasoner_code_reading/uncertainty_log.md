# Uncertainty Log

## 03_data_format.md

1. `尚未确认`：本地没有 `data/ST-Bench/` 原始 JSONL，无法抽取 3 条真实不同任务样本。当前字段说明来自 `data/dataset_info.json`、推理/训练/评估代码和已有推理输出。

2. `根据代码推断，未由真实数据验证`：普通时序样本应包含 `input`、`output`、`timeseries`；text-only 应只有 `input`、`output`；image-only 应包含 `input`、`output`、`images`。

3. `根据代码推断，未由真实数据验证`：`input` 中的 `<ts><ts/>` 占位符数量应与 `timeseries` 列表长度一致；真实数据需要下载后核对。

4. `尚未确认`：`ST-SFT` 在 `data/dataset_info.json` 中注册，但本次静态搜索未发现官方 Qwen3-8B 脚本直接使用这些 dataset 名称；它和 `ST-CoT` 的真实差异需要看数据内容。

5. `尚未确认`：`evaluation/evaluate.py` 默认读取 `data/reasoning/*.jsonl`，而推理和 dataset registry 指向 `data/ST-Bench/ST-Test/*.jsonl`；需要下载数据后确认 README 的 evaluation 命令是否缺少 `--dataset`。

6. `尚未确认`：本地已有 `exp_STReasoner-8B/*/generated_answer.json` 没有 `num_tokens` 字段，而当前推理脚本会写该字段；这些输出可能来自旧版脚本或不同导出逻辑。

## 04_prompt_construction.md

1. `尚未确认`：本地没有原始 ST-Bench JSONL，不能确认所有任务的 `input` 是否都严格包含角色说明、Node 占位符、`Graph Structure` 和问题四段。

2. `尚未确认`：真实样本中 `<ts><ts/>` 数量是否总是等于 `timeseries` 列表长度，需要下载数据后逐行检查。

3. `尚未确认`：SFT CoT 数据的 `output` 是否已经包含 `<think>/<answer>`，还是主要由 template/default_system 约束生成格式，需要查看真实 `ST-CoT/*.jsonl`。

4. `尚未确认`：`prompt_suffix` 与原始 `input` 是否存在重复输出格式说明，需要下载数据后检查原始 `input` 末尾。

## 05_inference_flow.md

1. `尚未确认`：当前没有实际运行 inference，因此 `generated_answer.json` 的当前脚本格式尚未由新运行结果验证；本地已有示例输出缺少当前代码会写出的 `num_tokens`。

2. `尚未确认`：代码阅读显示 `--max_tokens` 和 `--temperature` 对 `vllm-ts` worker 可能不生效，因为 worker 使用内部 `SamplingParams`；需要小规模运行或代码修复后验证。

3. `尚未确认`：真实 ST-Test 中所有 `timeseries` 嵌套格式是否都能被 `prepare_batches()` 正确转为 float list，需要下载数据后逐样本检查。

4. `尚未确认`：单卡 sanity check 是否能加载合并后的 STReasoner-8B，需要本地模型、vLLM 环境和显存条件验证。

## 06_model_overview.md

1. `尚未确认`：本地没有完整 `base_model/Qwen3-8B/` 权重目录，尚未验证 `initial_model.py` 是否能成功加载自定义 `Qwen3TSForCausalLM` 并初始化保存 `ts_encoder`。
2. `尚未确认`：没有实际跑 forward，因此尚未用真实 batch 验证 HF 路径中 `_merge_input_ids_with_time_series_features()` 的长度扩展与 `labels` 对齐是否和训练时完全一致。
3. `尚未确认`：`base_model/Config-Qwen3-8B/config.json` 中 `model_type` 是 `"qwen3"`，但训练侧 `TIMESERIES_MODELS` 注册的是 `"chatts"` 和 `"qwen3ts"`；这是否影响 `timeseries_sft_lr`、冻结 TS 模块或 LoRA target 的自动识别，需要在训练配置阅读或小规模 dry check 中确认。
4. `尚未确认`：vLLM 路径和 HF 路径的 TS merge 实现不完全相同；当前只确认了静态流程，尚未用同一条样本比较两条路径的 token/embedding 对齐结果。

## 07_time_series_encoder.md

1. `尚未确认`：当前没有真实 ST-Bench batch 和实际 forward 结果，因此尚未用运行结果验证 `sp_encoding()` 输出、batch padding、`TimeSeriesEmbedding.reshape()` 三者在所有真实样本上的形状完全一致。
2. `尚未确认`：`sp_encoding()` docstring 允许 1D 或 2D 原始序列，但 `07_time_series_encoder.md` 的形状解释主要针对常见 1D 序列；真实数据中是否存在 2D 序列以及 2D 情况下语义如何，需要下载数据后检查。
3. `尚未确认`：`07_time_series_encoder.md` 对 patchify 设计动机的解释来自代码结构和常见 Transformer 多模态设计，没有引用论文原文逐句确认。

## 08_embedding_merge.md

1. `尚未确认`：没有实际 forward 验证复杂 batch 下 `final_attention_mask`、`position_ids`、`new_token_positions` 是否和训练/推理期望完全一致。
2. `尚未确认`：没有用真实样本验证 prompt 中 `<ts><ts/>` 顺序、`timeseries` 列表顺序、`patch_cnt` 顺序和 `time_series_features` 顺序是否在所有数据路径中严格一致。
3. `尚未确认`：没有展开 vLLM 内部 `merge_multimodal_embeddings()` 源码，也没有比较 HF 与 vLLM 两条路径的最终 embedding 对齐结果。
4. `尚未确认`：generation 防重复注入逻辑已从代码确认，但没有通过真实 `generate()` 小样本运行验证。

## 09_generation_logic.md

1. `尚未确认`：没有通过真实 HF `generate()` 小样本验证 `timeseries` 是否确实只在 first forward pass 使用。
2. `尚未确认`：没有验证扩展后的 `attention_mask` 经父类 `_update_model_kwargs_for_generation()` 后，在所有 cache 实现上都能正确追加新 token。
3. `尚未确认`：没有验证 `cache_position` 在首轮 TS 扩展后是否和当前 Transformers 版本完全兼容。
4. `尚未确认`：官方 vLLM 推理路径的 prefill/decode cache 行为没有在 `09_generation_logic.md` 中展开源码验证。

## 10_sft_training_flow.md

1. `尚未确认`：未下载真实 ST-Align / ST-CoT 数据，因此 Stage 1 alignment 和 Stage 2 CoT 的真实样本内容、答案格式和任务分布尚未验证。
2. `尚未确认`：未运行训练，因此 `--timeseries_sft_lr` 是否实际生成 `llamafactory_group="timeseries"` 参数组、日志中是否出现 `ts_encoder_learning_rate` 尚未验证。
3. `尚未确认`：未运行 DeepSpeed，因此 `ds_config_3.json` 的 `auto` batch 参数和脚本 batch 参数在实际环境中的解析结果尚未验证。
4. `尚未确认`：Stage 2 “reasoning cold start” 这一命名来自论文/任务语义和脚本数据命名的对应；代码本身只显示它使用四类 CoT 数据继续 SFT。

## 11_sgrpo_rl.md

1. `尚未确认`：未下载真实 `data/ST-Bench/ST-RL/*.jsonl`，因此没有验证真实 prompt 中 `Graph Structure: ... please analyze` 片段是否总能被 `remove_graph_structure()` 正则正确删除。
2. `尚未确认`：未运行 Ray / EasyR1 / vLLM rollout，因此没有验证 original prompt 和 no-graph prompt 在真实 batch 中的 shape、padding、TS placeholder、`multi_modal_data["timeseries"]` 对齐是否完全正确。
3. `尚未确认`：尚未把论文 S-GRPO 数学公式逐项对照代码；当前只能确认代码实现了 “original reward 与 no-graph reward 对比 + spatial bonus 加权注入 GRPO score”。
4. `尚未确认`：`original_r > no_graph_r * 0.8` 这个阈值来自代码实现，尚未确认它是否与论文公式或实验设置完全一致。
5. `尚未确认`：text-only / image-only spatial 脚本属于 ablation 还是额外实验，需要结合论文实验表和真实数据说明确认。
6. `尚未确认`：未实际运行 Stage 3，因此 checkpoint 保存路径、WandB 日志、reward metrics 和 `spatial_reward` 曲线没有运行结果验证。

## 12_evaluation_flow.md

1. `尚未确认`：README 的 evaluation 命令没有显式传 `--dataset`，而 `evaluation/evaluate.py` 默认读取 `data/reasoning/*.jsonl`；如果本地只存在 `data/ST-Bench/ST-Test/*.jsonl`，是否需要补 `--dataset` 仍需下载数据后验证。
2. `尚未确认`：已有 `exp_STReasoner-8B` 的 forecasting metrics 没有 `mape` 和 `target_stats`，但当前 `evaluation/evaluate_qa.py` 会计算这些字段；已有结果是否来自旧版评估脚本尚未确认。
3. `尚未确认`：已有 `generated_answer.json` 没有 `num_tokens`，但当前推理脚本会写 `num_tokens`；token stats 是否能在当前推理输出上正常写入 `evaluation_metrics.json` 尚未运行验证。
4. `尚未确认`：alignment 分支中数值相对误差的缩进逻辑是否为 bug，以及是否影响任何官方实验结果，需要后续小样本或单元测试验证。
5. `尚未确认`：本次没有重新运行 evaluation，因此文档中的已有指标来自现有 `exp_STReasoner-8B` 文件，而不是本轮复现结果。

## 13_paper_to_code_mapping.md

1. `尚未确认`：本文没有逐字核对论文 PDF 原文，因此“论文中的含义”主要来自用户给出的论文题目、README 三阶段说明、脚本命名和代码行为。
2. `尚未确认`：`Graph Prompting` 没有定位到独立代码模块；当前只确认图结构以 `Graph Structure` 文本出现在 prompt 中，真实数据生成 graph prompt 的过程可能不在仓库内。
3. `尚未确认`：`ST-SFT` split 已在 `data/dataset_info.json` 注册，但没有在官方 Qwen3-8B 主线脚本中定位到使用入口；它与 `ST-CoT`、`ST-RL` 的真实职责差异需要看论文或真实数据。
4. `尚未确认`：论文公式级 S-GRPO 尚未逐项对照代码，尤其是 `original_r > no_graph_r * 0.8` 与 `spatial_reward_weight=0.1` 的公式来源和实验设置。
5. `尚未确认`：没有下载真实 ST-Bench，因此无法验证所有样本的 `input` 是否都已经预先包含完整 `Graph Structure` prompt。

## 14_single_sample_trace.md

1. `尚未确认`：本文使用的是结构化伪样本，不是从真实 ST-Bench JSONL 抽取的样本；真实字段内容和样本分布仍需下载数据后验证。
2. `尚未确认`：没有实际运行 vLLM 推理，因此 prompt replacement 后的 placeholder 数、TS patch embedding 数和最终 embedding 序列长度未由运行结果验证。
3. `尚未确认`：没有用真实样本验证 `<ts><ts/>` 数量是否始终等于 `timeseries` 列表长度。
4. `尚未确认`：伪输出中的 `num_tokens` 只是结构占位；当前脚本会写该字段，但本地已有 `exp_STReasoner-8B` 示例输出没有该字段。
5. `尚未确认`：没有实际运行 evaluation，因此伪样本评分逻辑只来自静态代码阅读。

## 15_environment_risks.md

1. `尚未确认`：真实机器上 Python 3.10、CUDA 12.8、torch 2.6.0、flash-attn 2.7.2.post1 的组合是否能一次安装成功；当前只是根据 README 和 requirements 做静态判断。
2. `尚未确认`：vLLM 0.8.5 与 `inference/vllm/chatts_vllm.py` 的自定义 TS multimodal 注册逻辑是否能在本地 GPU 环境正常加载 STReasoner-8B。
3. `尚未确认`：单卡 `max_samples=1` inference sanity check 的最低显存需求；代码中 `gpu_memory_utilization=0.95` 和 `CTX_LENGTH=6500` 只能说明风险，不等于实测显存。
4. `尚未确认`：已有 `exp_STReasoner-8B` 结果重新用当前 `evaluation/evaluate.py` 复算时，是否与已有 `evaluation_metrics.json` 完全一致。
5. `尚未确认`：RL Docker 镜像中的 vLLM 0.11.0 与本仓库 inference 侧 vLLM 0.8.5 自定义 TS patch 是否存在 API 差异；本周没有进入 Docker 运行验证。
6. `尚未确认`：如果硬件少于官方 README 的 8 x A100-SXM4-80GB，需要如何系统性缩小 batch、tensor parallel、rollout batch、DeepSpeed 和 Ray 配置；当前文档只给出风险方向。

## 16_group_meeting_ppt_outline.md

1. `尚未确认`：PPT 大纲没有新增实验或运行验证；所有未确认项继承 `03_data_format.md` 到 `15_environment_risks.md` 的记录。
2. `尚未确认`：PPT 中关于论文问题定义和 S-GRPO 公式语义的表述仍需后续与论文 PDF 原文逐项核对。
3. `尚未确认`：PPT 建议展示的已有 `exp_STReasoner-8B` 指标只能作为仓库已有结果，不能作为本周重跑复现结果。

## 17_group_meeting_qa.md

1. `尚未确认`：问答稿没有新增任何实验或运行验证；其中关于未跑通训练、未跑通大模型推理和未下载真实 ST-Bench 的边界，继承前面所有专题文档。
2. `尚未确认`：问答中关于“核心技术贡献”的表述来自 README、脚本和代码映射，仍需后续与论文 PDF 原文逐项核对。
3. `尚未确认`：问答中对“我的研究方向”的回答是通用表达，需要根据真实个人研究方向在组会前微调。

## 18_final_report.md

1. `尚未确认`：最终综合报告没有新增运行验证；其“已确认/未确认”结论来自前面静态代码阅读文档的汇总。
2. `尚未确认`：报告中的论文目标和核心贡献概括主要来自 README、脚本命名和代码映射，仍需后续与论文 PDF 原文逐项核对。
3. `尚未确认`：报告建议的下周计划尚未执行，尤其是下载真实 ST-Bench、复算已有结果和单样本 vLLM inference sanity check。

## 19_self_check.md

1. `尚未确认`：自检本身没有新增真实运行验证，只是对既有 Markdown 和静态代码阅读结论做一致性审查。
2. `尚未确认`：自检没有逐行机器验证每一句自然语言是否都有代码行号；它只抽查了关键结论、证据标签和高风险不一致点。
3. `尚未确认`：风险表中的补救动作尚未执行，尤其是真实 ST-Bench 下载、evaluation-only 复算、vLLM 单样本推理和论文公式核对。
