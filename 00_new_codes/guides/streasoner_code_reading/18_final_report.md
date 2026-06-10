# 18 STReasoner 代码阅读综合报告

本文是对 `docs/streasoner_code_reading/01_repo_map.md` 到 `17_group_meeting_qa.md` 的压缩复盘。结论来自静态代码阅读和仓库已有结果目录；没有运行完整训练、DeepSpeed、RL 或大模型推理。

## 1. 背景与目标

STReasoner 的论文目标是让 LLM 处理时空时间序列推理：既要读多个节点的历史数值序列，又要利用节点之间的空间结构关系，完成 forecasting、entity、etiological、correlation 等任务。

本周代码阅读目标不是复现论文指标，而是回答四个问题：

1. 数据样本如何从 JSONL 进入 prompt 和 time series tensor。
2. 普通 Qwen3 如何被扩展为能接收 time series 的 Qwen3TS。
3. SFT、RL、inference、evaluation 在仓库中如何连接。
4. 哪些结论已经由代码确认，哪些还需要真实数据或运行验证。

当前最重要的边界：我没有跑通完整训练，也没有跑大模型推理。已有 `exp_STReasoner-8B/` 只能作为仓库自带结果分析，不能说成本周复现结果。

## 2. 代码仓库结构

仓库可以按五条主线理解：

| 主线 | 关键位置 | 作用 |
|---|---|---|
| 模型改造 | `base_model/Config-Qwen3-8B/` | 定义 processor、config、`TimeSeriesEmbedding`、`Qwen3TSForCausalLM` 和 embedding merge |
| SFT | `src/train.py`, `src/llamafactory/`, `scripts/qwen3-8b/train_stage1*.sh` | Stage 1/2 监督微调入口 |
| RL / S-GRPO | `src/EasyR1/verl/`, `src/EasyR1/examples/reward_function/str.py`, `scripts/qwen3-8b/train_stage1+2+3*_spatial.sh` | Stage 3 GRPO / S-GRPO 训练 |
| Inference | `inference/inference_tsmllm_vllm.py`, `inference/llm_utils.py`, `inference/vllm/chatts_vllm.py` | 读取 ST-Test、构造 prompt、调用 vLLM TS 模型、写 prediction |
| Evaluation | `evaluation/evaluate.py`, `evaluation/evaluate_qa.py` | 读取 prediction 和 dataset，计算 accuracy / MAE / MAPE |

最核心的模型文件是 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`。其中 `TimeSeriesEmbedding` 定义在 `:43-179`，`Qwen3TSForCausalLM` 定义在 `:352` 附近，`_merge_input_ids_with_time_series_features()` 定义在 `:387-513`，forward 中调用 TS encoder 和 merge 在 `:580-588`。

## 3. 数据与任务

代码中 ST-Bench 主要通过路径和 dataset registry 组织，而不是一个单独 dataloader 类。`data/dataset_info.json` 注册了 ST-Align、ST-CoT、ST-SFT、ST-RL、ST-Test 等 split；主线字段是：

- `input`：完整题面，通常包含 node 描述、`<ts><ts/>` 占位符、`Graph Structure` 和问题。
- `timeseries`：真实数值时间序列列表，和 `<ts><ts/>` 顺序对应。
- `output`：标准答案，多选任务通常是 A-D，forecasting 任务是数值序列。

推理时，`inference/inference_tsmllm_vllm.py:98-149` 的 `load_dataset()` 和 `prepare_batches()` 直接把 `sample["input"]` 放入 `question_list`，把 `sample.get("timeseries", [])` 放入 `ts_list`。这说明推理阶段不再重新拼接 graph text 或 question，样本的 `input` 已经是完整 prompt 主体。

四类 reasoning 任务可以这样讲：

| 任务 | 输出形态 | 评估 |
|---|---|---|
| `reasoning_entity` | 多选 A-D | `evaluate_multiple_choice_predictions()` 算 accuracy |
| `reasoning_etiological` | 多选 A-D | 同上 |
| `reasoning_correlation` | 多选 A-D | 同上 |
| `reasoning_forecasting` | 数值序列 | `evaluate_forecasting_predictions()` 算 MAE / MAPE |

## 4. 模型结构

STReasoner 相比普通 Qwen3 的关键差异是：文本 token embedding 之外，新增了 time series embedding token。

模型输入可以概括为：

```text
文本 prompt:
  Node 0: <ts><ts/> ... Graph Structure ... question ...

真实数值:
  timeseries = [node0_series, node1_series, ...]

模型内部:
  text token embeddings + TS patch embeddings -> Qwen3Model -> lm_head
```

time series 接入分三步：

1. `processing_qwen3_ts.py:24-50` 的 `sp_encoding()` 对序列做归一化，生成 value/mask 结构。
2. `TimeSeriesEmbedding.forward()` 根据 mask 计算有效长度和 patch 数，公式是 `(valid_lengths + patch_size - 1) // patch_size`，见 `modeling_qwen3_ts.py:80-87`。
3. patch 经过 MLP 投影到 LLM hidden size，再由 merge 函数插入文本 embedding 序列，见 `modeling_qwen3_ts.py:171-179`、`:580-588`。

一个直观例子：如果某节点时间序列长度为 16，`patch_size=8`，则 patch 数是 2，会产生 2 个 TS embedding token。

需要强调的边界：代码中没有定位到独立 GNN 或 graph encoder。空间结构主要以 `Graph Structure` 文本出现在 prompt 中。

## 5. 训练流程

README 和脚本把训练分成三阶段：

1. Stage 1：time series alignment SFT  
   入口是 `scripts/qwen3-8b/train_stage1.sh`。它通过 `deepspeed --num_gpus 8 src/train.py --stage sft` 启动，使用 `--dataset "alignment"` 和 `--template "STReasoner-Align"`。

2. Stage 2：reasoning cold-start SFT  
   入口是 `scripts/qwen3-8b/train_stage1+2.sh` 或 `train_stage2.sh`。主线继续从 Stage 1 输出训练，使用 `entity_cot, etiological_cot, correlation_cot, forecasting_cot` 和 `STReasoner-CoT` 模板。

3. Stage 3：RL / S-GRPO  
   入口是 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh`。它调用 `python3 -m src.EasyR1.verl.trainer.main`，读取四类 `ST-Bench/ST-RL/*.jsonl`，reward function 是 `src/EasyR1/examples/reward_function/str.py:compute_score`。

训练风险很高：SFT 默认 8 卡 DeepSpeed ZeRO-3；RL 默认 8 卡、Ray、vLLM rollout、EasyR1/verl 和 Docker 环境。当前只做了静态分析，没有运行训练。

## 6. 推理与评估

推理主链路：

```text
inference/inference_tsmllm_vllm.py
  -> load_dataset()
  -> prepare_batches()
  -> append prompt_suffix
  -> LLMClient(engine="vllm-ts")
  -> llm_batch_generate()
  -> generated_answer.json
```

关键点：

- `DEFAULT_TASK_CONFIG` 把任务映射到 `data/ST-Bench/ST-Test/*.jsonl`，见 `inference/inference_tsmllm_vllm.py:41-61`。
- `LLMClient` 把文本 prompt 和 `multi_modal_data.timeseries` 一起交给 vLLM，见 `inference/llm_utils.py:311-336`。
- vLLM TS 模型注册发生在 `inference/vllm/chatts_vllm.py:761-765`。
- 输出 `generated_answer.json` 的主逻辑在 `inference/inference_tsmllm_vllm.py:296-322`。

评估主链路：

```text
evaluation/evaluate.py
  -> load_jsonl_dataset()
  -> load_prediction_files()
  -> evaluate_predictions_for_task()
  -> evaluation_metrics.json
```

评估不加载模型，只做文件级指标计算。多选任务抽 `<answer>` 中的 A-D 算 accuracy；forecasting 任务解析数值序列算 MAE / MAPE，见 `evaluation/evaluate_qa.py:198-330`。

当前注意点：README 的 evaluation 示例没有显式传 `--dataset`，但 `evaluation/evaluate.py` 默认路径是 `data/reasoning/*.jsonl`；如果真实数据只在 `data/ST-Bench/ST-Test/`，建议显式传 `--dataset`。

## 7. S-GRPO

我对 S-GRPO 的代码理解是：它不是单独训练一个 spatial reward model，而是在 GRPO 训练中加入一个 spatial 对照分支。

vanilla GRPO 和 S-GRPO 的主线配置大体相同：同一模型、同一组 ST-RL 数据、同一个 `str.py:compute_score` reward function、同样的 rollout `n=8`。S-GRPO 脚本额外打开：

```text
algorithm.enable_spatial_reward=true
algorithm.spatial_reward_weight=0.1
data.enable_spatial_reward=true
```

证据在 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`。

S-GRPO 的实现分两层：

1. 数据层：`RLHFDataset` 额外构造 no-graph prompt。`remove_graph_structure()` 删除 `Graph Structure:` 到 `please analyze` 前的片段，见 `src/EasyR1/verl/utils/dataset.py:35-56`、`:313-334`。
2. trainer 层：`RayPPOTrainer._compute_spatial_reward()` 比较 original reward 和 no-graph reward。如果 `original_r > no_graph_r * 0.8`，就给 spatial reward，见 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`。之后把 `spatial_reward_weight * spatial_reward` 加到 `token_level_scores`，见 `:724-729`。

尚未确认的是：论文公式中的 S-GRPO 是否逐项等同于代码中的 `0.8` 阈值和 `spatial_reward_weight=0.1` 设置。

## 8. 当前进展

已确认：

1. 仓库主线结构已经清楚：`base_model` 做 Qwen3TS，`llamafactory` 做 SFT，`EasyR1/verl` 做 RL，`inference` 做 vLLM TS 推理，`evaluation` 做指标。
2. 数据主字段是 `input / output / timeseries`，推理阶段直接使用 `input` 作为完整题面。
3. `TimeSeriesEmbedding` 是 time series encoder 的代码落点，会把时间序列 patchify 并投影到 Qwen hidden size。
4. embedding merge 是模型核心：TS embedding 会替换/扩展 `<ts><ts/>` 对应位置，和文本 embedding 进入同一 attention 序列。
5. S-GRPO 代码实现了 original/no-graph reward 对照和 spatial bonus 注入。
6. evaluation 不加载模型，适合作为下周最低风险复现入口。

未做：

1. 未下载真实 ST-Bench JSONL。
2. 未运行 vLLM 推理。
3. 未运行 DeepSpeed SFT。
4. 未运行 EasyR1 / Ray / RL。
5. 未把论文 S-GRPO 公式逐项和代码核对。

## 9. 风险与不确定点

最大环境风险：

- README 推荐 8 x A100-SXM4-80GB 和 CUDA 12.8，见 `README.md:56-58`。
- 主 `requirements.txt` 固定 `torch==2.6.0`、`transformers==4.52.4`、`vllm==0.8.5`。
- RL Docker 镜像是 `cu12.9-vllm0.11.0`，和 inference 主环境不完全一致。
- `requirements.txt` 中 flash-attn wheel 面向 Linux / cp310 / cu12 / torch2.6，Windows 原生环境风险高。

最大代码不确定点：

1. 真实样本中 `<ts><ts/>` 数量、`timeseries` 列表、`patch_cnt` 和 TS embedding 顺序是否在所有任务上严格一致。
2. vLLM 路径的 prompt replacement、`merge_multimodal_embeddings()` 和 cache 行为没有实际运行验证。
3. 当前已有 `exp_STReasoner-8B` 的 `generated_answer.json` 可能来自旧脚本，因为它和当前 inference 脚本的 `num_tokens` 字段不完全一致。
4. S-GRPO 论文公式和代码阈值/权重设置尚未逐项核对。

## 10. 下周计划

建议按最低风险到最高风险推进：

1. 下载真实 ST-Bench 数据，先只检查文件结构和字段，不加载模型。
2. 用真实 test JSONL + 仓库已有 `exp_STReasoner-8B` 跑 evaluation-only，验证指标脚本和 prediction 格式。
3. 如果模型、vLLM 0.8.5 和 GPU 就绪，只跑 `max_samples=1` 的 inference sanity check，不覆盖已有结果。
4. 用一条真实样本核对 `<ts><ts/>`、`timeseries`、patch 数和 embedding merge 的对齐。
5. 对照论文原文，核对 ST-SFT / ST-CoT / S-GRPO 公式和代码实现的关系。
6. 只有在环境和资源明确满足条件后，再评估是否尝试 Stage 1/2 小规模训练；Stage 3 RL 放到专门复现阶段。

### 3 个最重要收获

1. STReasoner 的核心模型改造不是把时间序列转成文本，而是把数值序列 patchify 成 embedding token 后插入 LLM。
2. 代码中的 spatial structure 主要通过 `Graph Structure` prompt 文本进入模型；没有定位到独立 GNN 模块。
3. S-GRPO 的核心实现是 original prompt 和 no-graph prompt 的 reward 对照，再把 spatial bonus 加进 GRPO score。

### 3 个最大不确定点

1. 真实 ST-Bench 样本格式和所有 `<ts><ts/>` / `timeseries` 对齐关系尚未验证。
2. vLLM TS 推理路径尚未实际跑通，HF merge 和 vLLM merge 的运行时一致性尚未验证。
3. 论文 S-GRPO 数学公式与代码中的 `original_r > no_graph_r * 0.8`、`spatial_reward_weight` 尚未逐项核对。

### 3 个下一步最小行动

1. 下载 ST-Bench，并写一个轻量检查脚本统计字段、任务、`<ts><ts/>` 数和 `timeseries` 长度。
2. 用已有 `exp_STReasoner-8B` 对真实 test set 跑一次 evaluation-only。
3. 在模型和环境就绪后，只对单任务单样本运行 `--max_samples 1` inference sanity check。
