# 01 仓库结构地图

> 证据标记说明：
> `已从代码确认` 表示结论直接来自本仓库文件、类、函数、脚本参数或 README。
> `合理推测` 表示基于代码结构和 README 说明推断，尚未通过运行或论文逐句对齐验证。
> `尚未确认` 表示当前静态阅读无法确认，需要后续数据、实验或论文公式对照。

## 1. 目录树

`已从代码确认`：当前实际项目根目录为 `D:\working\responsitories\reproduce\STReasoner\STReasoner`。仓库 2-3 层结构如下。

```text
STReasoner/
├── README.md
├── download_dataset.py
├── download_model.py
├── initial_model.py
├── model_merger.py
├── requirements.txt
├── base_model/
│   ├── base_model_config/
│   ├── Config-Qwen2.5-7B/
│   ├── Config-Qwen2.5-14B-Instruct/
│   ├── Config-Qwen3-4B-Instruct-2507/
│   ├── Config-Qwen3-8B/
│   └── Config-Qwen3-14B/
├── data/
│   └── dataset_info.json
├── ds_config/
│   ├── ds_config_0.json
│   ├── ds_config_1.json
│   ├── ds_config_2*.json
│   ├── ds_config_3*.json
│   ├── zero1_no_optimizer.json
│   └── zero3.json
├── evaluation/
│   ├── evaluate.py
│   └── evaluate_qa.py
├── exp_STReasoner-8B/
│   ├── reasoning_correlation-STReasoner-8B/
│   ├── reasoning_entity-STReasoner-8B/
│   ├── reasoning_etiological-STReasoner-8B/
│   └── reasoning_forecasting-STReasoner-8B/
├── figures/
│   ├── method.png
│   └── streasoner.png
├── inference/
│   ├── inference_tsmllm_vllm.py
│   ├── llm_utils.py
│   ├── prompt.json
│   ├── prompt_utils.py
│   └── vllm/
│       └── chatts_vllm.py
├── scripts/
│   ├── qwen3-4b-instruct/
│   ├── qwen3-8b/
│   ├── qwen3-14b/
│   └── qwen3-vl-8b-instruct/
└── src/
    ├── train.py
    ├── api.py
    ├── cli_demo.py
    ├── webui.py
    ├── llamafactory/
    │   ├── data/
    │   ├── hparams/
    │   ├── model/
    │   ├── train/
    │   ├── chat/
    │   ├── eval/
    │   └── webui/
    └── EasyR1/
        ├── examples/
        ├── verl/
        ├── scripts/
        ├── tests/
        └── verl.egg-info/
```

## 2. 功能分区

| 功能分区 | 代码位置 | 证据等级 | 作用说明 |
|---|---|---:|---|
| base model / 模型改造 | `base_model/Config-Qwen3-8B/`, `base_model/Config-Qwen3-14B/`, `base_model/Config-Qwen3-4B-Instruct-2507/`, `base_model/Config-Qwen2.5-*` | 已从代码确认 | 这些目录包含 `configuration_*_ts.py`、`processing_*_ts.py`、`modeling_*_ts.py`，对应在 Qwen 模型上加入时间序列处理器和 `ts_encoder`。例如 `Qwen3TSConfig` 在 `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py:25`，`Qwen3TSForCausalLM` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:352`。 |
| training / SFT | `src/train.py`, `src/llamafactory/train/`, `src/llamafactory/data/`, `src/llamafactory/model/model_utils/timeseries.py` | 已从代码确认 | SFT 入口是 `src/train.py:18` 的 `main()`，调用 `run_exp()`；`src/llamafactory/train/tuner.py:69-72` 根据 `finetuning_args.stage == "sft"` 分发到 `run_sft()`；`src/llamafactory/train/sft/workflow.py:41` 定义 SFT 主流程。 |
| RL / S-GRPO | `src/EasyR1/verl/`, `src/EasyR1/examples/reward_function/str.py`, `scripts/qwen3-*/train_*_w_spatial.sh` | 已从代码确认 | RL 主入口是 `python3 -m src.EasyR1.verl.trainer.main`，示例见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:7`；空间奖励开关在同脚本 `algorithm.enable_spatial_reward=true` 和 `data.enable_spatial_reward=true`，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`。 |
| inference | `inference/`, `inference/vllm/chatts_vllm.py` | 已从代码确认 | 推理脚本 `inference/inference_tsmllm_vllm.py:218` 定义 `main()`；它使用 `LLMClient(engine="vllm-ts")`，见 `inference/inference_tsmllm_vllm.py:79-83`；vLLM 时间序列模型注册在 `inference/vllm/chatts_vllm.py:761-765`。 |
| evaluation | `evaluation/evaluate.py`, `evaluation/evaluate_qa.py` | 已从代码确认 | `evaluation/evaluate.py:100` 是评估 CLI 入口；`evaluation/evaluate_qa.py:198` 处理 forecasting，`evaluation/evaluate_qa.py:278` 处理多选 reasoning 任务。 |
| data | `data/dataset_info.json`, 下载后的 `data/ST-Bench/` | 已从代码确认 | 数据集注册表把 `input`、`output`、`timeseries` 映射为 LLaMA-Factory 所需列，见 `data/dataset_info.json:2-7`；RL 数据路径如 `ST-Bench/ST-RL/entity_rl.jsonl` 在 `data/dataset_info.json:82-87`。 |
| scripts | `scripts/qwen3-8b/`, `scripts/qwen3-14b/`, `scripts/qwen3-4b-instruct/`, `scripts/qwen3-vl-8b-instruct/` | 已从代码确认 | 脚本按模型规模和训练阶段组织。README 给出 Stage 1/2 SFT 与 Stage 3 RL 调用示例，见 `README.md:86-137`。 |
| configs | `ds_config/`, `src/EasyR1/examples/config.yaml`, `base_model/*/config.json` | 已从代码确认 | `ds_config/` 是 SFT DeepSpeed 配置；RL 默认配置通过 `config=./src/EasyR1/examples/config.yaml` 注入，见 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:8`；模型配置由 `base_model/Config-Qwen3-8B/config.json` 等提供。 |
| third-party modified framework | `src/llamafactory/`, `src/EasyR1/` | 合理推测 | README 明确致谢 LLaMA-Factory、EasyR1、Verl、ChatTS 和 vLLM，见 `README.md:190`。本仓库把这些框架代码 vendored 到 `src/` 下，并加入时间序列和空间奖励相关改动。改动边界需要后续与上游仓库 diff 验证。 |
| 示例输出 | `exp_STReasoner-8B/` | 已从代码确认 | 该目录包含 `generated_answer.json` 和 `evaluation_metrics.json`，可作为推理输出和评估输出格式样例；它不是训练实现代码。 |

## 3. 关键文件优先级

| 优先级 | 文件路径 | 所属模块 | 主要作用 | 关键类/函数 | 为什么对理解论文重要 | 证据等级 |
|---:|---|---|---|---|---|---:|
| 1 | `README.md` | 项目总入口 | 给出论文任务、三阶段训练、数据、推理、评估命令。 | 三阶段表在 `README.md:44-50`；S-GRPO 命令在 `README.md:123-126`；推理评估入口在 `README.md:142-169`。 | 这是把论文概念和代码命令连起来的第一张地图，适合组会开场说明。 | 已从代码确认 |
| 2 | `data/dataset_info.json` | data | 注册 ST-Bench 各 split 和列映射。 | `alignment` 映射在 `data/dataset_info.json:2-7`；`entity_cot` 在 `data/dataset_info.json:18-23`；`entity_rl` 在 `data/dataset_info.json:82-87`；`forecasting_test` 在 `data/dataset_info.json:138-143`。 | 论文里的 ST-Bench、ST-Align、ST-CoT、ST-RL、ST-Test 在这里落到实际 JSONL 路径和字段。 | 已从代码确认 |
| 3 | `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py` | base model / 模型配置 | 定义 Qwen3 时间序列版本的配置类。 | `Qwen3TSConfig` 在 `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py:25`；`model_type = "qwen3"` 在 `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py:136`。 | 它说明模型仍以 Qwen3 配置体系为骨架，时间序列配置通过 `config.ts` 供模型类读取。 | 已从代码确认 |
| 4 | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py` | base model / processor | 把原始时间序列转成规范化数值张量，并把 `<ts><ts/>` 占位符替换为带统计信息的 prompt 片段。 | `sp_encoding()` 在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24`；`Qwen3TSProcessor` 在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:52`；`__call__()` 在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:72`；`encode_timeseries()` 在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:173`。 | 这是“时间序列进入 LLM prompt 和张量输入”的第一站，解释了数值序列如何被标准化、padding 和传入模型。 | 已从代码确认 |
| 5 | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py` | base model / 模型改造 | 定义时间序列编码器，并把时间序列 patch embedding 合并进 Qwen3 token embedding 序列。 | `TimeSeriesEmbedding` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43`；`Qwen3TSForCausalLM` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:352`；`self.ts_encoder = TimeSeriesEmbedding(config.ts)` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:363-364`；`_merge_input_ids_with_time_series_features()` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387`；`forward()` 中调用 `self.ts_encoder(timeseries)` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。 | 这是模型设计的核心文件，可用于解释论文中“LLM + dedicated time series encoder”的实现。 | 已从代码确认 |
| 6 | `initial_model.py` | base model / 初始化 | 初始化 `ts_encoder` 参数并保存模型。 | `initialize_ts_encoder()` 在 `initial_model.py:8`；遍历 `model.ts_encoder.named_parameters()` 在 `initial_model.py:13`；加载模型在 `initial_model.py:56`；保存模型在 `initial_model.py:70`。 | README 的 base model preparation 调用了该脚本，见 `README.md:90-92`；它是训练前让新增时间序列模块具备初始权重的步骤。 | 已从代码确认 |
| 7 | `src/train.py` | training / SFT entry | LLaMA-Factory 训练入口封装。 | `main()` 在 `src/train.py:18`；调用 `run_exp()` 在 `src/train.py:19`；TPU 兼容入口 `_mp_fn()` 在 `src/train.py:22`。 | Stage 1 和 Stage 2 SFT 脚本通过这个入口进入 LLaMA-Factory 训练流程。 | 已从代码确认 |
| 8 | `src/llamafactory/data/mm_plugin.py` | training / SFT data adapter | 注册 STReasoner 时间序列多模态插件，把 timeseries 交给 processor 编码。 | `ChatTSPlugin` 在 `src/llamafactory/data/mm_plugin.py:1999`；`get_mm_inputs()` 返回 `{"timeseries": processor.encode_timeseries(timeseries)}`，见 `src/llamafactory/data/mm_plugin.py:2002-2015`；`process_messages()` 调用 processor 替换文本，见 `src/llamafactory/data/mm_plugin.py:2018-2034`；`"streasoner": ChatTSPlugin` 在 `src/llamafactory/data/mm_plugin.py:2057-2059`。 | 它解释 LLaMA-Factory 原本的多模态抽象如何接入时间序列。 | 已从代码确认 |
| 9 | `src/llamafactory/data/template.py` | training / SFT prompt template | 注册 STReasoner 训练模板。 | `STReasoner-Align` 在 `src/llamafactory/data/template.py:1978-1992`；`STReasoner-CoT` 在 `src/llamafactory/data/template.py:1994-2008`；CoT 默认 system 要求 `<think>` 和 `<answer>`，见 `src/llamafactory/data/template.py:2004`。 | 它把论文中的 alignment 与 cold-start reasoning 阶段映射到两个训练模板。 | 已从代码确认 |
| 10 | `src/llamafactory/model/model_utils/timeseries.py` | training / SFT optimizer adapter | 管理时间序列模块是否训练、LoRA target 和单独学习率。 | `TimeSeriesModel` 在 `src/llamafactory/model/model_utils/timeseries.py:34`；注册 `qwen3ts` 在 `src/llamafactory/model/model_utils/timeseries.py:74-79`；`maybe_apply_timeseries_sft_lr()` 在 `src/llamafactory/model/model_utils/timeseries.py:186`；`patch_timeseries_modules_for_lora()` 在 `src/llamafactory/model/model_utils/timeseries.py:282`。 | 它解释 SFT 阶段如何让新增 `ts_encoder` 参与训练，并支持独立学习率。 | 已从代码确认 |
| 11 | `src/llamafactory/train/sft/workflow.py` | training / SFT workflow | 组织 tokenizer、template、dataset、model、collator、trainer。 | `run_sft()` 在 `src/llamafactory/train/sft/workflow.py:41`；加载 template 和 dataset 在 `src/llamafactory/train/sft/workflow.py:49-53`；构造 `SFTDataCollatorWith4DAttentionMask` 在 `src/llamafactory/train/sft/workflow.py:58-67`；训练和保存模型在 `src/llamafactory/train/sft/workflow.py:95-102`。 | 这是 Stage 1/2 监督微调的数据流主干。 | 已从代码确认 |
| 12 | `src/EasyR1/verl/trainer/ray_trainer.py` | RL / S-GRPO trainer | 组织 RL rollout、reward、KL、advantage、actor update，并实现 spatial reward 注入。 | `RayPPOTrainer` 在 `src/EasyR1/verl/trainer/ray_trainer.py:158`；`_compute_spatial_reward()` 在 `src/EasyR1/verl/trainer/ray_trainer.py:466`；生成 no-graph batch 在 `src/EasyR1/verl/trainer/ray_trainer.py:526-550`；比较 original 与 no-graph reward 在 `src/EasyR1/verl/trainer/ray_trainer.py:582-595`；加权加入 token score 在 `src/EasyR1/verl/trainer/ray_trainer.py:724-729`。 | 这是 S-GRPO 与 vanilla GRPO 的主要代码差异点。 | 已从代码确认 |
| 13 | `src/EasyR1/verl/trainer/core_algos.py` | RL / GRPO algorithm | 定义 advantage estimator、GRPO advantage、policy loss、KL 等核心算法。 | `AdvantageEstimator.GRPO` 在 `src/EasyR1/verl/trainer/core_algos.py:76-86`；`compute_advantage_return()` 在 `src/EasyR1/verl/trainer/core_algos.py:120`；`compute_grpo_outcome_advantage()` 在 `src/EasyR1/verl/trainer/core_algos.py:175-215`；`compute_policy_loss()` 在 `src/EasyR1/verl/trainer/core_algos.py:409`。 | 论文中的 GRPO 优化部分需要从这里理解优势标准化和策略损失实现。 | 已从代码确认 |
| 14 | `src/EasyR1/verl/utils/dataset.py` | RL / data loader | 为 RL 构造 prompt、multi-modal data、no-graph prompt 和 ground truth。 | `remove_graph_structure()` 在 `src/EasyR1/verl/utils/dataset.py:35`；`RLHFDataset` 在 `src/EasyR1/verl/utils/dataset.py:111`；时间序列 processor 分支在 `src/EasyR1/verl/utils/dataset.py:313-319`；spatial reward 的 no-graph prompt 在 `src/EasyR1/verl/utils/dataset.py:321-334`；no-graph position_ids 后处理在 `src/EasyR1/verl/utils/dataset.py:387-405`。 | 它说明 S-GRPO 对照分支不是另一个数据集，而是在同一条样本上移除 `Graph Structure` prompt 片段。 | 已从代码确认 |
| 15 | `inference/inference_tsmllm_vllm.py` | inference | 批量加载测试集、构造问题和时间序列输入、调用 vLLM、保存 `generated_answer.json`。 | 任务默认数据集在 `inference/inference_tsmllm_vllm.py:42-61`；`answer_question_list()` 在 `inference/inference_tsmllm_vllm.py:70`；`prepare_batches()` 在 `inference/inference_tsmllm_vllm.py:123`；`main()` 在 `inference/inference_tsmllm_vllm.py:218`；保存输出在 `inference/inference_tsmllm_vllm.py:319-322`。 | 它是复现论文测试输出和组会展示推理流程的主入口。 | 已从代码确认 |

## 4. 论文概念到代码位置映射

| 论文概念 | 代码位置 | 代码证据 | 阅读说明 | 证据等级 |
|---|---|---|---|---:|
| ST-Bench | `data/dataset_info.json`, `download_dataset.py` | ST-Bench split 路径出现在 `data/dataset_info.json:3`, `data/dataset_info.json:19`, `data/dataset_info.json:83`, `data/dataset_info.json:139`。 | 数据被拆成 alignment、CoT、SFT、RL、test，以及 text/image 变体。 | 已从代码确认 |
| 三阶段训练 | `README.md`, `scripts/qwen3-8b/` | 三阶段表在 `README.md:44-50`；Stage 1/2 SFT 命令在 `README.md:97-98`；Stage 3 RL 命令在 `README.md:123-126`。 | 组会可以先用 README 解释整体 pipeline，再进入脚本参数。 | 已从代码确认 |
| 时间序列编码 | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py` | `sp_encoding()` 在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24`；均值偏移、缩放和 `<ts>` prompt 生成在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:36-50`。 | 这是原始数值序列进入模型前的 preprocessing。 | 已从代码确认 |
| `<ts>` 占位符 | `processing_qwen3_ts.py`, `modeling_qwen3_ts.py`, `template.py` | prompt 按 `<ts><ts/>` 切分在 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:123-126`；模型按 `ts_token_start_index` 和 `ts_token_end_index` 找特殊 token，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:407-410`；模板插件设置 `timeseries_token="<ts>"`，见 `src/llamafactory/data/template.py:1991` 和 `src/llamafactory/data/template.py:2007`。 | 它连接文本 prompt 和时间序列 embedding 插入位置。 | 已从代码确认 |
| time series encoder / `ts_encoder` | `modeling_qwen3_ts.py`, `timeseries.py` | `TimeSeriesEmbedding` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43`；`Qwen3TSForCausalLM` 初始化 `ts_encoder` 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:363-364`；SFT 学习率覆盖在 `src/llamafactory/model/model_utils/timeseries.py:186-268`。 | 这是模型结构和训练可学习参数的交叉点。 | 已从代码确认 |
| SFT Stage 1 alignment | `scripts/qwen3-8b/train_stage1.sh`, `data/dataset_info.json`, `template.py` | README 把 Stage 1 描述为 alignment SFT，见 `README.md:48`；`alignment` 数据项在 `data/dataset_info.json:2-7`；`STReasoner-Align` 模板在 `src/llamafactory/data/template.py:1978-1992`。 | 代码层面能确认 alignment 数据和模板存在；脚本细节建议后续单独阅读。 | 已从代码确认 |
| SFT Stage 2 cold-start reasoning | `scripts/qwen3-8b/train_stage2.sh`, `data/dataset_info.json`, `template.py` | README 把 Stage 2 描述为 cold-start reasoning SFT，见 `README.md:49`；CoT 数据项如 `entity_cot` 在 `data/dataset_info.json:18-23`；`STReasoner-CoT` 模板在 `src/llamafactory/data/template.py:1994-2008`。 | CoT 模板要求输出 `<think>` 和 `<answer>`，和 reward 的格式要求一致。 | 已从代码确认 |
| GRPO | `src/EasyR1/verl/trainer/core_algos.py`, `src/EasyR1/verl/trainer/ray_trainer.py` | `AdvantageEstimator.GRPO` 在 `src/EasyR1/verl/trainer/core_algos.py:81-83`；GRPO advantage 在 `src/EasyR1/verl/trainer/core_algos.py:175-215`；trainer 中要求 `rollout.n > 1` 的逻辑在 `src/EasyR1/verl/trainer/ray_trainer.py:233-236`。 | 这是 vanilla GRPO 公式的代码入口。 | 已从代码确认 |
| S-GRPO / spatial-aware reward | `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh`, `ray_trainer.py`, `dataset.py` | 开关和权重在 `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh:29-31`；`_compute_spatial_reward()` 在 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`；no-graph prompt 在 `src/EasyR1/verl/utils/dataset.py:321-334`；空间奖励加权进入 token score 在 `src/EasyR1/verl/trainer/ray_trainer.py:724-729`。 | 代码实现的 S-GRPO 是“原始 prompt 与去 graph prompt 的 reward 对比 + 加权奖励”。 | 已从代码确认 |
| no-graph 对照 | `src/EasyR1/verl/utils/dataset.py`, `src/EasyR1/verl/trainer/ray_trainer.py` | `remove_graph_structure()` 删除 `Graph Structure: ... please analyze` 前的片段，见 `src/EasyR1/verl/utils/dataset.py:35-56`；trainer 生成 no-graph batch 在 `src/EasyR1/verl/trainer/ray_trainer.py:526-550`。 | 它是 spatial-aware 的对照组构造方式。 | 已从代码确认 |
| reward function | `src/EasyR1/examples/reward_function/str.py` | `format_reward()` 在 `src/EasyR1/examples/reward_function/str.py:12-15`；`accuracy_reward()` 在 `src/EasyR1/examples/reward_function/str.py:71-99`；`compute_score()` 在 `src/EasyR1/examples/reward_function/str.py:101-108`。 | 奖励由格式分和准确性分组成，numeric forecasting 使用相对误差奖励。 | 已从代码确认 |
| vLLM 推理 | `inference/inference_tsmllm_vllm.py`, `inference/llm_utils.py`, `inference/vllm/chatts_vllm.py` | 推理使用 `engine="vllm-ts"`，见 `inference/inference_tsmllm_vllm.py:79-83`；`worker_vllm_ts()` 加载 `inference.vllm.chatts_vllm` 并设置 `limit_mm_per_prompt={"timeseries": 50}`，见 `inference/llm_utils.py:142-149`；模型注册在 `inference/vllm/chatts_vllm.py:761-765`。 | 这是测试阶段支持时间序列多模态输入的关键。 | 已从代码确认 |
| evaluation metrics | `evaluation/evaluate.py`, `evaluation/evaluate_qa.py` | `evaluate.py:100-175` 负责 CLI 和保存 metrics；forecasting 的 MAE/MAPE 在 `evaluation/evaluate_qa.py:198-275`；多选准确率在 `evaluation/evaluate_qa.py:278-317`；任务分发在 `evaluation/evaluate_qa.py:320-330`。 | 组会汇报指标时，需要区分 forecasting 数值指标和 entity/etiological/correlation 多选准确率。 | 已从代码确认 |
| Text/Image alternative training | `data/dataset_info.json`, `scripts/qwen3-vl-8b-instruct/` | README 的 alternative training 表在 `README.md:175-180`；text 数据项如 `entity_cot_text` 在 `data/dataset_info.json:154-160`；image 数据项如 `entity_cot_image` 在 `data/dataset_info.json:266-271`。 | 这部分是论文 ablation 或扩展实验的候选入口。 | 已从代码确认 |

## 5. 本周优先阅读路径

1. `已从代码确认`：先读 `README.md:44-50` 和 `README.md:86-169`，建立三阶段训练、推理、评估的全局流程。
2. `已从代码确认`：读 `data/dataset_info.json`，重点看 `alignment`、`*_cot`、`*_rl`、`*_test` 四类 split，明确每个阶段读哪些文件和字段。
3. `已从代码确认`：读 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py` 与 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`，按“原始 timeseries -> `sp_encoding()` -> `<ts>` prompt -> `TimeSeriesEmbedding` -> embedding merge”的顺序画数据流图。
4. `已从代码确认`：读 `src/llamafactory/data/mm_plugin.py`、`src/llamafactory/data/template.py`、`src/llamafactory/train/sft/workflow.py`，回答“LLaMA-Factory 如何接入 STReasoner 的 timeseries 输入”。
5. `已从代码确认`：读 `src/EasyR1/verl/utils/dataset.py`、`src/EasyR1/verl/trainer/ray_trainer.py`、`src/EasyR1/verl/trainer/core_algos.py`，重点定位 GRPO 与 S-GRPO 的差异。
6. `已从代码确认`：最后读 `inference/inference_tsmllm_vllm.py`、`inference/llm_utils.py`、`inference/vllm/chatts_vllm.py`、`evaluation/evaluate.py`、`evaluation/evaluate_qa.py`，形成复现推理和评估的汇报闭环。

## 6. 尚未确认的问题

1. `尚未确认`：实际 ST-Bench JSONL 样例尚未下载和查看；当前仅能确认 `data/dataset_info.json` 中声明的路径和列映射。
2. `尚未确认`：论文中的 S-GRPO 数学公式与 `RayPPOTrainer._compute_spatial_reward()`、`compute_grpo_outcome_advantage()` 的逐项一致性尚未完成；当前只能确认代码中存在 no-graph reward 对比和加权注入。
3. `尚未确认`：所有 `scripts/qwen3-*` 与论文主实验、消融实验、text/image 变体之间的完整对应关系尚未建立；当前只确认 README 和脚本名给出的阶段关系。
4. `尚未确认`：`src/llamafactory/`、`src/EasyR1/`、`inference/vllm/chatts_vllm.py` 相比上游项目的具体改动范围尚未通过 git diff 或上游仓库对比确认。
5. `尚未确认`：`exp_STReasoner-8B/` 中示例结果是否完全对应论文表格数值尚未核验；当前只能确认其是已有推理输出和评估指标样例目录。
6. `合理推测`：`Graph Structure` 是 S-GRPO 空间信息对照的关键 prompt 片段，因为 `remove_graph_structure()` 只删除该片段，见 `src/EasyR1/verl/utils/dataset.py:35-56`；该 prompt 字段在真实数据中的具体格式需要下载数据后验证。

### 本文档核心结论

`已从代码确认`：本仓库围绕 STReasoner 分为五条主线：`base_model/` 做 Qwen 时间序列模型改造，`src/llamafactory/` 做 Stage 1/2 SFT，`src/EasyR1/` 做 Stage 3 GRPO/S-GRPO，`inference/` 做 vLLM 时间序列推理，`evaluation/` 做任务指标计算。

`已从代码确认`：S-GRPO 在代码中的核心链路是 `data.enable_spatial_reward` 生成 no-graph prompt，`algorithm.enable_spatial_reward` 启用 original/no-graph reward 对比，`spatial_reward_weight` 把空间奖励加到 `token_level_scores`。

### 组会可讲版本

`已从代码确认`：我本周先完成了仓库结构静态阅读。代码不是一个单文件训练脚本，而是把 Qwen3-TS 模型、LLaMA-Factory SFT、EasyR1 RL、vLLM 推理和自定义评估串成完整 pipeline。模型侧新增 `ts_encoder`，数据侧用 `<ts>` 占位符和 `timeseries` 字段把数值序列接入 prompt，训练侧先做 SFT 再做 RL。S-GRPO 的实现入口在 EasyR1 trainer 中，它会额外构造去掉 `Graph Structure` 的 no-graph prompt，与原 prompt 的 reward 做比较，再把空间奖励加到 GRPO 的 token-level score 里。

### 后续需要验证的问题

1. `尚未确认`：下载少量 ST-Bench 样例后，核对 `input` 中 `Graph Structure`、`timeseries` 数组和 `<ts><ts/>` 占位符的真实格式。
2. `尚未确认`：把论文 S-GRPO 公式逐项对照 `src/EasyR1/verl/trainer/ray_trainer.py:466-494`、`src/EasyR1/verl/trainer/ray_trainer.py:724-729` 和 `src/EasyR1/verl/trainer/core_algos.py:175-215`。
3. `尚未确认`：逐个阅读 `scripts/qwen3-8b/`、`scripts/qwen3-14b/`、`scripts/qwen3-4b-instruct/`、`scripts/qwen3-vl-8b-instruct/`，建立“论文实验表格 -> 训练脚本 -> checkpoint 名称”的对应表。
