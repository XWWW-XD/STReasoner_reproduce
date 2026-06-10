# 10 SFT 训练流程

本文档静态分析 STReasoner 的 SFT 训练入口，范围包括 `scripts/qwen3-8b/` 的 Stage 1 / Stage 2 脚本、`src/train.py`、`src/llamafactory` 的 SFT 分发逻辑、数据集注册、模板、多模态 time series 插件，以及 `ds_config/ds_config_3.json`。本文不运行训练。

术语约定：

- `已从代码确认`：能直接绑定到本仓库文件、类、函数或配置行。
- `根据代码推断，未由真实数据验证`：由静态代码串联得到，但没有训练日志或真实数据内容验证。
- `尚未确认`：证据不足，已写入 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 入口调用链

### 1.1 Shell 到 Python

`已从代码确认`：Qwen3-8B Stage 1 脚本入口是：

```bash
NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py ...
```

见 `scripts/qwen3-8b/train_stage1.sh:1`。Stage 1+2 和独立 Stage 2 脚本也使用同样入口，见 `scripts/qwen3-8b/train_stage1+2.sh:1` 和 `scripts/qwen3-8b/train_stage2.sh:1`。

SFT 调用链可以画成：

```text
bash scripts/qwen3-8b/train_stage1.sh
  -> deepspeed --num_gpus 8 --master_port=19901 src/train.py
    -> src/train.py:main()
      -> llamafactory.train.tuner.run_exp()
        -> read_args()
        -> get_train_args()
        -> _training_function()
          -> if finetuning_args.stage == "sft": run_sft()
            -> load_tokenizer()
            -> get_template_and_fix_tokenizer()
            -> get_dataset(..., stage="sft")
            -> load_model(...)
            -> SFTDataCollatorWith4DAttentionMask(...)
            -> CustomSeq2SeqTrainer(...)
            -> trainer.train()
```

对应证据：

- `src/train.py` 只调用 `run_exp()`，见 `src/train.py:15-28`。
- `run_exp()` 读取参数并调用 `_training_function()`，见 `src/llamafactory/train/tuner.py:102-118`。
- `_training_function()` 用 `get_train_args(args)` 解析参数，并在 `finetuning_args.stage == "sft"` 时调用 `run_sft()`，见 `src/llamafactory/train/tuner.py:52-72`。
- `run_sft()` 依次加载 tokenizer、template、dataset、model、data collator 和 trainer，见 `src/llamafactory/train/sft/workflow.py:41-93`。
- `training_args.do_train` 为真时调用 `trainer.train()`，保存模型、metrics 和 state，见 `src/llamafactory/train/sft/workflow.py:95-120`。

### 1.2 数据流入口

`已从代码确认`：数据集名字来自脚本的 `--dataset`，会映射到 `data/dataset_info.json`。Stage 1 的 `alignment` 指向 `ST-Bench/ST-Align/alignment_train.jsonl`，列映射是 `input -> prompt`、`output -> response`、`timeseries -> timeseries`，见 `data/dataset_info.json:1-9`。

`已从代码确认`：Stage 1+2 / Stage 2 使用四个 CoT 数据集：

- `entity_cot` -> `ST-Bench/ST-CoT/entity_cot.jsonl`
- `etiological_cot` -> `ST-Bench/ST-CoT/etiological_cot.jsonl`
- `correlation_cot` -> `ST-Bench/ST-CoT/correlation_cot.jsonl`
- `forecasting_cot` -> `ST-Bench/ST-CoT/forecasting_cot.jsonl`

见 `data/dataset_info.json:18-49`。

`已从代码确认`：dataset converter 会把原始 `input/output/timeseries` 转成内部字段 `_prompt/_response/_timeseries`，见 `src/llamafactory/data/converter.py:84-135`。

`已从代码确认`：SFT supervised processor 在存在 `timeseries` 时调用 `template.mm_plugin.process_messages(..., timeseries=timeseries)`，并检查 `<ts>` token 数量和 `timeseries` 数量是否匹配，见 `src/llamafactory/data/processor/supervised.py:31-75`。

`已从代码确认`：`ChatTSPlugin` 用 processor 把 time series 注入 prompt 文本，并通过 `processor.encode_timeseries(timeseries)` 生成模型输入中的 `timeseries` tensor，见 `src/llamafactory/data/mm_plugin.py:1999-2034`。

## 2. Stage 1 参数解释

Stage 1 脚本是 `scripts/qwen3-8b/train_stage1.sh`。

| 参数 | Stage 1 值 | 作用 | 证据 |
|---|---|---|---|
| `deepspeed --num_gpus` | `8` | 用 8 张 GPU 启动分布式训练 | `scripts/qwen3-8b/train_stage1.sh:1` |
| `--master_port` | `19901` | DeepSpeed/分布式通信端口 | `scripts/qwen3-8b/train_stage1.sh:1` |
| `--deepspeed` | `ds_config/ds_config_3.json` | 使用 DeepSpeed ZeRO-3 配置 | `scripts/qwen3-8b/train_stage1.sh:2` |
| `--stage` | `sft` | 进入 LLaMA-Factory 的 SFT 分支 | `scripts/qwen3-8b/train_stage1.sh:3`, `src/llamafactory/train/tuner.py:69-72` |
| `--model_name_or_path` | `./base_model/Qwen3-8B` | 从初始化后的 Qwen3TS base model 开始训练 | `scripts/qwen3-8b/train_stage1.sh:4` |
| `--dataset` | `alignment` | 使用 ST-Align alignment train 数据 | `scripts/qwen3-8b/train_stage1.sh:5`, `data/dataset_info.json:1-9` |
| `--interleave_probs` | `1` | 单数据集采样概率为 1 | `scripts/qwen3-8b/train_stage1.sh:6` |
| `--do_train` | true | 执行训练而非只评估/预测 | `scripts/qwen3-8b/train_stage1.sh:7`, `src/llamafactory/train/sft/workflow.py:95-120` |
| `--mix_strategy` | `interleave_over` | 数据混合策略；单数据集时影响很小 | `scripts/qwen3-8b/train_stage1.sh:8`, `src/llamafactory/hparams/data_args.py:66-73` |
| `--template` | `STReasoner-Align` | 使用 alignment 模板 | `scripts/qwen3-8b/train_stage1.sh:9`, `src/llamafactory/data/template.py:1978-1991` |
| `--finetuning_type` | `full` | 全参数微调 | `scripts/qwen3-8b/train_stage1.sh:10`, `src/llamafactory/hparams/finetuning_args.py:411-414` |
| `--output_dir` | `./output/Qwen3-8B-stage1` | 保存 Stage 1 checkpoint | `scripts/qwen3-8b/train_stage1.sh:11` |
| `--overwrite_output_dir` | true | 允许覆盖输出目录 | `scripts/qwen3-8b/train_stage1.sh:12` |
| `--per_device_train_batch_size` | `2` | 每张 GPU micro batch size | `scripts/qwen3-8b/train_stage1.sh:13` |
| `--gradient_accumulation_steps` | `32` | 梯度累积步数 | `scripts/qwen3-8b/train_stage1.sh:14` |
| `--lr_scheduler_type` | `cosine` | cosine 学习率调度 | `scripts/qwen3-8b/train_stage1.sh:15` |
| `--logging_steps` | `1` | 每步记录日志 | `scripts/qwen3-8b/train_stage1.sh:16` |
| `--save_steps` | `100` | 每 100 step 保存一次 | `scripts/qwen3-8b/train_stage1.sh:17` |
| `--learning_rate` | `1e-5` | 全局学习率 | `scripts/qwen3-8b/train_stage1.sh:18` |
| `--timeseries_sft_lr` | `1e-5` | 试图给 time-series encoder 单独设置 SFT 学习率 | `scripts/qwen3-8b/train_stage1.sh:19`, `src/llamafactory/hparams/finetuning_args.py:448-456` |
| `--warmup_ratio` | `0.02` | warmup 占总训练步比例 | `scripts/qwen3-8b/train_stage1.sh:20` |
| `--num_train_epochs` | `0` | 不按 epoch 控制训练 | `scripts/qwen3-8b/train_stage1.sh:21` |
| `--max_steps` | `1000` | Stage 1 实际训练 1000 optimization steps | `scripts/qwen3-8b/train_stage1.sh:22` |
| `--plot_loss` | true | 保存 loss 曲线 | `scripts/qwen3-8b/train_stage1.sh:23`, `src/llamafactory/train/sft/workflow.py:111-120` |
| `--fp16` | true | 使用 fp16 混合精度 | `scripts/qwen3-8b/train_stage1.sh:24`, `src/llamafactory/hparams/parser.py:397-405` |
| `--save_only_model` | true | 只保存模型权重相关内容 | `scripts/qwen3-8b/train_stage1.sh:25` |
| `--save_safetensors` | `False` | 不保存 safetensors 格式 | `scripts/qwen3-8b/train_stage1.sh:26` |
| `--preprocessing_num_workers` | `96` | 数据预处理进程数 | `scripts/qwen3-8b/train_stage1.sh:27`, `src/llamafactory/hparams/data_args.py:82-85` |
| `--trust_remote_code` | `True` | 允许加载自定义 Qwen3TS 模型/processor 代码 | `scripts/qwen3-8b/train_stage1.sh:28` |
| `--cutoff_len` | `10000` | tokenized 输入最大长度 | `scripts/qwen3-8b/train_stage1.sh:29`, `src/llamafactory/hparams/data_args.py:46-49` |

`根据代码推断，未由真实训练验证`：Stage 1 的有效全局 batch 大致是 `8 GPUs * 2 per_device_train_batch_size * 32 gradient_accumulation_steps = 512` 条样本/optimizer step。实际 token 数还受样本长度、padding、数据 collator 和 DeepSpeed 行为影响。

`已从代码确认`：DeepSpeed 配置 `ds_config/ds_config_3.json` 使用 ZeRO stage 3，且 batch size、micro batch size、gradient accumulation、fp16 enabled 和 gradient clipping 都设为 `"auto"`，由 HF/DeepSpeed 根据 CLI training args 接管，见 `ds_config/ds_config_3.json:1-24`。

## 3. Stage 2 参数解释

这里需要区分两个脚本：

- `scripts/qwen3-8b/train_stage1+2.sh`：README 主线，用 Stage 1 checkpoint 继续 Stage 2，输出 `Qwen3-8B-stage1+2`。
- `scripts/qwen3-8b/train_stage2.sh`：独立 Stage 2 变体，直接从 `base_model/Qwen3-8B` 训练 CoT 数据，输出 `Qwen3-8B-stage2`。

### 3.1 README 主线：train_stage1+2.sh

`已从代码确认`：README 把 `train_stage1.sh` 标注为 `STReasoner-8B-Align`，把 `train_stage1+2.sh` 标注为 `STReasoner-8B-CoT`，见 `README.md:96-104`。

`train_stage1+2.sh` 相比 Stage 1 的关键变化：

| 参数 | Stage 1 | Stage 1+2 | 变化含义 |
|---|---|---|---|
| `--model_name_or_path` | `./base_model/Qwen3-8B` | `./output/Qwen3-8B-stage1` | 从 Stage 1 alignment checkpoint 继续训练 |
| `--dataset` | `alignment` | `entity_cot,etiological_cot,correlation_cot,forecasting_cot` | 从 alignment 数据切换到四类 reasoning CoT 数据 |
| `--interleave_probs` | `1` | `0.25,0.25,0.25,0.25` | 四类任务均匀采样 |
| `--template` | `STReasoner-Align` | `STReasoner-CoT` | 切换到 CoT 输出格式模板 |
| `--output_dir` | `./output/Qwen3-8B-stage1` | `./output/Qwen3-8B-stage1+2` | 保存 Stage 1+2 checkpoint |
| `--max_steps` | `1000` | `400` | Stage 2 继续训练 400 steps |

证据见 `scripts/qwen3-8b/train_stage1.sh:4-22` 和 `scripts/qwen3-8b/train_stage1+2.sh:4-22`。

`已从代码确认`：`STReasoner-CoT` 模板的 default system 是：

```text
Output Format:
<think>Your step-by-step reasoning process that justifies your answer</think>
<answer>Your final answer</answer>
```

见 `src/llamafactory/data/template.py:1994-2007`。这说明 Stage 2 模板明确鼓励模型输出 reasoning trace 和 final answer。

### 3.2 独立 Stage 2：train_stage2.sh

`已从代码确认`：`train_stage2.sh` 和 `train_stage1+2.sh` 的 CoT 数据、模板、batch、学习率、`max_steps=400` 基本一致；主要差别是 `--model_name_or_path "./base_model/Qwen3-8B"`、`--output_dir "./output/Qwen3-8B-stage2"`，见 `scripts/qwen3-8b/train_stage2.sh:4-22`。

`根据代码推断，未由真实训练验证`：`train_stage2.sh` 更像 ablation/变体入口，用来观察不经过 Stage 1 alignment、直接在 CoT 数据上冷启动 SFT 的效果；README 主线则是先 Stage 1 再 Stage 2。

## 4. Stage 1 vs Stage 2

| 维度 | Stage 1 alignment | Stage 2 reasoning cold start / CoT |
|---|---|---|
| 主脚本 | `scripts/qwen3-8b/train_stage1.sh` | `scripts/qwen3-8b/train_stage1+2.sh` |
| 起始模型 | `./base_model/Qwen3-8B` | `./output/Qwen3-8B-stage1` |
| 数据 | `alignment` | `entity_cot, etiological_cot, correlation_cot, forecasting_cot` |
| 数据文件 | `ST-Bench/ST-Align/alignment_train.jsonl` | `ST-Bench/ST-CoT/*.jsonl` |
| 模板 | `STReasoner-Align` | `STReasoner-CoT` |
| system prompt | `"You are a helpful assistant."` | 要求 `<think>...</think><answer>...</answer>` |
| steps | 1000 | 400 |
| 学习率 | global `1e-5`, TS `1e-5` | global `1e-5`, TS `1e-5` |
| batch 相关 | 8 GPU, per-device 2, grad accum 32 | 同 Stage 1 |
| cutoff | 10000 | 10000 |
| DeepSpeed | ZeRO-3 `ds_config_3.json` | 同 Stage 1 |

`已从代码确认`：Stage 1 和 Stage 2 都是 `--stage sft`、`--finetuning_type full`，所以都会进入同一个 `run_sft()` 训练流程，见 `scripts/qwen3-8b/train_stage1.sh:3-10` 和 `scripts/qwen3-8b/train_stage1+2.sh:3-10`。

`根据代码推断，未由真实数据验证`：Stage 1 alignment 的训练目标是让初始化后的 Qwen3TS 模型学习基本的 spatial-temporal instruction following 和 time-series 输入对齐，因为它使用 `ST-Align/alignment_train.jsonl`、`STReasoner-Align` 模板和 `timeseries` 字段。

`根据代码推断，未由真实数据验证`：Stage 2 reasoning cold start 的训练目标是让 Stage 1 后的模型学习四类 reasoning 任务的 CoT 输出格式和任务解法，因为它使用四类 `*_cot` 数据集，并且模板 default system 明确要求 `<think>` 推理过程和 `<answer>` 最终答案。

## 5. 和论文方法的对应

`已从代码确认`：README 的训练流程先准备 base model，再运行 Stage 1 和 Stage 1+2 SFT，随后才进入 RL，见 `README.md:88-128`。因此 SFT 在整体 pipeline 中位于 RL 之前。

和论文方法可以对应为：

1. `已从代码确认`：模型结构层面，`base_model/Qwen3-8B` 应先由普通 Qwen3 复制自定义 Qwen3TS 文件并初始化 `ts_encoder`，README 命令见 `README.md:88-94`。
2. `已从代码确认`：Stage 1 使用 `ST-Align` 的 `alignment_train.jsonl`，数据列包含 `input/output/timeseries`，见 `data/dataset_info.json:1-9`。
3. `已从代码确认`：Stage 2 使用 `ST-CoT` 四类 reasoning 数据，见 `data/dataset_info.json:18-49`。
4. `已从代码确认`：两个 SFT 阶段都通过 `ChatTSPlugin` 把 `timeseries` 交给 processor 编码，见 `src/llamafactory/data/mm_plugin.py:1999-2034`。
5. `已从代码确认`：`SupervisedDatasetProcessor` 会把 prompt 部分 label mask 成 `IGNORE_INDEX`，只对 response target 计算监督 loss；相关逻辑见 `src/llamafactory/data/processor/supervised.py:57-102`。

### Time series encoder 学习率相关参数

`已从代码确认`：脚本设置了两个学习率：

- `--learning_rate 1e-5`：全局学习率，见 `scripts/qwen3-8b/train_stage1.sh:18` 和 `scripts/qwen3-8b/train_stage1+2.sh:18`。
- `--timeseries_sft_lr 1e-5`：time-series encoder 专用学习率，见 `scripts/qwen3-8b/train_stage1.sh:19` 和 `scripts/qwen3-8b/train_stage1+2.sh:19`。

`已从代码确认`：`FinetuningArguments` 定义 `train_timeseries_modules=True`，以及 `timeseries_sft_lr`，见 `src/llamafactory/hparams/finetuning_args.py:439-456`。

`已从代码确认`：`CustomSeq2SeqTrainer.create_optimizer()` 在 optimizer 创建后调用 `maybe_apply_timeseries_sft_lr()`，见 `src/llamafactory/train/sft/trainer.py:82-89`。该函数会尝试把 `ts_encoder` 参数拆成单独 param group，并设置 `llamafactory_group="timeseries"` 和 `lr=timeseries_sft_lr`，见 `src/llamafactory/model/model_utils/timeseries.py:186-268`。

`已从代码确认`：训练日志中还会尝试记录 `ts_encoder_learning_rate`，见 `src/llamafactory/train/sft/trainer.py:91-97` 和 `src/llamafactory/train/sft/workflow.py:95-102`。

`尚未确认的重要风险`：当前 `base_model/Config-Qwen3-8B/config.json` 和 `configuration_qwen3_ts.py` 的 `model_type` 是 `"qwen3"`，见 `base_model/Config-Qwen3-8B/config.json:61` 和 `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py:136`；但 `TIMESERIES_MODELS` 只注册了 `"chatts"` 和 `"qwen3ts"`，见 `src/llamafactory/model/model_utils/timeseries.py:67-79`。`maybe_apply_timeseries_sft_lr()` 只有在 `model.config.model_type` 命中注册表时才会覆盖 TS 学习率，见 `src/llamafactory/model/model_utils/timeseries.py:203-209`。因此脚本设置的 `--timeseries_sft_lr` 在当前静态代码下存在不生效风险，需要实际训练日志或修复注册表后验证。

### Batch / cutoff / DeepSpeed 相关参数

`已从代码确认`：控制训练规模和显存的主要参数是：

- GPU 数：`deepspeed --num_gpus 8`，见 `scripts/qwen3-8b/train_stage1.sh:1`。
- 每 GPU batch：`--per_device_train_batch_size 2`，见 `scripts/qwen3-8b/train_stage1.sh:13`。
- 梯度累积：`--gradient_accumulation_steps 32`，见 `scripts/qwen3-8b/train_stage1.sh:14`。
- 最大 token 长度：`--cutoff_len 10000`，见 `scripts/qwen3-8b/train_stage1.sh:29`。
- DeepSpeed 配置：`--deepspeed ds_config/ds_config_3.json`，见 `scripts/qwen3-8b/train_stage1.sh:2`。
- ZeRO stage：`"stage": 3`，见 `ds_config/ds_config_3.json:15-24`。
- FP16：脚本 `--fp16`，DeepSpeed `fp16.enabled="auto"`，见 `scripts/qwen3-8b/train_stage1.sh:24` 和 `ds_config/ds_config_3.json:7-14`。

`已从代码确认`：parser 会设置 `model_args.model_max_length = data_args.cutoff_len`，并设置 `training_args.remove_unused_columns = False`，后者对 multimodal dataset 很重要，见 `src/llamafactory/hparams/parser.py:343-346` 和 `src/llamafactory/hparams/parser.py:397-407`。

## 6. 组会讲法

### 本文档核心结论

1. `已从代码确认`：SFT 入口不是独立脚本逻辑，而是 DeepSpeed 启动 `src/train.py`，再进入 LLaMA-Factory 的 `run_exp -> run_sft -> CustomSeq2SeqTrainer`。
2. `已从代码确认`：Stage 1 使用 `alignment` 数据和 `STReasoner-Align` 模板，从 `base_model/Qwen3-8B` 训练 1000 steps。
3. `已从代码确认`：Stage 1+2 使用四类 `*_cot` 数据和 `STReasoner-CoT` 模板，从 `output/Qwen3-8B-stage1` 继续训练 400 steps。
4. `已从代码确认`：两个阶段都通过 `ChatTSPlugin` 和 processor 处理 `timeseries`，并用 supervised loss 训练 response。
5. `尚未确认`：`--timeseries_sft_lr` 是否真的命中 `ts_encoder` 参数组存在静态风险，因为模型配置 `model_type="qwen3"` 与 time-series 注册表的 `"qwen3ts"` 不一致。

### 组会可讲版本

SFT 分两步。第一步是 alignment：从初始化后的 Qwen3TS base model 出发，用 `ST-Align/alignment_train.jsonl` 做 1000 step 全参数 SFT，目标是让模型先学会 STReasoner 的基本输入格式，也就是文本 prompt、图结构描述和 time series 张量如何共同进入模型。

第二步是 reasoning cold start / CoT：从 Stage 1 checkpoint 继续训练，用 entity、etiological、correlation、forecasting 四类 CoT 数据，各 0.25 采样概率，训练 400 step。模板从 `STReasoner-Align` 换成 `STReasoner-CoT`，system prompt 要求输出 `<think>` 推理过程和 `<answer>` 最终答案。

训练入口可以在 PPT 上画成：

```text
train_stage1.sh / train_stage1+2.sh
  -> deepspeed src/train.py
  -> run_exp()
  -> get_train_args()
  -> run_sft()
  -> get_dataset + ChatTSPlugin
  -> load_model(Qwen3TS)
  -> CustomSeq2SeqTrainer.train()
```

需要提醒组会：脚本虽然设置了 `timeseries_sft_lr=1e-5`，但静态代码显示它依赖 `model.config.model_type` 命中 `"qwen3ts"` 或 `"chatts"` 注册；当前配置文件写的是 `"qwen3"`，所以这个专用学习率是否生效需要后续用日志验证。

### 后续需要验证的问题

1. `尚未确认`：未下载真实 ST-Align / ST-CoT 数据，因此 Stage 1 alignment 和 Stage 2 CoT 的真实样本内容、答案格式和任务分布尚未验证。
2. `尚未确认`：未运行训练，因此 `--timeseries_sft_lr` 是否实际生成 `llamafactory_group="timeseries"` 参数组、日志中是否出现 `ts_encoder_learning_rate` 尚未验证。
3. `尚未确认`：未运行 DeepSpeed，因此 `ds_config_3.json` 的 auto batch 参数和脚本 batch 参数在实际环境中的解析结果尚未验证。
4. `尚未确认`：Stage 2 “reasoning cold start” 这一命名来自论文/任务语义和脚本数据命名的对应；代码本身只显示它使用四类 CoT 数据继续 SFT。
