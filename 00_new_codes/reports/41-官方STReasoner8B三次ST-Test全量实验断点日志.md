# 41-官方 STReasoner-8B 三次 ST-Test 全量实验断点日志

## 当前目标

更正：本轮真正目标不是只跑 forecasting，而是按 `00_new_codes/reports/10-ST-TEST数据集效果.md` 的口径，跑完整 ST-Test 四任务：

- `reasoning_entity`
- `reasoning_etiological`
- `reasoning_correlation`
- `reasoning_forecasting`

已有 run1 是 `exp/sttest_full_*_6144/` 四任务。本轮补齐 run2 / run3：

- `exp/sttest_full_entity_6144_run2_official/`
- `exp/sttest_full_etiological_6144_run2_official/`
- `exp/sttest_full_correlation_6144_run2_official/`
- `exp/sttest_full_forecasting_6144_run2_official/`
- `exp/sttest_full_entity_6144_run3_official/`
- `exp/sttest_full_etiological_6144_run3_official/`
- `exp/sttest_full_correlation_6144_run3_official/`
- `exp/sttest_full_forecasting_6144_run3_official/`

注意：日志中间保留了最开始误按 forecasting-only 执行的历史过程；最终报告以 `00_new_codes/reports/42-官方STReasoner8B三次ST-Test全量实验结果.md` 为准。

## 已确认的代码入口

- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- 参考报告：
  - `00_new_codes/reports/10-ST-TEST数据集效果.md`
  - `00_new_codes/reports/17-stage2.4 w/o graph实验报告.md`
  - `00_new_codes/reports/39-MLP时间序列复述错误四任务定量分析报告.md`

本轮不要用 `00_new_codes` 里的实验 wrapper 跑主实验；`00_new_codes/tools` 只能作为后续分析工具。

## 模型状态

截至 `2026-06-08 06:19:10 UTC`，官方 `STReasoner-8B` 已完整落盘到：

```text
/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B
```

本轮最初尝试用 `huggingface-cli download` 续传 cache，但 cache 下载多次长时间无增长；随后安装 `aria2`，把官方仓库 snapshot 中的小文件复制到本地模型目录，再用 aria2 顺序补齐 4 个 safetensors shard。这样避免继续依赖不稳定的 HF cache 断点。

本地模型目录大小：

```text
16G
```

`model.safetensors.index.json` 能识别 4 个 shard，`weight_map` 中共有 `410` 个 tensor。

本地目录没有 `.aria2` 或 `.incomplete` 残片。

四个 shard 的 SHA256 已校验通过：

| 文件 | SHA256 |
| --- | --- |
| `model-00001-of-00004.safetensors` | `5b04c286439204440ee90a60046a1fd52631572eccb54466a647f4eb7d2d45b1` |
| `model-00002-of-00004.safetensors` | `f17c912fb2f0d6d5e511729e29308a3657d50c80116001597809ceb3b86eca50` |
| `model-00003-of-00004.safetensors` | `b0bef738377a5abe8878f5f8f9ac22a2a67cf7b1b3d71e0ad4749d8a151515c8` |
| `model-00004-of-00004.safetensors` | `414d09f55e6c6a8b07b33b0d4e083e44bdb04e124faa78ae1073df7a45d92efc` |

## 已做空间清理

用户明确要求删除之前训练的模型后，已删除：

- `base_model/Qwen3-4B-Instruct-2507`
- `00_new_codes/reports/artifacts/model_archive/stage1_4b_training_outputs_20260608_0213`

删除后 `/root/autodl-tmp` 数据盘空间从约 `18G` 可用提升到约 `27G` 可用。

当前最新空间状态：

- 系统盘 `/`：`30G` 总量，`3.3G` 已用，`27G` 可用，`11%`
- 数据盘 `/root/autodl-tmp`：`50G` 总量，`40G` 已用，`11G` 可用，`80%`
- GPU：`NVIDIA A100-PCIE-40GB`，当前空闲，显存使用约 `1 MiB`

## 当前下载进程

当前没有需要继续保留的下载进程。

注意：`rg` 当前环境不可用，后续查询用 `grep/find`。

## run2 命令

模型完整后执行：

```bash
env HF_HUB_DISABLE_XET=1 \
    HF_HOME=/root/autodl-tmp/cache/huggingface \
    HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface \
    TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
    /root/autodl-tmp/conda/envs/str-py310/bin/python \
    inference/inference_tsmllm_vllm.py \
    --task reasoning_forecasting \
    --dataset data/ST-Bench/ST-Test/forecasting_test.jsonl \
    --model_path /root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B \
    --num_gpus 1 \
    --num_gpus_per_process 1 \
    --max_tokens 6144 \
    --temperature 0.2 \
    --exp sttest_full_forecasting_6144_run2_official \
    --output_name generated_answer.json
```

评估：

```bash
env PYTHONPATH=. \
    /root/autodl-tmp/conda/envs/str-py310/bin/python \
    evaluation/evaluate.py \
    --task reasoning_forecasting \
    --dataset data/ST-Bench/ST-Test/forecasting_test.jsonl \
    --exp_path exp/sttest_full_forecasting_6144_run2_official \
    --pred_pattern generated_answer
```

## run2 守门检查

- `generated_answer.json` 存在。
- 样本数为 `280`。
- idx 覆盖 `0..279` 且无重复。
- `evaluation_metrics.json` 生成。
- 无明显空输出、模型加载错误、OOM、路径错误。

run2 正常后再跑 run3；run2 如果有严重问题，先停下来报告，不直接跑 run3。

## run3 命令

run2 守门通过后，执行同口径 run3：

```bash
env HF_HUB_DISABLE_XET=1 \
    HF_HOME=/root/autodl-tmp/cache/huggingface \
    HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface \
    TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface \
    /root/autodl-tmp/conda/envs/str-py310/bin/python \
    inference/inference_tsmllm_vllm.py \
    --task reasoning_forecasting \
    --dataset data/ST-Bench/ST-Test/forecasting_test.jsonl \
    --model_path /root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B \
    --num_gpus 1 \
    --num_gpus_per_process 1 \
    --max_tokens 6144 \
    --temperature 0.2 \
    --exp sttest_full_forecasting_6144_run3_official \
    --output_name generated_answer.json
```

评估：

```bash
env PYTHONPATH=. \
    /root/autodl-tmp/conda/envs/str-py310/bin/python \
    evaluation/evaluate.py \
    --task reasoning_forecasting \
    --dataset data/ST-Bench/ST-Test/forecasting_test.jsonl \
    --exp_path exp/sttest_full_forecasting_6144_run3_official \
    --pred_pattern generated_answer
```

## 当前断点

- 旧的 run1 已存在：`exp/sttest_full_forecasting_6144`
- run2 已完成：`exp/sttest_full_forecasting_6144_run2_official`
- run3 已完成：`exp/sttest_full_forecasting_6144_run3_official`
- 下一步：基于这三组已有 full forecasting 输出，沿用报告 39 / 既有脚本口径分析三次结果稳定性；不要另起一套新脚本框架。

## 2026-06-08 追加执行记录

### 14:20-14:31 run2

使用仓库原始入口 `inference/inference_tsmllm_vllm.py`，模型路径为本地官方 8B：

```text
/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B
```

关键配置：

- task：`reasoning_forecasting`
- dataset：`data/ST-Bench/ST-Test/forecasting_test.jsonl`
- output：`exp/sttest_full_forecasting_6144_run2_official/generated_answer.json`
- max_tokens：`6144`
- temperature：`0.2`
- num_gpus：`1`
- num_gpus_per_process：`1`

推理日志：

- `run.log`：`exp/sttest_full_forecasting_6144_run2_official/run.log`
- 加载权重耗时：`10.68s`
- 模型加载显存：`15.3953 GiB`
- vLLM engine 初始化：`61.66s`
- 生成进度耗时：`8:33`
- 输入 token：`77434`，平均 `276.6`

守门检查：

- `generated_answer.json` 存在。
- prediction 行数 `280`。
- idx 覆盖 `0..279`，无重复。
- 空 response：`0`。
- raw response 缺少完整 `<answer>` 的 idx：`[4, 84, 262]`。

官方 evaluate：

```text
exp/sttest_full_forecasting_6144_run2_official/evaluation_metrics.json
```

结果：

| run | total | evaluated | missing | coverage | MAE | MAPE | missing idx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| run2 | 280 | 278 | 2 | 0.992857 | 64.297454 | 135.203768 | [84, 262] |

说明：run2 的 missing 是 evaluate parser 没抽到有效预测，不是没有 response。idx 84 和 262 输出很长且没有完整 `<answer>`。

### 14:32-14:41 run3

同一模型、同一官方入口、同一 dataset、同一 `max_tokens=6144`，输出目录：

```text
exp/sttest_full_forecasting_6144_run3_official
```

推理日志：

- `run.log`：`exp/sttest_full_forecasting_6144_run3_official/run.log`
- 加载权重耗时：`9.61s`
- 模型加载显存：`15.3953 GiB`
- vLLM engine 初始化：`59.95s`
- 生成进度耗时：`6:49`
- 输入 token：`77434`，平均 `276.6`

守门检查：

- `generated_answer.json` 存在。
- prediction 行数 `280`。
- idx 覆盖 `0..279`，无重复。
- 空 response：`0`。
- raw response 缺少完整 `<answer>` 的 idx：`[82]`。

官方 evaluate：

```text
exp/sttest_full_forecasting_6144_run3_official/evaluation_metrics.json
```

结果：

| run | total | evaluated | missing | coverage | MAE | MAPE | missing idx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| run3 | 280 | 279 | 1 | 0.996429 | 64.754811 | 133.539721 | [82] |

### 三次 full forecasting 当前结果

| run | 输出目录 | evaluated/total | missing | MAE | MAPE | raw response 缺完整 `<answer>` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| run1 | `exp/sttest_full_forecasting_6144` | 280/280 | 0 | 68.317056 | 123.289149 | [23, 206] |
| run2 | `exp/sttest_full_forecasting_6144_run2_official` | 278/280 | 2 | 64.297454 | 135.203768 | [4, 84, 262] |
| run3 | `exp/sttest_full_forecasting_6144_run3_official` | 279/280 | 1 | 64.754811 | 133.539721 | [82] |

### 关于多余脚本

中途曾短暂新建 `00_new_codes/tools/official_forecasting_runs_analysis/` 和 `00_new_codes/reports/artifacts/official_forecasting_runs_analysis/` 做额外汇总；用户提醒后确认没有必要，已经删除。后续不要再为这个目标另写新脚本，除非确实需要复用报告 39 的既有分析脚本做最小改动。

## 2026-06-08 最终完成状态

已按报告 10 的 ST-Test 四任务口径补齐 run2 / run3：

| run | entity | etiological | correlation | forecasting |
| --- | --- | --- | --- | --- |
| run1 | `exp/sttest_full_entity_6144/` | `exp/sttest_full_etiological_6144/` | `exp/sttest_full_correlation_6144/` | `exp/sttest_full_forecasting_6144/` |
| run2 | `exp/sttest_full_entity_6144_run2_official/` | `exp/sttest_full_etiological_6144_run2_official/` | `exp/sttest_full_correlation_6144_run2_official/` | `exp/sttest_full_forecasting_6144_run2_official/` |
| run3 | `exp/sttest_full_entity_6144_run3_official/` | `exp/sttest_full_etiological_6144_run3_official/` | `exp/sttest_full_correlation_6144_run3_official/` | `exp/sttest_full_forecasting_6144_run3_official/` |

核验结果：

- 每次 ST-Test 都包含四任务，共 `3273` 条 prediction。
- 所有 prediction 文件 idx 均完整、无空 response。
- 选择类任务 evaluate coverage 都是 `1.0`。
- forecasting run2 / run3 有少量 evaluate missing：run2 `[84, 262]`，run3 `[82]`。
- 官方推理/评估源码未修改。

run2 / run3 主要指标：

| task | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| entity accuracy | 0.747906 | 0.733668 | 0.731156 |
| etiological accuracy | 0.956522 | 0.927536 | 0.927536 |
| correlation accuracy | 0.831658 | 0.831030 | 0.822236 |
| forecasting MAE | 68.317056 | 64.297454 | 64.754811 |
| forecasting MAPE | 123.289149 | 135.203768 | 133.539721 |

运行时间：

- run2 四任务纯生成合计约 `1:43:13`。
- run3 四任务纯生成合计约 `1:37:47`。
- 按当前四任务分别启动 vLLM 的方式，一次完整 ST-Test 端到端约 `1小时45分-1小时50分`。

正式结果报告：

```text
00_new_codes/reports/42-官方STReasoner8B三次ST-Test全量实验结果.md
```
