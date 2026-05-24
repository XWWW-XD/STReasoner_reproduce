# 实验一：不同精度推理资源测试

## 当前执行状态

本轮按人工调整后的范围执行：只完成 `4bit_single` 的主测试 `tiny20` 20 条；论文样例、压力样例、`8bit_single`、`fp16_single`、`fp16_dual` 均不再继续执行，报告中保持空白。

本轮在 `4bit_single` 进入 `paper 1/4` 后人工中断。`paper_predictions.jsonl` 和 `stress_predictions.jsonl` 均为 0 行，因此 paper/stress 不纳入本轮统计。

## 样本与目录

- 主测试样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl`，共 20 条，四类任务各 5 条。
- 论文样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl`，本轮未执行。
- 压力测试样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl`，本轮未执行。
- 运行脚本：`repro_kaggle/experiments/scripts/stage1_script/run_experiment1_precision_resource.py`
- 机器可读结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource`
- 额外问题分析：`repro_kaggle/experiments/stage1_docs/experiment1_4bit_main20_issue_analysis.md`

## 证据查找

### 作者代码中的精度设置

- 训练脚本使用 fp16：
  - `scripts/qwen3-8b/train_stage1.sh:24`：`--fp16`
  - `scripts/qwen3-8b/train_stage2.sh:24`：`--fp16`
- 推理脚本中的 dtype：
  - `inference/llm_utils.py:97-98`：普通 vLLM worker 使用 `LLM(..., dtype='half')`，对应 fp16/half。
  - `inference/llm_utils.py:142` 之后的 `worker_vllm_ts` 未显式设置 dtype；实际 dtype 需要通过本实验记录加载后模型参数 dtype、量化配置和日志确认。

### 作者代码中的 max_new_tokens / max_tokens 设置

- 作者 vLLM 推理使用 `max_tokens=512`：
  - `inference/inference_tsmllm_vllm.py:64-68`：`SamplingParams(max_tokens=512, temperature=0.2)`
- 本实验使用 Hugging Face `generate`，对应参数记录为 `max_new_tokens=512`。
- Hugging Face 的 `max_new_tokens` 表示最多生成的新 token 数，不包含 prompt tokens。

## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | 本轮只统计主测试 tiny20 20 条 |
|   batch size   |                1                |
| max_new_tokens |               512               |

|           | dtype                        | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |
| --------- | ---------------------------- | -------: | -------: | -------: | ------- |
| 配置证据 | 加载方式 | 4bit |  |  |  |
|  | device_map | {"": 0} |  |  |  |
|  | 实际模型分布 | {"": 0} |  |  |  |
|  | use_cache | False |  |  |  |
|  | merge patch | True |  |  |  |
| 可运行证据 | 已完成样本 | main=20, paper=0, stress=0 |  |  |  |
|  | input tokens（平均值） | 696.850 |  |  |  |
|  | actual new tokens（平均值） | 512.000 |  |  |  |
|  | load 成功 | True |  |  |  |
|  | generate 成功率（main20） | 1.000 |  |  |  |
|  | decode 成功率（main20） | 1.000 |  |  |  |
|  | parse 成功率（main20） | 0.450 |  |  |  |
|  | 平均正确率（main20） | 0.150 |  |  |  |
| 资源与速度 | GPU 总显存 | {"gpu0": 14.563} |  |  |  |
|  | load 后显存 | {"gpu0": {"allocated_gib": 5.703, "reserved_gib": 6.740}} |  |  |  |
|  | generate 峰值显存 | {"gpu0": {"max_allocated_gib": 6.631, "max_reserved_gib": 6.744}} |  |  |  |
|  | 平均延迟与最高延迟 | 552.565 / 773.262 秒 |  |  |  |
|  | tokens/s | 0.979 |  |  |  |
| 失败阶段、失败原因 | 失败阶段、详细失败原因 | parse 失败 11 条；generate/decode 无失败 |  |  |  |

## 本轮统计口径

本轮只统计 `4bit_single` 的主测试 20 条。此前 prompt2 中规划的论文样例、压力样例和其他三组精度配置没有继续执行，本报告不对它们给出成功率。

## 主要结论

- `4bit_single` 加载成功，模型实际位于单卡 `cuda:0`，加载后显存约 5.703 GiB allocated / 6.740 GiB reserved。
- 已复用旧成功脚本中的 time-series merge runtime patch，日志记录 `MERGE_PATCH_APPLIED=True`。
- `main20` 的 generate 成功率为 20/20，说明之前的 time-series token merge 失败已被修复。
- `main20` 的 decode 成功率为 20/20，但 parse 成功率只有 9/20，最终正确 3/20。
- 20 条样本全部生成到 `actual_new_tokens=512`，说明输出普遍触顶，导致平均单条延迟约 552.565 秒，速度约 0.979 tokens/s。
- 主要问题已经从“输入/生成阶段崩溃”转为“输出格式与评测解析问题”和“生成过长导致速度不可接受”。
