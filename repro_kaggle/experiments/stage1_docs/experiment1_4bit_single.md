# 实验一：4bit 单卡主测试部分结果

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：512
- precision：`4bit`
- CUDA_VISIBLE_DEVICES：`0`
- requested device_map：`{"": 0}`
- time-series merge patch：已应用，日志中为 `MERGE_PATCH_APPLIED=True`

## 执行范围

本轮只完成主测试 `tiny20` 的 20 条样本。脚本进入 `paper 1/4` 后按人工要求中断，因此论文样例和压力样例不纳入本轮统计。

机器可读结果：

- `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/main_predictions.jsonl`
- `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/summary.json`
- `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/run.log`

## 运行摘要

- load 成功：True
- load 错误：None
- main 样本数：20
- paper 样本数：0
- stress 样本数：0
- main generate 成功率：1.0
- main decode 成功率：1.0
- main parse 成功率：0.45
- main 平均正确率：0.15
- 平均 input tokens：696.85
- 平均 actual new tokens：512.0
- 平均延迟：552.565 秒
- 最高延迟：773.262 秒
- 平均 tokens/s：0.979
- 峰值显存：`{"gpu0": {"max_allocated_gib": 6.631, "max_reserved_gib": 6.744}}`
- 失败阶段统计：`{"parse": 11}`

## 说明

本轮 20 条主测试样例均能完成 generate 和 decode；失败集中在 parse 阶段。所有样本的 `actual_new_tokens` 都达到 512，说明当前 Hugging Face generate 路径下输出没有提前停止，速度和解析质量都受到明显影响。
