# 实验一：4bit单卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：512
- precision：`4bit`
- CUDA_VISIBLE_DEVICES：`0`
- requested device_map：`{'': 0}`

## 运行摘要

- load 成功：True
- load 错误：None
- 正式 generate 成功率（三组合计）：0.0
- 正式 decode 成功率（三组合计）：0.0
- 正式 parse 成功率（三组合计）：0.0
- 正式平均正确率（三组合计）：0.0
- 主测试平均正确率：0.0
- 论文样例平均正确率：0.0
- 压力测试平均正确率：0.0
- 平均 input tokens：745.16
- 平均 actual new tokens：None
- 平均延迟：None 秒
- 最高延迟：None 秒
- 平均 tokens/s：None
- 峰值显存：`{"gpu0": {"max_allocated_gib": 6.631, "max_reserved_gib": 6.744}}`
- 瓶颈统计：`{"输入/生成瓶颈": 25}`

## 三组样例说明

- 主测试样例：计入正式成功率，并单独保留分组指标。
- 论文样例：计入正式成功率，并单独保留分组指标。
- 压力测试样例：计入正式成功率，并单独保留分组指标。

机器可读结果见：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single`
