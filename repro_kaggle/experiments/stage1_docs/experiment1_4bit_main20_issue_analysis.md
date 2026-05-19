# 实验一 4bit main20 问题分析

## 结论摘要

本轮只完成 `4bit_single` 的主测试 `tiny20` 20 条。结果显示，time-series merge runtime patch 生效后，之前 4bit 单卡在 generate 阶段的张量长度不匹配问题已经消失：20 条主测试样例全部 generate 成功、decode 成功。

但是本轮仍然明显有问题：20 条样例全部生成到 `max_new_tokens=512` 上限，平均单条延迟 552.565 秒，平均速度 0.979 tokens/s；parse 只成功 9/20，最终正确 3/20。也就是说，当前瓶颈已经从“模型无法生成”转为“输出过长、格式不稳定、解析成功率低”。

## 执行范围

- 已执行：`4bit_single` + main `tiny20` 20 条。
- 未执行：paper cases。
- 未执行：stress case。
- 未执行：`8bit_single`、`fp16_single`、`fp16_dual`。
- 中断位置：主测试 20 条全部完成后，脚本刚进入 `paper 1/4`，随后按人工要求中断。

文件状态确认：

- `main_predictions.jsonl`：20 行。
- `paper_predictions.jsonl`：0 行。
- `stress_predictions.jsonl`：0 行。
- `summary.json`：已写出，`counts = {"main": 20, "paper": 0, "stress": 0}`。

## 实现细节检查

### 模型加载

符合预期。

- precision：`4bit`
- quantization：BitsAndBytes `load_in_4bit=True`
- device_map：`{"": 0}`
- 实际模型分布：`{"": 0}`
- 首个参数 dtype：`torch.float16`
- use_cache：`False`
- 可见 GPU：1 张 Tesla T4，约 14.563 GiB

加载后显存：

- allocated：5.703 GiB
- reserved：6.740 GiB

### time-series merge patch

符合预期。

当前实验脚本没有重写作者源码，也没有改 Hugging Face 远程模型代码，而是在模型加载后复用旧成功脚本 `repro_kaggle/scripts/05_eval_sttest_tiny.py` 中已有的 `patch_timeseries_merge_device`。

当前 runner 中新增的调用位置在模型加载完成之后、样本循环之前：

- `load_model(...)`
- `apply_timeseries_merge_patch(model, logger)`
- 进入 main/paper/stress 样本循环

日志中已打印：

```text
Applied runtime patch for multi-GPU time-series merge device alignment.
MERGE_PATCH_APPLIED=True
```

这符合预期。更重要的是，20 条 main 样例均未再出现 `_merge_input_ids_with_time_series_features` 中的张量长度不匹配错误，说明 patch 对当前 4bit 单卡路径有效。

### 输入构造

符合预期。

当前 runner 与旧成功脚本一致，核心调用仍是：

```python
processor(text=prompt, timeseries=timeseries, return_tensors="pt")
```

没有使用 `chat_template`。输入会被手动 move 到 `first_model_device(model)`，本轮为 `cuda:0`。此前对 ST-Test `index=0` 的最小验证中确认：

- processor keys：`input_ids`, `attention_mask`, `timeseries`
- input_ids shape：`(1, 504)`
- attention_mask shape：`(1, 504)`
- timeseries shape：`(5, 96, 1)`
- timeseries dtype：`torch.float16`
- device：`cuda:0`

本轮 main20 没有出现 placeholder 数量、timeseries 节点数或 device move 相关错误。

### generate 与 decode

部分符合预期，但存在严重速度问题。

符合预期的部分：

- generate 成功：20/20
- decode 成功：20/20
- generate 阶段未出现 CUDA OOM
- generate 阶段未出现 time-series merge 张量长度错误
- 峰值显存约 6.631 GiB allocated / 6.744 GiB reserved，低于 T4 总显存

不符合预期或需要警惕的部分：

- 20/20 样例的 `actual_new_tokens` 都等于 512。
- 这说明生成全部触达 `max_new_tokens` 上限，没有自然 EOS 或提前停止。
- 平均延迟达到 552.565 秒，最高 773.262 秒。
- 平均速度只有 0.979 tokens/s。

这不是资源崩溃，但对后续批量实验非常不友好。如果保持 `max_new_tokens=512`，完整四组配置很难在有限运行窗口内稳定完成。

### parse 与正确率

不符合预期。

总体结果：

- parse 成功：9/20
- parse 失败：11/20
- 正确：3/20
- main 平均正确率：0.15
- parse 成功样本内正确率：3/9 = 0.3333

按任务拆分：

| 任务 | 样本数 | generate 成功 | parse 成功 | 正确 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forecasting | 5 | 5 | 0 | 0 | 518.070 秒 |
| entity | 5 | 5 | 4 | 1 | 648.640 秒 |
| etiological | 5 | 5 | 3 | 1 | 537.435 秒 |
| correlation | 5 | 5 | 2 | 1 | 506.116 秒 |

确定的 parse 失败原因：

- forecasting 5 条全部 parse 失败，失败信息包括 `json_array_parse_error` 和 `no_json_array`。
- entity 有 1 条 parse 失败，原因是 `no_answer_tag_or_choice`。
- etiological 有 2 条 parse 失败，原因是 `no_answer_tag_or_choice`。
- correlation 有 3 条 parse 失败，原因是 `no_answer_tag_or_choice`。

这说明当前输出没有稳定遵守评测脚本期望的格式。尤其 forecasting 需要可解析的数值数组，但 5 条都没有形成可解析结果。

## 样本级结果

| 序号 | sample_id | task | stage | parse | correct | actual_new_tokens | latency_sec | 确定失败原因 |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | tiny20_forecasting_01_line87 | forecasting | parse | False | False | 512 | 665.719 | json_array_parse_error |
| 2 | tiny20_forecasting_02_line116 | forecasting | parse | False | False | 512 | 488.735 | no_json_array |
| 3 | tiny20_forecasting_03_line174 | forecasting | parse | False | False | 512 | 401.960 | no_json_array |
| 4 | tiny20_forecasting_04_line227 | forecasting | parse | False | False | 512 | 395.946 | no_json_array |
| 5 | tiny20_forecasting_05_line239 | forecasting | parse | False | False | 512 | 637.990 | no_json_array |
| 6 | tiny20_entity_01_line257 | entity | parse | False | False | 512 | 772.784 | no_answer_tag_or_choice |
| 7 | tiny20_entity_02_line529 | entity | none | True | False | 512 | 480.565 | 无运行失败 |
| 8 | tiny20_entity_03_line670 | entity | none | True | False | 512 | 773.262 | 无运行失败 |
| 9 | tiny20_entity_04_line785 | entity | none | True | True | 512 | 535.116 | 无运行失败 |
| 10 | tiny20_entity_05_line1133 | entity | none | True | False | 512 | 681.474 | 无运行失败 |
| 11 | tiny20_etiological_01_line13 | etiological | none | True | False | 512 | 703.940 | 无运行失败 |
| 12 | tiny20_etiological_02_line137 | etiological | parse | False | False | 512 | 692.927 | no_answer_tag_or_choice |
| 13 | tiny20_etiological_03_line144 | etiological | none | True | True | 512 | 418.031 | 无运行失败 |
| 14 | tiny20_etiological_04_line154 | etiological | none | True | False | 512 | 444.622 | 无运行失败 |
| 15 | tiny20_etiological_05_line197 | etiological | parse | False | False | 512 | 427.657 | no_answer_tag_or_choice |
| 16 | tiny20_correlation_01_line91 | correlation | parse | False | False | 512 | 430.283 | no_answer_tag_or_choice |
| 17 | tiny20_correlation_02_line336 | correlation | parse | False | False | 512 | 431.204 | no_answer_tag_or_choice |
| 18 | tiny20_correlation_03_line528 | correlation | none | True | True | 512 | 427.664 | 无运行失败 |
| 19 | tiny20_correlation_04_line896 | correlation | none | True | False | 512 | 694.753 | 无运行失败 |
| 20 | tiny20_correlation_05_line1059 | correlation | parse | False | False | 512 | 546.674 | no_answer_tag_or_choice |

## 已确定的问题

### 问题一：原始 merge 错误已修复

确定。

修复前，4bit/8bit 单卡在 generate 的 time-series token merge 阶段报错，例如张量长度不匹配。修复后，4bit main20 全部 generate 成功，没有再落入 `_merge_input_ids_with_time_series_features` 报错。

### 问题二：输出全部触达 512 token 上限

确定。

20 条样本的 `actual_new_tokens` 全部为 512。这说明当前 generate 设置下模型没有提前停止，导致每条样本都跑满上限。这个行为直接解释了本轮运行极慢。

### 问题三：parse 成功率低

确定。

parse 只成功 9/20。失败集中在两类：

- forecasting 没有输出可解析 JSON 数组。
- classification 类任务没有稳定输出 `<answer>` 或可解析的 A/B/C/D 选项。

### 问题四：4bit 单卡显存不是主要瓶颈

确定。

本轮峰值显存约 6.631 GiB allocated / 6.744 GiB reserved，远低于 T4 的 14.563 GiB。当前 4bit 单卡的主要问题不是 OOM，而是生成耗时和输出格式。

### 问题五：本轮 summary 的字段名仍带有三组口径历史痕迹

确定。

`summary.json` 中仍存在 `official_denominator_note`，文字写的是“主测试 20 + 论文样例 4 + 压力测试 1”。但本轮实际 `counts` 是 `main=20, paper=0, stress=0`。因此本轮报告只按 main20 解读，不按三组合计解读。

## 后续建议

如果继续这个方向，建议先不扩展到 paper/stress 或其他精度，而是先解决两个更基本的问题：

- 控制生成长度或停止条件，避免每条样本都跑满 512 token。
- 明确输出格式约束，尤其 forecasting 的 JSON 数组格式，以及分类任务的 `<answer>A</answer>` 格式。

在这两个问题解决前，继续跑 8bit/fp16 的收益有限，因为会花大量时间重复得到“能生成但慢、解析不稳”的结果。
