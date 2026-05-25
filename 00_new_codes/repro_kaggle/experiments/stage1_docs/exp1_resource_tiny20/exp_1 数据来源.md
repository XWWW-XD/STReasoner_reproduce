# 三个来源

- 论文中提到的4个样例
- 4类推理任务 x 5条样例
- 压力测试：input_tokens最长的样例

## 目录对应关系

未填写

## 使用此subsets实验中应读取的文件

- 主测试成功率分母：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl`
- 分任务主测试文件：`forecasting_5.jsonl`、`entity_5.jsonl`、`etiological_5.jsonl`、`correlation_5.jsonl`
- 论文样例额外复跑：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl`
- 资源压力测试：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl`

## 已知限制和注意事项

- 长度为近似字符数，不是正式 tokenizer token 数，因为本地没有完整 Qwen/STReasoner tokenizer 词表。
- ST-Test 将时间序列保存在单独的 `timeseries` 字段，而 `input` 里使用 `<ts><ts/>` 占位符，因此 stress case 使用 prompt + 序列化 time-series 的合并长度排序。

## 不匹配的论文中示例

- Figure 1 是说明性示例，无法从公开 ST-Bench 文件中严格还原完整输入。
- 提供的 PDF 抽取文本显示 Appendix H 的 case-study 表为 Table 6-9；未发现 Table 10-12 的 case-study 条目。
- Appendix H 的 forecasting 样例 input 能匹配到一条 ST-Test 样本，但论文展示的 STReasoner prediction 与数据集 gold output 不一致；细节记录在 `paper_cases/matching_notes.md`。
