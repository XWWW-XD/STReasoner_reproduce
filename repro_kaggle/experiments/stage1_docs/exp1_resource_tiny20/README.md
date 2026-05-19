# exp1_resource_tiny20

## 实验目标

准备固定、可复用的样本集，用于后续比较 STReasoner/Qwen-8B 在 fp16、8bit、4bit 三种推理配置下的资源占用、generate 成功率、decode 成功率、parse 成功率和速度。本阶段只准备数据和文档，不运行模型。

## 目录对应关系

- tiny20 主测试数据：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/`
- tiny20 主测试文档：`repro_kaggle/experiments/stage1_docs/exp1_resource_tiny20/st_test_tiny20_seed20260519/`
- 论文样例数据：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/`
- 论文样例文档：`repro_kaggle/experiments/stage1_docs/exp1_resource_tiny20/paper_cases/`
- 压力测试数据：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/`
- 压力测试文档：`repro_kaggle/experiments/stage1_docs/exp1_resource_tiny20/stress_case/`

## 验收摘要

- tiny20 总数：20 条
- forecasting：5 条
- entity：5 条
- etiological：5 条
- correlation：5 条
- 论文样例 matched：4 条
- 论文样例 unmatched / 无法严格还原：4 条
- stress case：已选出 1 条

## 后续 fp16 / 8bit / 4bit 应读取的文件

- 主测试成功率分母：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl`
- 分任务主测试文件：`forecasting_5.jsonl`、`entity_5.jsonl`、`etiological_5.jsonl`、`correlation_5.jsonl`
- 论文样例额外复跑：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl`
- 资源压力测试：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl`

只有 `tiny20_all.jsonl` 中的 20 条样本计入主测试的 generate/decode/parse/速度成功率比较。matched 论文样例只作为额外定性检查或回归检查。unmatched 论文样例仅作记录。stress case 只用于资源压力测试，不计入 tiny20 成功率。

## 已知限制和注意事项

- 长度为近似字符数，不是正式 tokenizer token 数，因为本地没有完整 Qwen/STReasoner tokenizer 词表。
- ST-Test 将时间序列保存在单独的 `timeseries` 字段，而 `input` 里使用 `<ts><ts/>` 占位符，因此 stress case 使用 prompt + 序列化 time-series 的合并长度排序。
- Figure 1 是说明性示例，无法从公开 ST-Bench 文件中严格还原完整输入。
- 提供的 PDF 抽取文本显示 Appendix H 的 case-study 表为 Table 6-9；未发现 Table 10-12 的 case-study 条目。
- Appendix H 的 forecasting 样例 input 能匹配到一条 ST-Test 样本，但论文展示的 STReasoner prediction 与数据集 gold output 不一致；细节记录在 `paper_cases/matching_notes.md`。
