请帮我固定 STReasoner 复现阶段一中“实验一：不同精度推理资源测试”的样本集。

当前任务只做样本集准备，不运行模型。

背景：
我后续要比较 STReasoner 开源代码中的 Qwen-8B 模型在 fp16 / 8bit / 4bit 三种推理配置下的资源占用、generate 成功率、decode 成功率、parse 成功率和速度。为了可比性，三种精度必须使用同一批固定样本。

论文和抽取文本已经放在主文件夹的 paper/ 目录下，请优先使用 paper/ 中已有的 PDF 和 extracted markdown/text 文件。如果 extracted 文件中无法检索 Figure 1、Appendix H、Table 9-12 等关键信息，再自行从 PDF 中重新抽取文本。论文文本抽取只用于匹配论文样例，不要求排版完美。

输出目录：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/

请完成三组样本集：

一、主测试样本 tiny20

从 Hugging Face 数据集 Time-HD-Anonymous/ST-Bench 的 ST-Test 中抽取样本。

覆盖四类任务：
- forecasting
- entity
- etiological
- correlation

每类抽 5 条，共 20 条。
固定随机种子：20260519。

请先检查实际数据字段，不要假设字段名。抽样时尽量保留原始样本完整内容，同时额外记录 task、source_file、original_line_index 或可追溯 id。

保存到：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/

需要产出：
- tiny20_all.jsonl
- forecasting_5.jsonl
- entity_5.jsonl
- etiological_5.jsonl
- correlation_5.jsonl
- manifest.json
- README.md

manifest.json 中请记录：
- dataset repo
- dataset revision / commit，如果能获取就记录
- source file
- task
- original line index 或原始样本 id
- sample_id
- seed
- selection rule
- 字段摘要
- 如果能计算，记录 input text / prompt 的 token 或字符长度；如果无法确定正式 tokenizer，就先记录可用的近似长度，并在 README 中说明。

二、论文样例额外测试 paper_cases

请从论文正文和附录中找出所有展示过的 case study / 示例样例，尤其关注：
- Figure 1
- Appendix H
- Table 9
- Table 10
- Table 11
- Table 12

目标：
论文正文和附录中出现过、并且能还原完整输入的样例，全部纳入额外测试。

请在 ST-Test、ST-CoT-Text 或其他作者公开数据文件中尝试匹配这些论文样例的完整原始输入。

匹配优先级：
1. question 文本完全或近似匹配
2. options 匹配
3. graph structure 匹配
4. paper answer 匹配
5. 节点数、time series length、关键词等辅助匹配

如果能匹配到完整原始输入，保存为可复跑样例。
如果论文只展示了简化图、截断时间序列、占位符或不完整输入，无法严格还原，请记录为 unmatched，不要强行放入可复跑样例，也不要纳入成功率统计。

保存到：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/

需要产出：
- paper_cases_matched.jsonl
- paper_cases_unmatched.md
- paper_cases_manifest.json
- README.md

README 中请说明：
- 找到了哪些论文样例
- 哪些可以严格复跑
- 哪些无法还原完整输入
- 每个 matched 样例的匹配依据是什么

三、压力测试 stress case

从 ST-Test 全部可用测试样本中，选择 input tokens 或近似输入长度最长的 1 条样本，作为压力测试样本。

目标：
这个样本用于测试最容易触发显存压力的输入，不计入 tiny20 主测试成功率。

如果正式 tokenizer 尚不可用，可以先用字符长度或字段长度做近似排序；如果 tokenizer 可用，则优先使用 tokenizer 计算 input tokens。

保存到：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/

需要产出：
- stress_longest_input_1.jsonl
- stress_manifest.json
- README.md

README 中请说明：
- 这个样本来自哪个 source file
- 属于哪类任务
- 为什么被选为最长输入样本
- 使用的是 token 长度还是近似长度
- 如果它已经出现在 tiny20 中，也请标注；但仍然单独作为 stress case 保存。

总报告

请在：
repro_kaggle/experiments/stage1_docs/exp1_resource_tiny20/README.md

生成总说明，包含：
- tiny20 是否恰好 20 条
- 每类任务是否恰好 5 条
- paper cases 匹配到多少条，未匹配多少条
- stress case 是否成功选出 1 条
- 后续 fp16 / 8bit / 4bit 实验应该读取哪些文件
- 哪些样本计入主测试成功率，哪些只作为额外测试或压力测试

验收标准：
- 不运行模型
- 不修改原始数据
- 抽样可复现
- tiny20_all.jsonl 恰好 20 条
- 四类任务各 5 条
- stress case 恰好 1 条
- 论文样例必须区分 matched / unmatched
- 无法还原完整输入的论文样例不能纳入成功率统计
- 所有产物都保存在 repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/ 下