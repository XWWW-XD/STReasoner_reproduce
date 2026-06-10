## `PaperCases.jsonl` / `cases` 字段说明

- `input`：模型看到的完整文本 prompt。里面包含节点占位符 `<ts><ts/>`、`Graph Structure`、问题和选项。
- `timeseries`：与 `input` 中节点占位符对应的时序数值数组。外层列表按 Node 顺序排列，内层列表是该节点的时间序列。
- `output`：标准答案。选择题通常是 `<answer>A</answer>` 这类格式；forecasting 是数值列表字符串。
- `category`：样本类别，取值包括 `etiological`、`entity`、`correlation`、`forecasting`；混合任务 runner 按此字段分流。
- `sample_id`：本地整理后的唯一样本 ID，包含论文表格和原始行号信息。
- `source_file`：样本来自 ST-Bench 的哪个原始测试文件，例如 `ST-Test/entity_test.jsonl`。
- `original_line_index`：样本在原始 ST-Test 文件中的零基行号，方便回溯官方数据。
- `paper_case_id`：论文 case 的稳定 ID，例如 `appendix_h_table7_entity`。
- `paper_note`：整理备注。前三个 case 是严格匹配；forecasting case 说明了 ST-Test gold output 和论文展示的 STReasoner prediction 不同，本文件保留 ST-Test gold output 以便可复现实验评估。

## 使用 `PaperCases.jsonl` 的注意点

- `PaperCases.jsonl` 是混合任务文件，包含 etiological、entity、correlation、forecasting 各 1 条。它适合给现有 paper-case/SmartTest 风格 runner 按行读取、按 `category` 分流。
- 如果直接使用官方 `inference/inference_tsmllm_vllm.py` 和 `evaluation/evaluate.py`，它们一次只接收一个 `--task`，更适合按任务拆开或单条运行。不要把混合任务文件当成某一个单独 `reasoning_entity` 或 `reasoning_forecasting` 测试集整体评估。

