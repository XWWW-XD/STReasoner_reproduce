# 

## 1. 原始提示词

```text
我现在有一个 stage2 的实验脚本，需要完成：
- 修改后的新实验名称叫stage 2.2。
- 先从stage2的脚本复制一份代码。
- 把它当前使用的数据源切换成Stage 1 使用的数据集（论文中涉及到的样例）STReasoner_reproduce\00_new_codes\repro_kaggle\experiments\stage1_subsets\exp1_resource_tiny20\paper_cases。
- max_new_tokens设置为6144。
- parser逻辑尽量与evaluate文件夹相同，最好直接复用，当前parser过于严格。
- 脚本中的变量名尽量与源代码对齐，当前有些差别，而且raw_response和decode_text在当前代码中同时存在无意义。

完成后请你认为我该知道的全部，和提示词一起放在reports里。
```

## 2. 新增文件

新增 stage 2.2 推理脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py
```

该脚本由原 stage2 脚本复制而来：

```text
00_new_codes/repro_autodl/experiments/scripts/stage2_script/run_smarttest.py
```

新增 stage 2.2 数据快照目录：

```text
00_new_codes/repro_autodl/experiments/stage2_2_subsets/experiment1_paper_cases/
```

其中已通过 `prepare` 生成：

```text
paper_cases_matched.jsonl
prepare.log
prepare_summary.json
```

## 3. 实验名称

新脚本中实验名称统一为：

```text
stage 2.2
```

配置名为：

```python
CONFIG_NAME = "stage2.2_fp16_a100_single"
```

## 4. 数据源切换

原 stage2 脚本使用 SmartTest 两条样例：

```text
00_new_codes/repro_autodl/experiments/stage2_subsets/experiment1_smart_test/SmartTest.jsonl
```

stage 2.2 改为使用 Stage 1 的论文样例数据源：

```text
00_new_codes/repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl
```

脚本中的常量为：

```python
PAPER_CASE_SOURCE_DIR = REPRO_KAGGLE_ROOT / "experiments/stage1_subsets/exp1_resource_tiny20/paper_cases"
PAPER_CASE_SOURCE_PATH = PAPER_CASE_SOURCE_DIR / "paper_cases_matched.jsonl"
STAGE22_DATA_DIR = REPRO_AUTODL_ROOT / "experiments/stage2_2_subsets/experiment1_paper_cases"
STAGE22_DATA_PATH = STAGE22_DATA_DIR / "paper_cases_matched.jsonl"
```

`prepare` 命令会把 Stage 1 的 `paper_cases_matched.jsonl` 复制到 stage 2.2 的 AutoDL 实验目录，作为本次实验快照。

## 5. 样例数量和任务分布

已验证 stage 2.2 数据共有 4 条：

```text
0: paper_appendix_h_table6_etiological_line118, task=etiological
1: paper_appendix_h_table7_entity_line982, task=entity
2: paper_appendix_h_table8_correlation_line547, task=correlation
3: paper_appendix_h_table9_forecasting_line9, task=forecasting
```

任务分布：

```json
{
  "etiological": 1,
  "entity": 1,
  "correlation": 1,
  "forecasting": 1
}
```

因此不能继续沿用原 stage2 的 `--case forecasting/non_forecasting` 逻辑。原逻辑只适合 2 条 SmartTest；paper_cases 有 3 条非 forecasting 样例，必须按 `sample_index` 或 `sample_id` 选择。

## 6. max_new_tokens 修改

默认值已从 2048 改为：

```python
DEFAULT_MAX_NEW_TOKENS = 6144
```

运行时仍可通过命令行覆盖：

```bash
--max-new-tokens 6144
```

## 7. Parser 修改

原 stage2 脚本的 parser 过严，主要问题是：

- 要求 `<answer>...</answer>` 必须正好出现一次。
- forecasting 要求 answer 内容必须是严格 JSON list。
- 多选题要求 answer 内容必须是单个 A/B/C/D。

stage 2.2 中改为复用 `evaluation/evaluate_qa.py` 里的解析辅助函数：

```python
evaluate_qa._extract_tag_content(...)
evaluate_qa._parse_series(...)
evaluate_qa._normalize_choice(...)
```

新逻辑：

- forecasting：先按 `evaluate_qa._extract_tag_content` 抽取答案，再用 `evaluate_qa._parse_series` 解析数字序列。
- entity / etiological / correlation / causal：先抽取答案，再用 `evaluate_qa._normalize_choice` 标准化选项。
- 不再要求 answer tag 恰好出现一次。
- 不再要求 forecasting 输出必须是严格 JSON，只要 `evaluate_qa._parse_series` 能抽出数字序列即可。

注意：这里复用了 `evaluate_qa.py` 的私有辅助函数，目的是最大限度贴近当前 evaluation 文件夹逻辑。如果以后 `evaluation/evaluate_qa.py` 改名或删除这些函数，stage 2.2 脚本也需要同步调整。

## 8. response 字段对齐

原 stage2 脚本同时写：

```python
raw_response
decoded_text
```

这两个字段内容相同，语义重复。

stage 2.2 中改为只写一个字段：

```python
response
```

原因：

- `evaluation/evaluate_qa.py` 的 `load_prediction_files()` 本来就优先读取顶层 `response` 字段。
- `response` 比 `raw_response` / `decoded_text` 更贴近 evaluation 代码的数据结构。
- 避免同一段模型输出在一个 record 里重复保存。

对应记录字段包括：

```python
"response": None
"parsed_answer": None
"gold_answer": sample.get("output")
"parse_success": False
"parse_error": None
```

## 9. 新脚本运行方式

先准备数据：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py prepare --overwrite true
```

查看可运行样例：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py list
```

按样例序号运行单条：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run --sample-index 0 --overwrite true
```

按 sample_id 运行单条：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run --sample-id paper_appendix_h_table9_forecasting_line9 --overwrite true
```

运行全部 4 条：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run-all --overwrite true
```

如果中断后继续：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run-all --resume true
```

默认输出目录：

```text
00_new_codes/repro_autodl/experiments/stage2_2_results/experiment1_paper_cases/
```

主要输出：

```text
paper_cases_prediction.jsonl
paper_cases_summary.json
paper_cases_run.log
```

## 10. 已做验证

已通过 Python 语法编译：

```bash
python -m py_compile 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py
```

已通过 `prepare`：

```text
rows: 4
task_counts: {"etiological": 1, "entity": 1, "correlation": 1, "forecasting": 1}
```

已通过 `list`：

```text
0: sample_id=paper_appendix_h_table6_etiological_line118 task=etiological paper_case_id=appendix_h_table6_etiological
1: sample_id=paper_appendix_h_table7_entity_line982 task=entity paper_case_id=appendix_h_table7_entity
2: sample_id=paper_appendix_h_table8_correlation_line547 task=correlation paper_case_id=appendix_h_table8_correlation
3: sample_id=paper_appendix_h_table9_forecasting_line9 task=forecasting paper_case_id=appendix_h_table9_forecasting
```

已验证默认 `max_new_tokens`：

```text
6144
```

已验证 parser 示例：

```text
parse_model_answer("entity", "D. final") -> ("D", True, None)
parse_model_answer("forecasting", "The forecast is [19.86, 19.97, 20.05].") -> ([19.86, 19.97, 20.05], True, None)
```

本地没有 GPU / torch 环境，因此没有在本机启动模型推理。真正模型运行需要在 AutoDL 服务器上执行。

## 11. 需要知道的注意事项

1. `run-all` 是为了方便加的入口，但当前仍保持原 stage2 单样本运行结构，内部会按样例顺序逐条执行。

2. 当前脚本每条样例都会进入 `run_one_sample()` 的完整流程，包括模型加载。这样最贴近原 stage2 脚本，风险较低，但 4 条样例会重复加载模型，耗时会更长。后续如果需要提速，可以再改成“一次加载模型，循环跑 4 条样例”的结构。

3. `response` 字段是新的唯一模型输出文本字段。后续看结果时优先看：

```text
paper_cases_prediction.jsonl -> response
```

4. `parse_success` 现在是“按 evaluation parser 能否解析出候选答案”的诊断字段，不等于最终正确率。最终指标仍看 `official_metrics`。

5. stage 2.2 的 evaluation 路径已修正到仓库根目录：

```text
evaluation/evaluate_qa.py
```

原 stage2 脚本里 `AUTHOR_EVALUATE_QA = PROJECT_ROOT / "evaluation/evaluate_qa.py"` 在当前目录结构下会指向 `00_new_codes/evaluation/evaluate_qa.py`，该路径不存在。stage 2.2 已改为：

```python
AUTHOR_EVALUATE_QA = REPO_ROOT / "evaluation/evaluate_qa.py"
```

6. 如果 AutoDL 上路径结构和本机一致，直接运行即可。如果你只上传 `00_new_codes` 而没有仓库根目录的 `evaluation` 文件夹，脚本会找不到 `evaluate_qa.py`。

## 12. 建议执行顺序

在 AutoDL 服务器上进入仓库根目录后：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py prepare --overwrite true
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py list
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run-all --overwrite true
```

如果只想先做 forecasting 样例：

```bash
python 00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py run --sample-index 3 --overwrite true
```
