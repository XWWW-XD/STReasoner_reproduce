# Stage 2.2 paper_cases 重试与修复报告

日期：2026-05-29

## 结论

结果：
- 4条样例均跑通
- 原始效果不好，主要有以下问题：
- 脚本写的 `run-all` 每条样例都会重新加载一次 STReasoner-8B
- 
2. 模型加载失败分支里 `failure_type_from()` 参数数量写错，导致本应记录失败时二次崩溃。
3. parser 太朴素：选择题只看回答开头，forecasting 把全文所有数字都当预测值，导致模型已经写出的最终答案被评错。

修复后结果：

- 生成：4/4 成功。
- 解析：4/4 成功。
- 选择题：3/3 正确。
- forecasting：提取预测 `[20.02, 20.13, 20.23]`，gold 为 `[19.86, 19.97, 20.05]`，MAE `0.1667`，MAPE `0.8349%`。

仍然存在的问题：

- 模型没有按 `<answer>...</answer>` 输出，选择题多为自然语言末尾 `Answer: D` / `\boxed{C}`。
- forecasting 输出过长，重复大量 `</final_answer>`，`actual_new_tokens=6031`，接近 `max_new_tokens=6144`。
- 这次没有修改 prompt，也没有硬编码答案；改善来自运行器资源管理和 parser/评测提取逻辑。
- 进一步补充了格式诊断：raw response 的 `<answer>` 格式成功率是 `0/4`，但可从 raw response 通用抽取出规范化 `formatted_answer`。

## 实验与输出路径

数据源：

```text
00_new_codes/repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl
```

Stage 2.2 快照：

```text
00_new_codes/repro_autodl/experiments/stage2_2_subsets/experiment1_paper_cases/paper_cases_matched.jsonl
```

当前 4 条样例：

| index | task | sample_id | gold |
|---:|---|---|---|
| 0 | etiological | `paper_appendix_h_table6_etiological_line118` | `<answer>D</answer>` |
| 1 | entity | `paper_appendix_h_table7_entity_line982` | `<answer>C</answer>` |
| 2 | correlation | `paper_appendix_h_table8_correlation_line547` | `<answer>D</answer>` |
| 3 | forecasting | `paper_appendix_h_table9_forecasting_line9` | `[19.86, 19.97, 20.05]` |

主要输出：

```text
00_new_codes/repro_autodl/experiments/stage2_2_paper_cases/baseline_current_6144/
00_new_codes/repro_autodl/experiments/stage2_2_paper_cases/fixed_reuse_model_6144/
00_new_codes/repro_autodl/experiments/stage2_2_paper_cases/parserfix_reparse_existing_6144/
00_new_codes/repro_autodl/experiments/stage2_2_paper_cases/formatfix_reparse_existing_6144/
```

PDF 解析文本：

```text
paper/STReasoner_ACL_2026.txt
```

## 修改 1：修复 run-all 重复加载模型

修改文件：

```text
00_new_codes/repro_autodl/experiments/scripts/stage2_2_script/run_paper_cases.py
```

修改原因：

- 原 `run-all` 内部循环每条样例都调用 `run_one_sample()`。
- `run_one_sample()` 每次都会重新 `load_model_and_processors()`。
- 第 1 条后显存约 17G，第 2 条后显存约 33G，第 3 条加载模型时 OOM。

修改内容：

- `run-all` 开始时只加载一次 model / processor / tokenizer。
- `run_one_sample()` 增加可选的 `preloaded_model` / `preloaded_processor` / `preloaded_tokenizer` 参数。
- `run-all` 循环内复用已经加载好的模型。
- 保留单条 `run` 的原行为，单条运行仍可独立加载模型。

同时修复：

- 加载失败分支原来写成：

```python
failure_type_from(stage, False, False, False)
```

- 但函数只接收 3 个参数，已改为：

```python
failure_type_from(stage, False, False)
```

修改前结果：

- `baseline_current_6144` 跑到第 3 条时 OOM。
- 失败后又触发 `TypeError: failure_type_from() takes 3 positional arguments but 4 were given`。
- 未得到完整 4 条正式结果。

修改后结果：

- `fixed_reuse_model_6144` 4 条全部完成生成。
- 只加载一次模型，日志中后续样例显示 `Using preloaded model for this sample.`。

## 修改 2：修复 parser / evaluate 提取逻辑

修改文件：

```text
evaluation/evaluate_qa.py
```

修改原因：

模型输出里其实包含正确选择题答案，但旧 parser 没读到：

- etiological 末尾有 `Answer: D`，旧 parser 把整段文本当答案，parse failed。
- entity 末尾有 `\boxed{C}`，旧 parser 因为回答开头是 `Answer:`，误读成 `A`。
- correlation 末尾有 `Answer: D`，旧 parser 同样误读成 `A`。
- forecasting 输出里有最终 JSON `{"predictions": [20.02, 20.13, 20.23]}`，旧 parser 把全文所有数字都抽出来，导致 MAE 被严重放大。

修改内容：

- `_normalize_choice()` 新增：
  - 优先读取 `\boxed{C}` 这类最终答案。
  - 读取 `Answer: D` / `Final Answer: D`。
  - 读取末尾带判断词的 `Option C ... most/correct/best/...`。
  - 最后才回退到旧的开头字母匹配。
- `_parse_series()` 新增：
  - 优先读取 JSON 字段里的 `"predictions": [...]` / `"forecast": [...]` / `"answer": [...]`。
  - 其次读取最后一个短数字数组。
  - 最后才回退到旧的全文数字抽取。

没有做的事：

- 没有改 gold。
- 没有删除失败样例。
- 没有按 sample_id 写特例。
- 没有把正确答案硬编码进 parser。
- 没有改 prompt 诱导模型输出。

## 修改 3：修复 vLLM-TS worker 未使用传入 max_tokens 的问题

修改文件：

```text
inference/llm_utils.py
```

修改原因：

用户明确要求正式实验 `max_new_tokens / max_tokens` 必须是 `6144`。复查代码时发现：

- `inference/inference_tsmllm_vllm.py` 有 `--max_tokens` 参数，并会构造 `SamplingParams(max_tokens=args.max_tokens, ...)`。
- 但 `LLMClient` 把这个 `SamplingParams` 放进队列后，`worker_vllm_ts()` 实际调用 `llm.generate()` 时仍使用 worker 内部默认的 `sampling_params`。
- 因此 CLI 参数没有真正生效。之前 `max_tokens=512` 的 ST-Test 运行只能视为链路预跑，不能作为正式结果。

修改内容：

- 在 `worker_vllm()` 和 `worker_vllm_ts()` 中检查队列参数最后一项是否为 `SamplingParams`。
- 如果存在调用方传入的 `SamplingParams`，则用它调用 `llm.generate()`。
- 如果不存在，仍回退到 worker 内部默认参数，保持原有兼容性。

修改前后差异：

| 项目 | 修改前 | 修改后 |
|---|---|---|
| CLI `--max_tokens` | 被放入队列，但 worker 生成时未使用 | worker 生成时使用传入的 `SamplingParams` |
| ST-Test 正式运行 | 不能证明用了指定 token 上限 | 可按 `--max_tokens 6144` 运行 |
| prompt / gold / 数据 | 未修改 | 未修改 |

6144 smoke 验证：

- 命令：`reasoning_entity --max_samples 1 --max_tokens 6144 --exp sttest_smoke_entity_6144`
- 输出：`exp/sttest_smoke_entity_6144/generated_answer.json`
- gold：`<answer>A</answer>`
- raw response 末尾：`<answer>A</answer>`
- 当前 evaluate：`evaluated_samples=1`，`accuracy=1.0`

## 前后结果对比

### 生成链路

| 阶段 | 结果 |
|---|---|
| 历史 T4 fp16 | 模型加载 OOM，paper 4 条全失败 |
| 历史 8bit paper | 生成阶段 time-series merge 长度不匹配，paper 4 条全失败 |
| A100 smoke | 1 条 forecasting 能生成，但 `max_new_tokens=16` 被截断 |
| `baseline_current_6144` | 第 3 条重复加载模型时 OOM，未完整跑通 |
| `fixed_reuse_model_6144` | 4 条全部生成成功 |
| `parserfix_reparse_existing_6144` | 基于同一批生成文本，4 条全部解析成功 |
| `formatfix_reparse_existing_6144` | 基于同一批生成文本，补充 raw 格式诊断和 `formatted_answer` |

### parser 修复前后

| sample | task | gold | 修复前解析 | 修复后解析 | 修复后是否正确 |
|---|---|---|---|---|---|
| table6 | etiological | D | 整段文本，parse failed | D | 是 |
| table7 | entity | C | A | C | 是 |
| table8 | correlation | D | A | D | 是 |
| table9 | forecasting | `[19.86, 19.97, 20.05]` | 全文数字长列表 | `[20.02, 20.13, 20.23]` | MAE 0.1667 |

修复后汇总：

```text
generate_success_count = 4 / 4
parse_success_count    = 4 / 4
choice_accuracy        = 3 / 3 = 1.0
forecasting_mae        = 0.1667
forecasting_mape       = 0.8349%
```

### raw 格式与 formatted_answer

用户指出“模型不按照格式输出”是准确的。parser 修复只能解决“能否从模型回答中读出答案”，不能证明模型原始输出已经符合 `<answer>...</answer>` 规范。

因此补充了格式层诊断：

- `format_success`：raw response 是否严格包含 1 对 `<answer>...</answer>`。
- `format_error`：缺少 `<answer>`、answer tag 数量不匹配、误用 `<final_answer>` 等。
- `formatted_answer`：不改 raw response，只把 parser 抽到的答案规范化成 `<answer>...</answer>`，供保存和后续评估/报告使用。

基于已有 4 条输出重解析：

| sample | raw format | format_error | formatted_answer |
|---|---:|---|---|
| table6 etiological | 失败 | `missing_answer_tag` | `<answer>D</answer>` |
| table7 entity | 失败 | `missing_answer_tag` | `<answer>C</answer>` |
| table8 correlation | 失败 | `missing_answer_tag` | `<answer>D</answer>` |
| table9 forecasting | 失败 | `uses_final_answer_tag_instead_of_answer_tag`，且重复大量 `</final_answer>` | `<answer>[20.02, 20.13, 20.23]</answer>` |

结论：

- 模型 raw 输出格式仍不好，raw `<answer>` 格式成功率为 `0/4`。
- 这不影响“答案内容是否能被通用 parser 提取”的结论，但需要在报告中分开写：
  - `raw format success = 0/4`
  - `parse success = 4/4`
  - `formatted answer available = 4/4`
- 当前没有改 prompt，所以没有从生成端强迫模型输出 `<answer>`；只是做了通用后处理和诊断。

## 为什么之前看起来不好

之前不能简单说“模型不会”。

更准确地说：

1. 早期结果没有完整跑通正式 Stage 2.2 paper_cases。
2. 已有失败主要是资源和输入合并问题，不是答案质量问题。
3. 真正跑通后，模型的自然语言答案里已经有正确选择题答案。
4. 旧 parser 对非 `<answer>` 格式很脆弱，所以把正确答案评成错误。
5. forecasting 的模型预测不完全等于 gold，但已经比较接近；旧 parser 把历史值、步骤编号和预测值混在一起，导致指标虚高。

## 后续建议

- 如果只看 paper_cases 复现实验，本轮结果已经足够用于汇报：链路跑通，3 个选择题正确，forecasting 接近 gold。
- 如果继续优化，优先解决输出格式问题：模型不稳定输出 `<answer>`，并且 forecasting 会重复 `</final_answer>`。
- 但不要为了这 4 条样例改 prompt 或硬编码答案；可以考虑以后在通用推理脚本里加 stopping criteria 或更严格的格式化后处理。

## 2.3 ST-TEST 数据集验证论文真实效果

paper_cases 只能说明论文附录样例链路能跑通，不能代表论文真实整体效果。要验证论文中的真实效果，需要在完整 ST-Test 上按四类任务运行并评估：

- `reasoning_entity`
- `reasoning_etiological`
- `reasoning_correlation`
- `reasoning_forecasting`

本轮已补充下载并检查 ST-Test 四类数据：

| ST-Test 文件 | 样例数 |
|---|---:|
| `data/ST-Bench/ST-Test/correlation_test.jsonl` | 1592 |
| `data/ST-Bench/ST-Test/entity_test.jsonl` | 1194 |
| `data/ST-Bench/ST-Test/etiological_test.jsonl` | 207 |
| `data/ST-Bench/ST-Test/forecasting_test.jsonl` | 280 |
| 合计 | 3273 |

早期使用仓库原始推理入口做过四类任务各 1 条 smoke。当时命令里是 `--max_tokens 512`，不符合后续明确的 `6144` 要求；这里只作为“链路曾经跑通”的排查记录，不作为正式实验依据：

```bash
/root/autodl-tmp/conda/envs/str-py310/bin/python \
  inference/inference_tsmllm_vllm.py \
  --task <task> \
  --model_path /root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_samples 1 \
  --max_tokens 512 \
  --temperature 0.2 \
  --exp <exp_name> \
  --output_name generated_answer.json
```

评估使用仓库原始 `evaluation/evaluate.py`，但从文件路径直接执行时需要显式加 `PYTHONPATH=.`：

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/str-py310/bin/python \
  evaluation/evaluate.py \
  --task <task> \
  --dataset data/ST-Bench/ST-Test/<task_file>.jsonl \
  --exp_path exp/<exp_name> \
  --pred_pattern generated_answer
```

早期四类 smoke 结果如下：

| task | exp | 数据总数 | 本轮评估数 | 输出 token | 指标 |
|---|---|---:|---:|---:|---|
| `reasoning_entity` | `exp/sttest_smoke_entity_1` | 1194 | 1 | 407 | accuracy `0.0` |
| `reasoning_etiological` | `exp/sttest_smoke_etiological_1` | 207 | 1 | 366 | accuracy `1.0` |
| `reasoning_correlation` | `exp/sttest_smoke_correlation_1` | 1592 | 1 | 257 | accuracy `0.0` |
| `reasoning_forecasting` | `exp/sttest_smoke_forecasting_1` | 280 | 1 | 187 | MAE `29.75`，MAPE `158.8235%` |

这 4 条早期 smoke 的意义仅限于：

- ST-Test 数据已经在本地，四类任务文件齐全。
- 官方 vLLM 推理入口可以加载 `STReasoner-8B` 并生成。
- evaluate 入口可以对四类任务输出做评估。
- 当前只是链路验证，不是完整 ST-Test 论文效果复现，也不满足 `6144` 正式要求。

这 4 条 smoke 也提示两个现实问题：

- 单条样例结果波动很大，不能代表论文指标。
- 如果完整跑 3273 条，按当前每次启动 vLLM 的方式不合适；应使用一次加载后连续生成，或按任务长进程运行，避免每条/每小批反复加载模型。

### ST-Test 参数不合规预跑结果

这一节只保留为排查记录，不能作为正式结论。原因有两个：

1. 用户要求正式实验必须使用 `6144`。
2. 当时还未发现 `worker_vllm_ts()` 没有真正使用 CLI 传入的 `SamplingParams`。

当时预跑配置记录如下：

- 模型：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B`
- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- GPU：1 张 A100
- CLI 传入 `max_tokens=512`，但后续确认 worker 实际未正确使用 CLI 参数
- `temperature=0.2`
- `num_gpus=1`
- `num_gpus_per_process=1`

输出目录：

| task | 输出目录 |
|---|---|
| `reasoning_entity` | `exp/sttest_full_entity_512/` |
| `reasoning_etiological` | `exp/sttest_full_etiological_512/` |
| `reasoning_correlation` | `exp/sttest_full_correlation_512/` |
| `reasoning_forecasting` | `exp/sttest_full_forecasting_512/` |

预跑 evaluate 结果：

| task | 样例数 | evaluated | missing | coverage | 指标 | total input tokens | avg input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1194 | 0 | 1.0 | accuracy `0.742044` | 495068 | 414.63 |
| `reasoning_etiological` | 207 | 207 | 0 | 1.0 | accuracy `0.937198` | 79543 | 384.27 |
| `reasoning_correlation` | 1592 | 1592 | 0 | 1.0 | accuracy `0.862437` | 717765 | 450.86 |
| `reasoning_forecasting` | 280 | 280 | 0 | 1.0 | MAE `67.430537`，MAPE `138.878661%` | 77434 | 276.55 |

raw `<answer>` 格式诊断：

| task | 样例数 | 严格 1 对 `<answer>` | 缺少 `<answer>` | `<final_answer>` | 空输出 |
|---|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1188 | 5 | 0 | 0 |
| `reasoning_etiological` | 207 | 207 | 0 | 0 | 0 |
| `reasoning_correlation` | 1592 | 1566 | 23 | 0 | 0 |
| `reasoning_forecasting` | 280 | 280 | 0 | 0 | 0 |

这里要分清两件事：

- ST-Test 上 raw 格式明显比 paper_cases 好，大部分样例都严格输出了 `<answer>...</answer>`。
- 但本轮评估用的是当前仓库中的 `evaluation/evaluate.py` / `evaluation/evaluate_qa.py`，而 `evaluate_qa.py` 已在前面为 paper_cases 做过通用 parser 增强。因此报告时应写作“当前 evaluate 结果”，不要包装成完全未改动的原始官方 parser 结果。
- 上表是参数不合规预跑，不作为正式 ST-Test 结果。

### 正式 ST-Test 6144 运行结果

修复 `SamplingParams` 传递后，重新按 `max_tokens=6144` 完整运行 ST-Test 四类数据。本轮正式配置：

- 模型：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B`
- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- GPU：1 张 A100
- `max_tokens=6144`
- `temperature=0.2`
- `num_gpus=1`
- `num_gpus_per_process=1`

输出目录：

| task | 输出目录 |
|---|---|
| `reasoning_entity` | `exp/sttest_full_entity_6144/` |
| `reasoning_etiological` | `exp/sttest_full_etiological_6144/` |
| `reasoning_correlation` | `exp/sttest_full_correlation_6144/` |
| `reasoning_forecasting` | `exp/sttest_full_forecasting_6144/` |

当前 evaluate 结果：

| task | 样例数 | evaluated | missing | coverage | 指标 | total input tokens | avg input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1194 | 0 | 1.0 | accuracy `0.747906` | 495068 | 414.63 |
| `reasoning_etiological` | 207 | 207 | 0 | 1.0 | accuracy `0.956522` | 79543 | 384.27 |
| `reasoning_correlation` | 1592 | 1592 | 0 | 1.0 | accuracy `0.831658` | 717765 | 450.86 |
| `reasoning_forecasting` | 280 | 280 | 0 | 1.0 | MAE `68.317056`，MAPE `123.289149%` | 77434 | 276.55 |

raw `<answer>` 格式与 token 诊断：

| task | 样例数 | 严格 1 对 `<answer>` | 缺少 `<answer>` | 空输出 | response token 最大值 | 达到 6144 |
|---|---:|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1175 | 15 | 0 | 6043 | 0 |
| `reasoning_etiological` | 207 | 207 | 0 | 0 | 1060 | 0 |
| `reasoning_correlation` | 1592 | 1545 | 47 | 0 | 6071 | 0 |
| `reasoning_forecasting` | 280 | 278 | 1 | 0 | 6094 | 0 |

完整 raw 输出与 gold 对齐附件：

```text
00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl
```

该文件共 `3273` 行，每行包含：

- `task`
- `idx`
- `input`
- `gold_output`
- `raw_response`
- `parsed_prediction`
- `parsed_gold`
- `response_tokens_tokenizer`
- `<answer>` / `<final_answer>` tag 计数
- 是否空输出
- 是否达到 `6144`

汇总附件：

```text
00_new_codes/reports/artifacts/sttest_full_6144_summary.json
00_new_codes/reports/artifacts/sttest_full_6144_first3_preview.json
```

因此，本轮没有省略 ST-Test 的真实输出和正确输出；正文只放汇总，完整逐条证据在附件中。

当前本地可见的 Stage 2 SmartTest 结果只有 2 条样例：

```text
00_new_codes/repro_autodl/experiments/stage2_results/experiment1_smarttest/forecasting_prediction.jsonl
00_new_codes/repro_autodl/experiments/stage2_results/experiment1_smarttest/non_forecasting_prediction.jsonl
```

对应输入也只有：

```text
00_new_codes/repro_autodl/experiments/stage2_subsets/experiment1_smart_test/SmartTest.jsonl
```

该文件当前为 2 行，因此它只能作为小样例 smoke / sanity check，不能作为论文真实效果验证。

已有 SmartTest 2 条结果显示的现象和 paper_cases 一致：

- 模型能加载并生成。
- raw response 不稳定输出 `<answer>`。
- forecasting 容易长输出并重复 `(TRAINING DATA END)`。
- 旧评测逻辑因为要求 `<answer>` 或抽取过粗，容易把可读答案记为 parse failed 或错误。

因此，后续若要继续做更严格的“论文真实效果复现”，建议在本轮 6144 输出基础上继续核对：

1. 固定本轮 `exp/sttest_full_*_6144/` 输出，另备一份只读结果。
2. 如要对齐论文指标，需确认论文评估时的 `max_tokens`、temperature、parser 版本和是否有额外 stopping 规则。
3. 使用仓库原始 `inference/inference_tsmllm_vllm.py` / `evaluation/evaluate.py` 作为优先依据。
4. 避免每条样例重复加载 8B 模型；完整跑时按任务长进程运行。
5. 同时记录三类指标：
   - raw format success：模型是否严格输出 `<answer>`。
   - parse success / formatted answer：是否能通用抽取并规范化答案。
   - official metrics：选择题 accuracy，forecasting MAE / MAPE。
6. 不改 gold、不删样例、不为样例改 prompt；若要处理格式，只做通用 parser / stopping / 后处理，并保留 raw response。

本轮结论是：paper_cases 的答案内容质量已经较好，但 raw 格式不合格；ST-Test 四类真实数据已经完整跑完一轮 `max_tokens=6144` 验证。当前结果可作为本地真实数据验证结果，但如果要严肃声称复现论文整体效果，还需要进一步核对论文原始评估配置和 parser 版本。
