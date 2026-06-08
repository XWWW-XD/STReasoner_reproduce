# evaluate 与 raw 格式诊断的差异，及 paper_cases / ST-Test 对比说明

日期：2026-05-30

---

## 1. 评分 vs 格式诊断（不是两套分数）

| 维度 | 作者 evaluate（唯一正式口径） | raw 标签计数（报告附件，可选） |
|---|---|---|
| 问什么 | 能否从 `<answer>...</answer>` 内 parse 出答案并对比 gold | raw 里 `<answer>` / `</answer>` 各出现几次 |
| `Answer: D` 无标签 | **parse 失败** → coverage↓，不计入 accuracy | 计为缺少标签 |
| 指标 | accuracy / MAE / MAPE / **coverage** | `exact_answer_tag_count` 等 |
| 用途 | 论文 benchmark | 排查 prompting 是否生效；**不单独算分** |

`evaluation/evaluate_qa.py` 当前逻辑（已收严）：

- 选择题：`_extract_tag_content("answer")` 后只接受单个 A–D；
- forecasting：读标签内 JSON/短数组；**不**扫全文所有数字；
- **无** `Answer:` / `\boxed{}` 兜底。

因此：**应用 format prompt 后，evaluate 结果与「strict 标签计数」应大体一致**；不一致时先查 evaluate 实现是否被改过，而不是再造第二层 parser。

---

## 2. ST-Test 上为什么 `<answer>` 标签合规率高

### 2.1 官方推理会强制追加格式说明

作者 `inference/inference_tsmllm_vllm.py` 在生成前对每条 prompt 追加 `inference/prompt.json` 中的后缀，例如选择题：

```text
Output Format: <think>Your step-by-step reasoning process</think><answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>
```

数据集 jsonl 的 `input` 字段**通常不含**这段（paper_cases 对应的 4 条 ST-Test 行均为 `Output Format in input: False`），后缀由**推理脚本**追加，不是数据自带。

### 2.2 6144 全量标签汇总（报告 10 / 附件）

| task | 样例数 | 严格 1 对 `<answer>` | 缺少 `<answer>` | 空输出 |
|---|---:|---:|---:|---:|
| reasoning_entity | 1194 | 1175 | 15 | 0 |
| reasoning_etiological | 207 | 207 | 0 | 0 |
| reasoning_correlation | 1592 | 1545 | 47 | 0 |
| reasoning_forecasting | 280 | 278 | 1 | 0 |

另有少量 **标签重复**（如 entity 4 条 `open=2, close=2`），evaluate 可能 parse 失败或异常，需单独看 raw。

附件路径：

```text
00_new_codes/reports/artifacts/sttest_full_6144_summary.json
00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl
```

---

## 标签问题根本原因

**早期** Stage 2.2：`build_inputs()` 只用 `sample["input"]`，未追加 `prompt.json` 后缀。

**当前** Stage 2.2：`compose_model_prompt()` + 默认 `--format-prompt true`，与官方一致。

作者 ST-Test 链路：`inference/inference_tsmllm_vllm.py` + vLLM + **必加格式后缀**。

### 3.2 同一批题：ST-Test 6144 vs 早期 paper_cases（无后缀）raw 对比

| index | category | ST-Test（vLLM + 后缀）raw 末尾 | 早期 paper_cases（HF，无后缀） | 标签计数 |
|---:|---|---|---|---|
| 118 | etiological | `</think>\n\n<answer>D</answer>` | 两次 `Answer: D` | ST 1/1；旧 paper 0/0 |
| 982 | entity | `</think>\n\n<answer>C</answer>` | `\boxed{C}` | ST 1/1；旧 paper 0/0 |
| 547 | correlation | `</think>\n\n<answer>D</answer>` | `Answer: D` | ST 1/1；旧 paper 0/0 |
| 9 | forecasting | `<answer>[19.88, 19.90, 19.91]</answer>` | JSON + `</final_answer>`，无 `<answer>` | ST 1/1；旧 paper 0/0 |

旧 paper_cases 证据：`parserfix_reparse_existing_6144/paper_cases_prediction.jsonl`（**历史结果**，无 format-prompt）。

**在收严 evaluate 下**，旧 paper_cases 的 `Answer: D` **不能**再被 parser 抽成正确 accuracy；须重跑带 `--format-prompt true` 后再评。


---

## 6. 相关文件索引

| 用途 | 路径 |
|---|---|
| 作者格式后缀 | `inference/prompt.json`、`inference/prompt_utils.py` |
| 作者 ST-Test 推理 | `inference/inference_tsmllm_vllm.py` |
| 作者评估 | `evaluation/evaluate.py`、`evaluation/evaluate_qa.py` |
| paper_cases 脚本 | `00_new_codes/repro_autodl/experiments/scripts/stage2_2_run_paper_cases.py` |
| 维护规则 | `00_new_codes/guides/agents修改文件必读规则.md` |
| ST-Test 6144 附件 | `00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl` |
| 旧 paper_cases 预测（无 format-prompt） | `.../parserfix_reparse_existing_6144/`（历史） |

---

## 7. 总结

**评分只认作者 tag-first evaluate。** ST-Test 标签合规率高，是因为 vLLM 推理**每次都加了 `prompt.json` 后缀**；paper_cases 早期差，是因为 HF 路径**漏了这一步**——已在 Stage 2.2 默认补上；重跑后用 **coverage + accuracy** 验收，不必再维护 Strict/Official 双层。
