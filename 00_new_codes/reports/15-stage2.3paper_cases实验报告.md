# Stage 2.3 paper_cases 正式实验报告

## 结论

本次补做 `13-0531实验进度总结和实验计划.md` 中的 paper_cases 实验，最终采用仓库官方 vLLM 推理链路完成 Stage 2.3：

- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- 模型：`base_model/STReasoner-8B`
- 数据：`00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl`
- `max_tokens=6144`
- `temperature=0.2`
- `inference/prompt.json` 格式后缀由官方推理脚本追加

正式结果：

- 4/4 样例完成生成。
- 4/4 样例都有唯一一对 `<answer>...</answer>`。
- 0 条空输出。
- 0 条达到 `max_tokens=6144`。
- 选择题 3/3 正确。
- forecasting：MAE `0.073333`，MAPE `0.366537%`。

这说明 report13 的核心判断成立：paper_cases 早期格式差主要来自 HF 复现脚本链路差异；同 index 在官方 vLLM + `prompt.json` 后缀下格式合规。

## 重要诊断：为什么没有继续用 HF Stage 2.2 runner 直接 run-all

先按守门规则跑了 `stage2_2_run_paper_cases.py` 的第 0 条：

- `format-prompt=true`
- `flash_attention_2`
- `max-new-tokens=6144`
- `do_sample=false`

结果：

- 模型加载成功。
- 输入构造成功。
- `generate_success=true`。
- 但 `actual_new_tokens=1`。
- 唯一新 token 是 `151645 = <|im_end|>`。
- decode 后 `response=""`。

随后做了两个诊断：

1. `do_sample=true, temperature=0.2, top_p=0.95`：仍然首 token `<|im_end|>`。
2. `format-prompt=false`：能正常生成 1183 tokens，并输出 `Answer: D`，但没有 `<answer>` 标签，因此在 tag-first evaluate 下 parse failed。

因此判断：HF `model.generate()` 路径本身能生成，但追加官方 `prompt.json` 后缀后会在第 0 条上首 token 结束。为了完成 report13 要验证的“官方格式后缀是否能让 paper_cases 合规”，本次正式实验改用官方 vLLM 推理入口，而不是继续扩大 HF 空输出。

## 输出文件

正式 vLLM 汇总：

```text
00_new_codes/repro_autodl/experiments/results/stage2.3_paper_cases_report13_6144_vllm/
├── datasets/
│   ├── manifest.json
│   ├── 00_reasoning_etiological_paper_appendix_h_table6_etiological_line118.jsonl
│   ├── 01_reasoning_entity_paper_appendix_h_table7_entity_line982.jsonl
│   ├── 02_reasoning_correlation_paper_appendix_h_table8_correlation_line547.jsonl
│   └── 03_reasoning_forecasting_paper_appendix_h_table9_forecasting_line9.jsonl
├── paper_cases_vllm_predictions.jsonl
└── paper_cases_vllm_summary.json
```

官方推理输出：

```text
exp/stage2.3_paper_cases_report13_6144_vllm_00_etiological/
exp/stage2.3_paper_cases_report13_6144_vllm_01_entity/
exp/stage2.3_paper_cases_report13_6144_vllm_02_correlation/
exp/stage2.3_paper_cases_report13_6144_vllm_03_forecasting/
```

每个目录包含：

- `generated_answer.json`
- `evaluation_metrics.json`

HF 诊断输出：

```text
00_new_codes/repro_autodl/experiments/results/stage2.3_paper_cases_report13_6144/
00_new_codes/repro_autodl/experiments/results/stage2.3_paper_cases_report13_6144_sampling_diag/
00_new_codes/repro_autodl/experiments/results/stage2.3_paper_cases_report13_6144_noformat_diag/
```

## 正式结果明细

| case | task | gold | parsed answer | tag | metric |
|---|---|---|---|---|---|
| appendix_h_table6_etiological | reasoning_etiological | D | D | 1 对 | accuracy 1.0 |
| appendix_h_table7_entity | reasoning_entity | C | C | 1 对 | accuracy 1.0 |
| appendix_h_table8_correlation | reasoning_correlation | D | D | 1 对 | accuracy 1.0 |
| appendix_h_table9_forecasting | reasoning_forecasting | `[19.86, 19.97, 20.05]` | `[19.88, 19.90, 19.92]` | 1 对 | MAE 0.073333, MAPE 0.366537% |

## 指标汇总

| 指标 | 值 |
|---|---:|
| paper_cases 总数 | 4 |
| 完成生成 | 4 |
| 空输出 | 0 |
| 唯一 `<answer>` 标签对 | 4 |
| 达到 6144 上限 | 0 |
| 选择题正确数 | 3/3 |
| forecasting coverage | 1.0 |
| forecasting MAE | 0.073333 |
| forecasting MAPE | 0.366537% |

## 输入 token 记录

| case | input tokens |
|---|---:|
| appendix_h_table6_etiological | 293 |
| appendix_h_table7_entity | 396 |
| appendix_h_table8_correlation | 729 |
| appendix_h_table9_forecasting | 182 |

## 过程日志

- 16:45：重新阅读关键代码库文档和报告，确认 report13 要求：tag-first evaluate、`format-prompt=true`、`max_tokens/max_new_tokens=6144`。
- 16:45：确认 paper_cases 当前数据为 `00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl`，共 4 条。
- 16:45：确认 A100 空闲、模型权重完整、脚本可编译。
- 16:45：用 HF Stage 2.2 runner 跑第 0 条守门，首次发现脚本根目录推导错误。
- 16:48：修复 `stage2_2_run_paper_cases.py` 的根目录定位，让 `evaluation/evaluate_qa.py` 从真实仓库路径加载。
- 16:48：重跑 HF 第 0 条，出现 `actual_new_tokens=1`、空输出。
- 16:52：给 HF runner 增加生成 token 诊断字段，确认唯一新 token 是 `<|im_end|>`。
- 16:53：采样诊断仍首 token `<|im_end|>`。
- 16:54：`format-prompt=false` 诊断能正常生成长文本，但无 `<answer>` 标签。
- 16:57：将 paper_cases 拆成 4 个单任务 JSONL，准备走官方 vLLM 链路。
- 17:00：vLLM 第 0 条 etiological 完成，官方 evaluate accuracy 1.0。
- 17:02：vLLM 第 1 条 entity 完成，官方 evaluate accuracy 1.0。
- 17:05：vLLM 第 2 条 correlation 完成，官方 evaluate accuracy 1.0。
- 17:07：vLLM 第 3 条 forecasting 完成，官方 evaluate MAE 0.073333、MAPE 0.366537%。

## 本次代码修改

修改文件：

```text
00_new_codes/repro_autodl/experiments/scripts/stage2_2_run_paper_cases.py
00_new_codes/guides/修改文件必读规则.md
```

脚本修改：

- 修复 `00_new_codes` 根目录和仓库根目录识别。
- 修复 summary 中 `paper_cases_path` 记录实际 `--dataset`。
- 增加生成诊断字段：新 token id、token text、保留 special token 的 decode 预览。
- 增加可选生成参数：`--do-sample`、`--temperature`、`--top-p`，用于受控诊断。

没有修改：

- 没有改 `inference/prompt.json`。
- 没有改 gold。
- 没有改 `evaluation/evaluate_qa.py` parser 规则。
- 没有把 `Answer: D` 改写成 `<answer>D</answer>`。
- 没有使用 `sdpa`。
- 没有降低 `6144`。

## 后续建议

1. paper_cases 汇报应使用本次 vLLM 正式结果，不使用 HF `format-prompt=true` 空输出结果当模型能力结论。
2. HF Stage 2.2 runner 目前适合作为诊断链路；若后续还要让它承担正式 format-prompt 实验，需要专门排查“追加 `prompt.json` 后首 token EOS”的原因。
3. 如果目标是论文真实效果，仍以完整 ST-Test 四类任务的 `6144` vLLM 结果为主；paper_cases 只能作为 4 条论文样例回归。
