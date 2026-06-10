# 第六波：未闭合 thinking 取证与 strict×mention 交叉

脚本：[`_analysis/compute_wave6.py`](_analysis/compute_wave6.py)。承接 [`06`](06-第五波strict全文与parser分类.md) §6。

---

## 1. 12+1 题 all_runs_parser_issue：不是 input 截断

[`unclosed_thinking_forensics.json`](artifacts/unclosed_thinking_forensics.json) 对 correlation **12** 题 + entity **1024**（三 run 皆无 tag）逐 run 诊断。

| 现象 | 结论 |
| --- | --- |
| `generated_answer.json` 的 `num_tokens` | **输入** token 数（见 `inference_tsmllm_vllm.py`），**非**生成长度 |
| 无 `<answer>` | 与 **未闭合** `<think>` **1:1**（全 task/run） |
| response 末段 | 大量 **数字重复**（如 `96.59, 96.59, …` / `0.00, 170.00, …`） |
| response 长度 | 无 tag 样本 median **~5.8k–6.0k** 字符；有 tag 样本 median **~2.1k** |

**根因（39 run 次）**：`unclosed_thinking_numeric_loop` **25** + `unclosed_thinking_other` **14**。模型在 thinking 内 **陷入数值复读**，在写完 `</think><answer>` **之前** 触发生成上限（6144 max_tokens；数字密集时字符≈token），**不是 prompt/input 被截断**。

典型 idx：**191**（三 run 均无 close tag，末段 dominant `96.59` share≈0.92）、**384**（`0.00/170.00` 交替）。

---

## 2. 全局无 tag 率

| task | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| correlation | 47/1592 (**2.95%**) | 53 (**3.33%**) | 66 (**4.15%**) |
| entity | 15 (**1.26%**) | 10 | 11 |
| etiological | 0 | 0 | 0 |

run3 correlation 无 tag 略升，与 wave5「parser 噪声」一致，但 **绝对量仍小**（66 题）。

---

## 3. loose 假三不同 = 未闭合长 thinking 的副产物

[`strict_triple_mention_cross.json`](artifacts/strict_triple_mention_cross.json)（run1 thinking 体长度，含未闭合时从 open tag 到文末）：

| 分组 | correlation think_len median | broad mention median |
| --- | ---: | ---: |
| 全 corpus | 2092 | 3 |
| **strict 三不同** | 2300 | 2 |
| **loose 假三不同（parser 噪声）** | **5351** | 1 |
| flip 但非 triple | 2279 | 3 |
| 三 run 全对稳定 | 2031 | 3 |

**解读**

- **strict 三不同** 与 corpus / 普通 flip **长度接近**——是真选项漂移，不是 generation 失败。
- **loose 假三不同** think_len **≈2.5× corpus**，且 broad mention **更低**——evaluate 把 **未闭合 thinking 前缀** 当 pred，三 run 前缀微差 → 假「三不同」。
- strict 三不同 **不** 靠「写更多数」驱动（mention 略低于 corpus）。

entity 侧模式相同：loose 噪声 median think **5558** vs strict **2466**。

---

## 4. 与 evaluate 链路的关系

`load_prediction_files` → `_extract_tag_content`：无 tag 时 **整段 response 回退为 pred** → `_normalize_choice` 截断前缀 → loose triple + parse fail。

这 **不是** strict mismatch 口径问题，而是 **生成未完成 + evaluate 回退策略** 叠加。

---

## 5. 补分析摘要

| RQ | 结论 |
| --- | --- |
| RQ14：12 题无 tag 是否 input 截断？ | **否**；未闭合 thinking + 数值复读至 **~6k 字符** |
| RQ15：strict 三不同是否更长/更多 mention？ | **否**；与 flip 普通题相近；**假三不同** 才极长 |
| RQ16：无 tag 是否扩散？ | correlation run1→3：**2.95%→4.15%**；entity 稳定 ~1% |

---

## 6. 仍可继续（wave7+）

1. 对 66 题无 tag 估 **output token**（若 log 有 finish_reason 可对读）。
2. strict 三不同 × **题目图规模 / 序列长度**（ST-Test jsonl 可读）。
3. 受控 probe / 降 max_tokens 对比（需新推理）。

---

## 7. 附件

| 文件 | 内容 |
| --- | --- |
| [`unclosed_thinking_forensics.json`](artifacts/unclosed_thinking_forensics.json) | 12+1 题 + 全局无 tag 率 + 长度对比 |
| [`strict_triple_mention_cross.json`](artifacts/strict_triple_mention_cross.json) | strict / loose / flip / stable × think & mention |
| [`loose_triple_noise_breakdown.json`](artifacts/loose_triple_noise_breakdown.json) | loose−strict bucket 索引 |
| [`_analysis/compute_wave6.py`](_analysis/compute_wave6.py) | 本波脚本 |
