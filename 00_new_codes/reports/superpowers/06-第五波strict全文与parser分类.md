# 第五波：strict 三不同全文、parser 分类与 volatility 标签

脚本：[`_analysis/compute_wave5.py`](_analysis/compute_wave5.py)。承接 [`05`](05-第四波案例册与口径修正.md) §9。

---

## 1. Strict 三不同 aggregate

[`strict_triple_aggregate.json`](artifacts/strict_triple_aggregate.json)：

| task | 题数 | 与 flip 交集 | run1 mismatch | 三 run 全错 | 至少一对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| correlation | 18 | **17** | **0** | **1**（idx 119） | 17 |
| entity | 31 | **30** | **0** | **1** | 30 |

**flip 模式**（correlation）：WRW 7、RWW 6、WWR 4、WWW 1；entity：RWW 13、WRW 11、WWR 6、WWW 1。

**解读**：strict 三不同几乎 **全是 flip 题**，且 **与 run1 strict mismatch 零交集**——「三 run 真选了三个不同选项」主要由 **answer 层随机性** 驱动，而非 thinking 复述标签。

---

## 2. Strict 全文摘录（JSON）

[`strict_triple_excerpts.json`](artifacts/strict_triple_excerpts.json) 含 **49 题**逐 run：

- `think_tail`：thinking 末 **500** 字符
- `answer_snippet`：`<answer>` 片段
- `parser_class`、`correct`、`flip_pattern`

### 2.1 典型：correlation 119 — 唯一三 run 全错且 strict 三不同

| run | answer | 末段 reasoning 要点 |
| --- | --- | --- |
| run1 | **A** | Node1→Node2→Node9 传播 → 「eastern district」 |
| run2 | **C** | Node0 北方中心 → 「northern outposts」 |
| run3 | **D** | Node2/3/4 多向 → 「southern regions」 |

三 run 均在 thinking 末段 **自行编造地理叙事**，选项 **A/C/D 互斥**，gold **B**——属 **语义漂移 + 选项全错**，非 parser 问题。

### 2.2 典型：correlation 154 — strict 三不同 + 仅 run2 对

| run | answer | 要点 |
| --- | --- | --- |
| run1 | D | Node4 surge → 「western hub」 |
| run2 | **C** ✓ | 同 surge，改口 western |
| run3 | B | surge → 「eastern hub」 |

同一数值峰（Node4 t=39），三 run **方向标签叙事不一致** → 选项 D/C/B 三不同。

### 2.3 entity strict

31 题全文见 JSON；模式与 correlation 类似：**flip 为主、mismatch_run1=0**。idx **0** 等为 strict 三不同（catalog 与 excerpts 对照）。

---

## 3. Parser 异常 taxonomy（loose − strict）

[`parser_anomaly_taxonomy.json`](artifacts/parser_anomaly_taxonomy.json)：

### correlation（49 题）

| 样本级 bucket | 题数 |
| --- | ---: |
| two_runs_parser_issue | **34** |
| all_runs_parser_issue | **12** |
| one_run_parser_issue | **3** |

| run 级 parser class（49×3 次） | 次数 |
| --- | ---: |
| **no_answer_tag** | **107** |
| ok_single_letter | 40 |

**机制**：evaluate 的 `load_prediction_files` 在 **无 `<answer>`** 时把整段 response（常以 `<think>` 开头）当作 pred → `_normalize_choice` 截断成不同前缀串 → **假「三不同」**。典型：idx **16**（run2 无 tag，loose 串为 thinking 前缀）、idx **191**（三 run 皆无 tag，三串皆 thinking 前缀但仍被判 loose 三不同）。

### entity（9 题 loose 非 strict）

| bucket | 题数 |
| --- | ---: |
| one_run_parser_issue | **7** |
| all_runs_parser_issue | 1 |
| two_runs_parser_issue | 1 |

entity 的 strict 覆盖率高（31/40），剩余 9 题多为 **单 run 缺 tag**。

---

## 4. Volatility top-50 半自动标签

[`volatility_top50_labels.json`](artifacts/volatility_top50_labels.json)：top30 `correctness_std` + top20 `mae_range`。

| auto_tag | top50 内计数 |
| --- | ---: |
| flip_no_mismatch | 14 |
| two_run_flip_symmetric | 10 |
| high_mae_non_persistent | 8 |
| persistent_mismatch_mae_swing | 8 |
| flip_with_mismatch | 3 |
| loose_triple_parser_noise | 2 |
| mismatch_not_persistent | 4 |
| **strict_triple_diverse** | **1**（idx **154**） |

**解读**：correctness 高波动 top 以 **普通 flip** 为主；**strict 三不同** 在 flip 总体中占比小（18/261 correlation flip），故 top30 仅命中 1 题。MAE top 中 **persistent / 非 persistent 各半**，与 wave4 一致。

---

## 5. 补分析摘要

| RQ | 结论 |
| --- | --- |
| RQ11：strict 三不同是否由复述 mismatch 驱动？ | **否**；18+31 题 run1 mismatch **均为 0** |
| RQ12：49 correlation「假三不同」主因？ | **107/147 run 次无 `<answer>` tag**（evaluate 回退整段 response） |
| RQ13：strict 三不同是否常全错？ | **极少**；各 task 仅 **1** 题三 run 全错（correlation **119**） |

---

## 6. 仍可继续（wave7+）

wave6 已完成：见 [`07-第六波未闭合thinking取证.md`](07-第六波未闭合thinking取证.md)。

1. ~~对 **12 题 all_runs_parser_issue** 查截断原因~~ → **未闭合 thinking + 数值复读至 ~6k 字符**
2. ~~strict 三不同 × thinking / mention~~ → strict 与 flip 相近；**loose 假三不同** think 2.5× 长
3. 受控 probe（需新推理）。

---

## 7. 附件

| 文件 | 内容 |
| --- | --- |
| [`strict_triple_excerpts.json`](artifacts/strict_triple_excerpts.json) | 18+31 题全文摘录 |
| [`strict_triple_aggregate.json`](artifacts/strict_triple_aggregate.json) | strict × flip × mismatch 汇总 |
| [`parser_anomaly_taxonomy.json`](artifacts/parser_anomaly_taxonomy.json) | 58 题 loose−strict 分类 |
| [`volatility_top50_labels.json`](artifacts/volatility_top50_labels.json) | top50 半自动标签 |
| [`_analysis/compute_wave5.py`](_analysis/compute_wave5.py) | 本波脚本 |
