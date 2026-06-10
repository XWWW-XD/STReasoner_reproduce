# Etiological 与 paper_cases（Wave 10–11）

Artifact：[`etiological_and_paper_cases.json`](artifacts/etiological_and_paper_cases.json)。

---

## 1. Etiological 三 run（6144）

| 指标 | 值 |
| --- | ---: |
| n | 207 |
| flip_count | **27** |
| flip_rate | 13.04% |
| no_tag (三 run) | **0%** |
| think_len 中位 | **2700** |

独立 artifact：[`etiological_stability.json`](artifacts/etiological_stability.json)。

### flip 模式（三 run 对错串）

| 模式 | 计数 |
| --- | ---: |
| RRR | 178 |
| RWR | 9 |
| RRW | 8 |
| WRR | 4 |
| RWW | 3 |
| WRW | 2 |
| WWW | 2 |
| WWR | 1 |

**解读**：以 **`flip_indices`（27 题）** 为准；`flip_patterns` 按三 run 对错串枚举，合计 29 条组合计数，与 flip_count 口径不同。

flip idx 全表：10, 16, 19, 20, 27, 44, 47, 57, 59, 65, 72, 76, 90, 98, 100, 106, 110, 123, 127, 128, 143, 148, 158, 163, 175, 176, 183。

---

## 2. 512 vs 6144（同题）

见 [`09`](09-512与6144同题对照.md)：etiological flip **14**；6144 acc **0.9565** > 512 **0.9372**；无 tag 率均为 0。

---

## 3. Graph 消融

见 [`10`](10-graph消融样本级.md)：etiological flip_total=**27**（with vs w/o 单 run 配对），graph rescued **9**、harmed **6**。

---

## 4. Paper cases（4 题 × graph ±，已补全）

| paper_case_id | task | gold | baseline run1 | with | w/o | graph 影响 |
| --- | --- | --- | --- | --- | --- | --- |
| table6 eti | eti | D | D | D | D | spatial 36→23 |
| table7 entity | entity | C | C | C | C | spatial 42→20 |
| table8 corr | corr | D | D | D | D | spatial 40→25 |
| table9 fcst | fcst | [19.86…] | [19.88…] | MAE **0.12** | MAE **0.07** | w/o 略优 MAE |

数据源：`paired_results.jsonl` + `exp/stage2.4_graph_ablation_paper_cases_6144_*`。artifact：`etiological_and_paper_cases.json`。

**解读**：四题 choice 均对；graph 主要差在 **spatial 叙述量** 与 forecasting **MAE 微差**，非 rescue/harm 典型 flip 题。

---

## 5. 下一步（roadmap 引用）

- E3：4 题固定清单 + strict/loose 双口径 + graph ± 对照
- 扩展 etiological：68.6% loose 数值叙述（阶段一 `03`）+ strict mismatch=0 → 需 **新 strict 口径** 或人工窗口
