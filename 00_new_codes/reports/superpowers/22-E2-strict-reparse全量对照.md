# E2：T0-1 strict reparse 全量对照（Wave 23 / 执行包 E2）

脚本：[`_analysis/compute_e2_strict_reparse.py`](_analysis/compute_e2_strict_reparse.py) → [`artifacts/e2_strict_reparse_summary.json`](artifacts/e2_strict_reparse_summary.json)。

**Mechanism**：官方 `load_prediction_files` 无 `<answer>` 时回退整段 response → thinking 噪声进入 `_normalize_choice`。  
**Intervention**：T0-1——无 tag 标 `parse_fail`，禁止 fallback。  
**Falsifier**：loose 三不同应降至 strict 量级 → **correlation 67→18 已验证**。

---

## 1. parse_fail 率（三 run 6144）

| task | run1 | run2 | run3 | 任 run fail | 三 run 全 fail |
| --- | ---: | ---: | ---: | ---: | ---: |
| correlation | 2.95% (47) | 3.33% (53) | 4.15% (66) | 108 | **12** |
| entity | 1.26% (15) | 0.84% (10) | 0.92% (11) | 33 | 1 |
| forecasting | 0.71% (2) | 1.07% (3) | 0.36% (1) | 6 | 0 |
| etiological | 0% | 0% | 0% | 0 | 0 |

**parse_fail ≡ 无 `<answer>` tag**（`parse_fail_but_has_tag` 全任务为 0）。

---

## 2. acc：official（loose）vs strict（仅 parse_ok）

### correlation（run1 示例）

| 口径 | 分母 | correct | acc |
| --- | ---: | ---: | ---: |
| official / loose | 1592 | 1324 | **0.8317** |
| strict parse_ok | 1545 | 1324 | **0.8570** |

**correct 数不变、分母缩小**——fallback 曾把部分「本无法解析」样本算进分母并判错；strict 剔除 47 题后 acc 上升。**不是模型变强，是口径变干净。**

三 run strict acc（parse_ok）：run1 **0.8570**，run2 **0.8596**，run3 **0.8578**。

---

## 3. 三 run flip（对错标签不稳定）

| task | loose flip | strict flip | Δ |
| --- | ---: | ---: | ---: |
| correlation | 261 | **173** | −88 |
| entity | 357 | **333** | −24 |
| etiological | 27 | 27 | 0 |
| forecasting | 1 | 1 | 0 |

strict flip 仅在 **三 run 均 parse_ok** 的子集上计数；仍显著高于 etiological，说明 correlation/entity 的不稳定 **不全是 parser 假 flip**，但 **~34% correlation flip 来自 parser 通道**。

---

## 4. 三选项全不同（falsifier ✅）

| task | loose | strict | wave4 对照 |
| --- | ---: | ---: | ---: |
| correlation | **67** | **18** | 67 / 18 ✅ |
| entity | **40** | **31** | 40 / 31 ✅ |
| etiological | 1 | 1 | 1 / 1 ✅ |

roadmap falsifier **通过**：loose 三不同中 **49 题 correlation** 级噪声与 wave5 叙事一致。

---

## 5. T0-2 建议（未改 evaluation 源码）

报告层应并列：

- `acc_official`（当前 evaluate）
- `acc_parse_ok`（strict 分母）
- `parse_fail_rate`

本 artifact 已含 `official_metrics` / `loose_reparse` / `strict_reparse` 三列，可直接进 dashboard。

---

## 6. 执行边界

本报告为 superpowers **可执行项 E2 的交付**；T0-2 见 [`23`](23-T0-2-parse分层与flip归因.md)。**工作流到此终止**。

E1/E3/E4 见 [`14`](14-后续实验与优化计划.md)——**外部 GPU 规格**，非本环境、非 superpowers 待办；不在此列出「下一步行动」。

---

## 7. 附件

| 字段 | 路径 |
| --- | --- |
| 全量 JSON | `artifacts/e2_strict_reparse_summary.json` |
| strict 三不同 idx | 同上 `triple_diverse.*.e2_strict_recomputed.indices` |
