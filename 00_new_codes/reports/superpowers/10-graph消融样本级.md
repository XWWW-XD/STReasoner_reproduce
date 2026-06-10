# Graph 消融样本级 ledger（Wave 9）

对比：`stage2.4_graph_ablation_sttest_6144_without_graph_reasoning_*`（w/o）vs 含 graph 的 6144 baseline（run1）。Artifact：[`graph_ablation_sample_ledger.json`](artifacts/graph_ablation_sample_ledger.json)。

---

## 1. 任务汇总

| task | n | graph rescued (W→R) | graph harmed (R→W) | flip_total | rescued∩flip | rescued∩strict_mismatch_run1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correlation | 1592 | **161** | 77 | 261 | 58 | **30** |
| entity | 1194 | **226** | 114 | 357 | 89 | 0 |
| etiological | 207 | **9** | 6 | 27 | 3 | 0 |

**rescued**：w/o 错、with 对；**harmed**：反向。

---

## 2. 机制假说

- **Correlation +161**：graph 文本注入 prompt，帮助选项题在 **复述仍错** 时选对——与阶段一「flip ⊥ mismatch」相容（仅 30/161 rescued 同时 run1 strict mismatch）。
- **Entity +226 rescued 但 mismatch_run1=0**：graph 增益 **不在 strict 复述窗口**，而在 final choice——与 entity strict mismatch≈0 的口径一致。
- **Etiological 净 +3**：rescued 9、harmed 6，与三 run flip 27 同阶。

---

## 3. 样本索引（审计入口）

- correlation rescued 样例：6, 8, 24, 45, 47, …（artifact 列 25 个）
- correlation harmed 样例：35, 46, 78, 112, …（46、78 亦出现在 512/6144 flip 的 unclosed 根因）
- entity rescued：2, 5, 12, 14, 16, …

---

## 4. 与 macro +6pp

stage2.4 文档宏观 **~+6pp** 可由 correlation+entity 净 rescued（161+226−77−114=**196** 样本级翻转）支撑；**非**逐题 strict 复述改善。

---

## 5. Intervention 锚点（代码级见阶段三 §14 增补）

stage2.4 脚本对 prompt 做 **regex 去除 graph reasoning 段**；evaluate 仍只读 `<answer>`。E4 graph 对照应固定 **同一 checkpoint、同一 max_tokens**。
