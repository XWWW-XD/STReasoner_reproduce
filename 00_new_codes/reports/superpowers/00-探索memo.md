# 探索 memo（阶段 0–1）

归纳式研究的工作笔记。条目按「事实 / 对照 / 意外 / 悬而未决」组织；**不是**最终结论。

---

## A. 源码链路（事实）

| 观察 | 证据 |
| --- | --- |
| ST-Test 样本经 `inference/inference_tsmllm_vllm.py` 加载；`input` 进 prompt，`timeseries` 单独传入 vLLM | `prepare_batches()` 读 `sample["input"]` 与 `sample["timeseries"]` |
| Prompt 里是 `<ts><ts/>` 占位，**不是** raw 数值明文 | 样例见 `exp/.../generated_answer.json` 的 `question_text` |
| TS 经 `TimeSeriesEmbedding`：patch_size=8，多层 MLP，patch 间无 cross-attention | `inference/vllm/chatts_vllm.py` L51–82 |
| 生成超参由 CLI 传入：`temperature` 默认 0.2，`max_tokens` 默认 512 但 ST-Test 正式跑 6144 | `inference_tsmllm_vllm.py` L76–79, L209–214, L312 |
| worker 使用调用方传入的 `SamplingParams`（非 worker 内写死的 0.5） | `llm_utils.py` L186–188 |
| **评估不看 thinking**：`load_prediction_files` 对 full response 做 `_extract_tag_content`，只留 `<answer>` | `evaluation/evaluate_qa.py` L147–148 |
| 输出格式由 `inference/prompt.json` 强制：thinking + answer 两段 | 四任务均有 `<think>…</think><answer>…</answer>` |

**对照 run1/2/3**：`run.log` 显示同一模型路径 `base_model/STReasoner-8B`、同一 dataset 路径；每 task 单独起 vLLM 进程（非 3273 条一次加载）。

---

## B. 三 run 全局指标（对照）

来源：`superpowers/artifacts/metrics_table.json`（自 `evaluation_metrics.json` 读取）

| task | run1 | run2 | run3 | 印象 |
| --- | --- | --- | --- | --- |
| correlation acc | 0.8317 | 0.8310 | **0.8222** | run3 最低 |
| entity acc | 0.7479 | 0.7337 | 0.7312 | run2/3 均低于 run1 |
| etiological acc | 0.9565 | 0.9275 | 0.9275 | run2/3 同且低于 run1 |
| forecasting MAE | 68.32 | **64.30** | 64.75 | run2/3 更低 |
| forecasting MAPE | **123.29** | 135.20 | 133.54 | run2/3 更高 |
| forecasting missing | 0 | 2 (idx 84,262) | 1 (idx 82) | parser 未抽到预测，非空 response |

**意外**：task 指标 run 间有系统偏移（correlation/entity/etiological 降，forecasting MAE↓ 但 MAPE↑），不能用一个标量概括「三次一样」。

---

## C. 逐样本 final answer 稳定性（意外）

来源：`superpowers/artifacts/sample_outcomes.json`（3273 条 × 3 run，用当前 parser 判对错）

| task | 三 run 间「对错标签」翻转样本数 | 占该 task 比例 |
| --- | ---: | ---: |
| correlation | **261** | 16.4% |
| entity | **357** | 29.9% |
| etiological | 27 | 13.0% |
| forecasting（严格序列相等） | 1 | 0.4% |
| forecasting（MAE 差 >1） | 166 | 59.3% |

**事实**：choice 类任务在 **temperature=0.2** 下，final `<answer>` 仍大量随 run 变化。

**对照**：同一样本三次 run 的 **thinking 长度**几乎不变（median ~2100–2700 字符，见 `artifacts/response_length_stats.json`）。

**悬而未决**：run 间指标波动，多大程度来自 final answer 采样，而非 thinking 长度/结构？

**补分析（2026-05-30）**：见主报告 §9——run1→run3 correlation acc Δ **完全** 由 RW−WR=−15 题解释；**70%** flip 样本从未 strict mismatch。

---

## D. thinking 内数值复述（strict mismatch，对照）

来源：已有 `mlp_encoder_focused_analysis/{run1_recheck,run2,run3}/`（strict 口径，未重跑）；汇总见 `artifacts/mismatch_stability.json`。

| task | run1 mismatch 样本 | 三 run 交集 | 占 run1 比例 |
| --- | ---: | ---: | ---: |
| forecasting | 141 | 104 | 74% |
| correlation | 347 | 190 | 55% |
| entity | 0 | 0 | — |
| entity run2 only | — | idx **1175**（4 窗口） | 单次 run 独有 |

**意外**：复述 mismatch **部分 persistent、部分 drift**；correlation 约一半 run1 mismatch 在三次中不完全复现。

---

## E. mismatch 与 task 对错（意外，补分析后）

来源：`artifacts/mismatch_vs_correct.json`（mismatch 来自 artifact；对错用 evaluate parser）

**run1 correlation**：mismatch 组 309/347 答对（**89.0%**）；非 mismatch 组 1015/1245 答对（**81.5%**）→ mismatch 组 **更**常答对。

**run1 forecasting**：mismatch 组 0/141 答对（序列匹配意义下几乎全错）；非 mismatch 组 1/139 答对 → **方向相反**。

**悬而未决**：同一「thinking 复述对不上 raw」标签，在不同 task 上与 final 指标关系 **符号相反**——能否用任务形式（序列回归 vs 单选）解释？

---

## F. 从 memo 事后归纳的研究问题（阶段 2）

| ID | 问题（由哪些 memo 条目推出） | 补分析 | 完成标准 |
| --- | --- | --- | --- |
| **RQ1** | E + B：choice 任务 final answer 高翻转率下，全局 acc 波动多少来自采样而非数据/模型差异？ | 已算 flip 率；报告里对照 run 指标 | 能定量说出 correlation/entity flip 比例与 acc Δ 量级 |
| **RQ2** | E + D：thinking 内 strict mismatch 与 task 对错的关系为何在 forecasting vs correlation **反向**？ | `mismatch_vs_correct.json` 三 run | 两 task 各给出 mismatch/other 组正确率并排 |
| **RQ3** | D：复述 mismatch 标签 run 间 drift（~26–45% 非三 run 交集）主要来自「没写出数值列表」还是「写出但数字/窗口变了」？ | 抽 503/480、volatile idx 读 raw response | 至少 2 例机制分类 |
| **RQ4** | A + C：评估链路只看 `<answer>`，thinking 复述错误是否 **结构上** 无法影响官方 metrics？ | 源码引用 + RQ2 结果 | 明确「能/不能通过现有 evaluate 传导」 |

---

## G. 仍悬而未决 / 已闭合

- ~~entity 非 strict 复述~~ → **已宽口径计数**（591/1194 loose），见 `03` §4；仍缺与 raw 对齐的新 metric。
- ~~forecasting parser-missing vs mismatch~~ → **无重叠**（`forecasting_gaps.json`）。
- 训练目标：SFT 对 **全文 response** CE，**无** fidelity loss（`03` §4 只读对照）；非 ablation 因果。

---

## H. 阶段 2 深化（2026-05-30，见 `03-深化探索与终止说明.md`）

| 发现 | 要点 |
| --- | --- |
| Persistent core 分型 | correlation **35% constant_fill**；forecasting **63% local_drift** |
| MAE 波动 | persistent / never mismatch 的 MAE range>1 比例 **~56–58%**，无显著差 |
| Entity loose 叙述 | **591/1194** 有宽口径 Node+数字，strict mismatch **0** |
| 终止 | 受控 probe 需 **新推理**；**现有 exp 仍可迭代挖掘**（见 `04`） |

**补分析（2026-05-30）**：见主报告 §9——run1→run3 correlation acc Δ **完全** 由 RW−WR=−15 题解释；**70%** flip 样本从未 strict mismatch。

---

## I. 第三波（见 `04-现有数据第三波挖掘.md`）

| 新发现 | 要点 |
| --- | --- |
| 分型 × acc | persistent correlation：`constant_fill` **87.9%** vs `local_drift` **78.3%** |
| Flip × think_len | correlation flip **+470** 字符 vs 稳定对题 |
| MAE 账本 | run2/run3 各 **~135** 题优于 run1 → 全局 MAE↓ |
| 三选项全不同 | correlation **67**、entity **40**（loose）→ **strict 18+31**（`05`） |

---

## J. 第四波（见 `05-第四波案例册与口径修正.md`）

- strict 三不同 **18+31**；49 correlation loose 题为 parser/格式噪声
- wave6：**未闭合 thinking 数值复读** → ~6k 字符无 tag；loose 假三不同 think **2.5×** 长（`07`）
- `volatility_rankings.csv` **843 行**可扫表
- broad mention ⇒ mismatch，反向不成立

---

## K. 阶段二（见 `08-阶段二总览.md`）

| 轴 | headline |
| --- | --- |
| 512 vs 6144 | corr flip **163**；6144 无 tag ≈2× |
| graph w/o | rescued **161 / 226 / 9** |
| eti 三 run | flip **27/207** |
| Kaggle | 4×20 条，**0** tag，not comparable |

---

## L. 阶段三 + 外部 roadmap

- 源码 trace：`16`–`21`；总览 [`15-阶段三总览.md`](15-阶段三总览.md)
- **Superpowers 工作流已终止**（只读挖掘 + CPU 执行项 22–25）
- 外部规格（GPU/改源码）：[`14-后续实验与优化计划.md`](14-后续实验与优化计划.md)——**非本队列待办**
- 验证：[`_analysis/verify_reports.py`](_analysis/verify_reports.py) **PASS**

---

## M. Superpowers CPU 执行项（已完成）

- [`22`](22-E2-strict-reparse全量对照.md)：三不同 **67→18**；flip **261→173**
- [`23`](23-T0-2-parse分层与flip归因.md)：parser 诱发 flip **33.7%**（correlation）
- [`24`](24-真实不稳定flip账本.md)：173 题真实 flip 中 **132 题 mismatch=0**
- [`25`](25-T0-3-num_tokens字段取证.md)：`num_tokens` 为 input 计数（ratio 中位 **5.8×**）
