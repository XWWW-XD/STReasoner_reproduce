# 训练与 SFT 通路深读（Wave 18）

**Skills**: brainstorming → explore  
**Plan**: [`_plans/wave18.md`](_plans/wave18.md) | **Review**: [`_reviews/wave18-review.md`](_reviews/wave18-review.md)  
**Artifact**: [`training_trace.json`](artifacts/training_trace.json)

---

## 1. SupervisedProcessor（`src/llamafactory/data/processor/supervised.py`）

| 行号 | 机制 |
| ---: | --- |
| 82–84 | Source（prompt）→ `IGNORE_INDEX` |
| 87–96 | Target（**assistant 全文**：thinking + answer）→ **参与 CE** |
| 105–106 | 注释：`labels` = `<ignore>…<ignore> Y <eos>` — Y 含整段 CoT |
| 141–145 | debug：`valid_labels` decode — 可见 thinking 在 loss 内 |

Timeseries 经 `template.mm_plugin.process_token_ids`（L49）并入 multimodal input — **非** 单独 MSE 对齐 raw patch。

---

## 2. 与 inference / evaluate 三角

| 阶段 | 优化/计分 |
| --- | --- |
| Train | 全文 response CE |
| Inference | 采样至 `max_tokens` |
| Evaluate | `_extract_tag_content("answer")` 或 fallback |

→ **flip ⊥ strict mismatch**（阶段一）：训练不要求 thinking 数值窗对齐；evaluate 不看 thinking。

---

## 3. repro probe（只读登记）

- `repro_autodl/experiments/results/stage1_lora_*.jsonl`：LoRA probe **非** ST-Test 官方指标 — registry wave8 已标 pipeline。
- 官方脚本锚点：`scripts/qwen3-8b/` + [`10_sft_training_flow.md`](../../repro_kaggle/experiments/stage1_docs/streasoner_code_reading/10_sft_training_flow.md)

---

## 4. Intervention

| ID | 改哪里 |
| --- | --- |
| T2-3 | `supervised.py` target 构造 — thinking mask / 分段 loss |
| T3-1 | 仅 answer CE（Falsifier：acc 崩溃 → 依赖长 CoT 形式） |

---

## 5. Falsifier

若 T3-1 acc 保持且 flip 降 → thinking 对 official 指标 **边际小**（与 T1-6 answer-only 对照互证）。
