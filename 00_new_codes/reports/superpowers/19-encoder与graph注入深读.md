# TS encoder 与 graph 注入深读（Wave 19）

**Skills**: explore → 对照 report 39/37  
**Plan**: [`_plans/wave19.md`](_plans/wave19.md) | **Review**: [`_reviews/wave19-review.md`](_reviews/wave19-review.md)  
**Artifact**: [`encoder_trace.json`](artifacts/encoder_trace.json)

---

## 1. TimeSeriesEmbedding（`inference/vllm/chatts_vllm.py`）

| 行号 | 机制 |
| ---: | --- |
| 52–82 | `TimeSeriesEmbedding`：patch_size、num_layers、MLP `Linear→GELU→…→hidden_size` |
| 84–91 | `forward`：mask 末维 → `valid_lengths` → **patch_cnt = ceil(vl/patch_size)** |
| 71–72 | 无 position 时 **input_size = 1 × patch_size**（标量 patch，非 verbatim 文本） |

Raw TS **不以明文**进入 prompt（与阶段一 memo §A 一致）；模型见的是 **patch embedding**，非 `Node k [a-b]: v1,v2` 字符串。

---

## 2. Graph 注入路径

| 通道 | 说明 |
| --- | --- |
| **Prompt 文本** | stage2.4 w/o graph：删 spatial/edge 描述（report 17）— **推理侧**干预 |
| **权重** | SFT 含 graph 文本；w/o 仅改 prompt 未改权重 → T2-4 训练 vs 推理分叉 |
| **Embedding** | graph 结构不进 `TimeSeriesEmbedding.forward`；spatial 信息在 **自然语言 CoT** |

wave10：graph ± 样本级 rescued 161/226 — **选项层**收益，strict mismatch 常不变。

---

## 3. 与 MLP 复述瓶颈（report 39）

- Strict mismatch 要求 thinking 内 **数值窗与 raw patch 对齐** — MLP 压缩后 **不可** 逐点 verbatim。
- T1-5 probe（roadmap）：禁止 CoT、仅输出 Node 行 — 若仍高 mismatch → **放弃 prompt 逼复述**。

---

## 4. Intervention 锚点

| ID | 锚点 |
| --- | --- |
| T1-4 | prompt graph 文本 ±（wave10 ledger） |
| T2-4 | `supervised.py` 数据管线 w/o graph SFT vs 推理删图 |
| T3-2 | `chatts_vllm.py:52+` 架构/MSE 辅助（高成本） |

**Falsifier**：若 graph w/o 仅降 spatial mention 但 Δacc≈0 → 收益非「叙述更多」而是 **与 gold 对齐的推理链**（需 case 级）。
