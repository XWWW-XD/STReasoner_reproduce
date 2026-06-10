# 真实不稳定 flip 账本（Wave 25）

脚本：[`_analysis/compute_wave25_real_flip.py`](_analysis/compute_wave25_real_flip.py) → [`artifacts/real_instability_flip_ledger.json`](artifacts/real_instability_flip_ledger.json)。承接 T0-2 [`23`](23-T0-2-parse分层与flip归因.md)。

**定义**：三 run 均在 strict parse_ok 下，对错标签仍不一致的 idx（= strict flip 集）。

---

## 1. 规模

| task | loose flip | parser 诱发 | **真实不稳定** |
| --- | ---: | ---: | ---: |
| correlation | 261 | 88 | **173** |
| entity | 357 | 24 | **333** |
| etiological | 27 | 0 | **27** |

完整 idx 列表见 artifact `real_instability_indices`（correlation 173 条）。

---

## 2. correlation 真实 flip × strict mismatch

| 桶 | 数量 | 占比 |
| --- | ---: | ---: |
| 任 run 有 mismatch | 41 | 23.7% |
| 三 run 均有 mismatch | 12 | 6.9% |
| **全无 mismatch** | **132** | **76.3%** |

**结论**：在去掉 parser 噪声后，correlation **仍 173 题** strict 不稳定，其中 **132 题（76%）与 strict mismatch 无关**——主报告 §9.3 在 parse_ok 子集上仍成立。

---

## 3. correlation 模式（strict flip）

| 模式 | 计数 |
| --- | ---: |
| RRW | 35 |
| RWR | 32 |
| WRR | 30 |
| WRW | 28 |
| WWR | 27 |
| RWW | 21 |

entity / eti 模式见 artifact；entity/eti **real_flip_mismatch_any_run = 0**（strict mismatch 口径下仍无 overlap）。

---

## 4. 用途

- 审计入口：优先人工看 **132 题 mismatch=0 仍 flip** 的 correlation 样本
- 外部 Tier1-1（temperature 对照）若执行，应 **仅抽本 ledger idx**，而非 loose 261 题

---

## 5. 执行边界

Superpowers CPU 分析；不涉及新推理。
