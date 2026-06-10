# T0-2：parse 分层指标与 flip 归因（Wave 24）

脚本：[`_analysis/compute_t0_2.py`](_analysis/compute_t0_2.py) → [`artifacts/t0_2_parse_dashboard.json`](artifacts/t0_2_parse_dashboard.json)。承接 E2 [`22`](22-E2-strict-reparse全量对照.md)。

**Superpowers 可执行项**（CPU，不改 `evaluation/`）。

---

## 1. 分层 acc 表（correlation run1 示例）

| 指标 | 值 | 分母 |
| --- | ---: | ---: |
| acc_official | 0.8317 | 1592 |
| acc_strict_parse_ok | **0.8570** | 1545 |
| parse_fail_rate | 2.95% | 47 题 |

四任务 12 行全表见 artifact `dashboard[]`。规律：**official ≈ loose**；strict parse_ok 分母更小、acc 更高（correct 数不变）。

---

## 2. flip 归因（三 run 6144）

| task | loose flip | parser 诱发 | 占比 | 真实不稳定（strict flip） |
| --- | ---: | ---: | ---: | ---: |
| correlation | 261 | **88** | 33.7% | **173** |
| entity | 357 | **24** | 6.7% | **333** |
| etiological | 27 | 0 | 0% | **27** |

**解读**：

- correlation 约 **1/3 flip** 可在 strict parse 下消失 → 报告 flip 率应并列 loose / strict 两列。
- entity 多数 flip 在 parse_ok 子集仍存在 → **采样/推理不稳定** 仍是主因。
- etiological 无 parse_fail → flip 与 parser 无关。

parser 诱发样例 idx（correlation）：16, 46, 78, 98, …（artifact `parser_induced_indices_sample`）。

---

## 3. parse_fail × wave6 未闭合 thinking

| task | unclosed 焦点题 | 任 run parse_fail | 焦点 ⊆ parse_fail |
| --- | ---: | ---: | --- |
| correlation | 12 | 108 | **是**（12/12） |
| entity | 1 | 33 | **是**（1/1） |

T0-2 falsifier：**parse_fail 与 wave6 未闭合样本高度重叠**——未闭合 thinking 是 parse_fail 主因之一。

---

## 4. 执行边界

本项为 superpowers CPU 分析；Wave25 见 [`24`](24-真实不稳定flip账本.md)。`14` 中 Tier1+ 仍为**外部规格**。
