# T0-3：`num_tokens` 字段取证（Wave 26）

脚本：[`_analysis/compute_t0_3.py`](_analysis/compute_t0_3.py) → [`artifacts/t0_3_num_tokens_audit.json`](artifacts/t0_3_num_tokens_audit.json)。

**Mechanism**：`inference_tsmllm_vllm.py` L331 写入 `input_token_counts[idx]`，字段名却为 `num_tokens`。  
**Intervention（外部）**：双写字段或改名——需改 inference，superpowers **仅取证**。  
**Falsifier**：若 `num_tokens` 表示 output，则 unclosed 样本 ratio ≈1；实测 **≫1**。

---

## 1. 全局

| 指标 | 值 |
| --- | ---: |
| response_len / num_tokens 中位数 | **5.80** |
| p95 | **12.13** |
| wave6 未闭合样本 ratio 最小值 | **6.06** |

typical 样本：response 中位 ~2200 字符，`num_tokens` 中位 ~240（input）→ ratio ~9。

---

## 2. 解读

- `num_tokens` **与 response 长度数量级不符**，与 prompt/input token 量级一致 → 支持「**input 计数误标为 num_tokens**」。
- wave6 未闭合题（response ~5k–6k）：`num_tokens` 仍 ~630–730 → ratio **6–8×**，**非**「input 触顶导致截断」。

---

## 3. correlation run1 样例

| 字段 | 中位 |
| --- | ---: |
| response_len | ~2100 |
| num_tokens | ~451 |
| ratio | ~5.0 |

未闭合焦点题明细见 artifact `by_task_run.*.unclosed_focus_samples`。

---

## 4. 与 wave6 关系

wave6 结论「非 input 截断」在本 artifact 上 **定量复现**；metadata 误导性为 **文档/字段问题**，superpowers 侧已闭环；改字段属 `14` **外部 inference 规格**。

---

## 5. 执行边界

只读 `generated_answer.json`；未改 `inference/`。
