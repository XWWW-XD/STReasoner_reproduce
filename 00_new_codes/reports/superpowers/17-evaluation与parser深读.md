# Evaluation 与 parser 深读（Wave 17）

**Skills**: systematic-debugging → requesting-code-review  
**Plan**: [`_plans/wave17.md`](_plans/wave17.md) | **Review**: [`_reviews/wave17-review.md`](_reviews/wave17-review.md)  
**Artifact**: [`evaluation_trace.json`](artifacts/evaluation_trace.json)

---

## 1. 预测加载（`evaluation/evaluate_qa.py`）

| 行号 | 机制 |
| ---: | --- |
| 104–143 | `load_prediction_files`：扫 `*{pattern}*.json`，取 `response` 或 COT `responses[0]` |
| 144+ | 按 `idx` 建 dict — **不**读 thinking 与 gold 对齐窗 |

无 tag 的 raw json 仍被完整载入 → 下游 parser 决定 acc。

---

## 2. 无 tag 回退（核心）

```39:48:evaluation/evaluate_qa.py
def _extract_tag_content(text: str, tag: str = "answer") -> str:
    ...
    if match:
        ...
        return answer
    return text.strip()  # 无 <answer> 时整段 response
```

```25:35:evaluation/evaluate_qa.py
def _normalize_choice(text: Any) -> str:
    ...
    value = _extract_tag_content(value, "answer")
    match = re.match(r"\s*([A-Da-d])[\.\)\s-]*", value)
```

**机制链**：unclosed thinking 含大量数字与偶发 `A`/`B` 字符 → loose 口径可「误匹配」→ wave4 **假三不同**（67 loose vs 18 strict triple）。

---

## 3. Forecasting parser

| 行号 | 机制 |
| ---: | --- |
| 51–86 | `_parse_series`：JSON → `"predictions"` 键 → 末个 `[...]` 数字表 |
| 与 report 26 | 本地 smarttest **第二套** strict 扫描 ≠ 本文件全文逻辑 — **禁止混报** |

Kaggle wave12：**0** answer tag → choice 任务走整段回退或失败；forecasting 依赖 `_parse_series` 启发式。

---

## 4. 入口路由（`evaluation/evaluate.py`）

- Task 路由至 `evaluate_qa` 各函数 — ST-Test 四任务 acc/MAE 均经 **tag-first choice** 或 `_parse_series`。
- **Evaluate 只评 `<answer>` 内（choice）或解析出的序列（forecasting）** — thinking 改文案不直接改分，除非改变最终 tag 内容。

---

## 5. 与 flip/mismatch 解耦

- Strict mismatch（阶段一）测 thinking 与 raw TS — **evaluate 路径不读 mismatch**。
- ~70% flip 样本 strict mismatch=0（`flip_mismatch_cross.json`）与 **L25–35 只看 answer 层** 相容。

---

## 6. Intervention 锚点

| Tier | 改哪里 |
| --- | --- |
| T0-1 | 同一 json：official vs strict-missing vs 末段 A–D 诊断 |
| E2 | 已完成 — strict reparse 67→18 triple |
| T2-5 | 对齐 GitHub 官方 `_parse_series` 全文 offline 重评 |

**Falsifier**：若 strict-missing 与 official acc 差 <0.5pp → 通道3非主因（E2 已显示 substantial 差，通道3 **成立**）。
