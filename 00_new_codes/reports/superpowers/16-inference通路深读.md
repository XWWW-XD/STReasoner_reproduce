# Inference 通路深读（Wave 16）

**Skills**: explore → verification-before-completion → requesting-code-review  
**Plan**: [`_plans/wave16.md`](_plans/wave16.md) | **Review**: [`_reviews/wave16-review.md`](_reviews/wave16-review.md)  
**Artifact**: [`inference_trace.json`](artifacts/inference_trace.json)

---

## 1. 入口与 CLI（`inference/inference_tsmllm_vllm.py`）

| 行号 | 机制 |
| ---: | --- |
| 204–207 | `--max_tokens` CLI，**默认 512**（调试/非 ST-Test 主路径易误用） |
| 312 | `SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)` — **ST-Test 6144 由此传入** |
| 315–322 | `answer_question_list(..., sampling_params=sampling_params)` — params 进入 worker 队列 |
| 324–333 | 每题写 `response` + **`num_tokens`: `input_token_counts[idx]`**（见 §2） |
| 335–337 | 输出 `generated_answer.json` — evaluate 默认读取 |

**Input token 计数**（L300–306）：`text_tokens + ceil(ts_len/patch_size)`，与 wave6「非 input 截断」一致 — 字段名误导，非生成长度。

---

## 2. Worker 与 SamplingParams 传递（`inference/llm_utils.py`）

| 行号 | 机制 |
| ---: | --- |
| 156–162 | `worker_vllm_ts` 内 **默认** `SamplingParams(..., max_tokens=CTX_LENGTH, stop_token_ids=[151643,151645], ...)` |
| 186–189 | **若 batch 末位为 `SamplingParams`** → `request_sampling_params = batch_args[0][-1]`，**覆盖 worker 默认** |
| 189 | `llm.generate(batch_inputs, request_sampling_params, ...)` |

**结论**：report 09 修复点成立 — 主路径 `inference_tsmllm_vllm.py:312` 的 6144 **会传到 worker**；仅绕开 `answer_question_list` 的脚本才 stuck 在 `CTX_LENGTH` 默认。

**Stop 行为**：stop_token_ids 针对 chat 特殊 token；**不**阻止 thinking 段内数值复读至 `max_tokens`（wave6–7：~6k 字符、无 `</answer>`）。

---

## 3. Prompt 与格式（`inference/prompt.json`）

- Chat 模板要求 `` + `<answer>` 闭合（与 evaluate `_extract_tag_content` 对齐）。
- 生成未闭合 → evaluate 无 tag 时 **整段 response 进入 `_normalize_choice`**（见 wave17）。

---

## 4. 与阶段一/二 artifact 对照

| 现象 | 代码锚点 |
| --- | --- |
| `num_tokens` 与 output 长度无关 | `inference_tsmllm_vllm.py:331` |
| 6144 触顶无 answer | `max_tokens=args.max_tokens` + stop 未截断复读 |
| T0-3 ratio median 5.8× | input 字段 vs `len(response)` — 见 `25-T0-3-num_tokens字段取证.md` |

---

## 5. Falsifier

若 ST-Test run 使用 `max_tokens=512` CLI 默认且未覆盖 → corr flip/acc 应与 512 exp 趋同；现有 6144 run.log 显示 **6144**，故主结论不变。

**Intervention 锚点**（→ `14` §代码级）：T1-2 改 L312 `max_tokens` 或 stop；E1 在 L326–332 增 `output_token_count` / `finish_reason`。
