# Stage 2.2 paper_cases 重试与修复报告

日期：2026-05-29

## 结论

结果：
- 4条样例均跑通
- 原始效果不好，主要有以下问题：
- 脚本写的 `run-all` 每条样例都会重新加载一次 STReasoner-8B

## 实验与输出路径

数据集：paper_cases 4 samples

### 修复后实验效果

在源代码基础上，修复过reuse问题、parser逻辑问题

| index | task | len(response) | parsed_answer | gold_answer |
|---:|---|---:|---|---|
| 0 | etiological | 1183 | D | `<answer>D</answer>` |
| 1 | entity | 806 | C | `<answer>C</answer>` |
| 2 | correlation | 1354 | D | `<answer>D</answer>` |
| 3 | forecasting | 6031 | `[20.02, 20.13, 20.23]` | `[19.86, 19.97, 20.05]` |

```text
generate_success_count = 4 / 4
parse_success_count    = 4 / 4
choice_accuracy        = 3 / 3 = 100%
forecasting_mae        = 0.1667
forecasting_mape       = 0.8349%
```


## 具体修改

### 修复 parser / evaluate 提取逻辑

修改文件：

```text
evaluation/evaluate_qa.py
```

修改原因：

模型输出里其实包含正确选择题答案，但旧 parser 没读到：

- etiological 末尾有 `Answer: D`，旧 parser 把整段文本当答案，parse failed。
- entity 末尾有 `\boxed{C}`，旧 parser 因为回答开头是 `Answer:`，误读成 `A`。
- correlation 末尾有 `Answer: D`，旧 parser 同样误读成 `A`。
- forecasting 输出里有最终 JSON `{"predictions": [20.02, 20.13, 20.23]}`，旧 parser 把全文所有数字都抽出来，导致 MAE 被严重放大。

修改内容：

- `_normalize_choice()` 新增：
  - 优先读取 `\boxed{C}` 这类最终答案。
  - 读取 `Answer: D` / `Final Answer: D`。
  - 读取末尾带判断词的 `Option C ... most/correct/best/...`。
  - 最后才回退到旧的开头字母匹配。
- `_parse_series()` 新增：
  - 优先读取 JSON 字段里的 `"predictions": [...]` / `"forecast": [...]` / `"answer": [...]`。
  - 其次读取最后一个短数字数组。
  - 最后才回退到旧的全文数字抽取。

## 修复 vLLM-TS worker 未使用传入 max_tokens 的问题

修改文件：

```text
inference/llm_utils.py
```

修改原因：

用户明确要求正式实验 `max_new_tokens / max_tokens` 必须是 `6144`。复查代码时发现：

- `inference/inference_tsmllm_vllm.py` 有 `--max_tokens` 参数，并会构造 `SamplingParams(max_tokens=args.max_tokens, ...)`。
- 但 `LLMClient` 把这个 `SamplingParams` 放进队列后，`worker_vllm_ts()` 实际调用 `llm.generate()` 时仍使用 worker 内部默认的 `sampling_params`。
- 因此 CLI 参数没有真正生效。之前 `max_tokens=512` 的 ST-Test 运行只能视为链路预跑，不能作为正式结果。

修改内容：

- 在 `worker_vllm()` 和 `worker_vllm_ts()` 中检查队列参数最后一项是否为 `SamplingParams`。
- 如果存在调用方传入的 `SamplingParams`，则用它调用 `llm.generate()`。
- 如果不存在，仍回退到 worker 内部默认参数，保持原有兼容性。