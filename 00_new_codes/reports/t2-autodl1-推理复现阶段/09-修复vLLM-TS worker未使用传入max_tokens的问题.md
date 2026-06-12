日期：2026-05-29

### 修改文件：

```text
inference/llm_utils.py
```

### 修改原因：

复查代码时发现：

- `inference/inference_tsmllm_vllm.py` 有 `--max_tokens` 参数，并会构造 `SamplingParams(max_tokens=args.max_tokens, ...)`。
- 但 `LLMClient` 把这个 `SamplingParams` 放进队列后，`worker_vllm_ts()` 实际调用 `llm.generate()` 时仍使用 worker 内部默认的 `sampling_params`。
- 因此 CLI 参数没有真正生效。之前 `max_tokens=512` 的 ST-Test 运行只能视为链路预跑，不能作为正式结果。

### 修改内容：

- 在 `worker_vllm()` 和 `worker_vllm_ts()` 中检查队列参数最后一项是否为 `SamplingParams`。
- 如果存在调用方传入的 `SamplingParams`，则用它调用 `llm.generate()`。
- 如果不存在，仍回退到 worker 内部默认参数，保持原有兼容性。

### 修改前后差异：


| 项目                 | 修改前                   | 修改后                              |
| ------------------ | --------------------- | -------------------------------- |
| CLI `--max_tokens` | 被放入队列，但 worker 生成时未使用 | worker 生成时使用传入的 `SamplingParams` |
| ST-Test 正式运行       | 不能证明用了指定 token 上限     | 可按 `--max_tokens 6144` 运行        |
| prompt / gold / 数据 | 未修改                   | 未修改                              |


