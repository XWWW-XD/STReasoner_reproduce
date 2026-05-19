# 09 Generation 逻辑

本文档分析 HuggingFace 路径下 `Qwen3TSGenerationMixin`、`Qwen3TSForCausalLM.forward()`、`past_key_values`、`cache_position` 与扩展后 `attention_mask` 的关系。实际仓库推理脚本主要走 vLLM 的 `llm.generate()`，但 HF 路径的 generation 代码展示了 STReasoner 为 time series embedding 做的关键适配。

术语约定：

- `已从代码确认`：能直接绑定到本仓库文件、类、函数或配置行。
- `根据代码推断，未由真实数据验证`：由静态代码串联得到，但没有真实 `generate()` 或真实 batch 验证。
- `尚未确认`：证据不足，已写入 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 普通 LLM generation 流程

普通 causal LM 的自回归 generation 可以概括为：

```text
prompt input_ids
  -> first forward
  -> logits for next token
  -> choose next token
  -> append next token
  -> subsequent forward only processes new token with past_key_values cache
  -> repeat until stop condition
```

`根据代码推断，未由真实数据验证`：在 HuggingFace 风格的 causal LM 中，`prepare_inputs_for_generation()` 通常负责根据 `past_key_values` 裁剪本轮要送入模型的 `input_ids`，并维护 `attention_mask`、`cache_position`、`inputs_embeds` 等参数；`_update_model_kwargs_for_generation()` 通常负责把本轮 forward 输出中的 cache 和 mask 写回下一轮 `model_kwargs`。本仓库没有改写完整 HuggingFace `GenerationMixin`，而是在 `Qwen3TSGenerationMixin` 中调用 `super()` 复用这些标准逻辑，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:264-305`。

`已从代码确认`：普通 Qwen3 decoder 主体由 `Qwen3TSForCausalLM.forward()` 调用 `self.model(...)` 执行，`past_key_values`、`attention_mask`、`position_ids`、`cache_position` 都原样传给底层 `Qwen3Model`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。模型再通过 `lm_head` 把 hidden states 转为 next-token logits，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:607-614`。

普通文本 causal LM 的关键前提是：首轮处理完整 prompt，后续每轮只处理新 token；已经处理过的 prompt 信息通过 `past_key_values` 缓存复用。STReasoner 的难点在于首轮 prompt 会被 TS patch embedding 扩展，不能让后续每个 decoding step 都重新注入同一批 TS embedding。

## 2. STReasoner 的改动点

`已从代码确认`：STReasoner 在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py` 中新增 `Qwen3TSGenerationMixin`，位置是 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:221-347`。这个 mixin 改写了四个 generation 相关方法：

| 方法 | 位置 | 作用 |
|---|---|---|
| `prepare_inputs_for_generation()` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-277` | 在 generation 每一步准备模型输入，并控制 `timeseries` 只在首轮使用 |
| `_update_model_kwargs_for_generation()` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:279-305` | 把扩展后的 `attention_mask` 写回下一轮 generation kwargs |
| `generate()` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:307-332` | 允许调用方把 `timeseries` 作为参数传给 `generate()` |
| `_validate_model_kwargs()` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:334-346` | 临时移除并恢复 `timeseries`，让父类校验 kwargs 时不会把它当成非法参数 |

`已从代码确认`：`Qwen3TSForCausalLM` 继承了 `Qwen3TSPreTrainedModel` 和 `Qwen3TSGenerationMixin`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:348-352`。因此这个模型的 HF `generate()` 会使用这些 TS-aware generation 逻辑。

STReasoner 要改 generation 的原因是：

1. `已从代码确认`：`forward()` 首轮会把 `timeseries` 编码成 `ts_features`，再通过 `_merge_input_ids_with_time_series_features()` 扩展 `inputs_embeds`、`attention_mask`、`position_ids`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-589`。
2. `已从代码确认`：扩展后的 `attention_mask` 会被返回到 `Qwen3TSCausalLMOutputWithPast`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-625`。
3. `已从代码确认`：后续 decoding 需要使用已经扩展过的 `attention_mask`，所以 `_update_model_kwargs_for_generation()` 会在输出中发现 `outputs.attention_mask` 时写回 `model_kwargs["attention_mask"]`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:292-305`。

`已从代码确认`：实际 vLLM 推理脚本不是直接调用 HF `Qwen3TSForCausalLM.generate()`。`inference/llm_utils.py` 中 vLLM worker 调用的是 `llm.generate(batch_inputs, sampling_params, use_tqdm=False)`，见 `inference/llm_utils.py:158-177`；vLLM 模型通过自定义 multimodal processor 和 `merge_multimodal_embeddings()` 合并 TS embedding，见 `inference/vllm/chatts_vllm.py:699-709`。因此本文的 HF generation 逻辑主要用于理解模型代码和非 vLLM 路径。

## 3. First pass 与后续 decoding

### First forward pass

`已从代码确认`：首轮如果传入 `timeseries` 且 `timeseries.shape[0] > 0`，`forward()` 会执行 TS 编码和 embedding 合并：

```python
inputs_embeds = self.get_input_embeddings()(input_ids)
ts_features, patch_cnt = self.ts_encoder(timeseries)
inputs_embeds, attention_mask, position_ids, labels, new_token_positions = \
    self._merge_input_ids_with_time_series_features(...)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-589`。

这一步之后：

- `input_ids` 仍是原始 token id 序列，但底层 `Qwen3Model` 实际接收的是合并后的 `inputs_embeds`。
- `attention_mask` 已经扩展到包含 TS patch embedding 的长度。
- `position_ids` 根据扩展后的 `final_attention_mask` 重新计算。
- `past_key_values` 会由底层 `Qwen3Model` 返回，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-620`。

### 后续 decoding step

`已从代码确认`：`prepare_inputs_for_generation()` 的注释明确写道：time series 只在 first forward pass 处理，后续 generation steps 中它们已经嵌入到 `past_key_values` 里，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:239-244`。

核心逻辑是：

```python
has_ts = timeseries is not None and len(timeseries) > 0

if has_ts and past_key_values is not None:
    ...
    if past_length > 0:
        input_ids = input_ids[:, -1:]
        timeseries = None
        has_ts = False
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:245-263`。

因此，`已从代码确认`：只要 `past_key_values` 已存在且 `past_length > 0`，后续 decoding 只保留最后一个 token 作为本步输入，并把 `timeseries` 清空，避免重复编码和重复合并。

`根据代码推断，未由真实数据验证`：首轮之后，TS patch embedding 对 attention 的影响保存在 cache 中；后续 token 通过 `past_key_values` attend 到之前的文本和 TS patch 表示，而不是每一步重新把 TS 张量送进 `ts_encoder`。

## 4. Cache 相关逻辑

### past_key_values 如何判断是否已经处理过 prompt

`已从代码确认`：`prepare_inputs_for_generation()` 支持两种 cache 类型：

```python
if isinstance(past_key_values, Cache):
    past_length = past_key_values.seen_tokens
else:
    past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:249-255`。

含义是：

- 如果使用 Transformers 的 `Cache` 对象，就用 `seen_tokens` 判断已经处理的 token 数。
- 如果使用传统 tuple cache，就取第一层 key tensor 的序列长度维度 `shape[2]`。
- 只要 `past_length > 0`，代码认为首轮 prompt 已经被处理过，TS embedding 已经进入 cache。

`已从代码确认`：forward 接收 `past_key_values`，并把它传给底层 `self.model(...)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:522` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。forward 返回时又把 `outputs.past_key_values` 放回 `Qwen3TSCausalLMOutputWithPast`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-620`。

### cache_position 如何处理

`已从代码确认`：`prepare_inputs_for_generation()` 接收 `cache_position` 并传给父类：

```python
model_inputs = super().prepare_inputs_for_generation(
    input_ids=input_ids,
    past_key_values=past_key_values,
    attention_mask=attention_mask,
    inputs_embeds=inputs_embeds,
    cache_position=cache_position,
    **kwargs
)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-272`。

`已从代码确认`：`forward()` 也接收 `cache_position`，并传给底层 `self.model(...)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:528` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。

`根据代码推断，未由真实数据验证`：STReasoner 没有在自己的 mixin 中手动重算 `cache_position`，而是依赖父类 `GenerationMixin.prepare_inputs_for_generation()` 和底层 Qwen3Model 处理 cache position。STReasoner 自己手动重算的是 `position_ids`，位置在 `_merge_input_ids_with_time_series_features()`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:498-501`。

### attention_mask / position_ids 如何更新

`已从代码确认`：首轮 merge 中，`_merge_input_ids_with_time_series_features()` 创建 `final_attention_mask`，再用它计算 `position_ids`：

```python
position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:449-460` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:498-501`。

`已从代码确认`：`forward()` 把扩展后的 `attention_mask` 和 `position_ids` 传给底层 `Qwen3Model`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。

`已从代码确认`：`Qwen3TSCausalLMOutputWithPast` 比标准输出额外保留 `attention_mask`、`labels` 和 `new_token_positions`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:182-209`。

`已从代码确认`：generation 更新阶段，`_update_model_kwargs_for_generation()` 如果发现 `outputs.attention_mask` 非空，就先写入 `model_kwargs["attention_mask"]`，再调用父类更新逻辑：

```python
if hasattr(outputs, "attention_mask") and outputs.attention_mask is not None:
    model_kwargs["attention_mask"] = outputs.attention_mask

model_kwargs = super()._update_model_kwargs_for_generation(...)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:279-305`。

`根据代码推断，未由真实数据验证`：这个顺序的意图是让父类在后续 decoding 时基于“已经包含 TS patch 的 attention mask”继续追加新 token 的 mask，而不是基于原始短 prompt mask 继续更新。

## 5. 潜在 bug

### 5.1 重复注入 TS embedding

如果后续 decoding step 没有清空 `timeseries`，每一步都可能重新执行 `self.ts_encoder(timeseries)` 和 `_merge_input_ids_with_time_series_features()`。这样会导致：

- TS patch embedding 被重复插入。
- `attention_mask` 和 `position_ids` 长度不断异常增长。
- `past_key_values` 中的序列长度和本轮输入长度对不上。

`已从代码确认`：当前 HF 代码通过 `past_length > 0` 后设置 `timeseries = None` 避免这个问题，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:248-263`。

### 5.2 attention_mask 没有扩展或没有写回

如果 merge 后没有把扩展后的 `attention_mask` 写回 generation kwargs，后续 token 可能只看到原始文本长度的 mask，无法正确覆盖 TS patch embedding 位置。

`已从代码确认`：当前代码在输出对象中返回扩展后的 `attention_mask`，并在 `_update_model_kwargs_for_generation()` 中写回 `model_kwargs["attention_mask"]`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-625` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:279-305`。

`尚未确认`：没有通过真实 `generate()` 验证父类 `_update_model_kwargs_for_generation()` 在写回扩展 mask 后是否对所有 cache 实现都按预期追加新 token mask。

### 5.3 position_ids / cache_position 不一致

TS patch embedding 会让首轮实际输入序列比原始 `input_ids` 更长。如果 `position_ids` 或 `cache_position` 仍按原始短序列理解，就可能导致：

- RoPE/位置编码错位。
- cache 中 token 位置和下一步 token 位置不连续。
- 生成时注意力对齐错误，表现为输出混乱或运行时报 shape mismatch。

`已从代码确认`：当前代码在 merge 时重算 `position_ids`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:498-501`；`cache_position` 则传给父类和底层 Qwen3Model 处理，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-272` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。

`尚未确认`：没有实际运行验证 `cache_position` 在首轮 TS 扩展后是否和不同 Transformers 版本的 cache 实现完全兼容。

### 5.4 `inputs_embeds` 分支绕过 TS merge

`已从代码确认`：`forward()` 只有在 `inputs_embeds is None` 时才会从 `input_ids` 计算文本 embedding，并进入 TS merge 分支，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-589`。

因此，如果调用方直接传入 `inputs_embeds`，同时又传入 `timeseries`，当前代码不会再执行 `self.ts_encoder(timeseries)`。这可能是有意设计，也可能是需要调用方保证 `inputs_embeds` 已经包含 TS embedding。

`尚未确认`：当前没有发现仓库中 HF generation 路径直接传入自定义 `inputs_embeds` 的调用；如果后续复现中使用这个分支，需要单独验证。

### 5.5 vLLM 路径和 HF GenerationMixin 差异

`已从代码确认`：官方推理脚本使用 vLLM worker，调用 `llm.generate(batch_inputs, sampling_params, use_tqdm=False)`，见 `inference/llm_utils.py:158-177`。这条路径走 `inference/vllm/chatts_vllm.py` 的 multimodal processor 和 `merge_multimodal_embeddings()`，不直接走 HF `Qwen3TSGenerationMixin`。

风险是：如果只阅读 HF generation 逻辑，会误以为官方推理一定使用 `prepare_inputs_for_generation()`。实际上，当前 inference 主链路的 cache 和 decoding 管理由 vLLM 接管。

`尚未确认`：本文没有展开 vLLM 内部 decoding/cache 实现，也没有验证 vLLM 中 TS embedding 是否只在 prefill 阶段注入。

## 6. 组会讲法

### 本文档核心结论

1. `已从代码确认`：STReasoner 的 HF 模型通过 `Qwen3TSGenerationMixin` 改写 generation 准备和更新逻辑，核心位置是 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:221-347`。
2. `已从代码确认`：time series 在 first forward pass 中被编码并合并进 `inputs_embeds`；后续如果 `past_key_values` 已存在且 `past_length > 0`，代码会把 `timeseries` 清空，避免重复注入。
3. `已从代码确认`：首轮 merge 后扩展的 `attention_mask` 会被放进输出对象，并在 `_update_model_kwargs_for_generation()` 中写回下一轮 kwargs。
4. `已从代码确认`：`cache_position` 没有由 STReasoner 手动重算，而是传给父类 generation 逻辑和底层 Qwen3Model 处理。

### 组会可讲版本

普通 LLM 生成时，第一步处理完整 prompt，之后每次只处理新生成的一个 token，前面的上下文靠 `past_key_values` 缓存。STReasoner 的特殊点是：第一步不仅有文本 prompt，还要把时间序列 patch embedding 插入到 `<ts><ts/>` 的位置。这个插入会改变真实输入序列长度，所以 attention mask 和 position ids 也要跟着扩展。

STReasoner 的处理方式是：first pass 时正常编码 timeseries 并合并 embedding；一旦 `past_key_values` 显示 prompt 已经处理过，后续 decoding 就把 `timeseries` 设为 `None`，只保留最后一个 token 继续生成。这样 TS 信息通过 cache 保留下来，不会每一步重复插入。

可以用下面的流程图讲：

```text
first pass:
text prompt + timeseries
  -> TS encoder
  -> merge text embedding + TS embedding
  -> Qwen3 forward
  -> past_key_values + expanded attention_mask

next decoding steps:
new token + past_key_values
  -> timeseries = None
  -> Qwen3 forward with cache
  -> next token
```

### 后续需要验证的问题

1. `尚未确认`：没有通过真实 HF `generate()` 小样本验证 `timeseries` 是否确实只在 first forward pass 使用。
2. `尚未确认`：没有验证扩展后的 `attention_mask` 经父类 `_update_model_kwargs_for_generation()` 后，在所有 cache 实现上都能正确追加新 token。
3. `尚未确认`：没有验证 `cache_position` 在首轮 TS 扩展后是否和当前 Transformers 版本完全兼容。
4. `尚未确认`：官方 vLLM 推理路径的 prefill/decode cache 行为没有在本文中展开源码验证。
