# 08 Time Series Embedding 合并机制

本文档专门阅读 `Qwen3TSForCausalLM._merge_input_ids_with_time_series_features()`。主线以 HuggingFace 模型路径为准，因为该函数完整展示了文本 embedding 与 time series patch embedding 的重排逻辑；同时补充 vLLM 推理路径中 `merge_multimodal_embeddings()` 的对应机制。

术语约定：

- `已从代码确认`：能直接绑定到本仓库文件、类、函数或配置行。
- `根据代码推断，未由真实数据验证`：由静态代码串联得到，但没有真实 forward 或真实 batch 验证。
- `尚未确认`：证据不足，已写入 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 函数位置

`已从代码确认`：HF 路径的 embedding 合并函数是：

```python
Qwen3TSForCausalLM._merge_input_ids_with_time_series_features(
    self,
    time_series_features,
    inputs_embeds,
    input_ids,
    attention_mask,
    labels,
    patch_cnt,
)
```

定义位置是 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`。

`已从代码确认`：该函数由 `Qwen3TSForCausalLM.forward()` 调用。forward 先用 `self.get_input_embeddings()(input_ids)` 得到文本 embedding；如果 `timeseries is not None and timeseries.shape[0] > 0`，则调用 `self.ts_encoder(timeseries)` 得到 `ts_features, patch_cnt`，再进入 merge 函数，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-589`。

`已从代码确认`：vLLM 推理路径不走这个 HF merge 函数。vLLM 路径中，prompt replacement 在 `inference/vllm/chatts_vllm.py:352-401`，模型侧合并调用 `merge_multimodal_embeddings()`，见 `inference/vllm/chatts_vllm.py:699-709`。因此本文把 HF 函数作为主线，把 vLLM 作为实际推理路径的补充说明。

## 2. 输入输出

### 输入

`已从代码确认`：merge 函数输入来自 `forward()` 中的三条流：

| 输入 | 形状 / 内容 | 来源 | 作用 |
|---|---|---|---|
| `time_series_features` | `[sum(patch_cnt), embed_dim]` | `self.ts_encoder(timeseries)` | 所有 TS patch 的 embedding，按样本和节点顺序扁平拼接 |
| `inputs_embeds` | `[batch_size, sequence_length, embed_dim]` | `self.get_input_embeddings()(input_ids)` | 原始文本 token embedding，其中包括 `<ts>` 和 `<ts/>` token 的普通 embedding |
| `input_ids` | `[batch_size, sequence_length]` | tokenizer 输出 | 用来定位 `<ts>` / `<ts/>` 特殊 token |
| `attention_mask` | `[batch_size, sequence_length]` | tokenizer padding 结果 | 标记原始有效文本 token 和 pad token |
| `labels` | `[batch_size, sequence_length]` 或 `None` | 训练时标签 | merge 时和文本 token 一起移动；TS 位置保持 ignore |
| `patch_cnt` | `[num_timeseries_total]` | `TimeSeriesEmbedding.forward()` | 每条 time series 对应多少个 TS patch embedding |

对应代码：`forward()` 中得到 `inputs_embeds`、`ts_features` 和 `patch_cnt` 的位置在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-588`；`TimeSeriesEmbedding.forward()` 返回 `x, patch_cnt`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:171-179`。

### 输出

`已从代码确认`：merge 函数返回五个对象，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:513`：

```python
return final_embedding, final_attention_mask, position_ids, final_labels, new_token_positions
```

| 输出 | 作用 | 后续流向 |
|---|---|---|
| `final_embedding` | 合并文本 embedding 和 TS patch embedding 后的新 `inputs_embeds` | 传给底层 `self.model(...)` |
| `final_attention_mask` | 扩展后的 attention mask | 传给底层 `self.model(...)`，并在 generation 中继续更新 |
| `position_ids` | 基于扩展 mask 重新计算的位置 id | 传给底层 `self.model(...)` |
| `final_labels` | 扩展后的 labels，TS 位置默认为 `ignore_index` | 训练时用于 loss |
| `new_token_positions` | 原始 token 在扩展序列中的新位置，padding 位置设为 -1 | 返回到 `Qwen3TSCausalLMOutputWithPast` |

`已从代码确认`：合并后的 `inputs_embeds`、`attention_mask`、`position_ids` 会进入底层 `Qwen3Model`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`。返回对象中也保存了扩展后的 `attention_mask`、`labels`、`new_token_positions`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-625`。

## 3. 合并算法逐步解释

### Step 1：判断 padding 方向

`已从代码确认`：函数先检查 batch 的 `attention_mask` 左端和右端是否有 0：

```python
_left_padding = torch.any(attention_mask[:, 0] == 0)
_right_padding = torch.any(attention_mask[:, -1] == 0)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-405`。

如果 batch size 大于 1 且左右两侧都出现 padding，函数直接抛错：

```python
raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:392-400`。

`已从代码确认`：这个判断决定后续扩展序列时，新增位置是在左 padding 场景下右对齐，还是默认从左到右填充，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:442-459`。

### Step 2：定位 `<ts>` 和 `<ts/>`

`已从代码确认`：函数用两个特殊 token id 找 time series 占位符：

```python
special_ts_token_mask_start = input_ids == self.config.ts_token_start_index
special_ts_token_mask_end = input_ids == self.config.ts_token_end_index
special_ts_token_mask = special_ts_token_mask_start | special_ts_token_mask_end
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:407-410`。

其中 `ts_token_start_index=151669`，`ts_token_end_index=151670`，见 `base_model/Config-Qwen3-8B/config.json:85-86`；对应 token 是 `<ts>` 和 `<ts/>`，见 `base_model/Config-Qwen3-8B/added_tokens.json:8-9`。

`已从代码确认`：函数只用 `<ts>` 统计 time series 的条数：

```python
num_special_ts_tokens = torch.sum(special_ts_token_mask_start, dim=-1)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:412-414`。因此每个 `<ts>` 起始 token 对应一条 time series；`<ts/>` 主要作为结束占位 token，一起从普通文本 token 中排除。

### Step 3：按样本切分 `patch_cnt`

`已从代码确认`：`time_series_features` 是所有 TS patch 的扁平拼接，`patch_cnt` 是每条 TS 的 patch 数。函数用 `patch_index` 按 batch 样本顺序消耗 `patch_cnt`：

```python
patch_index = 0
for i in range(batch_size):
    num_ts_in_batch = num_special_ts_tokens[i]
    num_total_patches[i] = patch_cnt[patch_index : patch_index + num_ts_in_batch].sum() - 2 * num_ts_in_batch
    ...
    patch_index += num_ts_in_batch
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:416-430`。

这里的 `- 2 * num_ts_in_batch` 很关键：原始文本里每条 TS 已经占了两个 token，即 `<ts>` 和 `<ts/>`。如果一条 TS 需要 `k` 个 patch embedding，最终序列长度相对原始 token 序列增加 `k - 2`。所以 batch 中每个样本的扩展长度是：

```text
sum(patch_cnt_for_this_sample) - 2 * num_ts_in_sample
```

`根据代码推断，未由真实数据验证`：该函数隐含要求 `patch_cnt` 的顺序和 prompt 中 `<ts>` 出现顺序一致。这个顺序由 processor / data pipeline 保证，但本文没有用真实 batch 运行验证。

### Step 4：计算扩展后长度和文本 token 新位置

`已从代码确认`：扩展后最大长度为：

```python
max_embed_dim = sequence_length + num_total_patches.max()
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:432-433`。

然后函数找出所有非 TS token：

```python
batch_indices, non_ts_indices = torch.where(~special_ts_token_mask)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:435-437`。

`已从代码确认`：文本 token 的新位置由累计和计算：

```python
new_token_positions = torch.cumsum((special_ts_token_mask_start_with_size + 1), dim=-1) - 1
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:439-447`。

其中 `special_ts_token_mask_start_with_size` 对 `<ts>` 位置写入 `patch_cnt - 2`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:419-429`。这让 `<ts><ts/>` 两个占位 token 在最终序列中被替换为对应数量的 TS patch embedding。

### Step 5：创建最终 embedding 和 attention mask

`已从代码确认`：函数创建全 0 的 `final_embedding`：

```python
final_embedding = torch.zeros(
    batch_size, max_embed_dim, embed_dim,
    dtype=inputs_embeds.dtype,
    device=inputs_embeds.device
)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:449-452`。

`已从代码确认`：`final_attention_mask` 也从全 0 开始，然后根据 `attn_mask_cnt` 设置有效位置为 1：

```python
final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
for i in range(attention_mask.size(0)):
    if left_padding:
        final_attention_mask[i, max_embed_dim - attn_mask_cnt[i] :] = 1
    else:
        final_attention_mask[i, : attn_mask_cnt[i]] = 1
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:454-459`。

`已从代码确认`：`attn_mask_cnt` 初始来自原始 `attention_mask.sum(dim=-1)`，再加上每个样本新增的 TS patch 数，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:422-430`。

### Step 6：移动文本 embedding 和 labels

`已从代码确认`：函数把非 TS token 的原始 embedding 移到最终位置：

```python
final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_ts_indices]
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:467-475`。

如果 `labels is not None`，也按同样位置迁移 labels：

```python
final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_ts_indices]
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:461-477`。

`已从代码确认`：`final_labels` 初始化为 `self.config.ignore_index`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:461-465`。因此 TS patch 位置不会直接参与文本 token loss。

### Step 7：把 TS embedding 填入剩余位置

`已从代码确认`：函数先把所有位置设为 TS 候选位置，再把文本 token 已占用的位置排除：

```python
ts_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device)
ts_to_overwrite[batch_indices, text_to_overwrite] = False
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:479-483`。

随后通过 `reversed_cumsum` 排除 padding 对齐造成的多余空位：

```python
reversed_cumsum = ts_to_overwrite.flip(dims=[-1]).cumsum(-1).flip(dims=[-1]) - 1
ts_to_overwrite &= reversed_cumsum >= nb_ts_pad[:, None].to(target_device)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:485-486`。

`已从代码确认`：在真正填入前，函数校验要填的 TS 位置数是否等于 `time_series_features` 数量：

```python
if ts_to_overwrite.sum() != time_series_features.shape[:-1].numel():
    raise ValueError(...)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:488-493`。

通过校验后，TS patch embedding 被写入最终序列：

```python
final_embedding[ts_to_overwrite] = time_series_features.contiguous().reshape(-1, embed_dim).to(target_device)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:494`。

### Step 8：重算 position ids，处理 pad token

`已从代码确认`：函数根据扩展后的 attention mask 重算 `position_ids`：

```python
position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:498-501`。

`已从代码确认`：如果原始 `input_ids` 中有 `pad_token_id`，函数会把这些 pad token 对应的新位置 embedding 置 0：

```python
pad_batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_id)
if len(pad_batch_indices) > 0:
    indices_to_mask = new_token_positions[pad_batch_indices, pad_indices]
    final_embedding[pad_batch_indices, indices_to_mask] = 0
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:503-508`。

最后，原始 padding token 的 `new_token_positions` 被设为 -1：

```python
new_token_positions = new_token_positions.masked_fill(attention_mask == 0, -1)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:510-511`。

## 4. 特殊 Token 作用

`已从代码确认`：STReasoner 使用两个 TS 特殊 token：

| token | id | 配置位置 | 在 merge 中的作用 |
|---|---:|---|---|
| `<ts>` | 151669 | `base_model/Config-Qwen3-8B/config.json:85-86`, `base_model/Config-Qwen3-8B/added_tokens.json:8-9` | 标记一条 time series 的起始位置；函数用它统计每个样本有多少条 TS |
| `<ts/>` | 151670 | `base_model/Config-Qwen3-8B/config.json:85-86`, `base_model/Config-Qwen3-8B/added_tokens.json:8-9` | 标记 time series 占位符结束；在 merge 时和 `<ts>` 一起被排除出普通文本 token |

在 HF merge 中，`<ts>` 和 `<ts/>` 的共同作用是“占位”。合并前它们是普通 token id 和普通 token embedding；合并后它们的位置不再保留原本的 token embedding，而是被替换/扩展为 `patch_cnt` 个 TS patch embedding。

`已从代码确认`：vLLM 路径也使用 `ts_token_start_index` 作为 embedding placeholder，见 `inference/vllm/chatts_vllm.py:352-401`。vLLM 的 prompt replacement target 是 `[placeholder, placeholder + 1]`，也就是 `<ts>` 和 `<ts/>`，见 `inference/vllm/chatts_vllm.py:395-400`。

`已从代码确认`：vLLM 会根据每条 TS 的 patch 数补充 `<ts>` placeholder token：

```python
if num_placeholders < patch_cnt[item_idx]:
    tokens.extend([placeholder] * (patch_cnt[item_idx] - num_placeholders))
```

见 `inference/vllm/chatts_vllm.py:381-389`。随后模型侧用 `merge_multimodal_embeddings(input_ids, inputs_embeds, multimodal_embeddings, self.config.ts_token_start_index)` 合并，见 `inference/vllm/chatts_vllm.py:699-709`。

## 5. 多节点示例

这里用抽象 token，不使用真实数据。

假设一个 prompt 中有两个节点时间序列：

```text
A <ts><ts/> B <ts><ts/> C
```

token 化后可以抽象为：

```text
合并前 input_ids / inputs_embeds:
[A, <ts>, <ts/>, B, <ts>, <ts/>, C]
```

假设 `TimeSeriesEmbedding` 产生：

```text
patch_cnt = [2, 3]

time_series_features:
[
  ts0_0, ts0_1,
  ts1_0, ts1_1, ts1_2
]
```

`已从代码确认`：merge 函数会按 `<ts>` 出现顺序消费 `patch_cnt`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:416-430`。因此第一组 `<ts><ts/>` 对应 `patch_cnt[0]=2`，第二组 `<ts><ts/>` 对应 `patch_cnt[1]=3`。

最终序列可以示意为：

```text
合并后 final_embedding:
[A, ts0_0, ts0_1, B, ts1_0, ts1_1, ts1_2, C]
```

长度变化：

```text
原始长度 = 7
第一条 TS: 2 个 patch 替换 2 个 token，长度变化 0
第二条 TS: 3 个 patch 替换 2 个 token，长度变化 +1
最终长度 = 8
```

如果同一个 batch 中另一个样本更长，`max_embed_dim` 会取 batch 内最大扩展长度；较短样本通过 `final_attention_mask` 区分有效位置和 padding 位置，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:432-459`。

## 6. 潜在风险

### 6.1 TS token 数和 placeholder 不一致

`已从代码确认`：HF merge 中有最终校验：

```python
if ts_to_overwrite.sum() != time_series_features.shape[:-1].numel():
    raise ValueError(...)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:488-493`。如果需要填入的 TS 位置数和 `time_series_features` 数量不一致，函数会抛出 `ValueError`，提示 TS token 数和 time series 数不匹配。

`根据代码推断，未由真实数据验证`：导致不一致的常见原因可能包括：

- prompt 中 `<ts><ts/>` 数量和 `timeseries` 列表长度不一致。
- `patch_cnt` 顺序和 prompt 中 `<ts>` 出现顺序不一致。
- 某条 TS 的 patch 数计算和实际 `time_series_features` 数量不一致。

`已从代码确认`：processor 的非 vLLM 路径也会检查 `<ts><ts/>` 占位符数量和 `timeseries` 数量，数量不一致时抛错，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:122-143`。

### 6.2 attention mask 是否正确

`已从代码确认`：merge 函数会基于原始 `attention_mask.sum(dim=-1)` 加上新增 TS patch 数，生成 `final_attention_mask`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:422-459`。

`已从代码确认`：如果 batch 中同时出现左 padding 和右 padding，函数会认为 mask 非法并抛错，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-405`。

`尚未确认`：当前没有实际 forward 验证所有 batch padding 情况，因此还不能确认复杂 batch 下 `final_attention_mask`、`position_ids` 和 `new_token_positions` 在运行时都和训练/推理期望完全一致。这个问题已记录到 `uncertainty_log.md`。

### 6.3 generation 时是否重复注入 TS embedding

`已从代码确认`：`Qwen3TSGenerationMixin.prepare_inputs_for_generation()` 显式处理了重复注入问题。它判断如果 `timeseries` 存在且 `past_key_values` 已经有内容，并且 `past_length > 0`，就只保留最后一个 `input_ids`，并把 `timeseries = None`：

```python
if past_length > 0:
    input_ids = input_ids[:, -1:]
    timeseries = None
    has_ts = False
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-277`。

`已从代码确认`：`_update_model_kwargs_for_generation()` 还会把模型输出中的扩展后 `attention_mask` 写回 `model_kwargs`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:279-305`。这对应 TS embedding 扩展序列长度后的 generation mask 管理。

`尚未确认`：上述逻辑是静态代码确认，没有通过真实 `generate()` 小样本运行验证。尤其 vLLM 推理路径不走 HF `GenerationMixin`，而是走 vLLM 自己的 multimodal processing 和 cache 机制。

### 6.4 HF 路径和 vLLM 路径差异

`已从代码确认`：HF 路径由 `_merge_input_ids_with_time_series_features()` 手工重排 `inputs_embeds`、`attention_mask`、`position_ids`、`labels`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`。

`已从代码确认`：vLLM 路径先在 `_get_prompt_updates()` 中把 `<ts><ts/>` 替换/扩展为足够数量的 placeholder token，再调用 vLLM 的 `merge_multimodal_embeddings()`，见 `inference/vllm/chatts_vllm.py:352-401` 和 `inference/vllm/chatts_vllm.py:699-709`。

`尚未确认`：本文没有展开 vLLM 内部 `merge_multimodal_embeddings()` 源码，也没有用同一条样本比较 HF 和 vLLM 两条路径的最终 embedding 对齐结果。

## 7. 组会讲法

### 本文档核心结论

1. `已从代码确认`：HF 路径的核心合并函数是 `Qwen3TSForCausalLM._merge_input_ids_with_time_series_features()`，位于 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`。
2. `已从代码确认`：函数把 `<ts>` 和 `<ts/>` 当作占位符，不保留它们原始 token embedding，而是用 `TimeSeriesEmbedding` 产生的 TS patch embedding 替换/扩展这些位置。
3. `已从代码确认`：多个节点时间序列按 prompt 中 `<ts>` 出现顺序，与扁平的 `patch_cnt` 和 `time_series_features` 顺序对应。
4. `已从代码确认`：合并后会重建 `final_embedding`、`final_attention_mask`、`position_ids` 和可选 `final_labels`，再把 `final_embedding` 作为 `inputs_embeds` 送入底层 Qwen3Model。

### 组会可讲版本

可以把 embedding 合并讲成一句话：`<ts><ts/>` 在文本里不是普通内容，而是“给 TS embedding 留位置”的锚点。模型先把整段 prompt token 化，得到文本 embedding；同时把每个节点的时间序列编码成若干个 TS patch embedding。merge 函数扫描 `input_ids` 找到 `<ts>` 和 `<ts/>`，把普通文本 token 移到新位置，再把 TS patch embedding 填进这些占位符形成的空位，最终得到一个混合序列：

```text
合并前:
[文本A, <ts>, <ts/>, 文本B, <ts>, <ts/>, 文本C]

TS features:
[ts0_0, ts0_1, ts1_0, ts1_1, ts1_2]

合并后:
[文本A, ts0_0, ts0_1, 文本B, ts1_0, ts1_1, ts1_2, 文本C]
```

这一步之后，Qwen3 看到的已经不是单纯文本 embedding，而是文本 token embedding 和时间序列 patch embedding 混在同一个 Transformer 序列中。attention mask 和 position ids 也会同步扩展，让 Qwen3 可以在同一上下文里 attend 到问题文本、图结构文本和 TS embedding。

### 后续需要验证的问题

1. `尚未确认`：没有实际 forward 验证复杂 batch 下 `final_attention_mask`、`position_ids`、`new_token_positions` 是否和训练/推理期望完全一致。
2. `尚未确认`：没有用真实样本验证 prompt 中 `<ts><ts/>` 顺序、`timeseries` 列表顺序、`patch_cnt` 顺序和 `time_series_features` 顺序是否在所有数据路径中严格一致。
3. `尚未确认`：没有展开 vLLM 内部 `merge_multimodal_embeddings()` 源码，也没有比较 HF 与 vLLM 两条路径的最终 embedding 对齐结果。
4. `尚未确认`：generation 防重复注入逻辑已从代码确认，但没有通过真实 `generate()` 小样本运行验证。
