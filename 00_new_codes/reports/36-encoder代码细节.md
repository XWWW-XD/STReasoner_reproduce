
## 2. MLP encoder 代码事实

### 2.1 代码位置

| 项目 | 位置 |
| --- | --- |
| MLP encoder 类 | `base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py`，`TimeSeriesEmbedding` |
| encoder forward | `base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py`，`TimeSeriesEmbedding.forward` |
| TS + LLM 合并函数 | `base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py`，`Qwen3TSForCausalLM._merge_input_ids_with_time_series_features` |
| 模型 forward | `base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py`，`Qwen3TSForCausalLM.forward` |
| normalize / prompt reconstruction | `base_model/Qwen3-4B-Instruct-2507/processing_qwen3_ts.py`，`sp_encoding` 与 `Qwen3TSProcessor.__call__` |
| 配置 | `base_model/Qwen3-4B-Instruct-2507/config.json` 的 `ts` 字段 |

### 2.2 config 与 MLP 结构

`config.json` 中 `ts` 字段如下：

```json
{
  "embedding_dim": 16,
  "hidden_size": 2560,
  "max_length": 32768,
  "max_sequence_length": 32768,
  "num_features": 2,
  "num_layers": 5,
  "patch_size": 8,
  "use_layer_norm": false,
  "use_position_embedding": true,
  "use_position_idx": false
}
```

由代码可确认：

- `patch_size = 8`。
- 每个 patch 是 8 个原始数值。
- 开启 `use_position_embedding`，每个时间步拼接 16 维 position embedding。
- 单 patch MLP 输入维度是 `8 + 8 * 16 = 136`。
- MLP 是 5 个 Linear 层，前 4 个 Linear 后接 GELU，最后一个 Linear 输出。
- hidden dim / output dim 都是 `2560`。
- `TimeSeriesEmbedding.mlp` 中没有 dropout。
- `use_layer_norm=false`，没有启用 layer norm。
- 每个 patch 产生 1 个 2560 维 embedding。

### 2.3 raw time series 到 LLM inputs_embeds

流程如下：

1. prompt 中只有 `<ts><ts/>` placeholder，不直接展开原始时间序列文本。
2. `Qwen3TSProcessor.__call__` 调用 `sp_encoding(timeseries)`。
3. `sp_encoding` 做全序列 mean centering：`scaled_timeseries = timeseries - mean`。
4. 如果 centered 后最大绝对值大于等于 3，则除以 `max_abs / 3`。
5. prompt 中只写 metadata：`offset`、`scaling`、`length`、`max`、`min`、`left`、`right`。
6. 数值张量进入 `TimeSeriesEmbedding.forward` 后 reshape 成 `(batch, seq_len, num_features)`，最后一列作为 mask。
7. valid values 按 8 步 patchify，不足一 patch 时用最后一个有效值 padding。
8. 每个 patch 拼接 position embedding 后过 MLP，得到 patch embedding。
9. `Qwen3TSForCausalLM.forward` 调用 `_merge_input_ids_with_time_series_features`，把 `<ts>` placeholder 区间替换成 patch embeddings。
10. 扩展后的 `inputs_embeds` 进入 Qwen3 主干 self-attention。

### 2.4 MLP 是否直接建模 cross-patch / cross-node dependency

不直接建模。

代码依据：

- `TimeSeriesEmbedding.forward` 把每条序列切成 patch。
- 所有 patch 通过 `torch.cat(patches_list, dim=0)` 放到 batch 维。
- 同一个 MLP 独立处理每个 patch。
- encoder 内没有 patch-to-patch attention、RNN、卷积窗口、state-space、图邻接传播或节点间消息传递。

因此，cross-patch / cross-node / delayed dependency 只能在 patch embeddings 插入 LLM 后，由 Qwen3 self-attention 间接处理。
