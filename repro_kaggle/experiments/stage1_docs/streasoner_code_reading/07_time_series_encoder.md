# 07 Time Series Encoder 深读

本文档专门阅读 `TimeSeriesEmbedding`。主证据来自 HuggingFace 模型路径 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py`，并补充 `processing_qwen3_ts.py` 说明 `timeseries` 张量如何被构造。vLLM 和 EasyR1 中也有同名实现，本文只把它们作为推理/RL 侧镜像实现标注，不展开逐行分析。

术语约定：

- `已从代码确认`：能直接绑定到本仓库文件、类、函数或配置行。
- `根据代码推断，未由真实数据验证`：由静态代码串联得到，但没有真实 forward 或真实 batch 验证。
- `尚未确认`：证据不足，已写入 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 类和文件位置

`已从代码确认`：核心类是 `TimeSeriesEmbedding`，定义在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。它在 `Qwen3TSForCausalLM.__init__()` 中被实例化为 `self.ts_encoder = TimeSeriesEmbedding(config.ts)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:357-365`。

`已从代码确认`：`Qwen3TSForCausalLM.forward()` 在存在 `timeseries` 输入时调用 `self.ts_encoder(timeseries)`，随后把输出的 `ts_features` 和 `patch_cnt` 交给 `_merge_input_ids_with_time_series_features()`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。

`已从代码确认`：Qwen3-8B 的 TS encoder 配置位于 `base_model/Config-Qwen3-8B/config.json:73-83`：

| 配置项 | 值 | 含义 |
|---|---:|---|
| `embedding_dim` | 16 | 每个位置索引的 position embedding 维度 |
| `hidden_size` | 4096 | TS patch embedding 输出维度，和 Qwen3 hidden size 对齐 |
| `max_sequence_length` | 32768 | position embedding 支持的最大原始时序长度 |
| `num_features` | 2 | encoder 内部把每个时间点看作 `[value, mask]` 两个特征 |
| `num_layers` | 5 | MLP 线性层数量 |
| `patch_size` | 8 | 每 8 个有效时间点组成一个 patch |
| `use_layer_norm` | false | MLP 末尾不加 LayerNorm |
| `use_position_embedding` | true | patch 输入拼接可学习位置 embedding |
| `use_position_idx` | false | 不使用归一化位置数值作为额外特征 |

`已从代码确认`：vLLM 推理路径中也有一份 `TimeSeriesEmbedding`，见 `inference/vllm/chatts_vllm.py:52-184`；EasyR1/RL 路径也有类似实现，见 `src/EasyR1/verl/utils/chatts_vllm.py:53-193`。它们服务于 vLLM 多模态推理和 RL rollout，不改变本文对主模型文件的结论。

## 2. 输入张量

`已从代码确认`：原始 JSONL 中的 `timeseries` 不是直接送入 `TimeSeriesEmbedding`。它先经过 `Qwen3TSProcessor` 的 `sp_encoding()`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`。

`已从代码确认`：`sp_encoding()` 对原始序列做均值中心化和缩放，然后构造 value/mask 交错结构：

```python
result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)
```

见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:36-50`。

因此，对于常见的一维原始序列长度 `L`：

```text
原始 timeseries:
[x_0, x_1, ..., x_{L-1}]

sp_encoding 后的逻辑内容:
[[scaled_x_0, 1],
 [scaled_x_1, 1],
 ...
 [scaled_x_{L-1}, 1]]

代码实际 reshape 后的外部形态:
[2L, 1]
```

`已从代码确认`：processor 会给每条编码后的序列加 batch 维，形成 `encoded_ts[None, ...]`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:131-135` 和 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:183-186`。如果一个 batch 里多条序列长度不同，processor 会在 axis 1 上用 0 padding 到同一长度，再 `np.concatenate`，最后转成 `torch.float16`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:145-155` 和 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:188-198`。

`已从代码确认`：进入 `TimeSeriesEmbedding.forward()` 后，第一步是：

```python
batch_size = x.size(0)
x = x.reshape(batch_size, -1, self.num_features)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:79-82`。由于配置中 `num_features=2`，见 `base_model/Config-Qwen3-8B/config.json:78`，所以常见一维序列的内部形态会变成：

```text
[num_series, L_or_padded_L, 2]
```

其中最后一维的第 0 个特征是数值，第 1 个特征是 mask。

`根据代码推断，未由真实数据验证`：文档中把输入解释为“一维时间序列 -> value/mask 两特征”。`sp_encoding()` docstring 写原始输入可以是 1D 或 2D，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:28-33`，但当前静态阅读没有验证真实数据中是否存在 2D 原始序列，也没有验证 2D 情况下 reshape 后的语义是否仍是严格的 `[value, mask]` 成对结构。

## 3. Mask 与有效长度

`已从代码确认`：`TimeSeriesEmbedding.forward()` 把最后一个特征作为 mask：

```python
mask = x[:, :, -1].long()
valid_lengths = mask.sum(dim=1).long()
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-86`。

mask 的作用有两层：

1. `已从代码确认`：区分真实时间点和 processor padding 出来的位置。`sp_encoding()` 对真实时间点写入 `1`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:48`；processor 对不同长度序列做 batch padding 时使用 0，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:145-155`。
2. `已从代码确认`：决定每条序列真正参与 patchify 的长度。`valid_lengths = mask.sum(dim=1).long()` 直接按 mask 求和，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-86`。

一个直观例子：

```text
内部 x[i] 形态:
[[v0, 1],
 [v1, 1],
 [v2, 1],
 [0,  0],
 [0,  0]]

mask = [1, 1, 1, 0, 0]
valid_lengths = 3
```

`已从代码确认`：后续取真实数值时只取 `x[i, :vl, :1]`，也就是只拿有效长度内的第 0 个数值特征，排除 mask 特征，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:93-100`。

## 4. Patchify 过程

`已从代码确认`：patch 数量用下面公式计算：

```python
patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-87`。这等价于：

```text
patch_cnt = ceil(valid_lengths / patch_size)
```

`已从代码确认`：当前 Qwen3-8B 配置中 `patch_size=8`，见 `base_model/Config-Qwen3-8B/config.json:80`。

`已从代码确认`：对 batch 中第 `i` 条序列，代码读取：

```python
vl = valid_lengths[i].item()
pc = patch_cnt[i].item()
```

如果 `pc == 0`，直接跳过，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:93-97`。

`已从代码确认`：如果有效长度不能被 `patch_size` 整除，代码会把最后一个真实数值重复若干次，补齐到 `pc * patch_size`：

```python
total_padded_length = pc * self.patch_size
padding_length = total_padded_length - vl
last_value = xi[-1:, :]
padding = last_value.repeat(padding_length, 1)
xi = torch.cat([xi, padding], dim=0)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:99-115`。

补齐后，代码把一维连续数值切成 patch：

```python
xi = xi.reshape(pc, self.patch_size)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:117-119`。因此每个 patch 是连续 `patch_size` 个时间点的数值片段。

`已从代码确认`：当 `use_position_embedding=True` 时，代码同时为每个时间点创建位置索引，真实时间点用 `0..vl-1`，patch 内 padding 位置用 `self.padding_idx = max_sequence_length`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:56-60` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:104-119`。

## 5. MLP 编码

`已从代码确认`：`TimeSeriesEmbedding.__init__()` 先根据配置决定 MLP 输入维度，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-65`。

当前 Qwen3-8B 配置是 `use_position_embedding=True`，所以每个 patch 的输入由两部分拼接：

```text
数值 patch:
patch_size = 8 个 scalar -> 8 维

位置 embedding patch:
每个时间点 position embedding 维度 16
8 个时间点 -> 8 * 16 = 128 维

MLP 输入维度:
8 + 128 = 136
```

对应代码：

- position embedding 定义：`self.position_embedding = nn.Embedding(self.max_sequence_length + 1, self.embedding_dim)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:56-59`
- 输入维度：`input_size = 1 * self.patch_size + self.embedding_dim * self.patch_size`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:56-60`
- `embedding_dim=16`、`patch_size=8`，见 `base_model/Config-Qwen3-8B/config.json:73-83`

`已从代码确认`：forward 中实际拼接方式是先查 position embedding，再把数值 patch 和位置 embedding flatten 后 concat：

```python
batch_pos_emb = self.position_embedding(batch_position_indices)
xi = xi.unsqueeze(-1)
patch_input = torch.cat([
    xi.flatten(1),
    pos_emb.flatten(1)
], dim=1)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:146-169`。

`已从代码确认`：MLP 层结构由 `num_layers` 控制。代码先循环 `num_layers - 1` 次，每次添加 `Linear(input_size, hidden_size)` 和 `GELU()`，再添加最后一个 `Linear(input_size, hidden_size)`；如果 `use_layer_norm=True`，末尾才加 `LayerNorm`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:66-77`。

当前配置是：

```text
num_layers = 5
hidden_size = 4096
use_layer_norm = false
```

因此 MLP 可以概括为：

```text
Linear(136 -> 4096) + GELU
Linear(4096 -> 4096) + GELU
Linear(4096 -> 4096) + GELU
Linear(4096 -> 4096) + GELU
Linear(4096 -> 4096)
```

`已从代码确认`：如果配置改成 `use_position_idx=True`，输入维度会是 `2 * patch_size`；如果两个位置相关开关都关闭，输入维度是 `1 * patch_size`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:56-65`。但当前 Qwen3-8B 配置采用的是 `use_position_embedding=True`。

## 6. 输出 embedding

`已从代码确认`：所有 patch 输入会先拼成一个二维矩阵：

```python
x_patches = torch.cat(patches_list, dim=0)
x = self.mlp(x_patches)
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:171-179`。

因此输出 `x` 的形状是：

```text
[sum(patch_cnt), hidden_size]
```

在当前 Qwen3-8B 配置中：

```text
[所有序列 patch 总数, 4096]
```

`已从代码确认`：`TimeSeriesEmbedding.forward()` 返回两个对象：

```python
return x, patch_cnt
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:179`。

其中：

- `x`：所有 time series patch 的 embedding，按 batch 中序列顺序串起来。
- `patch_cnt`：每条 time series 对应的 patch 数，用于后续把扁平化的 TS embedding 分回每条序列，或用于文本/TS embedding 合并。

`已从代码确认`：HF 模型路径中，`Qwen3TSForCausalLM.forward()` 把 `ts_features, patch_cnt` 传入 `_merge_input_ids_with_time_series_features()`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-588`。该 merge 函数用 `patch_cnt` 计算每条样本需要扩展多少 embedding 位置，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:412-433`。

`已从代码确认`：vLLM 路径中，`get_multimodal_embeddings()` 会把扁平的 `ts_features` 按 `patch_cnt` 拆成 list，每条时间序列一个二维 tensor，见 `inference/vllm/chatts_vllm.py:673-697`。

## 7. 示例推导

假设某个节点的原始时间序列长度为 `L=16`，配置 `patch_size=8`。

`已从代码确认`：`sp_encoding()` 会为每个真实时间点生成一个数值和一个 mask。逻辑上可看成：

```text
[[v0, 1],
 [v1, 1],
 ...
 [v15, 1]]
```

进入 `TimeSeriesEmbedding.forward()` 后：

```text
valid_lengths = mask.sum(dim=1).long()
              = 16

patch_cnt = (valid_lengths + patch_size - 1) // patch_size
          = (16 + 8 - 1) // 8
          = 23 // 8
          = 2
```

因此：

```text
16 个时间点 -> 2 个 patch
patch 0: [v0, v1, ..., v7]
patch 1: [v8, v9, ..., v15]
```

每个 patch 经过 MLP 后生成一个 `hidden_size=4096` 的 TS embedding token：

```text
输出 ts_features 形状: [2, 4096]
```

`已从代码确认`：如果长度不是 8 的倍数，例如 `L=17`，公式会得到 `patch_cnt=3`；第三个 patch 会用最后一个真实数值重复 padding 到 8 个时间点，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:99-119`。

## 8. 设计动机

`根据代码推断，未由真实数据验证`：把时间序列 patchify 后再映射到 LLM embedding 空间，至少解决了三个工程问题。

第一，降低序列长度。若把每个时间点都作为一个 token，长度为 168 的节点会产生 168 个 TS token；当前 `patch_size=8` 时只产生 `ceil(168/8)=21` 个 TS embedding token。代码中的 `patch_cnt = ceil(valid_lengths / patch_size)` 直接体现了这个压缩比例，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-87`。

第二，让连续局部时间片段先形成局部表示。一个 patch 包含连续 8 个时间点；MLP 接收整个 patch 的数值和位置 embedding，而不是孤立处理单点，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:117-119` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:163-169`。

第三，对齐 LLM hidden size。TS patch 经过 MLP 后输出 `hidden_size=4096`，和 Qwen3 的 token embedding 维度一致，见 `base_model/Config-Qwen3-8B/config.json:17`、`base_model/Config-Qwen3-8B/config.json:73-83` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:66-77`。这使得后续 `_merge_input_ids_with_time_series_features()` 可以把 TS embedding 和文本 embedding 放进同一个 `inputs_embeds` 序列，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:474-494`。

与图像 patch embedding / ViT 的类比可以这样讲：

| 维度 | ViT 图像 patch | STReasoner 时间序列 patch |
|---|---|---|
| 切分对象 | 图像二维网格中的小块 | 一维时间轴上的连续片段 |
| patch 内容 | 像素块 | 连续 `patch_size=8` 个时间点 |
| 投影目标 | Transformer hidden size | Qwen3 hidden size |
| 位置意识 | 图像位置 embedding | 时间位置 embedding 或位置索引 |

但不要过度类比：`已从代码确认`，这里的 patch 是一维时间片段，不是二维图像 patch；encoder 是 MLP，不是 ViT patch projection 加完整 ViT encoder。STReasoner 的空间关系主要来自 prompt 中的 `Graph Structure` 和任务数据流，而不是 `TimeSeriesEmbedding` 本身显式建模图结构。

## 9. 组会讲法

### 本文档核心结论

1. `已从代码确认`：`TimeSeriesEmbedding` 是 STReasoner 把数值时间序列接入 Qwen3 embedding 空间的核心模块，位置在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43-179`。
2. `已从代码确认`：输入不是裸数值序列，而是 processor 生成的 value/mask 结构；encoder 内部 reshape 为 `[num_series, seq_len, num_features=2]`，并用最后一个特征作为 mask，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:48` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:79-86`。
3. `已从代码确认`：patch 数是 `ceil(valid_lengths / patch_size)`，当前 `patch_size=8`。长度 16 的节点会产生 2 个 time series embedding token。
4. `已从代码确认`：每个 TS patch 最终被 MLP 映射到 `hidden_size=4096`，从而能和 Qwen3 文本 token embedding 合并进入同一个 Transformer 输入序列。

### 组会可讲版本

可以把 Time Series Encoder 讲成三步：

```text
原始时间序列
  -> processor 归一化，并构造 [value, mask]
  -> TimeSeriesEmbedding 按 8 个时间点切成一个 patch
  -> MLP 把每个 patch 投影成 4096 维 embedding
```

这里 mask 的作用是告诉模型哪些时间点是真实数据、哪些是为了 batch 对齐补出来的 0。`valid_lengths` 由 mask 求和得到，patch 数由 `ceil(valid_lengths / 8)` 得到。比如一个节点长度 16，正好切成两个 patch，所以最终对应 2 个 TS embedding token。

和 ViT 的类比只讲到“切 patch 再投影”：ViT 切的是图像块，STReasoner 切的是一维时间片段；两者都把原始连续信号变成 Transformer 可以处理的 embedding token。但 STReasoner 的空间图信息不是这个 encoder 自己建的，而是由 prompt 里的 `Graph Structure` 和后续 LLM attention 共同使用。

### 后续需要验证的问题

1. `尚未确认`：当前没有真实 ST-Bench batch 和实际 forward 结果，因此尚未用运行结果验证 `sp_encoding()` 输出、batch padding、`TimeSeriesEmbedding.reshape()` 三者在所有真实样本上的形状完全一致。
2. `尚未确认`：`sp_encoding()` docstring 允许 1D 或 2D 原始序列，但本文的形状解释主要针对常见 1D 序列；真实数据中是否存在 2D 序列以及 2D 情况下语义如何，需要下载数据后检查。
3. `尚未确认`：本文对 patchify 设计动机的解释来自代码结构和常见 Transformer 多模态设计，没有引用论文原文逐句确认。
