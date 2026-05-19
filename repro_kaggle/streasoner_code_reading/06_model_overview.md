# 06 模型结构总览

本文档阅读范围：`base_model/Config-Qwen3-8B/` 下的 `modeling_qwen3_ts.py`、`configuration_qwen3_ts.py`、`processing_qwen3_ts.py`、`config.json`、`tokenizer_config.json`、`added_tokens.json`、`special_tokens_map.json`、`chat_template.jinja`，并补充实际 vLLM 推理路径中的 `inference/vllm/chatts_vllm.py`。

术语约定：

- `已从代码确认`：可以直接绑定到本仓库文件、类、函数或配置字段。
- `根据代码推断，未由真实数据验证`：由多处静态逻辑串起来得到，但没有加载真实权重或跑 forward。
- `尚未确认`：当前仓库静态阅读不足以确认，已同步写入 `docs/streasoner_code_reading/uncertainty_log.md`。

## 1. 普通 Qwen3 vs STReasoner

### 1.1 总体差异

| 维度 | 普通 Qwen3 | STReasoner / Qwen3TS | 证据状态 |
|---|---|---|---|
| 模型骨架 | 文本 `input_ids` 经 token embedding 进入 Qwen3 decoder，再经 `lm_head` 输出 next-token logits | 仍复用 HuggingFace `Qwen3Model` 作为语言模型主体，同时增加 `ts_encoder` 处理时间序列 | `已从代码确认`：`Qwen3TSForCausalLM.__init__()` 中 `self.model = Qwen3Model(config)`，`self.lm_head = nn.Linear(...)`，`self.ts_encoder = TimeSeriesEmbedding(config.ts)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:357-365` |
| 输入模态 | 只有文本 token | 文本 token + `timeseries` 张量；文本中用 `<ts><ts/>` 标记时间序列插入位置 | `已从代码确认`：`Qwen3TSForCausalLM.forward()` 增加 `timeseries` 参数，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:515-531`；processor 注释说明 prompt 含 `<ts><ts/>`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:72-93` |
| 时间序列编码 | 无 | `TimeSeriesEmbedding` 将数值序列切成 patch，经 MLP 映射到 Qwen hidden size | `已从代码确认`：`TimeSeriesEmbedding` 定义在 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43`；MLP 构建与 patch 处理见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:66-79` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:79-179` |
| embedding 合并 | token embedding 序列直接进入 Qwen3 | 先得到文本 embedding，再用 TS embedding 替换/扩展 `<ts>` 位置，最后把混合 embedding 送入 Qwen3 | `已从代码确认`：HF 路径在 `_merge_input_ids_with_time_series_features()` 合并，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513`；vLLM 路径使用 `merge_multimodal_embeddings()`，见 `inference/vllm/chatts_vllm.py:699-709` |
| tokenizer / processor | Qwen tokenizer + chat template | 增加 `<ts>`、`<ts/>` 特殊 token，并把 `AutoProcessor` 映射到 `Qwen3TSProcessor` | `已从代码确认`：`config.json` 的 `auto_map` 见 `base_model/Config-Qwen3-8B/config.json:7-12`；`added_tokens.json` 中 `<ts>`/`<ts/>` id 见 `base_model/Config-Qwen3-8B/added_tokens.json:8-9` |
| 推理部署 | 标准 HF/vLLM Qwen3 | 需要 `trust_remote_code=True` 或 vLLM 注册自定义 `Qwen3TSForCausalLM` | `已从代码确认`：`initial_model.py` 用 `AutoModelForCausalLM.from_pretrained(... trust_remote_code=True)`，见 `initial_model.py:54-60`；vLLM 注册见 `inference/vllm/chatts_vllm.py:760-765` |

`已从代码确认`：STReasoner 没有在 `modeling_qwen3_ts.py` 中重写 Qwen3 decoder block，而是导入 `transformers.models.qwen3.modeling_qwen3.Qwen3Model` 和 `Qwen3PreTrainedModel`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:28`。因此模型结构可以概括为：

```text
Qwen3TSForCausalLM
├── self.model: Qwen3Model                 # 原 Qwen3 decoder 主体
├── self.lm_head: Linear(hidden, vocab)     # 语言模型输出头
└── self.ts_encoder: TimeSeriesEmbedding    # 新增时间序列 patch encoder
```

`已从代码确认`：`base_model/Config-Qwen3-8B/config.json` 中 `architectures` 是 `Qwen3TSForCausalLM`，`AutoModelForCausalLM` 指向 `modeling_qwen3_ts.Qwen3TSForCausalLM`，`AutoProcessor` 指向 `processing_qwen3_ts.Qwen3TSProcessor`，见 `base_model/Config-Qwen3-8B/config.json:1-12`。这说明下载普通 Qwen3 权重后，需要把这些自定义文件复制进本地模型目录，README 也给出 `cp -rf base_model/Config-Qwen3-8B/* base_model/Qwen3-8B/`，见 `README.md:90-94`。

`尚未确认`：当前本地没有 `base_model/Qwen3-8B/` 完整权重目录，只有 `base_model/Config-Qwen3-8B/` 配置与代码模板。因此本文没有验证真实权重加载后 `ts_encoder` 参数是否成功初始化和保存。

## 2. 新增模块

### 2.1 base_model / HuggingFace 路径

| 类 / 函数 | 位置 | 作用 | 证据状态 |
|---|---|---|---|
| `Qwen3TSConfig` | `base_model/Config-Qwen3-8B/configuration_qwen3_ts.py:25` | 自定义配置类，继承 `PretrainedConfig`，保留 Qwen3 主要配置字段 | `已从代码确认` |
| `sp_encoding()` | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24` | 对单条时间序列做均值中心化、缩放，并生成统计 prompt 片段 | `已从代码确认` |
| `Qwen3TSProcessor` | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:52` | 同时处理文本 prompt 和 `timeseries`，输出 tokenizer 结果与 TS 张量 | `已从代码确认` |
| `Qwen3TSProcessor.__call__()` | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:72` | 处理 `<ts><ts/>` 占位符、编码 TS、padding TS、返回 `BatchFeature` | `已从代码确认` |
| `Qwen3TSProcessor.encode_timeseries()` | `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:173` | 训练多模态插件中直接把 `timeseries` 转为模型输入张量 | `已从代码确认` |
| `TimeSeriesEmbedding` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:43` | 新增 TS encoder；按 `patch_size` 切分序列并映射到 `hidden_size` | `已从代码确认` |
| `Qwen3TSCausalLMOutputWithPast` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:182` | 在标准 CausalLM 输出上额外保存扩展后的 `attention_mask`、`labels`、`new_token_positions` | `已从代码确认` |
| `Qwen3TSGenerationMixin` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:221` | 让 generation 接受 `timeseries`，并在后续 token 生成时避免重复处理 TS | `已从代码确认` |
| `Qwen3TSPreTrainedModel` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:348` | 绑定 `config_class = Qwen3TSConfig` | `已从代码确认` |
| `Qwen3TSForCausalLM` | `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:352` | STReasoner 的 HF 模型主体，组合 `Qwen3Model`、`lm_head` 和 `ts_encoder` | `已从代码确认` |

### 2.2 vLLM 推理路径

实际 `inference/inference_tsmllm_vllm.py` 使用 vLLM，因此还需要关注 `inference/vllm/chatts_vllm.py` 中的并行实现。

| 类 / 函数 | 位置 | 作用 | 证据状态 |
|---|---|---|---|
| `TimeSeriesEmbedding` | `inference/vllm/chatts_vllm.py:52` | vLLM 路径下的 TS encoder，实现与 HF 路径高度相似 | `已从代码确认` |
| `get_patch_cnt()` | `inference/vllm/chatts_vllm.py:189` | 根据 mask 和 `patch_size` 计算 patch 数 | `已从代码确认` |
| `Qwen2TSProcessingInfo` | `inference/vllm/chatts_vllm.py:202` | 声明 HF config/processor 获取方式，以及每个 prompt 最多 50 条 `timeseries` | `已从代码确认` |
| `Qwen2TSDummyInputsBuilder` | `inference/vllm/chatts_vllm.py:214` | vLLM profiling/dummy input 构造，dummy 文本为重复 `<ts><ts/>` | `已从代码确认` |
| `TimeSeriesProcessorItems` | `inference/vllm/chatts_vllm.py:250` | vLLM multimodal parser 中的原始 time series item 容器 | `已从代码确认` |
| `TimeSeriesEmbeddingItems` | `inference/vllm/chatts_vllm.py:275` | vLLM multimodal parser 中的已 embedding item 容器 | `已从代码确认` |
| `Qwen2TSDataParser` | `inference/vllm/chatts_vllm.py:282` | 把 `multi_modal_data["timeseries"]` 解析为 vLLM multimodal item | `已从代码确认` |
| `Qwen2TSMultiModalProcessor` | `inference/vllm/chatts_vllm.py:307` | 调用 HF `Qwen3TSProcessor`，设置 `vllm_flag=True`，并定义 prompt replacement | `已从代码确认` |
| `Qwen3TSForCausalLM` | `inference/vllm/chatts_vllm.py:587` | vLLM 注册用的 Qwen3TS 模型，内部通过 `init_vllm_registered_model(... architectures=["Qwen3ForCausalLM"])` 复用 vLLM 的 Qwen3 | `已从代码确认` |
| `ModelRegistry.register_model("Qwen3TSForCausalLM", ...)` | `inference/vllm/chatts_vllm.py:764` | 把自定义模型名注册进 vLLM | `已从代码确认` |

`已从代码确认`：vLLM 中处理器命名为 `Qwen2TS*`，但同一个 processor 被注册给 `Qwen3TSForCausalLM` 使用，见 `inference/vllm/chatts_vllm.py:582-587`。这里的 `Qwen2TS` 更像复用自 ChatTS/Qwen2 时代的命名，而不是说明底层一定是 Qwen2。

### 2.3 训练侧相关接入点

虽然本文重点是模型结构，但训练侧也有两个和模型结构强相关的接入点：

- `已从代码确认`：LLaMA-Factory 多模态插件 `ChatTSPlugin` 会把 `timeseries` 交给 processor 的 `encode_timeseries()`，见 `src/llamafactory/data/mm_plugin.py:1999-2015`。
- `已从代码确认`：`ChatTSPlugin.process_messages()` 会调用 `processor(text=text, timeseries=timeseries, tokenize=False)`，把文本中的 `<ts><ts/>` 替换成 processor 生成的 TS prompt 片段，见 `src/llamafactory/data/mm_plugin.py:2018-2034`。
- `已从代码确认`：训练模板 `STReasoner-Align` 与 `STReasoner-CoT` 都注册 `mm_plugin=get_mm_plugin("streasoner", timeseries_token="<ts>")`，见 `src/llamafactory/data/template.py:1979-1991` 和 `src/llamafactory/data/template.py:1994-2007`。

## 3. Forward 输入输出

### 3.1 HuggingFace forward

`已从代码确认`：HF 模型入口是 `Qwen3TSForCausalLM.forward()`，关键参数如下，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:515-531`：

| 参数 | 作用 | 说明 |
|---|---|---|
| `input_ids` | 文本 token id | 包含普通文本 token，也包含 `<ts>` / `<ts/>` 特殊 token |
| `timeseries` | 时间序列张量 | 送入 `self.ts_encoder`；静态代码显示实际会被 reshape 成 `[num_series, seq_len, num_features]` |
| `attention_mask` | 文本 attention mask | 如果插入 TS patch embedding，会被扩展成更长序列的 mask |
| `position_ids` | 位置 id | merge 后重新计算，传给 Qwen3Model |
| `inputs_embeds` | 可选预嵌入输入 | 如果已经提供，则代码不会再用 `input_ids` 计算 embedding，也不会进入 `timeseries` merge 分支 |
| `labels` | 训练标签 | merge 时和文本 token 对齐，TS 位置保持 ignore |
| `past_key_values` / `use_cache` | generation cache | 交给底层 `Qwen3Model` |

`已从代码确认`：forward 的核心过程如下：

```text
Qwen3TSForCausalLM.forward()
  1. inputs_embeds = self.get_input_embeddings()(input_ids)
  2. if timeseries is not None:
       ts_features, patch_cnt = self.ts_encoder(timeseries)
       inputs_embeds, attention_mask, position_ids, labels, new_token_positions =
           self._merge_input_ids_with_time_series_features(...)
  3. outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=..., position_ids=...)
  4. logits = self.lm_head(outputs.last_hidden_state)
  5. return Qwen3TSCausalLMOutputWithPast(...)
```

对应代码位置：

- `inputs_embeds = self.get_input_embeddings()(input_ids)`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:577-578`
- `self.ts_encoder(timeseries)`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:580-583`
- `_merge_input_ids_with_time_series_features(...)`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:587-589`
- 调用底层 `self.model(...)`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:595-605`
- `lm_head` 计算 logits：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:607-614`
- 返回 `Qwen3TSCausalLMOutputWithPast`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:617-625`

`已从代码确认`：`Qwen3TSGenerationMixin.prepare_inputs_for_generation()` 会把 `timeseries` 放进 `model_inputs`，但如果 `past_key_values` 已经存在且 `past_length > 0`，会把 `input_ids` 缩到最后一个 token，并把 `timeseries = None`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:229-277`。这表示 TS 只在 generation 的首个 forward 被编码，后续自回归步依赖 cache。

`根据代码推断，未由真实数据验证`：forward docstring 中写 `timeseries` 形状是 `(batch_size, num_patches, patch_size)`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:556-557`；但 processor 实际把一条序列编码为 value/mask 成对特征，再 reshape 为 `(-1, 1)`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:48`，`TimeSeriesEmbedding.forward()` 又用 `config.ts["num_features"] == 2` reshape，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:79-87` 和 `base_model/Config-Qwen3-8B/config.json:73-83`。因此更贴近代码的理解是：模型收到的是编码后的 TS 张量，最后一维/相邻特征中包含数值与 mask，patch 切分发生在 `TimeSeriesEmbedding` 内部，而不是外部直接传入 patch。

### 3.2 vLLM forward

`已从代码确认`：实际推理使用的 vLLM 模型入口是 `inference/vllm/chatts_vllm.py` 中的 `Qwen3TSForCausalLM.forward()`，见 `inference/vllm/chatts_vllm.py:711-734`。它的核心过程是：

```text
vLLM Qwen3TSForCausalLM.forward(input_ids, positions, **kwargs)
  1. ts_features = self.get_multimodal_embeddings(**kwargs)
  2. inputs_embeds = self.get_input_embeddings(input_ids, ts_features)
  3. hidden_states = self.language_model.model(input_ids=None, positions=..., inputs_embeds=...)
  4. compute_logits() 再调用 self.language_model.compute_logits(...)
```

对应代码位置：

- 解析 `timeseries`：`_parse_and_validate_ts_input()`，见 `inference/vllm/chatts_vllm.py:628-671`
- TS encoder：`get_multimodal_embeddings()` 内部调用 `self.ts_encoder(ts_input)`，见 `inference/vllm/chatts_vllm.py:673-697`
- 合并 embedding：`get_input_embeddings()` 调用 `merge_multimodal_embeddings(...)`，见 `inference/vllm/chatts_vllm.py:699-709`
- 调底层 vLLM Qwen3：`self.language_model.model(...)`，见 `inference/vllm/chatts_vllm.py:730-733`
- logits：`compute_logits()`，见 `inference/vllm/chatts_vllm.py:736-742`

`已从代码确认`：vLLM 路径中 `Qwen3TSForCausalLM.__init__()` 通过 `init_vllm_registered_model(... architectures=["Qwen3ForCausalLM"])` 复用 vLLM 已有 Qwen3ForCausalLM，同时增加 `self.ts_encoder = TimeSeriesEmbedding(config.ts)`，见 `inference/vllm/chatts_vllm.py:607-623`。

## 4. Embedding 合并机制

### 4.1 时间序列先被 processor 编码

`已从代码确认`：`sp_encoding()` 做了三件事，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50`：

1. 把输入 `timeseries` 转成 numpy 数组。
2. 计算均值 `mean`，做中心化；如果中心化后绝对值超过 3，则按最大绝对值除以 3 的比例缩放。
3. 生成统计文本片段：

```text
[offset=...|scaling=...|length=...|max=...|min=...|left=...|right=...]<ts>
```

如果 `eots_token=True`，还会追加 `<ts/>`。

`已从代码确认`：`sp_encoding()` 同时把时间序列编码成 value/mask 结构：

```python
result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)
```

见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:48`。这里的 `ones_like` 是有效值 mask；后续 `TimeSeriesEmbedding.forward()` 用最后一个特征作为 mask 计算有效长度，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-87`。

### 4.2 HF / SFT 路径：替换 prompt 并返回 TS tensor

`已从代码确认`：当 `vllm_flag=False` 时，`Qwen3TSProcessor.__call__()` 会按 `<ts><ts/>` 切分 prompt，然后对每个占位符对应的 `timeseries` 调用 `sp_encoding()`，把占位符替换成统计 prompt 片段，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:120-137`。

`已从代码确认`：如果 `timeseries` 数量与 `<ts><ts/>` 占位符数量不一致，会抛出 `ValueError`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:139-143`。

`已从代码确认`：processor 会把不同长度的 TS padding 到同一长度，拼成 `concatenated_ts`，并转成 `torch.float16`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:145-155`；最后把它放到 `outputs["timeseries"]`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:164-170`。

### 4.3 HF 模型内部：用 `<ts>` / `<ts/>` 决定插入位置

`已从代码确认`：`_merge_input_ids_with_time_series_features()` 会先在 `input_ids` 中找两个 TS 特殊 token：

```python
special_ts_token_mask_start = input_ids == self.config.ts_token_start_index
special_ts_token_mask_end = input_ids == self.config.ts_token_end_index
special_ts_token_mask = special_ts_token_mask_start | special_ts_token_mask_end
```

见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:407-410`。

`已从代码确认`：`TimeSeriesEmbedding.forward()` 根据 mask 算有效长度，再用 `ceil(valid_length / patch_size)` 得到 `patch_cnt`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:83-87`。`patch_size` 在配置中是 8，见 `base_model/Config-Qwen3-8B/config.json:73-83`。

`已从代码确认`：合并函数会根据每条 TS 的 `patch_cnt` 扩展序列长度，先把普通文本 token embedding 移到新位置，再把 TS patch embedding 填入剩余位置，见：

- 计算 batch 内 TS patch 总长度：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:412-433`
- 计算文本 token 的新位置：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:435-447`
- 创建 `final_embedding` 与 `final_attention_mask`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:449-460`
- 填入文本 embedding：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:474-477`
- 填入 TS embedding：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:479-494`
- 重算 `position_ids`：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:498-501`
- 返回合并后的 embedding/mask/position/labels：`base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:510-513`

`根据代码推断，未由真实数据验证`：合并函数把 `<ts>` 和 `<ts/>` 两个特殊 token 都视为占位符，不作为普通文本 embedding 保留；真正进入 Qwen3 的是对应数量的 TS patch embedding。代码里使用 `patch_cnt - 2` 扩展 `<ts>` 起始 token 的位置贡献，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:423-430`，这与“原本两个 token 占位，最终替换成 `patch_cnt` 个 TS embedding”的逻辑一致。

### 4.4 vLLM 路径：prompt replacement + multimodal embedding merge

`已从代码确认`：vLLM 路径中，HF processor 不直接改 prompt，而是设置 `vllm_flag=True`，见 `inference/vllm/chatts_vllm.py:312-330`。对应的 `Qwen3TSProcessor.__call__()` 会保留原始 prompt，同时为每条 TS 生成 `ts_tokens` 和 `encoded_ts_arrays`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:104-118`；最后返回 `outputs["timeseries"] = zip(ts_tokens, encoded_ts_arrays)`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:164-167`。

`已从代码确认`：vLLM 的 `_get_prompt_updates()` 使用 `hf_config.ts_token_start_index` 作为 placeholder，并把 target 设为 `[placeholder, placeholder + 1]`，也就是 `<ts>` 和 `<ts/>`，见 `inference/vllm/chatts_vllm.py:352-401`。它会根据 patch 数补充足够数量的 `<ts>` placeholder token，使 vLLM 有位置放 TS embedding。

`已从代码确认`：vLLM 模型合并时调用：

```python
inputs_embeds = merge_multimodal_embeddings(
    input_ids, inputs_embeds, multimodal_embeddings,
    self.config.ts_token_start_index
)
```

见 `inference/vllm/chatts_vllm.py:699-709`。这说明 vLLM 路径主要依赖 vLLM 的 multimodal embedding merge 工具，把 TS embedding 填到 `ts_token_start_index` 对应的位置。

## 5. 特殊 Token

### 5.1 TS 特殊 token

| token | id | 配置位置 | 模型使用位置 | 证据状态 |
|---|---:|---|---|---|
| `<ts>` | 151669 | `base_model/Config-Qwen3-8B/added_tokens.json:8-9`；`base_model/Config-Qwen3-8B/tokenizer_config.json:213-220`；`base_model/Config-Qwen3-8B/config.json:85-86` | `ts_token_start_index`；用于寻找 TS 起点和 vLLM placeholder | `已从代码确认` |
| `<ts/>` | 151670 | `base_model/Config-Qwen3-8B/added_tokens.json:8-9`；`base_model/Config-Qwen3-8B/tokenizer_config.json:221-228`；`base_model/Config-Qwen3-8B/config.json:85-86` | `ts_token_end_index`；HF merge 中和 `<ts>` 一起被视为 TS 占位符 | `已从代码确认` |

`已从代码确认`：`special_tokens_map.json` 把 `<ts>` 和 `<ts/>` 放在 `additional_special_tokens` 中，见 `base_model/Config-Qwen3-8B/special_tokens_map.json:1-5`。`tokenizer_config.json` 也声明 `additional_special_tokens` 包含这两个 token，见 `base_model/Config-Qwen3-8B/tokenizer_config.json:230-233`。

### 5.2 Qwen chat 相关 token

| token / 字段 | id 或值 | 作用 | 证据 |
|---|---:|---|---|
| `<|endoftext|>` | 151643 | config 中既是 `bos_token_id` 也是 `pad_token_id` | `base_model/Config-Qwen3-8B/config.json:13` 和 `base_model/Config-Qwen3-8B/config.json:65` |
| `<|im_end|>` | 151645 | `eos_token_id`，chat template 中用于结束 message | `base_model/Config-Qwen3-8B/config.json:14`；`base_model/Config-Qwen3-8B/chat_template.jinja:13-15` |
| `<|im_start|>` | 151644 | chat template 中用于开始 system/user/assistant message | `base_model/Config-Qwen3-8B/added_tokens.json:18-19`；`base_model/Config-Qwen3-8B/chat_template.jinja:13-15` 和 `base_model/Config-Qwen3-8B/chat_template.jinja:31-32` |
| `processor_class` | `Qwen3TSProcessor` | AutoProcessor 加载自定义 processor | `base_model/Config-Qwen3-8B/tokenizer_config.json:245` |
| `tokenizer_class` | `Qwen2Tokenizer` | tokenizer 类名沿用 Qwen2Tokenizer | `base_model/Config-Qwen3-8B/tokenizer_config.json:247` |

`已从代码确认`：chat template 在 `add_generation_prompt=True` 时追加 `<|im_start|>assistant\n`，见 `base_model/Config-Qwen3-8B/chat_template.jinja:84-89`。这与前文 prompt 构造文档中的 `LLMClient._apply_chat_template()` 对应。

## 6. 模型输入示意图

### 6.1 概念图

```text
原始 JSONL 样本
├── input:
│   "... Node 0 time series with length L: <ts><ts/>; ... question ..."
└── timeseries:
    [ts_0, ts_1, ...]

文本 token 流
input text
  -> tokenizer / chat template
  -> input_ids:
     [普通文本token, 151669(<ts>), 151670(<ts/>), 普通文本token, ...]
  -> Qwen3 token embedding:
     [text_emb_0, ts_placeholder_emb_start, ts_placeholder_emb_end, text_emb_1, ...]

TS embedding token 流
timeseries ts_i
  -> sp_encoding()
     value/mask 编码 + 统计 prompt 信息
  -> TimeSeriesEmbedding.forward()
     reshape 为 [seq_len, num_features=2]
     按 patch_size=8 切 patch
     每个 patch 经 MLP 映射到 hidden_size=4096
  -> ts_features:
     [ts_patch_emb_0, ts_patch_emb_1, ..., ts_patch_emb_k]

合并后进入语言模型
[text_emb_0,
 ts_patch_emb_0, ts_patch_emb_1, ..., ts_patch_emb_k,
 text_emb_1, ...]
  -> Qwen3Model
  -> lm_head
  -> generated answer token
```

### 6.2 和普通 LLM 输入的关键区别

`已从代码确认`：普通 LLM 只会看到 `<ts>` 和 `<ts/>` 两个普通文本 token；STReasoner 的模型代码会把这两个 token 当成占位符，并用 TS patch embedding 替换/扩展它们。这个行为依赖 `ts_token_start_index` 和 `ts_token_end_index`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:407-410`。

`已从代码确认`：TS embedding 的维度和 Qwen hidden size 对齐。配置中 `hidden_size=4096`，TS 配置中 `ts.hidden_size=4096`，见 `base_model/Config-Qwen3-8B/config.json:17` 和 `base_model/Config-Qwen3-8B/config.json:73-83`。`TimeSeriesEmbedding` 的最后一层输出也是 `self.hidden_size`，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:66-77`。

`根据代码推断，未由真实数据验证`：从 embedding 维度对齐和 merge 逻辑看，STReasoner 不是把完整数值序列展开成大量文本数字，而是把数值序列压成若干个连续的 hidden-size patch embedding，让 Qwen3 decoder 在同一个 attention 序列中同时 attend 文本 token 与 TS patch token。

## 7. 组会讲法

### 本文档核心结论

1. `已从代码确认`：STReasoner 的模型主体仍是 Qwen3，代码通过 `Qwen3TSForCausalLM` 包一层，在 Qwen3Model 前面增加 `TimeSeriesEmbedding` 和 embedding merge 机制，见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:352-365`。
2. `已从代码确认`：时间序列不是直接作为文本数字输入，而是由 `Qwen3TSProcessor` 编码为 value/mask 张量，再由 `TimeSeriesEmbedding` 切 patch 并映射到 `hidden_size=4096`，见 `base_model/Config-Qwen3-8B/processing_qwen3_ts.py:24-50` 和 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:79-179`。
3. `已从代码确认`：文本中的 `<ts>` / `<ts/>` 是关键锚点。HF 路径用 `_merge_input_ids_with_time_series_features()` 自己重排 embedding；vLLM 路径用 prompt replacement 与 `merge_multimodal_embeddings()` 完成合并，分别见 `base_model/Config-Qwen3-8B/modeling_qwen3_ts.py:387-513` 和 `inference/vllm/chatts_vllm.py:352-401`, `inference/vllm/chatts_vllm.py:699-709`。
4. `已从代码确认`：推理必须先注册自定义 vLLM 模型，否则 vLLM 不认识 `Qwen3TSForCausalLM`；注册代码在 `inference/vllm/chatts_vllm.py:760-765`。

### 组会可讲版本

这一部分可以用一句话概括：STReasoner 不是把时序数据转成一长串数字喂给普通 Qwen3，而是在 Qwen3 前面加了一个时间序列 patch encoder。文本 prompt 里保留 `<ts><ts/>` 作为插入位置，真实数值序列走另一条张量通道，经过归一化、value/mask 编码、patch 切分和 MLP，变成和 Qwen3 token embedding 同维度的 TS embedding。随后模型把普通文本 embedding 和 TS patch embedding 拼在同一个序列里，再交给 Qwen3 decoder 做自回归生成。

如果画 PPT，可以画成三段：

```text
Prompt text with <ts><ts/>
        │
        ├── tokenizer -> text embeddings
        │
Timeseries values
        └── Qwen3TSProcessor -> TimeSeriesEmbedding -> TS patch embeddings

text embeddings + TS patch embeddings
        -> Qwen3Model
        -> lm_head
        -> reasoning answer
```

组会上需要强调两点：

- 第一，`<ts>` 不是普通装饰符，它是模型定位 TS embedding 插入点的特殊 token。
- 第二，STReasoner 的空间时序能力在代码层面至少来自三处组合：数据 prompt 中的图结构文本、`timeseries` 数值张量、以及模型内新增的 TS encoder 和 embedding merge。

### 后续需要验证的问题

1. `尚未确认`：本地没有完整 `base_model/Qwen3-8B/` 权重目录，尚未验证 `initial_model.py` 是否能成功加载自定义 `Qwen3TSForCausalLM` 并初始化保存 `ts_encoder`。
2. `尚未确认`：没有实际跑 forward，因此尚未用真实 batch 验证 HF 路径中 `_merge_input_ids_with_time_series_features()` 的长度扩展与 `labels` 对齐是否和训练时完全一致。
3. `尚未确认`：`config.json` 中 `model_type` 是 `"qwen3"`，见 `base_model/Config-Qwen3-8B/config.json:61`；但训练侧 `TIMESERIES_MODELS` 注册的是 `"chatts"` 和 `"qwen3ts"`，见 `src/llamafactory/model/model_utils/timeseries.py:67-79`。这是否影响 `timeseries_sft_lr`、冻结 TS 模块或 LoRA target 的自动识别，需要在训练配置阅读或小规模 dry check 中确认。
4. `尚未确认`：vLLM 路径和 HF 路径的 TS merge 实现不完全相同；当前只确认了静态流程，尚未用同一条样本比较两条路径的 token/embedding 对齐结果。
