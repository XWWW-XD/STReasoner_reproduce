### Stage1 Loss 与 cutoff_len 源码追踪报告

日期：2026-06-14

### 1. 核心结论

用 superpowers 追源码后，可以确认：**Stage1 的 loss 是标准 causal LM SFT token-level cross entropy。**

它没有额外设计：
- ST-Align 专用 loss；
- 图结构 loss；
- 时序重建 loss；
- 显式多模态对齐 loss。

也就是说，Stage1 本质上是：

```text
带时间序列 embedding 输入的普通 causal LM SFT
```

### 2. loss 计算链路

#### 2.1 Trainer 没有改写 loss

源码位置：

```text
/root/autodl-tmp/STReasoner_reproduce/src/llamafactory/train/sft/trainer.py:114
```

关键代码：

```python
def compute_loss(self, model, inputs, *args, **kwargs):
    return super().compute_loss(model, inputs, *args, **kwargs)
```

这说明 `CustomSeq2SeqTrainer.compute_loss()` 没有自定义新的 loss，而是直接调用 HuggingFace `Seq2SeqTrainer` 的默认 loss 逻辑。

#### 2.2 哪些部分算loss

对应关系：

| 部分 | 是否算 loss |
| --- | --- |
| system prompt | 否 |
| user prompt | 否 |
| `<ts>` 占位所在的 prompt 区域 | 否 |
| assistant 标准答案 | 是 |
| padding | 否 |

#### 2.3 模型 forward 中计算 causal LM loss

源码位置：

```text
/root/autodl-tmp/STReasoner_reproduce/base_model/Qwen3-4B-Instruct-2507/modeling_qwen3_ts.py:615
```

关键代码：

```python
if labels is not None:
    loss = self.loss_function(
        logits=logits,
        labels=labels,
        vocab_size=self.config.vocab_size,
        **kwargs
    )
```

当前 Transformers 4.52.4 的 `ForCausalLMLoss` 逻辑可以概括为：

```python
logits = logits.float()
labels = pad(labels, (0, 1), value=-100)
shift_labels = labels[..., 1:]
loss = cross_entropy(
    logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction="mean"
)
```

对应公式是：

```text
loss = mean_{t where label_t != -100} CE(logits_{t-1}, label_t)
```

#### 2.4 总结loss计算方法

- 第 t-1 个位置的 hidden state 用来预测第 t 个 assistant token；
- prompt、padding、TS patch embedding 位置全部跳过；
- 最终 loss 是所有有效 assistant token 的平均 cross entropy。

### 6. cutoff_len 的源码结论

#### 6.1 cutoff_len 的含义

`cutoff_len` 是 LLaMA-Factory 预处理 SFT 样本时允许的最大 token 长度，大致控制：

```text
source_ids(prompt) + target_ids(answer) <= cutoff_len
```

如果样本超过 `cutoff_len`，会通过 `infer_seqlen()` 截断 source / target。

#### 6.2 STReasoner 的特殊点

STReasoner 中存在 `<ts>` 时间序列特殊 token。

如果截断破坏了 `<ts>` token 与时间序列 patch 的对应关系，样本会被 drop。

因此在 STReasoner 中，`cutoff_len` 不只是普通文本长度限制，还会影响：

```text
样本是否保留
<ts> 占位与 time-series patch 是否能正确匹配
```

另外，模型 forward 里会把 `<ts>` token 展开成 time-series patch embeddings，所以：

```text
预处理时的 token 长度
```

和

```text
模型 forward 后的最终 embedding 序列长度
```

不是同一个概念。

### 7. 原作者 Stage1 cutoff_len 不是 4096

结论明确：**原作者原代码 Stage1 的 cutoff_len 是 10000，不是 4096。**

证据如下。

官方脚本：

```text
scripts/qwen3-4b-instruct/train_stage1.sh
```

初始提交中就是：

```bash
--cutoff_len 10000
```

`git blame` 显示该设置来自原始提交：

```text
dbe9aa5 Initial commit
作者：LingFengGold
```

并且官方 Qwen3 stage1 / stage2 脚本基本都使用 `--cutoff_len 10000`，包括：

- `scripts/qwen3-4b-instruct/train_stage1.sh`
- `scripts/qwen3-8b/train_stage1.sh`
- `scripts/qwen3-14b/train_stage1.sh`
- 多数 stage2 相关脚本

因此，`10000` 更接近原作者正式训练设置。

### 8. 4096 的来源

`4096` 是后续复现实验中加入的 A100 smoke 脚本设置，不是原作者正式 Stage1 设置。

来源脚本：

```text
00_new_codes/repro_autodl/experiments/scripts/single_a100_qwen3_4b_stage1_smoke.sh
```

其中设置为：

```bash
--cutoff_len 4096
```

`git blame` 显示该设置来自：

```text
ddabbfe sync merged results
```

更早对应：

```text
088c445 Add single A100 stage1 reproduction artifacts
```

该脚本还包含：

```text
max_steps = 10
batch = 1
preprocessing_num_workers = 4
```

这些配置说明它是 A100 单卡 smoke / 低配验证用的保守设置，不是正式训练配置。

### 9. 当前 A800 Stage1 使用的是 cutoff_len=10000

当前 A800 Stage1 最终训练配置中使用的是：

```text
cutoff_len = 10000
```

这一点在以下位置一致：

```text
27-服务器2-A800-Qwen3-4B-Stage1最终训练配置.md
训练日志
```

因此，回答 cutoff_len 相关问题时，应区分：

| 设置 | 来源 | 含义 |
| --- | --- | --- |
| `cutoff_len=10000` | 原作者 Stage1 / 当前 A800 正式训练 | 正式训练方向 |
| `cutoff_len=4096` | 后续 A100 smoke 脚本 | 低配验证 / 保守排查设置 |