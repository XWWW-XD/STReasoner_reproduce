### 服务器2 A800 Qwen3-4B Stage1 训练过程记录

日期：2026-06-14

### 1. 主要依据


| 来源                                                                                      | 内容                                                                  |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `checkpoint-500-paused/trainer_state.json`                                              | 逐步指标变化，step 1–500 的 loss / grad_norm / learning rate / epoch        |
| `qwen3_4b_stage1_align_single_a800_20260612_214947_bf16_zero3opt_batch2_ga32_train.log` | 正式训练日志                                                              |
| 报告 27                                                                                   | 训练配置：bf16、ZeRO-3、optimizer CPU offload、batch=2、GA=32、max_steps=1000 |


`logging_steps=1`，因此每个 global step 都有一条训练记录。

### 2. 数据与训练配置

训练数据：


| 文件                                             | 用途        | 行数      |
| ---------------------------------------------- | --------- | ------- |
| `data/ST-Bench/ST-Align/alignment_train.jsonl` | Stage1 训练 | 194,212 |
| `data/ST-Bench/ST-Align/alignment_test.jsonl`  | 测试        | 40,512  |


正式训练配置：


| 配置项                   | 取值                             |
| --------------------- | ------------------------------ |
| 模型                    | Qwen3-4B                       |
| 任务                    | Stage1 alignment SFT           |
| 精度                    | bf16                           |
| DeepSpeed             | ZeRO-3 + optimizer CPU offload |
| micro batch           | 2                              |
| gradient accumulation | 32                             |
| GPU 数                 | 1                              |
| global batch          | 64                             |
| max_steps             | 1000                           |
| warmup_ratio          | 0.02                           |


这里的 batch 关系是：


| 名称                    | 含义                        | 本次取值            |
| --------------------- | ------------------------- | --------------- |
| micro batch           | 每次 forward 进入 GPU 的样本数    | 2               |
| gradient accumulation | 累积多少次 micro batch 后更新一次参数 | 32              |
| global batch          | 每个 global step 实际消费的样本数   | 2 × 32 × 1 = 64 |


因此，本次训练每次 forward 用 2 条，累积 32 次后更新一次参数，所以每个 global step 消费 64 条样本。

### 3. 数据实际怎么被使用

训练开始时会加载 `alignment_train.jsonl` 全部 194,212 条样本，但加载全部数据不等于已经训练完全部数据。

实际流程是：

1. 加载全部训练样本；
2. Dataloader 打乱顺序；
3. 每次取 2 条样本进入 GPU；
4. 对这 2 条样本做 forward / backward，计算 answer token 上的 SFT loss；
5. 重复 32 次，累计 64 条样本的梯度；
6. 做 1 次 optimizer update，global step 加 1；
7. 继续取下一批样本。

### 4. loss 的含义

训练日志里的 loss 不是全量数据评测分数，也不是模型自由生成后和答案比较得到的准确率。

SFT 训练使用 **teacher forcing**：模型看到输入和标准答案前缀，预测下一个标准答案 token，然后在标准答案 token 上计算 **cross entropy loss**。

因此，loss 表示的是“当前 batch 中，模型预测标准答案 token 的平均难度”。

数据还没完整过完一遍，但 loss 可以稳定在较低水平，原因包括：

- 模型不是从零训练，已有语言建模能力；
- alignment 输出通常较短，格式相对固定；
- teacher forcing 比真实推理更容易；
- loss 只反映当前 batch，不等于全量测试表现。

### 5. epoch≈0.165 的含义

本地 alignment 训练集有 194,212 条样本，global batch=64。

所以 1 个 epoch 大约需要：

`194,212 ÷ 64 ≈ 3,035 step`

对应关系：


| 位置        | 消费样本数              | 约等于          |
| --------- | ------------------ | ------------ |
| step 500  | 500 × 64 = 32,000  | 0.165 epoch  |
| step 1000 | 1000 × 64 = 64,000 | 0.33 epoch   |
| 1 epoch   | 194,212            | 约 3,035 step |


所以，step 500 的含义是：

- 训练计划完成 500 / 1000 = 50%；
- 但训练集只扫过约 16.5%；
- 因此 checkpoint-500 不是“数据训练了一半”，而是“训练计划跑到一半，但数据只过了约 1/6”。

与 HF 作者配置对比：


| 配置        | global batch | 1000 step 约消费样本数 | 相当于本地训练集   |
| --------- | ------------ | ---------------- | ---------- |
| 本次单卡      | 64           | 64,000           | 0.33 epoch |
| HF 作者 8 卡 | 512          | 512,000          | 2.6 epoch  |


两者虽然都是 1000 step，但实际看过的数据量差很多。

### 6. 关键指标

step 1 与 step 500 对比：


| 指标            | step 1  | step 500 |
| ------------- | ------- | -------- |
| loss          | 16.52   | 0.21     |
| grad_norm     | 595.2   | 2.78     |
| learning_rate | 0       | 5.18e-06 |
| ts_encoder_lr | 5.0e-07 | 5.16e-06 |
| epoch         | 0.00033 | 0.165    |


训练里程碑：


| step | loss  | grad_norm | learning_rate |
| ---- | ----- | --------- | ------------- |
| 1    | 16.52 | 595.2     | 0             |
| 50   | 0.50  | 48.3      | 9.98e-06      |
| 100  | 0.52  | 26.4      | 9.84e-06      |
| 200  | 0.24  | 5.67      | 9.20e-06      |
| 300  | 0.25  | 3.68      | 8.13e-06      |
| 400  | 0.26  | 3.63      | 6.74e-06      |
| 500  | 0.21  | 2.78      | 5.18e-06      |


补充观察：

- loss 最低点为 0.097，出现在 step 488；
- step 500 附近 loss 在 0.20 左右小幅波动，属于正常尾段波动；
- grad_norm 峰值 3157 出现在 step 15，属于 warmup 早期尖峰；
- 后续 grad_norm 快速回落到个位数，未见持续梯度爆炸。

### 7. 曲线结论

loss 曲线：

- step 1–30 快速下降，从 16.5 降到约 1；
- step 30–100 继续下降到约 0.5；
- step 100 后进入 0.2–0.3 平台。

grad_norm 曲线：

- 前 20 step 有明显尖峰；
- 后续快速回落并稳定到个位数；
- 未见持续梯度爆炸。

learning rate 曲线：

- warmup 约 20 step；
- 之后按 cosine 衰减；
- LLM 与 TS encoder 的 learning rate 走势同步。

epoch 曲线：

- step 500 时 epoch≈0.165；
- 表明当前只扫过训练集约 16.5%。

### 8. 阶段性判断

1. 训练过程稳定：loss 明显下降，grad_norm 后期稳定，没有持续发散。
2. checkpoint-500 可用：step 500 时 loss≈0.21，grad_norm≈2.78，训练状态健康。
3. 本次训练未训满：checkpoint-500 只完成计划 1000 step 的一半，且只扫过训练集约 16.5%。
4. 不能把 checkpoint-500 解释为“全量 alignment 已充分训练”：它更准确地说是 Stage1 alignment 的中途 checkpoint。
5. 若后续续训，需要注意 `--save_only_model` 保存的断点不包含 optimizer 状态，因此不是严格无缝续训。

### 9. 复现命令

重新生成训练曲线：

```bash
cd /root/autodl-tmp/STReasoner_reproduce
/root/autodl-tmp/conda/envs/str-py310/bin/python \
  00_new_codes/repro_autodl/experiments/scripts/plot_stage1_a800_ckpt500_metrics.py
```

输出目录：

```text
00_new_codes/reports/t3-autodl2-三阶段训练复现/artifacts/stage1_a800_ckpt500/
  training_curves.png
  training_loss.png
  training_summary.json
```

脚本位置：

```text
00_new_codes/repro_autodl/experiments/scripts/plot_stage1_a800_ckpt500_metrics.py
```

### 10. 一句话总结

本次 Stage1 alignment 训练在服务器2 A800 上稳定收敛：step 500 时 loss 降至约 0.21，grad_norm 回落到 2.78，训练状态健康；但由于单卡 global batch=64 且目标为 1000 step，step 500 只相当于扫过训练集约 16.5%，因此该 checkpoint 是稳定的中途结果，不代表全量 alignment 数据已完整训练一轮。



## 11. 可能原因

**1. 模型不是从零训练，已有语言建模能力。**

意思是：Qwen3-4B 本来就会预测文字，不是一个随机模型。它已经知道 `"Node 3"` 这种短文本怎么写，所以一开始 loss 下降很快。

**2. alignment 输出通常较短，格式相对固定。**

如果答案经常是：

```

```

```
Node 0
Node 1
Node 2
yes
no
A
B
```

那模型要学的输出形式比较简单。短答案的 token-level loss 容易降下来。

**3. teacher forcing 比真实推理更容易。**

训练时模型看到标准答案前缀；推理时没有。  
 所以训练 loss 低，不能直接说明真实生成也稳。

**4. loss 只反映当前 batch，不等于全量测试表现。**

日志里的 loss 是训练过程中局部 batch 的损失；测试集准确率要另外跑 evaluation 才知道。

