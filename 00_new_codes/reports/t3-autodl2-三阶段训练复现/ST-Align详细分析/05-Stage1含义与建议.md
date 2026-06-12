# 05 ST-Align 的 Stage 1 含义与建议

修改：2026-06-12  
前置：[02-数据解剖](02-数据解剖.md) · [03-样本形态与生成](03-样本形态与生成.md) · [04-探针交叉论证](04-探针交叉论证.md)

---

## 1. ST-Align 在 pipeline 里到底干什么

论文叙事：Stage 1 **alignment pretraining**——让文本、图描述与 **time series embedding** 对齐。

数据事实（153,700 条全量）：

| 实际监督 | 占比 |
|----------|------|
| 不读 TS 也能答 | **~58%**（46% 图 + 12% metadata） |
| 必须读 TS | **~42%**（38% 参数反演 + 4% 模式分类） |

因此 Stage 1 **名义**是 TS 对齐，**实际**混合了大量 **文本图结构 QA + 低熵 metadata QA + 高难度参数反演**。LoRA 探针表明：**前两者的饱和不能代表第三者的失败被掩盖**（loss 仍降）。

---

## 2. 设计缺陷链（用于确认「数据集有问题」）

```text
(1) Graph Structure 明文写入 input
      → 46% 样本不要求 ts_encoder 参与
      → spatial 探针快速饱和

(2) 同场景 ~363 题共享同一 timeseries 与前缀
      → 19 种 question stem
      → 梯度冗余；loss 可降而 TS 能力不升

(3) 25% 题为正弦 A/ω/φ 反演
      → 需从 patch 序列估计 SDE 参数
      → 2000 step 探针 0/15；失败为高频常数

(4) temporal 内 90% 为数值反演、仅 10% 为模式分类
      → 探针仅 evolution 4/5、kappa 2/5 有信号
      → 「alignment」阶段目标与能力要求不匹配

(5) 与 Stage 2 任务族断层
      → 无 Options / CoT / 四任务推理
      → Stage 1 终点无法预测 Stage 2 冷启动质量
```

**结论（可写进复现决策）**：ST-Align **作为格式 warm-up 合理**；**作为「教会模型读时序再进推理」的数据集，结构失衡且有探针证据**。你「大致知道有问题、效果不好」的判断 **被数据 + 探针交叉印证**。

---

## 3. 优点（保留，避免全盘否定）

- Gold 无标注噪声（仿真 ground truth）。
- 与下游 **input 协议一致**（专家前缀、`<ts>`、edge list）。
- 规模足够大，适合短程 SFT。
- 论文 ablation 仍显示 alignment **对最终 STReasoner 有互补**——问题在 **本阶段目标是否被误读为 temporal 已学好**。

---

## 4. Stage 1 训练决策建议（方向性，非命令）

以下 **不写具体 shell**；供服务器 2 全参方案（report 25）或后续改数据时参考。

### 4.1 探针与停训

- **必面板**：30 条健康 + **40 条 temporal balanced**（8 题型 × 5 条），与 report 17 对齐。
- **停训条件**：看 temporal 面板，**不看** train_loss 单曲线。
- **禁止**用 6 条 spatial 样例或 spatial 10/10 宣告 Stage 1 成功。

### 4.2 采样 / 权重（若可改 dataloader）

| 能力组 | 全库占比 | 建议方向 |
|--------|----------|----------|
| graph_text_only | 46% | **降权或每场景仅保留 1–2 题**；否则 ts_encoder 梯度被稀释 |
| scenario_metadata | 12% | 可保留但 **不必与 TS 联合 oversample** |
| ts_pattern_classification | 4% | **适度 oversample**（探针唯一部分成功区） |
| ts_numeric_inversion | 38% | **分层**：kappa / evolution 优先；A/ω/φ **降权或 Stage 1 后期再上** |

原则：**先让模型在「真读 TS」的低熵任务上稳定，再接触正弦反演**——curriculum，而非 15 万条均匀 shuffle。

### 4.3 场景去冗余

- 423 场景 × median 310 题 → 同前缀重复训练。
- 可选策略：每场景每 epoch **最多 K 条**（如 K=20–40），或按题型分层抽样，避免 363 倍重复。

### 4.4 对「官方 1000 step」的预期管理

- 1000 step × 8 卡 batch 仍 **远小于** 153,700 条一遍 epoch。
- 在 **当前数据设计下**，不应期望 Stage 1 结束即可 **数值反演 A/ω/φ**——探针已说明即使用 2000 step LoRA 亦 **0/5**。
- Stage 1 更现实的合格线：**格式正确 + evolution/kappa 部分可读 + spatial 不证明 TS**。

### 4.5 与 Stage 2 的衔接

- ST-Align **不覆盖** entity / etiological / correlation / forecasting 语义。
- Stage 1 效果差 **不必然** 杀死 Stage 2，但会 **抬高 CoT 冷启动难度**——应用 **Stage 2 探针** 单独验收，勿从 Stage 1 loss 外推。

---

## 5. 不建议的方向

- **不建议** 仅凭继续堆 Stage 1 step 解决 temporal 6/40（1500→2000 已平台）。
- **不建议** 用全库 loss 或 spatial 准确率替代 temporal balanced 面板。
- **不建议** 在未改数据权重的情况下，把 Stage 1 失败 solely 归因于 LoRA / 4B 规模（full FT 资源问题另论，见 report 15/25）。

---

## 6. 开放问题（本次未做）

- `alignment_test.jsonl` 评测协议与 train 分布是否一致（未下载分析）。
- 8×A100 full FT 1000 step 的 **同款 temporal 面板**（无本地 log 则无法对比）。
- 改权重 / curriculum 后的 **对照实验**（需另开实验，非本文范围）。

---

## 7. 一句话收束

**ST-Align 把近六成监督放在「不读时序也能答」的题上，又把最难的参数反演放在近四成「必须读时序」的题里；LoRA 探针显示 loss 下降与 temporal 读取能力脱钩——这是数据集结构问题，不是单靠调参能消掉的噪声。**

返回索引：[01-索引.md](01-索引.md) · 延伸：[06-为何训完仍无法复述时序.md](06-为何训完仍无法复述时序.md)
