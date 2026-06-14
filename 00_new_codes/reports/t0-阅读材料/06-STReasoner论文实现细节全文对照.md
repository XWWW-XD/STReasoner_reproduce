# STReasoner 论文实现细节 · 全文对照

修改：2026-06-13  
论文：[arXiv:2601.03248v3](https://arxiv.org/abs/2601.03248)（ACL 2026 Main，最后修订 2026-04-22）  
本地全文：`paper/STReasoner_ACL_2026.txt`（与 v3 PDF 对齐的提取文本）  
用途：复现三阶段训练时，按 **论文出处 → 数值 → 本仓库脚本** 逐项对照。

---

## 1. 论文与附录地图

| 附录/章节 | 页码（PDF） | 内容 |
| --- | --- | --- |
| **§3.3** | p.5 | 数据合成规模、ST-Align / ST-Bench 划分 |
| **§4.1** | p.5–6 | 模型架构（TS Encoder、patch、输入格式） |
| **§4.2** | p.6 | 三阶段训练流程（Align / CoT / S-GRPO） |
| **§5** | p.6–9 | 实验、基线、消融、RL 曲线 |
| **Appendix A** | p.15 | Related Work |
| **Appendix B.1–B.3** | p.16–18 | 数据合成、人工质检、QA 数据集生成 |
| **Appendix C** | p.18 | Task-Grounded Reward（format + task + λ） |
| **Appendix D** | p.18–19 | **Implementation Details（训练超参主表）** |
| **Appendix E** | p.19 | 五次 run 的 95% CI 结果 |
| **Appendix F** | p.19 | RL 1 epoch → 2 epoch scaling |
| **Appendix G** | p.20+ | 合成数据样例 Table 5 |
| **Appendix H** | 正文 §5.5 引用 | Case Study |
| **Appendix I** | B.3 / §3.2 引用 | ST-Bench QA 生成 prompt |
| **Appendix J** | B.3 / §3.2 引用 | ST-Align 模板 |
| **Appendix K** | B.1 引用 | 多 Agent 数据合成 prompt |
| **Appendix L** | §5.4 引用 | 空间推理使用率评估 prompt |

---

## 2. Appendix D · Implementation Details（原文结构化）

> 出处：**Appendix D**，`paper/STReasoner_ACL_2026.txt` 约 L2186–L2207（PDF p.18–19）

### 2.1 硬件与基座模型

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| Base model | **Qwen3-8B** (Yang et al., 2025a) | Appendix D 首句 |
| GPU | **8 × NVIDIA A100 80GB** | Appendix D |

### 2.2 Stage 1 · Alignment SFT

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| 数据 | **ST-Align** | Appendix D |
| 训练量 | **1,000 steps** | Appendix D |
| 框架 | **LlamaFactory** (Zheng et al., 2024) | Appendix D |
| LR scheduler | **cosine** | Appendix D |
| Warmup ratio | **0.2** | Appendix D |
| `per_device_train_batch_size` | **2** | Appendix D |
| `gradient_accumulation_steps` | **32** | Appendix D |
| LLM 学习率 | **1×10⁻⁵** | Appendix D |
| Time series encoder 学习率 | **1×10⁻⁵**（与 LLM 相同） | Appendix D |

**隐含全局 batch（论文未写，可推算）：**  
8 GPU × 2 × 32 = **512 samples / optimizer step**（与 `00_new_codes/guides/streasoner_code_reading/10_sft_training_flow.md` 一致）。

**§4.2 / §3.3 补充（非 Appendix D，但属 Stage 1 定义）：**

| 项 | 论文描述 | 出处 |
| --- | --- | --- |
| 目标 | 大规模 alignment，对齐文本与时间序列 embedding | §4.2 Stage 1 |
| ST-Align 规模 | **153,700** QA pairs | §3.3 |
| 三类问题 | Temporal / Spatial / Spatio-Temporal Characters | §3.2 ST-Align、Appendix J |

### 2.3 Stage 2 · CoT SFT

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| 数据 | **ST-CoT**（论文 Appendix D 写 “ST-CoT”；§4.2 称由 ST-SFT rejection sampling 得到） | Appendix D、§4.2 |
| 训练量 | **400 steps** | Appendix D |
| 框架 | LlamaFactory（与 Stage 1 相同超参表） | Appendix D |
| LR scheduler | cosine | Appendix D |
| Warmup ratio | **0.2** | Appendix D |
| Batch | per_device=**2**, grad_accum=**32** | Appendix D |
| 学习率 | LLM & TS encoder 均为 **1×10⁻⁵** | Appendix D |

**§4.2 Stage 2 数据构造（Implementation 相关）：**

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| 源问题集 | ST-SFT（ST-Bench 子集） | §4.2、Appendix B.3 |
| Rejection sampling | 每题用 **Claude-4.5-Sonnet** 采样 **5** 个候选回答 | §4.2 |
| 保留条件 | 仅保留 **最终答案正确** 的 reasoning trajectory | §4.2 |
| 四类任务 | T1 Etiological / T2 Entity / T3 Correlation / T4 Forecasting | §3.2、Table 4 |

**Table 4 · ST-CoT 各任务样本数（Appendix B.3, PDF p.17）：**

| Task | ST-CoT | ST-RL | ST-Test | Total |
| --- | ---: | ---: | ---: | ---: |
| T1 Etiological | 1,576 | 405 | 207 | 2,188 |
| T2 Entity | 4,974 | 2,451 | 1,194 | 8,619 |
| T3 Correlation | 11,124 | 3,231 | 1,592 | 15,947 |
| T4 Forecasting | 650 | 506 | 280 | 1,436 |
| **Total** | **18,324** | **6,593** | **3,273** | **28,190** |

### 2.4 Stage 3 · S-GRPO RL

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| 数据 | **ST-RL** | Appendix D |
| 训练量 | **1 epoch**（非固定 step 数） | Appendix D |
| 框架 | **EasyR1** (Zheng et al., 2025) + **verl** (Sheng et al., 2025) | Appendix D |
| Rollout group size **G** | **8** | Appendix D |
| Spatial reward **α** | **0.1** | Appendix D |
| Spatial tolerance **β** | **0.8** | Appendix D |
| LR scheduler | **linear** | Appendix D |
| Warmup ratio | **0.2** | Appendix D |
| `rollout_batch_size` | **128** | Appendix D |
| 学习率 | **1×10⁻⁷** | Appendix D |

**1 epoch 对应多少 step（论文未写 step 数，可推算）：**

- ST-RL 总样本 **6,593**（Table 4）  
- `rollout_batch_size=128` → 约 **⌈6593/128⌉ = 52** 个 RL step/epoch  
- 本仓库 README 合并 checkpoint 为 **`global_step_51`**（`README.md` L133–138），与上式一致。

**Appendix F · RL scaling（PDF p.19）：**

| 项 | 论文结论 | 出处 |
| --- | --- | --- |
| 对比 | **1 epoch vs 2 epoch** | Appendix F、Figure 11 |
| 现象 | 2 epoch 早期 accuracy reward 更高，后期饱和且更不稳定 | Appendix F |
| 主实验默认 | **1 epoch** | Appendix D |

---

## 3. Appendix C · Reward 设计（Stage 3 必备）

> 出处：**Appendix C**，L2096–L2184

| 项 | 公式/值 | 说明 |
| --- | --- | --- |
| Format reward | `r_format = I(valid output format)` | 必须 `<think>...</think><answer>...</answer>` |
| 选择题 task reward | `rtask=1` iff 预测选项等于 y | 离散标签 |
| 预测 task reward | 相对误差均值，ε=**10⁻⁹** | T4 Forecasting |
| 长度匹配 bonus | 预测序列长度与 GT 完全一致时 **+0.1**，clip 到 [0,1] | Appendix C |
| 组合 | **`r = (1−λ) r_task + λ r_format`** | — |
| **λ** | **0.5**（所有实验固定） | Appendix C |

**§4.2 · S-GRPO spatial reward（与 Appendix D 的 α、β 对应）：**

对同一问题，有图 prompt 与无图 prompt 各生成 G 个回答。最终用于 advantage 的 reward：

- 若 **`r_sp > β · r_ns`**，则 **`R = r_sp + α`**
- 否则 **`R = r_sp`**

其中 `r_sp`、`r_ns` 含 accuracy + format（Appendix C）。**G=8，α=0.1，β=0.8**（Appendix D）。

策略损失含 GRPO clip 与 **KL 惩罚** `−β_KL D_KL(π_θ ∥ π_ref)`（§4.2 公式；此处 β 为 KL 系数，与 spatial β **同名不同义**，Appendix D 只给出 spatial 的 α、β）。

---

## 4. §4.1 · 模型架构（训练前初始化相关）

> 出处：**§4.1**，L452–L477

| 项 | 论文描述 | 本仓库 |
| --- | --- | --- |
| TS 编码 | Patchify（Nie 2022）+ **5 层 MLP** Time Series Encoder | `modeling_qwen3_ts.py` |
| 输入格式 | `[Node1:⟨TS1⟩, Node2:⟨TS2⟩, …, Graph, Question]` | template + ChatTSPlugin |
| 归一化 | **Value-preserving normalization**（跟随 ChatTS） | `processing_qwen3_ts.py` `sp_encoding()` |
| Patch size | 正文未给数值 | 代码 **`patch_size=8`**（`inference_tsmllm_vllm.py` L50；checkpoint 内 config） |
| TS encoder 初始化 | 正文未写 | `initial_model.py`：**Xavier normal** 初始化 `ts_encoder` 权重 |

---

## 5. 数据管线（Implementation 上下文）

### 5.1 §3.3 + Appendix B（合成 → QA）

| 项 | 论文值 | 出处 |
| --- | --- | --- |
| 初始合成样本 | **1,200** pairs | §3.3 |
| 人工质检后保留 | **1,064** High Quality | Appendix B.2 |
| Train / Test 划分 | **80% / 20%**（按 spatio-temporal 样本） | §3.3 |
| ST-Align | 训练 split 生成，**153,700** QA | §3.3 |
| ST-Bench 划分 | **6 : 2 : 2** → ST-CoT / ST-RL / ST-Test | Appendix B.3 |
| 合成 LLM | **Claude-4.5-Sonnet** | Appendix B.1 |
| 图规模 | **3 / 5 / 10** nodes | Appendix B.1 |
| 领域 | 10 domains（Transportation, Energy, …） | Appendix B.1 |

### 5.2 Appendix B.3 · ST-Align 问题计数规则（摘要）

| 类别 | 计数规则 |
| --- | --- |
| Temporal | 每 node 每 drift pattern 一题；sinusoidal 额外问 A, ω, φ |
| Spatial | 每有序对 2 题（edge_relationship + indirect_connection）→ **2N²** |
| Spatio-temporal | **N + E + 2M**（node type / edge delay / modulation） |

---

## 6. 实验设置（非训练超参，但论文写明）

### 6.1 Appendix E · 置信区间（PDF p.19）

- **5 次独立 run**，报告 **95% CI**  
- 例：T1 ACC **95.39±0.74**，T2 **75.78±0.49**，T3 **87.20±0.47**，T4 MAE **65.61±0.06**

### 6.2 §5.4 · α 敏感性（Figure 5）

| α | 论文结论 |
| --- | --- |
| 0.2, **0.1** |  consistently strong（主实验取 **0.1**） |
| 1, 0.5, 0 | 至少一项指标变差 |

### 6.3 推理（论文正文未给默认 decoding；仓库实现）

论文 Table 1 报 **input tokens** 与 API 成本，**未在 Appendix D 写 temperature / max_tokens**。

本仓库 `inference/inference_tsmllm_vllm.py` 默认：

- `max_tokens=512`，`temperature=0.2`  
- 正式 ST-Test 实验报告建议 **`max_tokens=6144`**（见 `00_new_codes/reports/t0-阅读材料/01-pipeline数据通路.md`）

---

## 7. 本仓库官方脚本对照表

> 8B 主线：`scripts/qwen3-8b/`  
> 4B 同结构：`scripts/qwen3-4b-instruct/`

### 7.1 Stage 1 · `train_stage1.sh`

| 参数 | 论文 Appendix D | 本仓库脚本 | 一致？ |
| --- | --- | --- | --- |
| max_steps | 1000 | **1000** | ✅ |
| per_device_train_batch_size | 2 | **2** | ✅ |
| gradient_accumulation_steps | 32 | **32** | ✅ |
| learning_rate | 1e-5 | **1e-5** | ✅ |
| timeseries_sft_lr | 1e-5 | **1e-5** | ✅ |
| lr_scheduler_type | cosine | **cosine** | ✅ |
| warmup_ratio | **0.2** | **0.02** | ⚠️ **不一致** |
| 数据 | ST-Align | `alignment` | ✅ |
| 模板 | （Align 格式） | `STReasoner-Align` | ✅ |
| GPU 数 | 8 | `deepspeed --num_gpus 8` | ✅ |
| DeepSpeed | 未写 | ZeRO-3 `ds_config_3.json` | 补充 |
| cutoff_len | 未写 | **10000** | 补充 |
| fp16 | 未写 | **是** | 补充 |

### 7.2 Stage 2 · `train_stage1+2.sh`

| 参数 | 论文 | 脚本 | 一致？ |
| --- | --- | --- | --- |
| max_steps | 400 | **400** | ✅ |
| 其余 SFT 超参 | 同 Stage 1 | 同 Stage 1 | 同 warmup ⚠️ |
| 数据 | ST-CoT 四类 | `entity_cot,etiological_cot,correlation_cot,forecasting_cot` | ✅ |
| interleave | 未写 | **0.25×4** | 补充 |
| 起始 checkpoint | Stage 1 输出 | `./output/Qwen3-8B-stage1` | ✅ |
| 模板 | CoT 格式 | `STReasoner-CoT` | ✅ |

### 7.3 Stage 3 · `train_stage1+2+3_w_spatial.sh`

| 参数 | 论文 Appendix D | 脚本 | 一致？ |
| --- | --- | --- | --- |
| total_epochs | 1 | **1** | ✅ |
| rollout_batch_size | 128 | **128** | ✅ |
| G (rollout.n) | 8 | **8** | ✅ |
| actor lr | 1e-7 | **1e-7** | ✅ |
| lr_warmup_ratio | 0.2 | **0.2** | ✅ |
| α | 0.1 | `algorithm.spatial_reward_weight=**0.1**` | ✅ |
| β | 0.8 | 代码 `original_r > no_graph_r * **0.8**`（`ray_trainer.py` L487） | ✅ |
| enable_spatial_reward | S-GRPO | **true** | ✅ |
| 起始模型 | Stage 1+2 | `./output/Qwen3-8B-stage1+2` | ✅ |
| ST-RL 四文件 | 未列文件名 | `ST-Bench/ST-RL/*.jsonl` ×4 | ✅ |
| KL / clip 等 | 未写 | `config.yaml` 默认 `kl_coef=1e-2` 等 | 补充 |

### 7.4 三阶段步数一览（复现排期用）

| 阶段 | 论文训练量 | _optimizer step 含义 | 约步数 |
| --- | --- | --- | --- |
| **Stage 1** | 1000 steps | SFT optimizer step | **1000** |
| **Stage 2** | 400 steps | SFT optimizer step | **400** |
| **Stage 3** | 1 epoch | RL step（每 step 处理 rollout_batch 条 prompt） | **~51–52**（6593÷128） |

---

## 8. 复现时优先核对的差异点

1. **warmup_ratio**：论文 **0.2**，8B 脚本 **0.02**（差 10×）——若严格贴论文，应改脚本或确认作者实际 run 用的值。  
2. **Stage 3 用 step 还是 epoch**：论文只写 **1 epoch**；不要与 Stage 1/2 的 1000/400 混为一谈。  
3. **ST-CoT vs ST-SFT 命名**：Appendix D 写 ST-CoT；§4.2 先提 ST-SFT 再 rejection sampling 成 ST-CoT——指同一 Stage 2 数据管线。  
4. **§3.3 与 B.3 划分表述**：§3.3 写 ST-Bench 6:2 分 SFT/RL；B.3 写 **6:2:2** 含 ST-Test——以 **Table 4 数字** 为准。  
5. **推理超参**：不在 Appendix D；复现 Table 1 需另查实验脚本或联系作者。  
6. **4B / 14B**：论文主实验仅写 **Qwen3-8B**；仓库另提供 4B/14B 脚本，超参结构相同但需单独 smoke。

---

## 9. 快速索引：你要找的「多少步」

| 阶段 | 论文原文 | PDF 位置 |
| --- | --- | --- |
| Stage 1 | **1,000 steps** on ST-Align | Appendix D |
| Stage 2 | **400 steps** on ST-CoT | Appendix D |
| Stage 3 | **one epoch** on ST-RL（≈51 steps @ batch 128） | Appendix D + Table 4 推算 |

---

## 10. 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-06-13 | 初稿：通读 `paper/STReasoner_ACL_2026.txt` + 对照 `scripts/qwen3-8b/*`；Appendix A–L 地图 + D/C/B/F/E 全量 + 脚本 diff |
