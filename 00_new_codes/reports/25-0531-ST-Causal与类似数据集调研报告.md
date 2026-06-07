# 25 — ST-Causal 与类似数据集调研报告

> 对应 `15-0531晚上实验.md` §5「数据集调研」。  
> 调研时间：2026-05-31。主要来源：STReasoner 论文（arXiv:2601.03248）、HuggingFace `Time-HD-Anonymous/ST-Bench`、CausalRivers 官网与 GitHub、以及 Time-MQA / TSRBench / STARK 等公开资料。

---

## 1. 核心结论（先读）

| 问题 | 结论 |
|---|---|
| **ST-Causal 是否对应论文 §5.2？** | **是。** 代码中的 `ST-Causal/causal.jsonl` 即论文 *Zero-Shot Results on Real-World Data*（§5.2）基于 **CausalRivers** 构造的 real-world zero-shot causal QA 评测集。 |
| **与 ST-Test 的关系** | **独立 split。** ST-Test 四类（entity / etiological / correlation / forecasting）是合成 ST-Bench 主评测；ST-Causal **不参与** Stage 1–3 训练注册（不在 ST-CoT / ST-RL 中），仅 zero-shot 外推评测。 |
| **命名** | 仓库与 HF 均为 **ST-Causal**（Causal），不是 Casual。 |
| **其他论文是否用过 ST-Causal？** | **截至 2026-05-31，仅 STReasoner 原文 Table 2 报告该 QA 集结果**；CausalRivers 本体被大量因果发现工作使用，但「CausalRivers → LLM 因果 QA」这一转化目前几乎只为 STReasoner 服务。 |
| **本仓库能否直接跑** | 推理入口已支持 `--task reasoning_causal`；需 `python download_dataset.py` 下载 `ST-Causal/causal.jsonl`。注意 `evaluation/evaluate.py` 里 `reasoning_causal` 的默认 dataset 路径与推理脚本不一致（见 §4.3）。 |

---

## 2. ST-Causal 与论文 §5.2 的对应关系

### 2.1 论文怎么说的

论文 §5.2（Table 2）描述：

1. 以 **CausalRivers**（Stein et al., ICLR 2025）的因果真值图为依据；
2. 若存在有向边 \(A \to B\)，则定义「\(A\) 对 \(B\) 有 causal effect」；
3. 对每条这样的边生成一条 QA：**询问 node A 是否对 node B 有因果效应**；
4. **严格 zero-shot**：不在该数据上微调；
5. Table 2 结果：GPT-5.2 (Text) 22.32%、Claude-4.5-Sonnet 83.18%、**STReasoner (TS Encoder) 98.82%** ACC。

论文摘要亦明确：ST-Bench 含四类合成任务，**另附 real-world dataset for zero-shot evaluation**——即 §5.2 这一套，而非 ST-Test 本身。

### 2.2 代码与 HF 数据中的落点

| 层级 | 路径 / 入口 |
|---|---|
| 数据文件 | `data/ST-Bench/ST-Causal/causal.jsonl`（HF: [ST-Bench/ST-Causal](https://huggingface.co/datasets/Time-HD-Anonymous/ST-Bench/tree/main/ST-Causal)，约 14MB） |
| 训练注册 | `data/dataset_info.json` → `"causal"`（**仅注册，官方三阶段脚本未用于训练**） |
| 推理 | `inference/inference_tsmllm_vllm.py` → `reasoning_causal` |
| 评估 | `evaluation/evaluate_qa.py` 将 `reasoning_causal` 与 entity/etiological/correlation 一样走 **多选 accuracy** |
| Prompt 后缀 | `inference/prompt.json` → `reasoning_causal`（与 correlation 相同的多选 `<answer>` 格式） |

### 2.3 实测样本结构（2026-05-31 自 HF 拉取 `causal.jsonl`）

| 字段 | 值 |
|---|---|
| 样本数 | **1183** |
| 字段 | `input`, `output`, `timeseries`, `category`（与 ST-Test 一致） |
| `category` | 全部为 `causal_river` |
| 题干模板 | 固定为 2 节点、序列长 **365**、问句 *「Is there a causal effect from node 0 to node 1?」*，选项 A. Yes / B. No |
| `Graph Structure` | 固定 `Node 0->Node 1; Node 1->Node 0`（双向边，与 CausalRivers 中常见上下游关系一致） |
| `output` | **全部为 `<answer>A</answer>`**（1183/1183） |
| `timeseries` | 1183 条 **互不重复** 的 2 节点 × 365 点河流流量序列（来自 CausalRivers 不同测站对） |

**解读：** ST-Causal 是「**固定因果图 + 固定问法 + 变化的真实时序**」的外推测试：每条样本换一对 CausalRivers 测站时序，但标签与图结构模板相同。这与论文「对每条有向边生成一个 QA instance」一致，但 released 版本在问句与边上高度模板化；**不含负样本（B. No）**，评测 discriminability 偏弱（见 §6）。

---

## 3. CausalRivers 原始数据 vs ST-Causal

### 3.1 CausalRivers 原生格式

来源：[causalrivers.github.io](https://causalrivers.github.io/) · [GitHub CausalRivers/causalrivers](https://github.com/CausalRivers/causalrivers)

| 组件 | 格式 |
|---|---|
| 因果图 | 3 套 **NetworkX** 图（Bavaria / East Germany / Flood 等） |
| 时序 | **CSV**，15 分钟分辨率，2019–2023，666+494 测站 |
| 元数据 | 节点 ID 与测站对齐的 metadata 表 |
| 典型用途 | **因果发现算法** benchmark（VAR、PCMCI、CP 等），leaderboard 提交边预测 |

**没有** LLM 所需的 `input` / `<ts><ts/>` / 自然语言选项。

### 3.2 STReasoner 的构造链路（论文 + 数据反推）

```text
CausalRivers 真值图 + 河流 CSV
    → 采样节点对 / 子图
    → 提取对应时序窗口（365 点）
    → 写入 Graph Structure 文本 + <ts> 占位符
    → 生成 Yes/No 选择题（release 版均为 Yes）
    → ST-Causal/causal.jsonl
```

本仓库 **不含** 上述构造脚本；只有成品 `causal.jsonl`。

---

## 4. 本代码库支持的数据格式

### 4.1 ST-Bench 时序版（ST-Causal / ST-Test 同属此类）

每行 JSONL 最少：

| 字段 | 说明 |
|---|---|
| `input` | 含 `<ts><ts/>`、`Graph Structure: ...`、题干与选项 |
| `timeseries` | 与 `<ts>` 数量一致的嵌套 float 列表 |
| `output` | Gold：多选 `<answer>X</answer>` 或 forecasting 数值串 |
| `category` | 任务分流（ST-Causal 为 `causal_river`） |

推理：`input` + `timeseries` → vLLM TS 模型；评估：读 `output` 作 gold，tag-first 解析 `<answer>`。

### 4.2 变体 split（迁移其他数据集时需对齐）

| Split | 字段差异 | 代码支持 |
|---|---|---|
| ST-*-Text | 无 `timeseries` | `train_stage2_only_text.sh` 等 |
| ST-*-Image | `images` 代替/补充 TS | Qwen3-VL 脚本 |
| ST-Align / CoT / RL / Test | 同三字段 | 主训练与 ST-Test 评测 |

### 4.3 跑 ST-Causal 的注意事项

**推理（正确默认路径）：**

```bash
python inference/inference_tsmllm_vllm.py \
  --task reasoning_causal \
  --model_path <STReasoner-8B> \
  --max_tokens 6144
# 默认 dataset: data/ST-Bench/ST-Causal/causal.jsonl
```

**评估（需显式 `--dataset`，否则路径错误）：**

- `inference_tsmllm_vllm.py` → `data/ST-Bench/ST-Causal/causal.jsonl` ✓  
- `evaluation/evaluate.py` 内置 `DEFAULT_TASK_CONFIG` → `data/reasoning/causal.jsonl` ✗（与 ST-Bench 不一致）

建议：

```bash
PYTHONPATH=. python evaluation/evaluate.py \
  --task reasoning_causal \
  --dataset data/ST-Bench/ST-Causal/causal.jsonl \
  --exp_path <exp_dir>
```

**与 ST-Test 差异：** ST-Causal 序列长 **365**（ST-Test 常见 32–52），patch token 更多，显存与 `max_tokens` 压力更大；建议与 ST-Test 6144 设置对齐并监控 OOM。

---

## 5. 类似 / 相关数据集一览

按与 STReasoner **任务形态接近度** 排序。

### 5.1 高度相关（时空图 + 时序 + QA）

| 数据集 | 规模 | 模态 | 与 ST-Causal 关系 | 格式差异 |
|---|---:|---|---|---|
| **ST-Bench / ST-Test**（同仓库） | 3273 test | TS + 图 prompt | 同作者、同 JSONL 协议；**合成** vs **真实** | 任务更丰富（四类推理 + forecasting）；序列更短 |
| **TSRBench**（Yu et al., 2026） | 4125 | T / V / T+V | 含 **Causal Discovery** 等 15 任务 | 通用 benchmark JSON；**无** `<ts><ts/>` encoder 管线；需自建 adapter |
| **STReasoner correlation_test** | 1592 | TS + 图 | T3 空间相关/因果传播，与 ST-Causal 任务最近邻 | 合成数据；多跳/多选更复杂 |

### 5.2 时序 QA / 推理（无图结构或弱空间）

| 数据集 | 规模 | 特点 | 迁移难度 |
|---|---:|---|---|
| **Time-MQA / TSQA**（Kong et al., ACL 2025） | ~200k | 12 域、forecasting/imputation/QA 等 | 纯文本模板 `{Question}{Answer}`；需补 `timeseries` + 图或走 Text split |
| **ChatTS**（Xie et al.） | ~2.2k | 情境化 TS 问答 | 无 Graph Structure；ChatTS 自有格式 |
| **TSAQA**（2026） | ~210k | 分析型 QA | 选择题/开放题；非 ST-Bench 字段 |
| **MTBench**（Chen et al.） | ~42k | 偏 forecasting QA | 两域；无空间图 |
| **Time-R1 / Time-MQA-7B 训练集** | — | TS 推理 LM | 模型侧 benchmark，非直接 JSONL 兼容 |

### 5.3 空间 / 时空（偏地理或 CPS，非 TS encoder）

| 数据集 | 规模 | 说明 |
|---|---:|---|
| **STBench**（LwbXc, 2024, [GitHub](https://github.com/LwbXc/STBench)） | 60k+ | 地理时空 **纯文本** QA；与 STReasoner 的 ST-Bench **同名不同物** |
| **STARK**（NeurIPS 2025） | 14552 | CPS 时空推理；传感器多模态；[HF prquan/STARK_10k](https://huggingface.co/datasets/prquan/STARK_10k) |
| **STQAD**（2024） | 10k | 时空知识图谱 QA；KG 嵌入而非 raw TS |

### 5.4 因果（非时序 LLM QA）

| 数据集 | 说明 |
|---|---|
| **CausalRivers**（ICLR 2025） | ST-Causal 的**上游**；因果发现 leaderboard，非 QA |
| **Causal Judgment / BBH** | 故事型因果判断，无 multivariate TS |
| **CausalProbe** 等 | LLM 因果推理文本 benchmark |

---

## 6. 可行性与效果预测

### 6.1 直接跑 ST-Causal（推荐优先级：高）

| 维度 | 评估 |
|---|---|
| **数据获取** | `download_dataset.py` 一键；本地当前 **未下载** ST-Causal |
| **代码兼容** | **零格式转换**；与 ST-Test 同一推理/评估链路 |
| **工作量** | 低：单 task 一次 inference + evaluate |
| **预期 ACC** | 论文 **98.82%**；因 **100% gold 为 A**，随机/恒 A 基线亦 100%，实际需看 **模型是否稳定输出 `<answer>A</answer>`** 及 **coverage** |
| **风险** | ① 全正样本，无法测「拒因果」；② 365 长度 OOM；③ evaluate 默认路径需手动修正 |

### 6.2 从 CausalRivers 自建扩展集（优先级：中）

| 维度 | 评估 |
|---|---|
| **价值** | 可加入 **无边 / 反事实边 → B. No**、多节点子图、不同问法，提升评测可信度 |
| **工作量** | 高：需图采样 + CSV 对齐 + prompt 模板（可参考 ST-Causal 样例） |
| **预期** | ACC 应 **低于** 论文 98.82%；更接近真实 zero-shot 泛化 |
| **代码** | 构造后仍用 `input/output/timeseries` 三字段即可接入 |

### 6.3 迁移 TSRBench / Time-MQA 等（优先级：低–中）

| 数据集 | 可行性 | 预期效果 |
|---|---|---|
| TSRBench Causal Discovery 子任务 | 需写 **格式转换 + 可选图 prompt**；语义与 ST-Causal 最近 | 中等；可横向对比「专用 TS-LM vs 通用 LLM」 |
| Time-MQA 开放推理子集 | 可抽子集转 Text 或补 TS 列 | STReasoner 在合成 ST-Test 上已强；开放域可能 **明显低于** ST-Test |
| STBench (LwbXc) | 仅文本，走 ST-*-Text 或纯 LLM | **不能**测 TS encoder；与 STReasoner 定位不同 |
| CausalRivers 原生 CD | 输出边集合，非 QA | 需另写 metrics（F1/SHD），**不能**直接用 `evaluate_qa.py` |

---

## 7. ST-Causal / CausalRivers 的文献使用情况

### 7.1 CausalRivers（上游）

- **发表：** ICLR 2025（[OpenReview wmV4cIbgl6](https://openreview.net/forum?id=wmV4cIbgl6)）。
- **使用者：** 因果发现社区（VAR、PCMCI、DYNOTEARS、Causal Pretraining 等），见 [experiments/causal_discovery_zoo](https://github.com/CausalRivers/experiments/tree/main/causal_discovery_zoo) 与官网 leaderboard。
- **任务：** 从多元时序 **发现因果图**，指标为结构误差等——**不是**自然语言 QA。

### 7.2 ST-Causal QA（STReasoner 构造）

- **首次系统使用：** STReasoner（arXiv:2601.03248, ACL 2026）§5.2 Table 2。
- **其他论文：** 公开检索 **未发现** 第二篇在同一 QA 转化集上报告结果；TSRBench 等更新的 TS 推理 benchmark **未** 声明复用 ST-Causal。
- **效果归纳：**
  - STReasoner **98.82%** >> GPT-5.2 **22.32%** >>（Claude 83.18% 介于其间）；
  - 说明 **TS encoder + 图 prompt** 对该模板化 real-world 任务极强；
  - 但 **Claude 83% vs GPT 22%** 也提示：部分高性能可能来自 **题干/选项理解 + 图文本**，需结合 w/o graph 消融（类似 ST-Test Figure 6）才能分离「看时序」的贡献。

### 7.3 与 ST-Test T3（correlation）的关系

- **ST-Test correlation**：合成、多跳、四选一，测空间相关/传播推理。
- **ST-Causal**：真实河流、二选一 Yes/No、固定图模板。
- 二者共享 **「Graph Structure + multivariate TS + 多选」** 范式；ST-Causal 可视为 **real-world、简化版、偏直接因果边** 的外推测试。

---

## 8. 建议的后续实验顺序

1. **下载并 smoke test**：`reasoning_causal`，`max_samples=4`，确认 365 长度 + 6144 tokens 不 OOM。  
2. **全量 1183 条 + tag-first evaluate**：记录 accuracy、coverage、恒 A 基线对比。  
3. **w/ graph vs w/o graph**（同 ST-Test 实验设计）：验证 real-world 上是否仍依赖 Graph Structure。  
4. **（可选）** 从 CausalRivers 构造含 **No** 标签的子集，避免评测饱和。  
5. **（可选）** 与 ST-Test correlation 子集对比 error case，分析合成→真实的 gap。

---

## 9. 参考链接

| 资源 | URL |
|---|---|
| STReasoner 论文 | https://arxiv.org/abs/2601.03248 |
| ST-Bench (HF) | https://huggingface.co/datasets/Time-HD-Anonymous/ST-Bench |
| STReasoner 代码 | https://github.com/LingFengGold/STReasoner |
| CausalRivers | https://causalrivers.github.io/ |
| CausalRivers 论文 | https://arxiv.org/abs/2503.17452 |
| Time-MQA / TSQA | https://huggingface.co/datasets/Time-MQA/TSQA |
| TSRBench | https://tsrbench.github.io/ |
| STBench (LwbXc, 地理时空) | https://github.com/LwbXc/STBench |
| STARK | https://openreview.net/forum?id=zRhO4hizR8 |

---

## 10. 一句话收束

**ST-Causal 就是论文 §5.2 基于 CausalRivers 构造的 real-world zero-shot causal QA；与本仓库 ST-Test 格式完全兼容、不参与训练，目前几乎仅 STReasoner 使用过并在 Table 2 报告 98.82% ACC。** 复现优先跑通 `reasoning_causal`；若要做有判别力的真实世界评测，需要在 CausalRivers 上扩展负样本与多样图结构，而不是仅重复现有 1183 条全-A 模板。
