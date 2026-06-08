# ST-Bench 数据集专篇：四任务入门 + 百问速查

> **范围**：只谈 ST-Bench **数据集本身**（字段、规模、标签、合成方式、划分、评测接口），不涉及模型结构、训练或优化细节。  
> 依据：论文 *STReasoner*（ACL 2026 Appendix B）、[HF ST-Bench](https://huggingface.co/datasets/Time-HD-Anonymous/ST-Bench)、`data_generation/`、`evaluation/`、`inference/`、本仓库 `PaperCases.jsonl` / `SmartTest.jsonl`。

---

## 0. 问题索引（Q1–Q100 → 章节）

| 问题 | 节 | 问题 | 节 |
|---:|:---|---:|:---|
| Q1–Q3 | §1 | Q51–Q54 | §8 |
| Q4–Q11 | §2 | Q55–Q67 | §9 |
| Q12–Q15 | §3 | Q68–Q77 | §10 |
| Q16–Q25 | §4 | Q78–Q85 | §11 |
| Q26–Q32 | §5 | Q86–Q99 | §12 |
| Q33–Q43 | §6 | Q100 | §13 |
| Q44–Q50 | §7 | | |

---

## 1. 定位与四任务设计（Q1–Q3）

**Q1. ST-Bench 的整体定位？**  
**时空推理 benchmark**，不是纯 time series 预测集。每条样本 = 多节点数值时序 + 有向图（文本 edge list）+ 自然语言题 + 标准答案；考察在图约束下读时序、做推理（场景/实体/传播/预测）。

**Q2. 为何四个任务而非只做 forecasting？**  
论文 §3.2：时空推理包含 **全局场景推断、节点角色识别、跨节点传播关系、上下文预测** 四类能力；只做 forecasting 无法覆盖 QA 式空间因果与语义识别。四任务共用同一批合成场景，从不同角度出题。

**Q3. 四任务在数据集层面对应什么问题？**

| 代码 `category` | 论文 | 数据集层问题类型 | 答案形式 |
|---|---|---|---|
| `etiological` | T1 | 全图时序+拓扑 → **系统级场景**是什么 | 四选一 |
| `entity` | T2 | 指定节点 → **(name, description) 角色**是什么 | 四选一 |
| `correlation` | T3 | 指定节点对+时间窗 → **传播关系类型**（直连/多跳/无关系等） | 四选一 |
| `forecasting` | T4 | 给定 context+历史窗 → **目标节点未来 k 步数值** | 浮点序列 |

---

## 2. 输入格式、字段与表示（Q4–Q11）

**Q4. 四任务输入格式是否完全一致？**  
**前缀一致**，**题干与答案格式不同**：

- **相同**：`You are a spatial temporal analysis expert.` + 每节点 `Node i time series with length of L: <ts><ts/>` + `Graph Structure: Node a->Node b; ...` + `please analyze ... question:`
- **不同**：etiological 固定场景问句；entity 带目标节点；correlation 带节点对+`time steps a-b (1 time step = ...)`；forecasting 带 context、`predict ... next k steps`、`Historical observation window: a-b`
- **forecasting 额外**：`timeseries` 常截断到预测起点之前（见 `generate_reasoning_forecasting_QA.py`），序列长度可短于原始仿真全长

**Q5. 每条样例包含哪些字段？**

| 字段 | ST-Test 官方 | 说明 |
|---|---|---|
| `input` | ✅ | 完整文本 prompt（含选项时已在末尾） |
| `timeseries` | ✅ | `List[List[float]]`，外层=节点，内层=该节点序列 |
| `output` | ✅ | gold |
| `category` | ✅ | `entity` / `etiological` / `correlation` / `forecasting` |

**非官方、子集自建**：`sample_id`、`source_file`、`original_line_index`、`paper_case_id`、`task`、`dataset_repo`、`length_metrics` 等（见 `PaperCases.jsonl`、`SmartTest.jsonl`）。

**ST-CoT 额外**（训练用）：`cot_attempt`、`cot_mae`、`cot_normalized_mae`、`idx` 等；**ST-Align** 部分文件无 `category`。

**Q6. time series 如何表示？**  
- **存储**：每节点一条 **完整或截断后的浮点历史序列**（非单窗口切片文件字段）  
- **输入文本**：`<ts><ts/>` 占位；推理时由 processor 替换为 patch embedding  
- **题面**：correlation/forecasting 的 **时间窗口** 写在 question 里，模型需从完整序列中读对应 steps

**Q7. graph structure 如何表示？**  
**自然语言有向 edge list**，非邻接矩阵字段：  
`Graph Structure: Node 0->Node 2; Node 1->Node 2; ...`  
合成时由 `relationships` / `base_adjacency` 导出（`generate_reasoning_QA.py::build_graph_structure_description`）。

**Q8. node / entity / variable 含义？**  
- **Node**：图顶点编号（0-based），与 `timeseries[i]` 一一对应  
- **Entity**（任务名）：指「空间实体识别」——问某 **Node** 的语义角色，不是独立实体 ID 字段  
- **Variable**（生成 pipeline）：仿真里节点的物理/业务变量名（metadata），写入 entity 选项的 name/description；**数据集 JSONL 无单独 `variable` 列**  
→ 三者 **不是三个并列字段**；对外接口只有 **Node 编号 + 可选语义选项文本**

**Q9. 问题是自然语言还是模板？**  
**混合**：固定骨架（专家前缀、图描述、部分固定问句）+ **LLM 生成** 的选项/部分问句（Claude，见 Appendix B.3 / `data_generation/prompts/qa_generation/`）。  
- etiological：问句固定  
- entity：问句模板 + LLM 填四选项  
- correlation / forecasting：问句 largely LLM，带时间窗/节点占位

**Q10. 标准答案如何存储？**  
- QA 三任务：`"<answer>D</answer>"` 单字母  
- forecasting：`'[19.86, 19.97, 20.05]'` JSON 数组字符串（可无 `<answer>` 包裹；评测 parser 兼容两种）

**Q11. 选项存在哪里？**  
**嵌在 `input` 末尾**：`Options: A. ... B. ... C. ... D. ...`  
**无**独立 `options` 列；**不要**从 JSONL 拆改后再写回。

---

## 3. 四任务标签来源（Q12–Q15，Q72–Q77）

**Q12. etiological label 来源？**  
LLM 根据仿真 **macro scenario** 生成 4 条系统级场景摘要；第一条为真（与 `observation`  verbatim 一致），`output` 为对应字母。**不是**单独标注「原因节点 ID」或「传播路径字符串」——路径信息在时序+图中，label 是 **场景类型描述**。

**Q13. entity label 来源？**  
从仿真 metadata 取目标节点的 **真实 (name, description)** 作为正确选项；LLM 生成 3 个同领域错误角色。**识别的是节点角色/功能**，不是图外实体。

**Q14. correlation label 来源？**  
两类生成证据（代码 `direct_causal` / `indirect_causal`）：  
- **直连**：基于 `adjacency_modulation` 中单条边事件  
- **多跳**：串联多条边事件合成描述  
label = **关系类型/传播路径的自然语言描述** 的四选一，不是简单「相关/不相关」二分类。

**Q15. forecasting label 来源？**  
从 Network SDE 仿真轨迹 **直接截取** 目标节点未来 `prediction_length` 步的 raw 值（`generate_reasoning_forecasting_QA.py::_extract_prediction_values`），写入 `output`。**步数 k 因样本而异**（非全库固定 3）。

**Q72.** etiological 是否一定有明确原因？→ 合成场景设计有源汇/传播结构；label 是 **宏观场景匹配**，非逐事件因果链标注。  
**Q73.** correlation 是否区分 direct/indirect？→ **生成时分两类 prompt**；发布 JSONL **不存子类型字段**，只能从题干/选项语义判断。  
**Q74.** entity 候选是否同图？→ **是**，四选项均针对同一仿真图内节点角色。  
**Q75–Q76.** forecasting 预测窗/历史窗是否固定？→ **均不固定**；题面 `Historical observation window: a-b` 与 `next k steps` 逐条给出。  
**Q77.** 预测所有节点还是目标节点？→ **仅目标节点**（题面 `predict the value of node {id}`）。

---

## 4. 数据来源、合成与图结构（Q16–Q25）

**Q16. 是否区分 synthetic / real-world？**  
- **主 ST-Bench 四任务**：**全 synthetic**（1064 个高质量场景扩展为 28190 条 QA，Table 4）  
- **real-world**：论文 §5.2 **ST-Causal**（CausalRivers 因果图 QA），**不在**主四任务 benchmark 内

**Q17. synthetic 如何生成？**  
6-Agent **Network SDE pipeline**（`data_generation/demo_sts_sde.py`）：场景文本 → 结构化图+参数 → 时变邻接 → SDE 仿真 → 人工质检 1064 对 → LLM 批量出 QA（`generate_reasoning_QA.py` / `generate_reasoning_forecasting_QA.py`）。

**Q18. real-world 来自哪？**  
**ST-Causal**：CausalRivers（Stein et al., 2025）因果图上的「A 是否对 B 有因果效应」二选一 QA。

**Q19. 四任务是否都含 synthetic+real？**  
**四任务仅 synthetic**；real-world 仅 **ST-Causal** 子集。

**Q20. graph 真实还是合成？**  
**合成**；拓扑由 agent 生成，边权由 time-varying adjacency 控制。

**Q21. 节点数/边数/拓扑设置？**  
论文 Figure 10：**节点数约均匀覆盖 3 / 5 / 10**；边数随场景变；**无单一固定拓扑**。

**Q22. 是否有 directed graph？方向含义？**  
**有**。`Node i->Node j` = **i 向 j 的影响/传播方向**（如上游→下游、源→汇）。

**Q23. 是否有 weighted edge？**  
**仿真内部**有 `base_adjacency`（有向权重，默认边 0.1）及 **time-varying modulation**；**数据集文本不暴露权重数值**，只有边列表 + 时序后果。

**Q24. 是否有时间延迟？**  
**有**。pipeline 为每条边设 **propagation time lag**（Appendix B.1）；体现为时序上的 **滞后响应**，非单独 delay 字段。

**Q25. 是否有多跳关系？哪些任务用？**  
**有**。correlation 的 **indirect_causal** 子类专门考多跳；etiological/entity/forecasting 也可能需要沿图多跳读时序，但题型不总是显式问「路径」。

---

## 5. 规模与划分（Q26–Q32）

**Q26–Q27. 总量与各任务条数？**

| 划分 | etiological | entity | correlation | forecasting | 合计 |
|---|---:|---:|---:|---:|---:|
| ST-CoT | 1,576 | 4,974 | 11,124 | 650 | 18,324 |
| ST-RL | 405 | 2,451 | 3,231 | 506 | 6,593 |
| **ST-Test** | **207** | **1,194** | **1,592** | **280** | **3,273** |
| **Total** | 2,188 | 8,619 | 15,947 | 1,436 | **28,190** |

底层 **1064** 个时空场景；QA 条数 >> 场景数（一场景多题）。

**Q28. train / val / test？**  
- **无独立 validation**  
- **ST-CoT + ST-SFT**：Stage 2 SFT（6 份，CoT:SFT ≈ 6:2 来自同池 80% 场景）  
- **ST-RL**：Stage 3 RL（2 份）  
- **ST-Test**：held-out 评测（2 份场景），**不与训练重叠**  
- **ST-Align**：Stage 1 对齐（另一套 QA，非四任务）

**Q29. ST-Test 与完整 ST-Bench？**  
**ST-Test = 官方测试 split**（3273 条）；「完整 ST-Bench」= ST-CoT + ST-RL + ST-Test 全部 QA（28190）+ 底层 1064 场景。

**Q30. paper cases 是什么？**  
论文 **Appendix H Table 6–9 展示样例**；本仓库 `PaperCases.jsonl` **4 条**，从 ST-Test **按行号抽取**（非独立 benchmark）。  
→ **展示 +  smoke 对照**，报告 full benchmark 时 **不能** 用 4 条代替 3273 条。

**Q31. 当前代码读哪份数据？**

| 场景 | 路径 |
|---|---|
| 官方全量评测 | `data/ST-Bench/ST-Test/{task}_test.jsonl` |
| 论文 4 case | `00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl` |
| 小样本 20 条 | `SmartTest.jsonl`（每任务 5 条，`seed=20260519`） |
| 分析附件 | `00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl`（推理输出+gold，非原始集） |

**Q32. 实验样本与论文 benchmark 一致吗？**  
- 跑满 `ST-Test` 四文件 → **与论文 Table 4 测试集一致**  
- 仅 paper_cases / SmartTest / 自定义 JSONL → **subset**，需在报告中明示（§12）

---

## 6. 文件格式与 prompt 字段（Q33–Q43）

**Q33. 文件格式？**  
**JSONL**（一行一条）；HF 亦按子目录多文件发布。无 CSV 主格式。

**Q34. task 字段如何标识？**  
- 官方：**`category`** = 四任务名  
- 混合 runner：`--task reasoning_entity` 等（`evaluation/evaluate.py`），与文件名对应  
- 子集可加冗余 `task` 字段

**Q35. question / prompt / instruction？**  
**统一用 `input`**。训练注册表 `data/dataset_info.json` 映射 `prompt → input`。**无**单独 `question` 列。

**Q36. answer / label / ground_truth？**  
**统一用 `output`**。训练映射 `response → output`。评测读 `sample["output"]` 为 gold。

**Q37. 是否已含 formatted prompt？**  
**不含** Output Format 后缀。后缀由推理脚本追加（`inference/prompt.json`）。

**Q38. 代码重构造 prompt 会改原题吗？**  
- **官方链路**：`input` 原样 + 追加 `\n\nOutput Format: ...`（**不改题干**）  
- **Stage 2.2 graph ablation 等**：会改 `input` 文本（删图），属 **实验变体**，非数据集原貌

**Q39–Q40. options 与 QA 答案格式？**  
三 QA 任务选项在 `input`；答案 **A/B/C/D 单字母**（`<answer>X</answer>`）。

**Q41. forecasting 答案格式？**  
JSON 浮点数组字符串；推理后缀要求 `<answer>[v1, v2, ...]</answer>`。

**Q42–Q43. 是否有 explanation / CoT？**  
- **ST-Test**：**无** gold reasoning chain  
- **ST-CoT**：`output` 为 **模型 rejection sampling 选出的长 CoT 响应**（`generate_cot.py`），**非人工标注**；带 `cot_mae` 等质量字段

---

## 7. 长度、规模与复杂度（Q44–Q50）

**Q44–Q45. 样例长度与最长任务？**  
- 序列长度：**48 ~ 数千 steps**（论文 Fig 8）  
- **correlation** 往往最长（节点多、序列长、题干含时间窗说明）；**etiological** 题干相对短  
- `input` 字符数随节点数线性增（每节点一句 + 图边）

**Q46–Q48. 极端样例？**  
**有**：长序列（如 paper correlation case **240 steps×10 nodes**）、复杂图（10+ 边）、节点数 3/5/10 混布（论文称近似均匀）。

**Q49. 不同时间窗口？**  
**有**；correlation/forecasting 题面逐步指定；非全局统一窗长。

**Q50. 数据集内 normalization？**  
**JSONL 存仿真 raw 值**（生成时 `round(..., 2)`）。**value-preserving normalization** 在 **推理 processor**（`processing_qwen3_ts.py::sp_encoding`）对 **输入 TS** 做均值中心化/缩放，**不改变 `output` gold 数值**。

---

## 8. Normalization 与评测数值（Q51–Q54）

**Q51. 是否保留原始尺度？**  
**是**（数据集文件内为仿真物理量级浮点）。

**Q52. value-preserving normalization 的角色？**  
模型 **读入** 时保留 offset/scaling 元数据可还原幅度；属于 **编码阶段**，不是数据集字段。

**Q53–Q54. 对 entity / forecasting 的影响？**  
- **entity**：主看形态与拓扑，**不依赖** gold 数值匹配；norm 主要影响 embedding  
- **forecasting**：gold 为 **raw 尺度**；MAE/MAPE 在 **raw 预测 vs raw gold** 上算（`evaluate_qa.py`），与 processor norm 无关

---

## 9. 均衡性、泄露与图/文本依赖（Q55–Q67）

**Q55–Q58. 四任务难度/数量/长度/答案格式均衡？**  
**不均衡**：correlation 最多（1592 test），forecasting 最少（280）；答案格式 QA vs 数值序列 ** deliberately 不同**。

**Q59–Q60. 类别不平衡？选项偏置？**  
- 生成时用 `option_cycles` **轮换正确答案字母**，减轻 Always-A  
- 场景/domain 约 10 类均匀（论文）；**无**官方 difficulty 分层

**Q61–Q63. 泄露风险？**  
- **模板**：前缀与部分问句固定 → 存在 **格式泄露** 可能  
- **答案格式**：推理追加 Output Format → 训练/测试需一致  
- **图暴露答案**：选项文本 **不应** 含真实 node 编号路径；但强模型可能仅从 **时序形态** 猜对（见 Q64–Q66）

**Q64–Q66. 能否不看图/时序/只看文本？**  
数据集 **未官方标注** 每题依赖类型；论文 ablation 表明部分题 **可仅靠 TS 形态或文本** 做对，但 correlation 多跳题 **设计上需图+时序对齐**。需单独 ablation 实验，非字段级标记。

**Q67–Q68. 官方 ablation 版本？代码自建？**  
- HF 另有 **`ST-CoT-Text` / `ST-Test-Text`**（纯文本 TS，见 `dataset_info.json`）→ 文本模态 ablation  
- **`ST-Test-Text`** 在 HF datasets loader 曾失败（空/ schema 问题，见 `stbench_inspect.log`）  
- 本仓库 **graph ablation** 在 `stage2_4_graph_ablation_paper_cases.py` **代码删图**，非官方字段

---

## 10. 数据质量与事件标注（Q68–Q77）

**Q69. missing value？**  
**无显式 NaN 字段**；仿真可出现 **0.0**（如脉冲/关断），应视为 **有效数值** 非缺失。

**Q70. 异常值/突变？**  
**有**；由 SDE + adjacency modulation 故意产生尖峰、阶跃，服务 reasoning。

**Q71. event / shock / intervention 标注？**  
**无独立列**；事件语义在 **context 文本** 与 **adjacency_modulation.patterns**（生成侧），评测 JSONL 不导出。

（Q72–Q77 见 §3。）

---

## 11. 评测脚本与字段映射（Q78–Q85）

**Q78. 评测如何读答案？**  
`evaluation/evaluate.py` → `load_jsonl_dataset` 读 `output`；`load_prediction_files` 读 `exp/.../generated_answer.json` 的 `response` → 抽 `<answer>` 内容。

**Q79. QA 与 forecasting 是否不同脚本？**  
**同一入口**，`evaluate_predictions_for_task` 分发：  
- QA → `evaluate_multiple_choice_predictions`（accuracy）  
- forecasting → `evaluate_forecasting_predictions`（MAE/MAPE）

**Q80. 字段名不一致？**  
数据集用 `input`/`output`；训练框架用 `prompt`/`response`（`dataset_info.json` 映射）；预测文件用 `response`。**评测统一读 `output` vs 解析后的 prediction**。

**Q81. response 是模型还是数据集字段？**  
**模型输出**（`generated_answer.json`）；**不是**数据集原生列。

**Q82. parsed_answer？**  
**后处理字段**（本仓库 predictions.jsonl 等）；**不在**官方 ST-Test JSONL。

**Q83. parse fail 归因？**  
**几乎总是模型输出格式** 或 **未追加 Output Format**；gold `output` 对 QA/forecasting **规范**（evaluate 可直接 parse）。forecasting 缺 `<answer>` 时 parser 仍尝试抽 `[...]`。

**Q84. 原始答案够规范吗？**  
**是**；选项题 gold 为单字母 tag；forecasting 为 JSON 数组字符串。

**Q85. paper cases 与 ST-Test 格式？**  
**一致**（同样四列 + 子集 metadata）；`PaperCases.jsonl` 可直接走官方 evaluate。

---

## 12. 版本、subset 与报告写法（Q86–Q99）

**Q86. 不一致时代码适配？**  
Stage 2.2 对混合 JSONL 按 `category` 分流；CoT 额外列 **不影响** 仅读四列的推理/评测。

**Q87. task type vs question type？**  
`category` = 任务；correlation 的 direct/indirect **仅生成阶段**区分，**不落盘**。

**Q88–Q90. metadata / difficulty / 溯源？**  
- 官方测试集：**无** difficulty、graph_id  
- 生成侧有 `dataset_id`（QA 合成）；子集用 `source_file` + `original_line_index` 回溯 ST-Test  
- **不能**从单条 ST-Test 反查完整 pickle/scenario JSON（需生成源或 HF 全量）

**Q91. 能否追踪原始 TS 与 graph？**  
**能**：同条 `timeseries` + 从 `input` 解析 `Graph Structure`；与论文 figure 一致。

**Q92–Q95. 公开与版本？**  
- HF：`Time-HD-Anonymous/ST-Bench`（`download_dataset.py`）  
- SmartTest 记录 revision `1a6871632f295dc2a049860b2a7d08ae445c25da`  
- HF **datasets** 对 ST-CoT/ST-Align 多列 schema 可能报错 → **直接读 JSONL 更稳**  
- 本仓库 **默认不含** `data/ST-Bench/`（需下载）

**Q96–Q99. subset 报告怎么写？**

| 范围 | 报告写法 |
|---|---|
| paper_cases（4） | 「Appendix H 复现样例，非 full ST-Test」 |
| SmartTest（20） | 「ST-Test 分层抽样 n=5×4，seed=20260519」 |
| 单任务部分 idx | 明示 `{task}_test.jsonl` 行号/数量 |
| 全量 | 「ST-Test 3273 条，与论文 Table 4 一致」 |

---

## 13. 汇报 ST-Bench 的推荐维度（Q100）

写数据集章节建议 **六维**：

1. **任务类型**：四任务定义与 label 语义（§1、§3）  
2. **数据组成**：1064 synthetic 场景 → 28190 QA；ST-Causal real-world 另计（§4–§5）  
3. **字段格式**：`input` / `timeseries` / `output` / `category`；选项在 input 内（§2、§6）  
4. **规模划分**：CoT / RL / Test；无 val（§5）  
5. **答案形式**：QA 单字母 vs forecasting 数值序列；CoT 仅训练（§3、§6）  
6. **局限性**：合成偏置、模板泄露、无官方 per-sample 依赖标签、Text split  loader 问题、subset 与 full 不可混报（§9、§12）

---

## 附录 A. 四任务详解（直觉 + 样例）

### A.1 一句话对照

| 任务 | 中文 | 关键能力 |
|---|---|---|
| etiological | 场景推断 | 全图叙事 |
| entity | 实体识别 | 单节点角色 |
| correlation | 传播推理 | 节点对+时间窗 |
| forecasting | 上下文预测 | 历史窗→未来值 |

### A.2 共同输入前缀

```text
You are a spatial temporal analysis expert.
Node 0 time series with length of 48: <ts><ts/>; ...
Graph Structure: Node 0->Node 2; ...
please analyze the spatial temporal data and answer the following question: ...
```

### A.3 论文 Appendix H 四 case（本地路径）

| Case | ST-Test 行号 | 文件 |
|---|---:|---|
| etiological Table 6 | 118 | `etiological_test.jsonl` |
| entity Table 7 | 982 | `entity_test.jsonl` |
| correlation Table 8 | 547 | `correlation_test.jsonl` |
| forecasting Table 9 | 9 | `forecasting_test.jsonl` |

完整 JSONL：`00_new_codes/repro_autodl/experiments/stage2_subsets/paper_cases/PaperCases.jsonl`

### A.4 字段示例

```json
{
  "input": "... Graph Structure: ... question: ... Options: A. ... B. ...",
  "timeseries": [[...], [...]],
  "output": "<answer>D</answer>",
  "category": "correlation"
}
```

forecasting 的 `output` 示例：`"[19.86, 19.97, 20.05]"`

---

## 附录 B. 代码与 HF 路径对照

| 论文任务 | ST-Test 文件 | 推理 `--task` | 指标 |
|---|---|---|---|
| T1 Etiological | `etiological_test.jsonl` | `reasoning_etiological` | ACC |
| T2 Entity | `entity_test.jsonl` | `reasoning_entity` | ACC |
| T3 Correlation | `correlation_test.jsonl` | `reasoning_correlation` | ACC |
| T4 Forecasting | `forecasting_test.jsonl` | `reasoning_forecasting` | MAE/MAPE |

```bash
python download_dataset.py   # -> data/ST-Bench/
```

训练阶段目录：`ST-Align/`（非四任务）、`ST-CoT/`、 `ST-SFT/`、 `ST-RL/`、 `ST-Test/`、`ST-Causal/`、`ST-*-Text/`。

---

## 附录 C. 参考

- 论文 §3.2、Appendix B（合成）、Appendix H（样例）、Table 4（规模）  
- `00_new_codes/guides/dataset-ST-Bench使用说明.md`  
- `data_generation/generate_reasoning_QA.py`、`generate_reasoning_forecasting_QA.py`  
- `evaluation/evaluate_qa.py`、`inference/inference_tsmllm_vllm.py`、`inference/prompt.json`
