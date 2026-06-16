# HeaRTS / HEARTS 数据集深度调研（v2 · 文字分析版）

> 论文：**HeaRTS: Benchmarking LLM Reasoning on Health Time Series**（arXiv:2603.06638，ICML 2026 poster）  
> 官方：[GitHub yang-ai-lab/HEARTS](https://github.com/yang-ai-lab/HEARTS) · [HF yang-ai-lab/HEARTS](https://huggingface.co/datasets/yang-ai-lab/HEARTS)  
> 本报告版本：**v2**（2026-06-16 重做）— 按 Superpowers 流程重写；**不含统计图**，以评测范式、任务解剖、临床语义与方法论对照为主。  
> 本地数据：`/root/autodl-tmp/datasets/HEARTS/`（隔离，未写入 `STReasoner_reproduce/data/`）

---

## 1. 一句话定位：它测的不是「会不会看曲线」，而是「会不会在真实文件上完成健康时序任务」

HEARTS（发布名 HeaRTS）与多数时序 benchmark 的根本差异在于 **任务形态**：

| 维度 | 典型时序 benchmark | HEARTS |
|------|-------------------|--------|
| 输入 | 张量 / 固定长度窗口 | **原始文件**（CSV、NPY、JPG、音频等）落在 agent 工作目录 |
| 交互 | 单次 forward | **多轮**：读文件 → 写代码执行 → 观察输出 → 再推理 |
| 输出 | 类别或连续值 | **结构化 JSON** 或 `output/solution.json` 文件 |
| 能力假设 | 模式识别 / 预测 | **分层推理**：感知统计 → 事件推断 → 生成 → 时序演绎 |

论文将能力分为四维（后文 §3 展开）：**Perception、Inference、Generation、Deduction**。全 benchmark 含 16 数据集、110 任务、20,226 条测试样本；**本次本机仅持有 HF 发布的 10 条冻结样例**（`cgmacros/cgm_stat_calculation`），分析深度集中在「能读源码 + 能 load 样例」处，对全量 110 任务以论文 + 已克隆代码为据，不虚构本地文件。

---

## 2. 评测范式：CodeAct 闭环如何定义「做对」

### 2.1 冻结样例 → agent 工作区

`run_exp_freeze.py` 从 `fix_test_cases_dir/{dataset}/{task}/{index}.pkl` 读取 `pickle.load` 后的 dict，交给对应 `Experiment.run_agent`。以 CGMacros 为例，`base.py` 的 `save_data` 把 DataFrame 写成：

```text
agent_working/<query_id>/input/cgm.csv
```

图像任务则复制 `*.jpg` 到 `input/`。**模型从不直接「看见」内存里的 DataFrame**，而是与真实数据分析脚本一样，面对磁盘上的文件路径——这与临床/生信流水线一致，也提高了对 **工具使用与 I/O 理解** 的要求。

### 2.2 CodeAct agent：思考 → 执行代码 → 或最终答案

`agents/codeact/codeact_impl.py` 用 LangGraph + Jupyter 内核实现循环：

1. 模型输出 `<thought>` 推理；
2. 要么 `<execute>` Python（可 `pandas.read_csv`、绘图、数值计算）；
3. 要么 `<solution>` 直接给出符合格式的答案。

系统 prompt 明确要求 **不要打印整段长序列**（上下文长度约束），迫使 agent 写聚合逻辑而非暴力 dump。`only_thoughts_limit`（默认 30）限制「只思考不执行」的轮数，避免无限空谈。

这与 STReasoner 推理时「题干 + `<ts>` 占位符一次性进模型」完全不同：HEARTS 考察 **过程性工具推理**，而不只是单次 token 预测。

### 2.3 两种答案通道

| 类型 | 代表任务 | 答案如何提交 |
|------|----------|--------------|
| **对话内 JSON** | `cgm_stat_calculation`, `a1c_classification` | `agent.query` 返回的 final_answer 经 `OutputStringParser.parse_dict` + `json_repair` 解析 |
| **工作区文件** | `meal_forecasting`, `non_meal_imputation_*` | prompt 要求写入 `output/solution.json`；`parse_output(query_id=...)` 读文件 |

第二类任务额外考察 **文件协议遵守**（列表长度必须 30、顺序与分钟对齐等），解析失败计入 `Failures`。

### 2.4 评分与「推理」的关系

以 `cgm_stat_calculation` 为例，GT 是 `below`/`above` 两个百分比，指标为 sMAPE 族（论文 Perception · Stat.Calculation）。**做对**在工程上等于：

1. 正确读 CSV 列名 `Libre GL`；
2. 按阈值 70 / 180 mg/dL 计数（见源码 `calculate_stat`）；
3. 输出可解析 JSON，且数值与 GT 接近。

失败模式包括：列名搞错、把 mg/dL 当 mmol/L、百分比 vs 比例、JSON 外包裹 markdown、未执行代码而幻觉数字。论文报告 SOTA LLM Overall 仅 ~0.66（Naive ~0.61），说明 **大量失败来自流程而非单一公式不会背**。

---

## 3. 四维能力框架：论文在考什么认知步骤

### 3.1 Perception（感知）

从原始信号提取 **可度量的生理统计量**，而非分类标签。

- **Stat.Calculation**：TIR（time in range）、iAUC、自定义窗口统计——需要明确临床/生理定义（如 TIR 70–180 mg/dL 来自 ADA CGM 指南，源码注释链接 diabetes.org）。
- **Feat.Ext.**（全 benchmark 其他数据集）：从 ECG/EEG/音频等抽特征。

认知负荷：**规则清晰 + 数据清洗 + 聚合**；对 LLM 而言，难点在可靠执行而非医学名词。

### 3.2 Inference（推断）

在信号上回答 **是什么 / 何时 / 谁**：

- **Event Localization**：如 `meal_time_localization`——2 小时窗口内唯一进餐，从 CGM 上升沿定位 `meal_timestamp`（分钟）。
- **Physiological Classification**：`a1c_classification` 由全日 CGM 推断 normal / prediabetes / diabetes；`meal_img_classification` 用 CGM 曲线 + 四张餐食图做跨模态匹配。
- **Subject Profiling**：`fasting_glu_prediction`、`meal_react_comparison`——从时序推断个体指标或疾病状态差异。

认知负荷：**模式识别 + 领域先验**（餐后曲线形态、空腹段估计）；比 Perception 多一层 **语义映射**（曲线 → 临床类别）。

### 3.3 Generation（生成）

输出 **未来或缺失** 的数值序列：

- **Future Forecasting**：`meal_forecasting` 族——给定餐前 1h CGM，预测餐后 30 分钟每分钟血糖；变体控制是否提供前 3 天 CGM/餐食表、是否告知当餐营养信息（消融设计）。
- **Signal Imputation**：`non_meal_imputation_*`——2h 窗口内 30min CGM 被置 0，用剩余 CGM 及可选 HR/活动卡路里补全。

认知负荷：**时序动力学 + 多模态条件**；论文指出 LLM 常退化为插值/复制启发式，而非真正建模餐后动力学。

### 3.4 Deduction（演绎）

全 benchmark 含时序排序、纵向轨迹等任务；**当前克隆的 GitHub `main` 仅在 `exp/cgmacros` 暴露 14 个任务，不含 Deduction 实现文件**。论文 Table 3 中 SHHS、Gazebase 等数据集承担大量 Deduction 样本（如 4799 条在 SHHS），是 LLM 最弱维度之一——本地无法复现该子集，但应在汇报时说明 **Deduction 依赖未下载的睡眠/眼动等大数据集**。

---

## 4. CGMacros 十四任务解剖（源码级）

CGMacros 是代谢领域核心集，论文 test **2333** 条，模态为 CGM + HR + 餐食标注 + 餐食图像。下表基于 `exp/cgmacros/*.py` 类 docstring 与 prompt，说明 **agent 实际收到什么、必须产出什么、难在哪**。

| 任务 | 能力维 | 主要输入文件 | 输出 | 核心认知步骤 |
|------|--------|--------------|------|--------------|
| `cgm_stat_calculation` | Perception | `cgm.csv`（~24h） | `below`/`above` % | 阈值计数 → TIR 反义（低于/高于范围） |
| `iauc_calculation` | Perception | 餐后 2h CGM | `iauc` 标量 | 基线校正曲线下面积 |
| `a1c_classification` | Inference | 全日 CGM | `disease_status` 三分类 | 长期变异 + 高血糖暴露 → 糖化风险层级 |
| `meal_time_localization` | Inference | 2h CGM（分钟索引） | `meal_timestamp` | 单事件检测 / 上升沿定位 |
| `fasting_glu_prediction` | Inference | 全日 CGM | `fasting_glu` mg/dL | 识别空腹段并估计空腹血糖 |
| `meal_react_comparison` | Inference | `A.csv`/`B.csv` 两窗 | `normal_subject` A/B | 相似餐食下比较正常 vs 糖代谢异常餐后反应 |
| `meal_img_classification` | Inference | 4h CGM + 4×jpg | `chosen_image` | 跨模态：曲线形态 ↔ 餐食视觉 |
| `meal_forecasting` | Generation | 3d CGM + meal 表 + 餐前 1h | `solution.json` 30 点 | 有历史参照的短期预测 |
| `meal_forecasting_meal_info` | Generation | 同上 + 当餐营养 | 同上 | 加入当餐碳水等条件 |
| `meal_forecasting_no_ref` | Generation | 仅餐前 1h | 同上 | 无历史，考泛化 |
| `meal_forecasting_no_ref_meal_info` | Generation | 餐前 1h + 营养 | 同上 | 无历史 + 营养提示 |
| `non_meal_imputation_cgm_only` | Generation | 2h CGM，30min 掩码 | 30 点列表 | 非餐窗口、仅 CGM 自回归/插值 |
| `non_meal_imputation_hr` | Generation | CGM+HR | 同上 | 多模态生理耦合 |
| `non_meal_imputation_calories` | Generation | CGM+活动卡路里 | 同上 | 活动代谢与血糖联动 |

**任务族内的消融逻辑**（`meal_forecasting*` 四变体）是 HEARTS 设计亮点：同一预测目标下切换 **历史上下文** 与 **当餐信息**，可分离「记忆个体节律」与「利用当餐成分」的贡献。汇报时可强调：这不是 4 个无关任务，而是 **受控因素实验**。

---

## 5. 聚焦：`cgm_stat_calculation` 的临床语义与 GT 算法

### 5.1 临床背景：Time in Range 的对偶指标

prompt 要求两个 **互补** 百分比：

- `below`：CGM **< 70 mg/dL** 的时间占比（低血糖暴露）；
- `above`：CGM **> 180 mg/dL** 的时间占比（高血糖暴露）；
- 中间 [70, 180] 为 **in range**（TIR），临床用于 T1/T2 血糖管理目标，但 **不在 JSON 输出字段中**——agent 需理解「below/above」与 TIR 的关系：TIR ≈ 100% − below − above（在无缺失读数时）。

源码 `NORMAL_RANGE = (70, 180)`，`calculate_stat` 对每一分钟读数严格比较 `< 70` 与 `> 180`（边界值 70、180 本身 **不算** below 或 above）。

### 5.2 GT 如何生成（可复现）

```python
below_pct = count(v < 70) / N * 100
above_pct = count(v > 180) / N * 100
# 四舍五入到 4 位小数
```

本次用 `pickle.load` 对 10 条冻结样例 **重算与 GT 完全一致**（验收命令见 §10）。

### 5.3 本批 10 例的群体特征（非图表，描述性）

- 读数 **N = 1431**（约 24h、1/min），与冻结格式一致。
- **全部 `below = 0.0`**：10 例 `min(Libre GL)` 均 ≥ 79 mg/dL，无低血糖分钟——符合 CGMacros 筛选人群或采样日无 hypoglycemia 的常见情况。
- **`above` 与 `max(Libre GL)` 强相关**：例如 index 1 `max=289` → `above=17.05%`；index 9 `max=175` → `above=0%`。index 9 为 **完美 TIR 日**（100% in range）。
- 标准差高（如 index 1 `std≈45.7`）通常伴随更高 `above`，反映餐后尖峰更剧烈。

这些模式说明：该任务对 LLM **并非「猜一个数」**，而是要在 1431 行上正确计数；肉眼扫一眼曲线容易低估高频尖峰的分钟占比。

---

## 6. 十条冻结样例：题干、GT 与逐条解读

### 6.1 共用题干（官方英文原文）

```text
The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}
```

### 6.2 逐条表（`pickle.load` 统计 + GT）

| index | min | max | mean | std | TIR% | GT below | GT above | 解读 |
|------:|----:|----:|-----:|----:|-----:|---------:|---------:|------|
| 0 | 80.0 | 192.0 | 126.0 | 21.8 | 96.30 | 0.0 | 3.7037 | 轻度高血糖暴露；峰值刚超 180 |
| 1 | 90.0 | 289.0 | 145.5 | 45.7 | 82.95 | 0.0 | 17.051 | 明显餐后高峰；约 1/6 时间 >180 |
| 2 | 79.0 | 233.0 | 129.9 | 32.2 | 91.75 | 0.0 | 8.246 | 中等尖峰 |
| 3 | 93.0 | 231.0 | 134.7 | 40.1 | 82.67 | 0.0 | 17.3305 | 与 1 类似高 above |
| 4 | 92.0 | 206.0 | 130.0 | 32.0 | 84.98 | 0.0 | 15.0245 | 持续偏高波动 |
| 5 | 87.0 | 244.0 | 139.6 | 35.7 | 84.84 | 0.0 | 15.1642 | 均值偏高，尖峰显著 |
| 6 | 94.0 | 225.0 | 128.5 | 29.8 | 93.15 | 0.0 | 6.8484 | 控制较好 |
| 7 | 105.0 | 193.0 | 134.1 | 25.8 | 92.52 | 0.0 | 7.4773 | 空腹段较高，尖峰有限 |
| 8 | 92.0 | 223.0 | 136.9 | 33.1 | 84.91 | 0.0 | 15.0943 | 中高波动 |
| 9 | 97.0 | 175.0 | 128.7 | 22.2 | 100.0 | 0.0 | 0.0 | 全日无超范围读数 |

**正确答案示例（index 0）**：`{"below": 0.0, "above": 3.7037}`

完整摘录 JSON/Markdown：`artifacts/sample_qa_excerpts.json`、`artifacts/sample_qa_excerpts.md`（含 `glucose_stats` 与 `gt_verified_against_series: true`）。

---

## 7. 论文主要发现：成因分析（非 leaderboard 复述）

依据 README「Key findings」与 Table 2 摘要（`artifacts/table_leaderboard.md`）：

1. **LLM 弱于专用时序模型**  
   专用模型在固定窗口、固定模态上端到端训练；HEARTS 任务却要求 **读 CSV、写代码、处理列名与单位**。LLM 的优势在语言泛化，不在默认优化的数值流水线，Overall 0.66 vs Naive 0.61 说明 **部分任务靠启发式也能拿分**，但离专用上界仍远。

2. **与通用推理基准弱相关**  
   健康时序推理依赖 **领域文件习惯 + 生理约束**（餐时、采样率、掩码语义），与数学/逻辑 puzzle 技能重叠有限。不能把 HEARTS 高分等同于「强推理模型」。

3. **长序列与高采样率退化**  
   CodeAct 虽可用 pandas 聚合，但 **思考轮次与上下文** 仍受限；prompt 禁止打印全序列。高采样任务（PERG 1700Hz、EMG 2048Hz）更需要 **分块/降采样策略**，LLM 若不会写正确预处理，后续全错。

4. **输入格式改变绝对分、不改变难度排序**  
   文本化曲线 vs 原始文件 vs 图像，主要影响 **信息进入模型的通道**；任务相对难度稳定说明瓶颈在 **时序语义** 而非 tokenizer。

5. **同族模型失败模式相似**  
   scaling 不能自动获得生理常识；需工具链或领域微调。

---

## 8. 数据发布三层与本次证据边界

```text
Layer 1  HF frozen .pkl     → 本次：10 × cgm_stat_calculation（~437 KB）
Layer 2  prepare_data 代码  → 已浅克隆，exp/cgmacros 14 任务定义可读
Layer 3  16 原始生理库      → 未下载（部分 Restricted：SHHS、PhyMER、GLOBEM、Bridge2AI-voice）
```

论文 20,226 test cases **不等于** HF 当前发布范围。16 数据集开放状态见 `artifacts/table_datasets.md`：4 个 Restricted 占 test 样本大头（如 SHHS 4799），制约 **无申请条件下的全 benchmark 复现**。

**开源结论**：HF 数据集与 GitHub 代码仓库均为 **public、非 gated**；预检见 `artifacts/preflight_result.json`。

---

## 9. 与 STReasoner / ST-Bench 方法论对照

| 维度 | ST-Bench / STReasoner | HEARTS |
|------|----------------------|--------|
| 世界来源 | **合成** STS 场景 + SDE 模拟 | **真实** 多中心生理数据 |
| 输入形态 | 文本图结构 + `<ts>` 嵌套数值列表 | 磁盘 CSV/NPY/图像/音频 |
| 推理类型 | 场景/实体/传播/预测（图约束 QA） | 临床感知、进餐事件、补全与预测 |
| 训练角色 | 三阶段 SFT + RL 的训练主数据 | **评测 benchmark**（非训练集） |
| 模型接口 | TSM LLM 一次 forward | CodeAct 多轮代码执行 |
| 答案格式 | `<answer>` 选项或数值列表字符串 | JSON / solution.json |

**能否替代 ST-Bench 做训练？** **不能。** 字段、模态、优化目标均不兼容；HEARTS 不应写入 `STReasoner_reproduce/data/`。

**互补价值：** 若未来 STReasoner 类模型要评估 **真实 CGM 文件上的工具推理**，可借 HEARTS 的 `run_exp_freeze.py` 作 **零样本外部评测**；与 ST-Test 上的合成图推理 **正交**，可共同描述「时空推理」是否泛化到临床文件场景。

**任务语义对照（示意）：**

- ST-Bench `correlation`：图上两节点传播关系 → HEARTS `meal_react_comparison`：两条餐后曲线哪条来自正常人；
- ST-Bench `forecasting`：合成节点未来值 → HEARTS `meal_forecasting`：餐后 30min 真实 CGM；
- ST-Bench `entity`：节点角色标签 → HEARTS `a1c_classification`：全日 CGM → 疾病状态。

相似点是 **需要跨时间整合证据**；差异是 HEARTS **无显式图结构**，图在生理因果里隐含（餐食 → 血糖 → HR）。

---

## 10. 下载、复现与验收记录

### 10.1 本地路径

| 路径 | 内容 |
|------|------|
| `/root/autodl-tmp/datasets/HEARTS/frozen_test_cases/` | 10 个 pkl |
| `/root/autodl-tmp/datasets/HEARTS/code/` | HEARTS 框架浅克隆 |
| `artifacts/download_manifest.json` | `pkl_count=10`, `lfs_pointer_remaining=[]` |

下载曾遇 `huggingface_hub` 在代理下失败，最终用 `git clone` + `curl -x 127.0.0.1:17997` 拉 LFS 对象。

### 10.2 复现分析管道（**不含作图**）

```bash
export HF_HOME=/root/autodl-tmp/datasets/HEARTS/.hf_cache
export https_proxy=http://127.0.0.1:17997 http_proxy=http://127.0.0.1:17997
cd STReasoner_reproduce/00_new_codes/reports/new_datasets/hearts_survey
python3 inventory.py --root /root/autodl-tmp/datasets/HEARTS/frozen_test_cases \
  --code-root /root/autodl-tmp/datasets/HEARTS/code --out ../artifacts
python3 inspect_pkl.py --root ... --taxonomy ../artifacts/task_taxonomy.json \
  --out ../artifacts/pkl_schema_samples.json --excerpts-out ../artifacts/sample_qa_excerpts.json
python3 render_excerpts_md.py --in ../artifacts/sample_qa_excerpts.json --out ../artifacts/sample_qa_excerpts.md
```

### 10.3 跑 HEARTS 官方评测（需 API）

```bash
cd /root/autodl-tmp/datasets/HEARTS/code && uv sync
# 配置 .env 中 LLM API
uv run run_exp_freeze.py \
  --fix-test-cases-dir /root/autodl-tmp/datasets/HEARTS/frozen_test_cases \
  --dataset-name cgmacros --task cgm_stat_calculation --dry-run  # 先校验
```

### 10.4 v2 验收（2026-06-16 执行）

- 主报告不嵌入统计图（验收见 §10.4 shell）
- `pickle.load` 10 例 GT 与序列重算一致
- `sample_qa_excerpts.json` 每条含 `glucose_stats`
- `STReasoner_reproduce/data/` 无 HEARTS 子目录

---

## 11. 卸载

见 [02-HeaRTS-资源卸载说明.md](02-HeaRTS-资源卸载说明.md)。分析结束后可：

```bash
bash hearts_survey/uninstall.sh --data-only
```

仅删除 `/root/autodl-tmp/datasets/HEARTS/`，保留本报告与 `artifacts/`。

---

## 12. 引用

```bibtex
@article{hearts2026,
  title={HEARTS: Benchmarking LLM Reasoning on Health Time Series},
  author={Sirui Li and Shuhan Xiao and Mihir Joshi and Ahmed Metwally and Daniel McDuff and Wei Wang and Yuzhe Yang},
  journal={arXiv preprint arXiv:2603.06638},
  year={2026}
}
```

---

**附录：计划文件** `.cursor/plans/hearts_deep_analysis_v2.plan.md`（Superpowers v2 重规划）
