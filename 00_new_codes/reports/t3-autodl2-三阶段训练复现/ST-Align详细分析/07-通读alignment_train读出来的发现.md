# 07 通读 alignment_train 读出来的发现

修改：2026-06-12  
方法：LLM 连续通读 [alignment_train.jsonl](../../../../data/ST-Bench/ST-Align/alignment_train.jsonl) 多段场景 burst（非脚本抽样统计）；日志见 [artifacts/reading_log.md](artifacts/reading_log.md)。  
前置：[02–06](01-索引.md) 为对照；[06](06-为何训完仍无法复述时序.md) 讲架构与复述，本篇只补 **读数据亲证**。

---

## 0. 读完后最该带走的几句话

1. **一条样本不是「一个任务」，而是一张 simulation 的问答清单**：同一 tensor 先被问遍 temporal，再 spatial 50 题，再 metadata（见 L1–134 完整 burst）。  
2. **波形「能不能读」和「会不会被问到」是两件事**：demand_source 才堆 A/ω/φ；propagation 改问 κ/baseline/evolution — 但 **source 上也可出现根本没法当正弦拟合的序列，仍照样问 A/ω/φ**（L14991 一带）。  
3. **读 TS 最有说服力的正向样例**是 L1–3 的 node0 正弦与 L50000 的 mean_reverting；**最有说服力的负向样例**是 L132 调制窗与 node2 glitch 时间对齐，但 gold multiplier=15 **不在波形里**。  
4. 与 LoRA 探针 **6/40** 的衔接：数据里 **真正「波形与 gold 一致」的题是少数**；多数 temporal 要么重复问、要么不可拟合、要么在 350/1850 clip 上 — 探针全灭不意外（[04](04-探针交叉论证.md) **支持**）。

---

## 1. 场景 burst：读一条等于读几百条

以 **L1–134**（5×96）为例，同一 `timeseries` 上的展开顺序为：

```text
temporal（各 node 的 SDE 参）→ spatial（25 对 × 直连/间接）→ spatial_temporal（角色、lag、modulation）
```

亲历：L1–3 问 node0 的 A/ω/φ；L57–59 问 node2 的 evolution/baseline/κ（**不问**正弦三参）；L118+ 进入 spatial；L122+ 问 demand_source/propagation 与 lag；L132 问 `[40,44]` 调制。

**读数据收获**（02/03 的 stats 未强调）：  
- 模型在同一 prefix 下要 **切换数十种末句模板** — Stage1 很大一部分 capacity 在 **记问法→短答映射**，而非 TS。  
- [03](03-样本形态与生成.md) 的 median **310 题/场景** 是平均；亲历 burst **~130 行/场景** 只是其一，大场景 spatial 段更长（10 节点 ≈200 spatial 行）。

---

## 2. 波形亲历：三类节点、三种「可读性」

### 2.1 Source 正弦（可读，且与 gold 对齐）

**L1**，node0：96 点从 ~20 单调降至 ~4 再升回 ~20，**单周期**；gold A=8.0, ω=0.0654, φ=1.5708。  
粗算：中心 ~12，振幅 ~8，与 A=8 一致；96 步内一个谷底 — 与 ω 量级相符。  
→ **这类题是 ST-Align 里极少数「读 TS 真应该能学会」的题**；探针仍 0/5，说明 **1500–2000 step + 数据淹没** 比「题不可学」更贴数据（[04](04-探针交叉论证.md) **部分支持、需训练实验**）。

### 2.2 Propagation：脉冲、近零、但不是乱问正弦

**L1** node2：t≈41–47 序列 `…10.79, 2.23, 11.45, 0.27, 12.54, 0.0, 11.93, 0.0…`；**L57** gold `mean_reverting`，κ=0.2。  
生成脚本 **没有** 对 propagation 节点套 `sinusoidal_A` 模板（亲历 L1–200 仅 node0 有 A/ω/φ）。  
→ 修正一种误解：「下游节点都被问 A/ω/φ 所以 ill-posed」**过强** — 更精确是 **「propagation 问 κ/形态，但波形仍是耦合+clip 的混合过程，κ 也难从 TS 精估」**（探针 κ 2/5 与读感一致）。

### 2.3 调制窗与 glitch 共现（读得到的因果感，读不出 gold）

**L132**：`edge (0,2)` 在 `[40,44]` multiplier=**15**。  
同一 tensor 的 node2 在 **t=41–47** 出现上述脉冲。  
→ **读数据的新收获**：ST-Align **在现象级把「边调制」和「下游波形突变」绑在同一 simulation**；但 gold **15** 仍来自 `agent4` metadata，**不是** 从幅度比反推。  
→ 对 Stage1：模型可能被训练成 **「看到 glitch 就输出某个常见 multiplier」**（stats 里 15 占 59%），而非物理反演。

---

## 3. 与 02–06 的对读（支持 / 补充 / 新发现）

| 既有结论 | 通读判定 | 证据行 |
|----------|----------|--------|
| 46% 图结构不需 TS | **支持** | L118：4→3 直连 no，仅看 Graph |
| 25% 正弦 A/ω/φ | **支持**，但 **补充** | 主要打在 **demand_source**；L14991 显示 **source 也可波形不可拟合** |
| phase 错答 -1.5708 = 先验 | **支持** | 多簇 φ 问法；L14993 φ=-1.832 与波形 41 个 0 并存 |
| evolution 探针 4/5 | **支持** | L50000 窄幅 mean-reverting 肉眼可辨 |
| 423 场景 × 高冗余 | **补充** | 同场景内 **同一 node 同一参数因多 drift pattern 重复问**（L4728–4743 段，见日志） |
| Stage1 不教复述 | **支持（现象侧）** | 全库无「输出序列」问法；L119800 大尺度 TS 仍只问 ω |
| 06 架构无 decoder | **不重复** | 本篇不论证复述；只说明 **数据侧也未给逐点监督** |

---

## 4. 读数据才浮现的训练含义（不重复 05 的采样建议）

1. **探针面板应含「可读正弦」与「不可拟合 source」两类** — 仅 40 条 balanced 可能高估 κ/evolution、低估「假正弦题」噪声。  
2. **spatial 段在同一 prefix 上连续出现 50–200 次** — 若 loss 均匀，**ts_encoder 在 burst 前段 temporal 之后很快被 spatial CE 淹没**（亲历顺序 temporal→spatial）。  
3. **调制/ lag 题** 与 ST-Test 的 etiological/correlation **机制语言相近**，但 gold 仍是 **lookup** — Stage2 冷启动 gap 在 **推理链**，不在 **词表**（[05](05-Stage1含义与建议.md) **补充**）。

---

## 5. 与 06 的分工

| | [06](06-为何训完仍无法复述时序.md) | 本篇 07 |
|---|------|--------|
| 核心问题 | 为何不教/不能复述 raw TS | 读 jsonl 后 **哪些题真在读 TS、哪些在背表** |
| 证据 | encoder 结构、39 报告 | L1、L132、L14991、L50000、L119800 等行 |
| 结论 | 复述需新任务+decoder | **现有 temporal 大量「伪读 TS」或「重复问」** |

---

## 6. 仍未读够的部分

- `alignment_test.jsonl`  
- 全库是否 **所有** propagation 都不问 A/ω/φ（目前 L1–200 与生成逻辑支持，但未逐行穷举）  
- 8×A100 official 1000 step 探针

---

返回 [01-索引.md](01-索引.md)
