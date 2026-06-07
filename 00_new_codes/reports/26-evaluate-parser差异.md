# 26 — evaluate parser 与官方 [LingFengGold/STReasoner](https://github.com/LingFengGold/STReasoner) 差异报告

> 对比基准：GitHub `main` 分支 `evaluation/evaluate_qa.py`、`evaluation/evaluate.py`（2026-05-31 拉取 raw）。  
> 本地对象：`STReasoner_reproduce/evaluation/` 当前工作区 + git 历史（`dbe9aa5` → `58a2a7e`）。

---

## 1. 一句话结论

**选择题 parser：本地与官方一致，均为 tag-first（只认 `<answer>` 内且开头为 A–D）。**  
**Forecasting parser：本地 `_parse_series` 比官方多两步结构化抽取，并去掉「全文扫数字」兜底——这是本仓库相对官方唯一的实质性 parser 改动。**  
`evaluate.py` 入口逻辑与官方相同，仅多了中文注释块。  

另：复现脚本里存在 **第二套 parser**（`stage2_run_smarttest.py` 的 strict 模式），与 `evaluate_qa.py` 不一致，跑分必须以 `evaluation/evaluate.py` 为准。

---

## 2. 评测链路（parser 在何处生效）

```text
generated_answer.json
  └─ load_prediction_files()
       └─ _extract_tag_content(response, "answer")   ← 第 1 层：抽 tag 内容
            └─ predictions[idx] = 抽出的字符串
                 └─ evaluate_multiple_choice / evaluate_forecasting
                      └─ _normalize_choice() 或 _parse_series()   ← 第 2 层
```

官方 README 推理/评估循环见 [STReasoner README — Evaluation](https://github.com/LingFengGold/STReasoner#-evaluation)（四类 task，未写 `reasoning_causal`，但代码里已注册）。

---

## 3. 原逻辑（官方 GitHub main）

### 3.1 `_extract_tag_content(text, tag="answer")`

| 步骤 | 行为 |
|---|---|
| 1 | 正则 `<answer>\s*(.*?)\s*</answer>`（DOTALL + IGNORECASE） |
| 2 | 命中 → 取 group(1)，去 markdown 围栏 `` ``` ``，strip |
| 3 | 未命中 → **返回整段 `text.strip()`**（不做其它格式兼容） |

### 3.2 `_normalize_choice(text)` — 多选题

| 步骤 | 行为 |
|---|---|
| 1 | 对输入再调 `_extract_tag_content(..., "answer")`（gold 常已是 `<answer>D</answer>`，预测已在 load 阶段抽过 tag，此处多为幂等） |
| 2 | 仅当**字符串开头**匹配 `^\s*([A-Da-d])[\.\)\s-]*` → 返回大写字母 |
| 3 | 否则 → `value.lower()` 整段小写（通常 **无法** 与 gold 字母相等 → 算错） |

**不识别：** `Answer: D`、`\boxed{C}`、文末 Option 句子、全文搜索 A–D。

### 3.3 `_parse_series(text)` — forecasting（官方）

| 优先级 | 行为 |
|---|---|
| 1 | 已是 list / 单数字 → 直接 float |
| 2 | `json.loads` 成功且为 list 或 number → 使用 |
| 3 | **兜底：`re.findall(r"-?\d+\.?\d*", text)` 抽全文所有数字** |

### 3.4 `load_prediction_files`

- 支持 flat `response` 与 CoT 式 `responses[]`（取 attempt 最小的一条）。
- 支持外层 `{"results": [...]}` 包装。
- 每条：`predictions[idx] = _extract_tag_content(text)`。

### 3.5 `evaluate.py`

- CLI + `DEFAULT_TASK_CONFIG` + `extract_token_stats`（token 统计）。
- 与官方 raw 内容一致；本地仅多文件头 docstring。

---

## 4. 新逻辑（本仓库当前 `evaluation/evaluate_qa.py`）

### 4.1 与官方相同的部分

- `_extract_tag_content`：**逐行一致**
- `_normalize_choice`：**逐行一致**（注释写明「tag-first；与官方 evaluate 路径一致」）
- `load_prediction_files`、`evaluate_multiple_choice_predictions`、`evaluate_alignment_predictions` 主流程：**一致**
- `evaluate.py`：**一致**（+ 中文注释）

### 4.2 相对官方的改动

#### 改动 A — `_parse_series`（**唯一实质 diff**）

在 `json.loads` 失败后，**不再**全文 `findall` 所有数字，改为：

1. 倒序匹配 JSON 片段键：`"predictions"` / `"prediction"` / `"forecast"` / `"answer"` → 后的 `[...]`，递归 `_parse_series`
2. 倒序匹配文中所有 `[...]`（不含嵌套 `[]`），若其中数字个数 ≤ 20 → 采用
3. 都失败 → 返回 `[]`（该样本 forecasting **missing**，不计 MAE）

git：`0be4f2b..HEAD` 中 `evaluation/evaluate_qa.py` 仅此函数体变化（另加中文注释）。

#### 改动 B — forecasting 指标（官方后续 commit 已有，本仓库继承）

`0be4f2b` 起增加 **MAPE**、`target_stats`；Initial commit 只有 MAE。这与 GitHub 当前 main 一致，**不是**复现 agent 私改。

#### 改动 C — `evaluate.py` 文件头

增加 11 行中文说明注释，**无行为变化**。

### 4.3 曾讨论但未留在 evaluate 里的「宽 parser」

`09-stage2.2效果调试.md` 记录过临时方案：在 `_normalize_choice` 中增加 `\boxed{}`、`Answer:`、末尾 Option 等规则。  

**当前 `evaluate_qa.py` 中不存在这些规则**；0531 计划（`13-2.3提示词存档.md`、`19-...`）明确 **收严为 tag-first**，与官方对齐。  
该宽逻辑若存在过，**未进入当前 git 的 evaluate_qa.py**（或已回滚）。

---

## 5. 行为对比表（实测）

在「先 `_extract_tag_content` 再 normalize / parse」的官方管线下测试：

### 5.1 多选题

| 模型 raw 输出（示意） | 官方 & 本地 evaluate 解析 | 能否计分 |
|---|---|---|
| `...<answer>D</answer>` | `D` | ✓ |
| `<answer>B</answer>` | `B` | ✓ |
| 仅 `D`（无 tag，load 后整段为 `D`） | `D` | ✓ |
| 长 CoT 末尾 `Answer: D`（无 tag） | 整段小写字符串，非单字母 | ✗ |
| `\boxed{C}`（无 tag） | 非 A–D 开头 | ✗ |
| 正文以 `A` 开头但真实答案在文末 | 可能误读为 `A` | 偶发 ✓/✗ |

**结论：** 无 `<answer>` 标签时，官方与本地 evaluate **都不会**给分；差别不在 MC parser。

### 5.2 Forecasting

| 预测文本（tag 内或 load 后） | 官方 `_parse_series` | 本地 `_parse_series` |
|---|---|---|
| `[19.86, 19.97, 20.05]` | 3 个数 | 相同 |
| `{"predictions": [20.02, 20.13, 20.23]}` | 3 个数（json.loads） | 相同 |
| `Step 1.5 ... Final [20.02, 20.13, 20.23]` | **5 个数**（含 1.5, 2.3） | **3 个数**（取最后 `[]`） |
| 无任何可解析结构 | 可能抽到零散数字 | `[]` → missing |

**结论：** CoT 里含大量中间数字时，**本地 MAE 可能比官方更低或更高**（取决于是否误抽/漏抽）；有规范 `<answer>[...]</answer>` 时两者一致。

---

## 6. 本仓库内「多套 parser」对照

| 位置 | 用途 | 与官方 evaluate 关系 |
|---|---|---|
| `evaluation/evaluate_qa.py` | **正式计分** | MC 同官方；FC 见 §4.2 改动 A |
| `stage2_2_run_paper_cases.py` → `parse_model_answer()` | paper_cases 中间态解析 | **调用同一套** `_extract_tag_content` / `_normalize_choice` / `_parse_series` |
| `stage2_run_smarttest.py` → `parse_model_answer()` | SmartTest 实验 | **更严**：必须恰好 1 个 `<answer>`，内容必须 `^[A-D]$` 或 strict forecasting；**≠ evaluate** |
| `repro_kaggle/scripts/05_eval_sttest_tiny.py` 等 | 早期 tiny 实验 | 曾用全文搜 A–D；**不应**与官方 evaluate 混比 |

**规则：** 论文对齐、ST-Test 6144 附件、paper_cases 正式结论 → 只用  
`PYTHONPATH=. python evaluation/evaluate.py --dataset data/ST-Bench/...`。

---

## 7. 官方与本地 git 时间线（evaluate_qa.py）

| Commit | 变更 |
|---|---|
| `dbe9aa5` Initial | 与 GitHub main 同结构的 tag-first MC + 全文数字 FC |
| `0be4f2b` | + MAPE / target_stats（与官方一致） |
| `0be4f2b..HEAD` | **仅** `_parse_series` 结构化抽取 + 中文注释 |

`evaluation/evaluate.py`：自 initial 以来仅 + docstring 注释。

---

## 8. 两处与 parser 无关但影响分数的已知问题

### 8.1 `evaluate_alignment_predictions` 相对误差分支（官方即有）

当 `target_float != 0` 且 `|target| > 1e-6` 时，代码算了 `rel_error` 但 **未** 写入 `rel_sum`；只有 `|target| ≤ 1e-6` 分支才累加。  

对齐任务若 numeric 样本为主，**official 与 local 同样受影响**；与 ST-Test 四类 reasoning 无关。

### 8.2 默认 dataset 路径

`evaluate.py` 的 `DEFAULT_TASK_CONFIG` 指向 `data/reasoning/*.jsonl`，实际数据在 `data/ST-Bench/ST-Test/`。  

**与 parser 无关**，但不显式 `--dataset` 会 FileNotFound；与 `inference_tsmllm_vllm.py` 路径不一致。

---

## 9. 复现建议

1. **计分只认** `evaluation/evaluate.py` + 显式 `--dataset data/ST-Bench/...`。  
2. **不要**用 SmartTest strict parser 的结果对比 ST-Test accuracy。  
3. **不要**在评测前把 raw 改成 `<answer>`（违反 agents 规则）；应靠推理 + `prompt.json` 让模型自己出 tag。  
4. 对比论文 forecasting MAE 时，注明本仓库 `_parse_series` 与 GitHub main 在「CoT 含多余数字」样本上可能 **不一致**；规范 tag 输出时一致。  
5. 若需与官方 **bit-for-bit** 对齐 FC parser，只需把 `_parse_series` 末尾改回官方两行 `findall` 兜底（不建议在未记录原因时改）。

---

## 10. 代码锚点（便于 diff）

**官方与本地相同的 MC 核心：**

```python
# evaluation/evaluate_qa.py — _normalize_choice（官方 main 同构）
value = _extract_tag_content(value, "answer")
match = re.match(r"\s*([A-Da-d])[\.\)\s-]*", value)
if match:
    return match.group(1).upper()
return value.lower()
```

**本地独有 — `_parse_series` 追加段（官方为全文 findall）：**

```python
prediction_lists = re.findall(
    r'"(?:predictions?|forecast|answer)"\s*:\s*(\[[^\[\]]+\])', ...)
# ... bracket_lists 倒序 ...
return []  # 官方此处为: return [float(n) for n in re.findall(...)]
```

---

## 11. 收束

| 组件 | 官方 GitHub | 本仓库当前 | 是否影响 ST-Test 多选 ACC |
|---|---|---|---|
| `_extract_tag_content` | tag-first | 相同 | 否（同逻辑） |
| `_normalize_choice` | tag + 开头 A–D | 相同 | 否 |
| `_parse_series` | 全文数字兜底 | 结构化 `[]` / JSON 键 | 否（仅 forecasting） |
| `evaluate.py` | 标准 CLI | + 注释 | 否 |
| 复现脚本 strict parser | 无 | SmartTest 专用 | **是**（若误用 SmartTest 解析去比 ACC） |

**最终答案：** 相对 [LingFengGold/STReasoner](https://github.com/LingFengGold/STReasoner) 官方 evaluate，本仓库 **没有** 改选择题 parser；**只** 改窄了 forecasting 的数字抽取。历史上 agent 曾设想放宽 MC parser，但 **当前 evaluate 已与官方 tag-first 对齐**。
