# 12 Evaluation 流程

本文只做静态代码阅读和已有结果目录检查，不重新训练、不重新推理、不启动大模型。

结论标记约定：

- `已从代码确认`：能直接绑定到文件、函数、行号。
- `根据代码推断，未由真实运行验证`：来自静态代码和文件结构，但没有本次重新执行验证。
- `尚未确认`：需要真实数据、运行结果或论文对照才能确认。

## 1. 入口命令

`已从代码确认`：官方 README 的 evaluation 命令是对四个 reasoning task 循环调用 `evaluation/evaluate.py`，只传 `--task` 和 `--exp_path`，见 `README.md:161-170`。

```bash
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do
    python evaluation/evaluate.py \
        --task $task \
        --exp_path exp/$task-qwen3_8b_grpo_stage1+2+3_w_spatial
done
```

`已从代码确认`：`evaluation/evaluate.py:100-179` 是 CLI 主入口。它做四件事：

1. 解析参数：`--exp_path`、`--dataset`、`--task`、`--pred_pattern`、`--repo_root`，见 `evaluation/evaluate.py:100-132`。
2. 解析 repo root、实验目录和 dataset 路径，见 `evaluation/evaluate.py:134-158`。
3. 调用 `load_jsonl_dataset()` 和 `load_prediction_files()` 读取标准答案和预测，见 `evaluation/evaluate.py:160-161`。
4. 调用 `evaluate_predictions_for_task()` 计算指标，追加 token stats，然后写出 `evaluation_metrics.json`，见 `evaluation/evaluate.py:163-175`。

`已从代码确认`：当前 `DEFAULT_TASK_CONFIG` 默认数据路径是 `data/reasoning/*.jsonl` 和 `data/alignment/alignment_test.jsonl`，见 `evaluation/evaluate.py:56-81`。但本仓库数据注册表中的 ST-Test 路径是 `data/ST-Bench/ST-Test/*.jsonl`，见 `data/dataset_info.json:114-145`。

`尚未确认`：README 的 evaluation 命令没有显式传 `--dataset`。如果本地只通过 `download_dataset.py` 下载到 `data/ST-Bench/`，而没有额外创建 `data/reasoning/`，默认评估路径可能找不到数据。是否官方环境中还会额外放置 `data/reasoning/`，本次未验证。

## 2. 参数说明

| 参数 | 是否必需 | 默认值 | 代码行为 | 证据 |
|---|---:|---|---|---|
| `--exp_path` | 是 | 无 | 指向实验输出目录；相对路径会拼到 `repo_root` 下；目录不存在则抛 `FileNotFoundError` | `evaluation/evaluate.py:102-107`, `evaluation/evaluate.py:139-146` |
| `--dataset` | 否 | `None` | 如果传入就作为评估 JSONL；如果不传，则按 `--task` 从 `DEFAULT_TASK_CONFIG` 取默认路径 | `evaluation/evaluate.py:108-113`, `evaluation/evaluate.py:148-156` |
| `--task` | 是 | 无 | 控制默认 dataset 和指标分支；支持 alignment、forecasting、多选类 reasoning | `evaluation/evaluate.py:114-119`, `evaluation/evaluate_qa.py:320-330` |
| `--pred_pattern` | 否 | `generated_answer` | 在 `exp_path` 目录下筛选文件名包含该子串、且以 `.json` 结尾的预测文件 | `evaluation/evaluate.py:120-125`, `evaluation/evaluate_qa.py:82-90` |
| `--repo_root` | 否 | `evaluation/` 的上一级 | 用来解析相对 `exp_path` 和相对 `dataset`；不传时自动取项目根目录 | `evaluation/evaluate.py:126-137`, `evaluation/evaluate.py:156` |

`已从代码确认`：`resolve_path(path, repo_root)` 只做“如果是绝对路径就原样返回，否则拼接 repo_root”，见 `evaluation/evaluate.py:84-87`。

`已从代码确认`：`evaluate.py` 没有 `--max_samples` 参数，也没有只评估前 N 条的内置选项；所有样本量由传入的 dataset 文件和 prediction 文件决定，见参数定义 `evaluation/evaluate.py:100-132`。

## 3. Prediction 文件格式

`已从代码确认`：当前推理脚本写出的 prediction 文件默认是 `generated_answer.json`，每个条目包含 `idx`、`question_text`、`response`、`num_tokens`，见 `inference/inference_tsmllm_vllm.py:308-321`。

当前代码期望的扁平格式：

```json
[
  {
    "idx": 0,
    "question_text": "...",
    "response": "<think>...</think><answer>...</answer>",
    "num_tokens": 1234
  }
]
```

`已从代码确认`：`load_prediction_files(exp_dir, pattern)` 会扫描 `exp_dir` 下所有文件名包含 `pattern` 且以 `.json` 结尾的文件；如果 JSON 顶层是 `{"results": [...]}`，则改读 `results` 字段，见 `evaluation/evaluate_qa.py:82-100`。

`已从代码确认`：对每个 prediction entry，评估只强依赖 `idx` 和预测文本。预测文本支持两种结构：

1. 扁平结构：entry 直接有 `"response"` 字段，见 `evaluation/evaluate_qa.py:107-110`。
2. COT-style 结构：entry 有 `"responses"` list，按 `attempt` 排序后取第一个 response，见 `evaluation/evaluate_qa.py:111-118`。

`已从代码确认`：`load_prediction_files()` 会对预测文本调用 `_extract_tag_content(text, "answer")`，优先抽取 `<answer>...</answer>` 内部内容；因此后续指标一般看到的是最终答案而不是完整 CoT，见 `evaluation/evaluate_qa.py:36-45`、`evaluation/evaluate_qa.py:123-127`。

`已从代码确认`：本地已有 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json` 是扁平 list 格式，条目字段为 `idx/question_text/response`，第一条样例见 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:1-5`。

`已从代码确认`：本地已有四个 `generated_answer.json` 都没有 `num_tokens` 字段。静态 JSON 解析显示四个目录的首条 keys 都是 `idx,question_text,response`，且 `HasNumTokens=False`。

`根据代码推断，未由真实运行验证`：已有结果可能来自旧版推理脚本或不同导出逻辑；当前 `inference/inference_tsmllm_vllm.py` 会写 `num_tokens`，见 `inference/inference_tsmllm_vllm.py:311-316`。

## 4. 指标计算逻辑

### 4.1 Dataset 读取

`已从代码确认`：`load_jsonl_dataset(path)` 逐行读取 JSONL，空行跳过，并给每条样本补默认 `idx=行号`，见 `evaluation/evaluate_qa.py:69-79`。

`已从代码确认`：评估函数用 dataset 顺序中的枚举下标 `idx` 来找 prediction，而不是用样本内部 `sample["idx"]`。例如 forecasting 和 multiple choice 都是 `for idx, sample in enumerate(dataset)` 后 `predictions.get(idx)`，见 `evaluation/evaluate_qa.py:213-216`、`evaluation/evaluate_qa.py:286-289`。

`根据代码推断，未由真实运行验证`：如果 prediction 的 `idx` 与 dataset 行号不一致，评估会把预测对错样本，或者把预测视为 missing。

### 4.2 Forecasting

`已从代码确认`：`reasoning_forecasting` 走 `evaluate_forecasting_predictions()`，见 `evaluation/evaluate_qa.py:198-275` 和分发函数 `evaluation/evaluate_qa.py:320-327`。

Forecasting 逻辑：

1. 用 `_parse_series()` 把标准答案 `sample["output"]` 和预测文本解析成 float list，见 `evaluation/evaluate_qa.py:47-66`、`evaluation/evaluate_qa.py:213-216`。
2. 如果 target 或 prediction 解析不到数值，则记为 missing，见 `evaluation/evaluate_qa.py:218-221`。
3. 如果预测长度短于 target，用最后一个预测值 padding；如果长于 target，则截断，见 `evaluation/evaluate_qa.py:223-227`。
4. 逐点计算绝对误差并取当前样本 MAE，再对样本 MAE 求平均，见 `evaluation/evaluate_qa.py:233-237` 和 `evaluation/evaluate_qa.py:258-265`。
5. 对非零 target 计算 MAPE，跳过接近 0 的 target 值，见 `evaluation/evaluate_qa.py:239-248` 和 `evaluation/evaluate_qa.py:258-265`。
6. 额外记录 target 的 mean、abs_mean、min、max、total_values，见 `evaluation/evaluate_qa.py:252-273`。

`已从代码确认`：当前代码存在 `mae`、`mape` 和 `target_stats` 指标，见 `evaluation/evaluate_qa.py:258-273`。

`已从代码确认`：已有 forecasting metrics 文件只包含 `mae`，没有 `mape` 和 `target_stats`，见 `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/evaluation_metrics.json:2-8`。

`尚未确认`：已有 forecasting metrics 是否由旧版 `evaluate_qa.py` 生成，或者是否由人工裁剪过，本次没有运行评估验证。

### 4.3 多选 reasoning

`已从代码确认`：`reasoning_entity`、`reasoning_etiological`、`reasoning_correlation`、`reasoning_causal` 都走 `evaluate_multiple_choice_predictions()`，见 `evaluation/evaluate_qa.py:278-317` 和 `evaluation/evaluate_qa.py:326-329`。

多选逻辑：

1. target 来自 `sample["output"]`，prediction 来自 `predictions[idx]`，见 `evaluation/evaluate_qa.py:286-289`。
2. `_normalize_choice()` 会先抽取 `<answer>` 内容，再用正则读取开头的 A-D 字母并转成大写，见 `evaluation/evaluate_qa.py:23-33`。
3. 标准化后的 prediction 与 target 相等则计为 correct，见 `evaluation/evaluate_qa.py:293-300`。
4. 输出 `task/total_samples/evaluated_samples/missing_predictions/coverage/accuracy`，见 `evaluation/evaluate_qa.py:309-316`。

`已从代码确认`：代码中的多选指标名是 `accuracy`，不是大写 `ACC`。本地已有 metrics 也使用 `"accuracy"` 键，见 `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/evaluation_metrics.json:2-7`、`exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/evaluation_metrics.json:2-7`、`exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/evaluation_metrics.json:2-7`。

### 4.4 Alignment

`已从代码确认`：`alignment` 任务走 `evaluate_alignment_predictions()`，见 `evaluation/evaluate_qa.py:142-195` 和 `evaluation/evaluate_qa.py:323-325`。

`已从代码确认`：alignment 输出指标包括 `overall_score`、`exact_match`、`relative_accuracy`，见 `evaluation/evaluate_qa.py:185-194`。

`尚未确认`：alignment 的数值相对分支在当前代码中存在可疑缩进：当 `abs(target_float) > 1e-6` 时只计算 `rel_error`，没有把 rel score 加入 `overall_sum`；`rel_score` 的更新位于 `else` 分支内，见 `evaluation/evaluate_qa.py:168-176`。本次任务不运行 alignment，因此没有验证实际影响。

### 4.5 Token Stats

`已从代码确认`：`extract_token_stats()` 会扫描预测文件，读取每条 entry 的 `num_tokens`，输出 `total_input_tokens`、`avg_input_tokens`、`samples_with_token_info`，见 `evaluation/evaluate.py:13-53`。

`已从代码确认`：只有当至少一个 entry 有数值型 `num_tokens` 时，token stats 才会并入 metrics；否则返回空 dict，见 `evaluation/evaluate.py:40-53`、`evaluation/evaluate.py:165-168`。

`已从代码确认`：本地已有 `exp_STReasoner-8B` 的 `generated_answer.json` 没有 `num_tokens` 字段，因此已有 `evaluation_metrics.json` 也没有 token stats。

## 5. 已有结果目录

`已从代码确认`：本地已有结果目录是 `exp_STReasoner-8B/`，包含四个任务子目录；每个子目录都有 `generated_answer.json` 和 `evaluation_metrics.json`。

| 目录 | generated entries | metrics 内容 | 可用于汇报 |
|---|---:|---|---|
| `exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/` | 280 | `mae=65.5934736377407`, `coverage=1.0` | forecasting 定量结果和若干预测案例 |
| `exp_STReasoner-8B/reasoning_entity-STReasoner-8B/` | 1194 | `accuracy=0.7571189279731994`, `coverage=1.0` | entity 多选准确率和案例 |
| `exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/` | 207 | `accuracy=0.9565217391304348`, `coverage=1.0` | etiological 多选准确率和案例 |
| `exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/` | 1592 | `accuracy=0.8712311557788944`, `coverage=1.0` | correlation 多选准确率和案例 |

证据：

- metrics 文件行：`exp_STReasoner-8B/reasoning_forecasting-STReasoner-8B/evaluation_metrics.json:2-8`、`exp_STReasoner-8B/reasoning_entity-STReasoner-8B/evaluation_metrics.json:2-7`、`exp_STReasoner-8B/reasoning_etiological-STReasoner-8B/evaluation_metrics.json:2-7`、`exp_STReasoner-8B/reasoning_correlation-STReasoner-8B/evaluation_metrics.json:2-7`。
- generated 文件字段样例：`exp_STReasoner-8B/reasoning_entity-STReasoner-8B/generated_answer.json:1-5`。

`根据代码推断，未由真实运行验证`：不重新训练时，可以直接复用这些已有结果做组会汇报：

1. 定量页：展示 entity / etiological / correlation 的 `accuracy`，forecasting 的 `mae`。
2. 定性页：从 `generated_answer.json` 选 1-2 条样例，展示题面中的 `Graph Structure`、`<ts><ts/>` 占位符、模型 `<think>` 和 `<answer>`。
3. 诚实说明：这些结果是仓库自带或本地已有结果，不是本周重新训练或重新推理得到；当前代码新增的 `mape` 和 token stats 没有出现在已有 metrics 中。

## 6. 小规模验证建议

`已从代码确认`：evaluation 本身是轻量脚本，不加载模型、不启动 vLLM、不跑训练。它只读 JSONL dataset 和 JSON prediction，然后写 `evaluation_metrics.json`，见 `evaluation/evaluate.py:160-174`。

如果 ST-Bench 已经下载，并且只想验证一个已有结果目录，可以显式传 `--dataset`，避免默认 `data/reasoning/*.jsonl` 路径问题：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset data/ST-Bench/ST-Test/entity_test.jsonl \
  --exp_path exp_STReasoner-8B/reasoning_entity-STReasoner-8B
```

`根据代码推断，未由真实运行验证`：上面命令会覆盖该目录下的 `evaluation_metrics.json`，因为输出路径固定为 `os.path.join(exp_dir, "evaluation_metrics.json")`，见 `evaluation/evaluate.py:172-174`。如果要保留已有 metrics，建议使用单独的临时 `exp_path` 目录做 sanity check。

`已从代码确认`：脚本没有 `--max_samples`。真正的小规模 sanity check 需要准备一个很小的 JSONL dataset 和一个只含对应 `idx` 的 `generated_answer.json`；评估会按小 dataset 的行数计算指标，见 `evaluation/evaluate_qa.py:69-79`、`evaluation/evaluate_qa.py:82-127`。

小规模 sanity check 的最小结构：

```text
tmp_eval/
  tiny_test.jsonl
  tiny_exp/
    generated_answer.json
```

多选任务的 `tiny_test.jsonl` 每行至少需要：

```json
{"input": "question text", "output": "A"}
```

对应 `tiny_exp/generated_answer.json`：

```json
[
  {"idx": 0, "question_text": "question text", "response": "<answer>A</answer>"}
]
```

命令：

```bash
python evaluation/evaluate.py \
  --task reasoning_entity \
  --dataset tmp_eval/tiny_test.jsonl \
  --exp_path tmp_eval/tiny_exp
```

`根据代码推断，未由真实运行验证`：forecasting sanity check 也类似，只是 `output` 和 `<answer>` 里放数值序列，例如 `[1.0, 2.0, 3.0]`。

## 7. 组会讲法

### 本文档核心结论

1. `已从代码确认`：evaluation 入口是 `evaluation/evaluate.py`，核心指标在 `evaluation/evaluate_qa.py`。
2. `已从代码确认`：prediction 文件按 `idx` 对齐 dataset 行号；评估优先抽取 `<answer>...</answer>`，不会直接比较完整 CoT。
3. `已从代码确认`：多选任务输出 `accuracy`；forecasting 当前代码输出 `mae`、`mape` 和 target stats；token stats 依赖 prediction entry 中的 `num_tokens`。
4. `已从代码确认`：已有 `exp_STReasoner-8B` 可以直接提供四个任务的已有结果：entity 75.71%，etiological 95.65%，correlation 87.12%，forecasting MAE 65.59。
5. `尚未确认`：已有 metrics 没有 MAPE 和 token stats，和当前代码不完全一致；需要重新运行 evaluation 才能得到当前代码版本的完整指标。

### 组会可讲版本

评估流程很直接：推理阶段生成 `generated_answer.json`，里面每条样本有 `idx` 和模型 `response`。评估脚本读取测试集 JSONL，再按 `idx` 把预测和标准答案对齐。对于多选任务，它从 `<answer>` 里抽取 A-D 选项并计算 accuracy；对于 forecasting，它把 `<answer>` 里的数字序列解析出来，和标准序列对齐长度后计算 MAE，当前代码还会计算 MAPE。

我没有重新训练或重新推理，但本地已有 `exp_STReasoner-8B` 四个任务结果，可以用于组会展示现有 checkpoint 的表现和输出格式。需要强调的是：这些已有 metrics 缺少当前代码支持的 MAPE 和 token stats，所以更适合作为代码阅读阶段的参考结果，不应说成本周完整复现实验结果。

### 后续需要验证的问题

1. 下载 ST-Bench 后，确认 README 默认 evaluation 命令是否能找到 dataset；如果不能，需要在复现命令中显式加 `--dataset data/ST-Bench/ST-Test/*.jsonl`。
2. 重新运行一次 evaluation，确认当前代码是否会在 forecasting metrics 中写入 `mape` 和 `target_stats`。
3. 用当前推理脚本重新生成小样本 prediction，确认 `num_tokens` 是否进入 `generated_answer.json`，以及 token stats 是否进入 `evaluation_metrics.json`。
4. 检查 alignment 分支的相对误差缩进逻辑，确认是否为 bug，以及是否影响论文主实验。
