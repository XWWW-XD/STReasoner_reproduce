## 2.3 ST-TEST 数据集验证论文真实效果

在数据集 ST-Test 上按四类任务运行并评估：

- `reasoning_entity`
- `reasoning_etiological`
- `reasoning_correlation`
- `reasoning_forecasting`

数据集下载后位置和样例数量

| ST-Test 文件 | 样例数 |
|---|---:|
| `data/ST-Bench/ST-Test/correlation_test.jsonl` | 1592 |
| `data/ST-Bench/ST-Test/entity_test.jsonl` | 1194 |
| `data/ST-Bench/ST-Test/etiological_test.jsonl` | 207 |
| `data/ST-Bench/ST-Test/forecasting_test.jsonl` | 280 |
| 合计 | 3273 |

### 推理命令、评估命令

```bash
/root/autodl-tmp/conda/envs/str-py310/bin/python \
  inference/inference_tsmllm_vllm.py \
  --task <task> \
  --model_path /root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B \
  --num_gpus 1 \
  --num_gpus_per_process 1 \
  --max_samples 1 \
  --max_tokens 512 \
  --temperature 0.2 \
  --exp <exp_name> \
  --output_name generated_answer.json
```

评估使用仓库原始 `evaluation/evaluate.py`，但从文件路径直接执行时需要显式加 `PYTHONPATH=.`：

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/str-py310/bin/python \
  evaluation/evaluate.py \
  --task <task> \
  --dataset data/ST-Bench/ST-Test/<task_file>.jsonl \
  --exp_path exp/<exp_name> \
  --pred_pattern generated_answer
```

## ST-Test 6144 运行结果

修复 `SamplingParams` 传递后，重新按 `max_tokens=6144` 完整运行 ST-Test 四类数据。本轮正式配置：

- 模型：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B`
- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- GPU：1 张 A100
- `max_tokens=6144`
- `temperature=0.2`
- `num_gpus=1`
- `num_gpus_per_process=1`

输出目录：

| task | 输出目录 |
|---|---|
| `reasoning_entity` | `exp/sttest_full_entity_6144/` |
| `reasoning_etiological` | `exp/sttest_full_etiological_6144/` |
| `reasoning_correlation` | `exp/sttest_full_correlation_6144/` |
| `reasoning_forecasting` | `exp/sttest_full_forecasting_6144/` |

当前 evaluate 结果：

| task | 样例数 | evaluated | missing | coverage | 指标 | total input tokens | avg input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1194 | 0 | 1.0 | accuracy `0.747906` | 495068 | 414.63 |
| `reasoning_etiological` | 207 | 207 | 0 | 1.0 | accuracy `0.956522` | 79543 | 384.27 |
| `reasoning_correlation` | 1592 | 1592 | 0 | 1.0 | accuracy `0.831658` | 717765 | 450.86 |
| `reasoning_forecasting` | 280 | 280 | 0 | 1.0 | MAE `68.317056`，MAPE `123.289149%` | 77434 | 276.55 |

raw `<answer>` 格式与 token 诊断：

| task | 样例数 | 严格 1 对 `<answer>` | 缺少 `<answer>` | 空输出 | response token 最大值 | 达到 6144 |
|---|---:|---:|---:|---:|---:|---:|
| `reasoning_entity` | 1194 | 1175 | 15 | 0 | 6043 | 0 |
| `reasoning_etiological` | 207 | 207 | 0 | 0 | 1060 | 0 |
| `reasoning_correlation` | 1592 | 1545 | 47 | 0 | 6071 | 0 |
| `reasoning_forecasting` | 280 | 278 | 1 | 0 | 6094 | 0 |

附件里对 raw 的「标签计数」（`answer_tag_open_count` 等）仅作**报告诊断**，与 evaluate 使用同一 tag-first 原则；**coverage 低即 parse 失败**，不存在「strict 失败但 official 仍高分」的口径。

完整 raw 输出与 gold 对齐附件：

```text
00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl
```

该文件共 `3273` 行，每行包含：

- `task`
- `idx`
- `input`
- `gold_output`
- `raw_response`
- `parsed_prediction`
- `parsed_gold`
- `response_tokens_tokenizer`
- `<answer>` / `<final_answer>` tag 计数
- 是否空输出
- 是否达到 `6144`

汇总附件：

```text
00_new_codes/reports/artifacts/sttest_full_6144_summary.json
00_new_codes/reports/artifacts/sttest_full_6144_first3_preview.json
```
