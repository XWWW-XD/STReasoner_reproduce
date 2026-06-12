## 2.3 ST-TEST 数据集验证论文真实效果

## 补充下载并检查 ST-Test 四类数据：


| ST-Test 文件                                     | 样例数  |
| ---------------------------------------------- | ---- |
| `data/ST-Bench/ST-Test/correlation_test.jsonl` | 1592 |
| `data/ST-Bench/ST-Test/entity_test.jsonl`      | 1194 |
| `data/ST-Bench/ST-Test/etiological_test.jsonl` | 207  |
| `data/ST-Bench/ST-Test/forecasting_test.jsonl` | 280  |
| 合计                                             | 3273 |


## ST-Test max_new_tokens=512

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

配置记录如下：

- 模型：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B`
- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py`
- GPU：1 张 A100
- CLI 传入 `max_tokens=512`，但后续确认 worker 实际未正确使用 CLI 参数
- `temperature=0.2`
- `num_gpus=1`
- `num_gpus_per_process=1`

输出目录：


| task                    | 输出目录                               |
| ----------------------- | ---------------------------------- |
| `reasoning_entity`      | `exp/sttest_full_entity_512/`      |
| `reasoning_etiological` | `exp/sttest_full_etiological_512/` |
| `reasoning_correlation` | `exp/sttest_full_correlation_512/` |
| `reasoning_forecasting` | `exp/sttest_full_forecasting_512/` |


预跑 evaluate 结果：


| task                    | 样例数  | evaluated | missing | coverage | 指标                                 | total input tokens | avg input tokens |
| ----------------------- | ---- | --------- | ------- | -------- | ---------------------------------- | ------------------ | ---------------- |
| `reasoning_entity`      | 1194 | 1194      | 0       | 1.0      | accuracy `0.742044`                | 495068             | 414.63           |
| `reasoning_etiological` | 207  | 207       | 0       | 1.0      | accuracy `0.937198`                | 79543              | 384.27           |
| `reasoning_correlation` | 1592 | 1592      | 0       | 1.0      | accuracy `0.862437`                | 717765             | 450.86           |
| `reasoning_forecasting` | 280  | 280       | 0       | 1.0      | MAE `67.430537`，MAPE `138.878661%` | 77434              | 276.55           |


raw `<answer>` 格式诊断：


| task                    | 样例数  | 严格 1 对 `<answer>` | 缺少 `<answer>` | `<final_answer>` | 空输出 |
| ----------------------- | ---- | ----------------- | ------------- | ---------------- | --- |
| `reasoning_entity`      | 1194 | 1188              | 5             | 0                | 0   |
| `reasoning_etiological` | 207  | 207               | 0             | 0                | 0   |
| `reasoning_correlation` | 1592 | 1566              | 23            | 0                | 0   |
| `reasoning_forecasting` | 280  | 280               | 0             | 0                | 0   |


这里要分清两件事：

- 【已评估增强】评估用的是当前仓库中的 `evaluation/evaluate.py` / `evaluation/evaluate_qa.py`，而 `evaluate_qa.py` 已在前面为 paper_cases 做过通用 parser 增强

## 正式 ST-Test 6144 运行结果

修复 `SamplingParams` 传递后，重新按 `max_tokens=6144` 完整运行 ST-Test 四类数据。本轮正式配置：

- 模型：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B`
- 推理入口：`inference/inference_tsmllm_vllm.py`
- 评估入口：`evaluation/evaluate.py` ，【与源论文保持一致】
- GPU：1 张 A100
- `max_tokens=6144`
- `temperature=0.2`
- `num_gpus=1`
- `num_gpus_per_process=1`

输出目录：


| task                    | 输出目录                                |
| ----------------------- | ----------------------------------- |
| `reasoning_entity`      | `exp/sttest_full_entity_6144/`      |
| `reasoning_etiological` | `exp/sttest_full_etiological_6144/` |
| `reasoning_correlation` | `exp/sttest_full_correlation_6144/` |
| `reasoning_forecasting` | `exp/sttest_full_forecasting_6144/` |


### 当前 evaluate 结果：


| task                    | 样例数  | evaluated | missing | coverage | 指标                                 | total input tokens | avg input tokens |
| ----------------------- | ---- | --------- | ------- | -------- | ---------------------------------- | ------------------ | ---------------- |
| `reasoning_entity`      | 1194 | 1194      | 0       | 1.0      | accuracy `0.747906`                | 495068             | 414.63           |
| `reasoning_etiological` | 207  | 207       | 0       | 1.0      | accuracy `0.956522`                | 79543              | 384.27           |
| `reasoning_correlation` | 1592 | 1592      | 0       | 1.0      | accuracy `0.831658`                | 717765             | 450.86           |
| `reasoning_forecasting` | 280  | 280       | 0       | 1.0      | MAE `68.317056`，MAPE `123.289149%` | 77434              | 276.55           |


### raw `<answer>` 格式与 token 诊断：


| task                    | 样例数  | 严格 1 对 `<answer>` | 缺少 `<answer>` | 空输出 | response token 最大值 | 达到 6144 |
| ----------------------- | ---- | ----------------- | ------------- | --- | ------------------ | ------- |
| `reasoning_entity`      | 1194 | 1175              | 15            | 0   | 6043               | 0       |
| `reasoning_etiological` | 207  | 207               | 0             | 0   | 1060               | 0       |
| `reasoning_correlation` | 1592 | 1545              | 47            | 0   | 6071               | 0       |
| `reasoning_forecasting` | 280  | 278               | 1             | 0   | 6094               | 0       |


另有少量 **标签重复**（如 entity 4 条 `open=2, close=2`），evaluate 可能 parse 失败或异常，需单独看 raw。



### 结果文件

完整 raw 输出在 `exp/sttest_full_*_6144/generated_answer.json`；gold 在 `data/ST-Bench/ST-Test/`（四任务合计 3273 条）。不再维护全量拼接 jsonl。

汇总附件：

```text
00_new_codes/reports/artifacts/sttest_full_6144_summary.json
```

因此，本轮没有省略 ST-Test 的真实输出和正确输出；正文只放汇总，完整逐条证据在附件中。