# Stage2 SmartTest fp16 A100 单卡运行报告

## 结论摘要

本轮 stage2 SmartTest 在 AutoDL A100 环境下完成了两条样例的逐条运行：

- non_forecasting 样例：模型加载、generate、decode 均成功；严格 `<answer>...</answer>` 解析失败；官方选择题评测覆盖率为 1.0，accuracy 为 0.0。
- forecasting 样例：模型加载、generate、decode 均成功；严格 `<answer>...</answer>` 解析失败；官方 forecasting 评测覆盖率为 1.0，MAE 为 123.526，MAPE 为 96.6623%。
- fp16 单卡链路已打通：`torch.float16`、`cuda:0`、`flash_attention_2`、无 quantization、无 CPU/disk offload。
- 当前主要问题不是加载或显存，而是输出没有严格生成唯一 `<answer>...</answer>` 标签，导致脚本 strict parser 标记为 `parse_failed`。

本轮未运行 tiny20 全量、paper cases、stress，也未执行 run-all。

## 环境

| 项目 | 记录 |
| --- | --- |
| 机器 | AutoDL A100 |
| GPU | NVIDIA A100-SXM4-80GB |
| Python | 3.10.16 |
| PyTorch | 2.6.0+cu126 |
| CUDA | 12.6 |
| flash_attn | 2.7.2.post1 |
| attention backend | `flash_attention_2` |
| HF_HOME | `/cloud/cloud-ssd1/hf_cache` |
| TRANSFORMERS_CACHE | `/cloud/cloud-ssd1/hf_cache` |
| HF_HUB_CACHE | `/cloud/cloud-ssd1/hf_cache` |

运行时 `flash_attn import check: PASS`，说明此前的 flash-attn 动态库导入问题已经解决。

## 运行配置

| 项目 | 记录 |
| --- | --- |
| 模型 | `Time-HD-Anonymous/STReasoner-8B` |
| 配置名 | `fp16_a100_single` |
| batch size | 1 |
| max_new_tokens | 2048 |
| CUDA_VISIBLE_DEVICES | `0` |
| device_map | `{"": 0}` |
| dtype | `torch.float16` |
| quantization_config | None |
| CPU offload | disabled |
| disk offload | disabled |

模型实际分布为 `single_gpu`，未出现 CPU offload 或 disk offload。

## 样例与输出文件

| case | sample_id | task | source_file | original_index | 输出文件 |
| --- | --- | --- | --- | ---: | --- |
| non_forecasting | `tiny20_entity_02_line529` | entity | `ST-Test/entity_test.jsonl` | 529 | `repro_autodl/experiments/stage2_results/experiment1_smarttest/non_forecasting_prediction.jsonl` |
| forecasting | `tiny20_forecasting_01_line87` | forecasting | `ST-Test/forecasting_test.jsonl` | 87 | `repro_autodl/experiments/stage2_results/experiment1_smarttest/forecasting_prediction.jsonl` |

对应 summary 文件：

- `repro_autodl/experiments/stage2_results/experiment1_smarttest/non_forecasting_summary.json`
- `repro_autodl/experiments/stage2_results/experiment1_smarttest/forecasting_summary.json`

对应 log 文件：

- `repro_autodl/experiments/stage2_results/experiment1_smarttest/non_forecasting_run.log`
- `repro_autodl/experiments/stage2_results/experiment1_smarttest/forecasting_run.log`

## non_forecasting 结果

### 加载与资源

| 指标 | 值 |
| --- | ---: |
| load success | true |
| model load time | 412.834 sec |
| load 后 allocated | 15.384 GiB |
| load 后 reserved | 15.480 GiB |
| generate 前 allocated | 15.384 GiB |
| generate 后 allocated | 15.392 GiB |
| generate 峰值 allocated | 15.521 GiB |
| generate 峰值 reserved | 15.707 GiB |

第一次运行包含模型权重下载，4 个 safetensors shard 下载总耗时约 6 分 37 秒，因此 load time 明显偏长。

### 生成与评测

| 指标 | 值 |
| --- | ---: |
| input_tokens | 394 |
| actual_new_tokens | 893 |
| latency_sec | 77.181 |
| tokens_per_sec | 11.570 |
| generate_success | true |
| decode_success | true |
| strict parse_success | false |
| failure_type | `parse_failed` |
| first_error | `expected_exactly_one_answer_tag_got_0` |
| official coverage | 1.0 |
| official accuracy | 0.0 |

解释：

- 模型已成功生成 893 个新 token，且 decode 成功。
- strict parser 没有找到 `<answer>...</answer>`，所以记录为 `parse_failed`。
- 官方评测仍然执行，说明 prediction 文件中有可评测文本；但 entity 选择题 accuracy 为 0.0。
- 当前需要查看 `non_forecasting_prediction.jsonl` 中的 `decoded_text/raw_response`，确认模型是否输出了长推理但没有最终答案标签，或者答案不是 B。

## forecasting 结果

### 加载与资源

| 指标 | 值 |
| --- | ---: |
| load success | true |
| model load time | 44.334 sec |
| load 后 allocated | 15.384 GiB |
| load 后 reserved | 15.480 GiB |
| generate 前 allocated | 15.384 GiB |
| generate 后 allocated | 15.392 GiB |
| generate 后 reserved | 18.824 GiB |
| generate 峰值 allocated | 15.696 GiB |
| generate 峰值 reserved | 18.824 GiB |

第二次运行复用了本地缓存权重，因此模型加载时间从 412.834 秒下降到 44.334 秒。日志中出现 Hugging Face HEAD 请求 read timeout 并自动 retry，最终 processor/tokenizer 加载成功。

### 生成与评测

| 指标 | 值 |
| --- | ---: |
| input_tokens | 1000 |
| actual_new_tokens | 2048 |
| latency_sec | 375.926 |
| tokens_per_sec | 5.448 |
| generate_success | true |
| decode_success | true |
| strict parse_success | false |
| failure_type | `parse_failed` |
| first_error | `expected_exactly_one_answer_tag_got_0` |
| official coverage | 1.0 |
| official MAE | 123.526 |
| official MAPE | 96.6623 |
| target mean | 125.926 |
| target min/max | 72.17 / 226.55 |

解释：

- forecasting 样例实际生成 token 数正好等于 `max_new_tokens=2048`，说明生成打满上限，可能没有自然停止或没有输出符合预期的结束格式。
- strict parser 没有找到 `<answer>...</answer>`，所以记录为 `parse_failed`。
- 官方 forecasting 评测仍然覆盖 1 条样例，但 MAE 和 MAPE 都较高，说明官方解析到的数值与 gold 差距较大。
- 当前需要查看 `forecasting_prediction.jsonl` 中的 `decoded_text/raw_response`，确认模型输出中是否包含大量解释文本、重复文本或未闭合答案。

## 两次运行对比

| 指标 | non_forecasting | forecasting |
| --- | ---: | ---: |
| input_tokens | 394 | 1000 |
| actual_new_tokens | 893 | 2048 |
| latency_sec | 77.181 | 375.926 |
| tokens_per_sec | 11.570 | 5.448 |
| peak allocated GiB | 15.521 | 15.696 |
| peak reserved GiB | 15.707 | 18.824 |
| strict parse_success | false | false |
| official coverage | 1.0 | 1.0 |

观察：

- forecasting 输入更长，且生成打满 2048 token，因此耗时显著更长，tokens/s 更低。
- 两条样例都没有触发显存不足；A100 80GB 上 fp16 单卡推理资源余量充足。
- 两条样例都失败在 strict parser，而不是模型加载、generate 或 decode。

## 当前判断

本轮 stage2 的目标是验证 AutoDL A100 fp16 单卡推理链路。按运行日志看，链路已经打通：

- flash-attn 可导入；
- 模型权重可加载；
- 单卡 fp16 无 offload；
- 两条 SmartTest 样例均完成 generate/decode；
- 结果文件和 summary 文件均已写出。

但输出格式存在问题：

- strict parser 期望唯一 `<answer>...</answer>`；
- 两条样例均为 `expected_exactly_one_answer_tag_got_0`；
- non_forecasting 官方 accuracy 为 0；
- forecasting 官方 MAE/MAPE 较差，且生成打满 token 上限。

## 下一步建议

1. 先打开两条 prediction jsonl，查看 `decoded_text` 或 `raw_response` 的实际内容。
2. 如果模型输出有答案但没有 `<answer>` 标签，下一轮可考虑在 prompt/input 构造中追加答案格式约束。
3. 如果 forecasting 输出明显跑偏或重复，下一轮应单独分析 max_new_tokens、EOS、generation_config 和 prompt 模板。
4. 当前不要扩大到 tiny20 全量；先把这两条 SmartTest 的输出格式问题定位清楚。
