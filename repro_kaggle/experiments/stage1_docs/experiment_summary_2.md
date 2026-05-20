# 实验一：不同精度推理资源测试

## 当前停止点

- 本轮已完成 `4bit_single`，并通过配置后健康检查。
- 本轮 `8bit_single` 在 model loading 阶段失败，原因是当前工作区缺少 `repro_kaggle/scripts/03_load_streasoner_smoke.py`。
- 静态核查发现对应文件当前位于 `repro_kaggle/00_smoke_test_scripts/03_load_streasoner_smoke.py`，因此该失败是脚本路径/工作区文件移动问题，不是 8bit 显存结论。
- 按 prompt2 的规则，出现加载失败、decoded_text 缺失和关键字段缺失后立即停止；本轮没有继续运行 `fp16_single` 和 `fp16_dual`。
- 最小修改建议：恢复 `repro_kaggle/scripts/` 目录，或让 `run_experiment1_new_version.py` 的 `import_repro_loader()` / `import_timeseries_patch_module()` 兼容 `repro_kaggle/00_smoke_test_scripts/` 作为 fallback，然后从 `8bit_single` 重新开始。

## 样本与目录

- SmartTest 样例：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/smart_test/SmartTest.jsonl`，共 2 条，任务分布 `{"forecasting": 1, "entity": 1}`。
- 新运行脚本：`repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py`
- 机器可读结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource`
- 官方评测逻辑：`evaluation/evaluate_qa.py`

## 证据查找

### 作者代码中的精度设置

- 训练脚本使用 fp16：
  - `scripts/qwen3-8b/train_stage1.sh:24`：`--fp16`
  - `scripts/qwen3-8b/train_stage2.sh:24`：`--fp16`
- 推理脚本中的 dtype：
  - `inference/llm_utils.py:97-98`：普通 vLLM worker 使用 `LLM(..., dtype='half')`，对应 fp16/half。
  - `inference/llm_utils.py:142-149`：time-series vLLM worker 未显式设置 dtype；实际 dtype 需要通过本实验记录加载后模型参数 dtype、量化配置和日志确认。

### 作者代码中的 max_new_tokens / max_tokens 设置

- 作者 vLLM 推理使用 `max_tokens=512`：
  - `inference/inference_tsmllm_vllm.py:64-68`：`SamplingParams(max_tokens=512, temperature=0.2)`
- 本实验使用 Hugging Face `generate`，对应参数记录为 `max_new_tokens=2048`。
- Hugging Face 的 `max_new_tokens` 表示最多生成的新 token 数，不包含 prompt tokens。


## 分层口径

- A. Run Layer：只负责加载模型、构造输入、generate、decode、记录资源。
- B. Strict Diagnostic Layer：只诊断输出是否符合我们希望的机器可解析格式。
- C. Official Eval Layer：生成 `evaluate_qa.py` 可读取的 `generated_answer*.json`，并复用作者的解析和指标函数。

## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | SmartTest 2 条：1 条 forecasting + 1 条非 forecasting |
|   batch size   |                1                |
| max_new_tokens |               2048               |

|           | 指标 | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |
| --------- | ---- | -------: | -------: | -------: | ------- |
| 配置证据 | 加载方式 | 4bit | 8bit |  |  |
|  | device_map | {"": 0} |  |  |  |
|  | 实际模型分布 | {"label": "single_gpu", "devices": ["0"], "cuda_devices": ["0"], "has_cpu_offload": false, "has_disk_offload": false} |  |  |  |
|  | is_cpu_offload | False |  |  |  |
|  | use_cache | False |  |  |  |
| 可运行证据 | input tokens（平均值） | 697.000 |  |  |  |
|  | actual new tokens（平均值） | 2048.000 |  |  |  |
|  | load 成功 | True | False |  |  |
|  | generate 成功率 | 1.000 | 0.000 |  |  |
| 资源与速度 | GPU 总显存 | {"gpu0": 14.563} | {"gpu0": 14.563} |  |  |
|  | load 后显存 | {"gpu0": {"allocated_gib": 5.703, "reserved_gib": 6.74}} |  |  |  |
|  | generate 峰值显存 | {"gpu0": {"max_allocated_gib": 6.038, "max_reserved_gib": 6.744}} | {} |  |  |
|  | 平均延迟与最高延迟 | 3507.756 |  |  |  |
|  | tokens/s | 0.599 |  |  |  |
| 输出与评测 | decode 成功率 | 1.000 | 0.000 |  |  |
|  | strict diagnostic 成功率 | 0.000 | 0.000 |  |  |
|  | official choice accuracy | 0.000 |  |  |  |
|  | official forecasting MAE | 113.526 |  |  |  |
|  | official forecasting MAPE | 87.618 |  |  |  |
| 失败阶段、失败原因 | 失败阶段、详细失败原因 |  | model_loading 阶段失败：FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/STReasoner_reproduce/repro_kaggle/scripts/03_load_streasoner_smoke.py' |  |  |

## 说明

- 不再使用 `official_accuracy` / `official_parse_success_rate` 这类误导命名。
- forecasting 官方口径只报告 MAE / MAPE / coverage，不报告 exact accuracy。
- `strict diagnostic 成功率` 是我们自己的格式诊断，不等同于作者官方 parse 或 evaluation。
