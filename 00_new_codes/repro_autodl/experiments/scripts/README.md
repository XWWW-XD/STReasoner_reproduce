# `repro_autodl/experiments/scripts/`

AutoDL 上跑实验的入口（GPU 推理、训练 smoke、graph 消融）。分析只读脚本在 [`../../../scripts/`](../../../scripts/README.md)。

## Stage 2 复现

| 文件 | 作用 |
| --- | --- |
| `stage2_2_run_paper_cases.py` | Stage 2.2 paper_cases 4 条（HF generate，`max_tokens=6144`） |
| `stage2_4_graph_ablation_paper_cases.py` | Stage 2.4 paper_cases 图结构 ±（官方 vLLM + evaluate） |
| `stage2_4_graph_ablation_sttest.py` | Stage 2.4 ST-Test 全量 graph 消融薄封装 |
| `stage2_5_prepare_paper_cases.py` | 将 `PaperCases.jsonl` 拆成 per-task 单样本 JSONL |
| `stage2_5_collect_metrics.py` | 收集各 checkpoint 的 `evaluation_metrics.json` 对比 |

## Qwen3-4B 单卡训练链路

服务器1 Stage 1 LoRA 分段实验已归档（2026-06-04）；`stage1_lora_*` / `stage1_adapter_*` 脚本已删，见 `reports/t3-autodl2…/16`、`17`。后续训练见 `t3-autodl2…/25-服务器2新策略…`。

| 文件 | 作用 |
| --- | --- |
| `single_a100_qwen3_4b_env_check.sh` | GPU / 磁盘 / Python 环境检查 |
| `single_a100_prepare_qwen3_4b_ts.sh` | 校验权重、拷贝 TS config、`initial_model.py` |
| `single_a100_download_stbench_train_data.sh` | 下载 ST-Bench 训练数据 |
| `single_a100_qwen3_4b_model_check.py` | 检查本地 Qwen3-4B 权重是否齐全 |
| `single_a100_qwen3_4b_stage1_smoke.sh` | Stage 1 full finetune 10-step smoke（服务器1勿重复，见报告 15） |
| `single_a100_qwen3_4b_stage2_smoke.sh` | Stage 2（ST-CoT）smoke |
| `single_a100_qwen3_4b_stage3_smoke.sh` | Stage 3 smoke |
