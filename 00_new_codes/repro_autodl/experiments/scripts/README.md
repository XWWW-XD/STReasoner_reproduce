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

| 文件 | 作用 |
| --- | --- |
| `single_a100_qwen3_4b_env_check.sh` | GPU / 磁盘 / Python 环境检查 |
| `single_a100_prepare_qwen3_4b_ts.sh` | 校验权重、拷贝 TS config、`initial_model.py` |
| `single_a100_download_stbench_train_data.sh` | 下载 ST-Bench 训练数据 |
| `single_a100_qwen3_4b_model_check.py` | 检查本地 Qwen3-4B 权重是否齐全 |
| `single_a100_qwen3_4b_stage1_smoke.sh` | Stage 1 full finetune 10-step smoke |
| `single_a100_qwen3_4b_stage1_lora_smoke.sh` | Stage 1 LoRA 短步 smoke |
| `single_a100_qwen3_4b_stage1_lora_100steps.sh` | LoRA 100 steps |
| `single_a100_qwen3_4b_stage1_lora_500steps.sh` | LoRA 500 steps |
| `single_a100_qwen3_4b_stage1_lora_continue_500to1000.sh` | 500→1000 续训 |
| `single_a100_qwen3_4b_stage1_lora_continue_1000to1500.sh` | 1000→1500 续训 |
| `single_a100_qwen3_4b_stage1_lora_continue_1500to2000_save_state.sh` | 1500→2000，保留 trainer state |
| `single_a100_qwen3_4b_stage1_lora_resume_2000to2500_save_state.sh` | 从 checkpoint 恢复训到 2500 |
| `single_a100_qwen3_4b_stage1_lora_preflight_2000to2500.sh` | 2500 续训前检查 |
| `single_a100_qwen3_4b_stage1_lora_2500_postcheck.sh` | 2500 训后 adapter 加载检查 |
| `single_a100_qwen3_4b_stage1_adapter_load_check.py` | Stage 1 adapter 最小加载检查 |
| `single_a100_qwen3_4b_stage1_adapter_generate_check.py` | adapter 最小 generate 路径验证 |
| `single_a100_qwen3_4b_stage1_adapter_probe.py` | 小批量 ST-Align 生成探测 |
| `single_a100_qwen3_4b_stage2_smoke.sh` | Stage 2（ST-CoT）smoke |
| `single_a100_qwen3_4b_stage3_smoke.sh` | Stage 3 smoke |
