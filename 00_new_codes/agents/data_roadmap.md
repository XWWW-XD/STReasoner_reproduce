# STReasoner 数据与实验流程路线图

## 总览

- 主流程以仓库根目录代码为准：`README.md`、`data_generation/`、`data/`、`scripts/`、`src/`、`inference/`、`evaluation/`。
- `00_new_codes/` 是复现、裁剪、排查和补充脚本区；写正式流程说明时不要只看这里。
- 核心样本字段通常是 `input`、`output`、`timeseries`；推理输出文件使用 `response` 保存模型回答。

## 0. 环境、缓存、模型文件

- 入口 / 说明：
  - `README.md`
  - `requirements.txt`
  - `cache_config.py`
- 模型下载：
  - `download_model.py`
  - `base_model/`
  - `/root/autodl-tmp/cache/huggingface`
- 当前 AutoDL 复现可用模型：
  - `base_model/STReasoner-8B/`
  - HF repo 名：`Time-HD-Anonymous/STReasoner-8B`
- 说明：
  - `Qwen3TSForCausalLM` 是加载架构类名，不是另一个实验模型。
  - tokenizer 的 `.decode()` 是底层 API，不是正式实验阶段。

## 1. 数据下载

- 入口：
  - `download_dataset.py`
- 输出：
  - `data/ST-Bench/`
- 主要子集：
  - `data/ST-Bench/ST-Align/`
  - `data/ST-Bench/ST-SFT/`
  - `data/ST-Bench/ST-CoT/`
  - `data/ST-Bench/ST-RL/`
  - `data/ST-Bench/ST-Test/`
  - `data/ST-Bench/ST-Causal/`

## 2. 数据生成

- 总说明：
  - `data_generation/README.md`
- Stage 1：生成 STS 场景并运行 SDE 模拟
  - `data_generation/run_pipeline.py`
  - `data_generation/demo_sts_sde.py`
  - `data_generation/llm_client.py`
  - 输出到 `data_generation/batch_output/`
- Stage 2：从模拟结果生成 QA
  - `data_generation/generate_alignment_QA.py`
  - `data_generation/generate_reasoning_QA.py`
  - `data_generation/generate_reasoning_forecasting_QA.py`
  - prompt 在 `data_generation/prompts/`
  - 输出到 `data/alignment/`、`data/reasoning_before_filter/`
- Stage 3：过滤样本
  - `data/filter.py`
  - `data/reasoning_before_filter/` -> `data/reasoning/`
- Stage 4：CoT / RL 数据构造
  - `data_generation/generate_cot.py`
  - 依赖一次推理输出 `exp/<exp>/generated_answer.json`
- Stage 5：文本 / 图像变体
  - `data/convert_to_text.py`
  - `data/convert_to_image.py`
  - 输出到 `data/reasoning_text/`、`data/reasoning_image/`

## 3. 数据注册

- 数据集映射：
  - `data/dataset_info.json`
- SFT 训练读取字段：
  - `prompt` -> `input`
  - `response` -> `output`
  - `timeseries` -> `timeseries`

## 4. 基座模型准备

- 下载基座：
  - `download_model.py`
- 加入 STReasoner 自定义代码：
  - `base_model/Config-Qwen3-8B/`
  - `configuration_qwen3_ts.py`
  - `modeling_qwen3_ts.py`
  - `processing_qwen3_ts.py`
- 初始化时间序列编码器：
  - `initial_model.py`

## 5. Stage 1 / Stage 2 SFT

- 训练入口：
  - `scripts/qwen3-8b/train_stage1.sh`
  - `scripts/qwen3-8b/train_stage1+2.sh`
  - 其他模型尺寸在 `scripts/qwen3-14b/`、`scripts/qwen3-4b-instruct/`
- 训练框架：
  - `src/train.py`
  - `src/llamafactory/`
- Stage 1 数据：
  - `alignment`
  - 目标：时间序列对齐
- Stage 2 数据：
  - `entity_cot`
  - `etiological_cot`
  - `correlation_cot`
  - `forecasting_cot`
  - 目标：冷启动推理 / CoT SFT

## 6. Stage 3 RL

- 训练入口：
  - `scripts/qwen3-8b/train_stage1+2+3.sh`
  - `scripts/qwen3-8b/train_stage1+2+3_w_spatial.sh`
- RL 框架：
  - `src/EasyR1/`
  - `src/EasyR1/verl/trainer/main.py`
  - `src/EasyR1/examples/config.yaml`
- 奖励函数：
  - `src/EasyR1/examples/reward_function/str.py`
- RL 数据：
  - `data/ST-Bench/ST-RL/*.jsonl`
- 验证数据：
  - `data/ST-Bench/ST-Test/*.jsonl`

## 7. checkpoint 合并

- 入口：
  - `model_merger.py`
- 输入：
  - `checkpoints/easy_r1/.../actor/`
- 输出：
  - `checkpoints/easy_r1/.../actor/huggingface/`

## 8. 推理

- 入口：
  - `inference/inference_tsmllm_vllm.py`
- vLLM / ChatTS 注册：
  - `inference/vllm/chatts_vllm.py`
  - `inference/llm_utils.py`
- prompt 后缀：
  - `inference/prompt.json`
  - `inference/prompt_utils.py`
- 输入数据：
  - `data/ST-Bench/ST-Test/*.jsonl`
  - `data/ST-Bench/ST-Causal/causal.jsonl`
- 输出：
  - `exp/<task>-<model>/generated_answer.json`
- 推理阶段按代码看是：读数据 -> 组织 prompt 和 timeseries -> 调 vLLM 生成 `response` -> 保存结果。

## 9. 评测

- 入口：
  - `evaluation/evaluate.py`
- 具体指标：
  - `evaluation/evaluate_qa.py`
- 输入：
  - 测试集 JSONL
  - `exp/<task>-<model>/generated_answer.json`
- 输出：
  - `exp/<task>-<model>/evaluation_metrics.json`
- 评测阶段按代码看是：读 gold -> 读 prediction 的 `response` -> 按任务解析答案 -> 计算指标。

## 10. 当前复现实验区

- 复现脚本：
  - `00_new_codes/repro_autodl/experiments/scripts/`
  - `00_new_codes/repro_kaggle/`
- paper cases 数据：
  - `00_new_codes/repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/`
  - `00_new_codes/repro_autodl/experiments/stage2_2_subsets/experiment1_paper_cases/`
- smoke / 临时输出：
  - `00_new_codes/repro_autodl/experiments/stage2_2_smoke/`
- 报告：
  - `00_new_codes/reports/`
- agent 经验：
  - `00_new_codes/agents/readme.md`
  - `00_new_codes/agents/data_roadmap.md`
