# Kaggle 精度链路审计（Wave 11）

Artifact：[`kaggle_precision_audit.json`](artifacts/kaggle_precision_audit.json)。脚本：[`_analysis/compute_kaggle_precision.py`](_analysis/compute_kaggle_precision.py)。

---

## 1. 结论（首行）

**Results NOT comparable to official ST-Test 6144 vLLM runs.**

---

## 2. 扫描配置

| config | n_records | with_answer_tag | pipeline | comparable |
| --- | ---: | ---: | --- | --- |
| 4bit_single | 20 | 0 | hf_kaggle_not_official_vllm | false |
| 8bit_single | 20 | 0 | hf_kaggle_not_official_vllm | false |
| fp16_single | 20 | 0 | hf_kaggle_not_official_vllm | false |
| fp16_dual | 20 | 0 | hf_kaggle_not_official_vllm | false |

预测路径均在 `00_new_codes/repro_kaggle/experiments/stage1_results/experiment1_precision_resource/*/main_predictions.jsonl`。

---

## 3. 与官方分叉（见 [`20-官方与复现分叉对照.md`](20-官方与复现分叉对照.md)）

| 维度 | 官方 ST-Test | Kaggle stage1 |
| --- | --- | --- |
| 推理 | `inference/inference_tsmllm_vllm.py` + vLLM | HF `generate` |
| max_tokens | 6144（run.log） | ~2048 |
| 输出格式 | 含 `<answer>` 期望 | **0/20** 含 tag |
| 样本量 | 全量 ST-Test | smoke **20** |

---

## 4. 对优化的含义

- **不得**用 Kaggle fp16/4bit/8bit 排名驱动 ST-Test 主结论。
- 若要做精度–资源曲线，需 **先对齐 pipeline**（vLLM、6144、同一 evaluate），再谈 quant ablation。
- crosswalk 标记：`kaggle_not_comparable` → Tier **勿用**（见 [`14`](14-后续实验与优化计划.md)）。

---

## 5. Falsifier

对齐 pipeline 后，Kaggle fp16_single 在 20 题 smoke 上 with_answer_tag 应 >0，且 evaluate 路径与 `evaluation/evaluate.py` 一致；否则 audit 仍 fail。
