# 阶段三探索 memo（Wave 15–22）

**Strict redo**: 2026-05-30 — [`_plans/README.md`](_plans/README.md) wave15–22 + 深读报告 16–19 已人工扩写 path:line。

**主轴**：源码深读 + crosswalk；**只读**官方根目录代码。

---

## Wave 映射

| Wave | 报告 | Artifact | Plan |
| --- | --- | --- | --- |
| 15 | 本 memo | `code_registry.json` | `_plans/wave15.md` |
| 16 | [`16-inference通路深读.md`](16-inference通路深读.md) | `inference_trace.json` | `_plans/wave16.md` |
| 17 | [`17-evaluation与parser深读.md`](17-evaluation与parser深读.md) | `evaluation_trace.json` | `_plans/wave17.md` |
| 18 | [`18-训练与SFT通路深读.md`](18-训练与SFT通路深读.md) | `training_trace.json` | `_plans/wave18.md` |
| 19 | [`19-encoder与graph注入深读.md`](19-encoder与graph注入深读.md) | `encoder_trace.json` | `_plans/wave19.md` |
| 20 | [`20-官方与复现分叉对照.md`](20-官方与复现分叉对照.md) | `pipeline_diff_ledger.json` | `_plans/wave20.md` |
| 21 | [`21-代码与实验异常对照.md`](21-代码与实验异常对照.md) | `code_experiment_crosswalk.json` | `_plans/wave21.md` |
| 22 | [`15-阶段三总览.md`](15-阶段三总览.md) | — | `_plans/wave22.md` |

脚本：`compute_code_phase3.py`（line scan）+ 报告 **人工互证** 源码。

---

## 阶段三 headline（与 artifact + 源码一致）

1. **Inference**：`inference_tsmllm_vllm.py:331` `num_tokens`=input；worker L186–189 可接收 CLI `SamplingParams`。  
2. **Evaluate**：`evaluate_qa.py:39–48` 无 tag → 整段 strip。  
3. **Train**：`supervised.py:87–96` thinking+answer 全文 CE。  
4. **Encoder**：`chatts_vllm.py:52–82` MLP patch — 非 verbatim raw TS 文本。  
5. **Pipeline**：Kaggle/autodl ≠ official vLLM 6144（wave12/20）。

---

## 暂停条件（Wave22）

- P0 模块已在 `code_registry.json`  
- `21` crosswalk 5 行填满现象→代码→Tier  
- 无新未锚定假说 → **阶段三 superpowers 终止**

Roadmap GPU 项 → [`14`](14-后续实验与优化计划.md)。
