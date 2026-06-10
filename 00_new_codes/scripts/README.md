# `00_new_codes/scripts/`

只读分析脚本：默认读仓库根下 `exp/` + `data/ST-Bench/`，产物写入 `reports/artifacts/<任务名>/`。

## `mlp_encoder_focused_analysis/`

STReasoner-8B 全量 ST-Test 6144 输出的 MLP/复述边界分析（报告 37–39、43–44）。

| 脚本 | 作用 |
| --- | --- |
| `find_bad_cases.py` | 从 `exp/sttest_full_*_6144/` 找 strict node-window 复述 mismatch；支持 `--exp-suffix`、`--exp-layout bundled` |
| `summarize_task_level_stats.py` | 汇总四任务 mismatch 统计（依赖 `find_bad_cases` 产物 + `evaluate_qa`） |
| `count_ts_mentions.py` | thinking 中宽口径 time-series 提及次数（报告 39 §10） |
| `inspect_mlp_encoder.py` | 只读 TS encoder 配置/维度，写出 `mlp_encoder_facts.json` |
| `summarize_bad_cases.py` | 将 boundary JSON 整理为 Markdown 表格 |

典型顺序：`find_bad_cases.py` → `summarize_task_level_stats.py`；提及统计：`count_ts_mentions.py`。

GPU 推理 / 训练 / graph 消融 runner 在 [`../repro_autodl/experiments/scripts/`](../repro_autodl/experiments/scripts/README.md)。
