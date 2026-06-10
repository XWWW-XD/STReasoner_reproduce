# `00_new_codes/tools/`

通用小工具：不绑定某一实验目录结构，改路径后可在多处复用。

| 文件 | 作用 |
| --- | --- |
| `json_to_md_table.py` | 将任意 prediction JSONL 转为 Markdown 表格 / Excel，便于人工抽查字段 |
| `columns.py` | 用硬编码分段计数画 ST-Test response token 分布图 → `outputs/st_test_token_distribution_bw.png` |
| `outputs/` | `columns.py` 等脚本的图表输出目录 |

实验跑数、Stage 分析见 [`../scripts/README.md`](README.md) 与 [`../repro_autodl/experiments/scripts/README.md`](../repro_autodl/experiments/scripts/README.md)。
