# 38-encoder代码细节：执行日志

15:35：用户要求本轮“日志写在38，报告写在39”；确认 `00_new_codes/reports/38-encoder代码细节.md` 在文件系统中不存在，按用户要求新建本日志文件。

15:35：开始修订 MLP 时间序列复述错误分析，不继续把 37 作为最终报告，而是新写 `39-MLP复述四任务分析.md`。

15:35：核查现有 artifact，发现原先 `478` 是 forecasting 中抽到的复述窗口条数，不是唯一样本数；按 sample idx 去重后约为 142 个样本。

15:35：继续核查，发现现有抽取逻辑没有过滤“复述其实对得上”的窗口；开始修正 `find_bad_cases.py` 的 mismatch 定义。

15:35：修正 node/window 对齐逻辑：同一条模型复述同时尝试 0-based 和 1-based raw time series 对齐，保留误差更小的一种；只有 `max_abs_diff_between_stated_and_raw > 0.01` 才算对不上。

15:35：重新生成复述错误 artifact；保守口径后 forecasting 为 474 个对不上窗口、141/280 个对不上样本，correlation 为 1179 个对不上窗口、347/1592 个对不上样本。

15:35：新增 `summarize_task_level_stats.py`，只读已有 STReasoner-8B full ST-Test 6144 输出、原始 ST-Test JSONL、官方 evaluation metrics 和复述错误 artifact，不重跑模型。

15:35：生成四任务 artifact：`task_level_reconstruction_mismatch.*`、`task_level_metric_comparison.*`、`reconstruction_error_severity_by_task.*` 和 `manual_audit_top10_reconstruction_mismatch.json`。

15:35：统计四任务效果差异：forecasting 用当前 parser 重算 MAE/MAPE，correlation/entity/etiological 用 accuracy；entity 和 etiological 当前严格 node-window 数值列表口径下未发现可统计的对不上样本。

15:35：发现 forecasting 现有 `evaluation_metrics.json` 与当前 parser 重算结果不完全一致：已有文件为 evaluated=280、MAE=68.3171、MAPE=123.2891；当前 parser 重算为 evaluated=278、MAE=67.8199、MAPE=123.5208，缺失 idx 为 `[23, 206]`。

15:35：在 39 报告中明确：forecasting 分组 MAE/MAPE 只用于当前 parser 口径内的组间比较，不混用旧 metrics 文件。

15:35：写入 39 报告，包含四任务 `a/b` 统计、idx 全列表、分组效果对比、复述错误严重度、代表核查样本、原因讨论和下一步复述 probe 设计。

