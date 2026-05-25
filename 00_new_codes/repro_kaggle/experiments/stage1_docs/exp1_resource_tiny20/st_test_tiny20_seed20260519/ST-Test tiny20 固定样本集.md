
本文档说明 `exp1_resource_tiny20` 的 `tiny20` 主测试样本集。

- 数据集：`Time-HD-Anonymous/ST-Bench`
- 数据集 revision：`1a6871632f295dc2a049860b2a7d08ae445c25da`
- 随机种子：`20260519`
- 抽样规则：对每个 ST-Test 任务文件，使用 Python random.Random(20260519).sample(range(num_rows), 5) 抽取 5 个原始零基行号；随后按行号升序写出，保证文件顺序稳定。
- 数据目录：`repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519`
- 主文件：`tiny20_all.jsonl`

该样本集共 20 条 ST-Test 样本：forecasting、entity、etiological、correlation 四类各 5 条。
每个 JSONL 行保留原始 `input`、`timeseries`、`output`、`category` 字段，并额外加入可追溯元数据。

长度统计使用近似字符长度。ST-Test 的 `input` 中时间序列位置是 `<ts><ts/>` 占位符，因此 manifest 还额外记录序列化后的 `timeseries` 长度，以及 `input + timeseries` 的合并近似长度。
