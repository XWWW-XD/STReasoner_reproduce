# 选择说明

本次扫描了四个任务文件中的全部 ST-Test 原始样本。由于本地没有完整 Qwen/STReasoner tokenizer 词表，未计算正式 token 数；每个候选样本按以下近似指标排序：

`len(input) + len(json.dumps(timeseries, compact separators))`

这个指标比只看 `input` 更保守，因为 ST-Test 把时间序列数值保存在单独的 `timeseries` 字段中，而 prompt 里只有 `<ts><ts/>` 占位符。

最终选中的样本：

```json
{
  "sample_id": "stress_longest_etiological_line110",
  "task": "etiological",
  "source_file": "ST-Test/etiological_test.jsonl",
  "original_line_index": 110,
  "category": "etiological",
  "output": "<answer>A</answer>",
  "appears_in_tiny20": false,
  "length_metrics": {
    "input_char_length": 1356,
    "timeseries_serialized_char_length": 22363,
    "output_char_length": 20,
    "combined_input_timeseries_char_length": 23719
  }
}
```
