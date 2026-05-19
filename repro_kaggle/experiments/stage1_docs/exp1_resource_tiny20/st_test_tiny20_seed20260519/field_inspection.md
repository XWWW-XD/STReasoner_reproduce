# 字段检查

数据集仓库：`Time-HD-Anonymous/ST-Bench`
数据集 revision：`1a6871632f295dc2a049860b2a7d08ae445c25da`

ST-Test 直接从原始 JSONL 文件逐行读取，而不是使用 `datasets` 合并后的 split；这样可以保留 source file 和原始行号，方便后续追溯。

## forecasting

- 源文件：`ST-Test/forecasting_test.jsonl`
- 行数：280
- 字段摘要：
```json
{
  "columns": [
    "category",
    "input",
    "output",
    "timeseries"
  ],
  "category_counts": {
    "forecasting": 280
  },
  "timeseries_node_count_distribution": {
    "3": 115,
    "5": 92,
    "10": 73
  },
  "timeseries_length_distribution": {
    "32": 52,
    "33": 34,
    "34": 68,
    "35": 43,
    "36": 152,
    "37": 87,
    "38": 52,
    "39": 3,
    "40": 57,
    "41": 8,
    "42": 29,
    "43": 26,
    "44": 8,
    "45": 8,
    "46": 10,
    "47": 3,
    "48": 23,
    "49": 10,
    "50": 8,
    "51": 10,
    "52": 18,
    "53": 20,
    "55": 10,
    "56": 20,
    "58": 5,
    "59": 10,
    "60": 60,
    "61": 13,
    "62": 6,
    "64": 9,
    "65": 14,
    "66": 5,
    "67": 15,
    "68": 23,
    "69": 3,
    "70": 13,
    "71": 40,
    "72": 51,
    "73": 15,
    "74": 26,
    "75": 5,
    "76": 10,
    "77": 15,
    "78": 5,
    "80": 23,
    "82": 25,
    "83": 39,
    "84": 33,
    "85": 13,
    "86": 3,
    "87": 3,
    "88": 3,
    "89": 5,
    "90": 20,
    "95": 3,
    "97": 5,
    "98": 3,
    "99": 5,
    "100": 16,
    "101": 15,
    "103": 10,
    "104": 5,
    "108": 3,
    "118": 3,
    "119": 3,
    "120": 11,
    "121": 8,
    "122": 23,
    "140": 15,
    "148": 3,
    "150": 3,
    "151": 5,
    "152": 5,
    "161": 10,
    "165": 10,
    "179": 5,
    "180": 13,
    "181": 23,
    "182": 10,
    "184": 10,
    "185": 3,
    "200": 10,
    "201": 10,
    "202": 10,
    "240": 3,
    "261": 5
  }
}
```

## entity

- 源文件：`ST-Test/entity_test.jsonl`
- 行数：1194
- 字段摘要：
```json
{
  "columns": [
    "category",
    "input",
    "output",
    "timeseries"
  ],
  "category_counts": {
    "entity": 1194
  },
  "timeseries_node_count_distribution": {
    "3": 249,
    "5": 295,
    "10": 650
  },
  "timeseries_length_distribution": {
    "48": 2104,
    "56": 109,
    "60": 234,
    "72": 325,
    "84": 824,
    "90": 243,
    "96": 1560,
    "120": 225,
    "168": 1803,
    "192": 9,
    "240": 325,
    "288": 100,
    "360": 861
  }
}
```

## etiological

- 源文件：`ST-Test/etiological_test.jsonl`
- 行数：207
- 字段摘要：
```json
{
  "columns": [
    "category",
    "input",
    "output",
    "timeseries"
  ],
  "category_counts": {
    "etiological": 207
  },
  "timeseries_node_count_distribution": {
    "3": 83,
    "5": 59,
    "10": 65
  },
  "timeseries_length_distribution": {
    "48": 318,
    "56": 13,
    "60": 28,
    "72": 35,
    "84": 128,
    "90": 31,
    "96": 210,
    "120": 25,
    "168": 251,
    "192": 3,
    "240": 35,
    "288": 10,
    "360": 107
  }
}
```

## correlation

- 源文件：`ST-Test/correlation_test.jsonl`
- 行数：1592
- 字段摘要：
```json
{
  "columns": [
    "category",
    "input",
    "output",
    "timeseries"
  ],
  "category_counts": {
    "correlation": 1592
  },
  "timeseries_node_count_distribution": {
    "3": 248,
    "5": 413,
    "10": 931
  },
  "timeseries_length_distribution": {
    "48": 2831,
    "56": 159,
    "60": 344,
    "72": 495,
    "84": 1124,
    "90": 323,
    "96": 2180,
    "120": 315,
    "168": 2503,
    "192": 9,
    "240": 465,
    "288": 100,
    "360": 1271
  }
}
```
