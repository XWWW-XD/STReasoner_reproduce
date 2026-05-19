# 匹配记录

## appendix_h_table6_etiological

- 论文位置：Appendix H，Table 6（PDF 抽取文本）；用户提示中提到 Table 9-12，但这份 PDF 中四个附录 case 编号为 Table 6-9。
- 匹配来源：`ST-Test/etiological_test.jsonl` 第 `118` 行
- 数据集输出：`<answer>D</answer>`
- 备注：按任务、graph/question/options 片段和论文答案严格匹配。
- 匹配依据：
  - `Which etiological scenario can be inferred from the spatio-temporal data?`
  - `Node 0->Node 2; Node 1->Node 2; Node 2->Node 3; Node 3->Node 4`
  - `Urban pollution network with industrial and traffic sources dispersing through nodes`

## appendix_h_table7_entity

- 论文位置：Appendix H, Table 7
- 匹配来源：`ST-Test/entity_test.jsonl` 第 `982` 行
- 数据集输出：`<answer>C</answer>`
- 备注：按任务、graph/question/options 片段和论文答案严格匹配。
- 匹配依据：
  - `Which (name, description) pair should Node 9 correspond to?`
  - `Node 0->Node 1; Node 1->Node 3; Node 2->Node 3; Node 3->Node 4; Node 4->Node 5; Node 5->Node 6; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9`
  - `Final Point, Final monitoring point`

## appendix_h_table8_correlation

- 论文位置：Appendix H, Table 8
- 匹配来源：`ST-Test/correlation_test.jsonl` 第 `547` 行
- 数据集输出：`<answer>D</answer>`
- 备注：按任务、graph/question/options 片段和论文答案严格匹配。
- 匹配依据：
  - `Which statement best describes the relationship between Node 2 and Node 4 during time steps 177-180`
  - `Node 0->Node 2; Node 0->Node 6; Node 1->Node 2; Node 2->Node 3; Node 2->Node 5; Node 3->Node 4; Node 4->Node 8; Node 5->Node 7; Node 6->Node 7; Node 7->Node 8; Node 8->Node 9`
  - `Thermal signal propagates from Node 2 to Node 4 via Node 3`

## appendix_h_table9_forecasting

- 论文位置：Appendix H, Table 9
- 匹配来源：`ST-Test/forecasting_test.jsonl` 第 `9` 行
- 数据集输出：`[19.86, 19.97, 20.05]`
- 备注：原始 ST-Test input 与论文 prompt 匹配，但数据集 gold output 是 [19.86, 19.97, 20.05]，论文展示的 STReasoner prediction 是 [19.88, 19.89, 19.90]。该样例保留为可复跑原始输入，并在文档中记录答案差异。
- 匹配依据：
  - `Given the context Maximum solar heating period enhances thermal transfer, predict the value of node 2 for the next 3 steps`
  - `Historical observation window: 30-35`
  - `Node 0->Node 2; Node 1->Node 2; Node 2->Node 1`
