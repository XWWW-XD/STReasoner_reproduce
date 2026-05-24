# 实验一 2026-05-19 会话记录

记录时间：2026-05-19 18:42:51 UTC

## 当前决定

- 今天暂停继续推进新的单样例四配置实验。
- 不再启动新的实验进程。
- 不再停止现有实验进程，等待当前仍在运行的任务自行完成。
- 本记录只记录现场状态，不更新正式 `experiment_summary.md`。

## 当前仍在运行的任务

- 进程：PID `24296`，父进程 PID `24293`
- 配置：`8bit_single`
- 样例：`tiny20_entity_01_line257`
- 任务类型：`entity`
- `max_new_tokens`: `2048`
- 物理 GPU：GPU1
- 启动环境：`CUDA_VISIBLE_DEVICES=1`
- 结果目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048_attempt3_gpu1/`
- 当前阶段：`STAGE=generate_one_sample`
- 已有文件：
  - `partial_result.json`
  - `run.log`
- 18:42:51 UTC 的 GPU 状态：
  - GPU0: `0 MiB / 15360 MiB`, util `0%`
  - GPU1: `11521 MiB / 15360 MiB`, util `91%`

## 已中断或未完成的相关任务

### 4bit rerun 20260519_182957

- 目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182957/`
- 样例：`tiny20_entity_01_line257`
- 配置：`4bit_single`
- `max_new_tokens`: `2048`
- 运行到 `STAGE=generate_one_sample`
- `run.log` 记录结果：`KeyboardInterrupt`
- 当前只有：
  - `partial_result.json`
  - `run.log`
- 未看到 `result.json`。

### 8bit smoke_nonforecasting_8bit_2048

- 目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048/`
- 样例：`tiny20_entity_01_line257`
- 配置：`8bit_single`
- `max_new_tokens`: `2048`
- 运行记录停在 `STAGE=generate_one_sample`
- 当前只有：
  - `partial_result.json`
  - `run.log`
- 未看到 `result.json`。

## 代码状态说明

- 今天曾短暂开始给 `run_experiment1_new_version.py` 加单样例分支，但已经回滚。
- 截至本记录写入前，`run_experiment1_new_version.py` 没有保留这次新增的 diff。
- 未启动新的非 entity 单样例四配置实验。

## 下次继续前建议检查

- 先检查 `8bit_single/smoke_nonforecasting_8bit_2048_attempt3_gpu1/result.json` 是否已生成。
- 如果 attempt3 仍在跑，先不要并行启动新实验，避免 GPU 资源相互影响。
- 如果继续用户最初的新要求，再选择一个非 `tiny20_entity_01_line257` 同类样例，建议优先选非 entity 的选择题样例，例如 `tiny20_correlation_03_line528`，再按 prompt2 的四种模型配置执行并记录真实输出。
