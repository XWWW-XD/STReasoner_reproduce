# 实验一：8bit单卡

## 固定配置

- 模型：`STReasoner_8B` / `Time-HD-Anonymous/STReasoner-8B`
- batch size：1
- max_new_tokens：512
- precision：`8bit`
- CUDA_VISIBLE_DEVICES：`0`
- requested device_map：`{'': 0}`

## 运行摘要

- load 成功：True
- load 错误：None
- 正式 generate 成功率（三组合计）：0.0
- 正式 decode 成功率（三组合计）：0.0
- 正式 parse 成功率（三组合计）：0.0
- 正式平均正确率（三组合计）：0.0
- 主测试平均正确率：0.0
- 论文样例平均正确率：0.0
- 压力测试平均正确率：0.0
- 平均 input tokens：745.16
- 平均 actual new tokens：None
- 平均延迟：None 秒
- 最高延迟：None 秒
- 平均 tokens/s：None
- 峰值显存：`{"gpu0": {"max_allocated_gib": 9.387, "max_reserved_gib": 10.293}}`
- 瓶颈统计：`{"输入/生成瓶颈": 25}`

## 三组样例说明

- 主测试样例：计入正式成功率，并单独保留分组指标。
- 论文样例：计入正式成功率，并单独保留分组指标。
- 压力测试样例：计入正式成功率，并单独保留分组指标。

机器可读结果见：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single`

## Smoke：同样例 8bit / max_new_tokens=2048

记录时间：2026-05-19 18:31:29 UTC

### Attempt 1

- 目标：复用 4bit smoke 的同一条非 forecasting 样例，比较 8bit 在同样 `max_new_tokens=2048` 下的运行状态。
- 样例：`tiny20_entity_01_line257`
- task：`entity`
- official task：`reasoning_entity`
- local index：`5`
- gold：`<answer>C</answer>`
- 输出目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048`
- CUDA_VISIBLE_DEVICES：`0`
- 可见 GPU：1 张 Tesla T4，`14.563 GiB`

加载阶段：

- load 成功：true
- load time：`67.987 sec`
- precision：`8bit`
- quantization：`BitsAndBytesConfig(load_in_8bit=True)`
- processor：`Qwen3TSProcessor`
- tokenizer：`Qwen2TokenizerFast`
- model：`Qwen3TSForCausalLM`
- requested device_map：`{"": 0}`
- actual device_map：`{"": 0}`
- model distribution：single GPU，device `0`
- CPU offload：false
- disk offload：false
- `use_cache`：false
- first parameter dtype：`torch.float16`
- time-series merge patch：applied
- load 后显存：
  - GPU0 allocated：`8.859 GiB`
  - GPU0 reserved：`9.064 GiB`

generate 阶段：

- 已进入：`STAGE=generate_one_sample`
- 未产生正常 `record.run`
- 未写出 `SMOKE_DONE`
- 未捕获 Python exception
- `partial_result.json` 停在：`stage = "model_loaded"`
- `result.json`：未生成
- PTY session exit code：`1`
- 退出后 GPU 已释放，`nvidia-smi` 显示无运行 GPU 进程。

运行中观察：

- 生成约 5 分钟后，GPU0 memory used 约 `11521 MiB`，GPU utilization 约 `91%`。
- 发现退出前后，GPU0 memory used 曾到约 `11885 MiB`，GPU utilization 约 `68%`。
- Python 子进程一度显示为 defunct，随后 GPU 资源释放。

解释：

- 8bit 第一次 smoke 没有卡在模型加载，而是在 `model.generate` 内部或期间异常退出。
- 因为没有 Python exception 写入日志，当前更像底层/native 进程终止或外部终止，而不是脚本中可捕获的普通异常。
- 与 4bit 同样例相比：4bit 可以完成但很慢；8bit 单 T4 在 2048 上限下加载成功后未完成生成。

### Attempt 2

记录时间：2026-05-19 18:33:30 UTC

- 目标：第二次复跑同一 8bit smoke 设置。
- 样例：`tiny20_entity_01_line257`
- task：`entity`
- `max_new_tokens`：`2048`
- 输出目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048_attempt2`
- CUDA_VISIBLE_DEVICES：`0`

结果：

- 未完成模型加载。
- `result.json` 已写出，错误类型为 `OutOfMemoryError`。
- 失败发生在 `AutoModelForCausalLM.from_pretrained(...)` 加载 checkpoint shards 阶段，约在 `2/4` shard 后失败。
- 首次 `AutoModelForCausalLM` 加载失败后，脚本 fallback 到 `AutoModel.from_pretrained(...)`，再次触发同样 OOM。
- 报错核心信息：尝试分配 `1.16 GiB`，GPU0 总容量 `14.56 GiB`，只剩 `981.75 MiB` free。

并发环境说明：

- 复查 `nvidia-smi` 后发现，Attempt 2 失败时 GPU0 上已有另一个 `python3` 进程占用约 `7044 MiB`，命令显示为 4bit smoke rerun。
- 因此 Attempt 2 不是干净 GPU 环境下的 8bit 复跑。
- 该 Attempt 2 的结论应标记为：受 GPU0 并发占用影响的加载阶段 OOM，不应用来直接判断 8bit 单独运行时必然无法加载。

### Attempt 3

记录时间：2026-05-19 18:47:09 UTC

- 目标：避开 GPU0 上并发的 4bit rerun，改用物理 GPU1 做 8bit 单卡复跑。
- 样例：`tiny20_entity_01_line257`
- task：`entity`
- `max_new_tokens`：`2048`
- 输出目录：`repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048_attempt3_gpu1`
- CUDA_VISIBLE_DEVICES：`1`
- 进程内可见 GPU：1 张 Tesla T4；在进程内显示为 `gpu0`。

加载阶段：

- load 成功：true
- load time：`80.110 sec`
- precision：`8bit`
- quantization：`BitsAndBytesConfig(load_in_8bit=True)`
- requested device_map：`{"": 0}`
- actual device_map：`{"": 0}`
- model distribution：single GPU，进程内 device `0`
- CPU offload：false
- disk offload：false
- `use_cache`：false
- first parameter dtype：`torch.float16`
- load 后显存：
  - 进程内 GPU0 allocated：`8.859 GiB`
  - 进程内 GPU0 reserved：`9.064 GiB`

generate 阶段：

- 已进入：`STAGE=generate_one_sample`
- 运行约 5 分钟后观察：
  - 物理 GPU1 memory used：约 `11521 MiB`
  - 物理 GPU1 utilization：约 `98%`
- 由于用户决定今日结束，本 attempt 被手动中止。
- 中止方式：发送 KeyboardInterrupt。
- 中止位置：`model.generate(...)` 内部，具体栈在 bitsandbytes 8bit matmul / `int8_vectorwise_quant` 附近。
- 未生成最终回答。
- 未运行 strict diagnostic。
- 未运行 official eval。

结论：

- Attempt 3 证明：在干净的第二张 T4 上，8bit 可以加载并进入生成阶段。
- 截止手动中止时，8bit generate 仍在持续计算，未自然完成。
- 因为是用户中止，Attempt 3 不应记为模型自然失败；它记录的是 8bit 生成阶段长耗时/高显存占用的中间状态。

### 当日收尾状态

- 2026-05-19 18:47:09 UTC 已停止当前所有正在跑的 smoke/rerun 进程。
- 停止后 `nvidia-smi` 显示 GPU0/GPU1 均为 `0 MiB`、`0%` utilization。
