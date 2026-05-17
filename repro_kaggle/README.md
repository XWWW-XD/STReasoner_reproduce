# repro_kaggle README

## 1. 这个目录是干什么的

`repro_kaggle/` 是我们自己新增的 Kaggle T4 x2 低资源复现辅助目录，用来放环境恢复、数据检查、模型加载 smoke test 和小规模评测脚本。

## 2. 脚本运行顺序总览

| 顺序 | 脚本 | 作用 | 什么时候运行 | 成功标志 |
|---|---|---|---|---|
| 00 | `repro_kaggle/scripts/00_check_kaggle_env.py` | 检查 Kaggle、GPU、CUDA、Python 包和 HF cache | 每天启动后、安装依赖后 | 能看到 T4 x2、关键包版本和 cache 路径 |
| 01 | `repro_kaggle/scripts/01_setup_kaggle_t4.sh` | 配置 cache、安装不含 flash_attn 的依赖、升级 bitsandbytes | 新 Kaggle session 或环境丢失后 | 最后自动跑通 `00_check_kaggle_env.py` |
| 02 | `repro_kaggle/scripts/02_inspect_stbench.py` | 检查 ST-Bench 各 subset 是否能直接 `load_dataset` | 环境可用后、正式跑模型前 | 日志里出现 `LOAD_OK` / `LOAD_FAILED` 总结 |
| 03 | `repro_kaggle/scripts/03_load_streasoner_smoke.py` | 4bit 加载 STReasoner-8B，并尝试极小生成 | 数据确认后、跑样本前 | `MODEL_LOAD_PASS` |
| 04 | `repro_kaggle/scripts/04_run_one_sttest_sample.py` | 读取 1 条 ST-Test，走完整模型推理链路 | 03 通过后 | `ONE_SAMPLE_RUN_PASS`，并写出预测 JSON |
| 05 | `repro_kaggle/scripts/05_eval_sttest_tiny.py` | 跑 5-20 条 ST-Test 小评测 | 04 通过后 | 写出 summary；无失败时出现 `TINY_EVAL_PASS` |

## 3. 每个脚本详细说明

### 00_check_kaggle_env.py

- 作用：确认当前是不是 Kaggle T4 x2 环境，并打印 Python、PyTorch、CUDA、GPU、关键依赖和 Hugging Face cache 信息。
- 运行命令：`python repro_kaggle/scripts/00_check_kaggle_env.py`
- 看哪些输出：`cuda available`、`gpu count`、GPU 名称、`transformers` / `accelerate` / `bitsandbytes` / `datasets` 版本、`HF_HOME`。
- 失败时说明什么：通常说明还没打开 GPU、依赖没装好，或 cache 环境变量没有按 Kaggle 工作目录设置。

### 01_setup_kaggle_t4.sh

- 作用：恢复 Kaggle session 的基础 Python 环境。
- 运行命令：`bash repro_kaggle/scripts/01_setup_kaggle_t4.sh`
- 它内部做了什么：进入项目目录，设置 Hugging Face cache，基于 `requirements.txt` 生成不含 flash 的临时依赖文件，安装依赖，升级 `bitsandbytes`，最后运行 `00_check_kaggle_env.py`。
- 为什么不安装 flash_attn：Kaggle T4 上编译和版本匹配成本高，容易卡住；当前目标是低资源复现链路，优先使用 `sdpa` 或 `eager` 注意力后端。
- 成功后下一步：运行 `python repro_kaggle/scripts/02_inspect_stbench.py`。

### 02_inspect_stbench.py

- 作用：检查 `Time-HD-Anonymous/ST-Bench` 的各个 subset 是否能直接读取，并预览第一条样本字段。
- 运行命令：`python repro_kaggle/scripts/02_inspect_stbench.py`
- 它检查哪些 subset：`ST-Test`、`ST-Test-Text`、`ST-SFT`、`ST-CoT`、`ST-Align`。
- 输出文件：`repro_kaggle/outputs/02_stbench_inspect.log`
- 如何解读结果：`LOAD_OK` 表示该 subset 可直接读取；`LOAD_FAILED` 多半是字段 schema 不一致，不等于数据不存在。

### 03_load_streasoner_smoke.py

- 作用：用 4bit 方式加载 `Time-HD-Anonymous/STReasoner-8B`，确认 checkpoint、remote code、量化和设备分配能跑起来。
- 运行命令：`python repro_kaggle/scripts/03_load_streasoner_smoke.py --model_name Time-HD-Anonymous/STReasoner-8B --load_in_4bit true --attn_backend sdpa`
- `MODEL_LOAD_PASS` 代表什么：模型已经成功加载到当前机器，这是最关键的里程碑。
- `GENERATE_FAIL_BUT_MODEL_LOAD_PASS` 代表什么：模型加载成功，但极小生成测试失败；这通常说明下一步要处理 processor、输入格式或设备细节，而不是 checkpoint 完全不可用。
- 如果 OOM / device mismatch / bitsandbytes 报错怎么办：先确认运行过 `01_setup_kaggle_t4.sh`；OOM 可重启 session 后只跑 03；device mismatch 可改试 `--attn_backend eager`；bitsandbytes 报错优先重新跑 01。

### 04_run_one_sttest_sample.py

- 作用：从 `ST-Test` 取 1 条样本，构造输入并调用 STReasoner 生成预测。
- 运行命令：`python repro_kaggle/scripts/04_run_one_sttest_sample.py --model_name Time-HD-Anonymous/STReasoner-8B --load_in_4bit true --attn_backend sdpa --index 0`
- 输出文件：`repro_kaggle/outputs/one_sttest_prediction.json` 和 `repro_kaggle/outputs/04_run_one_sttest_sample.log`
- 成功标志：日志里出现 `ONE_SAMPLE_RUN_PASS`，JSON 里有 `prediction`。

### 05_eval_sttest_tiny.py

- 作用：在 `ST-Test` 上跑 5-20 条小规模评测，快速观察输出格式、解析成功率和粗略正确率。
- 运行命令：`python repro_kaggle/scripts/05_eval_sttest_tiny.py --model_name Time-HD-Anonymous/STReasoner-8B --max_samples 5`
- 输出文件：`repro_kaggle/outputs/sttest_tiny_predictions.jsonl`、`repro_kaggle/outputs/sttest_tiny_summary.json`、`repro_kaggle/outputs/05_sttest_tiny_eval.log`
- 成功标志：summary 写出 `num_total`、`num_failed`、`accuracy` 等字段；如果没有失败，日志里出现 `TINY_EVAL_PASS`。
