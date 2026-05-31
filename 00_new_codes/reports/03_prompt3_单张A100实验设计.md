请基于当前仓库 STReasoner_reproduce，编写一个 stage2 的最小可运行脚本。

本轮目标非常小：
只在 AutoDL A100 环境下，用 fp16 格式，逐条运行 SmartTest 中的两条样例。
先跑非预测样例，将结果和指标写入 jsonl / summary.json / log；再跑预测样例，也写入对应结果文件。文档整理下一轮再做，本轮不要生成 md 报告。

重要限制：
1. 不要跑 tiny20 全量。
2. 不要跑 paper cases。
3. 不要跑 stress。
4. 不要 run-all。
5. 不要伪造任何结果。
6. 本轮只写脚本、检查/生成 SmartTest、给出 AutoDL 上的运行命令。

一、实验阶段与目录
本实验归档为 stage2，不再把本轮新增结果写入 stage1 目录。

请使用或创建以下目录：

样本目录：
SmartTest 来源文件：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl

本轮 SmartTest 输出目录：
repro_autodl/experiments/stage2_subsets/experiment1_smart_test/

本轮 SmartTest 输出文件：
repro_autodl/experiments/stage2_subsets/experiment1_smart_test/SmartTest.jsonl

结果目录：
repro_autodl/experiments/stage2_results/experiment1_smarttest/

脚本目录：
repro_autodl/experiments/scripts/stage2_script/

新脚本文件名：
repro_autodl/experiments/scripts/stage2_script/run_smarttest.py

二、当前 AutoDL 环境

当前运行环境是 AutoDL A100 服务器：

项目根目录：
/cloud/cloud-ssd1/workspace/STReasoner_reproduce

数据盘：
/cloud/cloud-ssd1

Hugging Face 缓存目录：
/cloud/cloud-ssd1/hf_cache

GPU：
1 × NVIDIA A100-SXM4-80GB

系统：
Ubuntu 22.04

PyTorch：
2.6.0

CUDA：
12.6

脚本中不要硬编码 /kaggle/working。
脚本中不要默认把模型缓存、数据缓存、实验输出写入 /root。

运行前优先读取这些环境变量：
HF_HOME
TRANSFORMERS_CACHE
HF_HUB_CACHE

如果这些变量不存在，则脚本中默认设置为：
/cloud/cloud-ssd1/hf_cache

三、参考脚本

参考已有脚本：

repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py

请优先复用它已有的：
- 样本读取逻辑
- prompt / input 构造逻辑
- 模型加载逻辑
- generate / decode 逻辑
- parser 逻辑
- 字段记录逻辑

不要重新设计 parser；如果需要解析输出，请直接复用参考脚本中的 parser 函数，或从参考脚本 import / copy 最小必要函数，并在注释中标明来源。
不要重写 official evaluation。

四、SmartTest 样本

SmartTest 的两条样例应从上述 tiny20_all.jsonl 中复用已有 SmartTest.jsonl。

本轮 stage2 的 SmartTest 输出到：
repro_autodl/experiments/stage2_subsets/experiment1_smart_test/SmartTest.jsonl

其中包含两条样例：
1. 一条 non_forecasting 样例；
2. 一条 forecasting 样例。

五、模型运行配置
本轮只支持一种配置：fp16_a100_single

配置要求：
- 只使用一张 GPU：cuda:0
- CUDA_VISIBLE_DEVICES=0
- torch_dtype=torch.float16
- 不使用 quantization_config
- 不使用 CPU offload
- 不使用 disk offload
- batch size = 1
- max_new_tokens 默认 2048，但允许命令行传参修改
- attn_implementation 优先使用 sdpa；如果原脚本已有稳定默认值，可保持原逻辑

注意：
本轮只验证 fp16 单卡推理链路。
不要实现 4bit / 8bit / 双卡逻辑。

六、运行顺序

脚本必须支持逐条运行。

第一步，只跑非预测样例：
python repro_autodl/experiments/scripts/stage2_script/run_smarttest.py run \
  --case non_forecasting \
  --max-new-tokens 2048

第二步，只跑预测样例：
python repro_autodl/experiments/scripts/stage2_script/run_smarttest.py run \
  --case forecasting \
  --max-new-tokens 2048

可选支持：
--overwrite true/false
--resume true/false
--output-root

八、输出文件

结果目录：
repro_autodl/experiments/stage2_results/experiment1_smarttest/

非预测样例输出：
non_forecasting_prediction.jsonl
non_forecasting_summary.json
non_forecasting_run.log

预测样例输出：
forecasting_prediction.jsonl
forecasting_summary.json
forecasting_run.log

九、每条 prediction 必须记录字段

每条样例至少保存：

基础信息：
- case：non_forecasting / forecasting
- config：fp16_a100_single
- sample_id 或 original_index
- task/category
- source_file

输入与生成：
- input_preview
- input_tokens
- max_new_tokens
- actual_new_tokens
- raw_response
- decoded_text
- generate_success
- decode_success
- generate_error

资源与速度：
- gpu_name
- gpu_total_memory
- gpu_peak_memory
- latency_sec
- tokens_per_sec

解析与判断：
- parsed_answer
- gold_answer
- parse_success
- parse_error
- failure_type

failure_type 可以先粗略记录：
- no_failure
- load_failed
- generate_failed
- decode_failed
- parse_failed
- unknown
更细的瓶颈类型后续由人工根据 log 和输出再分析。

十、运行前环境检查

每次 prepare 或 run 开始时，把以下信息打印到终端，并写入对应 log：

- pwd
- git branch
- git status --short
- python --version
- torch.__version__
- torch.cuda.is_available()
- torch.version.cuda
- torch.cuda.get_device_name(0)
- nvidia-smi 简要信息
- df -h /cloud/cloud-ssd1
- HF_HOME
- TRANSFORMERS_CACHE
- HF_HUB_CACHE


十一、完成后请告诉我

请在最终回复中告诉我：

1. 新增/修改了哪些文件；
2. SmartTest.jsonl 是已经生成，还是需要在 AutoDL 上运行 prepare 生成；
3. 我在 AutoDL 上应该依次运行哪三条命令；
4. 跑完非预测样例后，我应该检查哪些输出文件；
5. 跑完预测样例后，我应该检查哪些输出文件；
6. 如果失败，我应该把哪些 log 文件发给你继续排查。