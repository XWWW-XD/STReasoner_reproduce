# Agent Notes

## 写报告经验

- 报告先写结论，再写证据和过程，方便快速判断问题是否已解决。
- 遇到多轮排查时，最终报告要合并成一份完整版本，避免读者在多份文件之间来回拼接。
- 技术细节从浅到深展开：先解释现象和根因，再补充命令、路径、日志和备份位置。

## 实验配置选择经验

- 先按仓库 requirements 选解释器；不能使用默认 /root/miniconda3/bin/python，本项目优先用 `/root/autodl-tmp/conda/envs/str-py310/bin/python`。
- 跑正式实验前先做最小链路检查：编译、数据、parser import、1 条样例 smoke。
- 模型后端优先保持脚本默认配置；不要为了跑通擅自切换 attention backend。
- 输出放独立 smoke 目录，避免污染正式实验结果。
- 卡在模型、缓存、依赖、显存时先停下记录，不直接大改代码。
- 下载模型前确认权重是否完整，不把只有 config/tokenizer 的目录当完整模型。
- 写流程和指标时优先看仓库根目录原始代码（00_new_codes是复现新增代码，其他是源代码）；不要把辅助脚本里的实现细节编成论文或官方实验阶段。

## 模型缓存与加载经验

- STReasoner-8B 完整权重在 `/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B/`。
- HuggingFace cache 使用 `/root/autodl-tmp/cache/huggingface`；本地模型已用 symlink 接入该 cache。
- 同一 AutoDL 数据盘正常关机/开机后通常不用重下模型，直接加载即可；释放实例、换机器、清盘才需要重下。
- 跑 smoke 或正式实验可加 `HF_HUB_OFFLINE=1`，用于确认脚本命中本地模型，不重新联网拉权重。
- Stage 2.2 当前脚本硬编码模型名 `Time-HD-Anonymous/STReasoner-8B`，不要误改成未确认的本地别名。
<<<<<<< HEAD
=======

## Stage 2.2 paper_cases 经验

- `run-all` 不应每条样例重新加载一次 8B 模型；优先一次加载、循环复用。
- 先区分“生成失败”和“评测提取失败”；paper_cases 里模型可能答对但没有 `<answer>` 标签。
- 选择题 parser 不要只看文本开头，要优先读末尾 `Answer: X`、`\boxed{X}` 等最终答案。
- forecasting parser 不要抽全文所有数字，优先读最终 JSON/list 预测数组。
- 修 parser 时只能写通用规则，不按 sample_id 或 gold 写特例。
- raw response 格式和 parser 结果要分开记：`format_success` 看原文是否严格有 `<answer>`，`formatted_answer` 是通用后处理结果。
- paper_cases / SmartTest 小样例不能当作论文整体效果；验证论文真实效果要另跑完整 ST-Test 四类任务。

## ST-Test 经验

- ST-Test 四类数据在 `data/ST-Bench/ST-Test/`；完整验证不要用 SmartTest 或 paper_cases 替代。
- 优先使用仓库原始 `inference/inference_tsmllm_vllm.py` 和 `evaluation/evaluate.py`。
- 从文件路径直接跑 `evaluation/evaluate.py` 时加 `PYTHONPATH=.`，否则可能 import 不到 `evaluation` 包。
- 完整跑 ST-Test 时不要每条样例重新加载 vLLM/8B 模型，按任务长进程连续生成。
- ST-Test 正式实验必须用 `max_tokens=6144`；低于 6144 的结果只能记为预跑/链路检查，不能当正式结果。
- `inference/llm_utils.py` 里 worker 要确认真的使用调用方传入的 `SamplingParams`，否则 CLI 的 `--max_tokens` 可能只是摆设。
- 本轮完整 ST-Test `max_tokens=6144` 输出在 `exp/sttest_full_*_6144/`；逐条 raw/gold 对齐在 `00_new_codes/reports/artifacts/sttest_full_6144_outputs_with_gold.jsonl`。

## 术语经验

- `Qwen3TSForCausalLM` 是 STReasoner-8B 的加载架构类名，不代表换成了别的实验模型。
>>>>>>> origin/autodl
- tokenizer 的 `.decode()` 只是 token id 转字符串的 API，不要把它写成独立实验阶段或正式指标。

## 必须遵守的规则

- 报告语言为中文
- 不要改我的提示词文件，要写内容自己在后面加文件

## 每次任务结束后要添加经验、写日志

- 经验继续添加到/root/autodl-tmp/STReasoner_reproduce/00_new_codes/agents/readme.md。如无必要不删除其他经验，如果冲突则报告等我同意后删除原经验。

- codex需在reports下建XX-YY-MM-DD-id，其中XX是你顺着序号往后数一个，YY-MM-DD是今天日期，id从1开始，表示是今天的第几个日志

<<<<<<< HEAD
- 内容需要你从你在对话框中的输出填写，填写HH-MM：发生了什么事情。


## 原则

- 尽量精简
- 比如不要一次实验的结果放prediction.json，non_prediction.json等多个json

## 文件名

- stage2包括2.1, 2.2等实验
=======
- 内容需要你从你在对话框中的输出填写，填写"HH:MM：发生了什么事情"。

- 请写日志的时候，将这次任务执行中对话框的所有阶段性进展全部写入日志文件中，不要再去修改措辞。

>>>>>>> origin/autodl
