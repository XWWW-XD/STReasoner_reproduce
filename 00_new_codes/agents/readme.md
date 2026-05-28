# Agent Notes

## 写报告经验

- 报告先写结论，再写证据和过程，方便快速判断问题是否已解决。
- 遇到多轮排查时，最终报告要合并成一份完整版本，避免读者在多份文件之间来回拼接。
- 技术细节从浅到深展开：先解释现象和根因，再补充命令、路径、日志和备份位置。

## 实验配置选择经验

- 先按仓库 requirements 选解释器；本项目优先用 `/root/autodl-tmp/conda/envs/str-py310/bin/python`。
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
- tokenizer 的 `.decode()` 只是 token id 转字符串的 API，不要把它写成独立实验阶段或正式指标。

## 必须遵守的规则

- 报告语言为中文
- 不要改我的提示词文件，要写内容自己在后面加文件

## 每次任务结束后要添加经验、写日志

- 经验继续添加到/root/autodl-tmp/STReasoner_reproduce/00_new_codes/agents/readme.md。如无必要不删除其他经验，如果冲突则报告等我同意后删除原经验。

- codex需在reports下建XX-YY-MM-DD-id，其中XX是你顺着序号往后数一个，YY-MM-DD是今天日期，id从1开始，表示是今天的第几个日志

- 内容需要你从你在对话框中的输出填写，填写HH-MM：发生了什么事情。


## 原则

- 尽量精简
- 比如不要一次实验的结果放prediction.json，non_prediction.json等多个json

## 文件名

- stage2包括2.1, 2.2等实验