# stage1 Code Map

画一张最小项目地图。

不是画得很漂亮，而是能回答：

数据从哪里来？
模型在哪里加载？
训练脚本调了哪个框架？
推理脚本在哪里？
评测脚本怎么计算指标？

主链路：
ST-Bench / 合成数据
    ↓
data_generation/ 生成数据
data/ 转换数据格式
download_dataset.py 下载数据
    ↓
base_model/ 自定义 Qwen + 时间序列处理器
    ↓
scripts/qwen3-8b/train_stage1.sh
scripts/qwen3-8b/train_stage2.sh
scripts/qwen3-8b/train_stage3.sh
    ↓
src/llamafactory/ 做 Stage 1/2 SFT
src/EasyR1/ 做 Stage 3 RL / GRPO
    ↓
inference/ 推理
    ↓
evaluation/ 评测

## 1. 项目目标
- 目标：在 Kaggle T4×2 上做 STReasoner 的资源受限复现
- 暂不做：完整 Stage 3 RL 训练

## 2. 主要目录
| 目录 | 猜测作用 | 证据文件 |
|---|---|---|
| src/llamafactory | SFT 训练框架 | |
| src/EasyR1 | RL / GRPO 训练框架 | |
| base_model | 自定义 Qwen + 时间序列模型结构 | |
| scripts | 运行入口 / 辅助脚本 | |
| outputs | 今日实验输出 | |

## 3. 训练阶段
| 阶段 | checkpoint | 作用 | Kaggle 可做性 |
|---|---|---|---|
| Stage 1 | STReasoner-8B-Align | 时间序列-文本对齐 | 可做推理/评测 |
| Stage 2 | STReasoner-8B-CoT | CoT 冷启动推理 | 可做推理/评测 |
| Stage 3 | STReasoner-8B | S-GRPO 后最终模型 | 可做推理/评测，暂不重训 |

## 4. 今日最小实验
- 环境检查：
- 数据集读取：
- 模型加载：
- 第一条样本推理：