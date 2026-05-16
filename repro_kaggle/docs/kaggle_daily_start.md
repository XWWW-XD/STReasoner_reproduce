
适用目标：
在 Kaggle T4×2 上复现 STReasoner，并用本地 VS Code 通过 Remote Tunnel 编辑和运行代码。

【维护提醒】：
这份文档是当前启动流程的最终版。以后如果 Codex 或其他自动化助手想修改这份文档，必须先获得我的明确同意；如果确实修改了，还要提醒我去 Obsidian 里的同一版笔记同步更新。

## 0. 概述

```text
从 GitHub pull代码
环境靠 setup 脚本重建
模型权重后续尽量做缓存 / Kaggle Dataset
/kaggle/working 只当临时工作区
```

---

## 1. Kaggle Notebook 设置

在界面右侧
- [ ] Internet: On
- [ ] Accelerator: GPU T4 x2

## 2. 启动 VS Code Tunnel

- [ ] 在 Kaggle Notebook 运行已有cell，启动vsc tunnel
- [ ] 然后在输出中点击连接，完成授权

## 3. 本地 VS Code 连接 Kaggle

- [ ] 本地 VS Code 中点击右下角图标，选择连接到隧道-github-kaggle_node

连接成功后，左下角应显示：

## 4. vsc安装插件

- [ ] 在 'kaggle-gpu-node' 中安装扩展以进行启用。
- [ ] 搜索Python、Pylance、Jupyter、Codex

## 5. Codex 登录

- [ ] 打开 Codex 面板 -> 点击 Sign in -> 选择 Sign in with Device Code -> 复制设备码
- [ ] 在弹出网页中登陆chatgpt账号，输入设备码

## 6. 打开或拉取STReasoner

在vsc终端进行如下操作：

如果项目不存在：
```bash
cd /kaggle/working
git clone https://github.com/XWWW-XD/STReasoner_reproduce.git STReasoner_reproduce
```

如果项目已经存在：
```bash
cd /kaggle/working/STReasoner_reproduce
git pull --ff-only
```

## 7. 恢复环境依赖

- [ ] 进入项目目录
- [ ] 运行一键恢复脚本
```bash
cd /kaggle/working/STReasoner_reproduce
bash repro_kaggle/scripts/setup_kaggle_t4.sh
```

这个脚本会自动执行：
- 重新生成 `requirements_no_flash.txt`
- 安装依赖
- 升级 `bitsandbytes`
- 检查 GPU / torch / transformers / datasets / bitsandbytes / vllm

## 8. （非必做）确认环境正常

- [ ] 看到 `gpu count: 2`
- [ ] 看到两张 `Tesla T4`
- [ ] `flash_attn` 显示 warning 没关系
- [ ] `bitsandbytes` 和 `vllm` 没有显示 `MISSING`

如果只想单独检查环境：
```bash

python repro_kaggle/scripts/check_kaggle_env.py

```

## 9. 结束后需要做

- [ ] 提交并推送代码
- [ ] 回 Kaggle 页面 Stop Session
