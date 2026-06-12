# Git 操作说明（AutoDL A800 环境）

> 日期：2026-06-12  
> 目的：固定本机仓库路径与分支约定，便于在 terminal 直接 `git pull` / `git push`，并与本地 Windows / 其他机器协同。

---

## 1. 固定信息（先看这里）

| 项 | 值 |
| --- | --- |
| 远程仓库 | https://github.com/XWWW-XD/STReasoner_reproduce |
| **本机固定路径** | `/root/autodl-tmp/STReasoner_reproduce` |
| **本机工作分支** | `autodl-A800` |
| 默认上游分支 | `main`（远程 `origin/main`） |
| Python（本项目） | `/root/autodl-tmp/conda/envs/str-py310/bin/python` |

以后在 terminal 里操作，**先进入固定目录**：

```bash
cd /root/autodl-tmp/STReasoner_reproduce
```

---

## 2. 分支说明

远程当前分支：

| 分支 | 用途 |
| --- | --- |
| `main` | 主开发线，最新合并（含 `local` 合入） |
| `autodl` | 历史 AutoDL 复现线，略早于当前 `main` |
| `local` | 本地实验合并线，已合入 `main` |
| `autodl-A800` | **本 A800 实例专用**（本地已建，远程待首次 push） |

本实例策略：

- 日常在 `autodl-A800` 上改代码、写报告、跑实验日志。
- 需要同步他人 `main` 更新时，在 `autodl-A800` 上 `git pull origin main` 或 `git merge origin/main`。
- **不要**在 AutoDL 上直接改 `main` 并强推，避免覆盖远程主分支。

---

## 3. 首次设置（本机已完成克隆时可跳过）

### 3.1 克隆到固定路径

```bash
cd /root/autodl-tmp
git clone https://github.com/XWWW-XD/STReasoner_reproduce.git STReasoner_reproduce
cd STReasoner_reproduce
```

### 3.2 创建并切换到 `autodl-A800`

```bash
git fetch origin
git checkout -b autodl-A800 origin/main   # 或基于当前 main
```

### 3.3 绑定远程分支（只需做一次）

远程尚无 `autodl-A800` 时，本地首次推送并建立跟踪：

```bash
git push -u origin autodl-A800
```

之后在本分支可直接：

```bash
git pull    # 拉 origin/autodl-A800
git push    # 推 origin/autodl-A800
```

若尚未执行过 `git push -u`，`git pull` 会提示没有上游分支——按 3.3 执行一次即可。

### 3.4 确认状态

```bash
git branch -vv
git status
git remote -v
```

期望：`autodl-A800` 带 `[origin/autodl-A800]` 跟踪信息，工作区干净或仅有预期改动。

---

## 4. 日常使用

### 4.1 拉取远程本分支更新

```bash
cd /root/autodl-tmp/STReasoner_reproduce
git checkout autodl-A800
git pull
```

### 4.2 同步 `main` 上的新提交到本分支

```bash
cd /root/autodl-tmp/STReasoner_reproduce
git checkout autodl-A800
git fetch origin
git merge origin/main
# 若有冲突，解决后：
git add <冲突文件>
git commit
```

### 4.3 提交并推送

```bash
cd /root/autodl-tmp/STReasoner_reproduce
git status
git add <路径>          # 例如报告、脚本、json/jsonl 实验结果
git commit -m "简述本次改动"
git push
```

报告与日志约定（与 `00_new_codes/guides/修改文件必读规则.md` 一致）：

- 报告放在 `00_new_codes/reports/` 下，按阶段子目录编号。
- 正文开头写日期；大改时补修改日期。
- 不要改提示词模板文件，需要内容时新建文件。

### 4.4 只看远程有什么新提交

```bash
git fetch origin
git log --oneline autodl-A800..origin/main    # main 比本分支多什么
git log --oneline origin/main..autodl-A800    # 本分支比 main 多什么
```

---

## 5. 网络与代理

### AutoDL 服务器

- 一般 **直连 GitHub** 即可；本机若设置了 `127.0.0.1:7897` 代理但代理未开，会导致 clone/pull 失败。
- 失败时先清代理再试：

```bash
unset https_proxy http_proxy all_proxy
git pull
```

### 本地 Windows（VS Code Remote-SSH）

- 本地终端可走代理；SSH 到 AutoDL 后执行 git 时以服务器网络为准。
- Remote-SSH 连接问题见同目录 `04-vsc ssh autodl连接问题排查.md`。

---

## 6. 什么应该提交、什么不要提交

仓库 `.gitignore` 已排除大文件与本地环境，**不要强行 `git add -f` 以下类型**：

| 不提交 | 说明 |
| --- | --- |
| `base_model/`、`checkpoints/` | 模型权重与训练 checkpoint |
| `*.safetensors`、`*.bin`、`*.pt`、`*.pth` | 权重文件 |
| `cache/`、`hf_cache/`、`.cache/` | HuggingFace / torch 缓存 |
| `wandb/` | 训练日志 |
| 默认 `exp/**` 中非 json/jsonl | 大量日志；**json/jsonl 实验结果按规则可跟踪** |
| `data/ST-Bench/**` 等大体积原始数据 | 用 `download_dataset.py` 本地下载 |

**通常应提交**：

- `00_new_codes/reports/**/*.md` 报告与说明
- `00_new_codes/repro_autodl/experiments/results/**/*.json` / `*.jsonl`（复现实验结果）
- `exp/**/*.json` / `*.jsonl`、`evaluation_metrics.json` 等（按 `.gitignore` 白名单）
- 脚本、配置、对 `inference/` / `evaluation/` 的修复

模型与数据路径（关机续用，无需每次重下）：

- STReasoner-8B：`/root/autodl-tmp/STReasoner_reproduce/base_model/STReasoner-8B/`
- HF 缓存：`/root/autodl-tmp/cache/huggingface`

这些目录在 git 外维护；换机器或清盘需重新下载。

---

## 7. 常见场景速查

| 场景 | 命令 |
| --- | --- |
| 每天开工拉代码 | `cd .../STReasoner_reproduce && git pull` |
| 同步 main | `git fetch origin && git merge origin/main` |
| 放弃未提交改动（慎用） | `git checkout -- <文件>` 或 `git restore <文件>` |
| 暂存当前改动去拉代码 | `git stash && git pull && git stash pop` |
| 查看当前分支 | `git branch -vv` |
| 换到 main 只读 | `git checkout main && git pull` |

---

## 8. 与仓库结构的对应关系

本仓库 = **官方 STReasoner 实现** + **`00_new_codes` 复现层**：

- 官方训练/推理/评测：`scripts/`、`src/`、`inference/`、`evaluation/`、`data/`
- AutoDL 复现脚本与结果：`00_new_codes/repro_autodl/`
- 报告与 artifacts：`00_new_codes/reports/`
- 代码阅读与 pipeline：`00_new_codes/guides/`

Git 只同步代码与约定格式的实验 json/jsonl；**训练、全量 ST-Test 推理、模型权重**在实例本地执行，不依赖 git 传大文件。

---

## 9. 故障排查

| 现象 | 处理 |
| --- | --- |
| `Connection refused` 到 127.0.0.1:7897 | `unset https_proxy http_proxy all_proxy` 后重试 |
| `no tracking information` | 执行 `git push -u origin autodl-A800` |
| `rejected` push | 先 `git pull`（或 `git pull --rebase`）再 push |
| 误 add 了大文件 | `git reset HEAD <文件>`，勿提交；大文件用 `git lfs` 或不要入库 |
| 不确定能否提交 | `git status` + 对照第 6 节 |

---

## 10. 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-06-12 | 初稿：固定路径、工作分支 `autodl-A100`（后更正为 `autodl-A800`，GPU 为 A800） |
