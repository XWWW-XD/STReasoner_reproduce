# HeaRTS 调研资源卸载说明

## 1. 会占用磁盘的内容

| 类型 | 路径 | 典型体积 |
|------|------|----------|
| HF 冻结数据 + git 元数据 | `/root/autodl-tmp/datasets/HEARTS/frozen_test_cases/` | ~0.5 MB（10 个 pkl） |
| HF 隔离缓存 | `/root/autodl-tmp/datasets/HEARTS/.hf_cache/` | 数 MB |
| HEARTS 代码浅克隆 | `/root/autodl-tmp/datasets/HEARTS/code/` | ~8 MB |
| 预检/清单 | `/root/autodl-tmp/datasets/HEARTS/preflight_result.json` | 很小 |
| 调研脚本 | `new_datasets/hearts_survey/` | 很小 |
| 图表与 JSON | `new_datasets/artifacts/` | 数 MB（v1 遗留 `fig_*.png` 可删，v2 报告不引用） |
| Markdown 报告 | `new_datasets/01-*.md`、`02-*.md` | 很小 |

**不会删除（复现保护区）：**

- `STReasoner_reproduce/data/`、checkpoints、models
- `/root/.vscode-server/`、`/root/.codex/`
- 任何 Stage 训练输出

## 2. 一键卸载

```bash
cd /root/autodl-tmp/STReasoner_reproduce/00_new_codes/reports/new_datasets/hearts_survey
chmod +x uninstall.sh   # 首次
```

| 级别 | 命令 | 效果 |
|------|------|------|
| **L1 推荐** | `bash uninstall.sh --data-only` | 只删 `/root/autodl-tmp/datasets/HEARTS/`，保留报告与 artifacts |
| L2 | `bash uninstall.sh --with-artifacts` | L1 + 删 `artifacts/` 与 `hearts_survey/` |
| L3 | `bash uninstall.sh --all` | 删除整个 `new_datasets/` |

## 3. 卸载后验证

```bash
test ! -e /root/autodl-tmp/datasets/HEARTS && echo "HEARTS data removed"
test -d /root/autodl-tmp/STReasoner_reproduce/data && echo "STReasoner data OK"
df -h /root/autodl-tmp
```

## 4. 重新下载（若日后需要）

```bash
export HF_HOME=/root/autodl-tmp/datasets/HEARTS/.hf_cache
export https_proxy=http://127.0.0.1:17997 http_proxy=http://127.0.0.1:17997
python3 hearts_survey/preflight.py --out /root/autodl-tmp/datasets/HEARTS/preflight_result.json
python3 hearts_survey/download.py
```

说明：`huggingface_hub.snapshot_download` 在本机代理环境下可能失败；`download.py` 使用 **git clone + curl resolve** 拉取真实 LFS 文件（非 130B 指针）。

## 5. 预检失败时

若 `preflight_result.json` 中 `pass: false`（磁盘不足、HF 非公开、网络不可达），**不要下载**。仅保留本说明与预检 JSON 即可。
