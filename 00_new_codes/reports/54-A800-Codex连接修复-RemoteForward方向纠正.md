# A800 Codex 连接修复 — RemoteForward 方向纠正

日期：2026-06-12

## 0. 结论

A800（2 服务器 / `autodl-A800`）右侧 VS Code Codex 插件已恢复可用。根因不是插件损坏、不是缺少 `auth.json`、也不是远端 `server-env-setup` 没写好，而是 **Windows SSH config 用了错误的转发方向**：

```text
错误：LocalForward 17997 127.0.0.1:7897   → 只在 Windows 本机监听 17997
正确：RemoteForward 17997 127.0.0.1:7897  → 在远端监听 17997，转回 Windows 7897
```

Codex `app-server` 和 `extensionHost` 运行在 **远端**，它们访问的是远端 `127.0.0.1:17997`。只有 `RemoteForward` 才能在远端建立代理入口。

## 1. 问题现象

| 项 | 描述 |
| --- | --- |
| 环境 | AutoDL A800，`connect.nma1.seetacloud.com:48192`，VS Code Host 别名 `autodl-A800` |
| 对比基准 | 1 服务器 Codex 右侧 UI 稳定可用 |
| 用户操作 | VS Code 右侧 Codex 插件选择设备码登录 |
| 失败表现 | 弹出错误，登录页不在 Windows 本地浏览器打开 |
| 使用方式 | Codex 在 VS Code Remote-SSH 中使用，不是 Cursor |

初期容易误判为「浏览器没弹」或「插件版本问题」。实际上登录请求在远端就失败了，还没走到 Windows `openExternal` 开浏览器那一步。

## 2. 报告链条

本次问题叠在报告 49 / 50 之后：

| 报告 | 已解决 |
| --- | --- |
| `49-VSCode远程与Codex稳定连接配置迁移清单.md` | 远端 VS Code Server / Codex 代理继承 |
| `50-2026-06-12-2.md` / `51-新服务器SSH免密登录...` | A800 远端配置写入、SSH 免密 |
| `52-新服务器VSCode-Codex设备码登录页不弹出排查.md` | 分层排查框架 |
| `52-回复新服务器AI-Codex远端17997转发方向排查.md` | 1 服务器硬证据、转发方向确认 |

报告 54 记录最终根因与修复。

## 3. 排查过程摘要

### 3.1 A800 远端配置已齐，但 17997 不通

A800 远端已具备：

- `/root/.vscode-server/server-env-setup`
- `/root/.vscode-server/data/User/settings.json` 与 `Machine/settings.json`
- `/root/.codex/codex-vscode-wrapper.sh`
- `/root/.bashrc` 代理与 `[autodl-A800]` 提示符
- SSH 免密（`authorized_keys` 指纹 `SHA256:R60e0Dly8FpQ+Z7pTBArLesXFqOGlsdrl72wqmXHMNk`）

`extensionHost` 和 `codex app-server` 进程环境里均有：

```text
HTTP_PROXY=http://127.0.0.1:17997
HTTPS_PROXY=http://127.0.0.1:17997
ALL_PROXY=http://127.0.0.1:17997
```

但故障时：

```bash
curl -I https://chatgpt.com --proxy http://127.0.0.1:17997
# Connection refused

python3 -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',17997))"
# ConnectionRefusedError
```

### 3.2 Codex 日志指向远端网络失败

`Codex.log` 关键错误：

```text
failed to request device code: error sending request for url
  (https://auth.openai.com/api/accounts/deviceauth/usercode)
```

以及大量 `TypeError: fetch failed`。与设备码登录页不弹、用户看到错误弹窗一致。

### 3.3 1 服务器对照实验

在 1 服务器上核对（详见 `52-回复...`）：

| 检查项 | 1 服务器结果 |
| --- | --- |
| `curl --proxy http://127.0.0.1:17997` | `HTTP/1.1 200 Connection established` |
| socket 直连 `127.0.0.1:17997` | `CONNECT_OK` |
| `/proc/net/tcp` | `0100007F:464D`（`127.0.0.1:17997` LISTEN） |
| 监听进程 | `sshd: root`（符合 SSH RemoteForward） |
| active `node` 包装 | 无（ELF 原始二进制） |
| `codex.real` 包装 | 无 |

结论：1 服务器稳定不靠 node/codex 二进制包装，靠的是 **远端 17997 真有 sshd 转发的代理入口**。

## 4. 根因：LocalForward 与 RemoteForward 方向错误

### 4.1 语义对比

| 配置 | 谁监听 17997 | 远端 Codex 能否用 `127.0.0.1:17997` |
| --- | --- | --- |
| `LocalForward 17997 127.0.0.1:7897` | Windows 本机 | ❌ 远端无监听 |
| `RemoteForward 17997 127.0.0.1:7897` | 远端服务器 | ✅ 转回 Windows `7897` |

记忆口诀：

```text
终端/浏览器在 Windows 要访问远端端口 → LocalForward
远端进程要访问 Windows 本地代理 → RemoteForward
```

Codex `app-server` / `extensionHost` 在远端，因此必须用 `RemoteForward`。

### 4.2 文档疏漏

报告 49 正确描述了「远端访问远端 `127.0.0.1:<PORT>`」，但仓库 snippet（`windows-ssh-config.snippet`）和报告 50 曾写成 `LocalForward`，导致按文档配置后远端代理入口仍不存在。

## 5. 修复内容

### 5.1 Windows SSH config（用户本机，关键修复）

在 `C:\Users\HUAWEI\.ssh\config` 的 `Host autodl-A800` 块：

```sshconfig
Host autodl-A800
  HostName connect.nma1.seetacloud.com
  Port 48192
  User root
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 20
  ServerAliveCountMax 6
  TCPKeepAlive yes
  ConnectTimeout 15
  ConnectionAttempts 3
  IPQoS none
  RemoteForward 17997 127.0.0.1:7897
  ExitOnForwardFailure yes
```

说明：

- `7897` 为 Windows 本地代理端口（Clash 等）。
- `ExitOnForwardFailure yes`：转发建不起来时 SSH 直接失败，避免「VS Code 看似连上、Codex 必挂」。
- 删除同 Host 块中的 `LocalForward 17997 ...`（若有）。

操作：关闭 VS Code 对 A800 的连接 → `Remote-SSH: Kill VS Code Server on Host...` → 重连 `autodl-A800` → Codex 重新设备码登录。

### 5.2 仓库与远端脚本修正（A800 上已做）

| 路径 | 变更 |
| --- | --- |
| `00_new_codes/repro_autodl/vsc_codex_remote_setup/windows-ssh-config.snippet` | `LocalForward` → `RemoteForward` + `ExitOnForwardFailure yes` |
| `00_new_codes/repro_autodl/vsc_codex_remote_setup/README.md` | 同步说明 |
| `/root/check-vsc-codex.sh` | 增加 socket 测试与 `/proc/net/tcp` 检查 |

## 6. 修复后验证（A800）

```bash
bash /root/check-vsc-codex.sh
```

成功标准：

```text
127.0.0.1:17997 CONNECT_OK
HTTP/1.1 200 Connection established   # curl --proxy
extensionHost / codex app-server 有 HTTP_PROXY 等环境变量
```

2026-06-12 修复后实测：

```text
127.0.0.1:17997 CONNECT_OK
HTTP/1.1 200 Connection established
HTTP/2 403   # Cloudflare 页面级响应，可忽略
```

用户确认 Codex 右侧 UI 已连接成功。

## 7. 不建议做的事（本次已验证可跳过）

- 不复制 1 服务器 `/root/.codex/auth.json`
- 不包装 VS Code Server `node` 或扩展目录 `codex`
- 不在代理未通时重装 Codex 插件
- 不把 `LocalForward` 当作远端 Codex 代理方案

## 8. 新机 / 新实例 checklist

1. 报告 51：SSH 免密（`authorized_keys`）。
2. 报告 49：远端 `server-env-setup`、`settings`、`codex-vscode-wrapper.sh`、`bashrc`。
3. **本报告**：Windows SSH config 写 `RemoteForward 17997 127.0.0.1:7897`（不是 `LocalForward`）。
4. VS Code Remote-SSH 四条保守设置（`useExecServer=false` 等）。
5. Kill VS Code Server 后重连。
6. `bash /root/check-vsc-codex.sh` 确认 `CONNECT_OK` + `Connection established`。
7. Codex 设备码登录（网页在 Windows 本地浏览器打开）。

## 9. 双机同时开启注意

1 服务器与 A800 各自远端 `127.0.0.1:17997` 互不冲突（不同机器的 localhost）。每台机器各自需要自己的 SSH 连接建立 `RemoteForward`。不要共用同一个 `Host autodl` 别名来回改 HostName。

## 10. 相关文件索引

```text
00_new_codes/reports/49-VSCode远程与Codex稳定连接配置迁移清单.md
00_new_codes/reports/51-新服务器SSH免密登录与反复输入密码排查.md
00_new_codes/reports/52-回复新服务器AI-Codex远端17997转发方向排查.md
00_new_codes/repro_autodl/vsc_codex_remote_setup/
/root/check-vsc-codex.sh
```
