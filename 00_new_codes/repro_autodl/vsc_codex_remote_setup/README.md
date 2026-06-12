# VS Code Remote + Codex 本机配置片段

对应报告：`00_new_codes/reports/49-VSCode远程与Codex稳定连接配置迁移清单.md`

## 本实例（A800）

| 项 | 值 |
| --- | --- |
| SSH Host 别名 | `autodl-A800` |
| HostName | `connect.nma1.seetacloud.com` |
| Port | `48192` |
| Git 工作分支 | `autodl-A800` |
| 远端代理端口 | `127.0.0.1:17997` |
| Windows 本地代理 | `127.0.0.1:7897` |
| SSH 转发 | `RemoteForward 17997 127.0.0.1:7897`（远端监听 17997 → Windows 7897） |

## SSH 免密（先做，见报告 51）

本机当前 **没有** `/root/.ssh/authorized_keys`，VS Code 会反复问密码。

在 Windows PowerShell **执行一次**（只输这一次密码）：

```powershell
cd <仓库路径>\00_new_codes\repro_autodl\vsc_codex_remote_setup
.\install-ssh-pubkey-windows.ps1
```

或一行：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh autodl-A800 bash /root/install-ssh-pubkey.sh
ssh -o BatchMode=yes autodl-A800 "echo SSH_KEY_OK"
```

成功应输出 `SSH_KEY_OK`。指纹应与 1 服务器一致：`SHA256:R60e0Dly8FpQ+Z7pTBArLesXFqOGlsdrl72wqmXHMNk`。

也可在 AutoDL 控制台 → SSH 公钥 → 粘贴 `id_ed25519.pub` 内容 → 重启实例。

## Windows 操作顺序

1. 把 `windows-ssh-config.snippet` 合并进 `C:\Users\HUAWEI\.ssh\config`（保留已有 `Host autodl` 等条目）。**必须是 `RemoteForward`，`LocalForward` 不能让远端 17997 监听。**
2. 把 `windows-local-settings.json` 四条写入 VS Code **User** settings。
3. `remote.SSH.remotePlatform` 加 `"autodl-A800": "linux"`。
4. 确认本机代理在 `7897` 已启动。
5. VS Code：`Remote-SSH: Connect to Host...` → **`autodl-A800`**。
6. `Remote-SSH: Kill VS Code Server on Host...` → 再重连。
7. 远端：`bash /root/check-vsc-codex.sh`。
8. Codex 重新登录。

## 远端已配置文件

- `/root/.vscode-server/server-env-setup`
- `/root/.vscode-server/data/User/settings.json`
- `/root/.vscode-server/data/Machine/settings.json`
- `/root/.codex/codex-vscode-wrapper.sh`
- `/root/.bashrc`（代理 + `[autodl-A800]` 提示符）
