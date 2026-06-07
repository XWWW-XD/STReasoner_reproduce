# 04_vsc连接远程服务器错误排查

## 1. 问题现象

在 VS Code 中通过 Remote-SSH 选择 `autodl` 主机连接 AutoDL 服务器时，连接过程非常卡，连接不久后报错：

```text
Failed to set up dynamic port forwarding connection over SSH to the VS Code Server. (Show log)
```

该报错说明 VS Code 已经开始走 SSH 连接流程，但在建立 VS Code Server 所需的端口转发通道时失败或超时。它不等价于普通 SSH 完全连不上。

## 2. 当前连接配置

本机 SSH 配置文件：

```text
C:\Users\HUAWEI\.ssh\config
```

当前 `autodl` 主机配置为：

```sshconfig
Host autodl region-9.autodl.pro
  HostName region-9.autodl.pro
  Port 49151
  User root
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 20
  ServerAliveCountMax 6
  TCPKeepAlive yes
  ConnectTimeout 15
  ConnectionAttempts 3
  IPQoS none
```

其中：

- `Host autodl`：给 VS Code 和终端使用的短别名。
- `HostName` / `Port`：来自 AutoDL 页面提供的 SSH 命令。
- `ServerAliveInterval 20` / `ServerAliveCountMax 6`：每 20 秒发送保活包，连续 6 次无响应才断开。
- `ConnectTimeout 15` / `ConnectionAttempts 3`：降低偶发网络抖动导致的连接失败概率。
- `IPQoS none`：避免部分网络环境下 SSH 交互卡顿。

## 3. 已完成的排查

### 3.1 普通 SSH 连接正常

执行：

```powershell
ssh -o BatchMode=yes autodl "echo SSH_OK; uname -a; command -v bash; command -v tar; command -v wget; command -v curl; df -h ~"
```

远端返回了：

```text
SSH_OK
Linux autodl-container-db401188fa-d0c86748 ...
/usr/bin/bash
/usr/bin/tar
/usr/bin/wget
/usr/bin/curl
overlay 30G 929M 30G 4% /
```

结论：

- SSH 登录链路可用。
- 远端基础环境正常。
- `bash`、`tar`、`wget`、`curl` 均存在。
- 根目录磁盘空间充足，不是磁盘满导致 VS Code Server 安装失败。

### 3.2 TCP 端口可达

执行：

```powershell
Test-NetConnection region-9.autodl.pro -Port 49151
```

结果显示：

```text
TcpTestSucceeded : True
```

结论：

- 本机可以访问 AutoDL 暴露的 SSH 端口。
- 不是端口不可达或服务器完全断开的情况。

### 3.3 动态端口转发本身可建立

模拟 VS Code Remote-SSH 的动态转发：

```powershell
ssh -N -D 127.0.0.1:51888 -o ExitOnForwardFailure=yes -o BatchMode=yes -vv autodl
```

日志中出现：

```text
Authenticated to region-9.autodl.pro ... using "publickey".
Local forwarding listening on 127.0.0.1 port 51888.
Remote: /root/.ssh/authorized_keys:1: key options: agent-forwarding port-forwarding pty user-rc x11-forwarding
```

结论：

- SSH 动态转发 `-D` 能建立。
- 远端没有禁用 `port-forwarding`。
- VS Code 报错不是因为 AutoDL 明确禁止端口转发。

### 3.4 通过 SOCKS 转发访问外网成功

执行：

```powershell
curl.exe --socks5-hostname 127.0.0.1:51888 -I --max-time 20 https://update.code.visualstudio.com
```

返回：

```text
HTTP/1.1 200 OK
```

结论：

- 动态端口转发不只是监听成功，也能实际转发网络请求。
- SSH 转发通道本身可用。

### 3.5 发现远端存在残留 VS Code Server 进程

执行：

```bash
ps -ef | grep -E 'vscode-server|code-f6cfa|node.*vscode' | grep -v grep
```

发现远端残留了 VS Code Server 相关进程，例如：

```text
/root/.vscode-server/code-f6cfa2ea2403534de03f069bdf160d06451ed282 --cli-data-dir /root/.vscode-server/cli agent host
.../server/node .../server/out/server-main.js ...
.../bootstrap-fork --type=extensionHost ...
.../bootstrap-fork --type=fileWatcher
.../bootstrap-fork --type=ptyHost
```

结论：

- VS Code 曾经启动到一半或启动成功后未正常退出。
- 残留进程可能干扰后续 Remote-SSH 连接。
- 这类残留状态容易造成 VS Code 本地端口转发、远端 server socket、token 或启动流程不一致。

## 4. 判断原因

综合排查结果，当前问题最可能不是以下原因：

- 不是 SSH 命令错误，因为 `ssh autodl` 可正常进入。
- 不是 AutoDL 端口不可达，因为 TCP 测试成功。
- 不是远端缺少基础工具，因为 `bash`、`tar`、`wget`、`curl` 都存在。
- 不是磁盘空间不足，因为根目录剩余空间充足。
- 不是 AutoDL 禁止端口转发，因为手动 `ssh -D` 和 SOCKS 测试成功。

更可能的原因是：

1. VS Code Remote-SSH 的新连接模式在该 AutoDL 网络环境中不稳定。

   当前 VS Code Remote-SSH 会使用较新的 exec server / 动态转发机制。AutoDL 的连接链路经过公网转发端口，延迟和抖动较明显，VS Code 在建立动态端口转发与 VS Code Server 通信时可能超时。

2. 远端残留 VS Code Server 进程导致状态不一致。

   远端已经存在 `/root/.vscode-server` 下的 server、agent、extensionHost、ptyHost 等进程。若上一次连接异常中断，这些进程可能继续占用 socket 或保持旧连接状态，导致下一次 VS Code 连接卡住或报端口转发失败。

3. AutoDL 服务器本身容易断连或卡顿。

   AutoDL 暴露的是临时公网 SSH 入口，和普通云服务器固定公网 IP 相比更容易出现短时间卡顿。普通终端 SSH 对这种卡顿容忍度较高，但 VS Code Remote-SSH 需要额外建立 server、端口转发、文件系统、扩展宿主等多个通道，因此更敏感。

## 5. 已做的处理

### 5.1 优化 SSH 保活配置

在 `C:\Users\HUAWEI\.ssh\config` 中为 `autodl` 添加了：

```sshconfig
ServerAliveInterval 20
ServerAliveCountMax 6
TCPKeepAlive yes
ConnectTimeout 15
ConnectionAttempts 3
IPQoS none
```

作用：

- 降低空闲连接被断开的概率。
- 网络轻微抖动时给 SSH 更多重试机会。
- 减少某些网络环境下 SSH 卡顿。

### 5.2 修改 VS Code Remote-SSH 设置

修改文件：

```text
C:\Users\HUAWEI\AppData\Roaming\Code\User\settings.json
```

新增：

```json
"remote.SSH.connectTimeout": 60,
"remote.SSH.useExecServer": false,
"remote.SSH.enableDynamicForwarding": false,
"remote.SSH.showLoginTerminal": true
```

作用：

- `connectTimeout: 60`：给 AutoDL 慢连接更多时间。
- `useExecServer: false`：关闭 Remote-SSH 较新的 exec server 连接模式，回退到更传统的连接方式。
- `enableDynamicForwarding: false`：避免 VS Code 强依赖动态端口转发，绕开当前报错点。
- `showLoginTerminal: true`：连接时显示登录终端，便于输入密码和观察卡住位置。

### 5.3 清理远端残留 VS Code Server 进程

已执行清理，最后确认：

```text
CLEAN
```

说明远端残留的 VS Code Server 进程已清掉。

## 6. 建议的连接流程

每次连接 AutoDL 前，先在本地 VS Code 终端测试普通 SSH：

```powershell
ssh autodl
```

如果可以进入远端：

```bash
exit
```

然后再用 VS Code：

```text
Ctrl + Shift + P
Remote-SSH: Connect to Host...
autodl
```

如果提示系统类型，选择：

```text
Linux
```

如果提示密码，输入 AutoDL 当前实例提供的密码。

注意：

- 第一次连接或清理 server 后，VS Code 需要重新启动/安装 VS Code Server，会比较慢。
- 不要连续快速点击多次连接，否则容易产生多个半启动进程。

## 7. 如果再次失败的处理步骤

如果仍然报：

```text
Failed to set up dynamic port forwarding connection over SSH to the VS Code Server.
```

建议按顺序操作：

### 7.1 先确认 SSH 仍可用

```powershell
ssh autodl
```

如果 SSH 都无法进入，优先检查：

- AutoDL 实例是否仍在运行。
- `C:\Users\HUAWEI\.ssh\config` 中的 `HostName` 是否是最新域名。
- `Port` 是否是最新端口。
- 密码是否是当前实例的新密码。

### 7.2 清理远端 VS Code Server 残留进程

进入远端后执行：

```bash
pkill -f vscode-server || true
pkill -f ".vscode-server" || true
exit
```

然后重新在 VS Code 中连接 `autodl`。

### 7.3 如果仍失败，再清理 VS Code Server 目录

仅在多次失败后执行：

```bash
rm -rf ~/.vscode-server
exit
```

然后重新连接 `autodl`。这会让 VS Code 重新安装远端 server，第一次会更慢。

### 7.4 查看 VS Code Remote-SSH 日志

在 VS Code 报错窗口点击：

```text
Show Log
```

重点查找以下关键词：

```text
dynamic port forwarding
exec server
Failed to parse remote port
Permission denied
Could not establish connection
```

如果需要继续排查，应保存日志最后 80 行。

## 8. 每次 AutoDL 换实例时需要改什么

AutoDL 每次新建实例后，SSH 命令和密码可能变化，例如：

```powershell
ssh -p 45678 root@region-12.autodl.pro
```

此时只需要修改：

```text
C:\Users\HUAWEI\.ssh\config
```

中的：

```sshconfig
HostName region-12.autodl.pro
Port 45678
```

其余保活和 VS Code 设置不需要每次重配。

推荐保留短别名：

```sshconfig
Host autodl
```

这样 VS Code 中始终选择 `autodl`，不用每次重新添加新主机。

## 9. 当前结论

本次问题的核心原因不是 SSH 账号密码错误，也不是 AutoDL 禁止端口转发，而是 VS Code Remote-SSH 在 AutoDL 这种临时公网 SSH 入口上建立 VS Code Server 通信通道时不稳定，并且远端残留的 VS Code Server 进程会进一步放大问题。

当前已经完成的修复方向是：

- 保留 `autodl` 短别名。
- 增加 SSH keepalive 和连接重试。
- 禁用 VS Code Remote-SSH 的 exec server 和动态转发模式。
- 清理远端残留 VS Code Server 进程。

后续如果再次出现同类问题，优先按“普通 SSH 测试 -> 清理远端 VS Code Server 进程 -> 重新连接 VS Code”的顺序处理。
