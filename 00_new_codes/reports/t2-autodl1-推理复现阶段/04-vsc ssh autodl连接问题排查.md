#### 1. 问题现象

这次的问题是：VS Code 通过 Remote-SSH 连接 AutoDL 时非常卡，最后报错：

```text
Failed to set up dynamic port forwarding connection over SSH to the VS Code Server.
```

这句话的意思不是“SSH 完全连不上”，而是：VS Code 已经开始连服务器了，但它后面要建立一个给 VS Code Server 用的通信通道，这一步失败了。

简单说：

```text
普通 ssh 可能还能用；
但 VS Code Remote-SSH 需要更多连接步骤，所以更容易卡住或失败。
```



#### 2. 当前 SSH 配置

本地 SSH 配置文件位置：

```text
C:\Users\HUAWEI\.ssh\config
```

当前 AutoDL 的配置大致是：

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

这里最重要的是：

```text
Host autodl
```

它是短别名。以后 VS Code 里直接选 `autodl` 就行，不用每次手动输入完整 SSH 命令。

其他几项主要是为了让连接更稳：

```text
ServerAliveInterval / ServerAliveCountMax：定期保活，减少空闲断开。
ConnectTimeout / ConnectionAttempts：连接慢的时候多给一点时间和重试机会。
IPQoS none：避免部分网络环境下 SSH 卡顿。
```

#### 3. 已经确认正常的部分

先测试了普通 SSH：

```powershell
ssh -o BatchMode=yes autodl "echo SSH_OK; uname -a; command -v bash; command -v tar; command -v wget; command -v curl; df -h ~"
```

结果正常返回了：

```text
SSH_OK
bash / tar / wget / curl 都存在
根目录空间充足
```

说明：

```text
SSH 账号和端口基本没问题；
服务器能登录；
远端基础工具齐全；
不是磁盘满导致 VS Code Server 装不上。
```

又测试了 AutoDL 的 SSH 端口：

```powershell
Test-NetConnection region-9.autodl.pro -Port 49151
```

结果：

```text
TcpTestSucceeded : True
```

说明本机能连到 AutoDL 暴露出来的 SSH 端口，不是端口完全不通。

还手动模拟了 VS Code 的动态转发：

```powershell
ssh -N -D 127.0.0.1:51888 -o ExitOnForwardFailure=yes -o BatchMode=yes -vv autodl
```

日志显示动态端口能正常监听，并且远端没有禁用端口转发。

再用这个 SOCKS 通道访问 VS Code 更新服务器，也返回了：

```text
HTTP/1.1 200 OK
```

说明：

```text
SSH 动态转发本身是可用的；
AutoDL 没有禁止 port forwarding；
问题更像是 VS Code Remote-SSH 自己的连接流程卡住。
```

#### 4. 当前判断

这次问题大概率不是：

```text
不是 SSH 命令写错；
不是 AutoDL 端口不可达；
不是密码或密钥完全错误；
不是远端缺少 bash / tar / wget / curl；
不是磁盘空间不够；
不是 AutoDL 禁用了端口转发。
```

更可能是两个问题叠加：

```text
1. AutoDL 的临时公网 SSH 入口本来就容易抖动，VS Code Remote-SSH 比普通 ssh 更敏感。
2. 远端残留的 VS Code Server 进程导致连接状态混乱。
```

普通 SSH 只需要建立一个终端连接，所以它能忍受一些网络抖动。

但 VS Code Remote-SSH 要做更多事情：

```text
登录服务器；
安装或启动 VS Code Server；
建立端口转发；
连接远端文件系统；
启动扩展宿主；
维持多个通信通道。
```

所以它比普通终端 ssh 更容易失败。

#### 5. 已经做过的修复

本地 SSH 配置里已经加了保活和重试参数：

```sshconfig
ServerAliveInterval 20
ServerAliveCountMax 6
TCPKeepAlive yes
ConnectTimeout 15
ConnectionAttempts 3
IPQoS none
```

作用是让 SSH 连接更耐受网络抖动。

VS Code 的 Remote-SSH 设置也改了：

```json
"remote.SSH.connectTimeout": 60,
"remote.SSH.useExecServer": false,
"remote.SSH.enableDynamicForwarding": false,
"remote.SSH.showLoginTerminal": true
```

含义是：

```text
connectTimeout: 60
给 AutoDL 慢连接更多时间。

useExecServer: false
关闭 Remote-SSH 较新的 exec server 模式，回退到更传统的连接方式。

enableDynamicForwarding: false
绕开当前容易报错的 dynamic port forwarding 流程。

showLoginTerminal: true
显示登录终端，方便看它到底卡在哪一步。
```

远端残留的 VS Code Server 进程也已经清理过，最后确认结果是：

```text
CLEAN
```

