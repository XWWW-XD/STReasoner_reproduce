# 服务器连接与 Codex 稳定连接完整攻略

合并来源：报告 49-54  
合并日期：2026-06-14

这份文档把 49-54 号报告合并成一份可复用攻略。以后换新 AutoDL / SeetaCloud / 远端 GPU 实例时，优先按本攻略做，不要只照旧报告中的某一个片段。

## 0. 最终结论

VS Code Remote-SSH 里的右侧 Codex 插件能稳定工作，依赖四层都通：

1. Windows 到远端服务器的 SSH 免密登录稳定。
2. Windows 本地代理通过 SSH `RemoteForward` 转发到远端 `127.0.0.1:17997`。
3. 远端 VS Code Server / `extensionHost` / `codex app-server` 都继承代理环境。
4. Codex 登录态重新在新服务器上登录，不复制旧服务器 token。

其中最容易踩坑的是第二层：

```text
错误：LocalForward 17997 127.0.0.1:7897
正确：RemoteForward 17997 127.0.0.1:7897
```

原因很简单但很关键：

```text
Codex app-server 和 extensionHost 运行在远端。
它们访问 127.0.0.1:17997 时，访问的是远端自己的 localhost。
所以必须让远端监听 17997，再把流量转回 Windows 本地代理端口。
```

记忆口诀：

```text
Windows 本地要访问远端服务 -> LocalForward
远端进程要访问 Windows 本地代理 -> RemoteForward
```

本攻略推荐的默认端口约定：

| 名称 | 推荐值 | 说明 |
| --- | --- | --- |
| Windows 本地代理端口 | `7897` | Clash / 代理软件实际监听端口，以本机为准。 |
| 远端代理入口端口 | `17997` | 远端服务器监听，给 VS Code / Codex 使用。 |
| 远端代理 URL | `http://127.0.0.1:17997` | 写入远端环境变量和 VS Code settings。 |

如果本地代理不是 `7897`，只改 SSH config 里的本地端口；远端仍建议统一用 `17997`。如果远端 `17997` 被占用，再统一换成别的端口，并同步修改所有远端配置。

## 1. 一次性总流程

新服务器按这个顺序做：

1. Windows 确认本地代理已启动，例如 `127.0.0.1:7897`。
2. Windows 准备 SSH key，并把公钥放到新服务器 `/root/.ssh/authorized_keys`。
3. Windows `C:\Users\HUAWEI\.ssh\config` 新增独立 Host，例如 `autodl-A800` 或 `autodl-2`。
4. Host 块里写 keepalive、`IdentityFile`、`RemoteForward 17997 127.0.0.1:7897`、`ExitOnForwardFailure yes`。
5. Windows VS Code User settings 写入 Remote-SSH 保守设置。
6. 用 VS Code Remote-SSH 连接该 Host。
7. 远端写入：
   - `/root/.vscode-server/server-env-setup`
   - `/root/.codex/codex-vscode-wrapper.sh`
   - `/root/.vscode-server/data/User/settings.json`
   - `/root/.vscode-server/data/Machine/settings.json`
   - `/root/.bashrc` 代理环境变量
8. 执行 `Remote-SSH: Kill VS Code Server on Host...`，再重连。
9. 远端运行 `/root/check-vsc-codex.sh` 或本文检查命令。
10. 右侧 Codex 插件重新设备码登录。

不要跳过 SSH 免密。VS Code Remote-SSH 会建立多次 SSH 连接，如果只靠 root 密码，常会反复弹密码，后续代理和 Codex 排查都会被干扰。

## 2. Windows SSH 免密登录

### 2.1 检查或生成 key

在 Windows PowerShell：

```powershell
dir $env:USERPROFILE\.ssh\id_ed25519*
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

应至少存在：

```text
id_ed25519
id_ed25519.pub
```

如果没有，生成：

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519 -C "autodl-vscode"
```

注意：

- `id_ed25519` 是私钥，不要复制到服务器报告、仓库或聊天中。
- `id_ed25519.pub` 是公钥，可以放到服务器。
- 如果 key 设置了 passphrase，要配置 Windows `ssh-agent`，否则 VS Code 可能反复问 passphrase。

### 2.2 添加公钥到新服务器

优先方法：在 AutoDL / SeetaCloud 控制台添加 SSH 公钥。复制：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

把整行粘贴到控制台的 SSH 公钥配置里，然后重启或重新打开实例。

如果只能先用密码登录，可以从 Windows 追加：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh root@<SERVER_HOST> -p <SERVER_PORT> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chown -R root:root ~/.ssh"
```

如果已经进入远端，也可在远端执行：

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cat >> /root/.ssh/authorized_keys <<'EOF'
<把 Windows id_ed25519.pub 的整行公钥粘贴到这里>
EOF
chmod 600 /root/.ssh/authorized_keys
chown -R root:root /root/.ssh
```

报告 51 记录的旧机公钥指纹是：

```text
SHA256:R60e0Dly8FpQ+Z7pTBArLesXFqOGlsdrl72wqmXHMNk
```

这只是核对用，不需要把私钥或旧服务器文件复制过来。

### 2.3 验证免密

Windows PowerShell：

```powershell
ssh -o BatchMode=yes <HOST_ALIAS> "echo SSH_KEY_OK; hostname; whoami"
```

成功应看到：

```text
SSH_KEY_OK
<server-hostname>
root
```

`BatchMode=yes` 的意义是：key 不通时直接失败，不会退回密码输入。这比“手动输密码能连上”更准确。

### 2.4 如果仍然反复要密码

先检查 VS Code 选的是正确 Host alias，不要把两台服务器都叫 `autodl`。

Windows PowerShell：

```powershell
ssh -G <HOST_ALIAS> | findstr /i "hostname port user identityfile identitiesonly"
```

应该看到：

```text
user root
identityfile ...id_ed25519
identitiesonly yes
hostname <SERVER_HOST>
port <SERVER_PORT>
```

强制只用公钥测试：

```powershell
ssh -vvv -o PreferredAuthentications=publickey -o PasswordAuthentication=no <HOST_ALIAS> "echo SSH_KEY_OK"
```

日志判断：

| 日志现象 | 含义 |
| --- | --- |
| `Offering public key` | 本地确实拿 key 去试了。 |
| `Server accepts key` | 服务器接受公钥，免密应成功。 |
| `Permission denied (publickey)` | 服务器没有对应公钥，或权限不对。 |
| 没有 `Offering public key` | 本地没有用到正确私钥。 |

远端权限检查：

```bash
ls -ld /root /root/.ssh
ls -l /root/.ssh/authorized_keys
wc -l /root/.ssh/authorized_keys
ssh-keygen -lf /root/.ssh/authorized_keys
```

推荐权限：

```text
/root/.ssh                  700
/root/.ssh/authorized_keys  600
owner                       root:root
```

如果提示的是私钥 passphrase，而不是服务器 root 密码，配置 Windows `ssh-agent`：

```powershell
Get-Service ssh-agent
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
ssh-add -l
```

## 3. Windows SSH config 模板

路径：

```text
C:\Users\HUAWEI\.ssh\config
```

推荐每台机器一个独立别名，不要所有实例都叫 `autodl`。

新服务器模板：

```sshconfig
Host <HOST_ALIAS>
  HostName <SERVER_HOST>
  Port <SERVER_PORT>
  User root
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 20
  ServerAliveCountMax 6
  TCPKeepAlive yes
  ConnectTimeout 15
  ConnectionAttempts 3
  IPQoS none
  RemoteForward 17997 127.0.0.1:<WINDOWS_PROXY_PORT>
  ExitOnForwardFailure yes
```

A800 修复后实测可用配置：

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

参数说明：

| 参数 | 作用 |
| --- | --- |
| `IdentityFile ~/.ssh/id_ed25519` | 指定私钥。 |
| `IdentitiesOnly yes` | 只用指定 key，避免 SSH 乱试 key。 |
| `ServerAliveInterval 20` | SSH 每 20 秒保活一次。 |
| `ServerAliveCountMax 6` | 容忍短暂网络抖动。 |
| `TCPKeepAlive yes` | 开启 TCP keepalive。 |
| `ConnectTimeout 15` | 连接慢时不要太快放弃。 |
| `ConnectionAttempts 3` | 自动多试几次。 |
| `IPQoS none` | 避免部分网络环境里 SSH 卡顿或丢包。 |
| `RemoteForward 17997 127.0.0.1:7897` | 远端监听 `17997`，转回 Windows 本地代理 `7897`。 |
| `ExitOnForwardFailure yes` | 转发建不起来时 SSH 直接失败，避免 VS Code 看似连上但 Codex 必挂。 |

如果 Host 块里曾有：

```sshconfig
LocalForward 17997 127.0.0.1:7897
```

请删除或改成 `RemoteForward`。`LocalForward` 只会在 Windows 本机监听 `17997`，不能让远端 Codex 用到代理。

## 4. Windows VS Code Remote-SSH 设置

VS Code User settings 写入：

```json
{
  "remote.SSH.connectTimeout": 60,
  "remote.SSH.useExecServer": false,
  "remote.SSH.enableDynamicForwarding": false,
  "remote.SSH.showLoginTerminal": true
}
```

建议同时为 Host alias 指定远端平台：

```json
{
  "remote.SSH.remotePlatform": {
    "autodl-A800": "linux"
  }
}
```

含义：

| 设置 | 作用 |
| --- | --- |
| `remote.SSH.connectTimeout=60` | 给远端慢连接更长等待时间。 |
| `remote.SSH.useExecServer=false` | 关闭较新的 exec server 模式，使用更保守链路。 |
| `remote.SSH.enableDynamicForwarding=false` | 绕开旧机曾报错的 dynamic forwarding 流程。 |
| `remote.SSH.showLoginTerminal=true` | 连接卡住时能看到登录终端输出。 |

稳定后可以再尝试恢复默认；迁移时先按这组保守设置。

## 5. 远端 VS Code / Codex 配置

以下命令在远端服务器执行。默认远端代理端口是 `17997`。

### 5.1 写 VS Code Server 环境

文件：

```text
/root/.vscode-server/server-env-setup
```

内容：

```bash
export HTTP_PROXY=http://127.0.0.1:17997
export HTTPS_PROXY=http://127.0.0.1:17997
export http_proxy=http://127.0.0.1:17997
export https_proxy=http://127.0.0.1:17997
export ALL_PROXY=http://127.0.0.1:17997
export all_proxy=http://127.0.0.1:17997
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export NODE_USE_ENV_PROXY=1
```

这个文件让 VS Code Server 启动出的后台进程更容易继承代理。写完后必须 Kill VS Code Server 并重连，旧进程不会自动继承。

### 5.2 写 Codex wrapper

文件：

```text
/root/.codex/codex-vscode-wrapper.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

CODEX_PROXY="${CODEX_PROXY:-http://127.0.0.1:17997}"

export HTTP_PROXY="$CODEX_PROXY"
export HTTPS_PROXY="$CODEX_PROXY"
export http_proxy="$CODEX_PROXY"
export https_proxy="$CODEX_PROXY"
export ALL_PROXY="$CODEX_PROXY"
export all_proxy="$CODEX_PROXY"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NODE_USE_ENV_PROXY="${NODE_USE_ENV_PROXY:-1}"

while IFS= read -r candidate; do
  [ -x "$candidate" ] || continue
  case "$candidate" in
    /root/.codex/codex-vscode-wrapper.sh) continue ;;
  esac
  exec "$candidate" "$@"
done < <(
  find /root/.vscode-server/extensions -maxdepth 4 -type f \
    \( -path '*/openai.chatgpt-*/bin/linux-x86_64/codex.real' \
       -o -path '*/openai.chatgpt-*/bin/linux-x86_64/codex' \) \
    2>/dev/null | sort -r
)

echo "Unable to find bundled Codex executable under /root/.vscode-server/extensions/openai.chatgpt-*" >&2
exit 127
```

权限：

```bash
chmod 755 /root/.codex/codex-vscode-wrapper.sh
```

这个 wrapper 的目标是：设置代理环境变量，再执行 VS Code ChatGPT/Codex 扩展自带的 `codex` 二进制。优先不要直接改扩展目录里的二进制。

### 5.3 写 VS Code 远端 settings

两个文件都建议写，内容保持一致：

```text
/root/.vscode-server/data/User/settings.json
/root/.vscode-server/data/Machine/settings.json
```

内容：

```json
{
  "chatgpt.cliExecutable": "/root/.codex/codex-vscode-wrapper.sh",
  "http.proxy": "http://127.0.0.1:17997",
  "http.proxySupport": "override",
  "http.proxyStrictSSL": false
}
```

这一步对右侧 UI 很关键：它让 VS Code 和 ChatGPT/Codex 扩展知道远端要走哪个代理，并指定 Codex CLI wrapper。

### 5.4 写 shell 代理

为了让远程终端里的 `curl`、`git`、`codex` 等命令也走同一代理，可在 `/root/.bashrc` 追加：

```bash
# proxy for Codex / VS Code Remote
export HTTP_PROXY=http://127.0.0.1:17997
export HTTPS_PROXY=http://127.0.0.1:17997
export http_proxy=http://127.0.0.1:17997
export https_proxy=http://127.0.0.1:17997
export ALL_PROXY=http://127.0.0.1:17997
export all_proxy=http://127.0.0.1:17997
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

也可以给不同服务器设置不同提示符，避免双机时误操作：

```bash
export PS1="[autodl-A800] $PS1"
```

注意：`/root/.bashrc` 只保证终端 shell。右侧 Codex UI 是否稳定，最终看 `extensionHost` 和 `codex app-server` 的环境变量。

### 5.5 重启 VS Code Server

写完远端配置后，在 VS Code 命令面板执行：

```text
Remote-SSH: Kill VS Code Server on Host...
```

然后重新连接对应 Host alias。

不要只重开终端。终端重开不能保证 VS Code Server、extension host、Codex app-server 全部重新继承环境。

## 6. 远端代理必须先通

右侧 Codex UI、`codex app-server` 和 VS Code `extensionHost` 都运行在远端。它们访问：

```text
127.0.0.1:17997
```

访问的是远端自己的 localhost，不是 Windows 本机 localhost。

所以配置完成后，远端第一优先级验证是：

```bash
curl -I https://chatgpt.com --proxy http://127.0.0.1:17997 --max-time 20
```

成功时通常能看到：

```text
HTTP/1.1 200 Connection established
HTTP/2 403
```

其中 `HTTP/1.1 200 Connection established` 是关键，说明代理 tunnel 建起来了。后面的 `HTTP/2 403` 是 Cloudflare 页面级响应，不代表代理失败。

失败典型表现：

```text
Connection refused
Failed to connect to 127.0.0.1 port 17997
```

这说明远端没有监听 `17997`，优先检查 Windows SSH config 的 `RemoteForward`、本地代理端口、VS Code 连接的 Host alias、以及是否 Kill VS Code Server 后重连。

## 7. 一键检查脚本

如果远端已有：

```text
/root/check-vsc-codex.sh
```

直接运行：

```bash
bash /root/check-vsc-codex.sh
```

如果没有，可以创建下面这个检查脚本：

```bash
#!/usr/bin/env bash
set -u
if [[ -z "${CODEX_PROXY:-}" ]]; then
  export CODEX_PROXY="http://127.0.0.1:17997"
fi
PROXY_PORT="${CODEX_PROXY##*:}"
export PROXY_PORT

echo "== host =="
hostname
pwd

echo "== remote 127.0.0.1:${PROXY_PORT} socket =="
python3 - <<'PY'
import os, socket
port = int(os.environ.get("PROXY_PORT", "17997"))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(("127.0.0.1", port))
    print(f"127.0.0.1:{port} CONNECT_OK")
except Exception as e:
    print(f"127.0.0.1:{port} CONNECT_FAIL", repr(e))
finally:
    s.close()
PY

echo "== /proc/net/tcp listen on ${PROXY_PORT} (hex) =="
PORT_HEX=$(printf '%04X' "${PROXY_PORT}")
awk -v hex=":${PORT_HEX}" 'NR==1 || tolower($2) ~ hex || tolower($3) ~ hex' /proc/net/tcp /proc/net/tcp6 2>/dev/null || true

echo "== proxy tunnel =="
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20 || true

echo "== vscode/codex processes =="
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep || true

echo "== process proxy env =="
for pid in $(ps -eo pid,cmd | awk '/extensionHost|codex app-server/ && !/awk/ {print $1}'); do
  echo "PID $pid"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Ei '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|http_proxy|https_proxy|all_proxy|NO_PROXY|no_proxy|NODE_USE_ENV_PROXY|CODEX_)=' | sort || true
done

echo "== config files =="
test -f /root/.vscode-server/server-env-setup && echo "server-env-setup: ok"
test -f /root/.codex/codex-vscode-wrapper.sh && echo "codex wrapper: ok"
test -f /root/.vscode-server/data/User/settings.json && echo "User settings: ok"
test -f /root/.vscode-server/data/Machine/settings.json && echo "Machine settings: ok"

echo "== latest Codex logs =="
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -3
```

成功标准：

```text
127.0.0.1:17997 CONNECT_OK
HTTP/1.1 200 Connection established
extensionHost 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
codex app-server 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
server-env-setup / wrapper / User settings / Machine settings 都存在
```

`ss -tlnp | grep 17997` 没输出不一定说明失败。报告 52 里，1 服务器曾出现 `ss` 没输出，但 socket、curl、`/proc/net/tcp` 都证明端口可用。因此优先相信：

```bash
python socket 直连
/proc/net/tcp
curl --proxy
```

## 8. 手工验证命令

### 8.1 Windows 验证 SSH config

```powershell
ssh -G <HOST_ALIAS> | findstr /i "hostname port user identityfile identitiesonly remoteforward exitonforwardfailure"
```

重点看：

```text
identityfile ...id_ed25519
identitiesonly yes
remoteforward 17997 127.0.0.1:<WINDOWS_PROXY_PORT>
exitonforwardfailure yes
```

### 8.2 Windows A/B 测试 RemoteForward

先关闭 VS Code 对该服务器的连接，然后 Windows PowerShell：

```powershell
ssh -N -R 17997:127.0.0.1:7897 <HOST_ALIAS>
```

保持这个窗口不关。另开 PowerShell：

```powershell
ssh <HOST_ALIAS> "curl -I https://chatgpt.com --proxy http://127.0.0.1:17997 --max-time 20"
```

如果出现：

```text
HTTP/1.1 200 Connection established
```

说明方向确认无误，应把 SSH config 固定为：

```sshconfig
RemoteForward 17997 127.0.0.1:7897
ExitOnForwardFailure yes
```

### 8.3 远端检查 17997 监听

```bash
python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(("127.0.0.1", 17997))
    print("127.0.0.1:17997 CONNECT_OK")
except Exception as e:
    print("127.0.0.1:17997 CONNECT_FAIL", repr(e))
finally:
    s.close()
PY
```

`17997` 的十六进制端口是 `464D`：

```bash
awk 'NR==1 || tolower($2) ~ /:464d/ || tolower($3) ~ /:464d/' /proc/net/tcp /proc/net/tcp6 2>/dev/null
```

成功时类似：

```text
0100007F:464D 00000000:0000 0A
```

解释：

| 字段 | 含义 |
| --- | --- |
| `0100007F` | `127.0.0.1` |
| `464D` | `17997` |
| `0A` | TCP `LISTEN` |

报告 52 中，1 服务器还反查到监听进程是：

```text
sshd: root
```

这正符合 SSH `RemoteForward`：远端 sshd 持有监听端口，并把流量转回 Windows 本地代理。

### 8.4 远端检查后台进程代理环境

```bash
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
```

找到 PID 后：

```bash
tr '\0' '\n' < /proc/<PID>/environ | grep -Ei 'proxy|NODE_USE_ENV_PROXY|CODEX'
```

最低成功标准：

```text
extensionHost 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
codex app-server 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
```

如果终端有代理、但这两个后台进程没有代理，右侧 Codex UI 仍可能登录失败或 `reconnecting`。

### 8.5 远端查 Codex 日志

查最新日志：

```bash
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head
```

按关键字查：

```bash
grep -RInEi 'auth|login|sign|signin|device|code|browser|openExternal|external|uri|callback|oauth|token|wham|account|accounts|reconnect|fetch failed|Connection refused|401|unauthorized|invalidated' \
  /root/.vscode-server/data/logs/*/exthost*/openai.chatgpt/Codex.log 2>/dev/null | tail -n 200
```

查 extension host：

```bash
grep -RInEi 'openai.chatgpt|auth|login|sign|device|browser|openExternal|external|uri|callback|oauth|token|Authorize|PendingMigrationError|navigator' \
  /root/.vscode-server/data/logs/*/exthost*/remoteexthost.log 2>/dev/null | tail -n 200
```

日志判断：

| 日志关键词 | 更可能的问题 |
| --- | --- |
| `Connection refused` / `fetch failed` / `http/request failed` | 远端代理端口不通，或后台进程没有代理。 |
| `failed to request device code` | 设备码请求还没到本地浏览器，多半是远端网络失败。 |
| `/wham/accounts/check` | Codex 后端账号检查网络失败。 |
| `401` / `unauthorized` / `token_invalidated` | 登录态无效，需要重新登录。 |
| `device` / `oauth` / `callback` 后没有浏览器动作 | 可能是本地 VS Code / Windows 浏览器打开链路。 |
| `PendingMigrationError: navigator is now a global in nodejs` | 插件版本和 VS Code Server 兼容性可疑。 |
| 没有任何 Codex 激活日志 | 插件没有启动，先检查远程扩展安装。 |

当前旧机日志里出现过但通常可忽略的噪声：

```text
Received broadcast but no handler is configured
ignoring interface.defaultPrompt[0]
ignoring interface.icon_small
ignoring interface.icon_large
Codex could not find bubblewrap on PATH
```

只要右侧 UI 能正常对话，这些不是优先处理对象。

## 9. Codex 设备码登录页不弹出

通过 VS Code Remote-SSH 使用右侧 Codex 插件时，设备码网页应该在 Windows 本地浏览器打开，而不是在远端 Linux 服务器里打开。

链路是：

```text
Windows 本地 VS Code 窗口
    ↕ Remote-SSH
远端 VS Code Server / extensionHost / codex app-server
    ↕ RemoteForward 17997 -> Windows 7897
OpenAI 登录服务
```

如果登录网页没弹，先分层：

1. 远端是否能通过 `127.0.0.1:17997` 请求 `auth.openai.com` / `chatgpt.com`。
2. `extensionHost` 和 `codex app-server` 是否带代理环境。
3. Codex 日志是否生成了 device code / oauth / callback。
4. Windows 本地 VS Code 是否成功调用默认浏览器打开 URL。
5. 插件版本是否和当前 VS Code Server 有兼容性问题。

操作顺序：

1. 先跑 `/root/check-vsc-codex.sh`。
2. 如果 `CONNECT_FAIL` 或 `Connection refused`，先修 `RemoteForward`。
3. 如果远端代理通，但后台进程没有代理，检查 `server-env-setup`、远端 settings、wrapper，然后 Kill VS Code Server。
4. 如果后台进程有代理但日志是 `401` / `token_invalidated`，重新登录 Codex。
5. 如果日志显示已进入 device / oauth 流程但浏览器不弹：
   - 看 VS Code 右下角通知中心；
   - 看 Codex 侧边栏是否有 URL / device code / Open in browser / Copy code；
   - 手动复制链接到 Windows 浏览器；
   - 确认 Windows 默认浏览器可用；
   - 尝试 `Developer: Reload Window`、`Remote-SSH: Kill VS Code Server on Host...` 后重连。
6. 如果出现 `PendingMigrationError: navigator is now a global in nodejs` 且 UI 异常，可考虑重装或回退 OpenAI ChatGPT/Codex 扩展。不要手改扩展的 minified JS。

A800 的最终故障不是浏览器打开链路，而是远端设备码请求失败：

```text
failed to request device code: error sending request for url
https://auth.openai.com/api/accounts/deviceauth/usercode
TypeError: fetch failed
```

根因是远端 `127.0.0.1:17997` 没监听，因为 Windows SSH config 写成了 `LocalForward`。

## 10. 常见故障决策表

| 现象 | 优先判断 | 处理 |
| --- | --- | --- |
| VS Code 反复要 root 密码 | SSH key 没配好，或 VS Code 选错 Host | 先用 `ssh -o BatchMode=yes <HOST_ALIAS>` 验证免密。 |
| Windows 命令行免密成功，VS Code 仍问密码 | VS Code 没用同一 SSH config / Host alias | 检查 Remote-SSH Config File、选择正确 alias、Kill VS Code Server。 |
| 远端 `curl --proxy 127.0.0.1:17997` connection refused | 远端没有代理入口 | 检查 `RemoteForward`、本地代理端口、Host alias、`ExitOnForwardFailure`。 |
| 远端 socket OK，但 curl 不通 | 本地代理没开或代理端口不对 | Windows 确认 Clash 等代理监听端口。 |
| 终端 curl 通，但右侧 Codex reconnect | 后台进程没继承代理 | 查 `/proc/<PID>/environ`，修 `server-env-setup` / settings / wrapper 后 Kill Server。 |
| 后台进程有代理，但日志 `401` | 登录态问题 | 新服务器重新登录 Codex，不复制 `auth.json`。 |
| 设备码登录网页不弹 | 先分清远端请求失败还是本地浏览器没打开 | 查 Codex.log 和 VS Code 通知中心。 |
| `ss` 看不到 17997 | 不一定失败 | 用 socket、`/proc/net/tcp`、`curl --proxy` 交叉验证。 |
| 同时开两台服务器时只有一台失败 | 单机配置问题 | 不要改稳定机器，先确认当前 hostname 和 Host alias。 |

## 11. 不建议做的事

不要做这些：

- 不要把 Windows 私钥 `id_ed25519` 复制到服务器。
- 不要把密码写进报告、脚本或 git。
- 不要把 1 服务器和新服务器都叫 `autodl`。
- 不要把 `LocalForward` 当成远端 Codex 的代理方案。
- 不要复制旧服务器 `/root/.codex/auth.json`。
- 不要复制旧服务器整个 `/root/.vscode-server`。
- 不要在代理未通时反复重装 Codex 插件。
- 不要在 SSH 免密未通时排 Codex 登录。
- 不要优先包装 VS Code Server 的 `node`。
- 不要优先修改扩展目录里的 `codex` 二进制。
- 不要手改 OpenAI Codex 插件的 `extension.js`。
- 不要因为配置新服务器而改坏已经稳定的旧服务器。

## 12. 何时才考虑 node / codex 二进制包装

报告 49 曾记录历史上尝试过包装 VS Code Server `node` 或扩展目录里的 `codex`。但报告 52 和 54 的最终证据表明：

```text
1 服务器稳定可用时，active node 是原始 ELF。
1 服务器稳定可用时，扩展目录没有依赖 codex.real 包装。
A800 修复后，也不需要包装 node / codex 二进制。
```

新服务器只有在同时满足下面条件时，才考虑更侵入式方案：

1. `RemoteForward` 已确认远端 `17997` 可用。
2. `/root/.vscode-server/server-env-setup` 已写。
3. User / Machine settings 已写。
4. wrapper 已写并有执行权限。
5. VS Code Server 已 Kill 并重连。
6. `extensionHost` 或 `codex app-server` 仍然没有任何代理环境。

否则优先修普通配置，不要改 VS Code Server 或扩展包文件。

## 13. 双机或多机同时开启

建议别名：

```sshconfig
Host autodl-1
  HostName <SERVER1_HOST>
  Port <SERVER1_PORT>
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

Host autodl-2
  HostName <SERVER2_HOST>
  Port <SERVER2_PORT>
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

两台服务器都可以使用远端 `127.0.0.1:17997`，因为：

```text
1 服务器的 127.0.0.1 是 1 服务器自己。
2 服务器的 127.0.0.1 是 2 服务器自己。
```

它们不会因为端口号相同而互相抢占。但每台服务器都需要自己的 SSH 连接建立 `RemoteForward`。

双机排查时先定位当前机器：

```bash
hostname
pwd
df -h
git remote -v
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
curl -I https://chatgpt.com --proxy http://127.0.0.1:17997 --max-time 20
```

如果只有新服务器失败，旧服务器不要动。如果两台都失败，再考虑 Windows 本地代理、网络或账号侧问题。

## 14. A800 最终修复记录

A800 环境：

| 项 | 值 |
| --- | --- |
| Host alias | `autodl-A800` |
| HostName | `connect.nma1.seetacloud.com` |
| SSH Port | `48192` |
| Windows 本地代理 | `127.0.0.1:7897` |
| 远端代理入口 | `127.0.0.1:17997` |

故障时 A800 已有远端配置：

- `/root/.vscode-server/server-env-setup`
- `/root/.vscode-server/data/User/settings.json`
- `/root/.vscode-server/data/Machine/settings.json`
- `/root/.codex/codex-vscode-wrapper.sh`
- `/root/.bashrc` 代理环境
- SSH 免密

并且 `extensionHost` / `codex app-server` 进程环境里有：

```text
HTTP_PROXY=http://127.0.0.1:17997
HTTPS_PROXY=http://127.0.0.1:17997
ALL_PROXY=http://127.0.0.1:17997
```

但故障时：

```bash
curl -I https://chatgpt.com --proxy http://127.0.0.1:17997
# Connection refused
```

根因：

```text
Windows SSH config 使用了 LocalForward。
远端 127.0.0.1:17997 没有监听。
Codex 设备码请求在远端失败，网页还没机会在 Windows 浏览器弹出。
```

修复：

```sshconfig
RemoteForward 17997 127.0.0.1:7897
ExitOnForwardFailure yes
```

修复后：

```text
127.0.0.1:17997 CONNECT_OK
HTTP/1.1 200 Connection established
HTTP/2 403
```

用户确认右侧 Codex UI 已连接成功。

## 15. 配套文件索引

本攻略合并自：

```text
00_new_codes/reports/49-VSCode远程与Codex稳定连接配置迁移清单.md
00_new_codes/reports/50-2026-06-12-2.md
00_new_codes/reports/51-新服务器SSH免密登录与反复输入密码排查.md
00_new_codes/reports/52-回复新服务器AI-Codex远端17997转发方向排查.md
00_new_codes/reports/53-新服务器VSCode-Codex设备码登录页不弹出排查.md
00_new_codes/reports/54-A800-Codex连接修复-RemoteForward方向纠正.md
```

配套仓库片段：

```text
00_new_codes/repro_autodl/vsc_codex_remote_setup/windows-ssh-config.snippet
00_new_codes/repro_autodl/vsc_codex_remote_setup/windows-local-settings.json
00_new_codes/repro_autodl/vsc_codex_remote_setup/install-ssh-pubkey-windows.ps1
00_new_codes/repro_autodl/vsc_codex_remote_setup/README.md
```

远端实用脚本：

```text
/root/check-vsc-codex.sh
/root/install-ssh-pubkey.sh
```

后续新服务器如果只想看最短流程，读：

```text
第 0 节：最终结论
第 1 节：一次性总流程
第 3 节：Windows SSH config 模板
第 5 节：远端 VS Code / Codex 配置
第 7 节：一键检查脚本
第 10 节：常见故障决策表
```
