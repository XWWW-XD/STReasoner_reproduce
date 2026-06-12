# VSCode 远程与 Codex 稳定连接配置迁移清单

## 0. 结论

这台 AutoDL 主机现在能比较稳定地通过 VS Code Remote-SSH 使用右侧 Codex UI，核心不是某一个单点配置，而是两层问题都处理好了：

1. VS Code Remote-SSH 到 AutoDL 的 SSH 连接要稳定。
2. VS Code 远端后台进程要能继承代理，尤其是 `extensionHost` 和 `codex app-server`。

新主机优先照这个顺序复刻：

1. 本地 Windows SSH 配置保留 keepalive、重试、`IPQoS none`。
2. 本地 VS Code Remote-SSH 设置关闭容易出问题的 exec server / dynamic forwarding。
3. 先确认代理端口真的转发到了远端。
4. 在远端写入 `/root/.vscode-server/server-env-setup`。
5. 在远端写入 `/root/.codex/codex-vscode-wrapper.sh`。
6. 在远端 VS Code `User/settings.json` 和 `Machine/settings.json` 写入代理和 `chatgpt.cliExecutable`。
7. 重启 VS Code Server，再检查 `extensionHost` 和 `codex app-server` 的环境变量。

当前旧机稳定使用的代理端口是：

```bash
http://127.0.0.1:17997
```

但这个端口不要机械照抄到新主机。新主机必须先确认本地代理或端口转发到远端后的实际端口，然后统一替换下文的 `<PORT>`。

## 1. 报告依据

本报告核对了两类信息：

| 来源 | 作用 |
| --- | --- |
| `00_new_codes/reports/t2-autodl复现阶段/04-vsc ssh autodl连接问题排查.md` | 记录 VS Code Remote-SSH 动态端口转发失败、SSH keepalive 和 Remote-SSH 设置。 |
| `00_new_codes/reports/t2-autodl复现阶段/05-Codex重连修复.md` | 记录 Codex UI `reconnecting` 的根因、代理继承修复方案和当前稳定口径。 |
| 当前主机 `/root/.vscode-server/` | 核对现在实际生效的 VS Code server、settings、日志和进程。 |
| 当前主机 `/root/.codex/` | 核对 Codex wrapper 和 Codex 配置。 |
| 当前进程 `/proc/<PID>/environ` | 确认 `extensionHost` 和 `codex app-server` 真实继承了代理环境。 |

注意：没有读取或记录 `/root/.codex/auth.json` 这类认证文件内容。新主机不建议复制旧 token，应重新登录。

## 2. 新主机最小复刻流程

### 2.1 本地 Windows SSH config

旧报告记录的 Windows SSH 配置路径：

```text
C:\Users\HUAWEI\.ssh\config
```

新主机要把 `HostName` 和 `Port` 换成新 AutoDL 实例给出的地址和端口，其余稳定性参数可以沿用：

```sshconfig
Host autodl <NEW_AUTODL_HOST>
  HostName <NEW_AUTODL_HOST>
  Port <NEW_AUTODL_SSH_PORT>
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

这里最重要的是保留短别名：

```text
Host autodl
```

之后 VS Code Remote-SSH 里直接选 `autodl`，不要每次手动拼完整 SSH 命令。

这些参数的作用：

| 参数 | 作用 |
| --- | --- |
| `ServerAliveInterval 20` | SSH 每 20 秒保活一次，减少空闲断开。 |
| `ServerAliveCountMax 6` | 连续多次保活失败才断，容忍短暂网络抖动。 |
| `TCPKeepAlive yes` | 开启 TCP keepalive。 |
| `ConnectTimeout 15` | 连接慢时不要太快放弃。 |
| `ConnectionAttempts 3` | 自动多试几次。 |
| `IPQoS none` | 避免部分网络环境中 SSH 卡顿或丢包。 |

### 2.2 本地 VS Code Remote-SSH 设置

旧报告记录过这组设置对 AutoDL 比较稳：

```json
{
  "remote.SSH.connectTimeout": 60,
  "remote.SSH.useExecServer": false,
  "remote.SSH.enableDynamicForwarding": false,
  "remote.SSH.showLoginTerminal": true
}
```

含义：

| 设置 | 作用 |
| --- | --- |
| `remote.SSH.connectTimeout=60` | 给 AutoDL 慢连接更长等待时间。 |
| `remote.SSH.useExecServer=false` | 关闭 Remote-SSH 较新的 exec server 模式，回退更传统链路。 |
| `remote.SSH.enableDynamicForwarding=false` | 绕开旧机曾报错的 dynamic port forwarding 流程。 |
| `remote.SSH.showLoginTerminal=true` | 连接卡住时能看到登录终端输出。 |

如果新 VS Code 版本行为不同，可以在稳定后再尝试恢复默认；但迁移时先按这组保守设置做。

### 2.3 先确认远端代理端口可用

右侧 Codex UI、`codex app-server` 和 VS Code `extensionHost` 都运行在远端。它们访问 `127.0.0.1:<PORT>` 时，访问的是远端自己的 localhost，不是 Windows 本机 localhost。

所以新主机第一步不是写配置，而是确认代理端口已经转发到远端：

```bash
export CODEX_PROXY=http://127.0.0.1:<PORT>
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20
```

当前旧机用 `17997` 时，返回里能看到：

```text
HTTP/1.1 200 Connection established
HTTP/2 403
```

这里的 `403` 是 Cloudflare 页面级响应，不代表代理端口失败。真正要避免的是：

```text
Connection refused
Failed to connect to 127.0.0.1 port <PORT>
```

如果这里连不上，后面配置写得再漂亮，右侧 Codex UI 也还是容易 `reconnecting`。

### 2.4 写 VS Code Server 启动环境

新主机远端创建：

```text
/root/.vscode-server/server-env-setup
```

内容按实际端口替换 `<PORT>`：

```bash
export HTTP_PROXY=http://127.0.0.1:<PORT>
export HTTPS_PROXY=http://127.0.0.1:<PORT>
export http_proxy=http://127.0.0.1:<PORT>
export https_proxy=http://127.0.0.1:<PORT>
export ALL_PROXY=http://127.0.0.1:<PORT>
export all_proxy=http://127.0.0.1:<PORT>
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export NODE_USE_ENV_PROXY=1
```

这个文件的作用是让 VS Code Server 启动出的后台进程更容易继承代理。写完之后必须重启 VS Code Server，否则旧进程不会自动带上新环境。

### 2.5 写 Codex wrapper

新主机远端创建：

```text
/root/.codex/codex-vscode-wrapper.sh
```

内容按实际端口替换 `<PORT>`：

```bash
#!/usr/bin/env bash
set -euo pipefail

CODEX_PROXY="${CODEX_PROXY:-http://127.0.0.1:<PORT>}"

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

赋予执行权限：

```bash
chmod 755 /root/.codex/codex-vscode-wrapper.sh
```

### 2.6 写 VS Code 远端 settings

当前旧机同时写了两个位置：

```text
/root/.vscode-server/data/User/settings.json
/root/.vscode-server/data/Machine/settings.json
```

新主机也建议两处保持一致。按实际端口替换 `<PORT>`：

```json
{
  "chatgpt.cliExecutable": "/root/.codex/codex-vscode-wrapper.sh",
  "http.proxy": "http://127.0.0.1:<PORT>",
  "http.proxySupport": "override",
  "http.proxyStrictSSL": false
}
```

这一步对右侧 UI 很关键：它让 VS Code 和 ChatGPT/Codex 扩展知道远端要走哪个代理，并指定 Codex CLI wrapper。

### 2.7 写 shell 代理

为了让远程终端里的 `curl`、`git`、`codex` 等命令也走同一代理，可以在 `/root/.bashrc` 追加：

```bash
# proxy for Codex / VS Code Remote
export HTTP_PROXY=http://127.0.0.1:<PORT>
export HTTPS_PROXY=http://127.0.0.1:<PORT>
export http_proxy=http://127.0.0.1:<PORT>
export https_proxy=http://127.0.0.1:<PORT>
export ALL_PROXY=http://127.0.0.1:<PORT>
export all_proxy=http://127.0.0.1:<PORT>
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

这个只保证终端 shell。不要误以为终端能用，右侧 Codex UI 就一定能用；右侧 UI 还要看 `extensionHost` 和 `codex app-server` 是否继承代理。

### 2.8 重启 VS Code Server

写完上面配置后，需要重启远端 VS Code Server。推荐在 VS Code 命令面板执行：

```text
Remote-SSH: Kill VS Code Server on Host...
```

然后重新连接 `autodl`。

不推荐一上来手动乱杀所有 node；先用 VS Code 自带命令最干净。

## 3. 当前旧机真实快照

### 3.1 VS Code 远端 settings

当前旧机 `/root/.vscode-server/data/User/settings.json`：

```json
{
  "chatgpt.cliExecutable": "/root/.codex/codex-vscode-wrapper.sh",
  "http.proxy": "http://127.0.0.1:17997",
  "http.proxySupport": "override",
  "http.proxyStrictSSL": false
}
```

当前旧机 `/root/.vscode-server/data/Machine/settings.json` 内容相同。

### 3.2 VS Code Server 环境文件

当前旧机 `/root/.vscode-server/server-env-setup`：

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

### 3.3 Codex wrapper

当前旧机 `/root/.codex/codex-vscode-wrapper.sh`：

- 权限：`755`
- 默认端口：`http://127.0.0.1:17997`
- 作用：设置代理环境变量后，再查找并执行 VS Code ChatGPT/Codex 扩展自带的 `codex` 二进制。
- 备份：存在 `/root/.codex/codex-vscode-wrapper.sh.backup-20260531-2320`。

### 3.4 shell 代理

当前旧机 `/root/.bashrc` 第 110-118 行写了：

```bash
export HTTP_PROXY=http://127.0.0.1:17997
export HTTPS_PROXY=http://127.0.0.1:17997
export http_proxy=http://127.0.0.1:17997
export https_proxy=http://127.0.0.1:17997
export ALL_PROXY=http://127.0.0.1:17997
export all_proxy=http://127.0.0.1:17997
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

### 3.5 当前运行中的 VS Code / Codex 进程

当前运行中的 VS Code Server 版本目录：

```text
/root/.vscode-server/bin/3c631b164c239e7aeaaae7c626b46c527b361af2
```

关键进程：

```text
node ... --type=extensionHost --transformURIs --useHostProxy=false
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036/bin/linux-x86_64/codex app-server --analytics-default-enabled
```

当前机器上有两个 OpenAI ChatGPT/Codex 扩展目录：

```text
openai.chatgpt-26.602.71036
openai.chatgpt-26.609.30741
```

当前实际运行的 app-server 来自 `26.602.71036`。

### 3.6 关键成功证据

当前检查 `/proc/<PID>/environ`，`extensionHost` 和 `codex app-server` 都有：

```text
HTTP_PROXY=http://127.0.0.1:17997
HTTPS_PROXY=http://127.0.0.1:17997
ALL_PROXY=http://127.0.0.1:17997
http_proxy=http://127.0.0.1:17997
https_proxy=http://127.0.0.1:17997
all_proxy=http://127.0.0.1:17997
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

这比“终端里能 curl”更重要。右侧 UI 稳不稳，最终看这两个后台进程。

### 3.7 当前 active node 没有被包装

当前 active VS Code Server 的 node：

```text
/root/.vscode-server/bin/3c631b164c239e7aeaaae7c626b46c527b361af2/node
```

检查结果是 ELF 二进制，不是 bash wrapper。

另外：

```text
/root/.vscode-server/bin/8761a5560cfd65fdd19ce7e2bd18dab5c0a4d84e/node
```

是旧历史修复遗留的 shell wrapper，但它不是当前 active server。因此新主机不要优先复制“包装 node”这种侵入式操作。

### 3.8 当前远端 SSH server

当前 `/etc/ssh/sshd_config` 里 relevant 项：

```text
UsePAM yes
#AllowTcpForwarding yes
#GatewayPorts no
X11Forwarding yes
#ClientAliveInterval 0
#ClientAliveCountMax 3
```

`AllowTcpForwarding` 虽然是注释，但 OpenSSH 默认是允许。旧报告也曾验证过普通 SSH 和动态转发本身可用。

`/root/.ssh/authorized_keys` 当前存在 1 行公钥。新机照常放自己的公钥，不要复制不理解的密钥材料。

## 4. 历史上做过但新机不优先照抄的动作

### 4.1 包装扩展目录里的 codex 二进制

历史修复曾考虑或做过把扩展自带的：

```text
/root/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex
```

替换成 wrapper，并把原始文件保存成 `codex.real` 或 backup。

当前旧机检查结果：两个扩展目录里现在只有 `codex`，没有 `codex.real`。也就是说，当前稳定状态不依赖“直接修改扩展二进制”。

新机优先不要动扩展目录。先用：

```text
/root/.codex/codex-vscode-wrapper.sh
chatgpt.cliExecutable
server-env-setup
```

如果检查发现 `codex app-server` 仍没有代理，再考虑针对当前扩展版本做包装。

### 4.2 包装 VS Code Server 的 node

历史修复曾包装某个 VS Code Server 版本目录的 `node`。这可以强制 `extensionHost` 继承代理，但缺点也明显：

- 路径和 VS Code Server 版本强绑定。
- VS Code 更新后会失效。
- 改错可能导致 VS Code Server 起不来。

当前 active server 的 `node` 是原始 ELF，说明现在不需要这一步。

新机只有在确认：

```text
server-env-setup 已写；
VS Code Server 已重启；
extensionHost 仍没有 HTTP_PROXY / HTTPS_PROXY；
```

才考虑包装当前 active `node`。

### 4.3 复制旧机 Codex auth

不要复制：

```text
/root/.codex/auth.json
```

新机配置好代理后重新登录 Codex。这样更干净，也避免 token 失效或泄露。

## 5. 新机验证清单

### 5.1 验证 SSH 基础连接

在 Windows PowerShell：

```powershell
ssh -o BatchMode=yes autodl "echo SSH_OK; uname -a; command -v bash; command -v tar; command -v wget; command -v curl; df -h ~"
```

应看到：

```text
SSH_OK
bash / tar / wget / curl 存在
磁盘空间正常
```

### 5.2 验证远端代理端口

在远端终端：

```bash
export CODEX_PROXY=http://127.0.0.1:<PORT>
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20
```

能建立 tunnel 才继续。

### 5.3 验证后台进程继承代理

重新连接 VS Code 后，在远端执行：

```bash
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
```

找到 PID 后检查：

```bash
tr '\0' '\n' < /proc/<PID>/environ | grep -Ei 'proxy|NODE_USE_ENV_PROXY|CODEX'
```

最低成功标准：

```text
extensionHost 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
codex app-server 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
```

如果终端有代理、但这两个进程没有代理，右侧 UI 仍可能 `reconnecting`。

### 5.4 验证 Codex 日志

当前旧机日志位置类似：

```text
/root/.vscode-server/data/logs/<timestamp>/exthost*/openai.chatgpt/Codex.log
```

查最新日志：

```bash
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' | sort -nr | head
```

重点看错误类型。

网络或代理问题通常是：

```text
fetch failed
http/request failed
/wham/accounts/check
Connection refused
Failed to connect to 127.0.0.1 port <PORT>
```

认证问题通常是：

```text
401
token_invalidated
unauthorized
```

这两类不要混着处理。前者先修代理继承，后者才重新登录。

## 6. 当前日志里可以忽略的噪声

当前旧机 Codex 日志里还有一些 warning，例如：

```text
Received broadcast but no handler is configured
ignoring interface.defaultPrompt[0]
ignoring interface.icon_small
ignoring interface.icon_large
```

这些不是本次 `reconnecting` 的根因。只要右侧 UI 能正常对话，就不用优先处理。

如果看到：

```text
Codex could not find bubblewrap on PATH
```

也不一定是致命错误；Codex 可以使用 bundled bubblewrap。不要把它误判成右侧 UI 断连主因。

## 7. 常见误区

### 7.1 终端 Codex 能用，不等于右侧 UI 能用

终端由 shell 启动，读 `/root/.bashrc`。右侧 UI 由 VS Code 远端扩展启动，关键是 `extensionHost` 和 `codex app-server` 的环境变量。

### 7.2 不要沿用旧端口 `17897`

旧报告曾出现过 `17897`，但当前旧机稳定端口是 `17997`。新机仍要以实际转发端口为准。

### 7.3 不要先改 VS Code Server node

当前旧机已经证明，active node 不包装也能稳定。新机先做非侵入式配置。

### 7.4 不要把 auth token 当成配置迁移

Codex 登录态不是普通配置。新主机配置好网络后重新登录，不复制旧 token。

### 7.5 AutoDL 实例的 SSH 端口会变

每次新实例都要按 AutoDL 页面给出的 HostName / Port 更新本地 SSH config。

## 8. 一键检查命令

新机配置完后，可以直接跑这段检查：

```bash
PORT=<PORT>
export CODEX_PROXY=http://127.0.0.1:${PORT}

echo "== proxy tunnel =="
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20

echo "== vscode/codex processes =="
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep

echo "== process proxy env =="
for pid in $(ps -eo pid,cmd | awk '/extensionHost|codex app-server/ && !/awk/ {print $1}'); do
  echo "PID $pid"
  tr '\0' '\n' < "/proc/$pid/environ" | grep -Ei '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|http_proxy|https_proxy|all_proxy|NO_PROXY|no_proxy|NODE_USE_ENV_PROXY|CODEX_)=' | sort
done

echo "== latest Codex logs =="
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head
```

期望结果：

- `curl` 能建立代理 tunnel。
- 能看到 `extensionHost` 和 `codex app-server`。
- 两个进程环境里都有大小写 proxy。
- 最新日志没有持续刷新 `fetch failed` / `Connection refused`。

## 9. 如果新机仍然 reconnect

按下面顺序处理：

1. 先确认 `curl --proxy` 能不能通过远端 `127.0.0.1:<PORT>` 出去。
2. 再确认 `/root/.vscode-server/server-env-setup` 是否存在且端口正确。
3. 再确认 `User/settings.json` 和 `Machine/settings.json` 是否都写了 `http.proxy` 和 `chatgpt.cliExecutable`。
4. 重启 VS Code Server，不要只重开终端。
5. 查 `extensionHost` 和 `codex app-server` 的 `/proc/<PID>/environ`。
6. 如果后台进程没代理，才考虑包装当前 active VS Code Server `node` 或当前扩展目录的 `codex`。
7. 如果后台进程有代理但日志是 `401` / `token_invalidated`，重新登录 Codex。

## 10. 当前机器对新机最有价值的经验

这次真正让连接变稳定的经验是：

- SSH 层面，用 keepalive、重试、`IPQoS none` 降低 AutoDL 网络抖动影响。
- VS Code Remote-SSH 层面，关闭旧机容易触发问题的 exec server / dynamic forwarding。
- Codex UI 层面，不只配置终端代理，而是让 VS Code `extensionHost` 和 `codex app-server` 都继承代理。
- 当前稳定方案优先靠普通配置文件，不靠修改 VS Code Server 或扩展二进制。
- 新机不要复制 token，不要硬套旧端口，先验证代理端口和后台进程环境。

## 11. 1 服务器和 2 服务器可能同时开启时的配置建议

当前这台已经稳定连接的机器可以称为 `1服务器`。后续新开的机器可以称为 `2服务器`。如果两台 AutoDL 服务器以后可能同时开，不建议都继续叫 `autodl`，否则 VS Code、SSH config、终端记录和报告里很容易混淆。

### 11.1 本地 SSH alias 要分开

Windows 本地 `C:\Users\HUAWEI\.ssh\config` 建议改成两个明确别名：

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
```

建议以后 VS Code Remote-SSH 里分别连接：

```text
autodl-1
autodl-2
```

不要频繁把同一个 `Host autodl` 改来改去指向不同机器。这样做短期能连，但长期会导致：

- 不知道当前 VS Code 窗口连的是哪台机器。
- `Kill VS Code Server on Host` 时容易杀错目标。
- 报告、日志、模型路径和实验路径混在一起。
- SSH known_hosts 或端口变化排查更费劲。

### 11.2 两台服务器的远端配置各自独立

`/root/.vscode-server/`、`/root/.codex/`、`/root/.bashrc` 都在远端机器自己的磁盘上。也就是说：

```text
1服务器的 /root/.vscode-server 只影响 1服务器
2服务器的 /root/.vscode-server 只影响 2服务器
```

不要以为 1 服务器配好了，2 服务器会自动继承。2 服务器仍然要单独写：

- `/root/.vscode-server/server-env-setup`
- `/root/.codex/codex-vscode-wrapper.sh`
- `/root/.vscode-server/data/User/settings.json`
- `/root/.vscode-server/data/Machine/settings.json`
- `/root/.bashrc` 中的 proxy 环境变量

### 11.3 代理端口可以相同，但必须逐台验证

如果两台服务器都通过 SSH remote forward 把本地代理转发到远端，那么两台机器上都写：

```bash
http://127.0.0.1:17997
```

理论上可以同时成立，因为：

```text
1服务器的 127.0.0.1 是 1服务器自己
2服务器的 127.0.0.1 是 2服务器自己
```

它们不是同一个 localhost，所以不会因为都叫 `127.0.0.1:17997` 就互相抢端口。

但前提是：两台服务器各自的 SSH 连接都真的把代理转发到了远端同一个端口。每台机器都要分别验证：

```bash
export CODEX_PROXY=http://127.0.0.1:17997
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20
```

如果 2 服务器上这个命令 `Connection refused`，说明 2 服务器没有拿到这个远端端口。此时不要改 1 服务器；只检查 2 服务器的 SSH 转发、VS Code 连接和远端配置。

### 11.4 如果本地代理端口不同，要逐台统一替换

如果 2 服务器实际可用端口不是 `17997`，例如变成 `<SERVER2_PROXY_PORT>`，则只在 2 服务器里统一替换：

```text
/root/.vscode-server/server-env-setup
/root/.codex/codex-vscode-wrapper.sh
/root/.vscode-server/data/User/settings.json
/root/.vscode-server/data/Machine/settings.json
/root/.bashrc
```

1 服务器已经稳定，不要因为配置 2 服务器而改动 1 服务器的端口。

### 11.5 VS Code 窗口和终端提示要刻意区分

同时开两台机器时，建议：

- VS Code 窗口标题、工作区目录、终端里都确认当前在哪台服务器。
- 报告中写清楚 `1服务器` 或 `2服务器`。
- 运行模型下载、删除模型、清理缓存、推送 git 前，先执行：

```bash
hostname
pwd
df -h
git remote -v
```

如果要更醒目，可以在两台机器的 `/root/.bashrc` 中分别设置提示符，例如：

```bash
export PS1="[autodl-1] $PS1"
```

或：

```bash
export PS1="[autodl-2] $PS1"
```

这不是必须配置，但双机同时开时很有用。

### 11.6 Codex 登录态不要跨机器复制

两台服务器的 Codex 登录态应各自维护。不要把 1 服务器的：

```text
/root/.codex/auth.json
```

复制到 2 服务器。2 服务器配置好代理后重新登录 Codex。

### 11.7 双机排查时先定位是哪台机器的问题

以后如果出现 reconnect，不要直接改全局配置，先判断：

```text
是 autodl-1 reconnect？
还是 autodl-2 reconnect？
还是本地网络/代理本身挂了？
```

最小判断方法：

```bash
hostname
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
curl -I https://chatgpt.com --proxy http://127.0.0.1:<PORT> --max-time 20
```

如果只有 2 服务器失败，1 服务器不要动；如果两台都失败，再考虑本地代理、网络或账号侧问题。
