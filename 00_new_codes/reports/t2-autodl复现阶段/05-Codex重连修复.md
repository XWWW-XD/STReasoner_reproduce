# Codex 重连修复

## 1. 问题现象

VS Code 右侧 Codex UI 一直显示 `reconnecting`，但远程终端里的 `codex` CLI 可以正常使用。

这个现象说明：

- Codex 账号本身不一定坏；
- 终端里的 codex 能联网；
- 但 VS Code 右侧 UI 启动的后台 Codex 服务没有正常联网。

所以这次不是项目代码问题，也不是 STReasoner 项目环境问题，而是 VS Code 远程扩展的网络环境问题。

## 2. 根因

根因是：VS Code 右侧 Codex UI 的后台进程没有继承代理环境变量。

终端里能用，是因为终端 shell 里已经有代理：

```bash
HTTP_PROXY=http://127.0.0.1:17897
HTTPS_PROXY=http://127.0.0.1:17897
http_proxy=http://127.0.0.1:17897
https_proxy=http://127.0.0.1:17897
```

但右侧 Codex UI 不是直接运行在这个终端里。

它实际由 VS Code 远程扩展启动，主要涉及两层后台进程：

```text
VS Code extension host
→ 启动 Codex app-server
→ app-server 访问 ChatGPT / Codex 后端
```

问题就在这里：终端有代理，不代表 extension host 和 app-server 也有代理。

所以出现了这个差异：

- 终端 codex CLI：能用，因为 shell 有代理。
- 右侧 Codex UI：reconnecting，因为后台进程没走代理。

## 3. 关键证据

日志里反复出现网络错误：

```text
TypeError: fetch failed
http/request failed: error sending request for url (https://chatgpt.com/backend-api/wham/apps)
/wham/accounts/check
```

这类错误更像网络请求失败，不像模型代码报错。

同时，当时没有持续看到明确的认证错误，例如：

```text
401
token_invalidated
```

所以判断重点不是"重新登录 Codex"，而是"让 VS Code 后台进程也能走代理"。

进一步检查旧进程环境变量时发现：

- 旧 Codex app-server 没有 HTTP_PROXY / HTTPS_PROXY
- 旧 VS Code extension host 也没有 HTTP_PROXY / HTTPS_PROXY

这就解释了为什么终端能用，但右侧 UI 一直 reconnect。

## 4. 第一次修复为什么不够

第一次修复做了两件事：

1. 添加 `/root/.codex/codex-vscode-wrapper.sh`
2. 在 VS Code 远程 settings.json 里设置 `chatgpt.cliExecutable` 和 `http.proxy`

这个方向是对的：希望 Codex 扩展启动时走带代理的 wrapper。

但后续发现，实际运行中的 Codex app-server 仍然来自扩展自带的二进制：

```text
/root/.vscode-server/extensions/openai.chatgpt-.../bin/linux-x86_64/codex app-server
```

也就是说，扩展没有完全按 `chatgpt.cliExecutable` 启动后台服务，或者旧进程还在继续运行。

更重要的是：即使 app-server 后来带了代理，VS Code extension host 自己仍然可能没代理。

所以第一次修复后，右侧 UI 仍可能 reconnect。

## 5. 最终修复思路

最终不是只修 Codex CLI，而是修两层后台进程：

- 第一层：Codex app-server 必须带代理。
- 第二层：VS Code extension host 也必须带代理。

因为右侧 UI 的网络请求不一定都由 app-server 发出，有些检查请求可能由 extension host 直接发出。

## 6. 最终做了哪些修改

### 6.1 保留 VS Code 用户设置

保留远程 VS Code 设置：

```text
/root/.vscode-server/data/User/settings.json
```

其中包括：

```json
{
  "chatgpt.cliExecutable": "/root/.codex/codex-vscode-wrapper.sh",
  "http.proxy": "http://127.0.0.1:17897",
  "http.proxySupport": "override"
}
```

这部分有用，但单靠它不够。

### 6.2 包装 Codex 扩展自带的 codex 二进制

把 Codex 扩展实际调用的 `codex` 二进制替换成 wrapper。

原始二进制保留为：

- `codex.real`
- `codex.backup-20260528-115141`

新的 wrapper 会先设置代理：

```bash
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
```

然后再执行真实的 `codex.real`。

这样即使 VS Code 扩展直接调用它自带的 `codex`，也会自动带上代理。

### 6.3 添加 VS Code Server 启动环境文件

新增：

```text
/root/.vscode-server/server-env-setup
```

里面也写入代理环境变量。

作用是：让以后 VS Code Server 新启动时，更容易让服务端进程继承代理。

### 6.4 包装 VS Code Server 的 node

把当前 VS Code Server 版本目录里的 `node` 也包装了一层。

原始二进制保留为：

- `node.real`
- `node.backup-20260528-115535`

新的 wrapper 会设置：

```bash
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
export NODE_USE_ENV_PROXY="${NODE_USE_ENV_PROXY:-1}"
```

然后执行真实的 `node.real`。

这一步是关键，因为 VS Code extension host 本身就是由这个 node 启动的。

也就是说：

- 包装 codex：解决 Codex app-server 代理问题。
- 包装 node：解决 VS Code extension host 代理问题。

### 6.5 重启旧后台进程

杀掉旧的 app-server 和 extension host，让 VS Code 自动拉起新进程。

这样新的 wrapper 才会生效。

## 7. 修复后如何确认成功

修复后，新进程显示为：

```text
codex.real app-server --analytics-default-enabled
node.real ... --type=extensionHost
```

这说明 wrapper 已经生效：wrapper 设置好环境变量后，把控制权交给了真实二进制。

检查新进程环境变量后确认有：

```text
HTTP_PROXY=http://127.0.0.1:17897
HTTPS_PROXY=http://127.0.0.1:17897
http_proxy=http://127.0.0.1:17897
https_proxy=http://127.0.0.1:17897
NODE_USE_ENV_PROXY=1
```

日志里也出现了正常启动信息：

```text
Codex extension activated
spawned Codex app-server
app-server initialized
Features enabled
app routes mounted
```

同时，之前反复刷新的网络错误不再继续出现：

```text
/wham/accounts/check TypeError: fetch failed
wham/apps http/request failed
```

这说明右侧 Codex UI 已恢复连接。

## 8. 哪些日志不用紧张

修复后还有一些日志，例如：

```text
unsupported feature enablement auth_elicitation
goals feature is disabled
```

这些不是 reconnect 的根因，也不是网络失败。只要右侧 UI 能正常对话，就不用优先处理。

## 9. 下次 reconnect 的排查顺序

**第一步**，先确认终端代理是否可用：

```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
curl -I https://chatgpt.com --max-time 20
```

**第二步**，检查 Codex app-server 和 VS Code extension host 是否带代理：

```bash
ps -ef | grep -E 'codex|extensionHost' | grep -v grep
```

找到 PID 后检查：

```bash
tr '\0' '\n' < /proc/<PID>/environ | grep -i proxy
```

如果没有 `HTTP_PROXY` / `HTTPS_PROXY`，大概率还是代理继承问题。

**第三步**，看日志类型。

如果日志是：

```text
fetch failed
http/request failed
/wham/accounts/check
```

优先按网络/代理问题处理。

如果日志是：

```text
401
token_invalidated
unauthorized
```

才考虑重新登录 Codex。
