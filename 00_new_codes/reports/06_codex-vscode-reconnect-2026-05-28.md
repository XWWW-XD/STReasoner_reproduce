# Codex VS Code 右侧 UI Reconnecting 问题修复报告

Date: 2026-05-28

## 结论先说

这次右侧 Codex UI 一直 `reconnecting`，最终确认不是项目代码问题，也不是当前 `~/.codex/auth.json` 丢失或登录失效导致的。根因是：

**远程 VS Code 里的 Codex 扩展进程没有继承代理环境变量，导致它访问 ChatGPT 后端接口时直接连外网，连接失败。**

终端里的 `codex` CLI 能用，是因为终端 shell 里有代理：

- `HTTP_PROXY=http://127.0.0.1:17897`
- `HTTPS_PROXY=http://127.0.0.1:17897`
- `http_proxy=http://127.0.0.1:17897`
- `https_proxy=http://127.0.0.1:17897`

但是右侧 VS Code UI 不是直接复用这个终端进程。它由 VS Code extension host 启动自己的 Codex `app-server`，这些后台进程一开始没有代理变量，所以 UI 会反复重连。

现在已经成功恢复：你已经能在 VS Code 右侧 UI 里继续对话。

## 用浅一点的话解释

可以把这件事理解成三层：

1. 终端里的 `codex`

   这是你手动在 shell 里运行的 CLI。它继承了终端里的代理变量，所以能联网。

2. VS Code 右侧 Codex UI

   这是 VS Code 扩展的 WebView 界面。它自己需要访问 ChatGPT 的账号、任务、功能开关等接口。

3. Codex app-server

   这是 VS Code 扩展背后拉起来的本地服务。右侧 UI 会通过它和 Codex 后端交互。

问题就在于：第 1 层有代理，第 2 和第 3 层一开始没有代理。于是终端能用，右侧 UI 却一直 `reconnecting`。

## 一开始为什么没完全修好

第一次修复时做了两件事：

- 添加 `/root/.codex/codex-vscode-wrapper.sh`
- 在 `/root/.vscode-server/data/User/settings.json` 里设置：
  - `chatgpt.cliExecutable`
  - `http.proxy`
  - `http.proxySupport`

这个思路是对的：希望 VS Code Codex 扩展启动 Codex 时走一个带代理的 wrapper。

但后续检查发现，实际运行中的 app-server 仍然是扩展自带的二进制：

```text
/root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/bin/linux-x86_64/codex app-server --analytics-default-enabled
```

也就是说，扩展没有完全按我们设置的 `chatgpt.cliExecutable` 来启动后台服务，或者已有旧进程还在继续工作。

同时，VS Code extension host 自己也没有代理变量。即使 app-server 后来加了代理，extension host 自己发起的 `/wham/accounts/check` 之类请求仍然可能失败。

所以第一次修复后，右侧 UI 仍然会 reconnect。

## 关键证据

日志里反复出现的是网络层错误，例如：

```text
TypeError: fetch failed
http/request failed: error sending request for url (https://chatgpt.com/backend-api/wham/apps)
/wham/accounts/check
```

旧 app-server 进程的环境变量里没有：

```text
HTTP_PROXY
HTTPS_PROXY
http_proxy
https_proxy
```

旧 extension host 进程也没有这些代理变量。

另一方面，当时没有继续看到明确的：

```text
401
token_invalidated
```

因此这次主要问题不是“登录 token 已失效”，而是“VS Code 后台进程没走代理”。

## 最终做了什么修改

没有修改项目代码、数据集、实验输出，也没有打印或改动 `~/.codex/auth.json` 的 token 内容。

### 1. 保留第一次添加的 VS Code 用户设置

文件：

```text
/root/.vscode-server/data/User/settings.json
```

内容包含：

```json
{
  "chatgpt.cliExecutable": "/root/.codex/codex-vscode-wrapper.sh",
  "http.proxy": "http://127.0.0.1:17897",
  "http.proxySupport": "override"
}
```

这部分仍然有价值，但单靠它不够。

### 2. 包装 VS Code Codex 扩展实际调用的 Codex 二进制

被替换为 wrapper 的路径：

```text
/root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/bin/linux-x86_64/codex
```

原始二进制保留为：

```text
/root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/bin/linux-x86_64/codex.real
/root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/bin/linux-x86_64/codex.backup-20260528-115141
```

新的 wrapper 会先设置代理：

```bash
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
```

然后执行：

```text
codex.real
```

这样可以保证 VS Code 扩展即使直接调用它自带的 `codex`，也会带上代理。

### 3. 添加 VS Code Server 启动环境文件

新增：

```text
/root/.vscode-server/server-env-setup
```

里面设置同样的代理变量。它的作用是让以后 VS Code Server 新启动时，服务端进程更容易继承代理。

### 4. 包装 VS Code Server 的 node

被替换为 wrapper 的路径：

```text
/root/.vscode-server/bin/f6cfa2ea2403534de03f069bdf160d06451ed282/node
```

原始二进制保留为：

```text
/root/.vscode-server/bin/f6cfa2ea2403534de03f069bdf160d06451ed282/node.real
/root/.vscode-server/bin/f6cfa2ea2403534de03f069bdf160d06451ed282/node.backup-20260528-115535
```

新的 wrapper 会设置：

```bash
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
export NODE_USE_ENV_PROXY="${NODE_USE_ENV_PROXY:-1}"
```

然后执行：

```text
node.real
```

这一步是关键补丁：它让新启动的 VS Code extension host 自己也带上代理，而不只是 Codex app-server 带代理。

### 5. 重启相关后台进程

先停止旧的 Codex app-server：

```bash
kill 8498
```

后来为了让新的 node wrapper 生效，停止了当时的 extension host 和 app-server：

```bash
kill 27071 27030
```

VS Code 随后自动拉起了新的 extension host 和新的 Codex app-server。

## 修复后的验证结果

新的 app-server 进程变成：

```text
/root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/bin/linux-x86_64/codex.real app-server --analytics-default-enabled
```

新的 extension host 进程变成：

```text
/root/.vscode-server/bin/.../node.real --dns-result-order=ipv4first ... --type=extensionHost ...
```

这说明 wrapper 已经生效：进程名显示为 `codex.real` 和 `node.real`，因为 wrapper 设置完环境变量后把控制权交给了真实二进制。

检查 `/proc/<pid>/environ` 后确认，新 extension host 和新 app-server 都有：

```text
HTTP_PROXY=http://127.0.0.1:17897
HTTPS_PROXY=http://127.0.0.1:17897
http_proxy=http://127.0.0.1:17897
https_proxy=http://127.0.0.1:17897
NODE_USE_ENV_PROXY=1
```

修复后日志出现：

```text
11:55:23 Codex extension activated
11:55:23 spawned Codex app-server
11:55:25 app-server initialized
11:55:32 Features enabled
11:55:32 app routes mounted
```

在最终重启之后，之前导致 reconnect 的这类错误没有继续刷：

```text
/wham/accounts/check TypeError: fetch failed
wham/apps http/request failed
```

后面还有一些不影响主功能的日志，例如：

```text
unsupported feature enablement auth_elicitation
goals feature is disabled
```

这些不是连接失败，不属于这次 reconnect 的根因。

## 以后还需要再修吗

如果还是这台服务器，并且代理端口仍然是：

```text
127.0.0.1:17897
```

通常不需要再修。现在的 wrapper 会让当前 VS Code Server 版本和当前 Codex 扩展版本继续带代理启动。

但下面几种情况可能需要再检查一次：

1. AutoDL 或代理端口变了

   例如代理不再监听 `127.0.0.1:17897`，那就需要把 wrapper 里的端口改成新的。

2. VS Code Server 更新了

   当前 node wrapper 位于这个 VS Code Server 版本目录：

   ```text
   /root/.vscode-server/bin/f6cfa2ea2403534de03f069bdf160d06451ed282/
   ```

   如果 VS Code 自动更新到另一个 commit，新目录下会有新的 `node`，旧 wrapper 不一定还覆盖新版本。

3. Codex VS Code 扩展更新了

   当前 Codex 扩展目录是：

   ```text
   /root/.vscode-server/extensions/openai.chatgpt-26.519.32039-linux-x64/
   ```

   如果扩展版本更新，新的扩展目录里会有新的 bundled `codex`，旧 wrapper 不一定还覆盖新版本。

4. 重新安装或清空了 `/root/.vscode-server`

   这种情况下 wrapper 和 `server-env-setup` 可能都会消失，需要重新做。

## 如果以后又 reconnect，先看什么

先不要急着重新登录。按这个顺序判断：

1. 看是否还是代理问题

   检查新 app-server 和 extension host 的环境变量里有没有：

   ```text
   HTTP_PROXY=http://127.0.0.1:17897
   HTTPS_PROXY=http://127.0.0.1:17897
   ```

2. 看日志错误类型

   如果是：

   ```text
   TypeError: fetch failed
   http/request failed
   ```

   多半还是网络或代理环境问题。

   如果是：

   ```text
   401
   token_invalidated
   Unauthorized
   ```

   才更像登录状态问题。

3. 只有明确出现认证错误时，再考虑重新登录

   例如：

   ```bash
   codex logout
   codex login --device-auth
   ```

   这次修复中没有执行这一步，因为当前问题已经证明主要是代理继承问题。

## 当前状态

已成功恢复。你现在能在 VS Code 右侧 Codex UI 里继续对话，说明：

- 右侧 UI 已经能连接后台；
- app-server 已经正常初始化；
- extension host 的网络请求已经不再卡在原来的 `fetch failed` reconnect 循环上。

