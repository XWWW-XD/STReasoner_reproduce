# 新服务器 VSCode Codex 设备码登录页不弹出排查

## 0. 先校正问题对象

这次问题不是当前 `1服务器` 的右侧 Codex UI 打不开。当前 `1服务器` 已经能通过 VS Code 右侧 Codex 插件正常对话。

用户说的问题是：

```text
2服务器刚开始配置；
在 2服务器 的 VS Code 右侧 Codex 插件里选择设备码登录；
但是登录网页没有弹出来。
```

因此，当前 `1服务器` 的日志只能作为稳定基准，不能直接当成 `2服务器` 的故障证据。`2服务器` 需要单独检查 SSH、VS Code Server、Codex 插件、代理和本地浏览器打开链路。

## 1. 这个问题属于哪一层

报告 49 和 50 分别解决了两个前置问题：

| 报告 | 解决的问题 |
| --- | --- |
| `49-VSCode远程与Codex稳定连接配置迁移清单.md` | VS Code Remote / Codex app-server / extensionHost 的代理继承和 reconnect。 |
| `50-新服务器SSH免密登录与反复输入密码排查.md` | 新服务器 SSH 免密，避免 VS Code 反复要求 root 密码。 |

本报告 51 解决的是第三层：

```text
右侧 VS Code Codex 插件选择设备码登录后，登录页没有在本地浏览器弹出。
```

这个问题可能发生在四个地方：

1. `2服务器` 的 SSH/VS Code Remote 还不稳定，插件没有完整启动。
2. `2服务器` 的 Codex app-server 或 extensionHost 没有代理，无法向 OpenAI 后端请求登录流程。
3. 登录 URL 已经生成，但 VS Code 没能在 Windows 本地浏览器打开。
4. 当前 Codex 插件版本和 VS Code Server 有兼容性问题，导致登录 UI / openExternal 逻辑异常。

不要一上来复制 token 或删除配置。先定位是哪一层坏了。

## 2. 设备码登录页应该在哪里弹出

通过 VS Code Remote-SSH 使用右侧 Codex 插件时，涉及两端：

```text
Windows 本地 VS Code 窗口
    ↕ Remote-SSH
AutoDL 远端 VS Code Server / extensionHost / codex app-server
```

设备码登录时，网页应该在 **Windows 本地浏览器** 打开，而不是在远端 AutoDL 服务器里打开。

所以如果没有弹网页，需要同时看两边：

- 远端是否成功请求到登录信息。
- 本地 VS Code 是否成功调用浏览器打开 URL。

远端配置能解决代理和 app-server；但“浏览器弹不弹”最终还要看 Windows 本地 VS Code、默认浏览器、通知弹窗、URI opener。

## 3. 2 服务器第一优先级：先确认 SSH 和代理前置条件

### 3.1 先确认 SSH 免密

在 Windows PowerShell：

```powershell
ssh -o BatchMode=yes autodl-2 "echo SSH_KEY_OK; hostname; whoami"
```

期望：

```text
SSH_KEY_OK
<2服务器 hostname>
root
```

如果这里失败，先按报告 50 修 SSH key。SSH 还在反复问密码时，不要急着排 Codex 登录。

### 3.2 再确认远端代理端口

进入 `2服务器` 远端终端后：

```bash
export CODEX_PROXY=http://127.0.0.1:<PORT>
curl -I https://chatgpt.com --proxy "$CODEX_PROXY" --max-time 20
```

如果返回类似：

```text
HTTP/1.1 200 Connection established
```

说明远端能通过代理出去。

如果是：

```text
Connection refused
Failed to connect to 127.0.0.1 port <PORT>
```

说明 `2服务器` 没有拿到代理端口。先按报告 49 配远端代理，不要继续折腾登录页。

### 3.3 检查 2 服务器后台进程是否带代理

在 `2服务器`：

```bash
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
```

找到 PID 后：

```bash
tr '\0' '\n' < /proc/<PID>/environ | grep -Ei 'proxy|NODE_USE_ENV_PROXY|CODEX'
```

最低要求：

```text
extensionHost 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
codex app-server 有 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
```

如果终端有代理，但这两个后台进程没有代理，右侧 Codex UI 仍可能登录失败或 reconnect。

## 4. 如果代理没问题但网页不弹，优先查本地打开链路

当远端代理和后台进程都没问题，但选择设备码登录后网页不弹，优先判断：

```text
登录 URL 是否生成了？
VS Code 是否把 URL 交给本地浏览器？
Windows 默认浏览器是否拦截或没响应？
```

可以按这个顺序做：

1. 看 VS Code 右下角通知中心，有时登录链接不是自动弹浏览器，而是显示在通知里。
2. 看 Codex 侧边栏是否出现设备码、复制链接、Open in browser、Copy code 一类按钮。
3. 在 VS Code 命令面板运行：

   ```text
   Codex: Open Codex Sidebar
   Codex: New Codex Agent
   ```

   然后重新触发登录。

4. 如果页面没有自动打开，但 UI 给了 URL 或 device code，手动复制到 Windows 本地浏览器打开。
5. 确认 Windows 默认浏览器可用，且 VS Code 没有被系统拦截打开外部链接。

这一层不是远端 Linux 命令能完全修好的，因为浏览器在 Windows 本地。

## 5. 2 服务器需要收集的日志

在 `2服务器` 上执行：

```bash
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head
```

取最新的 `Codex.log`，再查：

```bash
grep -RInEi 'auth|login|sign|signin|device|code|browser|openExternal|external|uri|callback|oauth|token|wham|account|accounts|reconnect|fetch failed|Connection refused|401|unauthorized|invalidated' \
  /root/.vscode-server/data/logs/*/exthost*/openai.chatgpt/Codex.log 2>/dev/null | tail -n 200
```

同时查 extension host：

```bash
grep -RInEi 'openai.chatgpt|auth|login|sign|device|browser|openExternal|external|uri|callback|oauth|token|Authorize|PendingMigrationError|navigator' \
  /root/.vscode-server/data/logs/*/exthost*/remoteexthost.log 2>/dev/null | tail -n 200
```

日志判断：

| 日志关键词 | 更可能的问题 |
| --- | --- |
| `Connection refused` / `fetch failed` / `http/request failed` | 远端代理或后台进程没有代理。 |
| `401` / `unauthorized` / `token_invalidated` | 登录态无效，需要重新登录。 |
| `device` / `oauth` / `callback` 后没有浏览器动作 | 可能是本地 VS Code / Windows 浏览器打开链路。 |
| `PendingMigrationError: navigator is now a global in nodejs` | 插件版本和 VS Code Server 兼容性可疑。 |
| 没有任何 Codex 激活日志 | 插件没有启动，先检查 VS Code 远程扩展安装。 |

## 6. 当前 1 服务器作为基准看到的情况

当前 `1服务器` 的基准状态：

- 右侧 VS Code Codex 插件可以工作。
- `extensionHost` 和 `codex app-server` 都有 `HTTP_PROXY=http://127.0.0.1:17997`。
- 当前安装并激活的插件是 `openai.chatgpt-26.609.30741`。
- 旧版 `openai.chatgpt-26.602.71036` 目录仍存在，但已在 `.obsolete` 中标记。
- 当前远端 `/root/.codex/auth.json` 存在，但不读取、不复制内容。

当前 `1服务器` 日志中有几个可参考点：

```text
Activating Codex extension
Spawning codex app-server
Initialize received
Features enabled ... auth_elicitation ...
IAB_LIFECYCLE webview captured browser use session route ...
```

这说明 1 服务器的插件和 app-server 至少能启动、能进入 webview 会话。

但当前 1 服务器日志不能证明 2 服务器问题的根因。2 服务器必须单独取日志。

## 7. 当前值得注意的插件版本问题

当前 1 服务器在 `openai.chatgpt-26.609.30741` 激活时，`remoteexthost.log` 出现过：

```text
PendingMigrationError: navigator is now a global in nodejs
```

这个错误来自：

```text
/root/.vscode-server/extensions/openai.chatgpt-26.609.30741/out/extension.js
```

它没有阻止 1 服务器 app-server 启动，也没有阻止当前对话继续，但如果 2 服务器也出现“插件启动了但登录 UI 不正常”，这个版本兼容性要列为嫌疑。

如果 2 服务器日志也有同样错误，且代理、SSH 都确认正常，可以考虑：

1. `Developer: Reload Window`。
2. `Remote-SSH: Kill VS Code Server on Host...` 后重连。
3. 卸载并重新安装 OpenAI Codex / ChatGPT 扩展。
4. 如果刚升级到 `26.609.30741` 后才出问题，考虑临时回退到之前能工作的扩展版本，或等待扩展/VS Code 修复。

不要优先手动改扩展的 `extension.js`。这是插件发布包，不适合手改。

## 8. 2 服务器建议操作顺序

推荐顺序：

1. 按报告 50 配好 `autodl-2` SSH 免密。
2. 按报告 49 配好 `2服务器` 的：

   ```text
   /root/.vscode-server/server-env-setup
   /root/.codex/codex-vscode-wrapper.sh
   /root/.vscode-server/data/User/settings.json
   /root/.vscode-server/data/Machine/settings.json
   /root/.bashrc
   ```

3. 重启 VS Code Server。
4. 在 `2服务器` 验证 `curl --proxy`。
5. 在 `2服务器` 验证 `extensionHost` 和 `codex app-server` 都带 proxy。
6. 重新打开右侧 Codex 插件，选择设备码登录。
7. 如果浏览器不弹：
   - 看 VS Code 通知中心；
   - 看侧边栏是否给 URL / code；
   - 手动复制链接到 Windows 浏览器；
   - 抓取第 5 节日志。

## 9. 不建议做的事

不要做这些：

- 不要把 1 服务器的 `/root/.codex/auth.json` 复制到 2 服务器。
- 不要把 1 服务器的 `.vscode-server` 整个复制到 2 服务器。
- 不要在没确认代理之前反复重装插件。
- 不要在没确认 SSH 免密之前排 Codex 登录。
- 不要把 1 服务器和 2 服务器都叫 `autodl`。
- 不要手改 OpenAI Codex 插件的 minified JS。

## 10. 最小结论

2 服务器右侧 Codex 选择设备码登录后网页不弹，最可能不是一个单独的“服务器密码”问题，而是下面三类之一：

1. `2服务器` 还没完整复刻 49/50，导致插件或 app-server 登录链路不完整。
2. 远端已经请求到登录流程，但 Windows 本地 VS Code 没有成功打开浏览器。
3. 当前 OpenAI Codex 插件版本和 VS Code Server 的组合在登录 UI 上有兼容性问题。

下一步最有价值的证据不是继续猜，而是在 `2服务器` 上抓：

```bash
ps -eo pid,etime,cmd | grep -E 'extensionHost|codex app-server' | grep -v grep
tr '\0' '\n' < /proc/<PID>/environ | grep -Ei 'proxy|CODEX'
find /root/.vscode-server/data/logs -path '*/openai.chatgpt/Codex.log' -printf '%T@ %p\n' | sort -nr | head
grep -RInEi 'auth|device|oauth|browser|openExternal|fetch failed|Connection refused|401|token_invalidated|PendingMigrationError' /root/.vscode-server/data/logs/* 2>/dev/null | tail -n 200
```

拿到这些后，才能判断是远端代理、插件版本，还是本地浏览器弹窗链路。

