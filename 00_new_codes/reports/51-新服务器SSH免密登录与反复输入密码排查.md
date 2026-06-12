# 新服务器 SSH 免密登录与反复输入密码排查

## 0. 结论

新服务器不停要求输入密码，通常不是服务器上还要额外配置“密码”，而是 **2 服务器没有配置好 SSH 公钥免密登录**，或者本地 VS Code/SSH 没有使用正确的私钥。

当前 1 服务器已经稳定，SSH 登录配置的核心是：

- Windows 本地 SSH config 指定 `IdentityFile ~/.ssh/id_ed25519`。
- 远端 `/root/.ssh/authorized_keys` 中保存对应的公钥。
- 远端 `/root/.ssh` 权限正确。
- VS Code Remote-SSH 连接使用 SSH config 里的 host alias。

如果 2 服务器还没有同样放入公钥，VS Code Remote-SSH 会在多个连接阶段反复要求 root 密码。这是正常现象：VS Code 不是只建立一个 SSH 连接，它会登录、启动 server、建通道、启动扩展，因此密码认证会被问很多次。解决办法是把 2 服务器也配置成公钥登录。

## 1. 当前 1 服务器状态

当前机器检查结果：

```text
/root/.ssh 权限：700
/root/.ssh/authorized_keys 权限：600
authorized_keys 行数：1
公钥类型：ED25519
公钥备注：Gitee SSH Key
```

只记录指纹，不记录公钥原文：

```text
SHA256:R60e0Dly8FpQ+Z7pTBArLesXFqOGlsdrl72wqmXHMNk
```

当前 `/etc/ssh/sshd_config` 里：

```text
PubkeyAuthentication 默认开启
AuthorizedKeysFile 默认使用 .ssh/authorized_keys
PasswordAuthentication 默认允许
PermitRootLogin yes
```

也就是说，1 服务器不是靠禁用密码登录稳定下来的，而是因为公钥已经配置好了。

## 2. 本地 Windows SSH config 应该怎么写

建议不要把两台服务器都叫 `autodl`。如果 1 服务器和 2 服务器可能同时开，本地 `C:\Users\HUAWEI\.ssh\config` 建议写成：

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

关键是：

```sshconfig
IdentityFile ~/.ssh/id_ed25519
IdentitiesOnly yes
```

`IdentityFile` 告诉 SSH 用哪个私钥。`IdentitiesOnly yes` 避免 SSH 把一堆乱七八糟的 key 都试一遍，导致认证混乱或提前失败。

## 3. 2 服务器免密登录推荐配置方法

### 3.1 先确认本地有公钥

在 Windows PowerShell 上执行：

```powershell
dir $env:USERPROFILE\.ssh\id_ed25519*
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

应该至少有：

```text
id_ed25519
id_ed25519.pub
```

注意：

- `id_ed25519` 是私钥，不要发给别人，不要复制到服务器报告里。
- `id_ed25519.pub` 是公钥，可以放到服务器的 `authorized_keys`。

如果本地没有这对 key，再生成：

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519 -C "autodl-vscode"
```

如果它问 passphrase，可以直接回车留空；如果设置了 passphrase，后面要配 `ssh-agent`，否则 VS Code 可能反复问 passphrase。

### 3.2 方法 A：通过 AutoDL 页面添加公钥

如果 AutoDL 页面支持 SSH 公钥配置，优先用这个方式：

1. 在 Windows PowerShell 执行：

   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub
   ```

2. 复制整行公钥。
3. 粘贴到 AutoDL 的 SSH 公钥/免密登录配置里。
4. 重启或重新打开 2 服务器实例后测试：

   ```powershell
   ssh -o BatchMode=yes autodl-2 "echo SSH_KEY_OK; hostname"
   ```

如果返回 `SSH_KEY_OK`，说明免密已经通了。

### 3.3 方法 B：已经能用密码登录时，手动追加 authorized_keys

如果 AutoDL 页面不好用，也可以先用密码登录 2 服务器，然后手动追加公钥。

在 Windows PowerShell 执行：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh root@<SERVER2_HOST> -p <SERVER2_PORT> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

或者，如果已经通过 VS Code/SSH 进了 2 服务器，就在 2 服务器上执行：

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cat >> /root/.ssh/authorized_keys <<'EOF'
<把 Windows 上 id_ed25519.pub 的整行公钥粘贴到这里>
EOF
chmod 600 /root/.ssh/authorized_keys
chown -R root:root /root/.ssh
```

然后从 Windows 重新测试：

```powershell
ssh -o BatchMode=yes autodl-2 "echo SSH_KEY_OK; hostname"
```

`BatchMode=yes` 的意义是：如果 key 不通，它不会退回密码输入，而是直接失败。这个测试比“能不能手动输密码连上”更准确。

## 4. 如果还是一直要密码，按这个顺序查

### 4.1 确认 VS Code 选的是正确 Host

如果你在 VS Code Remote-SSH 里选的是旧的 `autodl`，但实际想连 2 服务器，可能还在走旧配置。

建议明确选择：

```text
autodl-1
autodl-2
```

不要在同一个 `Host autodl` 下面来回改 HostName 和 Port。

### 4.2 确认本地 config 生效

Windows PowerShell：

```powershell
ssh -G autodl-2 | findstr /i "hostname port user identityfile identitiesonly"
```

应该看到：

```text
user root
identityfile ...id_ed25519
identitiesonly yes
hostname <SERVER2_HOST>
port <SERVER2_PORT>
```

如果 `identityfile` 不是 `id_ed25519`，说明 SSH config 没写对或 VS Code 没用这个 alias。

### 4.3 强制只用公钥测试

Windows PowerShell：

```powershell
ssh -vvv -o PreferredAuthentications=publickey -o PasswordAuthentication=no autodl-2 "echo SSH_KEY_OK"
```

看日志里的关键字：

| 日志现象 | 含义 |
| --- | --- |
| `Offering public key` | 本地确实拿 key 去试了。 |
| `Server accepts key` | 服务器接受公钥，免密应成功。 |
| `Permission denied (publickey)` | 服务器没有对应公钥，或权限不对。 |
| 没有 `Offering public key` | 本地没有用到正确私钥。 |

### 4.4 检查 2 服务器远端权限

在 2 服务器上：

```bash
ls -ld /root /root/.ssh
ls -l /root/.ssh/authorized_keys
wc -l /root/.ssh/authorized_keys
ssh-keygen -lf /root/.ssh/authorized_keys
```

推荐权限：

```text
/root/.ssh              700
/root/.ssh/authorized_keys 600
owner                   root:root
```

如果权限太松，OpenSSH 可能拒绝使用 `authorized_keys`。

### 4.5 如果问的是私钥 passphrase，不是服务器密码

有时 VS Code 弹出的不是服务器 root 密码，而是本地私钥 passphrase。区别：

- 服务器密码：通常提示 `root@host's password`。
- 私钥 passphrase：通常提示 `Enter passphrase for key ...id_ed25519`。

如果是 passphrase，配置 Windows `ssh-agent`：

```powershell
Get-Service ssh-agent
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
ssh-add -l
```

之后 VS Code 通常不会反复问 passphrase。

## 5. 配好后 VS Code 仍问密码怎么办

先在 Windows PowerShell 跑：

```powershell
ssh -o BatchMode=yes autodl-2 "echo SSH_KEY_OK"
```

结果分两种：

1. 如果 PowerShell 这里都失败，说明 SSH key 还没配通，不要先怪 VS Code。
2. 如果 PowerShell 成功，但 VS Code 还问密码，说明 VS Code Remote-SSH 可能没有选对 host alias，或者使用了不同的 SSH config 路径。

VS Code 侧可以检查：

- Remote-SSH 选择的是 `autodl-2`。
- `Remote.SSH: Config File` 指向 `C:\Users\HUAWEI\.ssh\config`。
- 必要时执行 `Remote-SSH: Kill VS Code Server on Host...` 后重新连接。

## 6. 不建议做的事

不要为了省事做这些：

- 不要把 Windows 私钥 `id_ed25519` 复制到服务器。
- 不要把 1 服务器的 `/root/.ssh/authorized_keys` 盲目覆盖到 2 服务器；如果要复用，只追加你自己的公钥，不覆盖。
- 不要在 key 没配通前关闭服务器密码登录，否则可能把自己锁在外面。
- 不要把 1 服务器和 2 服务器都叫 `autodl`。
- 不要把密码写进报告、脚本或 git。

## 7. 最小成功标准

2 服务器配置完成后，至少满足：

```powershell
ssh -o BatchMode=yes autodl-2 "echo SSH_KEY_OK; hostname; whoami"
```

返回类似：

```text
SSH_KEY_OK
<server-hostname>
root
```

然后 VS Code Remote-SSH 再连接 `autodl-2`，正常情况下就不会反复要求服务器密码。

## 8. 和报告 49 的关系

报告 49 主要解决的是：

```text
VS Code/Codex 连接稳定、代理继承、右侧 UI reconnect
```

本报告 51 主要解决的是：

```text
SSH 登录免密、新服务器反复输入密码
```

推荐顺序是：

1. 先按本报告 51 配好 `autodl-A800` 免密 SSH（脚本：`00_new_codes/repro_autodl/vsc_codex_remote_setup/install-ssh-pubkey-windows.ps1`，远端 `/root/install-ssh-pubkey.sh`）。
2. 再按报告 49 配好 VS Code Server / Codex proxy。

如果 SSH 还在反复问密码，先不要急着处理 Codex 代理；基础 SSH 免密是前置条件。

