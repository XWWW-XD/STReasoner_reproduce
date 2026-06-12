# 在 Windows PowerShell 运行（会提示一次 root 密码，成功后免密）
#   cd <仓库>\00_new_codes\repro_autodl\vsc_codex_remote_setup
#   .\install-ssh-pubkey-windows.ps1

$ErrorActionPreference = "Stop"
$HostAlias = "autodl-A800"
$PubKey = Join-Path $env:USERPROFILE ".ssh\id_ed25519.pub"

if (-not (Test-Path $PubKey)) {
    Write-Host "未找到 $PubKey ，先生成密钥：" -ForegroundColor Yellow
    ssh-keygen -t ed25519 -f (Join-Path $env:USERPROFILE ".ssh\id_ed25519") -C "autodl-vscode" -N ""
}

Write-Host "公钥指纹（本地）："
ssh-keygen -lf $PubKey

Write-Host "上传公钥到 $HostAlias （请输入一次服务器密码）..."
Get-Content $PubKey | ssh $HostAlias "bash /root/install-ssh-pubkey.sh"

Write-Host "免密测试..."
ssh -o BatchMode=yes $HostAlias "echo SSH_KEY_OK; hostname; whoami"
Write-Host "若看到 SSH_KEY_OK 则成功。请 Kill VS Code Server 后重连 $HostAlias 。" -ForegroundColor Green
