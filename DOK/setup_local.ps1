# ─────────────────────────────────────────────────────────────────────────────
# FX LLM DOK セットアップスクリプト (Windows PowerShell)
# 実行: cd f:\FX && .\DOK\setup_local.ps1
# ─────────────────────────────────────────────────────────────────────────────
param(
    [string]$DockerTag     = "fx-llm:latest",
    [string]$DockerHubUser = "",       # Docker Hub ユーザー名 (pushする場合)
    [switch]$PushImage,                # Docker Hub に push する場合
    [switch]$SkipBuild                 # ビルドをスキップ (push のみ)
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FxRoot    = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  FX LLM DOK セットアップ" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ── STEP 1: SSH 鍵ペア生成 ──────────────────────────────────────────────────
$SshDir      = "$ScriptDir\ssh"
$PrivKeyPath = "$SshDir\id_ed25519_dok"
$PubKeyPath  = "$SshDir\id_ed25519_dok.pub"
$AuthKeysPath= "$SshDir\authorized_keys"

if (-not (Test-Path $SshDir)) { New-Item -ItemType Directory -Path $SshDir | Out-Null }

if (-not (Test-Path $PrivKeyPath)) {
    Write-Host "[1/4] SSH 鍵ペア生成..." -ForegroundColor Yellow
    ssh-keygen -t ed25519 -f $PrivKeyPath -N "" -C "fx-llm-dok"
    Write-Host "  秘密鍵: $PrivKeyPath" -ForegroundColor Green
    Write-Host "  公開鍵: $PubKeyPath" -ForegroundColor Green
} else {
    Write-Host "[1/4] SSH 鍵ペアは既に存在します: $PrivKeyPath" -ForegroundColor Green
}

# authorized_keys に公開鍵をコピー
if (Test-Path $PubKeyPath) {
    Copy-Item -Path $PubKeyPath -Destination $AuthKeysPath -Force
    Write-Host "  authorized_keys 更新完了" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] 公開鍵が見つかりません: $PubKeyPath" -ForegroundColor Red
    exit 1
}

# .gitignore に SSH 秘密鍵を追加
$GitignorePath = "$ScriptDir\.gitignore"
$GitignoreContent = @"
ssh/id_ed25519_dok
ssh/id_ed25519_dok.pub
"@
Set-Content -Path $GitignorePath -Value $GitignoreContent
Write-Host "  .gitignore 作成: $GitignorePath" -ForegroundColor Gray

# ── STEP 2: Docker ビルド ────────────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[2/4] Docker イメージビルド: $DockerTag" -ForegroundColor Yellow
    Write-Host "  (初回は 10-20分かかります...)" -ForegroundColor Gray

    Push-Location $FxRoot
    try {
        docker build -f "DOK\Dockerfile" -t $DockerTag .
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Docker ビルド失敗" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host "  ビルド完了: $DockerTag" -ForegroundColor Green
} else {
    Write-Host "[2/4] Docker ビルド: スキップ" -ForegroundColor Gray
}

# ── STEP 3: Docker Hub に push (オプション) ──────────────────────────────────
if ($PushImage) {
    if (-not $DockerHubUser) {
        Write-Host "[3/4] Docker Hub ユーザー名を入力 (-DockerHubUser <name>)" -ForegroundColor Red
        exit 1
    }
    $RemoteTag = "${DockerHubUser}/fx-llm:latest"
    Write-Host ""
    Write-Host "[3/4] Docker Hub に push: $RemoteTag" -ForegroundColor Yellow
    docker tag $DockerTag $RemoteTag
    docker push $RemoteTag
    Write-Host "  push 完了: $RemoteTag" -ForegroundColor Green
    Write-Host ""
    Write-Host "  さくら DOK のイメージ欄に入力:"
    Write-Host "    $RemoteTag" -ForegroundColor Cyan
} else {
    Write-Host "[3/4] Docker Hub push: スキップ (-PushImage で有効化)" -ForegroundColor Gray
}

# ── STEP 4: SSH 接続設定表示 ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] 接続情報" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ─── SSH 接続 ───────────────────────────────" -ForegroundColor Cyan
Write-Host "  ssh -i $PrivKeyPath root@<DOK_IP_ADDRESS>" -ForegroundColor White
Write-Host ""
Write-Host "  ─── データアップロード ──────────────────────" -ForegroundColor Cyan
Write-Host "  scp -i $PrivKeyPath" -ForegroundColor White
Write-Host "      USDJPY_M1_*.csv root@<DOK_IP>:/workspace/data/" -ForegroundColor White
Write-Host ""
Write-Host "  ─── ダッシュボード ──────────────────────────" -ForegroundColor Cyan
Write-Host "  http://<DOK_IP>:7860" -ForegroundColor White
Write-Host ""
Write-Host "  ─── 環境変数 (DOK 設定画面で指定) ──────────" -ForegroundColor Cyan
Write-Host "  LLM_MODEL_ID=Qwen/Qwen3-8B" -ForegroundColor Gray
Write-Host "  LLM_EPOCHS=10" -ForegroundColor Gray
Write-Host "  LLM_BATCH=8" -ForegroundColor Gray
Write-Host "  LLM_GRAD_ACCUM=8" -ForegroundColor Gray
Write-Host "  LLM_SKIP_DATASET=1  # データセット生成済みの場合" -ForegroundColor Gray
Write-Host "  LLM_RESUME=1        # チェックポイントから再開" -ForegroundColor Gray
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  セットアップ完了！" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# SSH config に追記 (オプション確認)
$AddToSshConfig = Read-Host "SSH config (~/.ssh/config) に追加しますか? (y/N)"
if ($AddToSshConfig -eq 'y' -or $AddToSshConfig -eq 'Y') {
    $SshConfigPath = "$env:USERPROFILE\.ssh\config"
    $DokAlias = Read-Host "ホスト名エイリアスを入力 (例: dok-fx)"
    $DokIp    = Read-Host "DOK の IP アドレスを入力"
    $SshEntry = @"

Host $DokAlias
    HostName $DokIp
    User root
    IdentityFile $PrivKeyPath
    ServerAliveInterval 60
    ServerAliveCountMax 3
"@
    Add-Content -Path $SshConfigPath -Value $SshEntry
    Write-Host "  SSH config に追加しました: $SshConfigPath" -ForegroundColor Green
    Write-Host "  接続コマンド: ssh $DokAlias" -ForegroundColor Cyan
}
