# ============================================================
# FX AI EA - Google Cloud TPU v6e-1 自動セットアップスクリプト
# 実行方法: powershell -ExecutionPolicy Bypass -File setup_tpu.ps1
# ============================================================
param(
    [string]$ProjectId   = "",   # gcloud project ID
    [string]$Zone        = "us-east5-a",
    [string]$VmName      = "fx-ea-tpu-v6e",
    [string]$DockerImage = "frtggroup/fx-ea-tpu:latest"
)

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

# ── 0. プロジェクト ID 入力 ────────────────────────────────────────────
if (-not $ProjectId) {
    $ProjectId = Read-Host "Google Cloud プロジェクト ID を入力してください"
}
if (-not $ProjectId) { Write-Error "プロジェクト ID が指定されていません"; exit 1 }

# ── 1. gcloud SDK インストール確認 ────────────────────────────────────
Write-Host "`n[1/6] gcloud CLI 確認中..." -ForegroundColor Cyan
$gcloudPath = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloudPath) {
    Write-Host "  gcloud CLI が見つかりません → Google Cloud SDK をダウンロード中..." -ForegroundColor Yellow
    $installer = "$env:TEMP\GoogleCloudSDKInstaller.exe"
    Invoke-WebRequest -Uri "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe" `
                      -OutFile $installer -UseBasicParsing
    Write-Host "  インストーラー起動中... (ウィザードの指示に従ってください)"
    Start-Process -FilePath $installer -Wait
    # PATH を再読み込み
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    $gcloudPath = Get-Command gcloud -ErrorAction SilentlyContinue
    if (-not $gcloudPath) {
        Write-Error "gcloud のインストール後に PowerShell を再起動して再実行してください"
        exit 1
    }
}
Write-Host "  gcloud OK: $(gcloud version --format='value(Google Cloud SDK)')" -ForegroundColor Green

# ── 2. 認証 ────────────────────────────────────────────────────────────
Write-Host "`n[2/6] Google Cloud 認証..." -ForegroundColor Cyan
$authCheck = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1
if (-not $authCheck -or $authCheck -match "No credentialed") {
    Write-Host "  ブラウザで Google アカウントにログインしてください..."
    gcloud auth login
    gcloud auth application-default login
}
Write-Host "  認証済みアカウント: $authCheck" -ForegroundColor Green

# ── 3. プロジェクト設定 ────────────────────────────────────────────────
Write-Host "`n[3/6] プロジェクト設定: $ProjectId" -ForegroundColor Cyan
gcloud config set project $ProjectId
gcloud services enable tpu.googleapis.com compute.googleapis.com --quiet
Write-Host "  TPU API 有効化 OK" -ForegroundColor Green

# ── 4. TPU VM 作成 ────────────────────────────────────────────────────
Write-Host "`n[4/6] TPU VM 作成: $VmName (v6e-1, $Zone, spot)..." -ForegroundColor Cyan
$existingVm = gcloud compute tpus tpu-vm list --zone=$Zone --filter="name=$VmName" `
              --format="value(name)" 2>&1
if ($existingVm -match $VmName) {
    Write-Host "  VM '$VmName' は既に存在します → スキップ" -ForegroundColor Yellow
} else {
    gcloud compute tpus tpu-vm create $VmName `
        --zone=$Zone `
        --accelerator-type=v6e-1 `
        --version=v2-alpha-tpuv6e `
        --spot
    if ($LASTEXITCODE -ne 0) { Write-Error "TPU VM 作成失敗"; exit 1 }
    Write-Host "  TPU VM 作成 OK" -ForegroundColor Green
    Write-Host "  起動まで 2 分待機中..." -ForegroundColor Yellow
    Start-Sleep -Seconds 120
}

# ── 5. VM に Docker + コンテナ起動スクリプトを転送して実行 ──────────────
Write-Host "`n[5/6] VM に Docker セットアップスクリプトを送信中..." -ForegroundColor Cyan

$remoteScript = @"
#!/bin/bash
set -e

# Docker インストール
if ! command -v docker &>/dev/null; then
    echo '[*] Docker インストール中...'
    sudo apt-get update -q
    sudo apt-get install -y --no-install-recommends docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker \$USER
    echo '[OK] Docker インストール完了'
fi

# 既存コンテナ停止
docker stop fx-ea-tpu 2>/dev/null || true
docker rm   fx-ea-tpu 2>/dev/null || true

# 最新イメージ取得
echo '[*] Docker イメージ取得中: $DockerImage'
sudo docker pull $DockerImage

# コンテナ起動
# --device=/dev/accel0  : v6e TPU チップへのアクセス
# --privileged --net=host: TPU ランタイム + ダッシュボード公開
echo '[*] コンテナ起動中...'
sudo docker run -d \
    --name fx-ea-tpu \
    --privileged \
    --net=host \
    --restart=unless-stopped \
    -e PJRT_DEVICE=TPU \
    -e TPU_NUM_DEVICES=1 \
    -e TPU_ACCELERATOR_TYPE=v6e-1 \
    $DockerImage

echo ''
echo '======================================================'
echo '  FX AI EA コンテナ起動完了!'
echo "  ダッシュボード: http://\$(curl -s ifconfig.me):8080"
echo '  ログ確認: sudo docker logs -f fx-ea-tpu'
echo '======================================================'
"@

# スクリプトを一時ファイルに書いてSCPで転送
$tmpScript = "$env:TEMP\tpu_setup_remote.sh"
$remoteScript | Set-Content -Path $tmpScript -Encoding utf8

gcloud compute tpus tpu-vm scp $tmpScript "${VmName}:/tmp/tpu_setup_remote.sh" `
    --zone=$Zone --strict-host-key-checking=no

gcloud compute tpus tpu-vm ssh $VmName --zone=$Zone `
    --strict-host-key-checking=no `
    --command="bash /tmp/tpu_setup_remote.sh"

if ($LASTEXITCODE -ne 0) { Write-Error "リモートセットアップ失敗"; exit 1 }

# ── 6. 外部 IP 表示 ───────────────────────────────────────────────────
Write-Host "`n[6/6] セットアップ完了!" -ForegroundColor Green
$ip = gcloud compute tpus tpu-vm describe $VmName --zone=$Zone `
      --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>&1
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "  ダッシュボード URL: http://${ip}:8080" -ForegroundColor Yellow
Write-Host "  SSH 接続:"
Write-Host "    gcloud compute tpus tpu-vm ssh $VmName --zone=$Zone"
Write-Host "  ログ確認 (SSH後):"
Write-Host "    sudo docker logs -f fx-ea-tpu"
Write-Host "  停止:"
Write-Host "    gcloud compute tpus tpu-vm delete $VmName --zone=$Zone"
Write-Host "=====================================================" -ForegroundColor Green
