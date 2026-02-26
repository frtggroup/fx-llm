# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA Docker ローカルセットアップ
# - SSH 鍵ペア生成 (未作成の場合)
# - authorized_keys に公開鍵をコピー
# - docker build & run のコマンドを表示
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$RepoRoot  = Split-Path $PSScriptRoot -Parent
$DokEaDir  = $PSScriptRoot
$KeyFile   = "$DokEaDir\ssh\fx_ea_id_rsa"
$AuthKeys  = "$DokEaDir\ssh\authorized_keys"

Write-Host "=== FX AI EA Docker セットアップ ===" -ForegroundColor Cyan

# ── SSH 鍵生成 ────────────────────────────────────────────────────────────────
if (-not (Test-Path "$KeyFile.pub")) {
    Write-Host "[*] SSH 鍵ペアを生成中..."
    New-Item -ItemType Directory -Force "$DokEaDir\ssh" | Out-Null
    ssh-keygen -t rsa -b 4096 -f "$KeyFile" -N "" -C "fx-ea-docker"
    Write-Host "[OK] SSH 鍵生成完了"
} else {
    Write-Host "[*] SSH 鍵は既に存在します: $KeyFile.pub"
}

# ── authorized_keys に公開鍵を設定 ───────────────────────────────────────────
$pubKey = Get-Content "$KeyFile.pub" -Raw
# コメント行を除いた既存の鍵を保持しつつ追加
$existing = @()
if (Test-Path $AuthKeys) {
    $existing = Get-Content $AuthKeys | Where-Object { $_ -notmatch "^#" -and $_.Trim() -ne "" }
}
if ($existing -notcontains $pubKey.Trim()) {
    Add-Content -Path $AuthKeys -Value $pubKey.Trim()
    Write-Host "[OK] 公開鍵を authorized_keys に追加しました"
} else {
    Write-Host "[*] 公開鍵は既に authorized_keys に登録されています"
}

# ── Docker ビルド情報 ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Docker ビルド & 実行コマンド ===" -ForegroundColor Yellow
Write-Host ""
Write-Host "# ビルド (FX ルートから実行):" -ForegroundColor Cyan
Write-Host "  docker build -f DOK_EA/Dockerfile -t fx-ea:latest $RepoRoot"
Write-Host ""
Write-Host "# ローカル実行 (GPU):" -ForegroundColor Cyan
Write-Host "  docker run --gpus all -p 7860:7860 -p 2222:22 ``"
Write-Host "    -e DATA_URL='https://your-storage/USDJPY_M1.csv' ``"
Write-Host "    -e H100_MODE=1 ``"
Write-Host "    -v `"$DokEaDir\data:/workspace/data`" ``"
Write-Host "    fx-ea:latest"
Write-Host ""
Write-Host "# Sakura DOK 環境変数:" -ForegroundColor Cyan
Write-Host "  H100_MODE=1"
Write-Host "  DATA_URL=<CSV の直リンク URL>"
Write-Host ""
Write-Host "# SSH 接続:" -ForegroundColor Cyan
Write-Host "  ssh -i `"$KeyFile`" -p 2222 root@<DOK_IP>"
Write-Host ""
Write-Host "# ダッシュボード:" -ForegroundColor Cyan
Write-Host "  http://<DOK_IP>:7860"
Write-Host ""
Write-Host "=== 完了 ===" -ForegroundColor Green
