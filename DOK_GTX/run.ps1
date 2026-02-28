# GTX / ローカル GPU コンテナ起動スクリプト (Windows PowerShell)
# GPU設定・並列数は run_train.py が実測 VRAM から自動計算

$IMAGE = "frtggroup/fx-ea-gtx:latest"
$NAME  = "ea"

# 既存コンテナを停止・削除
$existing = docker ps -a --format "{{.Names}}" 2>$null | Where-Object { $_ -eq $NAME }
if ($existing) {
    Write-Host "[*] 既存コンテナを停止中..."
    docker stop $NAME 2>$null
    docker rm   $NAME 2>$null
}

Write-Host "[*] コンテナ起動 (GPU: --gpus all)..."
docker run -d `
    --name $NAME `
    --gpus all `
    -p 8080:8080 `
    --shm-size 2gb `
    --restart unless-stopped `
    -e NVIDIA_VISIBLE_DEVICES=all `
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
    $IMAGE

Write-Host ""
Write-Host "[OK] コンテナ起動完了"
Write-Host "     ダッシュボード : http://localhost:8080"
Write-Host "     ログ確認       : docker logs -f $NAME"
Write-Host "     停止           : docker stop $NAME"
