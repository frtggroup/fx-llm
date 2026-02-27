#!/bin/bash
# GTX 1080 Ti コンテナ起動スクリプト
# docker compose の GPU パススルー問題を回避するため docker run を直接使用
set -e

IMAGE="frtggroup/fx-ea-gtx:latest"
NAME="fx-ea-gtx"

# 既存コンテナを停止・削除
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[*] 既存コンテナを停止中..."
    docker stop "$NAME" 2>/dev/null || true
    docker rm   "$NAME" 2>/dev/null || true
fi

# 最新イメージを取得
echo "[*] イメージ更新..."
docker pull "$IMAGE"

echo "[*] コンテナ起動 (GPU: --gpus all)..."
docker run -d \
    --name "$NAME" \
    --gpus all \
    -p 8080:8080 \
    -p 2222:22 \
    --shm-size 2gb \
    --restart unless-stopped \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e H100_MODE=0 \
    -e DASHBOARD_PORT=8080 \
    -e MAX_PARALLEL=2 \
    -e VRAM_PER_TRIAL=5 \
    -e BT_CAPITAL=150000 \
    -e BT_LEVERAGE=1000 \
    -e BT_RISK_PCT=1.0 \
    -e S3_ENDPOINT=https://s3.isk01.sakurastorage.jp \
    -e S3_REGION=jp-north-1 \
    -e S3_ACCESS_KEY=9LZ71SXF347BKM7MEPPM \
    -e "S3_SECRET_KEY==u/j64oP8sQvtRXjjNY33jF2S/gOY5WMVgYcdnqG" \
    -e S3_BUCKET=fxea \
    -e S3_PREFIX=checkpoint_gtx \
    -e DATA_URL=https://fxea.s3.isk01.sakurastorage.jp/USDJPY_M1.csv \
    -e DATA_PATH=/workspace/data/USDJPY_M1.csv \
    "$IMAGE"

echo ""
echo "[OK] コンテナ起動完了"
echo "     ダッシュボード : http://localhost:8080"
echo "     ログ確認       : docker logs -f $NAME"
echo "     停止           : docker stop $NAME"
