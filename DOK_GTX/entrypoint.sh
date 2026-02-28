#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA GTX版 エントリポイント
# H100版と違い /opt/artifact は使わず、ボリュームマウントで永続化
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA ランダムサーチ on GTX 1080 Ti"
echo "======================================================"

# ── GPU 確認 + CUDA MPS 起動 ────────────────────────────────────────────────
echo "[*] GPU 確認..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
  && echo "[OK] GPU 検出" \
  || echo "[WARN] nvidia-smi 使用不可 (CPU モードで続行)"

echo "[*] CUDA MPS デーモン起動..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
nvidia-cuda-mps-control -d 2>/dev/null \
  && echo "[OK] CUDA MPS 起動完了" \
  || echo "[WARN] CUDA MPS 起動失敗 (続行)"

# ── SSH サーバー ────────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true   # ホスト鍵が未生成の場合に生成
/usr/sbin/sshd -D &
SSH_PID=$!
sleep 1
if kill -0 "$SSH_PID" 2>/dev/null; then
    echo "[OK] SSH サーバー起動 (PID: $SSH_PID)"
else
    echo "[WARN] SSH サーバー起動失敗 (続行)"
fi

# ── 環境変数 / パス ──────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
# NODE_ID / H100_MODE / MAX_PARALLEL / VRAM_PER_TRIAL は
# run_train.py が実測 GPU VRAM から自動計算するため設定不要

echo "[*] 設定:"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(未設定)}"
echo "    GPU設定      : run_train.py 起動後に自動検出"

# ── CSV 自動ダウンロード: GDrive → DATA_URL の順で試みる ─────────────────────
mkdir -p "$(dirname ${DATA_PATH})"
if [ ! -f "${DATA_PATH}" ] || [ ! -s "${DATA_PATH}" ]; then
    echo "[*] CSV が見つかりません。自動ダウンロードを試みます..."
    python -c "
import sys, os
sys.path.insert(0, '/workspace/ai_ea')
from pathlib import Path
import gdrive
dst   = os.environ.get('DATA_PATH', '/workspace/data/USDJPY_H1.csv')
fname = os.path.basename(dst)
if not gdrive.GDRIVE_ENABLED:
    print('[WARN] GDrive 無効')
    sys.exit(1)
ok = gdrive.download(fname, Path(dst))
if ok and Path(dst).exists() and Path(dst).stat().st_size > 0:
    print(f'[OK] GDrive から CSV ダウンロード完了 ({Path(dst).stat().st_size/1e6:.1f} MB): {fname}')
    sys.exit(0)
print('[WARN] GDrive に CSV (', fname, ') が見つかりません')
sys.exit(1)
" && echo "[OK] CSV 準備完了 (GDrive)" || {
        if [ -n "${DATA_URL}" ]; then
            echo "[*] DATA_URL からダウンロード中: ${DATA_URL}"
            wget -q --show-progress -O "${DATA_PATH}" "${DATA_URL}" \
              && echo "[OK] CSV ダウンロード完了 $(du -h ${DATA_PATH} | cut -f1)" \
              || echo "[ERROR] CSV ダウンロード失敗"
        else
            echo "[WARN] GDrive にも DATA_URL にも CSV がありません"
            echo "[WARN] $(basename ${DATA_PATH}) を GDrive フォルダにアップロードするか"
            echo "[WARN] docker run -e DATA_URL=<url> で指定してください"
        fi
    }
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── ダッシュボードサーバー起動 ────────────────────────────────────────────
echo "[*] ダッシュボード起動 (port 8080)..."
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 2
if kill -0 "$DASH_PID" 2>/dev/null; then
    echo "[OK] ダッシュボード起動 (PID: $DASH_PID)"
else
    echo "[WARN] ダッシュボード起動失敗"
    cat /workspace/dashboard.log 2>/dev/null | tail -20
fi

# ── stop.flag クリア ────────────────────────────────────────────────────────
rm -f /workspace/stop.flag

# ── SIGTERM ハンドリング ─────────────────────────────────────────────────────
TRAIN_PID=""
_graceful_stop() {
    echo "[*] シグナル受信 → run_train.py に SIGTERM を送信..."
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID"
        for i in $(seq 1 30); do
            kill -0 "$TRAIN_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    fi
    echo "[OK] graceful shutdown 完了"
}
trap '_graceful_stop' SIGTERM SIGINT

# ── 学習ループ起動 ──────────────────────────────────────────────────────────
echo ""
echo "[*] ランダムサーチ開始  (GPU設定は自動検出)"
echo ""
python /workspace/ai_ea/run_train.py 2>&1 | tee /workspace/train_run.log &
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "  学習完了!"
    echo "  ダッシュボード : http://localhost:8080"
    echo "======================================================"
else
    echo "  [ERROR] 学習がエラーで終了 (exit=$EXIT_CODE)"
fi

echo "[*] コンテナ待機中..."
wait
