#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA TPU/GPU 汎用エントリポイント
# デバイスは run_train.py が自動検出: TPU (XLA) > GPU (CUDA) > CPU
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA ランダムサーチ on TPU/GPU"
echo "======================================================"

# ── 1. デバイス確認 ─────────────────────────────────────────────────────────
echo "[*] デバイス確認..."
python -c "
import sys
# TPU 検出
try:
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    devs = xm.get_xla_supported_devices()
    print(f'[OK] TPU 検出: {dev}  利用可能チップ数={len(devs)}')
    sys.exit(0)
except Exception as e:
    print(f'[INFO] TPU なし ({e})')

# GPU 検出
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'[OK] GPU 検出: {name}  VRAM={vram:.0f} GB')
    else:
        print('[WARN] GPU なし (CPU モードで続行)')
except Exception as e:
    print(f'[WARN] デバイス検出失敗: {e}')
" 2>&1 || true

# ── 2. CUDA MPS (GPU 使用時のみ有効) ────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
nvidia-cuda-mps-control -d 2>/dev/null && echo "[OK] CUDA MPS 起動" \
  || echo "[INFO] CUDA MPS スキップ (TPU または非対応 GPU)"

# ── 3. SSH サーバー ──────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
sleep 1
kill -0 "$SSH_PID" 2>/dev/null && echo "[OK] SSH 起動 (PID: $SSH_PID)" \
  || echo "[WARN] SSH 起動失敗 (続行)"

# ── 4. 環境変数 / パス ──────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
# TPU XLA 設定
export XLA_USE_BF16="${XLA_USE_BF16:-1}"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
# NODE_ID / MAX_PARALLEL / VRAM_PER_TRIAL は run_train.py が自動計算

echo "[*] 設定:"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(未設定)}"
echo "    XLA_USE_BF16 : ${XLA_USE_BF16}"
echo "    PJRT_DEVICE  : ${PJRT_DEVICE}"
echo "    GPU設定      : run_train.py 起動後に自動検出"

# ── 5. ダッシュボード起動 ────────────────────────────────────────────────────
echo "[*] ダッシュボード起動 (port 8080)..."
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 3
kill -0 "$DASH_PID" 2>/dev/null && echo "[OK] ダッシュボード起動 (PID: $DASH_PID)" \
  || { echo "[WARN] ダッシュボード起動失敗"; cat /workspace/dashboard.log 2>/dev/null | tail -10; }

# ── 6. CSV 自動ダウンロード: GDrive → DATA_URL ─────────────────────────────
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
print('[WARN] GDrive に CSV が見つかりません')
sys.exit(1)
" && echo "[OK] CSV 準備完了 (GDrive)" || {
        if [ -n "${DATA_URL}" ]; then
            echo "[*] DATA_URL からダウンロード中: ${DATA_URL}"
            wget -q --show-progress -O "${DATA_PATH}" "${DATA_URL}" \
              && echo "[OK] CSV ダウンロード完了 $(du -h ${DATA_PATH} | cut -f1)" \
              || echo "[ERROR] CSV ダウンロード失敗"
        else
            echo "[WARN] GDrive にも DATA_URL にも CSV がありません"
            echo "[WARN] $(basename ${DATA_PATH}) を GDrive フォルダにアップロードしてください"
        fi
    }
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── 7. stop.flag クリア ──────────────────────────────────────────────────────
rm -f /workspace/stop.flag

# ── 8. 学習ループ起動 ────────────────────────────────────────────────────────
echo ""
echo "[*] ランダムサーチ開始 (デバイス・並列数は run_train.py が自動検出)"
echo "    停止: http://<IP>:8080 → 学習停止ボタン"
echo "    または: touch /workspace/stop.flag"
echo ""

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

python /workspace/ai_ea/run_train.py 2>&1 | tee /workspace/train_run.log &
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "  学習完了!"
    echo "  ダッシュボード : http://<IP>:8080"
    echo "======================================================"
else
    echo "  [ERROR] 学習がエラーで終了 (exit=$EXIT_CODE)"
    echo "  ログ: /workspace/train_run.log"
fi

echo "[*] コンテナ待機中..."
wait
