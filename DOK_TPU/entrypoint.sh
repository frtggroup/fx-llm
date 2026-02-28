#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA TPU エントリポイント (Google Cloud TPU v4/v5/Trillium)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA ランダムサーチ on Google Cloud TPU"
echo "  PJRT_DEVICE=${PJRT_DEVICE:-TPU}  TPU_NUM_DEVICES=${TPU_NUM_DEVICES:-auto}"
echo "======================================================"

# ── 1. TPU 確認 (PJRT ランタイム) ────────────────────────────────────────────
echo "[*] TPU 確認 (PJRT_DEVICE=${PJRT_DEVICE:-TPU})..."
python -c "
import os
os.environ.setdefault('PJRT_DEVICE', 'TPU')
try:
    import torch_xla.core.xla_model as xm
    devs = xm.get_xla_supported_devices()
    print(f'[OK] TPU デバイス検出: {len(devs)} チップ')
    for d in devs:
        print(f'     {d}')
except Exception as e:
    print(f'[WARN] TPU 未検出: {e}')
    print('[INFO] CPU モードで続行します')
" 2>&1 || echo "[WARN] TPU 確認スクリプト失敗 (CPU モードで続行)"

# ── 2. SSH サーバー ────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
sleep 1
kill -0 "$SSH_PID" 2>/dev/null && echo "[OK] SSH 起動 (PID: $SSH_PID)" || echo "[WARN] SSH 起動失敗"

# ── 3. 環境変数 ───────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
# NODE_ID / MAX_PARALLEL / VRAM_PER_TRIAL は run_train.py が TPU から自動検出

# XLA コンパイルキャッシュ: 2回目以降のコンパイルを大幅高速化
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_persistent_cache=true"
export XLA_CACHE_DIR="/tmp/xla_cache"
mkdir -p /tmp/xla_cache
echo "[OK] XLA キャッシュ: /tmp/xla_cache"

echo "[*] 設定:"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(未設定)}"
echo "    TPU_NUM_DEV  : ${TPU_NUM_DEVICES:-自動}"
echo "    GPU設定      : run_train.py 起動後に TPU から自動検出"

# ── 4. CSV 自動ダウンロード: GDrive → DATA_URL ───────────────────────────
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

# ── 5. ダッシュボード起動 ────────────────────────────────────────────────
echo "[*] ダッシュボード起動 (port 8080)..."
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 3
kill -0 "$DASH_PID" 2>/dev/null \
    && echo "[OK] ダッシュボード起動 (PID: $DASH_PID)  -> http://0.0.0.0:8080" \
    || { echo "[WARN] ダッシュボード起動失敗"; tail -20 /workspace/dashboard.log 2>/dev/null; }

# ── 6. stop.flag クリア ─────────────────────────────────────────────────
rm -f /workspace/stop.flag

# ── 7. graceful shutdown ─────────────────────────────────────────────────
TRAIN_PID=""
_graceful_stop() {
    echo "[*] シグナル受信 → run_train.py に SIGTERM 送信..."
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && {
        kill -TERM "$TRAIN_PID"
        for i in $(seq 1 30); do kill -0 "$TRAIN_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    }
    echo "[OK] graceful shutdown 完了"
}
trap '_graceful_stop' SIGTERM SIGINT

# ── 8. 学習ループ起動 ────────────────────────────────────────────────────
echo ""
echo "[*] TPU ランダムサーチ開始  (TPU設定は run_train.py が自動検出)"
echo "    停止するには: http://<IP>:8080  →「学習停止」ボタン"
echo "    または:       touch /workspace/stop.flag"
echo ""

python /workspace/ai_ea/run_train.py 2>&1 | tee /workspace/train_run.log &
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
[ $EXIT_CODE -eq 0 ] \
    && echo "======================================================"  \
    && echo "  学習完了!  ダッシュボード: http://<IP>:8080" \
    && echo "======================================================"  \
    || echo "  [ERROR] 学習がエラーで終了しました (exit=$EXIT_CODE)"

echo "[*] コンテナ待機中..."
wait
