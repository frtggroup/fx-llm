#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA GTX版 エントリポイント
# H100版と違い /opt/artifact は使わず、ボリュームマウントで永続化
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA ランダムサーチ on GTX 1080 Ti"
echo "======================================================"

# ── GPU 確認 ────────────────────────────────────────────────────────────────
echo "[*] GPU 確認..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
  && echo "[OK] GPU 検出" \
  || echo "[WARN] nvidia-smi 使用不可 (CPU モードで続行)"

# ── SSH サーバー ────────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
echo "[OK] SSH サーバー起動 (PID: $SSH_PID)"

# ── 環境変数 / パス ──────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export H100_MODE="${H100_MODE:-0}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export VRAM_PER_TRIAL="${VRAM_PER_TRIAL:-8}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_M1.csv}"

echo "[*] 設定:"
echo "    H100_MODE    : ${H100_MODE} (GTX=0)"
echo "    MAX_PARALLEL : ${MAX_PARALLEL} 並列"
echo "    VRAM/試行    : ${VRAM_PER_TRIAL} GB"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    S3_ENDPOINT  : ${S3_ENDPOINT:-未設定}"

# ── CSV データ自動ダウンロード ──────────────────────────────────────────────
if [ ! -f "${DATA_PATH}" ] || [ ! -s "${DATA_PATH}" ]; then
    if [ -n "${DATA_URL}" ]; then
        echo "[*] CSV ダウンロード中: ${DATA_URL}"
        mkdir -p "$(dirname ${DATA_PATH})"
        wget -q --show-progress -O "${DATA_PATH}" "${DATA_URL}" \
          && echo "[OK] CSV ダウンロード完了 $(du -h ${DATA_PATH} | cut -f1)" \
          || echo "[ERROR] CSV ダウンロード失敗"
    else
        echo "[WARN] DATA_URL 未設定 — ${DATA_PATH} を手動でコピーしてください"
    fi
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── ダッシュボードサーバー起動 ────────────────────────────────────────────
echo "[*] ダッシュボード起動 (port 7860)..."
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
echo "[*] ランダムサーチ開始  (並列=${MAX_PARALLEL}  VRAM/試行=${VRAM_PER_TRIAL}GB)"
echo ""
python /workspace/ai_ea/run_train.py 2>&1 | tee /workspace/train_run.log &
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "  学習完了!"
    echo "  ダッシュボード : http://localhost:7860"
    echo "======================================================"
else
    echo "  [ERROR] 学習がエラーで終了 (exit=$EXIT_CODE)"
fi

echo "[*] コンテナ待機中..."
wait
