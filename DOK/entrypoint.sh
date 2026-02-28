#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA 統合エントリポイント
# GPU を自動検出し、Vast.ai / Sakura DOK / ローカル 全環境で動作します。
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA 並列ランダムサーチ (統合イメージ)"
echo "======================================================"

# ── 1. GPU / TPU 確認 + torch_xla 自動インストール ───────────────────────────
echo "[*] デバイス確認..."

# ── 1a. NVIDIA GPU チェック ──────────────────────────────────────────────────
if nvidia-smi --query-gpu=name,memory.total,driver_version \
              --format=csv,noheader 2>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "[OK] GPU 検出: ${GPU_NAME}"

# ── 1b. TPU チェック ─────────────────────────────────────────────────────────
# Google Cloud TPU: /dev/accel0 が存在するか、TPU 関連の環境変数が設定されている
elif [ -e /dev/accel0 ] \
  || [ -e /dev/vfio/0 ] \
  || [ -n "${TPU_NAME}" ] \
  || [ -n "${TPU_ACCELERATOR_TYPE}" ] \
  || [ -n "${COLAB_TPU_ADDR}" ]; then

    TPU_TYPE="${TPU_ACCELERATOR_TYPE:-${TPU_NAME:-TPU}}"
    echo "[OK] TPU 検出: ${TPU_TYPE}"

    # torch_xla がまだインストールされていなければ自動インストール
    if ! python3 -c "import torch_xla" 2>/dev/null; then
        echo "[*] torch_xla をインストール中 (初回のみ 2〜3 分かかります)..."
        # PyTorch バージョンに合わせた torch_xla を取得
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.5.0")
        pip install --no-cache-dir \
            "torch_xla[tpu]==${TORCH_VER}" \
            -f https://storage.googleapis.com/libtpu-releases/index.html \
        && echo "[OK] torch_xla インストール完了" \
        || echo "[WARN] torch_xla インストール失敗 — CPU モードで続行"
    else
        echo "[OK] torch_xla インストール済み"
    fi

    GPU_NAME="TPU (${TPU_TYPE})"

else
    GPU_NAME="CPU"
    echo "[WARN] GPU/TPU が検出できません (CPU モードで続行)"
fi

# Python に デバイス名を渡す (run_train.py の GPU_NAME 変数で使用)
export GPU_NAME

# ── 2. CUDA MPS デーモン起動 (GPU並列スループット向上) ─────────────────────
echo "[*] CUDA MPS デーモン起動..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
nvidia-cuda-mps-control -d 2>/dev/null \
  && echo "[OK] CUDA MPS 起動完了" \
  || echo "[WARN] CUDA MPS 起動失敗 (非特権モードまたは未対応GPU — 続行)"

# ── 3. SSH サーバー ──────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ ! -s /root/.ssh/authorized_keys ]; then
    echo "[WARN] authorized_keys が空です"
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
sleep 1
if kill -0 "$SSH_PID" 2>/dev/null; then
    echo "[OK] SSH サーバー起動 (PID: $SSH_PID)"
else
    echo "[WARN] SSH サーバー起動失敗 (続行)"
fi

# ── 4. 永続ストレージのセットアップ ─────────────────────────────────────────
# Sakura DOK: /opt/artifact が存在する場合はそこにシンボリックリンクを張る
# Vast.ai / ローカル: /workspace/ai_ea 直下を使用 (ボリュームマウントで永続化)
ARTIFACT=/opt/artifact
if [ -d "${ARTIFACT}" ] || [ -b "${ARTIFACT}" ]; then
    echo "[*] Sakura DOK モード: /opt/artifact を永続ストレージとして使用"
    mkdir -p "${ARTIFACT}/data" \
             "${ARTIFACT}/ai_ea/trials" \
             "${ARTIFACT}/ai_ea/top100" \
             "${ARTIFACT}/ai_ea/top_cache" \
             "${ARTIFACT}/checkpoint"

    [ ! -L /workspace/data ]          && rm -rf /workspace/data          && ln -sf "${ARTIFACT}/data"             /workspace/data
    [ ! -L /workspace/ai_ea/trials ]   && rm -rf /workspace/ai_ea/trials   && ln -sf "${ARTIFACT}/ai_ea/trials"    /workspace/ai_ea/trials
    [ ! -L /workspace/ai_ea/top100 ]   && rm -rf /workspace/ai_ea/top100   && ln -sf "${ARTIFACT}/ai_ea/top100"    /workspace/ai_ea/top100
    [ ! -L /workspace/ai_ea/top_cache ] && rm -rf /workspace/ai_ea/top_cache && ln -sf "${ARTIFACT}/ai_ea/top_cache" /workspace/ai_ea/top_cache

    export TORCHINDUCTOR_CACHE_DIR="${ARTIFACT}/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[OK] 永続ストレージセットアップ完了 (Sakura DOK)"
    df -h "${ARTIFACT}" | tail -1 || true
else
    echo "[*] Vast.ai / ローカルモード: /workspace を直接使用"
    mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache
fi

# ── 5. 環境変数 / パス ──────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
export DASHBOARD_PORT

echo "[*] 設定:"
echo "    GPU          : ${GPU_NAME}"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(未設定)}"
echo "    DASHBOARD    : port ${DASHBOARD_PORT}"
echo "    GPU設定      : run_train.py 起動後に自動検出"

# ── 6. ダッシュボードサーバー起動 ──────────────────────────────────────────
echo "[*] ダッシュボード起動 (port ${DASHBOARD_PORT})..."
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 3
if kill -0 "$DASH_PID" 2>/dev/null; then
    echo "[OK] ダッシュボード起動 (PID: $DASH_PID)"
    echo "     -> http://0.0.0.0:${DASHBOARD_PORT}"
else
    echo "[WARN] ダッシュボード起動失敗 — ログ:"
    tail -20 /workspace/dashboard.log 2>/dev/null || true
fi

# ── 7. CSV 自動ダウンロード: GDrive → DATA_URL の順 ─────────────────────────
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
              || echo "[ERROR] CSV ダウンロード失敗 — DATA_PATH を手動で配置してください"
        else
            echo "[WARN] CSV がありません: GDrive フォルダに USDJPY_H1.csv をアップロードするか"
            echo "[WARN] docker run -e DATA_URL=<url> で指定してください"
        fi
    }
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── 8. stop.flag をクリア ──────────────────────────────────────────────────
rm -f /workspace/stop.flag

# ── 9. 学習ループ起動 ────────────────────────────────────────────────────────
echo ""
echo "[*] 並列ランダムサーチ開始  (GPU・並列数は自動検出)"
echo "    停止するには: http://<IP>:${DASHBOARD_PORT}  →「学習停止」ボタン"
echo "    または:       touch /workspace/stop.flag"
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
    echo "  ダッシュボード : http://<IP>:${DASHBOARD_PORT}"
    echo "  ベストモデル   : http://<IP>:${DASHBOARD_PORT}/download/best"
    echo "  TOP100 DL      : http://<IP>:${DASHBOARD_PORT}/download/model/1"
    echo "  全結果 JSON    : http://<IP>:${DASHBOARD_PORT}/download/results"
    echo "======================================================"
else
    echo "  [ERROR] 学習がエラーで終了しました (exit=${EXIT_CODE})"
    echo "  ログ: /workspace/train_run.log"
fi

echo "[*] コンテナ待機中 (ダッシュボード・SSH は継続稼働)..."
wait
