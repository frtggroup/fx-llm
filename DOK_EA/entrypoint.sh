#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA H100 コンテナ エントリポイント v3
# 1. SSH サーバー起動
# 2. /opt/artifact を永続ストレージとしてセットアップ (Sakura DOK)
# 3. ダッシュボードサーバー起動 (port 7860)
# 4. CSV データ自動ダウンロード (DATA_URL が設定されている場合)
# 5. 並列ランダムサーチ学習ループ起動 (run_train.py)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA 並列ランダムサーチ on Sakura DOK / H100"
echo "======================================================"

# ── 1. GPU 確認 ────────────────────────────────────────────────────────────
echo "[*] GPU 確認..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
  && echo "[OK] GPU 検出" \
  || echo "[WARN] nvidia-smi 使用不可 (CPU モードで続行)"

# ── 2. SSH サーバー ────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ ! -s /root/.ssh/authorized_keys ]; then
    echo "[WARN] authorized_keys が空です"
    echo "[WARN] docker exec <container> sh -c 'echo \"<pubkey>\" >> /root/.ssh/authorized_keys'"
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
echo "[OK] SSH サーバー起動 (PID: $SSH_PID)"

# ── 3. /opt/artifact 永続ストレージのセットアップ (Sakura DOK) ──────────────
echo "[*] /opt/artifact 永続ストレージをセットアップ..."
ARTIFACT=/opt/artifact

# 永続ディレクトリ作成
mkdir -p "${ARTIFACT}/data" \
         "${ARTIFACT}/ai_ea/trials" \
         "${ARTIFACT}/ai_ea/top100" \
         "${ARTIFACT}/ai_ea/top_cache" \
         "${ARTIFACT}/checkpoint"

# /workspace/data → /opt/artifact/data シンボリックリンク
if [ ! -L /workspace/data ]; then
    rm -rf /workspace/data
    ln -sf "${ARTIFACT}/data" /workspace/data
    echo "[OK] /workspace/data → ${ARTIFACT}/data"
fi

# 学習結果ディレクトリも永続化
for d in trials top100 top_cache; do
    if [ ! -L "/workspace/ai_ea/${d}" ]; then
        rm -rf "/workspace/ai_ea/${d}"
        ln -sf "${ARTIFACT}/ai_ea/${d}" "/workspace/ai_ea/${d}"
        echo "[OK] /workspace/ai_ea/${d} → ${ARTIFACT}/ai_ea/${d}"
    fi
done

# チェックポイント
if [ ! -L /workspace/data/checkpoint ]; then
    ln -sf "${ARTIFACT}/checkpoint" /workspace/data/checkpoint 2>/dev/null || true
fi

echo "[OK] 永続ストレージセットアップ完了"
df -h "${ARTIFACT}" | tail -1

# ── 4. 環境変数 / パス ──────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export H100_MODE="${H100_MODE:-1}"
export MAX_PARALLEL="${MAX_PARALLEL:-3}"
export VRAM_PER_TRIAL="${VRAM_PER_TRIAL:-10}"
export DATA_PATH="${DATA_PATH:-/opt/artifact/data/USDJPY_M1.csv}"

echo "[*] 設定:"
echo "    H100_MODE    : ${H100_MODE}"
echo "    MAX_PARALLEL : ${MAX_PARALLEL} 並列"
echo "    VRAM/試行    : ${VRAM_PER_TRIAL} GB"
echo "    DATA_PATH    : ${DATA_PATH}"

# ── 5. CSV データ自動ダウンロード ──────────────────────────────────────────
if [ ! -f "${DATA_PATH}" ] || [ ! -s "${DATA_PATH}" ]; then
    if [ -n "${DATA_URL}" ]; then
        echo "[*] CSV ダウンロード中: ${DATA_URL}"
        wget -q --show-progress -O "${DATA_PATH}" "${DATA_URL}" \
          && echo "[OK] CSV ダウンロード完了 $(du -h ${DATA_PATH} | cut -f1)" \
          || echo "[ERROR] CSV ダウンロード失敗 — /workspace/data に手動でコピーしてください"
    else
        echo "[WARN] DATA_URL 未設定"
        echo "[WARN] ${DATA_PATH} を手動で配置してください"
        echo "[WARN] docker cp USDJPY_M1.csv <container>:/workspace/data/"
    fi
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── 6. ダッシュボードサーバー起動 (バックグラウンド) ─────────────────────
echo "[*] ダッシュボード起動 (port 7860)..."
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 2
if kill -0 "$DASH_PID" 2>/dev/null; then
    echo "[OK] ダッシュボード起動 (PID: $DASH_PID)"
    echo "     -> http://0.0.0.0:7860"
else
    echo "[WARN] ダッシュボード起動失敗 — ログ: /workspace/dashboard.log"
    cat /workspace/dashboard.log 2>/dev/null | tail -20
fi

# ── 7. stop.flag をクリア ──────────────────────────────────────────────────
rm -f /workspace/stop.flag

# ── 8. 並列ランダムサーチ学習ループ起動 ────────────────────────────────────
echo ""
echo "[*] 並列ランダムサーチ開始  (並列=${MAX_PARALLEL}  VRAM/試行=${VRAM_PER_TRIAL}GB)"
echo "    停止するには: http://<DOK_IP>:7860  →「学習停止」ボタン"
echo "    または:       touch /workspace/stop.flag"
echo ""
python /workspace/ai_ea/run_train.py 2>&1 | tee /workspace/train_run.log
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "  学習完了!"
    echo "  ダッシュボード : http://<DOK_IP>:7860"
    echo "  ベストモデル   : http://<DOK_IP>:7860/download/best"
    echo "  TOP100 DL      : http://<DOK_IP>:7860/download/model/1"
    echo "  全結果 JSON    : http://<DOK_IP>:7860/download/results"
    echo "======================================================"
else
    echo "  [ERROR] 学習がエラーで終了しました (exit=$EXIT_CODE)"
    echo "  ログ: /workspace/train_run.log"
fi

# コンテナを終了させずに待機 (ダッシュボード・SSH を継続)
echo "[*] コンテナ待機中 (ダッシュボード・SSH は継続稼働)..."
wait
