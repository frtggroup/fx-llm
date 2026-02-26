#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX LLM H100 コンテナ エントリポイント
# 実行順序:
#   1. UFW ファイアウォール設定
#   2. SSH サーバー起動
#   3. ダッシュボードサーバー起動 (バックグラウンド)
#   4. パイプライン実行 (データセット → 訓練 → バックテスト)
# ─────────────────────────────────────────────────────────────────────────────
set -e
# UFW/sysctl などの非致命的エラーは無視するヘルパー
ignore_err() { "$@" || true; }

echo "======================================================"
echo "  FX LLM Fine-tuning on Sakura DOK / H100 80GB"
echo "======================================================"

# ── 1. UFW ────────────────────────────────────────────────────────────────────
# DOK コンテナは iptables 権限がないため UFW はスキップ
# ポート管理は DOK の設定画面 (HTTP:7860, SSH:有効) で行う
echo "[*] UFW: DOK コンテナでは不要 (DOKがポート管理) → スキップ"

# ── 2. SSH サーバー ───────────────────────────────────────────────────────────
echo "[*] SSH サーバー起動..."
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
# authorized_keys が既に COPY されている場合はそのまま使用
if [ ! -f /root/.ssh/authorized_keys ] || [ ! -s /root/.ssh/authorized_keys ]; then
    echo "[WARN] authorized_keys が空です。SSH公開鍵を設定してください。"
    echo "[WARN] docker exec <container> sh -c 'echo \"<pubkey>\" >> /root/.ssh/authorized_keys'"
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
echo "[OK] SSH サーバー起動 (PID: $SSH_PID)"

# ── 3. 環境変数 / パス ────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:/workspace/src:${PYTHONPATH}"
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
mkdir -p /workspace/data /workspace/output /workspace/reports /workspace/hf_cache

# GPU確認
echo "[*] GPU 確認..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 使用不可"

# ── 4. ダッシュボード起動 (バックグラウンド) ──────────────────────────────────
echo "[*] ダッシュボード起動 (port 7860)..."
python /workspace/src/dashboard_server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 2
echo "[OK] ダッシュボード起動 (PID: $DASH_PID)"
echo "     → http://0.0.0.0:7860"

# ── 5. パイプライン引数 ───────────────────────────────────────────────────────
# 環境変数で上書き可能
MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-8B}"
EPOCHS="${LLM_EPOCHS:-10}"
BATCH="${LLM_BATCH:-8}"
GRAD_ACCUM="${LLM_GRAD_ACCUM:-8}"
LORA_R="${LLM_LORA_R:-64}"
LORA_ALPHA="${LLM_LORA_ALPHA:-128}"
MAX_LENGTH="${LLM_MAX_LENGTH:-1024}"
LR="${LLM_LR:-5e-5}"
SKIP_DATASET="${LLM_SKIP_DATASET:-}"
RESUME="${LLM_RESUME:-}"

echo "[*] パイプライン設定:"
echo "    モデル      : ${MODEL_ID}"
echo "    エポック数  : ${EPOCHS}"
echo "    バッチサイズ: ${BATCH} x ${GRAD_ACCUM} = $((BATCH * GRAD_ACCUM)) (実効)"
echo "    LoRA rank   : ${LORA_R}  alpha: ${LORA_ALPHA}"
echo "    max_length  : ${MAX_LENGTH}"
echo "    学習率      : ${LR}"

PIPELINE_ARGS=(
    "--model_id"   "${MODEL_ID}"
    "--epochs"     "${EPOCHS}"
    "--batch"      "${BATCH}"
    "--grad_accum" "${GRAD_ACCUM}"
    "--lora_r"     "${LORA_R}"
    "--lora_alpha" "${LORA_ALPHA}"
    "--max_length" "${MAX_LENGTH}"
    "--lr"         "${LR}"
)
[ -n "${SKIP_DATASET}" ] && PIPELINE_ARGS+=("--skip_dataset")
[ -n "${RESUME}"       ] && PIPELINE_ARGS+=("--resume")

# ── 6. パイプライン実行 ───────────────────────────────────────────────────────
echo ""
echo "[*] パイプライン開始..."
python /workspace/src/pipeline.py "${PIPELINE_ARGS[@]}"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "  全工程完了！"
    echo "  ダッシュボード: http://<DOK_IP>:7860"
    echo "  レポートDL:     http://<DOK_IP>:7860/download/report"
    echo "  モデルDL:       http://<DOK_IP>:7860/download/adapter"
    echo "======================================================"
else
    echo "  [ERROR] パイプラインがエラーで終了しました (exit=$EXIT_CODE)"
fi

# コンテナを終了させずに待機 (ダッシュボードとSSHを維持)
echo "[*] コンテナ待機中 (ダッシュボード・SSH は継続稼働)..."
wait
