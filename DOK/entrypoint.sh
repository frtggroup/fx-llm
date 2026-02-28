#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA 統合エントリポイント
# オプション不要 - GPU/TPU/CPU を Python で完全自動検出
# 対応: Vast.ai / Sakura DOK / Google Cloud / ローカル / TPU VM
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================================"
echo "  FX AI EA 並列ランダムサーチ (統合イメージ)"
echo "======================================================"

# ── 0a. NTP 時刻同期 (S3 RequestTimeTooSkewed 防止) ──────────────────────────
# S3 署名検証は±15分以内の時刻一致が必要。コンテナ起動時にクロックを同期する。
if command -v ntpdate &>/dev/null; then
    ntpdate -u pool.ntp.org &>/dev/null && echo "[*] NTP 同期完了 (ntpdate)" || true
elif command -v chronyc &>/dev/null; then
    chronyc makestep &>/dev/null && echo "[*] NTP 同期完了 (chronyc)" || true
fi

# ── 0b. torch_xla が CUDA_VISIBLE_DEVICES を空にするのを防ぐ ──────────────────
# torch_xla はインポート時に CUDA_VISIBLE_DEVICES="" を設定する場合がある。
# デバイス検出前にリセットして GPU が見えるようにする。
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ "${CUDA_VISIBLE_DEVICES}" = "" ]; then
    if [ -n "${NVIDIA_VISIBLE_DEVICES}" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "none" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "[*] CUDA_VISIBLE_DEVICES を 0 にリセット (torch_xla 干渉防止)"
    fi
fi
export PJRT_DEVICE=CUDA   # torch_xla が CPU にフォールバックするのを防ぐ

# ── 1. デバイス自動検出 (--gpus / --privileged 不要) ─────────────────────────
echo "[*] デバイス検出中..."

DEVICE_INFO=$(python3 - <<'PYEOF'
import os, sys, subprocess

# ── 1. nvidia-smi で先に確認 (torch より先に実行 = torch_xla 干渉を回避) ──────
try:
    r = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10)
    if r.returncode == 0 and r.stdout.strip():
        parts = r.stdout.strip().split(',')
        name = parts[0].strip()
        vram = round(float(parts[1].strip()) / 1024, 1) if len(parts) > 1 else 0
        print(f"GPU|{name}|{vram}")
        sys.exit(0)
except Exception:
    pass

# ── 2. NVIDIA_VISIBLE_DEVICES チェック (Vast.ai 等はこれで確認できる) ────────
nv = os.environ.get('NVIDIA_VISIBLE_DEVICES', '')
if nv and nv not in ('none', 'void', 'NoDevFiles'):
    # PJRT_DEVICE=CUDA を設定して torch_xla が CPU にデフォルトしないようにする
    os.environ['PJRT_DEVICE'] = 'CUDA'
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            print(f"GPU|{name}|{vram}")
            sys.exit(0)
    except Exception:
        pass
    # torch.cuda が使えない場合: nvidia-smi を再試行 (引数なしで存在確認)
    try:
        r2 = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10)
        if r2.returncode == 0 and r2.stdout.strip():
            parts = r2.stdout.strip().split(',')
            name2 = parts[0].strip()
            vram2 = round(float(parts[1].strip()) / 1024, 1) if len(parts) > 1 else 0
            print(f"GPU|{name2}|{vram2}")
            sys.exit(0)
    except Exception:
        pass
    # GPU 確認失敗 → TPU VM 上で NVIDIA_VISIBLE_DEVICES=all が誤設定されている可能性
    # fall-through して TPU チェックに進む (Unknown GPU として終了しない)

# ── 3. torch.cuda 直接確認 (NVIDIA_VISIBLE_DEVICES がない環境向け) ──────────
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"GPU|{name}|{vram}")
        sys.exit(0)
except Exception:
    pass

# ── 4. TPU チェック (物理デバイス確認必須 / torch_xla importだけでは不十分) ──
tpu_hw = (os.path.exists('/dev/accel0') or
          os.path.exists('/dev/vfio/0')  or
          bool(os.environ.get('TPU_NAME')) or
          bool(os.environ.get('TPU_ACCELERATOR_TYPE')) or
          bool(os.environ.get('COLAB_TPU_ADDR')))
if tpu_hw:
    tpu_type = os.environ.get('TPU_ACCELERATOR_TYPE',
               os.environ.get('TPU_NAME', 'TPU'))
    print(f"TPU|TPU ({tpu_type})|0")
    sys.exit(0)

# ── 5. CPU ───────────────────────────────────────────────────────────────────
print("CPU|CPU|0")
PYEOF
)

DEVICE_TYPE=$(echo "$DEVICE_INFO" | cut -d'|' -f1)
GPU_NAME=$(echo "$DEVICE_INFO"    | cut -d'|' -f2)
GPU_VRAM=$(echo "$DEVICE_INFO"    | cut -d'|' -f3)

case "$DEVICE_TYPE" in
    GPU)
        echo "[OK] GPU 検出: ${GPU_NAME} (${GPU_VRAM} GB)"
        ;;
    TPU)
        echo "[OK] TPU 検出: ${GPU_NAME}"
        # torch_xla 確認 / 未インストールならフォールバックインストール
        if python3 -c "import torch_xla" 2>/dev/null; then
            echo "[OK] torch_xla 利用可能"
        else
            echo "[*] torch_xla インストール中..."
            TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.5.0")
            pip install --no-cache-dir \
                "torch_xla==${TORCH_VER}" \
                -f https://storage.googleapis.com/libtpu-releases/index.html \
            && echo "[OK] torch_xla インストール完了" \
            || echo "[WARN] torch_xla インストール失敗 — CPU モードで続行"
        fi
        ;;
    *)
        echo "[WARN] GPU/TPU 未検出 — CPU モードで続行"
        ;;
esac

# Python に デバイス名を渡す
export GPU_NAME
export DEVICE_TYPE
export GPU_VRAM

# ── 2. CUDA MPS (特権不要・失敗しても続行) ───────────────────────────────────
if [ "$DEVICE_TYPE" = "GPU" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
    nvidia-cuda-mps-control -d 2>/dev/null \
      && echo "[OK] CUDA MPS 起動完了" \
      || true   # 特権なし環境では失敗するが無視
fi

# ── 3. SSH サーバー ──────────────────────────────────────────────────────────
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
SSH_PID=$!
sleep 1
kill -0 "$SSH_PID" 2>/dev/null \
  && echo "[OK] SSH サーバー起動 (PID: $SSH_PID)" \
  || echo "[WARN] SSH 起動失敗 (続行)"

# ── 4. 永続ストレージ ─────────────────────────────────────────────────────────
ARTIFACT=/opt/artifact
if [ -d "${ARTIFACT}" ] || [ -b "${ARTIFACT}" ]; then
    echo "[*] Sakura DOK モード: /opt/artifact を使用"
    mkdir -p "${ARTIFACT}/data" "${ARTIFACT}/ai_ea/trials" \
             "${ARTIFACT}/ai_ea/top100" "${ARTIFACT}/ai_ea/top_cache"
    [ ! -L /workspace/data ]             && rm -rf /workspace/data             && ln -sf "${ARTIFACT}/data"              /workspace/data
    [ ! -L /workspace/ai_ea/trials ]     && rm -rf /workspace/ai_ea/trials     && ln -sf "${ARTIFACT}/ai_ea/trials"      /workspace/ai_ea/trials
    [ ! -L /workspace/ai_ea/top100 ]     && rm -rf /workspace/ai_ea/top100     && ln -sf "${ARTIFACT}/ai_ea/top100"      /workspace/ai_ea/top100
    [ ! -L /workspace/ai_ea/top_cache ]  && rm -rf /workspace/ai_ea/top_cache  && ln -sf "${ARTIFACT}/ai_ea/top_cache"   /workspace/ai_ea/top_cache
    export TORCHINDUCTOR_CACHE_DIR="${ARTIFACT}/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[OK] Sakura DOK ストレージ設定完了"
else
    echo "[*] クラウド/ローカルモード: /workspace を使用"
    mkdir -p /workspace/data /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 /workspace/ai_ea/top_cache
fi

# ── 5. 環境変数 ──────────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/ai_ea:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo "[*] 設定:"
echo "    デバイス     : ${DEVICE_TYPE} / ${GPU_NAME}"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(未設定)}"
echo "    DASHBOARD    : port ${DASHBOARD_PORT}"

# ── 6. ダッシュボード起動 ─────────────────────────────────────────────────────
python /workspace/ai_ea/server.py > /workspace/dashboard.log 2>&1 &
DASH_PID=$!
sleep 3
kill -0 "$DASH_PID" 2>/dev/null \
  && echo "[OK] ダッシュボード起動 port ${DASHBOARD_PORT} (PID: $DASH_PID)" \
  || { echo "[WARN] ダッシュボード起動失敗:"; cat /workspace/dashboard.log 2>/dev/null | tail -10 || true; }

# ── 7. CSV 自動取得 ───────────────────────────────────────────────────────────
mkdir -p "$(dirname ${DATA_PATH})"
if [ ! -s "${DATA_PATH}" ]; then
    python3 - <<'PYEOF'
import sys, os, urllib.request, ssl
sys.path.insert(0, '/workspace/ai_ea')
from pathlib import Path

dst = Path(os.environ.get('DATA_PATH', '/workspace/data/USDJPY_H1.csv'))

# 方法1: S3 直接URL (最優先・高速)
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://frorit-2022.softether.net:18004')
S3_BUCKET   = os.environ.get('S3_BUCKET',   'fxea')
S3_PREFIX   = os.environ.get('S3_PREFIX',   'mix')
s3_url = f'{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}/data/USDJPY_H1.csv'
print(f'[*] S3 から CSV 取得中: {s3_url}')
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(s3_url, context=ctx, timeout=30) as resp:
        data = resp.read()
    if len(data) > 100000:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f'[OK] S3 CSV 取得完了 ({len(data)/1e6:.1f} MB)')
        sys.exit(0)
    print(f'[WARN] S3 レスポンスが小さすぎる ({len(data)} bytes)')
except Exception as e:
    print(f'[WARN] S3 取得失敗: {e}')

print('[ERROR] S3 CSV 取得失敗'); sys.exit(1)
PYEOF
    STATUS=$?
    if [ $STATUS -ne 0 ] && [ -n "${DATA_URL}" ]; then
        echo "[*] DATA_URL からダウンロード中..."
        wget -q -O "${DATA_PATH}" "${DATA_URL}" \
          && echo "[OK] CSV ダウンロード完了" \
          || echo "[ERROR] CSV ダウンロード失敗"
    fi
else
    echo "[*] CSV 既存: $(du -h ${DATA_PATH} | cut -f1)"
fi

# ── 8. 学習ループ起動 ─────────────────────────────────────────────────────────
rm -f /workspace/stop.flag
echo ""
echo "[*] 並列ランダムサーチ開始"
echo "    ダッシュボード: http://0.0.0.0:${DASHBOARD_PORT}"
echo ""

_STOP_REQUESTED=0

_graceful_stop() {
    echo "[*] 停止シグナル受信..."
    _STOP_REQUESTED=1
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -TERM "$TRAIN_PID"
    sleep 5
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    echo "[OK] 停止完了"
}
trap '_graceful_stop' SIGTERM SIGINT

# ── 自動再起動ループ ──────────────────────────────────────────────────────────
# run_train.py がクラッシュしても自動復旧する。stop.flag があれば再起動しない。
RESTART_COUNT=0
while true; do
    python /workspace/ai_ea/run_train.py 2>&1 | tee -a /workspace/train_run.log &
    TRAIN_PID=$!
    wait $TRAIN_PID
    EXIT_CODE=$?

    # stop.flag または SIGTERM/SIGINT があれば終了
    if [ "$_STOP_REQUESTED" -eq 1 ] || [ -f /workspace/stop.flag ]; then
        echo "===== 学習完了 | ダッシュボード: http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    # 正常終了も終了
    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== 学習完了 | ダッシュボード: http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[RESTART #${RESTART_COUNT}] run_train.py 異常終了 (exit=${EXIT_CODE}) → 5秒後に再起動..."
    sleep 5
done

echo "[*] コンテナ待機中..."
wait
