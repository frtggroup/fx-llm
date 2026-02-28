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
# PJRT_DEVICE は docker run -e PJRT_DEVICE=TPU 等で明示的に指定された場合はそれを尊重
# 未設定の場合のみ CUDA にデフォルト設定 (torch_xla が CPU にフォールバックするのを防ぐ)
if [ -z "${PJRT_DEVICE}" ]; then
    export PJRT_DEVICE=CUDA
fi

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
        # PJRT_DEVICE を確実に TPU に設定 (entrypoint の CUDA 上書きを打ち消す)
        export PJRT_DEVICE=TPU
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

# ── XLA コンパイルキャッシュ (TPU 専用) ──────────────────────────────────────
# XLA は初回コンパイル結果をファイルにキャッシュし、2回目以降は再利用する。
# S3 に永続化することでコンテナ/VM 再作成後もキャッシュが復元される。
if [ "$DEVICE_TYPE" = "TPU" ]; then
    export XLA_CACHE_DIR=/workspace/xla_cache
    mkdir -p "${XLA_CACHE_DIR}"
    # XLA_FLAGS に書くと torch_xla が GPU 専用フラグを追加してTPUでFatalクラッシュする。
    # 正しいTPU向けキャッシュ設定は XLA_PERSISTENT_CACHE_PATH 環境変数を使う。
    export XLA_PERSISTENT_CACHE_PATH="${XLA_CACHE_DIR}"
    echo "[*] XLA キャッシュ設定: ${XLA_CACHE_DIR}"
fi

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

# ─── XLA キャッシュ S3 アップロード ─────────────────────────────────────────
_xla_cache_upload() {
    [ "$DEVICE_TYPE" != "TPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    echo "[*] XLA キャッシュを S3 へ保存中..."
    python3 - <<'PYEOF'
import os, sys, pathlib
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ.get('XLA_CACHE_DIR', '/workspace/xla_cache'))
    files = [f for f in cache_dir.rglob('*') if f.is_file()]
    if not files:
        print('[INFO] XLA キャッシュ: ファイルなし (スキップ)')
        sys.exit(0)
    s3 = boto3.client('s3',
        endpoint_url=os.environ.get('S3_ENDPOINT', ''),
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
        aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
        verify=False)
    bucket    = os.environ.get('S3_BUCKET',  'fxea')
    s3_prefix = os.environ.get('S3_PREFIX',  'mix') + '/xla_cache'
    count = 0
    for f in files:
        rel = f.relative_to(cache_dir)
        s3.upload_file(str(f), bucket, f'{s3_prefix}/{rel}')
        count += 1
    print(f'[OK] XLA キャッシュ S3 保存: {count}件 → s3://{bucket}/{s3_prefix}/')
except Exception as e:
    print(f'[WARN] XLA キャッシュ S3 保存失敗: {e}')
PYEOF
}

# ─── XLA キャッシュ S3 ダウンロード ─────────────────────────────────────────
_xla_cache_download() {
    [ "$DEVICE_TYPE" != "TPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    echo "[*] XLA キャッシュを S3 から復元中 (並列16スレッド)..."
    python3 - <<'PYEOF'
import os, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ.get('XLA_CACHE_DIR', '/workspace/xla_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket    = os.environ.get('S3_BUCKET',  'fxea')
    s3_prefix = os.environ.get('S3_PREFIX',  'mix') + '/xla_cache/'

    def make_client():
        return boto3.client('s3',
            endpoint_url=os.environ.get('S3_ENDPOINT', ''),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
            verify=False)

    # ファイル一覧を取得
    s3 = make_client()
    paginator = s3.get_paginator('list_objects_v2')
    tasks = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel = key[len(s3_prefix):]
            if not rel:
                continue
            dst = cache_dir / rel
            # 既存ファイルはスキップ (サイズ一致チェック)
            if dst.exists() and dst.stat().st_size == obj['Size']:
                continue
            tasks.append((key, dst, obj['Size']))

    if not tasks:
        print(f'[OK] XLA キャッシュ: 全ファイル既存またはS3未存在 (スキップ)')
    else:
        print(f'[*] XLA キャッシュ: {len(tasks)}件をダウンロード中...', flush=True)
        def download(args):
            key, dst, _ = args
            dst.parent.mkdir(parents=True, exist_ok=True)
            make_client().download_file(bucket, key, str(dst))
            return dst.name

        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(download, t): t for t in tasks}
            for f in as_completed(futs):
                try:
                    f.result()
                    done += 1
                    if done % 50 == 0:
                        print(f'  ... {done}/{len(tasks)}', flush=True)
                except Exception as e:
                    print(f'  [WARN] DL失敗: {e}')
        print(f'[OK] XLA キャッシュ復元完了: {done}/{len(tasks)}件')
except Exception as e:
    print(f'[INFO] XLA キャッシュ復元スキップ: {e}')
PYEOF
}

_graceful_stop() {
    echo "[*] 停止シグナル受信..."
    _STOP_REQUESTED=1
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -TERM "$TRAIN_PID"
    sleep 5
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    # XLA キャッシュを最終アップロード
    _xla_cache_upload
    [ -n "$XLA_SYNC_PID" ] && kill "$XLA_SYNC_PID" 2>/dev/null || true
    echo "[OK] 停止完了"
}
trap '_graceful_stop' SIGTERM SIGINT

# XLA キャッシュを S3 から復元 (TPU のみ / 失敗しても続行)
_xla_cache_download || true

# warmup 進捗 JSON を S3 から復元 (TPU のみ / 再起動時にスキップ判定に使用)
if [ "$DEVICE_TYPE" = "TPU" ] && [ -n "$S3_ENDPOINT" ]; then
    python3 - <<'PYEOF'
import os, sys, pathlib
try:
    import boto3, urllib3; urllib3.disable_warnings()
    s3 = boto3.client('s3',
        endpoint_url=os.environ.get('S3_ENDPOINT',''),
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY',''),
        aws_secret_access_key=os.environ.get('S3_SECRET_KEY',''),
        verify=False)
    bucket = os.environ.get('S3_BUCKET','fxea')
    prefix = os.environ.get('S3_PREFIX','mix') + '/warmup_progress/'
    paginator = s3.get_paginator('list_objects_v2')
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']; rel = key[len(prefix):]
            if not rel: continue
            dst = pathlib.Path('/workspace') / rel
            s3.download_file(bucket, key, str(dst))
            count += 1
    if count: print(f'[OK] warmup 進捗復元: {count}件')
    else: print('[INFO] warmup 進捗: S3 にまだありません (初回)')
except Exception as e:
    print(f'[INFO] warmup 進捗復元スキップ: {e}')
PYEOF
fi

# XLA キャッシュ定期アップロード (10分ごと、バックグラウンド)
XLA_SYNC_PID=""
if [ "$DEVICE_TYPE" = "TPU" ] && [ -n "$S3_ENDPOINT" ]; then
    (while true; do sleep 600; _xla_cache_upload; done) &
    XLA_SYNC_PID=$!
    echo "[*] XLA キャッシュ自動同期 開始 (10分ごと, PID: ${XLA_SYNC_PID})"
fi

# ── XLA 全パターン事前コンパイル (TPU のみ / キャッシュ未完了時のみ) ───────────
if [ "$DEVICE_TYPE" = "TPU" ]; then
    # xmp.spawn で TPU_NUM_DEVICES 枚を並列コンパイル (シングルチップ時はシングル実行)
    python3 /workspace/ai_ea/warmup_xla.py 2>&1 | tee -a /workspace/train_run.log
    # warmup 後にキャッシュと進捗 JSON を S3 へ保存
    _xla_cache_upload || true
    if [ -n "$S3_ENDPOINT" ]; then
        python3 - <<'PYEOF'
import os, sys, pathlib
try:
    import boto3, urllib3; urllib3.disable_warnings()
    s3 = boto3.client('s3',
        endpoint_url=os.environ.get('S3_ENDPOINT',''),
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY',''),
        aws_secret_access_key=os.environ.get('S3_SECRET_KEY',''),
        verify=False)
    bucket = os.environ.get('S3_BUCKET','fxea')
    prefix = os.environ.get('S3_PREFIX','mix') + '/warmup_progress'
    count = 0
    for f in pathlib.Path('/workspace').glob('xla_warmup_rank_*.json'):
        s3.upload_file(str(f), bucket, f'{prefix}/{f.name}')
        count += 1
    if count: print(f'[OK] warmup 進捗 S3 保存: {count}件')
except Exception as e:
    print(f'[WARN] warmup 進捗 S3 保存失敗: {e}')
PYEOF
    fi
fi

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
    # クラッシュログがあれば末尾を表示
    if [ -f /workspace/crash.log ]; then
        echo "--- crash.log (末尾20行) ---"
        tail -20 /workspace/crash.log
        echo "----------------------------"
    fi
    sleep 5
done

echo "[*] コンテナ待機中..."
wait
