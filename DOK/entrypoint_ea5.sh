#!/bin/bash
# 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# FX AI EA 邨ｱ蜷医お繝ｳ繝医Μ繝昴う繝ｳ繝・# 繧ｪ繝励す繝ｧ繝ｳ荳崎ｦ・- GPU/TPU/CPU 繧・Python 縺ｧ螳悟・閾ｪ蜍墓､懷・
# 蟇ｾ蠢・ Vast.ai / Sakura DOK / Google Cloud / 繝ｭ繝ｼ繧ｫ繝ｫ / TPU VM
# 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
set -e

# 笏笏 繝ｪ繧ｽ繝ｼ繧ｹ蛻ｶ髯舌ｒ蜈ｨ鬆・岼MAX (FD譫ｯ貂・・繝励Ο繧ｻ繧ｹ謨ｰ繝ｻ繝｡繝｢繝ｪ遲峨・蛻ｶ髯占ｧ｣髯､) 笏笏笏笏笏笏笏笏笏笏笏
# sysctl 縺ｧ繧ｫ繝ｼ繝阪Ν繝代Λ繝｡繝ｼ繧ｿ繧呈怙螟ｧ蛹・sysctl -w fs.file-max=1048576          2>/dev/null || true
sysctl -w fs.nr_open=1048576           2>/dev/null || true
sysctl -w kernel.pid_max=4194304       2>/dev/null || true
sysctl -w kernel.threads-max=4194304   2>/dev/null || true
sysctl -w vm.max_map_count=1048576     2>/dev/null || true
sysctl -w kernel.msgmax=134217728      2>/dev/null || true
sysctl -w kernel.msgmnb=134217728      2>/dev/null || true
sysctl -w net.core.somaxconn=65535     2>/dev/null || true
sysctl -w net.core.netdev_max_backlog=65535 2>/dev/null || true

# prlimit 縺ｧ蜈ｨ繝ｪ繧ｽ繝ｼ繧ｹ繧樽AX (root迚ｹ讓ｩ繧ｳ繝ｳ繝・リ縺ｯDocker繝上・繝我ｸ企剞繧剃ｸ頑嶌縺榊庄閭ｽ)
_set_limit() { prlimit --"$1"="$2":"$2" --pid $$ 2>/dev/null || true; }
_set_limit nofile   1048576     # 繧ｪ繝ｼ繝励ΦFD謨ｰ
_set_limit nproc    4194304     # 繝励Ο繧ｻ繧ｹ/繧ｹ繝ｬ繝・ラ謨ｰ
_set_limit stack    unlimited   # 繧ｹ繧ｿ繝・け繧ｵ繧､繧ｺ
_set_limit memlock  unlimited   # 繝ｭ繝・け蜿ｯ閭ｽ繝｡繝｢繝ｪ
_set_limit core     unlimited   # 繧ｳ繧｢繝繝ｳ繝励し繧､繧ｺ
_set_limit fsize    unlimited   # 譛螟ｧ繝輔ぃ繧､繝ｫ繧ｵ繧､繧ｺ
_set_limit data     unlimited   # 繝・・繧ｿ繧ｻ繧ｰ繝｡繝ｳ繝・_set_limit rss      unlimited   # 蟶ｸ鬧舌Γ繝｢繝ｪ
_set_limit as       unlimited   # 莉ｮ諠ｳ繧｢繝峨Ξ繧ｹ遨ｺ髢・_set_limit locks    unlimited   # 繝輔ぃ繧､繝ｫ繝ｭ繝・け謨ｰ
_set_limit sigpending 4194304   # 菫晉蕗繧ｷ繧ｰ繝翫Ν謨ｰ
_set_limit msgqueue 134217728   # 繝｡繝・そ繝ｼ繧ｸ繧ｭ繝･繝ｼ繝舌う繝域焚
_set_limit rtprio   99          # 繝ｪ繧｢繝ｫ繧ｿ繧､繝蜆ｪ蜈亥ｺｦ
_set_limit nice     -20         # nice蛟､縺ｮ荳矩剞

echo "[*] FD荳企剞: $(ulimit -n)  繝励Ο繧ｻ繧ｹ荳企剞: $(ulimit -u)"

echo "======================================================"
echo "  FX AI EA 荳ｦ蛻励Λ繝ｳ繝繝繧ｵ繝ｼ繝・(邨ｱ蜷医う繝｡繝ｼ繧ｸ)"
echo "======================================================"

# 笏笏 SSH 繧呈怙蜆ｪ蜈医〒襍ｷ蜍・(繝・ヰ繧､繧ｹ讀懷・繧医ｊ蜑・ 蛻晄悄蛹紋ｸｭ縺ｧ繧よ磁邯壹〒縺阪ｋ繧医≧縺ｫ) 笏笏笏笏笏笏
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
_SSH_PID=$!
sleep 1
kill -0 "$_SSH_PID" 2>/dev/null \
  && echo "[OK] SSH 繧ｵ繝ｼ繝舌・譌ｩ譛溯ｵｷ蜍・(PID: $_SSH_PID)" \
  || echo "[WARN] SSH 譌ｩ譛溯ｵｷ蜍募､ｱ謨・(邯夊｡・"

# 笏笏 0a. NTP 譎ょ綾蜷梧悄 (S3 RequestTimeTooSkewed 髦ｲ豁｢) 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# S3 鄂ｲ蜷肴､懆ｨｼ縺ｯﾂｱ15蛻・ｻ･蜀・・譎ょ綾荳閾ｴ縺悟ｿ・ｦ√ゅさ繝ｳ繝・リ襍ｷ蜍墓凾縺ｫ繧ｯ繝ｭ繝・け繧貞酔譛溘☆繧九・if command -v ntpdate &>/dev/null; then
    ntpdate -u pool.ntp.org &>/dev/null && echo "[*] NTP 蜷梧悄螳御ｺ・(ntpdate)" || true
elif command -v chronyc &>/dev/null; then
    chronyc makestep &>/dev/null && echo "[*] NTP 蜷梧悄螳御ｺ・(chronyc)" || true
fi

# 笏笏 0b. torch_xla 縺・CUDA_VISIBLE_DEVICES 繧堤ｩｺ縺ｫ縺吶ｋ縺ｮ繧帝亟縺・笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# torch_xla 縺ｯ繧､繝ｳ繝昴・繝域凾縺ｫ CUDA_VISIBLE_DEVICES="" 繧定ｨｭ螳壹☆繧句ｴ蜷医′縺ゅｋ縲・# 繝・ヰ繧､繧ｹ讀懷・蜑阪↓繝ｪ繧ｻ繝・ヨ縺励※ GPU 縺瑚ｦ九∴繧九ｈ縺・↓縺吶ｋ縲・if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ "${CUDA_VISIBLE_DEVICES}" = "" ]; then
    if [ -n "${NVIDIA_VISIBLE_DEVICES}" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "none" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "[*] CUDA_VISIBLE_DEVICES 繧・0 縺ｫ繝ｪ繧ｻ繝・ヨ (torch_xla 蟷ｲ貂蛾亟豁｢)"
    fi
fi
# PJRT_DEVICE 縺ｯ docker run -e PJRT_DEVICE=TPU 遲峨〒譏守､ｺ逧・↓謖・ｮ壹＆繧後◆蝣ｴ蜷医・縺昴ｌ繧貞ｰ企㍾
# 譛ｪ險ｭ螳壹・蝣ｴ蜷医・縺ｿ CUDA 縺ｫ繝・ヵ繧ｩ繝ｫ繝郁ｨｭ螳・(torch_xla 縺・CPU 縺ｫ繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ縺吶ｋ縺ｮ繧帝亟縺・
if [ -z "${PJRT_DEVICE}" ]; then
    export PJRT_DEVICE=CUDA
fi

# 笏笏 1. 繝・ヰ繧､繧ｹ閾ｪ蜍墓､懷・ (--gpus / --privileged 荳崎ｦ・ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
echo "[*] 繝・ヰ繧､繧ｹ讀懷・荳ｭ..."

DEVICE_INFO=$(python3 - <<'PYEOF' || echo "CPU|CPU|0"
import os, sys, subprocess

# 笏笏 1. nvidia-smi 縺ｧ蜈医↓遒ｺ隱・(torch 繧医ｊ蜈医↓螳溯｡・= torch_xla 蟷ｲ貂峨ｒ蝗樣∩) 笏笏笏笏笏笏
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

# 笏笏 2. NVIDIA_VISIBLE_DEVICES 繝√ぉ繝・け (Vast.ai 遲峨・縺薙ｌ縺ｧ遒ｺ隱阪〒縺阪ｋ) 笏笏笏笏笏笏笏笏
nv = os.environ.get('NVIDIA_VISIBLE_DEVICES', '')
if nv and nv not in ('none', 'void', 'NoDevFiles'):
    # PJRT_DEVICE=CUDA 繧定ｨｭ螳壹＠縺ｦ torch_xla 縺・CPU 縺ｫ繝・ヵ繧ｩ繝ｫ繝医＠縺ｪ縺・ｈ縺・↓縺吶ｋ
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
    # torch.cuda 縺御ｽｿ縺医↑縺・ｴ蜷・ nvidia-smi 繧貞・隧ｦ陦・(蠑墓焚縺ｪ縺励〒蟄伜惠遒ｺ隱・
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
    # GPU 遒ｺ隱榊､ｱ謨・竊・TPU VM 荳翫〒 NVIDIA_VISIBLE_DEVICES=all 縺瑚ｪ､險ｭ螳壹＆繧後※縺・ｋ蜿ｯ閭ｽ諤ｧ
    # fall-through 縺励※ TPU 繝√ぉ繝・け縺ｫ騾ｲ繧 (Unknown GPU 縺ｨ縺励※邨ゆｺ・＠縺ｪ縺・

# 笏笏 3. torch.cuda 逶ｴ謗･遒ｺ隱・(NVIDIA_VISIBLE_DEVICES 縺後↑縺・腸蠅・髄縺・ 笏笏笏笏笏笏笏笏笏笏
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"GPU|{name}|{vram}")
        sys.exit(0)
except Exception:
    pass

# 笏笏 4. TPU 繝√ぉ繝・け (迚ｩ逅・ョ繝舌う繧ｹ遒ｺ隱榊ｿ・・/ torch_xla import縺縺代〒縺ｯ荳榊香蛻・ 笏笏
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

# 笏笏 5. CPU 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
print("CPU|CPU|0")
PYEOF
)

DEVICE_TYPE=$(echo "$DEVICE_INFO" | cut -d'|' -f1)
GPU_NAME=$(echo "$DEVICE_INFO"    | cut -d'|' -f2)
GPU_VRAM=$(echo "$DEVICE_INFO"    | cut -d'|' -f3)

case "$DEVICE_TYPE" in
    GPU)
        echo "[OK] GPU 讀懷・: ${GPU_NAME} (${GPU_VRAM} GB)"
        ;;
    TPU)
        echo "[OK] TPU 讀懷・: ${GPU_NAME}"
        # PJRT_DEVICE 繧堤｢ｺ螳溘↓ TPU 縺ｫ險ｭ螳・(entrypoint 縺ｮ CUDA 荳頑嶌縺阪ｒ謇薙■豸医☆)
        export PJRT_DEVICE=TPU
        # torch_xla 遒ｺ隱・/ 譛ｪ繧､繝ｳ繧ｹ繝医・繝ｫ縺ｪ繧峨ヵ繧ｩ繝ｼ繝ｫ繝舌ャ繧ｯ繧､繝ｳ繧ｹ繝医・繝ｫ
        if python3 -c "import torch_xla" 2>/dev/null; then
            echo "[OK] torch_xla 蛻ｩ逕ｨ蜿ｯ閭ｽ"
        else
            echo "[*] torch_xla 繧､繝ｳ繧ｹ繝医・繝ｫ荳ｭ..."
            TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.5.0")
            pip install --no-cache-dir \
                "torch_xla==${TORCH_VER}" \
                -f https://storage.googleapis.com/libtpu-releases/index.html \
            && echo "[OK] torch_xla 繧､繝ｳ繧ｹ繝医・繝ｫ螳御ｺ・ \
            || echo "[WARN] torch_xla 繧､繝ｳ繧ｹ繝医・繝ｫ螟ｱ謨・窶・CPU 繝｢繝ｼ繝峨〒邯夊｡・
        fi
        ;;
    *)
        echo "[WARN] GPU/TPU 譛ｪ讀懷・ 窶・CPU 繝｢繝ｼ繝峨〒邯夊｡・
        ;;
esac

# 笏笏 XLA 繧ｳ繝ｳ繝代う繝ｫ繧ｭ繝｣繝・す繝･ (TPU 蟆ら畑) 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# XLA 縺ｯ蛻晏屓繧ｳ繝ｳ繝代う繝ｫ邨先棡繧偵ヵ繧｡繧､繝ｫ縺ｫ繧ｭ繝｣繝・す繝･縺励・蝗樒岼莉･髯阪・蜀榊茜逕ｨ縺吶ｋ縲・# S3 縺ｫ豌ｸ邯壼喧縺吶ｋ縺薙→縺ｧ繧ｳ繝ｳ繝・リ/VM 蜀堺ｽ懈・蠕後ｂ繧ｭ繝｣繝・す繝･縺悟ｾｩ蜈・＆繧後ｋ縲・if [ "$DEVICE_TYPE" = "TPU" ]; then
    # XLA繧ｭ繝｣繝・す繝･縺ｯ繝ｭ繝ｼ繧ｫ繝ｫSSD蜆ｪ蜈・(gcsfuse譖ｸ縺崎ｾｼ縺ｿI/O縺ｮ繝懊ヨ繝ｫ繝阪ャ繧ｯ繧貞屓驕ｿ)
    # 迺ｰ蠅・､画焚 XLA_PERSISTENT_CACHE_PATH 縺ｧ荳頑嶌縺榊庄閭ｽ
    export XLA_CACHE_DIR="${XLA_PERSISTENT_CACHE_PATH:-/workspace/local_xla}"
    mkdir -p "${XLA_CACHE_DIR}"
    # XLA_FLAGS 縺ｫ譖ｸ縺上→ torch_xla 縺・GPU 蟆ら畑繝輔Λ繧ｰ繧定ｿｽ蜉縺励※TPU縺ｧFatal繧ｯ繝ｩ繝・す繝･縺吶ｋ縲・    # 豁｣縺励＞TPU蜷代￠繧ｭ繝｣繝・す繝･險ｭ螳壹・ XLA_PERSISTENT_CACHE_PATH 迺ｰ蠅・､画焚繧剃ｽｿ縺・・    export XLA_PERSISTENT_CACHE_PATH="${XLA_CACHE_DIR}"
    echo "[*] XLA 繧ｭ繝｣繝・す繝･險ｭ螳・ ${XLA_CACHE_DIR}"
    # torch_xla 縺・LIBTPU_INIT_ARGS 縺ｫ髱槫ｯｾ蠢懊ヵ繝ｩ繧ｰ繧定ｿｽ蜉縺励※libtpu縺後け繝ｩ繝・す繝･縺吶ｋ縺ｮ繧帝亟縺・    # 遨ｺ譁・ｭ怜・繧偵そ繝・ヨ縺励※縺翫￥縺ｨ torch_xla 縺ｮ setdefault 縺御ｸ頑嶌縺阪＠縺ｪ縺・    export LIBTPU_INIT_ARGS=""
    echo "[*] LIBTPU_INIT_ARGS 繧偵け繝ｪ繧｢ (髱槫ｯｾ蠢懊ヵ繝ｩ繧ｰ髦ｲ豁｢)"
    # 繧ｹ繧ｫ繝ｩ繝ｼ蛟､繧偵す繝ｳ繝懊Ν縺ｨ縺励※迚ｹ蛻･謇ｱ縺・＠縺ｪ縺・竊・蜷御ｸHLO繧ｰ繝ｩ繝輔〒逡ｰ縺ｪ繧九せ繧ｫ繝ｩ繝ｼ繧貞・蛻ｩ逕ｨ縺・    # 荳崎ｦ√↑蜀阪さ繝ｳ繝代う繝ｫ繧帝亟縺・(torch_xla 2.x 謗ｨ螂ｨ險ｭ螳・
    export XLA_NO_SPECIAL_SCALARS=1
    echo "[*] XLA_NO_SPECIAL_SCALARS=1 險ｭ螳・(荳崎ｦ√↑蜀阪さ繝ｳ繝代う繝ｫ髦ｲ豁｢)"
fi

# Python 縺ｫ 繝・ヰ繧､繧ｹ蜷阪ｒ貂｡縺・export GPU_NAME
export DEVICE_TYPE
export GPU_VRAM

# 笏笏 2. CUDA MPS (迚ｹ讓ｩ荳崎ｦ√・螟ｱ謨励＠縺ｦ繧らｶ夊｡・ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
if [ "$DEVICE_TYPE" = "GPU" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
    nvidia-cuda-mps-control -d 2>/dev/null \
      && echo "[OK] CUDA MPS 襍ｷ蜍募ｮ御ｺ・ \
      || true   # 迚ｹ讓ｩ縺ｪ縺礼腸蠅・〒縺ｯ螟ｱ謨励☆繧九′辟｡隕・fi

# 笏笏 4. 豌ｸ邯壹せ繝医Ξ繝ｼ繧ｸ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
ARTIFACT=/opt/artifact
if [ -d "${ARTIFACT}" ] || [ -b "${ARTIFACT}" ]; then
    echo "[*] Sakura DOK 繝｢繝ｼ繝・ /opt/artifact 繧剃ｽｿ逕ｨ"
    mkdir -p "${ARTIFACT}/data" "${ARTIFACT}/fx-ea5/trials" \
             "${ARTIFACT}/fx-ea5/top100" "${ARTIFACT}/fx-ea5/top_cache"
    [ ! -L /workspace/data ]             && rm -rf /workspace/data             && ln -sf "${ARTIFACT}/data"              /workspace/data
    [ ! -L /workspace/fx-ea5/trials ]     && rm -rf /workspace/fx-ea5/trials     && ln -sf "${ARTIFACT}/fx-ea5/trials"      /workspace/fx-ea5/trials
    [ ! -L /workspace/fx-ea5/top100 ]     && rm -rf /workspace/fx-ea5/top100     && ln -sf "${ARTIFACT}/fx-ea5/top100"      /workspace/fx-ea5/top100
    [ ! -L /workspace/fx-ea5/top_cache ]  && rm -rf /workspace/fx-ea5/top_cache  && ln -sf "${ARTIFACT}/fx-ea5/top_cache"   /workspace/fx-ea5/top_cache
    export TORCHINDUCTOR_CACHE_DIR="${ARTIFACT}/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[OK] Sakura DOK 繧ｹ繝医Ξ繝ｼ繧ｸ險ｭ螳壼ｮ御ｺ・
else
    echo "[*] 繧ｯ繝ｩ繧ｦ繝・繝ｭ繝ｼ繧ｫ繝ｫ繝｢繝ｼ繝・ /workspace 繧剃ｽｿ逕ｨ"
    mkdir -p /workspace/data /workspace/fx-ea5/trials \
             /workspace/fx-ea5/top100 /workspace/fx-ea5/top_cache
    # torch.compile inductor 繧ｭ繝｣繝・す繝･繧・/workspace 縺ｫ豌ｸ邯壼喧
    # (繝・ヵ繧ｩ繝ｫ繝・~/.cache/torch/inductor/ 縺ｯ繧ｳ繝ｳ繝・リ蜀崎ｵｷ蜍輔〒豸医∴繧九◆繧・
    export TORCHINDUCTOR_CACHE_DIR="/workspace/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
fi

# 笏笏 5. 迺ｰ蠅・､画焚 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
export PYTHONPATH="/workspace/fx-ea5:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo "[*] 險ｭ螳・"
echo "    繝・ヰ繧､繧ｹ     : ${DEVICE_TYPE} / ${GPU_NAME}"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(譛ｪ險ｭ螳・}"
echo "    DASHBOARD    : port ${DASHBOARD_PORT}"

# 笏笏 6. 繝繝・す繝･繝懊・繝芽ｵｷ蜍・(繧ｯ繝ｩ繝・す繝･譎り・蜍募・襍ｷ蜍・ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
_start_dashboard() {
    while true; do
        echo "[DASH] server.py 襍ｷ蜍・$(date '+%H:%M:%S')" >> /workspace/dashboard.log
        python /workspace/fx-ea5/server.py >> /workspace/dashboard.log 2>&1
        EXIT_CODE=$?
        echo "[DASH] server.py 邨ゆｺ・(exit=$EXIT_CODE) 竊・5遘貞ｾ後↓蜀崎ｵｷ蜍・$(date '+%H:%M:%S')" \
            >> /workspace/dashboard.log
        sleep 5
    done
}
_start_dashboard &
DASH_PID=$!
sleep 3
curl -s --connect-timeout 3 http://127.0.0.1:${DASHBOARD_PORT}/api/status > /dev/null 2>&1 \
  && echo "[OK] 繝繝・す繝･繝懊・繝芽ｵｷ蜍・port ${DASHBOARD_PORT} (PID: $DASH_PID)" \
  || { echo "[WARN] 繝繝・す繝･繝懊・繝芽ｵｷ蜍募､ｱ謨・"; cat /workspace/dashboard.log 2>/dev/null | tail -5 || true; }

# 笏笏 7. CSV 閾ｪ蜍募叙蠕・笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
mkdir -p "$(dirname ${DATA_PATH})"
if [ ! -s "${DATA_PATH}" ]; then
    python3 - <<'PYEOF' || true
import sys, os, urllib.request, ssl
sys.path.insert(0, '/workspace/fx-ea5')
from pathlib import Path

dst = Path(os.environ.get('DATA_PATH', '/workspace/data/USDJPY_H1.csv'))

# 譁ｹ豕・: S3 逶ｴ謗･URL (譛蜆ｪ蜈医・鬮倬・
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://frorit-2022.softether.net:18004')
S3_BUCKET   = os.environ.get('S3_BUCKET',   'fxea')
S3_PREFIX   = os.environ.get('S3_PREFIX',   'mix')
s3_url = f'{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}/data/USDJPY_H1.csv'
print(f'[*] S3 縺九ｉ CSV 蜿門ｾ嶺ｸｭ: {s3_url}')
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(s3_url, context=ctx, timeout=30) as resp:
        data = resp.read()
    if len(data) > 100000:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f'[OK] S3 CSV 蜿門ｾ怜ｮ御ｺ・({len(data)/1e6:.1f} MB)')
        sys.exit(0)
    print(f'[WARN] S3 繝ｬ繧ｹ繝昴Φ繧ｹ縺悟ｰ上＆縺吶℃繧・({len(data)} bytes)')
except Exception as e:
    print(f'[WARN] S3 蜿門ｾ怜､ｱ謨・ {e}')

print('[ERROR] S3 CSV 蜿門ｾ怜､ｱ謨・); sys.exit(1)
PYEOF
    STATUS=$?
    if [ $STATUS -ne 0 ] && [ -n "${DATA_URL}" ]; then
        echo "[*] DATA_URL 縺九ｉ繝繧ｦ繝ｳ繝ｭ繝ｼ繝我ｸｭ..."
        wget -q -O "${DATA_PATH}" "${DATA_URL}" \
          && echo "[OK] CSV 繝繧ｦ繝ｳ繝ｭ繝ｼ繝牙ｮ御ｺ・ \
          || echo "[ERROR] CSV 繝繧ｦ繝ｳ繝ｭ繝ｼ繝牙､ｱ謨・
    fi
else
    echo "[*] CSV 譌｢蟄・ $(du -h ${DATA_PATH} | cut -f1)"
fi

# 笏笏 8. 蟄ｦ鄙偵Ν繝ｼ繝苓ｵｷ蜍・笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
rm -f /workspace/stop.flag

echo ""
echo "[*] 荳ｦ蛻励Λ繝ｳ繝繝繧ｵ繝ｼ繝・幕蟋・
echo "    繝繝・す繝･繝懊・繝・ http://0.0.0.0:${DASHBOARD_PORT}"
echo ""

_STOP_REQUESTED=0

# 笏笏笏 XLA 繧ｭ繝｣繝・す繝･ S3 繧｢繝・・繝ｭ繝ｼ繝・(ZIP蝨ｧ邵ｮ) 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
_xla_cache_upload() {
    [ "$DEVICE_TYPE" != "TPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    python3 - <<'PYEOF'
import logging, os, pathlib, tempfile, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ.get('XLA_CACHE_DIR', '/workspace/local_xla'))
    bucket    = os.environ.get('S3_BUCKET',  'fxea')
    s3_prefix = os.environ.get('S3_PREFIX',  'mix') + '/xla_cache'

    # 蜑榊屓蜷梧悄莉･髯阪↓螟画峩縺輔ｌ縺溘ヵ繧｡繧､繝ｫ縺ｮ縺ｿ繧｢繝・・繝ｭ繝ｼ繝・(蜈ｨ莉ｶI/O遶ｶ蜷医ｒ髦ｲ豁｢)
    marker = cache_dir / '.last_s3_sync'
    last_sync = marker.stat().st_mtime if marker.exists() else 0
    files = [f for f in cache_dir.rglob('*')
             if f.is_file() and f.name != '.last_s3_sync'
             and f.stat().st_mtime > last_sync]
    if not files:
        import sys; sys.exit(0)
    print(f'[*] XLA 繧ｭ繝｣繝・す繝･ S3 蜷梧悄: {len(files)}莉ｶ (譁ｰ隕・譖ｴ譁ｰ縺ｮ縺ｿ)', flush=True)

    def make_client():
        return boto3.client('s3',
            endpoint_url=os.environ.get('S3_ENDPOINT', ''),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
            verify=False)

    def upload(f):
        rel = f.relative_to(cache_dir)
        s3_key = f'{s3_prefix}/{rel}.zip'
        client = make_client()
        for attempt in range(3):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                    zf.write(str(f), f.name)
                client.upload_file(str(tmp_path), bucket, s3_key)
                return f.name
            except Exception as e:
                if tmp_path and tmp_path.exists(): tmp_path.unlink(missing_ok=True)
                if attempt < 2: time.sleep(2 ** attempt)
                else: raise e
            finally:
                if tmp_path and tmp_path.exists(): tmp_path.unlink(missing_ok=True)
        return f.name

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(upload, f): f for f in files}
        for fut in as_completed(futs):
            try: fut.result(); done += 1
            except Exception as e: print(f'  [WARN] upload failed: {e}')
    marker.touch()
    print(f'[OK] XLA S3 蜷梧悄: {done}/{len(files)}莉ｶ螳御ｺ・, flush=True)
except Exception as e:
    print(f'[WARN] XLA 繧ｭ繝｣繝・す繝･ S3 菫晏ｭ伜､ｱ謨・ {e}')
PYEOF
}

# 笏笏笏 XLA 繧ｭ繝｣繝・す繝･ S3 繝繧ｦ繝ｳ繝ｭ繝ｼ繝・(ZIP隗｣蜃榊ｯｾ蠢・ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
_xla_cache_download() {
    [ "$DEVICE_TYPE" != "TPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    echo "[*] XLA 繧ｭ繝｣繝・す繝･繧・S3 縺九ｉ蠕ｩ蜈・ｸｭ (ZIP隗｣蜃・/ 荳ｦ蛻・0繧ｹ繝ｬ繝・ラ)..."
    python3 - <<'PYEOF'
import io, logging, os, pathlib, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ.get('XLA_CACHE_DIR', '/workspace/local_xla'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket    = os.environ.get('S3_BUCKET',  'fxea')
    s3_prefix = os.environ.get('S3_PREFIX',  'mix') + '/xla_cache/'

    def make_client():
        return boto3.client('s3',
            endpoint_url=os.environ.get('S3_ENDPOINT', ''),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
            verify=False)

    # 繝輔ぃ繧､繝ｫ荳隕ｧ繧貞叙蠕・(.zip / 髱栩ip 荳｡蟇ｾ蠢・
    s3 = make_client()
    paginator = s3.get_paginator('list_objects_v2')
    tasks = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel = key[len(s3_prefix):]
            if not rel:
                continue
            is_zip = rel.endswith('.zip')
            # 繝ｭ繝ｼ繧ｫ繝ｫ繝代せ: .zip 繧帝勁縺・◆逶ｸ蟇ｾ繝代せ
            local_rel = rel[:-4] if is_zip else rel
            dst = cache_dir / local_rel
            # 繝ｭ繝ｼ繧ｫ繝ｫ繝輔ぃ繧､繝ｫ縺梧里縺ｫ蟄伜惠縺吶ｌ縺ｰ繧ｹ繧ｭ繝・・
            if dst.exists():
                continue
            tasks.append((key, dst, is_zip))

    if not tasks:
        print(f'[OK] XLA 繧ｭ繝｣繝・す繝･: 蜈ｨ繝輔ぃ繧､繝ｫ譌｢蟄倥∪縺溘・S3譛ｪ蟄伜惠 (繧ｹ繧ｭ繝・・)')
    else:
        print(f'[*] XLA 繧ｭ繝｣繝・す繝･: {len(tasks)}莉ｶ繧偵ム繧ｦ繝ｳ繝ｭ繝ｼ繝我ｸｭ...', flush=True)
        def download(args):
            key, dst, is_zip = args
            dst.parent.mkdir(parents=True, exist_ok=True)
            if is_zip:
                buf = io.BytesIO()
                make_client().download_fileobj(bucket, key, buf)
                buf.seek(0)
                with zipfile.ZipFile(buf) as zf:
                    names = zf.namelist()
                    if names:
                        dst.write_bytes(zf.read(names[0]))
            else:
                make_client().download_file(bucket, key, str(dst))
            return dst.name

        done = 0
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(download, t): t for t in tasks}
            for f in as_completed(futs):
                try:
                    f.result()
                    done += 1
                    if done % 50 == 0:
                        print(f'  ... {done}/{len(tasks)}', flush=True)
                except Exception as e:
                    print(f'  [WARN] DL螟ｱ謨・ {e}')
        print(f'[OK] XLA 繧ｭ繝｣繝・す繝･蠕ｩ蜈・ｮ御ｺ・ {done}/{len(tasks)}莉ｶ')
except Exception as e:
    print(f'[INFO] XLA 繧ｭ繝｣繝・す繝･蠕ｩ蜈・せ繧ｭ繝・・: {e}')
PYEOF
}

# 笏笏笏 torch inductor 繧ｭ繝｣繝・す繝･ S3 繧｢繝・・繝ｭ繝ｼ繝・(ZIP蝨ｧ邵ｮ) 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
_inductor_cache_upload() {
    [ "$DEVICE_TYPE" != "GPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    [ -z "$TORCHINDUCTOR_CACHE_DIR" ] && return 0
    python3 - <<'PYEOF'
import logging, os, pathlib, tempfile, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    if not cache_dir.exists():
        import sys; sys.exit(0)
    bucket    = os.environ.get('S3_BUCKET', 'fxea')
    s3_prefix = os.environ.get('S3_PREFIX', 'mix') + '/torch_inductor_cache'

    marker    = cache_dir / '.last_s3_sync'
    last_sync = marker.stat().st_mtime if marker.exists() else 0
    files = [f for f in cache_dir.rglob('*')
             if f.is_file() and f.name != '.last_s3_sync'
             and f.stat().st_mtime > last_sync]
    if not files:
        import sys; sys.exit(0)
    print(f'[*] inductor 繧ｭ繝｣繝・す繝･ S3 蜷梧悄: {len(files)}莉ｶ (譁ｰ隕・譖ｴ譁ｰ縺ｮ縺ｿ)', flush=True)

    def make_client():
        return boto3.client('s3',
            endpoint_url=os.environ.get('S3_ENDPOINT', ''),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
            verify=False)

    def upload(f):
        rel    = f.relative_to(cache_dir)
        s3_key = f'{s3_prefix}/{rel}.zip'
        client = make_client()
        for attempt in range(3):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                    zf.write(str(f), f.name)
                client.upload_file(str(tmp_path), bucket, s3_key)
                return f.name
            except Exception as e:
                if tmp_path and tmp_path.exists(): tmp_path.unlink(missing_ok=True)
                if attempt < 2: time.sleep(2 ** attempt)
                else: raise e
            finally:
                if tmp_path and tmp_path.exists(): tmp_path.unlink(missing_ok=True)

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(upload, f): f for f in files}
        for fut in as_completed(futs):
            try: fut.result(); done += 1
            except Exception as e: print(f'  [WARN] upload failed: {e}')
    marker.touch()
    print(f'[OK] inductor S3 蜷梧悄: {done}/{len(files)}莉ｶ螳御ｺ・, flush=True)
except Exception as e:
    print(f'[WARN] inductor 繧ｭ繝｣繝・す繝･ S3 菫晏ｭ伜､ｱ謨・ {e}')
PYEOF
}

# 笏笏笏 torch inductor 繧ｭ繝｣繝・す繝･ S3 繝繧ｦ繝ｳ繝ｭ繝ｼ繝・(ZIP隗｣蜃榊ｯｾ蠢・ 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
_inductor_cache_download() {
    [ "$DEVICE_TYPE" != "GPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    [ -z "$TORCHINDUCTOR_CACHE_DIR" ] && return 0
    echo "[*] inductor 繧ｭ繝｣繝・す繝･繧・S3 縺九ｉ蠕ｩ蜈・ｸｭ..."
    python3 - <<'PYEOF'
import io, logging, os, pathlib, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
try:
    import boto3, urllib3
    urllib3.disable_warnings()
    cache_dir = pathlib.Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket    = os.environ.get('S3_BUCKET', 'fxea')
    s3_prefix = os.environ.get('S3_PREFIX', 'mix') + '/torch_inductor_cache/'

    def make_client():
        return boto3.client('s3',
            endpoint_url=os.environ.get('S3_ENDPOINT', ''),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', ''),
            verify=False)

    s3 = make_client()
    paginator = s3.get_paginator('list_objects_v2')
    tasks = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel = key[len(s3_prefix):]
            if not rel:
                continue
            is_zip  = rel.endswith('.zip')
            local_rel = rel[:-4] if is_zip else rel
            dst = cache_dir / local_rel
            if dst.exists():
                continue
            tasks.append((key, dst, is_zip))

    if not tasks:
        print('[OK] inductor 繧ｭ繝｣繝・す繝･: 蜈ｨ繝輔ぃ繧､繝ｫ譌｢蟄倥∪縺溘・S3譛ｪ蟄伜惠 (繧ｹ繧ｭ繝・・)')
    else:
        print(f'[*] inductor 繧ｭ繝｣繝・す繝･: {len(tasks)}莉ｶ繧偵ム繧ｦ繝ｳ繝ｭ繝ｼ繝我ｸｭ...', flush=True)
        def download(args):
            key, dst, is_zip = args
            dst.parent.mkdir(parents=True, exist_ok=True)
            if is_zip:
                buf = io.BytesIO()
                make_client().download_fileobj(bucket, key, buf)
                buf.seek(0)
                with zipfile.ZipFile(buf) as zf:
                    names = zf.namelist()
                    if names:
                        dst.write_bytes(zf.read(names[0]))
            else:
                make_client().download_file(bucket, key, str(dst))
            return dst.name

        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(download, t): t for t in tasks}
            for f in as_completed(futs):
                try:
                    f.result(); done += 1
                    if done % 50 == 0:
                        print(f'  ... {done}/{len(tasks)}', flush=True)
                except Exception as e:
                    print(f'  [WARN] DL螟ｱ謨・ {e}')
        print(f'[OK] inductor 繧ｭ繝｣繝・す繝･蠕ｩ蜈・ｮ御ｺ・ {done}/{len(tasks)}莉ｶ')
except Exception as e:
    print(f'[INFO] inductor 繧ｭ繝｣繝・す繝･蠕ｩ蜈・せ繧ｭ繝・・: {e}')
PYEOF
}

_graceful_stop() {
    echo "[*] 蛛懈ｭ｢繧ｷ繧ｰ繝翫Ν蜿嶺ｿ｡..."
    _STOP_REQUESTED=1
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -TERM "$TRAIN_PID"
    sleep 5
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    # 繧ｭ繝｣繝・す繝･繧呈怙邨ゅい繝・・繝ｭ繝ｼ繝・    _xla_cache_upload
    _inductor_cache_upload
    [ -n "$XLA_SYNC_PID" ] && kill "$XLA_SYNC_PID" 2>/dev/null || true
    [ -n "$INDUCTOR_SYNC_PID" ] && kill "$INDUCTOR_SYNC_PID" 2>/dev/null || true
    echo "[OK] 蛛懈ｭ｢螳御ｺ・
}
trap '_graceful_stop' SIGTERM SIGINT

# XLA 繧ｭ繝｣繝・す繝･繧・S3 縺九ｉ蠕ｩ蜈・(TPU 縺ｮ縺ｿ / 螟ｱ謨励＠縺ｦ繧らｶ夊｡・
# XLA_SKIP_DOWNLOAD=1 縺ｮ蝣ｴ蜷医・繧ｹ繧ｭ繝・・ (繝・ぅ繧ｹ繧ｯ遽邏・Δ繝ｼ繝・
if [ "$DEVICE_TYPE" = "TPU" ] && [ "${XLA_SKIP_DOWNLOAD:-0}" != "1" ]; then
    _XLA_CACHE_DIR="${XLA_CACHE_DIR:-/workspace/xla_cache}"
    _AVAIL_GB=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [ "$_AVAIL_GB" -lt 15 ] && [ -d "$_XLA_CACHE_DIR" ]; then
        _CACHE_CNT=$(ls "$_XLA_CACHE_DIR" 2>/dev/null | wc -l)
        _DEL_CNT=$(( _CACHE_CNT / 4 ))
        [ "$_DEL_CNT" -lt 100 ] && _DEL_CNT=100
        echo "[*] 繝・ぅ繧ｹ繧ｯ遨ｺ縺・${_AVAIL_GB}GB 竊・xla_cache 蜿､縺・${_DEL_CNT}莉ｶ 繧貞炎髯､縺励※繧ｹ繝壹・繧ｹ遒ｺ菫・
        ls -t "$_XLA_CACHE_DIR" | tail -"$_DEL_CNT" | xargs -I{} rm -f "$_XLA_CACHE_DIR/{}" 2>/dev/null || true
        echo "[*] xla_cache 蜑企勁蠕・ $(ls $_XLA_CACHE_DIR 2>/dev/null | wc -l)莉ｶ"
    fi
    _xla_cache_download || true
else
    [ "${XLA_SKIP_DOWNLOAD:-0}" = "1" ] && echo "[*] XLA_SKIP_DOWNLOAD=1: S3繧ｭ繝｣繝・す繝･繝繧ｦ繝ｳ繝ｭ繝ｼ繝峨ｒ繧ｹ繧ｭ繝・・"
fi

# warmup 騾ｲ謐・JSON 繧・S3 縺九ｉ蠕ｩ蜈・(TPU 縺ｮ縺ｿ / 蜀崎ｵｷ蜍墓凾縺ｫ繧ｹ繧ｭ繝・・蛻､螳壹↓菴ｿ逕ｨ)
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
    if count: print(f'[OK] warmup 騾ｲ謐怜ｾｩ蜈・ {count}莉ｶ')
    else: print('[INFO] warmup 騾ｲ謐・ S3 縺ｫ縺ｾ縺縺ゅｊ縺ｾ縺帙ｓ (蛻晏屓)')
except Exception as e:
    print(f'[INFO] warmup 騾ｲ謐怜ｾｩ蜈・せ繧ｭ繝・・: {e}')
PYEOF
fi

# 笏笏 XLA 蜈ｨ繝代ち繝ｼ繝ｳ莠句燕繧ｳ繝ｳ繝代う繝ｫ (TPU 縺ｮ縺ｿ) 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# warmup_xla.py 縺後ヱ繧ｿ繝ｼ繝ｳ1蛟句ｮ御ｺ・☆繧九◆縺ｳ縺ｫ譁ｰ隕上く繝｣繝・す繝･繝輔ぃ繧､繝ｫ繧担3縺ｸ蜊ｳ譎ゅい繝・・繝ｭ繝ｼ繝峨☆繧九・# 縺薙・繝悶Ο繝・け縺悟ｮ御ｺ・☆繧九∪縺ｧ蟄ｦ鄙偵・髢句ｧ九＠縺ｪ縺・・if [ "$DEVICE_TYPE" = "TPU" ] && [ "${WARMUP_SKIP_ALL:-0}" != "1" ]; then
    echo "[*] XLA 莠句燕繧ｳ繝ｳ繝代う繝ｫ髢句ｧ・(螳御ｺ・ｾ後↓蟄ｦ鄙帝幕蟋・"
    python3 /workspace/fx-ea5/warmup_xla.py 2>&1 | tee -a /workspace/train_run.log

    # warmup 螳御ｺ・ｾ・ 谿句ｭ倥く繝｣繝・す繝･繝輔ぃ繧､繝ｫ繧貞酔譛溘い繝・・繝ｭ繝ｼ繝・(蜿悶ｊ縺薙⊂縺鈴亟豁｢)
    echo "[*] XLA 繧ｭ繝｣繝・す繝･ S3 譛邨ょ酔譛滉ｸｭ..."
    _xla_cache_upload || true

    # warmup 騾ｲ謐・JSON 繧・S3 縺ｸ菫晏ｭ・    if [ -n "$S3_ENDPOINT" ]; then
        python3 - <<'PYEOF'
import os, pathlib
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
        for attempt in range(5):
            try:
                s3.upload_file(str(f), bucket, f'{prefix}/{f.name}')
                count += 1
                break
            except Exception as e:
                import time
                if attempt < 4: time.sleep(2 ** attempt)
                else: print(f'[WARN] warmup 騾ｲ謐・S3 菫晏ｭ伜､ｱ謨・{f.name}: {e}')
    if count:
        print(f'[OK] warmup 騾ｲ謐・S3 菫晏ｭ・ {count}莉ｶ')
except Exception as e:
    print(f'[WARN] warmup 騾ｲ謐・S3 菫晏ｭ伜､ｱ謨・ {e}')
PYEOF
    fi
    echo "[OK] XLA 繧ｳ繝ｳ繝代う繝ｫ・・3蜷梧悄 螳御ｺ・竊・蟄ｦ鄙帝幕蟋・

    # WARMUP_ONLY=1: 繧ｳ繝ｳ繝代う繝ｫ縺ｮ縺ｿ縺ｧ邨ゆｺ・(隍・焚VM荳ｦ蛻謡armup譎ゅ↓菴ｿ逕ｨ)
    if [ "${WARMUP_ONLY:-0}" = "1" ]; then
        echo "[*] WARMUP_ONLY=1: XLA繧ｳ繝ｳ繝代う繝ｫ螳御ｺ・ゅさ繝ｳ繝・リ繧堤ｵゆｺ・＠縺ｾ縺吶・
        exit 0
    fi
fi

# GPU: inductor 繧ｭ繝｣繝・す繝･繧・S3 縺九ｉ蠕ｩ蜈・(襍ｷ蜍墓凾, 螟ｱ謨励＠縺ｦ繧らｶ夊｡・
if [ "$DEVICE_TYPE" = "GPU" ] && [ -n "$S3_ENDPOINT" ]; then
    _inductor_cache_download || true
fi

# 蟄ｦ鄙剃ｸｭ縺ｮ譁ｰ隕上く繝｣繝・す繝･ (train.py 縺檎函謌・ 繧貞ｮ壽悄逧・↓S3縺ｸ繝舌ャ繧ｯ繧｢繝・・ (10蛻・＃縺ｨ)
XLA_SYNC_PID=""
if [ "$DEVICE_TYPE" = "TPU" ] && [ -n "$S3_ENDPOINT" ]; then
    (while true; do sleep 600; _xla_cache_upload; done) &
    XLA_SYNC_PID=$!
    echo "[*] 蟄ｦ鄙剃ｸｭXLA繧ｭ繝｣繝・す繝･閾ｪ蜍募酔譛・髢句ｧ・(10蛻・＃縺ｨ, PID: ${XLA_SYNC_PID})"
fi

# GPU: inductor 繧ｭ繝｣繝・す繝･繧貞ｭｦ鄙剃ｸｭ縺ｫ螳壽悄繝舌ャ繧ｯ繧｢繝・・ (10蛻・＃縺ｨ)
INDUCTOR_SYNC_PID=""
if [ "$DEVICE_TYPE" = "GPU" ] && [ -n "$S3_ENDPOINT" ]; then
    (while true; do sleep 600; _inductor_cache_upload; done) &
    INDUCTOR_SYNC_PID=$!
    echo "[*] 蟄ｦ鄙剃ｸｭ inductor 繧ｭ繝｣繝・す繝･閾ｪ蜍募酔譛・髢句ｧ・(10蛻・＃縺ｨ, PID: ${INDUCTOR_SYNC_PID})"
fi

# 笏笏 閾ｪ蜍募・襍ｷ蜍輔Ν繝ｼ繝・笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
# run_train.py 縺後け繝ｩ繝・す繝･縺励※繧り・蜍募ｾｩ譌ｧ縺吶ｋ縲Ｔtop.flag 縺後≠繧後・蜀崎ｵｷ蜍輔＠縺ｪ縺・・RESTART_COUNT=0
while true; do
    python /workspace/fx-ea5/run_train.py 2>&1 | tee -a /workspace/train_run.log &
    TRAIN_PID=$!
    wait $TRAIN_PID
    EXIT_CODE=$?

    # stop.flag 縺ｾ縺溘・ SIGTERM/SIGINT 縺後≠繧後・邨ゆｺ・    if [ "$_STOP_REQUESTED" -eq 1 ] || [ -f /workspace/stop.flag ]; then
        echo "===== 蟄ｦ鄙貞ｮ御ｺ・| 繝繝・す繝･繝懊・繝・ http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    # 豁｣蟶ｸ邨ゆｺ・ｂ邨ゆｺ・    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== 蟄ｦ鄙貞ｮ御ｺ・| 繝繝・す繝･繝懊・繝・ http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[RESTART #${RESTART_COUNT}] run_train.py 逡ｰ蟶ｸ邨ゆｺ・(exit=${EXIT_CODE}) 竊・5遘貞ｾ後↓蜀崎ｵｷ蜍・.."
    # 繧ｯ繝ｩ繝・す繝･繝ｭ繧ｰ縺後≠繧後・譛ｫ蟆ｾ繧定｡ｨ遉ｺ
    if [ -f /workspace/crash.log ]; then
        echo "--- crash.log (譛ｫ蟆ｾ20陦・ ---"
        tail -20 /workspace/crash.log
        echo "----------------------------"
    fi
    sleep 5
done

echo "[*] 繧ｳ繝ｳ繝・リ蠕・ｩ滉ｸｭ..."
wait
