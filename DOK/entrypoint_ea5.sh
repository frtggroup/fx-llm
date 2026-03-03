#!/bin/bash
# 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# FX AI EA 驍ｨ・ｱ陷ｷ蛹ｻ縺顔ｹ晢ｽｳ郢晏現ﾎ懃ｹ晄亢縺・ｹ晢ｽｳ郢昴・# 郢ｧ・ｪ郢晏干縺咏ｹ晢ｽｧ郢晢ｽｳ闕ｳ蟠趣ｽｦ繝ｻ- GPU/TPU/CPU 郢ｧ繝ｻPython 邵ｺ・ｧ陞ｳ謔溘・髢ｾ・ｪ陷榊｢難ｽ､諛ｷ繝ｻ
# 陝・ｽｾ陟｢繝ｻ Vast.ai / Sakura DOK / Google Cloud / 郢晢ｽｭ郢晢ｽｼ郢ｧ・ｫ郢晢ｽｫ / TPU VM
# 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
set -e

# 隨渉隨渉 郢晢ｽｪ郢ｧ・ｽ郢晢ｽｼ郢ｧ・ｹ陋ｻ・ｶ鬮ｯ闊鯉ｽ定怦・ｨ鬯・・蟯ｼMAX (FD隴ｫ・ｯ雋ゅ・繝ｻ郢晏干ﾎ溽ｹｧ・ｻ郢ｧ・ｹ隰ｨ・ｰ郢晢ｽｻ郢晢ｽ｡郢晢ｽ｢郢晢ｽｪ驕ｲ蟲ｨ繝ｻ陋ｻ・ｶ鬮ｯ蜊・ｧ・｣鬮ｯ・､) 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# sysctl 邵ｺ・ｧ郢ｧ・ｫ郢晢ｽｼ郢晞亂ﾎ晉ｹ昜ｻ｣ﾎ帷ｹ晢ｽ｡郢晢ｽｼ郢ｧ・ｿ郢ｧ蜻域呵棔・ｧ陋ｹ繝ｻsysctl -w fs.file-max=1048576          2>/dev/null || true
sysctl -w fs.nr_open=1048576           2>/dev/null || true
sysctl -w kernel.pid_max=4194304       2>/dev/null || true
sysctl -w kernel.threads-max=4194304   2>/dev/null || true
sysctl -w vm.max_map_count=1048576     2>/dev/null || true
sysctl -w kernel.msgmax=134217728      2>/dev/null || true
sysctl -w kernel.msgmnb=134217728      2>/dev/null || true
sysctl -w net.core.somaxconn=65535     2>/dev/null || true
sysctl -w net.core.netdev_max_backlog=65535 2>/dev/null || true

# prlimit 邵ｺ・ｧ陷茨ｽｨ郢晢ｽｪ郢ｧ・ｽ郢晢ｽｼ郢ｧ・ｹ郢ｧ讓ｽAX (root霑夲ｽｹ隶難ｽｩ郢ｧ・ｳ郢晢ｽｳ郢昴・繝ｪ邵ｺ・ｯDocker郢昜ｸ翫・郢晄・・ｸ莨∝応郢ｧ蜑・ｽｸ鬆大ｶ檎ｸｺ讎雁ｺ・妙・ｽ)
_set_limit() { prlimit --"$1"="$2":"$2" --pid $$ 2>/dev/null || true; }
_set_limit nofile   1048576     # 郢ｧ・ｪ郢晢ｽｼ郢晏干ﾎｦFD隰ｨ・ｰ
_set_limit nproc    4194304     # 郢晏干ﾎ溽ｹｧ・ｻ郢ｧ・ｹ/郢ｧ・ｹ郢晢ｽｬ郢昴・繝ｩ隰ｨ・ｰ
_set_limit stack    unlimited   # 郢ｧ・ｹ郢ｧ・ｿ郢昴・縺醍ｹｧ・ｵ郢ｧ・､郢ｧ・ｺ
_set_limit memlock  unlimited   # 郢晢ｽｭ郢昴・縺題愾・ｯ髢ｭ・ｽ郢晢ｽ｡郢晢ｽ｢郢晢ｽｪ
_set_limit core     unlimited   # 郢ｧ・ｳ郢ｧ・｢郢敖郢晢ｽｳ郢晏干縺礼ｹｧ・､郢ｧ・ｺ
_set_limit fsize    unlimited   # 隴崢陞滂ｽｧ郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ郢ｧ・ｵ郢ｧ・､郢ｧ・ｺ
_set_limit data     unlimited   # 郢昴・繝ｻ郢ｧ・ｿ郢ｧ・ｻ郢ｧ・ｰ郢晢ｽ｡郢晢ｽｳ郢昴・_set_limit rss      unlimited   # 陝ｶ・ｸ鬯ｧ闊湖鍋ｹ晢ｽ｢郢晢ｽｪ
_set_limit as       unlimited   # 闔会ｽｮ隲・ｳ郢ｧ・｢郢晏ｳｨﾎ樒ｹｧ・ｹ驕ｨ・ｺ鬮｢繝ｻ_set_limit locks    unlimited   # 郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ郢晢ｽｭ郢昴・縺題ｬｨ・ｰ
_set_limit sigpending 4194304   # 闖ｫ譎芽風郢ｧ・ｷ郢ｧ・ｰ郢晉ｿｫﾎ晁ｬｨ・ｰ
_set_limit msgqueue 134217728   # 郢晢ｽ｡郢昴・縺晉ｹ晢ｽｼ郢ｧ・ｸ郢ｧ・ｭ郢晢ｽ･郢晢ｽｼ郢晁・縺・ｹ晏沺辟・_set_limit rtprio   99          # 郢晢ｽｪ郢ｧ・｢郢晢ｽｫ郢ｧ・ｿ郢ｧ・､郢晢｣ｰ陷・ｽｪ陷井ｺ･・ｺ・ｦ
_set_limit nice     -20         # nice陋滂ｽ､邵ｺ・ｮ闕ｳ遏ｩ蜑・
echo "[*] FD闕ｳ莨∝応: $(ulimit -n)  郢晏干ﾎ溽ｹｧ・ｻ郢ｧ・ｹ闕ｳ莨∝応: $(ulimit -u)"

echo "======================================================"
echo "  FX AI EA 闕ｳ・ｦ陋ｻ蜉ｱﾎ帷ｹ晢ｽｳ郢敖郢晢｣ｰ郢ｧ・ｵ郢晢ｽｼ郢昴・(驍ｨ・ｱ陷ｷ蛹ｻ縺・ｹ晢ｽ｡郢晢ｽｼ郢ｧ・ｸ)"
echo "======================================================"

# 隨渉隨渉 SSH 郢ｧ蜻域呵怕・ｪ陷亥現縲定･搾ｽｷ陷阪・(郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ隶諛ｷ繝ｻ郢ｧ蛹ｻ・願恆繝ｻ 陋ｻ譎・ｄ陋ｹ邏具ｽｸ・ｭ邵ｺ・ｧ郢ｧ繧育｣・け螢ｹ縲堤ｸｺ髦ｪ・狗ｹｧ蛹ｻ竕ｧ邵ｺ・ｫ) 隨渉隨渉隨渉隨渉隨渉隨渉
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
_SSH_PID=$!
sleep 1
kill -0 "$_SSH_PID" 2>/dev/null \
  && echo "[OK] SSH 郢ｧ・ｵ郢晢ｽｼ郢晁・繝ｻ隴鯉ｽｩ隴帶ｺｯ・ｵ・ｷ陷阪・(PID: $_SSH_PID)" \
  || echo "[WARN] SSH 隴鯉ｽｩ隴帶ｺｯ・ｵ・ｷ陷榊供・､・ｱ隰ｨ繝ｻ(驍ｯ螟奇ｽ｡繝ｻ"

# 隨渉隨渉 0a. NTP 隴弱ｇ邯ｾ陷ｷ譴ｧ謔・(S3 RequestTimeTooSkewed 鬮ｦ・ｲ雎・ｽ｢) 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# S3 驗ゑｽｲ陷ｷ閧ｴ・､諛・ｽｨ・ｼ邵ｺ・ｯ・ゑｽｱ15陋ｻ繝ｻ・ｻ・･陷繝ｻ繝ｻ隴弱ｇ邯ｾ闕ｳﾂ髢ｾ・ｴ邵ｺ謔滂ｽｿ繝ｻ・ｦ竏堋繧・＆郢晢ｽｳ郢昴・繝ｪ隘搾ｽｷ陷榊｢灘・邵ｺ・ｫ郢ｧ・ｯ郢晢ｽｭ郢昴・縺醍ｹｧ雋樣・隴帶ｺ倪・郢ｧ荵敖繝ｻif command -v ntpdate &>/dev/null; then
    ntpdate -u pool.ntp.org &>/dev/null && echo "[*] NTP 陷ｷ譴ｧ謔・楜蠕｡・ｺ繝ｻ(ntpdate)" || true
elif command -v chronyc &>/dev/null; then
    chronyc makestep &>/dev/null && echo "[*] NTP 陷ｷ譴ｧ謔・楜蠕｡・ｺ繝ｻ(chronyc)" || true
fi

# 隨渉隨渉 0b. torch_xla 邵ｺ繝ｻCUDA_VISIBLE_DEVICES 郢ｧ蝣､・ｩ・ｺ邵ｺ・ｫ邵ｺ蜷ｶ・狗ｸｺ・ｮ郢ｧ蟶昜ｺ溽ｸｺ繝ｻ隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# torch_xla 邵ｺ・ｯ郢ｧ・､郢晢ｽｳ郢晄亢繝ｻ郢晏沺蜃ｾ邵ｺ・ｫ CUDA_VISIBLE_DEVICES="" 郢ｧ螳夲ｽｨ・ｭ陞ｳ螢ｹ笘・ｹｧ蜿･・ｰ・ｴ陷ｷ蛹ｻ窶ｲ邵ｺ繧・ｽ狗ｸｲ繝ｻ# 郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ隶諛ｷ繝ｻ陷鷹亂竊鍋ｹ晢ｽｪ郢ｧ・ｻ郢昴・繝ｨ邵ｺ蜉ｱ窶ｻ GPU 邵ｺ迹夲ｽｦ荵昶斡郢ｧ荵晢ｽ育ｸｺ繝ｻ竊鍋ｸｺ蜷ｶ・狗ｸｲ繝ｻif [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ "${CUDA_VISIBLE_DEVICES}" = "" ]; then
    if [ -n "${NVIDIA_VISIBLE_DEVICES}" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "none" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "[*] CUDA_VISIBLE_DEVICES 郢ｧ繝ｻ0 邵ｺ・ｫ郢晢ｽｪ郢ｧ・ｻ郢昴・繝ｨ (torch_xla 陝ｷ・ｲ雋り崟莠溯ｱ・ｽ｢)"
    fi
fi
# PJRT_DEVICE 邵ｺ・ｯ docker run -e PJRT_DEVICE=TPU 驕ｲ蟲ｨ縲定ｭ丞ｮ茨ｽ､・ｺ騾ｧ繝ｻ竊楢ｬ悶・・ｮ螢ｹ・・ｹｧ蠕娯螺陜｣・ｴ陷ｷ蛹ｻ繝ｻ邵ｺ譏ｴ・檎ｹｧ雋橸ｽｰ莨√裟
# 隴幢ｽｪ髫ｪ・ｭ陞ｳ螢ｹ繝ｻ陜｣・ｴ陷ｷ蛹ｻ繝ｻ邵ｺ・ｿ CUDA 邵ｺ・ｫ郢昴・繝ｵ郢ｧ・ｩ郢晢ｽｫ郢晞メ・ｨ・ｭ陞ｳ繝ｻ(torch_xla 邵ｺ繝ｻCPU 邵ｺ・ｫ郢晁ｼ斐°郢晢ｽｼ郢晢ｽｫ郢晁・繝｣郢ｧ・ｯ邵ｺ蜷ｶ・狗ｸｺ・ｮ郢ｧ蟶昜ｺ溽ｸｺ繝ｻ
if [ -z "${PJRT_DEVICE}" ]; then
    export PJRT_DEVICE=CUDA
fi

# 隨渉隨渉 1. 郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ髢ｾ・ｪ陷榊｢難ｽ､諛ｷ繝ｻ (--gpus / --privileged 闕ｳ蟠趣ｽｦ繝ｻ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
echo "[*] 郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ隶諛ｷ繝ｻ闕ｳ・ｭ..."

DEVICE_INFO=$(python3 - <<'PYEOF' || echo "CPU|CPU|0"
import os, sys, subprocess

# 隨渉隨渉 1. nvidia-smi 邵ｺ・ｧ陷亥現竊馴￡・ｺ髫ｱ繝ｻ(torch 郢ｧ蛹ｻ・願怦蛹ｻ竊楢楜貅ｯ・｡繝ｻ= torch_xla 陝ｷ・ｲ雋ょｳｨ・定摎讓｣竏ｩ) 隨渉隨渉隨渉隨渉隨渉隨渉
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

# 隨渉隨渉 2. NVIDIA_VISIBLE_DEVICES 郢昶・縺臥ｹ昴・縺・(Vast.ai 驕ｲ蟲ｨ繝ｻ邵ｺ阮呻ｽ檎ｸｺ・ｧ驕抵ｽｺ髫ｱ髦ｪ縲堤ｸｺ髦ｪ・・ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
nv = os.environ.get('NVIDIA_VISIBLE_DEVICES', '')
if nv and nv not in ('none', 'void', 'NoDevFiles'):
    # PJRT_DEVICE=CUDA 郢ｧ螳夲ｽｨ・ｭ陞ｳ螢ｹ・邵ｺ・ｦ torch_xla 邵ｺ繝ｻCPU 邵ｺ・ｫ郢昴・繝ｵ郢ｧ・ｩ郢晢ｽｫ郢晏現・邵ｺ・ｪ邵ｺ繝ｻ・育ｸｺ繝ｻ竊鍋ｸｺ蜷ｶ・・    os.environ['PJRT_DEVICE'] = 'CUDA'
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            print(f"GPU|{name}|{vram}")
            sys.exit(0)
    except Exception:
        pass
    # torch.cuda 邵ｺ蠕｡・ｽ・ｿ邵ｺ蛹ｻ竊醍ｸｺ繝ｻ・ｰ・ｴ陷ｷ繝ｻ nvidia-smi 郢ｧ雋槭・髫ｧ・ｦ髯ｦ繝ｻ(陟大｢鍋・邵ｺ・ｪ邵ｺ蜉ｱ縲定氛莨懈Β驕抵ｽｺ髫ｱ繝ｻ
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
    # GPU 驕抵ｽｺ髫ｱ讎奇ｽ､・ｱ隰ｨ繝ｻ遶翫・TPU VM 闕ｳ鄙ｫ縲・NVIDIA_VISIBLE_DEVICES=all 邵ｺ迹夲ｽｪ・､髫ｪ・ｭ陞ｳ螢ｹ・・ｹｧ蠕娯ｻ邵ｺ繝ｻ・玖愾・ｯ髢ｭ・ｽ隲､・ｧ
    # fall-through 邵ｺ蜉ｱ窶ｻ TPU 郢昶・縺臥ｹ昴・縺醍ｸｺ・ｫ鬨ｾ・ｲ郢ｧﾂ (Unknown GPU 邵ｺ・ｨ邵ｺ蜉ｱ窶ｻ驍ｨ繧・ｽｺ繝ｻ・邵ｺ・ｪ邵ｺ繝ｻ

# 隨渉隨渉 3. torch.cuda 騾ｶ・ｴ隰暦ｽ･驕抵ｽｺ髫ｱ繝ｻ(NVIDIA_VISIBLE_DEVICES 邵ｺ蠕娯・邵ｺ繝ｻ閻ｸ陟・・鬮・ｸｺ繝ｻ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"GPU|{name}|{vram}")
        sys.exit(0)
except Exception:
    pass

# 隨渉隨渉 4. TPU 郢昶・縺臥ｹ昴・縺・(霑夲ｽｩ騾・・繝ｧ郢晁・縺・ｹｧ・ｹ驕抵ｽｺ髫ｱ讎奇ｽｿ繝ｻ・ｰ繝ｻ/ torch_xla import邵ｺ・ｰ邵ｺ莉｣縲堤ｸｺ・ｯ闕ｳ讎企ｦ呵崕繝ｻ 隨渉隨渉
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

# 隨渉隨渉 5. CPU 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
print("CPU|CPU|0")
PYEOF
)

DEVICE_TYPE=$(echo "$DEVICE_INFO" | cut -d'|' -f1)
GPU_NAME=$(echo "$DEVICE_INFO"    | cut -d'|' -f2)
GPU_VRAM=$(echo "$DEVICE_INFO"    | cut -d'|' -f3)

case "$DEVICE_TYPE" in
    GPU)
        echo "[OK] GPU 隶諛ｷ繝ｻ: ${GPU_NAME} (${GPU_VRAM} GB)"
        ;;
    TPU)
        echo "[OK] TPU 隶諛ｷ繝ｻ: ${GPU_NAME}"
        # PJRT_DEVICE 郢ｧ蝣､・｢・ｺ陞ｳ貅倪・ TPU 邵ｺ・ｫ髫ｪ・ｭ陞ｳ繝ｻ(entrypoint 邵ｺ・ｮ CUDA 闕ｳ鬆大ｶ檎ｸｺ髦ｪ・定ｬ・侭笆雎ｸ蛹ｻ笘・
        export PJRT_DEVICE=TPU
        # torch_xla 驕抵ｽｺ髫ｱ繝ｻ/ 隴幢ｽｪ郢ｧ・､郢晢ｽｳ郢ｧ・ｹ郢晏現繝ｻ郢晢ｽｫ邵ｺ・ｪ郢ｧ蟲ｨ繝ｵ郢ｧ・ｩ郢晢ｽｼ郢晢ｽｫ郢晁・繝｣郢ｧ・ｯ郢ｧ・､郢晢ｽｳ郢ｧ・ｹ郢晏現繝ｻ郢晢ｽｫ
        if python3 -c "import torch_xla" 2>/dev/null; then
            echo "[OK] torch_xla 陋ｻ・ｩ騾包ｽｨ陷ｿ・ｯ髢ｭ・ｽ"
        else
            echo "[*] torch_xla 郢ｧ・､郢晢ｽｳ郢ｧ・ｹ郢晏現繝ｻ郢晢ｽｫ闕ｳ・ｭ..."
            TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.5.0")
            pip install --no-cache-dir \
                "torch_xla==${TORCH_VER}" \
                -f https://storage.googleapis.com/libtpu-releases/index.html \
            && echo "[OK] torch_xla 郢ｧ・､郢晢ｽｳ郢ｧ・ｹ郢晏現繝ｻ郢晢ｽｫ陞ｳ蠕｡・ｺ繝ｻ \
            || echo "[WARN] torch_xla 郢ｧ・､郢晢ｽｳ郢ｧ・ｹ郢晏現繝ｻ郢晢ｽｫ陞滂ｽｱ隰ｨ繝ｻ遯ｶ繝ｻCPU 郢晢ｽ｢郢晢ｽｼ郢晏ｳｨ縲帝け螟奇ｽ｡繝ｻ
        fi
        ;;
    *)
        echo "[WARN] GPU/TPU 隴幢ｽｪ隶諛ｷ繝ｻ 遯ｶ繝ｻCPU 郢晢ｽ｢郢晢ｽｼ郢晏ｳｨ縲帝け螟奇ｽ｡繝ｻ
        ;;
esac

# 隨渉隨渉 XLA 郢ｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ (TPU 陝・ｉ逡・ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# XLA 邵ｺ・ｯ陋ｻ譎丞ｱ鍋ｹｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ驍ｨ蜈域｣｡郢ｧ蛛ｵ繝ｵ郢ｧ・｡郢ｧ・､郢晢ｽｫ邵ｺ・ｫ郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･邵ｺ蜉ｱﾂ繝ｻ陜玲ｨ貞ｲｼ闔会ｽ･鬮ｯ髦ｪ繝ｻ陷讎願懸騾包ｽｨ邵ｺ蜷ｶ・狗ｸｲ繝ｻ# S3 邵ｺ・ｫ雎鯉ｽｸ驍ｯ螢ｼ蝟ｧ邵ｺ蜷ｶ・狗ｸｺ阮吮・邵ｺ・ｧ郢ｧ・ｳ郢晢ｽｳ郢昴・繝ｪ/VM 陷蝣ｺ・ｽ諛医・陟募ｾ鯉ｽらｹｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･邵ｺ謔滂ｽｾ・ｩ陷医・・・ｹｧ蠕鯉ｽ狗ｸｲ繝ｻif [ "$DEVICE_TYPE" = "TPU" ]; then
    # XLA郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･邵ｺ・ｯ郢晢ｽｭ郢晢ｽｼ郢ｧ・ｫ郢晢ｽｫSSD陷・ｽｪ陷医・(gcsfuse隴厄ｽｸ邵ｺ蟠趣ｽｾ・ｼ邵ｺ・ｿI/O邵ｺ・ｮ郢晄㈱繝ｨ郢晢ｽｫ郢晞亂繝｣郢ｧ・ｯ郢ｧ雋槫ｱ馴ｩ包ｽｿ)
    # 霑ｺ・ｰ陟・・・､逕ｻ辟・XLA_PERSISTENT_CACHE_PATH 邵ｺ・ｧ闕ｳ鬆大ｶ檎ｸｺ讎雁ｺ・妙・ｽ
    export XLA_CACHE_DIR="${XLA_PERSISTENT_CACHE_PATH:-/workspace/local_xla}"
    mkdir -p "${XLA_CACHE_DIR}"
    # XLA_FLAGS 邵ｺ・ｫ隴厄ｽｸ邵ｺ荳岩・ torch_xla 邵ｺ繝ｻGPU 陝・ｉ逡醍ｹ晁ｼ釆帷ｹｧ・ｰ郢ｧ螳夲ｽｿ・ｽ陷会｣ｰ邵ｺ蜉ｱ窶ｻTPU邵ｺ・ｧFatal郢ｧ・ｯ郢晢ｽｩ郢昴・縺咏ｹ晢ｽ･邵ｺ蜷ｶ・狗ｸｲ繝ｻ    # 雎・ｽ｣邵ｺ蜉ｱ・杁PU陷ｷ莉｣・郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･髫ｪ・ｭ陞ｳ螢ｹ繝ｻ XLA_PERSISTENT_CACHE_PATH 霑ｺ・ｰ陟・・・､逕ｻ辟夂ｹｧ蜑・ｽｽ・ｿ邵ｺ繝ｻﾂ繝ｻ    export XLA_PERSISTENT_CACHE_PATH="${XLA_CACHE_DIR}"
    echo "[*] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･髫ｪ・ｭ陞ｳ繝ｻ ${XLA_CACHE_DIR}"
    # torch_xla 邵ｺ繝ｻLIBTPU_INIT_ARGS 邵ｺ・ｫ鬮ｱ讒ｫ・ｯ・ｾ陟｢諛翫Ψ郢晢ｽｩ郢ｧ・ｰ郢ｧ螳夲ｽｿ・ｽ陷会｣ｰ邵ｺ蜉ｱ窶ｻlibtpu邵ｺ蠕後￠郢晢ｽｩ郢昴・縺咏ｹ晢ｽ･邵ｺ蜷ｶ・狗ｸｺ・ｮ郢ｧ蟶昜ｺ溽ｸｺ繝ｻ    # 驕ｨ・ｺ隴√・・ｭ諤懊・郢ｧ蛛ｵ縺晉ｹ昴・繝ｨ邵ｺ蜉ｱ窶ｻ邵ｺ鄙ｫ・･邵ｺ・ｨ torch_xla 邵ｺ・ｮ setdefault 邵ｺ蠕｡・ｸ鬆大ｶ檎ｸｺ髦ｪ・邵ｺ・ｪ邵ｺ繝ｻ    export LIBTPU_INIT_ARGS=""
    echo "[*] LIBTPU_INIT_ARGS 郢ｧ蛛ｵ縺醍ｹ晢ｽｪ郢ｧ・｢ (鬮ｱ讒ｫ・ｯ・ｾ陟｢諛翫Ψ郢晢ｽｩ郢ｧ・ｰ鬮ｦ・ｲ雎・ｽ｢)"
    # 郢ｧ・ｹ郢ｧ・ｫ郢晢ｽｩ郢晢ｽｼ陋滂ｽ､郢ｧ蛛ｵ縺咏ｹ晢ｽｳ郢晄㈱ﾎ晉ｸｺ・ｨ邵ｺ蜉ｱ窶ｻ霑夲ｽｹ陋ｻ・･隰・ｽｱ邵ｺ繝ｻ・邵ｺ・ｪ邵ｺ繝ｻ遶翫・陷ｷ蠕｡・ｸﾂHLO郢ｧ・ｰ郢晢ｽｩ郢晁ｼ斐帝｡・ｰ邵ｺ・ｪ郢ｧ荵昴○郢ｧ・ｫ郢晢ｽｩ郢晢ｽｼ郢ｧ雋槭・陋ｻ・ｩ騾包ｽｨ邵ｺ繝ｻ    # 闕ｳ蟠趣ｽｦ竏壺・陷髦ｪ縺慕ｹ晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ郢ｧ蟶昜ｺ溽ｸｺ繝ｻ(torch_xla 2.x 隰暦ｽｨ陞ゑｽｨ髫ｪ・ｭ陞ｳ繝ｻ
    export XLA_NO_SPECIAL_SCALARS=1
    echo "[*] XLA_NO_SPECIAL_SCALARS=1 髫ｪ・ｭ陞ｳ繝ｻ(闕ｳ蟠趣ｽｦ竏壺・陷髦ｪ縺慕ｹ晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ鬮ｦ・ｲ雎・ｽ｢)"
fi

# Python 邵ｺ・ｫ 郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ陷ｷ髦ｪ・定ｲゑｽ｡邵ｺ繝ｻexport GPU_NAME
export DEVICE_TYPE
export GPU_VRAM

# 隨渉隨渉 2. CUDA MPS (霑夲ｽｹ隶難ｽｩ闕ｳ蟠趣ｽｦ竏壹・陞滂ｽｱ隰ｨ蜉ｱ・邵ｺ・ｦ郢ｧ繧会ｽｶ螟奇ｽ｡繝ｻ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
if [ "$DEVICE_TYPE" = "GPU" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
    nvidia-cuda-mps-control -d 2>/dev/null \
      && echo "[OK] CUDA MPS 隘搾ｽｷ陷榊供・ｮ蠕｡・ｺ繝ｻ \
      || true   # 霑夲ｽｹ隶難ｽｩ邵ｺ・ｪ邵ｺ遉ｼ閻ｸ陟・・縲堤ｸｺ・ｯ陞滂ｽｱ隰ｨ蜉ｱ笘・ｹｧ荵昶ｲ霎滂ｽ｡髫輔・fi

# 隨渉隨渉 4. 雎鯉ｽｸ驍ｯ螢ｹ縺帷ｹ晏現ﾎ樒ｹ晢ｽｼ郢ｧ・ｸ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
ARTIFACT=/opt/artifact
if [ -d "${ARTIFACT}" ] || [ -b "${ARTIFACT}" ]; then
    echo "[*] Sakura DOK 郢晢ｽ｢郢晢ｽｼ郢昴・ /opt/artifact 郢ｧ蜑・ｽｽ・ｿ騾包ｽｨ"
    mkdir -p "${ARTIFACT}/data" "${ARTIFACT}/fx-ea5/trials" \
             "${ARTIFACT}/fx-ea5/top100" "${ARTIFACT}/fx-ea5/top_cache"
    [ ! -L /workspace/data ]             && rm -rf /workspace/data             && ln -sf "${ARTIFACT}/data"              /workspace/data
    [ ! -L /workspace/fx-ea5/trials ]     && rm -rf /workspace/fx-ea5/trials     && ln -sf "${ARTIFACT}/fx-ea5/trials"      /workspace/fx-ea5/trials
    [ ! -L /workspace/fx-ea5/top100 ]     && rm -rf /workspace/fx-ea5/top100     && ln -sf "${ARTIFACT}/fx-ea5/top100"      /workspace/fx-ea5/top100
    [ ! -L /workspace/fx-ea5/top_cache ]  && rm -rf /workspace/fx-ea5/top_cache  && ln -sf "${ARTIFACT}/fx-ea5/top_cache"   /workspace/fx-ea5/top_cache
    export TORCHINDUCTOR_CACHE_DIR="${ARTIFACT}/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[OK] Sakura DOK 郢ｧ・ｹ郢晏現ﾎ樒ｹ晢ｽｼ郢ｧ・ｸ髫ｪ・ｭ陞ｳ螢ｼ・ｮ蠕｡・ｺ繝ｻ
else
    echo "[*] 郢ｧ・ｯ郢晢ｽｩ郢ｧ・ｦ郢昴・郢晢ｽｭ郢晢ｽｼ郢ｧ・ｫ郢晢ｽｫ郢晢ｽ｢郢晢ｽｼ郢昴・ /workspace 郢ｧ蜑・ｽｽ・ｿ騾包ｽｨ"
    mkdir -p /workspace/data /workspace/fx-ea5/trials \
             /workspace/fx-ea5/top100 /workspace/fx-ea5/top_cache
    # torch.compile inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ繝ｻ/workspace 邵ｺ・ｫ雎鯉ｽｸ驍ｯ螢ｼ蝟ｧ
    # (郢昴・繝ｵ郢ｧ・ｩ郢晢ｽｫ郢昴・~/.cache/torch/inductor/ 邵ｺ・ｯ郢ｧ・ｳ郢晢ｽｳ郢昴・繝ｪ陷蟠趣ｽｵ・ｷ陷崎ｼ斐定ｱｸ蛹ｻ竏ｴ郢ｧ荵昶螺郢ｧ繝ｻ
    export TORCHINDUCTOR_CACHE_DIR="/workspace/torch_inductor_cache"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
fi

# 隨渉隨渉 5. 霑ｺ・ｰ陟・・・､逕ｻ辟・隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
export PYTHONPATH="/workspace/fx-ea5:${PYTHONPATH}"
export DATA_PATH="${DATA_PATH:-/workspace/data/USDJPY_H1.csv}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo "[*] 髫ｪ・ｭ陞ｳ繝ｻ"
echo "    郢昴・繝ｰ郢ｧ・､郢ｧ・ｹ     : ${DEVICE_TYPE} / ${GPU_NAME}"
echo "    DATA_PATH    : ${DATA_PATH}"
echo "    GDRIVE       : ${GDRIVE_FOLDER_ID:-(隴幢ｽｪ髫ｪ・ｭ陞ｳ繝ｻ}"
echo "    DASHBOARD    : port ${DASHBOARD_PORT}"

# 隨渉隨渉 6. 郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢晁歓・ｵ・ｷ陷阪・(郢ｧ・ｯ郢晢ｽｩ郢昴・縺咏ｹ晢ｽ･隴弱ｊ繝ｻ陷榊供繝ｻ隘搾ｽｷ陷阪・ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
_start_dashboard() {
    while true; do
        echo "[DASH] server.py 隘搾ｽｷ陷阪・$(date '+%H:%M:%S')" >> /workspace/dashboard.log
        python /workspace/fx-ea5/server.py >> /workspace/dashboard.log 2>&1
        EXIT_CODE=$?
        echo "[DASH] server.py 驍ｨ繧・ｽｺ繝ｻ(exit=$EXIT_CODE) 遶翫・5驕倩ｲ橸ｽｾ蠕娯・陷蟠趣ｽｵ・ｷ陷阪・$(date '+%H:%M:%S')" \
            >> /workspace/dashboard.log
        sleep 5
    done
}
_start_dashboard &
DASH_PID=$!
sleep 3
curl -s --connect-timeout 3 http://127.0.0.1:${DASHBOARD_PORT}/api/status > /dev/null 2>&1 \
  && echo "[OK] 郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢晁歓・ｵ・ｷ陷阪・port ${DASHBOARD_PORT} (PID: $DASH_PID)" \
  || { echo "[WARN] 郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢晁歓・ｵ・ｷ陷榊供・､・ｱ隰ｨ繝ｻ"; cat /workspace/dashboard.log 2>/dev/null | tail -5 || true; }

# 隨渉隨渉 7. CSV 髢ｾ・ｪ陷榊供蜿呵輔・隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
mkdir -p "$(dirname ${DATA_PATH})"
if [ ! -s "${DATA_PATH}" ]; then
    python3 - <<'PYEOF' || true
import sys, os, urllib.request, ssl
sys.path.insert(0, '/workspace/fx-ea5')
from pathlib import Path

dst = Path(os.environ.get('DATA_PATH', '/workspace/data/USDJPY_H1.csv'))

# 隴・ｽｹ雎輔・: S3 騾ｶ・ｴ隰暦ｽ･URL (隴崢陷・ｽｪ陷亥現繝ｻ鬯ｮ蛟ｬﾂ繝ｻ
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://frorit-2022.softether.net:18004')
S3_BUCKET   = os.environ.get('S3_BUCKET',   'fxea')
S3_PREFIX   = os.environ.get('S3_PREFIX',   'mix')
s3_url = f'{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}/data/USDJPY_H1.csv'
print(f'[*] S3 邵ｺ荵晢ｽ・CSV 陷ｿ髢・ｾ蠍ｺ・ｸ・ｭ: {s3_url}')
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(s3_url, context=ctx, timeout=30) as resp:
        data = resp.read()
    if len(data) > 100000:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f'[OK] S3 CSV 陷ｿ髢・ｾ諤懶ｽｮ蠕｡・ｺ繝ｻ({len(data)/1e6:.1f} MB)')
        sys.exit(0)
    print(f'[WARN] S3 郢晢ｽｬ郢ｧ・ｹ郢晄亢ﾎｦ郢ｧ・ｹ邵ｺ謔滂ｽｰ荳奇ｼ・ｸｺ蜷ｶ邃・ｹｧ繝ｻ({len(data)} bytes)')
except Exception as e:
    print(f'[WARN] S3 陷ｿ髢・ｾ諤懶ｽ､・ｱ隰ｨ繝ｻ {e}')

print('[ERROR] S3 CSV 陷ｿ髢・ｾ諤懶ｽ､・ｱ隰ｨ繝ｻ); sys.exit(1)
PYEOF
    STATUS=$?
    if [ $STATUS -ne 0 ] && [ -n "${DATA_URL}" ]; then
        echo "[*] DATA_URL 邵ｺ荵晢ｽ臥ｹ敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晄・・ｸ・ｭ..."
        wget -q -O "${DATA_PATH}" "${DATA_URL}" \
          && echo "[OK] CSV 郢敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晉甥・ｮ蠕｡・ｺ繝ｻ \
          || echo "[ERROR] CSV 郢敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晉甥・､・ｱ隰ｨ繝ｻ
    fi
else
    echo "[*] CSV 隴鯉ｽ｢陝・・ $(du -h ${DATA_PATH} | cut -f1)"
fi

# 隨渉隨渉 8. 陝・ｽｦ驗吝・ﾎ晉ｹ晢ｽｼ郢晁挙・ｵ・ｷ陷阪・隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
rm -f /workspace/stop.flag

echo ""
echo "[*] 闕ｳ・ｦ陋ｻ蜉ｱﾎ帷ｹ晢ｽｳ郢敖郢晢｣ｰ郢ｧ・ｵ郢晢ｽｼ郢昴・蟷戊沂繝ｻ
echo "    郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢昴・ http://0.0.0.0:${DASHBOARD_PORT}"
echo ""

_STOP_REQUESTED=0

# 隨渉隨渉隨渉 XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 郢ｧ・｢郢昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢昴・(ZIP陜ｨ・ｧ驍ｵ・ｮ) 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
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

    # 陷第ｦ雁ｱ楢惺譴ｧ謔・脂・･鬮ｯ髦ｪ竊楢棔逕ｻ蟲ｩ邵ｺ霈費ｽ檎ｸｺ貅倥Ψ郢ｧ・｡郢ｧ・､郢晢ｽｫ邵ｺ・ｮ邵ｺ・ｿ郢ｧ・｢郢昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢昴・(陷茨ｽｨ闔会ｽｶI/O驕ｶ・ｶ陷ｷ蛹ｻ・帝ｫｦ・ｲ雎・ｽ｢)
    marker = cache_dir / '.last_s3_sync'
    last_sync = marker.stat().st_mtime if marker.exists() else 0
    files = [f for f in cache_dir.rglob('*')
             if f.is_file() and f.name != '.last_s3_sync'
             and f.stat().st_mtime > last_sync]
    if not files:
        import sys; sys.exit(0)
    print(f'[*] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 陷ｷ譴ｧ謔・ {len(files)}闔会ｽｶ (隴・ｽｰ髫輔・隴厄ｽｴ隴・ｽｰ邵ｺ・ｮ邵ｺ・ｿ)', flush=True)

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
    print(f'[OK] XLA S3 陷ｷ譴ｧ謔・ {done}/{len(files)}闔会ｽｶ陞ｳ蠕｡・ｺ繝ｻ, flush=True)
except Exception as e:
    print(f'[WARN] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 闖ｫ譎擾ｽｭ莨懶ｽ､・ｱ隰ｨ繝ｻ {e}')
PYEOF
}

# 隨渉隨渉隨渉 XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 郢敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢昴・(ZIP髫暦ｽ｣陷・ｦ奇ｽｯ・ｾ陟｢繝ｻ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
_xla_cache_download() {
    [ "$DEVICE_TYPE" != "TPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    echo "[*] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ繝ｻS3 邵ｺ荵晢ｽ芽包ｽｩ陷医・・ｸ・ｭ (ZIP髫暦ｽ｣陷・・/ 闕ｳ・ｦ陋ｻ繝ｻ0郢ｧ・ｹ郢晢ｽｬ郢昴・繝ｩ)..."
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

    # 郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ闕ｳﾂ髫包ｽｧ郢ｧ雋槫徐陟輔・(.zip / 鬮ｱ譬ｩip 闕ｳ・｡陝・ｽｾ陟｢繝ｻ
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
            # 郢晢ｽｭ郢晢ｽｼ郢ｧ・ｫ郢晢ｽｫ郢昜ｻ｣縺・ .zip 郢ｧ蟶晏求邵ｺ繝ｻ笳・ｶ・ｸ陝・ｽｾ郢昜ｻ｣縺・            local_rel = rel[:-4] if is_zip else rel
            dst = cache_dir / local_rel
            # 郢晢ｽｭ郢晢ｽｼ郢ｧ・ｫ郢晢ｽｫ郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ邵ｺ譴ｧ驥檎ｸｺ・ｫ陝・ｼ懈Β邵ｺ蜷ｶ・檎ｸｺ・ｰ郢ｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ
            if dst.exists():
                continue
            tasks.append((key, dst, is_zip))

    if not tasks:
        print(f'[OK] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･: 陷茨ｽｨ郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ隴鯉ｽ｢陝・･竏ｪ邵ｺ貅倥・S3隴幢ｽｪ陝・ｼ懈Β (郢ｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ)')
    else:
        print(f'[*] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･: {len(tasks)}闔会ｽｶ郢ｧ蛛ｵ繝郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晄・・ｸ・ｭ...', flush=True)
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
                    print(f'  [WARN] DL陞滂ｽｱ隰ｨ繝ｻ {e}')
        print(f'[OK] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･陟包ｽｩ陷医・・ｮ蠕｡・ｺ繝ｻ {done}/{len(tasks)}闔会ｽｶ')
except Exception as e:
    print(f'[INFO] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･陟包ｽｩ陷医・縺帷ｹｧ・ｭ郢昴・繝ｻ: {e}')
PYEOF
}

# 隨渉隨渉隨渉 torch inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 郢ｧ・｢郢昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢昴・(ZIP陜ｨ・ｧ驍ｵ・ｮ) 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
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
    print(f'[*] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 陷ｷ譴ｧ謔・ {len(files)}闔会ｽｶ (隴・ｽｰ髫輔・隴厄ｽｴ隴・ｽｰ邵ｺ・ｮ邵ｺ・ｿ)', flush=True)

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
    print(f'[OK] inductor S3 陷ｷ譴ｧ謔・ {done}/{len(files)}闔会ｽｶ陞ｳ蠕｡・ｺ繝ｻ, flush=True)
except Exception as e:
    print(f'[WARN] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 闖ｫ譎擾ｽｭ莨懶ｽ､・ｱ隰ｨ繝ｻ {e}')
PYEOF
}

# 隨渉隨渉隨渉 torch inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 郢敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢昴・(ZIP髫暦ｽ｣陷・ｦ奇ｽｯ・ｾ陟｢繝ｻ 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
_inductor_cache_download() {
    [ "$DEVICE_TYPE" != "GPU" ] && return 0
    [ -z "$S3_ENDPOINT" ] && return 0
    [ -z "$TORCHINDUCTOR_CACHE_DIR" ] && return 0
    echo "[*] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ繝ｻS3 邵ｺ荵晢ｽ芽包ｽｩ陷医・・ｸ・ｭ..."
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
        print('[OK] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･: 陷茨ｽｨ郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ隴鯉ｽ｢陝・･竏ｪ邵ｺ貅倥・S3隴幢ｽｪ陝・ｼ懈Β (郢ｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ)')
    else:
        print(f'[*] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･: {len(tasks)}闔会ｽｶ郢ｧ蛛ｵ繝郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晄・・ｸ・ｭ...', flush=True)
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
                    print(f'  [WARN] DL陞滂ｽｱ隰ｨ繝ｻ {e}')
        print(f'[OK] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･陟包ｽｩ陷医・・ｮ蠕｡・ｺ繝ｻ {done}/{len(tasks)}闔会ｽｶ')
except Exception as e:
    print(f'[INFO] inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･陟包ｽｩ陷医・縺帷ｹｧ・ｭ郢昴・繝ｻ: {e}')
PYEOF
}

_graceful_stop() {
    echo "[*] 陋帶㊧・ｭ・｢郢ｧ・ｷ郢ｧ・ｰ郢晉ｿｫﾎ晁愾蠍ｺ・ｿ・｡..."
    _STOP_REQUESTED=1
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -TERM "$TRAIN_PID"
    sleep 5
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" || true
    # 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ蜻域咎お繧・＞郢昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢昴・    _xla_cache_upload
    _inductor_cache_upload
    [ -n "$XLA_SYNC_PID" ] && kill "$XLA_SYNC_PID" 2>/dev/null || true
    [ -n "$INDUCTOR_SYNC_PID" ] && kill "$INDUCTOR_SYNC_PID" 2>/dev/null || true
    echo "[OK] 陋帶㊧・ｭ・｢陞ｳ蠕｡・ｺ繝ｻ
}
trap '_graceful_stop' SIGTERM SIGINT

# XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ繝ｻS3 邵ｺ荵晢ｽ芽包ｽｩ陷医・(TPU 邵ｺ・ｮ邵ｺ・ｿ / 陞滂ｽｱ隰ｨ蜉ｱ・邵ｺ・ｦ郢ｧ繧会ｽｶ螟奇ｽ｡繝ｻ
# XLA_SKIP_DOWNLOAD=1 邵ｺ・ｮ陜｣・ｴ陷ｷ蛹ｻ繝ｻ郢ｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ (郢昴・縺・ｹｧ・ｹ郢ｧ・ｯ驕ｽﾂ驍上・ﾎ皮ｹ晢ｽｼ郢昴・
if [ "$DEVICE_TYPE" = "TPU" ] && [ "${XLA_SKIP_DOWNLOAD:-0}" != "1" ]; then
    _XLA_CACHE_DIR="${XLA_CACHE_DIR:-/workspace/xla_cache}"
    _AVAIL_GB=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [ "$_AVAIL_GB" -lt 15 ] && [ -d "$_XLA_CACHE_DIR" ]; then
        _CACHE_CNT=$(ls "$_XLA_CACHE_DIR" 2>/dev/null | wc -l)
        _DEL_CNT=$(( _CACHE_CNT / 4 ))
        [ "$_DEL_CNT" -lt 100 ] && _DEL_CNT=100
        echo "[*] 郢昴・縺・ｹｧ・ｹ郢ｧ・ｯ驕ｨ・ｺ邵ｺ繝ｻ${_AVAIL_GB}GB 遶翫・xla_cache 陷ｿ・､邵ｺ繝ｻ${_DEL_CNT}闔会ｽｶ 郢ｧ雋樒ｎ鬮ｯ・､邵ｺ蜉ｱ窶ｻ郢ｧ・ｹ郢晏｣ｹ繝ｻ郢ｧ・ｹ驕抵ｽｺ闖ｫ繝ｻ
        ls -t "$_XLA_CACHE_DIR" | tail -"$_DEL_CNT" | xargs -I{} rm -f "$_XLA_CACHE_DIR/{}" 2>/dev/null || true
        echo "[*] xla_cache 陷台ｼ∝求陟輔・ $(ls $_XLA_CACHE_DIR 2>/dev/null | wc -l)闔会ｽｶ"
    fi
    _xla_cache_download || true
else
    [ "${XLA_SKIP_DOWNLOAD:-0}" = "1" ] && echo "[*] XLA_SKIP_DOWNLOAD=1: S3郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢敖郢ｧ・ｦ郢晢ｽｳ郢晢ｽｭ郢晢ｽｼ郢晏ｳｨ・堤ｹｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ"
fi

# warmup 鬨ｾ・ｲ隰舌・JSON 郢ｧ繝ｻS3 邵ｺ荵晢ｽ芽包ｽｩ陷医・(TPU 邵ｺ・ｮ邵ｺ・ｿ / 陷蟠趣ｽｵ・ｷ陷榊｢灘・邵ｺ・ｫ郢ｧ・ｹ郢ｧ・ｭ郢昴・繝ｻ陋ｻ・､陞ｳ螢ｹ竊楢抄・ｿ騾包ｽｨ)
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
    if count: print(f'[OK] warmup 鬨ｾ・ｲ隰先懶ｽｾ・ｩ陷医・ {count}闔会ｽｶ')
    else: print('[INFO] warmup 鬨ｾ・ｲ隰舌・ S3 邵ｺ・ｫ邵ｺ・ｾ邵ｺ・ｰ邵ｺ繧・ｽ顔ｸｺ・ｾ邵ｺ蟶呻ｽ・(陋ｻ譎丞ｱ・')
except Exception as e:
    print(f'[INFO] warmup 鬨ｾ・ｲ隰先懶ｽｾ・ｩ陷医・縺帷ｹｧ・ｭ郢昴・繝ｻ: {e}')
PYEOF
fi

# 隨渉隨渉 XLA 陷茨ｽｨ郢昜ｻ｣縺｡郢晢ｽｼ郢晢ｽｳ闔蜿･辯慕ｹｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ (TPU 邵ｺ・ｮ邵ｺ・ｿ) 隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# warmup_xla.py 邵ｺ蠕後Τ郢ｧ・ｿ郢晢ｽｼ郢晢ｽｳ1陋溷唱・ｮ蠕｡・ｺ繝ｻ笘・ｹｧ荵昶螺邵ｺ・ｳ邵ｺ・ｫ隴・ｽｰ髫穂ｸ翫￥郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ郢ｧ諡・邵ｺ・ｸ陷奇ｽｳ隴弱ｅ縺・ｹ昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢晏ｳｨ笘・ｹｧ荵敖繝ｻ# 邵ｺ阮吶・郢晄じﾎ溽ｹ昴・縺醍ｸｺ謔滂ｽｮ蠕｡・ｺ繝ｻ笘・ｹｧ荵昶穐邵ｺ・ｧ陝・ｽｦ驗吝・繝ｻ鬮｢蜿･・ｧ荵晢ｼ邵ｺ・ｪ邵ｺ繝ｻﾂ繝ｻif [ "$DEVICE_TYPE" = "TPU" ] && [ "${WARMUP_SKIP_ALL:-0}" != "1" ]; then
    echo "[*] XLA 闔蜿･辯慕ｹｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ鬮｢蜿･・ｧ繝ｻ(陞ｳ蠕｡・ｺ繝ｻ・ｾ蠕娯・陝・ｽｦ驗吝ｸ晏ｹ戊沂繝ｻ"
    python3 /workspace/fx-ea5/warmup_xla.py 2>&1 | tee -a /workspace/train_run.log

    # warmup 陞ｳ蠕｡・ｺ繝ｻ・ｾ繝ｻ 隹ｿ蜿･・ｭ蛟･縺冗ｹ晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢晁ｼ斐＜郢ｧ・､郢晢ｽｫ郢ｧ雋樣・隴帶ｺ倥＞郢昴・繝ｻ郢晢ｽｭ郢晢ｽｼ郢昴・(陷ｿ謔ｶ・顔ｸｺ阮吮鰍邵ｺ驤ｴ莠溯ｱ・ｽ｢)
    echo "[*] XLA 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ S3 隴崢驍ｨ繧・・隴帶ｻ会ｽｸ・ｭ..."
    _xla_cache_upload || true

    # warmup 鬨ｾ・ｲ隰舌・JSON 郢ｧ繝ｻS3 邵ｺ・ｸ闖ｫ譎擾ｽｭ繝ｻ    if [ -n "$S3_ENDPOINT" ]; then
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
                else: print(f'[WARN] warmup 鬨ｾ・ｲ隰舌・S3 闖ｫ譎擾ｽｭ莨懶ｽ､・ｱ隰ｨ繝ｻ{f.name}: {e}')
    if count:
        print(f'[OK] warmup 鬨ｾ・ｲ隰舌・S3 闖ｫ譎擾ｽｭ繝ｻ {count}闔会ｽｶ')
except Exception as e:
    print(f'[WARN] warmup 鬨ｾ・ｲ隰舌・S3 闖ｫ譎擾ｽｭ莨懶ｽ､・ｱ隰ｨ繝ｻ {e}')
PYEOF
    fi
    echo "[OK] XLA 郢ｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ繝ｻ繝ｻ3陷ｷ譴ｧ謔・陞ｳ蠕｡・ｺ繝ｻ遶翫・陝・ｽｦ驗吝ｸ晏ｹ戊沂繝ｻ

    # WARMUP_ONLY=1: 郢ｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ邵ｺ・ｮ邵ｺ・ｿ邵ｺ・ｧ驍ｨ繧・ｽｺ繝ｻ(髫阪・辟啖M闕ｳ・ｦ陋ｻ隰｡armup隴弱ｅ竊楢抄・ｿ騾包ｽｨ)
    if [ "${WARMUP_ONLY:-0}" = "1" ]; then
        echo "[*] WARMUP_ONLY=1: XLA郢ｧ・ｳ郢晢ｽｳ郢昜ｻ｣縺・ｹ晢ｽｫ陞ｳ蠕｡・ｺ繝ｻﾂ繧・＆郢晢ｽｳ郢昴・繝ｪ郢ｧ蝣､・ｵ繧・ｽｺ繝ｻ・邵ｺ・ｾ邵ｺ蜷ｶﾂ繝ｻ
        exit 0
    fi
fi

# GPU: inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ繝ｻS3 邵ｺ荵晢ｽ芽包ｽｩ陷医・(隘搾ｽｷ陷榊｢灘・, 陞滂ｽｱ隰ｨ蜉ｱ・邵ｺ・ｦ郢ｧ繧会ｽｶ螟奇ｽ｡繝ｻ
if [ "$DEVICE_TYPE" = "GPU" ] && [ -n "$S3_ENDPOINT" ]; then
    _inductor_cache_download || true
fi

# 陝・ｽｦ驗吝宴・ｸ・ｭ邵ｺ・ｮ隴・ｽｰ髫穂ｸ翫￥郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･ (train.py 邵ｺ讙主・隰後・ 郢ｧ雋橸ｽｮ螢ｽ謔・ｧ繝ｻ竊鉄3邵ｺ・ｸ郢晁・繝｣郢ｧ・ｯ郢ｧ・｢郢昴・繝ｻ (10陋ｻ繝ｻ・・ｸｺ・ｨ)
XLA_SYNC_PID=""
if [ "$DEVICE_TYPE" = "TPU" ] && [ -n "$S3_ENDPOINT" ]; then
    (while true; do sleep 600; _xla_cache_upload; done) &
    XLA_SYNC_PID=$!
    echo "[*] 陝・ｽｦ驗吝宴・ｸ・ｭXLA郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･髢ｾ・ｪ陷榊供驟碑ｭ帙・鬮｢蜿･・ｧ繝ｻ(10陋ｻ繝ｻ・・ｸｺ・ｨ, PID: ${XLA_SYNC_PID})"
fi

# GPU: inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･郢ｧ雋橸ｽｭ・ｦ驗吝宴・ｸ・ｭ邵ｺ・ｫ陞ｳ螢ｽ謔・ｹ晁・繝｣郢ｧ・ｯ郢ｧ・｢郢昴・繝ｻ (10陋ｻ繝ｻ・・ｸｺ・ｨ)
INDUCTOR_SYNC_PID=""
if [ "$DEVICE_TYPE" = "GPU" ] && [ -n "$S3_ENDPOINT" ]; then
    (while true; do sleep 600; _inductor_cache_upload; done) &
    INDUCTOR_SYNC_PID=$!
    echo "[*] 陝・ｽｦ驗吝宴・ｸ・ｭ inductor 郢ｧ・ｭ郢晢ｽ｣郢昴・縺咏ｹ晢ｽ･髢ｾ・ｪ陷榊供驟碑ｭ帙・鬮｢蜿･・ｧ繝ｻ(10陋ｻ繝ｻ・・ｸｺ・ｨ, PID: ${INDUCTOR_SYNC_PID})"
fi

# 隨渉隨渉 髢ｾ・ｪ陷榊供繝ｻ隘搾ｽｷ陷崎ｼ釆晉ｹ晢ｽｼ郢昴・隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉隨渉
# run_train.py 邵ｺ蠕後￠郢晢ｽｩ郢昴・縺咏ｹ晢ｽ･邵ｺ蜉ｱ窶ｻ郢ｧ繧翫・陷榊供・ｾ・ｩ隴鯉ｽｧ邵ｺ蜷ｶ・狗ｸｲ・ｴtop.flag 邵ｺ蠕娯旺郢ｧ蠕後・陷蟠趣ｽｵ・ｷ陷崎ｼ費ｼ邵ｺ・ｪ邵ｺ繝ｻﾂ繝ｻRESTART_COUNT=0
while true; do
    python /workspace/fx-ea5/run_train.py 2>&1 | tee -a /workspace/train_run.log &
    TRAIN_PID=$!
    wait $TRAIN_PID
    EXIT_CODE=$?

    # stop.flag 邵ｺ・ｾ邵ｺ貅倥・ SIGTERM/SIGINT 邵ｺ蠕娯旺郢ｧ蠕後・驍ｨ繧・ｽｺ繝ｻ    if [ "$_STOP_REQUESTED" -eq 1 ] || [ -f /workspace/stop.flag ]; then
        echo "===== 陝・ｽｦ驗呵ｲ橸ｽｮ蠕｡・ｺ繝ｻ| 郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢昴・ http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    # 雎・ｽ｣陝ｶ・ｸ驍ｨ繧・ｽｺ繝ｻ・るお繧・ｽｺ繝ｻ    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== 陝・ｽｦ驗呵ｲ橸ｽｮ蠕｡・ｺ繝ｻ| 郢敖郢昴・縺咏ｹ晢ｽ･郢晄㈱繝ｻ郢昴・ http://0.0.0.0:${DASHBOARD_PORT} ====="
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[RESTART #${RESTART_COUNT}] run_train.py 騾｡・ｰ陝ｶ・ｸ驍ｨ繧・ｽｺ繝ｻ(exit=${EXIT_CODE}) 遶翫・5驕倩ｲ橸ｽｾ蠕娯・陷蟠趣ｽｵ・ｷ陷阪・.."
    # 郢ｧ・ｯ郢晢ｽｩ郢昴・縺咏ｹ晢ｽ･郢晢ｽｭ郢ｧ・ｰ邵ｺ蠕娯旺郢ｧ蠕後・隴幢ｽｫ陝・ｽｾ郢ｧ螳夲ｽ｡・ｨ驕会ｽｺ
    if [ -f /workspace/crash.log ]; then
        echo "--- crash.log (隴幢ｽｫ陝・ｽｾ20髯ｦ繝ｻ ---"
        tail -20 /workspace/crash.log
        echo "----------------------------"
    fi
    sleep 5
done

echo "[*] 郢ｧ・ｳ郢晢ｽｳ郢昴・繝ｪ陟輔・・ｩ貊会ｽｸ・ｭ..."
wait
