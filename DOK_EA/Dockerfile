# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA 並列ランダムサーチ on Sakura DOK / H100 80GB  (v2)
#
# Build (from repo root):
#   docker build -f DOK_EA/Dockerfile -t fx-ea:latest .
#
# Run (Sakura DOK / docker run):
#   docker run --gpus all -p 7860:7860 -p 2222:22 \
#     -e DATA_URL="https://your-storage/USDJPY_M1.csv" \
#     -e H100_MODE=1 \
#     -e MAX_PARALLEL=3 \
#     -e VRAM_PER_TRIAL=10 \
#     -v /host/output:/workspace/ai_ea \
#     fx-ea:latest
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # H100 メモリ断片化防止
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    # TF32 を明示的に有効化 (A100/H100 デフォルト)
    TORCH_ALLOW_TF32=1 \
    # H100 モード: 大型モデル・バッチサイズ・エポック数
    H100_MODE=1 \
    # 並列数 / 1試行あたりの最低 VRAM (GB)
    MAX_PARALLEL=3 \
    VRAM_PER_TRIAL=10 \
    # データパス (DATA_URL で上書き可能)
    DATA_PATH=/workspace/data/USDJPY_M1.csv \
    WORKSPACE=/workspace

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        curl \
        git \
        vim \
        wget \
        tmux \
        htop \
        rsync \
        gcc \
        g++ \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
COPY DOK_EA/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── SSH setup ────────────────────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo "PermitRootLogin yes"         >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"   >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes"    >> /etc/ssh/sshd_config && \
    echo "AllowTcpForwarding yes"      >> /etc/ssh/sshd_config

COPY DOK_EA/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
COPY DOK_EA/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# ── ディレクトリ作成 ─────────────────────────────────────────────────────────
RUN mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache

WORKDIR /workspace

# 22=SSH  7860=Dashboard
EXPOSE 22 7860

ENTRYPOINT ["/workspace/entrypoint.sh"]
