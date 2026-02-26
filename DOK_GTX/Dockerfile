# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA ランダムサーチ - GTX 1080 Ti / ローカル Docker 対応版
#
# H100版との違い:
#   - H100_MODE=0 (FP16 AMP, torch.compile無効, 小型モデル優先)
#   - MAX_PARALLEL=1 (VRAM 11GB)
#   - VRAM_PER_TRIAL=8
#   - /opt/artifact 不使用 (ローカルボリュームマウントで永続化)
#
# Build:
#   docker build -f DOK_GTX/Dockerfile -t frtggroup/fx-ea-gtx:latest .
#
# Run:
#   docker run --gpus all -p 8080:8080 -p 2222:22 \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/output:/workspace/ai_ea/trials \
#     -e DATA_URL="https://your-storage/USDJPY_M1.csv" \
#     frtggroup/fx-ea-gtx:latest
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # GTX 1080 Ti: メモリ断片化防止
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
    # GTX 1080 Ti (Pascal CC6.1): TF32/BF16非対応 → FP16 AMPのみ
    H100_MODE=0 \
    # GTX 1080 Ti VRAM 11GB → 1〜2並列
    MAX_PARALLEL=1 \
    VRAM_PER_TRIAL=8 \
    # データパス
    DATA_PATH=/workspace/data/USDJPY_M1.csv \
    WORKSPACE=/workspace \
    # Sakura オブジェクトストレージ
    S3_ENDPOINT=https://s3.isk01.sakurastorage.jp \
    S3_REGION=jp-north-1 \
    S3_ACCESS_KEY=9LZ71SXF347BKM7MEPPM \
    S3_BUCKET=fxea \
    S3_PREFIX=checkpoint_gtx

# S3_SECRET_KEY は = で始まるため別行で定義
ENV S3_SECRET_KEY="=u/j64oP8sQvtRXjjNY33jF2S/gOY5WMVgYcdnqG"

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
COPY DOK_GTX/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── SSH setup ────────────────────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo "PermitRootLogin yes"              >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"        >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes"         >> /etc/ssh/sshd_config && \
    echo "AllowTcpForwarding yes"           >> /etc/ssh/sshd_config && \
    echo "AllowStreamLocalForwarding yes"   >> /etc/ssh/sshd_config && \
    echo "PrintMotd no"                     >> /etc/ssh/sshd_config && \
    echo "PrintLastLog no"                  >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes"                    >> /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes"        >> /etc/ssh/sshd_config
RUN echo 'if [[ $- != *i* ]]; then return; fi' >> /root/.bashrc

# H100版と同じSSHキーを共用
COPY DOK_EA/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
COPY DOK_GTX/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# ── ディレクトリ作成 ─────────────────────────────────────────────────────────
RUN mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache

WORKDIR /workspace

# 22=SSH  8080=Dashboard
EXPOSE 22 8080

ENTRYPOINT ["/workspace/entrypoint.sh"]
