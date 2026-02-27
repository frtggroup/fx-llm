# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA 並列ランダムサーチ on Sakura DOK / H100 80GB  (v2)
#
# Build (from repo root):
#   docker build -f DOK_EA/Dockerfile -t fx-ea:latest .
#
# Run (Sakura DOK / docker run):
#   docker run --gpus all -p 8080:8080 -p 2222:22 \
#     -e DATA_URL="https://your-storage/USDJPY_M1.csv" \
#     -e H100_MODE=1 \
#     -e MAX_PARALLEL=6 \
#     -e VRAM_PER_TRIAL=6 \
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
    # H100 80GB: 6並列 × 6GB = 36GB使用, 残44GBはバッファ
    MAX_PARALLEL=6 \
    VRAM_PER_TRIAL=6 \
    # データパス (DATA_URL で上書き可能)
    DATA_PATH=/workspace/data/USDJPY_M1.csv \
    WORKSPACE=/workspace \
    # Sakura オブジェクトストレージ (docker-compose未使用時のフォールバック)
    S3_ENDPOINT=https://s3.isk01.sakurastorage.jp \
    S3_REGION=jp-north-1 \
    S3_ACCESS_KEY=9LZ71SXF347BKM7MEPPM \
    S3_BUCKET=fxea \
    S3_PREFIX=checkpoint

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
COPY DOK_EA/requirements.txt /tmp/requirements.txt
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
    # 非対話コマンド実行のため bash profile を無効化しない
    echo "PermitUserEnvironment yes"        >> /etc/ssh/sshd_config
# SSH 接続時にブラケットペーストモードが有効にならないよう .bashrc を設定
RUN echo 'if [[ $- != *i* ]]; then return; fi' >> /root/.bashrc

COPY DOK_EA/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
# 学習実行時に生成されるランタイムファイルをイメージから除去
RUN rm -f /workspace/ai_ea/all_results.json \
          /workspace/ai_ea/best_result.json \
          /workspace/ai_ea/last_result.json \
          /workspace/ai_ea/progress.json \
          /workspace/ai_ea/llm_progress.json \
          /workspace/ai_ea/norm_params.json \
          /workspace/ai_ea/norm_params_best.json
COPY DOK_EA/entrypoint.sh /workspace/entrypoint.sh
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
