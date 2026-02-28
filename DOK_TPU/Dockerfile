# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA ランダムサーチ  TPU (Google Cloud TPU v4/v5/v6e Trillium) 対応
#
# ベースイメージ: python:3.11-slim
# torch_xla は pip から取得 (GCR 認証不要 → GitHub Actions でビルド可能)
#
# デプロイ先: Google Cloud TPU VM
#   gcloud compute tpus tpu-vm create fx-ea-tpu-v6e \
#     --zone=us-central1-b --accelerator-type=v6e-1 \
#     --version=v2-alpha-tpuv6e --spot
#   ssh <tpu-vm>
#   docker run -d --privileged --net=host \
#     -e PJRT_DEVICE=TPU -e TPU_NUM_DEVICES=1 \
#     frtggroup/fx-ea-tpu:latest
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # PJRT ランタイム指定 (XRT 旧ランタイムより高速・v6e/v5e/v5p に必須)
    PJRT_DEVICE=TPU \
    # XLA フラグ: TPU メモリ断片化防止
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    # PyTorch XLA: プロファイリング無効化
    PT_XLA_DEBUG=0 \
    # デフォルト TPU チップ数 (v6e-1 = 1チップ)
    TPU_NUM_DEVICES=1 \
    DATA_PATH=/workspace/data/USDJPY_H1.csv \
    WORKSPACE=/workspace \
    DASHBOARD_PORT=8080 \
    GDRIVE_FOLDER_ID=1RhIqnpyzQrSuEgIAD7KroNWfELdElS6L \
    S3_ENDPOINT=https://s3.isk01.sakurastorage.jp \
    S3_REGION=jp-north-1 \
    S3_ACCESS_KEY=9LZ71SXF347BKM7MEPPM \
    S3_BUCKET=fxea \
    S3_PREFIX=mix

ENV S3_SECRET_KEY="=u/j64oP8sQvtRXjjNY33jF2S/gOY5WMVgYcdnqG"

ARG GDRIVE_OAUTH_CLIENT_ID=""
ARG GDRIVE_OAUTH_CLIENT_SECRET=""
ARG GDRIVE_OAUTH_REFRESH_TOKEN=""
ENV GDRIVE_OAUTH_CLIENT_ID="${GDRIVE_OAUTH_CLIENT_ID}" \
    GDRIVE_OAUTH_CLIENT_SECRET="${GDRIVE_OAUTH_CLIENT_SECRET}" \
    GDRIVE_OAUTH_REFRESH_TOKEN="${GDRIVE_OAUTH_REFRESH_TOKEN}"

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
        libssl-dev \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python packages (torch + torch_xla TPU 版) ───────────────────────────────
# torch_xla[tpu] は libtpu を pip から取得するため GCR 認証不要
COPY DOK_TPU/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        torch_xla[tpu]==2.5.1 \
        -f https://storage.googleapis.com/libtpu-releases/index.html \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── SSH setup ────────────────────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo "PermitRootLogin yes"              >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"        >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes"         >> /etc/ssh/sshd_config && \
    echo "AllowTcpForwarding yes"           >> /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes"        >> /etc/ssh/sshd_config
RUN echo 'if [[ $- != *i* ]]; then return; fi' >> /root/.bashrc

COPY DOK_EA/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
RUN rm -f /workspace/ai_ea/all_results.json \
          /workspace/ai_ea/best_result.json \
          /workspace/ai_ea/last_result.json \
          /workspace/ai_ea/progress.json \
          /workspace/ai_ea/norm_params.json \
          /workspace/ai_ea/norm_params_best.json
COPY DOK_TPU/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

RUN mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache

WORKDIR /workspace

EXPOSE 22 8080

ENTRYPOINT ["/workspace/entrypoint.sh"]
