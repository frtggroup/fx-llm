# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA3 統合イメージ  (GTX / H100 / H200 / Vast.ai / ローカル / TPU 全対応)
# fx-ea3: 70特徴量 × 25列 (基底+逐次差分24本) = N_FEATURES=1750
# S3バケット prefix: mix2
#
# Build:
#   docker build -f DOK3/Dockerfile -t frtgroup/fx-ea3:latest .
#
# Run (ローカル GTX):
#   docker run --gpus all -p 8080:8080 -p 2222:22 frtgroup/fx-ea3:latest
#
# Run (Vast.ai / クラウド):
#   docker run --gpus all --privileged --net=host frtgroup/fx-ea3:latest
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_ALLOW_TF32=1 \
    DATA_PATH=/workspace/data/USDJPY_H1.csv \
    WORKSPACE=/workspace \
    DASHBOARD_PORT=8080 \
    GDRIVE_FOLDER_ID=1RhIqnpyzQrSuEgIAD7KroNWfELdElS6L \
    S3_ENDPOINT=https://frorit-2022.softether.net:18004 \
    S3_REGION=us-east-1 \
    S3_ACCESS_KEY=mioroot \
    S3_BUCKET=fxea \
    S3_PREFIX=mix2

ENV S3_SECRET_KEY="Yakrty1484!#"

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
        ntpdate \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
COPY DOK3/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── torch_xla + libtpu (GPU・TPU 両対応) ────────────────────────────────────
RUN pip install --no-cache-dir \
        "torch_xla[tpu]==2.6.0" \
        -f https://storage.googleapis.com/libtpu-releases/index.html \
        -f https://storage.googleapis.com/libtpu-wheels/index.html \
    && echo "[OK] torch_xla 2.6.0 + libtpu SDK インストール完了" \
    || echo "[WARN] torch_xla インストールスキップ (GPU/CPU イメージとして続行)"

# ── tpu-info ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir tpu-info \
    && echo "[OK] tpu-info インストール完了" \
    || echo "[WARN] tpu-info インストールスキップ"

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

COPY DOK3/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY fx-ea3/ /workspace/fx-ea3/
RUN rm -f /workspace/fx-ea3/all_results.json \
          /workspace/fx-ea3/best_result.json \
          /workspace/fx-ea3/last_result.json \
          /workspace/fx-ea3/progress.json \
          /workspace/fx-ea3/llm_progress.json \
          /workspace/fx-ea3/norm_params.json \
          /workspace/fx-ea3/norm_params_best.json
COPY DOK3/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

RUN mkdir -p /workspace/data \
             /workspace/fx-ea3/trials \
             /workspace/fx-ea3/top100 \
             /workspace/fx-ea3/top_cache

WORKDIR /workspace

# 22=SSH  8080=Dashboard  7860=Sakura DOK nginx 互換
EXPOSE 22 8080 7860

ENTRYPOINT ["/workspace/entrypoint.sh"]
