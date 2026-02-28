# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA 統合イメージ  (GTX / H100 / H200 / Vast.ai / ローカル 全対応)
#
# GPU は起動時に自動検出し、並列数・モデルサイズ・バッチサイズを最適設定します。
#
# Build:
#   docker build -f DOK/Dockerfile -t frtgroup/fx-ea:latest .
#
# Run (ローカル GTX):
#   docker run --gpus all -p 8080:8080 -p 2222:22 frtgroup/fx-ea:latest
#
# Run (Vast.ai / クラウド):
#   docker run --gpus all --privileged --net=host frtgroup/fx-ea:latest
#
# 環境変数 (任意):
#   DATA_URL          CSVダウンロードURL (GDriveにない場合のフォールバック)
#   DASHBOARD_PORT    ダッシュボードポート (デフォルト: 8080)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_ALLOW_TF32=1 \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
COPY DOK/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# torch_xla は TPU 環境でのみ必要なため、起動時に entrypoint.sh が自動インストールします。
# GPU / CPU 環境ではインストール不要 (イメージサイズ削減)。

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

COPY DOK/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
RUN rm -f /workspace/ai_ea/all_results.json \
          /workspace/ai_ea/best_result.json \
          /workspace/ai_ea/last_result.json \
          /workspace/ai_ea/progress.json \
          /workspace/ai_ea/llm_progress.json \
          /workspace/ai_ea/norm_params.json \
          /workspace/ai_ea/norm_params_best.json
COPY DOK/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

RUN mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache

WORKDIR /workspace

# 22=SSH  8080=Dashboard  7860=Sakura DOK nginx 互換
EXPOSE 22 8080 7860

ENTRYPOINT ["/workspace/entrypoint.sh"]
