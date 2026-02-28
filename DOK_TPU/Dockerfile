# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA ランダムサーチ  TPU (Google Cloud TPU v4/v5/Trillium) 対応
#
# ベースイメージ: PyTorch XLA 公式 (torch_xla + libtpu 同梱)
#
# デプロイ先: Google Cloud TPU VM
#   gcloud compute tpus tpu-vm create fx-ea-tpu \
#     --zone=us-central2-b --accelerator-type=v4-8 \
#     --version=tpu-ubuntu2204-base
#   gcloud compute tpus tpu-vm ssh fx-ea-tpu --zone=us-central2-b \
#     --command="docker run -d --privileged --net=host \
#       -e DATA_URL=<url> \
#       frtggroup/fx-ea-tpu:latest"
#
# 環境変数 (すべて自動検出のため通常設定不要):
#   NODE_ID         : 自動 (tpu_v4 など)
#   TPU_NUM_DEVICES : TPU チップ数 (デフォルト 4)
#   DATA_URL        : CSV ダウンロード URL (GDrive になければ指定)
# ─────────────────────────────────────────────────────────────────────────────
FROM us-central1-docker.pkg.dev/tpu-pytorch/torch-xla-release/torch_xla:r2.5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # PJRT ランタイム指定 (XRT 旧ランタイムより高速・v6e/v5e/v5p に必須)
    PJRT_DEVICE=TPU \
    # XLA フラグ: TPU メモリ断片化防止
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    # PyTorch XLA: プロファイリング無効化 (本番では不要)
    PT_XLA_DEBUG=0 \
    # v6e-1 などシングルチップ構成のデフォルト (自動検出される場合は不要)
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
COPY DOK_TPU/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
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
