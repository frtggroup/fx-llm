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
    S3_PREFIX=mix

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
COPY DOK/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── torch_xla + libtpu (GPU・TPU 両対応 / ビルド時に組み込み) ────────────────
# torch_xla[tpu] で libtpu.so も含めてインストール。
# ・GPU/CPU 環境: libtpu は読み込まれない (安全に無視される)
# ・TPU VM 環境 : --net=host + tpu-runtime.service で XLA が有効化される
# インストール失敗してもビルドを止めない (GPU のみ使用なら不要)
RUN pip install --no-cache-dir \
        "torch_xla[tpu]==2.6.0" \
        -f https://storage.googleapis.com/libtpu-releases/index.html \
        -f https://storage.googleapis.com/libtpu-wheels/index.html \
    && echo "[OK] torch_xla 2.6.0 + libtpu SDK インストール完了" \
    || echo "[WARN] torch_xla インストールスキップ (GPU/CPU イメージとして続行)"

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
