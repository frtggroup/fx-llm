# ─────────────────────────────────────────────────────────────────────────────
# FX AI EA ランダムサーチE- GTX 1080 Ti / ローカル Docker 対応版
#
# H100版との違い:
#   - H100_MODE=0 (FP16 AMP, torch.compile無効, 小型モチE��優允E
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
    # GTX 1080 Ti: メモリ断牁E��防止
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
    # GTX 1080 Ti (Pascal CC6.1): TF32/BF16非対忁EↁEFP16 AMPのみ
    H100_MODE=0 \
    # GTX 1080 Ti VRAM 11GB ↁE2並刁EÁE5GB = 10GB
    MAX_PARALLEL=2 \
    VRAM_PER_TRIAL=5 \
    # チE�Eタパス (H1 CSV 直接使用 ↁEM1リサンプル不要で高送E
    DATA_PATH=/workspace/data/USDJPY_H1.csv \
    WORKSPACE=/workspace \
    # Sakura オブジェクトストレージ
    S3_ENDPOINT=https://s3.isk01.sakurastorage.jp \
    S3_REGION=jp-north-1 \
    S3_ACCESS_KEY=9LZ71SXF347BKM7MEPPM \
    S3_BUCKET=fxea \
    S3_PREFIX=mix

# S3_SECRET_KEY は = で始まるため別行で定義
ENV S3_SECRET_KEY="=u/j64oP8sQvtRXjjNY33jF2S/gOY5WMVgYcdnqG"

# Google Drive 共有ストレージ (S3 より優先。実行時に docker run -e で上書きも可)
ENV NODE_ID=gtx1080ti \
    GDRIVE_FOLDER_ID=1RhIqnpyzQrSuEgIAD7KroNWfELdElS6L
ENV GDRIVE_CREDENTIALS_BASE64="ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAidGVzdC00ODI3MTAiLAogICJwcml2YXRlX2tleV9pZCI6ICIwOWM4YzExZGVmMTE3N2QzY2I5YTYzZGYyNzQ3YmVmMGVmNDQ3NDI1IiwKICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdmdJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLZ3dnZ1NrQWdFQUFvSUJBUUM2ZkdHNk9HZFJySGpRXG45ZXZVd2p0SThMUEtDMzdLdGJpc3JRZVhBcmJrd1R2WDEvSWdlSkJVdVdRbGdiaGhUTDloZXRoLys2VmdBaFdBXG5zR2JacDVLOUsxOHh1WEk2LzllNEFPeWhSc2RBY3JsZERFUFZIdWo4ZS9pdE5LcGI5R0ZON2VESjluWEU2TnhmXG5xejR4WVNqdk9KdkRoMDV5VmIvalZvY05FNEdKMC9obG96YUt5NGVGZ0pCSzdvOXh1dDMxdzRhTFZtOGZRcSs0XG52WElkN29CRmVNTnl5dGpYWklVRHZ0STRLM3gyM0F0QVcwdVNiWHMrMy9Fa1RlYXdSNW41VkJXZk1Sa25EMVpDXG5QV2xVcW1zcXhBbHVka3NjVHFhQWNlYW5VUGdEb0s1ME9OSGNBS2cxblUyWkU3N3lWTnBFRkk0T0tNZVZjeFczXG5LR3hCVXZuL0FnTUJBQUVDZ2dFQUhLQ1FiZ0pQbGdHWWNGZDRFdXFZS0R2UEZiS0xEYm8wdGNsQUU0WDJwb29sXG5VaDZUekRsRnlzUEU1RjhUKzFmSlAwRVk4QmlreSthMS8xSFZFQStCM3FsRVRVRlBFRkNMQUhPZjM2Z2FXVUhqXG5RNjNvMGRRVFp3THU2MFBjN1EzSVRDcTZxYlk4WVBSd00vMTVLQ0ptb3NaNVh0d3NCakEwcTBUeXFXYTNxWDBPXG5HNXZ1Zk5Ub0QxQ0tJRGs1NWFTNjNjZzc3ZlA5QTVmWnhTdmR1ODdJODV4L3d5UUYrUnJCZ3dQRC8wV2hSNnlnXG5GeVdLYnBGbUMxVmhlaEgvaXlGUmhNVVdkTkphQWRDclVxdWNmOThvWDFnYklmZEExTFczNDdBZUNONkpIU3J5XG51UlA4bDdta1YxVGhrR2NXajFtUEFGV0lpWEt5eXRodzdsZE5Wb2JIVlFLQmdRRGRNa1A3ZCt1STZxcW53OGVEXG5ZdXJqWUE1Uy9ubFYyOGZCUXlLTzZWOGdWdUdGS0tacktDRjU4SFhmVk5VMncyNG9zTGNCU0dQeHFRQUx0c3VzXG5keGRRako4SGI5b2NBeityblBmam1jVDVJcmdENVB5NEhTam9XMDN3eU9RRVZkNC92ZFNiNEUyekM0YUh6ZDA3XG5ETWpjUXQxR2wveFlxR0wxY2NmQUk4SlF0UUtCZ1FEWDAveUpLVFRpcFk3OTN0Zm9KT0kxNy9taUZVZ3pzekVhXG5iZnIvNnJDUUFKSFcyUTJDNm9VWkp5QTFtQW4xVHgrUmEyQzNXU1RRRUtxYXNhZGpkSmlQbWtRRXJ1RmJLS2Q0XG5sYVk5RG51SVlyLzlTbWptUlJKcCtOQXk5eFMrV2ZPTHp6SXQrWnRFQmJ2T1AxVUdweHlvSThpVVVtYzFJdzIyXG40UVp5NUVVMFl3S0JnUURKK1pYOXpyaHhZdVhGSG8zVk9NT0dMeldsTHR3WVBBRWdSVzhPMDZXYTMvbzVIcThtXG45WlByeGxlN1A3a1lza2FxUFFJOTE1MzdFZmdLdWpVc0crcGNFS3M5blRLcGxZWFQ3QmFsNnNqalBJS09lODhQXG5Sd0VOZWlHbkZhaXZGTUZGcnM2dUxwaUVQcWZ1NVZsM0MrdDB4RkZNbm9jN1pUWkhMRWZhK0V2OGFRS0JnRkpnXG5jUVVSRzBSZW1JTkJhUHNNQ09NNVl2aUFVdnlZcW1kQ0R3bGExR1ZDTW1ma2xJbWhaWWZDTFVNSGN0V1VpNkgvXG5iL3pMSTRmZXRKOHlxNGNFVzY2RDNPNUVxWE1YdzBQOUttUW5nUEwvYmZRSlVKeURSSkg2QzgxSURUelNMR2p5XG5mbDZaNUs5RDJQalJwMnhRb0ZsMHZ2d05yNkxTYkV1ZSs4MGlYMSt0QW9HQkFJaVlSMURaTnBYYXFtaEJnQmcyXG54NGQ1cE1HbncyRTJIVy9tQlJCY2dqNG5JNVlld3llZDBqUWhMOHNOTWIwRjZxYmR2cERMK3k3NVFxd3Bvell6XG5ReTcrMG5teDBOdWlGRkVtNjFTYkZGRmg1aXpJV2k4NzArV1FQWXc2RjhmendxczlOV2d6UDRGZlFiNlE2UEhkXG52cjl2Z3ZtVjMweXRYZnlCL2FoNnhhYXZcbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgImNsaWVudF9lbWFpbCI6ICJmeGVhLTg2MkB0ZXN0LTQ4MjcxMC5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMTc4Njg5OTg2NTYyMzM0MzE2MzQiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L2Z4ZWEtODYyJTQwdGVzdC00ODI3MTAuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"

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

# H100版と同じSSHキーを�E用
COPY DOK_EA/ssh/authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# ── Source files ─────────────────────────────────────────────────────────────
COPY ai_ea/ /workspace/ai_ea/
# 学習実行時に生�Eされるランタイムファイルをイメージから除去
RUN rm -f /workspace/ai_ea/all_results.json \
          /workspace/ai_ea/best_result.json \
          /workspace/ai_ea/last_result.json \
          /workspace/ai_ea/progress.json \
          /workspace/ai_ea/llm_progress.json \
          /workspace/ai_ea/norm_params.json \
          /workspace/ai_ea/norm_params_best.json
COPY DOK_GTX/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# ── チE��レクトリ作�E ─────────────────────────────────────────────────────────
RUN mkdir -p /workspace/data \
             /workspace/ai_ea/trials \
             /workspace/ai_ea/top100 \
             /workspace/ai_ea/top_cache

WORKDIR /workspace

# 22=SSH  8080=Dashboard
EXPOSE 22 8080

ENTRYPOINT ["/workspace/entrypoint.sh"]
