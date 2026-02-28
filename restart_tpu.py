import subprocess, os

IP = '35.188.129.15'
KEY = os.path.expanduser(r'~\.ssh\google_compute_engine')
SSH_USER = 'yu'

# シークレットは環境変数または tpu_secrets.py から読み込む
try:
    from tpu_secrets import (GDRIVE_FOLDER_ID, GDRIVE_OAUTH_CLIENT_ID,
                              GDRIVE_OAUTH_CLIENT_SECRET, GDRIVE_OAUTH_REFRESH_TOKEN)
except ImportError:
    GDRIVE_FOLDER_ID           = os.environ.get('GDRIVE_FOLDER_ID', '')
    GDRIVE_OAUTH_CLIENT_ID     = os.environ.get('GDRIVE_OAUTH_CLIENT_ID', '')
    GDRIVE_OAUTH_CLIENT_SECRET = os.environ.get('GDRIVE_OAUTH_CLIENT_SECRET', '')
    GDRIVE_OAUTH_REFRESH_TOKEN = os.environ.get('GDRIVE_OAUTH_REFRESH_TOKEN', '')

script = (
    "set -e\n"
    "echo '[*] 最新イメージをプル...'\n"
    "sudo docker pull frtgroup/fx-ea:latest\n"
    "echo '[*] 旧コンテナ停止・削除...'\n"
    "sudo docker stop fx-ea-tpu 2>/dev/null || true\n"
    "sudo docker rm   fx-ea-tpu 2>/dev/null || true\n"
    "echo '[*] コンテナ起動...'\n"
    "sudo docker run -d --name fx-ea-tpu --restart unless-stopped"
    " --net=host"          # TPU tpu-runtime.service (port 8353) に接続するため必須
    " --privileged"
    f" -e PJRT_DEVICE=TPU"
    f" -e NODE_ID=tpu_v6e"
    f" -e GDRIVE_FOLDER_ID={GDRIVE_FOLDER_ID}"
    f" -e GDRIVE_OAUTH_CLIENT_ID={GDRIVE_OAUTH_CLIENT_ID}"
    f" -e GDRIVE_OAUTH_CLIENT_SECRET={GDRIVE_OAUTH_CLIENT_SECRET}"
    f" -e GDRIVE_OAUTH_REFRESH_TOKEN={GDRIVE_OAUTH_REFRESH_TOKEN}"
    " frtgroup/fx-ea:latest\n"
    "echo '[OK] 起動完了'\n"
    "sudo docker ps | grep fx-ea-tpu\n"
)

script_bytes = script.replace('\r\n', '\n').encode('utf-8')
result = subprocess.run(
    ['ssh', '-i', KEY,
     '-o', 'StrictHostKeyChecking=no',
     '-o', 'UserKnownHostsFile=/dev/null',
     f'{SSH_USER}@{IP}', 'bash'],
    input=script_bytes,
    timeout=180
)
print('exit code:', result.returncode)
