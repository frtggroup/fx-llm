"""
TPU VM 起動スクリプト
  - v6e-1 スポット VM を複数ゾーンで試行作成
  - Docker CE インストール & frtggroup/fx-ea-v2 コンテナ起動
  - 3時間後に自動で VM を停止・削除
  - Ctrl+C でも即時削除可能

使い方:
  python tpu_start.py                      # 3時間後に自動削除
  python tpu_start.py --hours 5            # 5時間後に自動削除
  python tpu_start.py --no-auto-delete     # 自動削除しない
"""

import subprocess, sys, time, threading, argparse, textwrap
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────────────────────────────
DOCKER_IMAGE  = "frtgroup/fx-ea:latest"
VM_NAME       = "fx-ea-tpu-v6e"
ACCEL_TYPE    = "v6e-4"
TPU_VERSION   = "v2-alpha-tpuv6e"
PROJECT       = "project-c7a2ed3f-0395-4b76-967"
SSH_KEY       = str(Path.home() / ".ssh" / "google_compute_engine")
SSH_USER      = "yu"
AUTO_DELETE_H = 3   # デフォルト自動削除時間 (時間)

# v6e-4 が利用可能なゾーン候補 (空き率が高い順)
ZONES = [
    "us-central1-b",
    "us-central1-a",
    "us-central1-c",
    "us-east5-a",
    "us-east5-b",
    "us-east5-c",
    "europe-west4-a",
    "europe-west4-b",
]

GCLOUD = None   # gcloud パスは _find_gcloud() で解決

# ── gcloud パス解決 ───────────────────────────────────────────────────────────
def _find_gcloud() -> str:
    import shutil, os
    candidates = [
        shutil.which("gcloud"),
        str(Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"),
        "/usr/bin/gcloud",
        "/usr/local/bin/gcloud",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("gcloud が見つかりません。Google Cloud SDK をインストールしてください。")


def run(cmd: list, check=True, capture=False) -> subprocess.CompletedProcess:
    """gcloud コマンドを実行"""
    full_cmd = [GCLOUD] + cmd
    if capture:
        r = subprocess.run(full_cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"コマンド失敗: {r.stderr.strip()}")
        return r
    else:
        return subprocess.run(full_cmd, check=check)


def propagate_ssh_key(zone: str):
    """gcloud で SSH 鍵を TPU VM に伝播させる (plink ホストキープロンプトは無視)"""
    print("[*] SSH 鍵を VM に伝播中 (gcloud)...")
    r = run([
        "compute", "tpus", "tpu-vm", "ssh", VM_NAME,
        f"--zone={zone}",
        "--command=echo PROPAGATED",
        "--ssh-flag=-o BatchMode=yes",
        "--ssh-flag=-o StrictHostKeyChecking=no",
    ], check=False, capture=True)
    # plink が "-o" を拒否しても鍵伝播自体は成功している場合が多い
    if "Propagating SSH public key" in (r.stdout + r.stderr) or \
       "PROPAGATED" in (r.stdout + r.stderr):
        print("[OK] SSH 鍵伝播完了")
    else:
        print("[WARN] gcloud SSH 終了 (鍵は伝播済みの可能性あり、OpenSSH で続行)")


def ssh(ip: str, remote_cmd: str, timeout=120) -> bool:
    """OpenSSH でリモートコマンドを実行 (バイト送信で CRLF 変換を回避)"""
    import subprocess
    # text=True にすると Windows で \n→\r\n 変換され bash が失敗するためバイト送信
    script_bytes = remote_cmd.replace('\r\n', '\n').encode('utf-8')
    result = subprocess.run(
        ["ssh", "-i", SSH_KEY,
         "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ConnectTimeout=15",
         f"{SSH_USER}@{ip}", "bash"],
        input=script_bytes,
        timeout=timeout,
    )
    return result.returncode == 0


# ── VM 操作 ───────────────────────────────────────────────────────────────────
def vm_exists(zone: str) -> bool:
    r = run(["compute", "tpus", "tpu-vm", "list",
             f"--zone={zone}", f"--filter=name={VM_NAME}",
             "--format=value(name)"], capture=True, check=False)
    return VM_NAME in r.stdout


def create_vm(zone: str) -> bool:
    """TPU VM を作成。成功なら True"""
    print(f"  [{zone}] VM 作成中...")
    r = run([
        "compute", "tpus", "tpu-vm", "create", VM_NAME,
        f"--zone={zone}",
        f"--accelerator-type={ACCEL_TYPE}",
        f"--version={TPU_VERSION}",
        "--spot",
    ], check=False)
    return r.returncode == 0


def get_vm_ip(zone: str) -> str:
    r = run([
        "compute", "tpus", "tpu-vm", "describe", VM_NAME,
        f"--zone={zone}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ], capture=True)
    return r.stdout.strip()


def delete_vm(zone: str, silent=False):
    """VM を削除"""
    if not silent:
        print(f"\n[削除] {VM_NAME} ({zone}) を削除中...")
    run([
        "compute", "tpus", "tpu-vm", "delete", VM_NAME,
        f"--zone={zone}", "--quiet",
    ], check=False)
    if not silent:
        print("[完了] VM 削除完了")


def ensure_firewall():
    """ダッシュボード (8080) の GCP ファイアウォールルールを確保する"""
    rule_name = "fx-ea-dashboard"
    r = run([
        "compute", "firewall-rules", "describe", rule_name,
        f"--project={PROJECT}", "--format=value(name)",
    ], capture=True, check=False)
    if rule_name in r.stdout:
        print(f"[OK] ファイアウォール '{rule_name}' は既に存在します")
        return
    print(f"[*] ファイアウォールルール '{rule_name}' を作成中 (tcp:8080)...")
    run([
        "compute", "firewall-rules", "create", rule_name,
        f"--project={PROJECT}",
        "--direction=INGRESS",
        "--priority=1000",
        "--network=default",
        "--action=ALLOW",
        "--rules=tcp:8080",
        "--source-ranges=0.0.0.0/0",
        "--description=FX AI EA dashboard port",
    ], check=False)
    print(f"[OK] ファイアウォールルール '{rule_name}' 作成完了")


# ── Docker セットアップ ────────────────────────────────────────────────────────
def _build_setup_sh(image: str) -> str:
    """Docker セットアップ + コンテナ起動シェルスクリプトを生成"""
    return f"""\
set -e
echo '=== Docker CE インストール (未インストール時のみ) ==='
if ! command -v docker &>/dev/null; then
  sudo apt-get remove -y docker docker-engine docker.io docker-compose 2>/dev/null || true
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
  ARCH=$(dpkg --print-architecture)
  echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $CODENAME stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io
  sudo systemctl enable --now docker
fi
echo "[OK] Docker: $(sudo docker --version)"

echo '=== 既存コンテナ停止・削除 ==='
sudo docker stop fx-ea-tpu 2>/dev/null || true
sudo docker rm   fx-ea-tpu 2>/dev/null || true

echo '=== イメージ取得: {image} ==='
sudo docker pull {image}

echo '=== コンテナ起動 ==='
sudo docker run -d \\
  --name fx-ea-tpu \\
  --privileged \\
  --net=host \\
  --shm-size=16g \\
  -v /dev/shm:/dev/shm \\
  --restart=unless-stopped \\
  -e PJRT_DEVICE=TPU \\
  -e TPU_NUM_DEVICES=4 \\
  -e TPU_ACCELERATOR_TYPE=v6e-4 \\
  -e MAX_PARALLEL=1 \\
  {image}

sleep 8
sudo docker ps --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'
echo '--- コンテナログ (最新20行) ---'
sudo docker logs fx-ea-tpu --tail 20 2>&1 || true
EXT_IP=$(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo '?')
echo ""
echo "=============================="
echo " ダッシュボード: http://$EXT_IP:8080"
echo "=============================="
"""


# ── 自動削除タイマー ──────────────────────────────────────────────────────────
_zone_used = None
_timer: threading.Timer = None

def _auto_delete(zone: str, hours: float):
    """指定時間後に VM を自動削除"""
    print(f"\n[自動削除] {hours}時間経過 → VM を削除します...")
    delete_vm(zone)
    print("[自動削除] 完了")


def schedule_auto_delete(zone: str, hours: float):
    global _timer
    secs = hours * 3600
    _timer = threading.Timer(secs, _auto_delete, args=(zone, hours))
    _timer.daemon = True
    _timer.start()
    t = time.strftime("%H:%M:%S", time.localtime(time.time() + secs))
    print(f"[自動削除] {hours}時間後 ({t}) に VM を自動削除します")
    print(f"           Ctrl+C で即時削除、--no-auto-delete で無効化")


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    global GCLOUD, _zone_used

    parser = argparse.ArgumentParser(
        description="TPU VM 起動 & Docker コンテナ自動管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            例:
              python tpu_start.py               # 3時間後に自動削除
              python tpu_start.py --hours 6     # 6時間後に自動削除
              python tpu_start.py --no-auto-delete  # 自動削除なし
        """)
    )
    parser.add_argument("--hours",          type=float, default=AUTO_DELETE_H,
                        help=f"自動削除までの時間 (デフォルト: {AUTO_DELETE_H}時間)")
    parser.add_argument("--no-auto-delete", action="store_true",
                        help="自動削除を無効にする")
    parser.add_argument("--zone",           default=None,
                        help="ゾーンを固定指定 (省略時は自動選択)")
    parser.add_argument("--image",          default=DOCKER_IMAGE,
                        help=f"Docker イメージ (デフォルト: {DOCKER_IMAGE})")
    args = parser.parse_args()

    GCLOUD = _find_gcloud()
    print(f"gcloud: {GCLOUD}")

    # プロジェクト設定
    run(["config", "set", "project", PROJECT], check=False)

    # ファイアウォールルール確保 (port 8080)
    ensure_firewall()

    # ── VM 作成 ──────────────────────────────────────────────────────────────
    zone = args.zone
    if zone:
        zones_to_try = [zone]
    else:
        zones_to_try = ZONES

    created_zone = None
    for z in zones_to_try:
        # 既存VM確認
        if vm_exists(z):
            print(f"[スキップ] {VM_NAME} は {z} に既に存在します")
            created_zone = z
            break
        if create_vm(z):
            print(f"[OK] {z} で VM 作成成功!")
            created_zone = z
            break
        print(f"  → {z} 空きなし、次のゾーンへ...")

    if not created_zone:
        print("[ERROR] 全ゾーンで VM 作成失敗。時間をおいて再試行してください。")
        sys.exit(1)

    _zone_used = created_zone

    # IP 取得 (起動待ち)
    print("VM IP 取得中...")
    ip = ""
    for _ in range(12):
        time.sleep(10)
        ip = get_vm_ip(created_zone)
        if ip:
            break
    if not ip:
        print("[ERROR] IP 取得失敗")
        sys.exit(1)
    print(f"[OK] VM IP: {ip}")

    # SSH 鍵伝播 (gcloud 経由で authorized_keys に登録)
    propagate_ssh_key(created_zone)

    # SSH 接続確認 (最大3分待つ)
    print("SSH 接続待ち...")
    for attempt in range(18):
        try:
            ok = ssh(ip, "echo 'SSH OK'", timeout=20)
            if ok:
                print("[OK] SSH 接続成功")
                break
        except Exception:
            pass
        time.sleep(10)
        print(f"  待機中... ({attempt+1}/18)")
    else:
        print("[ERROR] SSH 接続タイムアウト")
        sys.exit(1)

    # ── Docker セットアップ & コンテナ起動 ──────────────────────────────────
    print("\n=== Docker セットアップ & コンテナ起動 ===")
    ok = ssh(ip, _build_setup_sh(args.image), timeout=300)
    if not ok:
        print("[WARN] セットアップに問題が発生しました。ログを確認してください。")

    # ── 自動削除タイマー ─────────────────────────────────────────────────────
    if not args.no_auto_delete:
        schedule_auto_delete(created_zone, args.hours)

    print(f"""
======================================================
  FX AI EA TPU コンテナ 起動完了!
  VM   : {VM_NAME} @ {created_zone}
  IP   : {ip}
  ダッシュボード: http://{ip}:8080
  SSH  : ssh -i {SSH_KEY} {SSH_USER}@{ip}
  ログ : ssh 後 → sudo docker logs -f fx-ea-tpu
  停止 : python tpu_stop.py
======================================================
""")

    # Ctrl+C で即時削除
    _start = time.time()
    try:
        print("待機中... (Ctrl+C で即時削除)")
        while True:
            time.sleep(60)
            if _timer and not args.no_auto_delete:
                elapsed = time.time() - _start
                remain_h = max(0, args.hours - elapsed / 3600)
                remain_m = int(remain_h * 60)
                print(f"  稼働中... 自動削除まで残り {remain_m}分 ({remain_h:.1f}時間)")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 即時削除します...")
        if _timer:
            _timer.cancel()
        delete_vm(created_zone)


if __name__ == "__main__":
    main()
