"""
TPU VM 停止・削除スクリプト

使い方:
  python tpu_stop.py                  # VM 削除のみ
  python tpu_stop.py --container-stop # コンテナを graceful 停止してから削除
  python tpu_stop.py --list           # 実行中の TPU VM を一覧表示
"""

import subprocess, sys, argparse
from pathlib import Path

VM_NAME  = "fx-ea-tpu-v6e"
PROJECT  = "project-c7a2ed3f-0395-4b76-967"
SSH_KEY  = str(Path.home() / ".ssh" / "google_compute_engine")
SSH_USER = "yu"
ZONES    = [
    "us-central1-b", "us-central1-a", "us-central1-c",
    "us-east5-a", "us-east5-b", "us-east5-c",
    "europe-west4-a", "europe-west4-b",
]

def _find_gcloud() -> str:
    import shutil
    candidates = [
        shutil.which("gcloud"),
        str(Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"),
        "/usr/bin/gcloud",
        "/usr/local/bin/gcloud",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("gcloud が見つかりません")


def run(gcloud, cmd, capture=False, check=True):
    full = [gcloud] + cmd
    if capture:
        return subprocess.run(full, capture_output=True, text=True, check=False)
    return subprocess.run(full, check=check)


def find_running_vms(gcloud) -> list[tuple[str, str]]:
    """実行中の {VM_NAME} を (name, zone) のリストで返す"""
    found = []
    for zone in ZONES:
        r = run(gcloud, [
            "compute", "tpus", "tpu-vm", "list",
            f"--zone={zone}",
            "--format=value(name,state)",
        ], capture=True)
        for line in r.stdout.strip().splitlines():
            parts = line.split()
            if parts and parts[0] == VM_NAME:
                state = parts[1] if len(parts) > 1 else "UNKNOWN"
                found.append((VM_NAME, zone, state))
    return found


def stop_container(ip: str):
    """コンテナを graceful 停止"""
    print(f"[コンテナ停止] {ip} の fx-ea-tpu を停止中...")
    subprocess.run(
        ["ssh", "-i", SSH_KEY,
         "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ConnectTimeout=10",
         f"{SSH_USER}@{ip}", "bash"],
        input="sudo docker stop fx-ea-tpu 2>&1 && echo '[OK] コンテナ停止完了'",
        text=True, timeout=30,
    )


def get_ip(gcloud, zone) -> str:
    r = run(gcloud, [
        "compute", "tpus", "tpu-vm", "describe", VM_NAME,
        f"--zone={zone}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ], capture=True)
    return r.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="TPU VM 停止・削除")
    parser.add_argument("--container-stop", action="store_true",
                        help="コンテナを graceful 停止してから VM 削除")
    parser.add_argument("--list",           action="store_true",
                        help="実行中の TPU VM を一覧表示して終了")
    parser.add_argument("--zone",           default=None,
                        help="ゾーンを直接指定 (省略時は自動検索)")
    args = parser.parse_args()

    gcloud = _find_gcloud()
    run(gcloud, ["config", "set", "project", PROJECT], check=False)

    # ── 一覧表示 ──────────────────────────────────────────────────────────────
    if args.list:
        print(f"TPU VM '{VM_NAME}' を検索中...")
        vms = find_running_vms(gcloud)
        if vms:
            print(f"\n{'VM名':<20} {'ゾーン':<20} {'状態'}")
            print("-" * 55)
            for name, zone, state in vms:
                print(f"{name:<20} {zone:<20} {state}")
        else:
            print("実行中の VM が見つかりませんでした。")
        return

    # ── VM 検索 ───────────────────────────────────────────────────────────────
    if args.zone:
        zones_to_check = [args.zone]
    else:
        print(f"VM '{VM_NAME}' を検索中...")
        zones_to_check = ZONES

    target_zone = None
    for zone in zones_to_check:
        r = run(gcloud, [
            "compute", "tpus", "tpu-vm", "list",
            f"--zone={zone}", f"--filter=name={VM_NAME}",
            "--format=value(name)",
        ], capture=True)
        if VM_NAME in r.stdout:
            target_zone = zone
            print(f"[発見] {VM_NAME} @ {target_zone}")
            break

    if not target_zone:
        print(f"[INFO] VM '{VM_NAME}' が見つかりませんでした。既に削除済みの可能性があります。")
        return

    # ── コンテナ graceful 停止 ────────────────────────────────────────────────
    if args.container_stop:
        ip = get_ip(gcloud, target_zone)
        if ip:
            try:
                stop_container(ip)
            except Exception as e:
                print(f"[WARN] コンテナ停止失敗 (続行): {e}")

    # ── VM 削除 ───────────────────────────────────────────────────────────────
    print(f"[削除] {VM_NAME} ({target_zone}) を削除中...")
    r = run(gcloud, [
        "compute", "tpus", "tpu-vm", "delete", VM_NAME,
        f"--zone={target_zone}", "--quiet",
    ], check=False)

    if r.returncode == 0:
        print(f"[完了] VM '{VM_NAME}' を削除しました。課金は停止しています。")
    else:
        print(f"[ERROR] 削除失敗 (exit={r.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()
