#!/usr/bin/env python3
"""warmup_10vms.py — 複数TPU VMを並列起動してXLA全パターンを高速コンパイル

使い方:
  python warmup_10vms.py              # 10台起動→warmup監視→9台削除
  python warmup_10vms.py --n 5        # 5台で実行
  python warmup_10vms.py --dry-run    # テスト (VM操作なし)
  python warmup_10vms.py --keep 2     # VM-2を残す (デフォルト: 0)

仕組み:
  - 各VMがWARMUP_ONLY=1で起動 → warmup_xla.py でS3クレームを使って分散コンパイル
  - 10台×4チップ=40チップで522パターンを並列処理 (理論上: 単独の約10倍速)
  - S3の warmup_claims フォルダが5分間空になったら完了判定
  - KEEP_IDX のVMを残して他を削除
"""
import argparse, base64, concurrent.futures, subprocess, sys, time
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────────────────────────────
N_VMS      = 10
VM_PREFIX  = "fx-ea-wu"
ACCEL_TYPE = "v5litepod-4"
VERSION    = "tpu-ubuntu2204-base"
IMAGE      = "frtgroup/fx-ea:latest"
CONTAINER  = "fx-ea-warmup"

# v5litepod-4 が確保しやすいゾーン (VM-i は ZONES[i % len] から順に試す)
ZONES = [
    "us-west4-a", "us-west4-b", "us-west4-c",
    "us-east1-c", "us-east1-d",
    "us-east5-a", "us-east5-b", "us-east5-c",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
]

S3_ENDPOINT   = "https://frorit-2022.softether.net:18004"
S3_ACCESS_KEY = "mioroot"
S3_SECRET_KEY = "Yakrty1484!#"
S3_BUCKET     = "fxea"
S3_PREFIX     = "mix"

DONE_IDLE_SEC = 300  # クレームが0件になってから何秒待つか (5分)


# ── ユーティリティ ─────────────────────────────────────────────────────────────
def _run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def vm_name(idx: int) -> str:
    return f"{VM_PREFIX}-{idx}"


def _env_b64(warmup_only: bool = True) -> str:
    """S3_SECRET_KEY の !# をシェルエスケープ問題なしで渡すためbase64エンコード"""
    content = "\n".join([
        f"S3_ENDPOINT={S3_ENDPOINT}",
        f"S3_ACCESS_KEY={S3_ACCESS_KEY}",
        f"S3_SECRET_KEY={S3_SECRET_KEY}",
        f"S3_BUCKET={S3_BUCKET}",
        f"S3_PREFIX={S3_PREFIX}",
        "DEVICE_TYPE=TPU",
        "PJRT_DEVICE=TPU",
        "TPU_NUM_DEVICES=4",
        "TPU_ACCELERATOR_TYPE=v5litepod-4",
        "MAX_PARALLEL=4",
        f"WARMUP_ONLY={'1' if warmup_only else '0'}",
    ])
    return base64.b64encode(content.encode()).decode()


# ── VM 作成 ────────────────────────────────────────────────────────────────────
def create_vm(idx: int, dry_run: bool) -> tuple:
    """1台のTPU VMを作成/起動。成功したゾーンを (idx, zone) で返す。"""
    name = vm_name(idx)
    # 各VMをずらしたゾーンから試す (ラウンドロビンで負荷分散)
    zone_order = ZONES[idx % len(ZONES):] + ZONES[:idx % len(ZONES)]

    for zone in zone_order:
        state = _run(
            f"gcloud compute tpus tpu-vm describe {name} --zone={zone} "
            f"--format=value(state) 2>/dev/null"
        ).stdout.strip()

        if state == "READY":
            print(f"[VM-{idx}] {name}@{zone} 既存・起動済み", flush=True)
            return idx, zone

        elif state == "STOPPED":
            print(f"[VM-{idx}] {zone} 停止中→起動", flush=True)
            if dry_run:
                return idx, zone
            if _run(f"gcloud compute tpus tpu-vm start {name} --zone={zone}").returncode == 0:
                print(f"[VM-{idx}] {name}@{zone} 起動成功", flush=True)
                return idx, zone

        elif not state:  # 存在しない
            print(f"[VM-{idx}] {zone} で新規作成中...", flush=True)
            if dry_run:
                return idx, zone
            res = _run(
                f"gcloud compute tpus tpu-vm create {name} --zone={zone} "
                f"--accelerator-type={ACCEL_TYPE} --version={VERSION}"
            )
            if res.returncode == 0:
                print(f"[VM-{idx}] {name}@{zone} 作成成功", flush=True)
                return idx, zone
            else:
                print(f"[VM-{idx}] {zone} 失敗: {res.stderr.strip()[:100]}", flush=True)

    print(f"[VM-{idx}] 全ゾーン失敗", flush=True)
    return idx, None


# ── コンテナデプロイ ───────────────────────────────────────────────────────────
def deploy(idx: int, zone: str, dry_run: bool) -> bool:
    """VMにSSHしてWARMUP_ONLY=1でコンテナをデプロイ。"""
    name    = vm_name(idx)
    env_b64 = _env_b64()

    # base64デコードでenv fileを作成 → !# のシェルエスケープ問題を完全回避
    ssh_cmd = (
        f"gcloud compute tpus tpu-vm ssh {name} --zone={zone} "
        f"--strict-host-key-checking=no --command="
        f"\"echo {env_b64} | base64 -d > /tmp/fx-ea.env && "
        f'sudo docker stop {CONTAINER} 2>/dev/null || true && '
        f'sudo docker rm   {CONTAINER} 2>/dev/null || true && '
        f'sudo docker pull {IMAGE} && '
        f"sudo docker run -d --name {CONTAINER} --privileged --net=host "
        f"-v /workspace:/workspace --env-file /tmp/fx-ea.env {IMAGE} && "
        f"echo DEPLOYED_OK\""
    )

    if dry_run:
        print(f"[VM-{idx}] DRY: deploy {name}@{zone}", flush=True)
        return True

    print(f"[VM-{idx}] {name}@{zone} SSH接続待機 (20秒)...", flush=True)
    time.sleep(20)

    res = _run(ssh_cmd)
    ok  = res.returncode == 0 and "DEPLOYED_OK" in res.stdout
    if ok:
        print(f"[VM-{idx}] デプロイ完了", flush=True)
    else:
        print(f"[VM-{idx}] デプロイ失敗:\n{res.stderr.strip()[:300]}", flush=True)
    return ok


# ── warmup 完了監視 ────────────────────────────────────────────────────────────
def _count_claims() -> int:
    """S3 warmup_claims フォルダのオブジェクト数。-1=エラー"""
    try:
        import boto3, urllib3, botocore.config
        urllib3.disable_warnings()
        s3 = boto3.client(
            "s3", endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY,
            verify=False, config=botocore.config.Config(max_pool_connections=5)
        )
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/warmup_claims/")
        return resp.get("KeyCount", 0)
    except Exception as e:
        print(f"[MONITOR] S3確認失敗: {e}", flush=True)
        return -1


def wait_warmup_done(dry_run: bool):
    """S3クレームを30秒ごとに監視。{DONE_IDLE_SEC}秒間クレーム0件で完了判定。"""
    if dry_run:
        print("[MONITOR] DRY: 3秒待機でwarmup完了シミュレート", flush=True)
        time.sleep(3)
        return

    print(f"[MONITOR] warmup監視開始 (クレームが{DONE_IDLE_SEC}秒間=0件→完了)", flush=True)
    started       = False
    last_nonzero  = None
    last_count    = -99

    while True:
        count = _count_claims()
        ts    = datetime.now().strftime("%H:%M:%S")

        if count != last_count:
            print(f"[MONITOR] {ts} アクティブクレーム: {count}件", flush=True)
            last_count = count

        if count > 0:
            started      = True
            last_nonzero = time.time()
        elif count == 0:
            if not started:
                print(f"[MONITOR] {ts} warmup未開始 (クレームなし)... 待機", flush=True)
            else:
                idle = time.time() - (last_nonzero or time.time())
                if idle >= DONE_IDLE_SEC:
                    print(f"[MONITOR] warmup完了!", flush=True)
                    return

        time.sleep(30)


# ── VM 削除 ────────────────────────────────────────────────────────────────────
def delete_vm(idx: int, zone: str, dry_run: bool) -> bool:
    name = vm_name(idx)
    print(f"[CLEANUP] {name}@{zone} 削除中...", flush=True)
    if dry_run:
        print(f"[CLEANUP] DRY: delete {name}@{zone}", flush=True)
        return True
    res = _run(f"gcloud compute tpus tpu-vm delete {name} --zone={zone} --quiet")
    if res.returncode == 0:
        print(f"[CLEANUP] {name}@{zone} 削除完了", flush=True)
        return True
    print(f"[CLEANUP] {name}@{zone} 削除失敗: {res.stderr.strip()[:100]}", flush=True)
    return False


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="VM操作をスキップ (テスト用)")
    parser.add_argument("--n",    type=int, default=N_VMS,  help=f"VM台数 (デフォルト: {N_VMS})")
    parser.add_argument("--keep", type=int, default=0,       help="残すVMのインデックス (デフォルト: 0)")
    args = parser.parse_args()

    n, keep = args.n, args.keep
    print(f"=== warmup_10vms.py ({'DRY RUN' if args.dry_run else '本番'}) ===")
    print(f"[*] {n}台のTPU {ACCEL_TYPE} を並列起動、XLA 522パターンを分散コンパイル")
    print(f"[*] 完了後: VM-{keep} ({vm_name(keep)}) を残して他 {n-1}台を削除\n")

    # ── Step 1: VM並列作成 ──────────────────────────────────────────────────────
    print(f"[STEP 1] {n}台のVM並列作成", flush=True)
    vm_zones: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(create_vm, i, args.dry_run): i for i in range(n)}
        for fut in concurrent.futures.as_completed(futs):
            idx, zone = fut.result()
            if zone:
                vm_zones[idx] = zone

    print(f"\n[STEP 1完了] 起動成功: {len(vm_zones)}/{n}台")
    for i, z in sorted(vm_zones.items()):
        print(f"  VM-{i}: {vm_name(i)}@{z}")

    if not vm_zones:
        print("[ERROR] VMが1台も起動できませんでした")
        sys.exit(1)

    # ── Step 2: コンテナ並列デプロイ ────────────────────────────────────────────
    print(f"\n[STEP 2] {len(vm_zones)}台にWARMUP_ONLY=1でコンテナをデプロイ", flush=True)
    deployed: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(deploy, i, vm_zones[i], args.dry_run): i for i in vm_zones}
        for fut in concurrent.futures.as_completed(futs):
            idx = futs[fut]
            if fut.result():
                deployed[idx] = vm_zones[idx]

    print(f"\n[STEP 2完了] デプロイ成功: {len(deployed)}/{len(vm_zones)}台")

    # ── Step 3: warmup完了監視 ──────────────────────────────────────────────────
    print(f"\n[STEP 3] S3クレームを監視してwarmup完了を待機...", flush=True)
    wait_warmup_done(args.dry_run)

    # ── Step 4: keep以外のVMを削除 ──────────────────────────────────────────────
    to_delete = {i: z for i, z in vm_zones.items() if i != keep}
    print(f"\n[STEP 4] VM-{keep} を残して他 {len(to_delete)}台を並列削除", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(delete_vm, i, z, args.dry_run) for i, z in to_delete.items()]
        concurrent.futures.wait(futs)

    keep_zone = vm_zones.get(keep, "unknown")
    print(f"\n=== 完了 ===")
    print(f"残存VM : {vm_name(keep)}@{keep_zone}")
    print(f"削除VM : {len(to_delete)}台")
    train_b64 = _env_b64(warmup_only=False)
    print(f"\n[次のステップ] 残存VMで学習を開始する場合:")
    print(f"  gcloud compute tpus tpu-vm ssh {vm_name(keep)} --zone={keep_zone} --command=\"\\")
    print(f"    sudo docker stop {CONTAINER} 2>/dev/null; sudo docker rm {CONTAINER} 2>/dev/null; \\")
    print(f"    echo {train_b64} | base64 -d > /tmp/fx-ea.env && \\")
    print(f"    sudo docker run -d --name {CONTAINER} --privileged --net=host \\")
    print(f"    -v /workspace:/workspace --env-file /tmp/fx-ea.env {IMAGE}\"")



if __name__ == "__main__":
    main()
