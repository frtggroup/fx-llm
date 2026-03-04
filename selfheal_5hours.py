#!/usr/bin/env python3
"""
selfheal_3hours.py
============================================================
1. 60秒ごとに vast.ai インスタンスの学習状態を確認
2. 学習停止を検出 → S3からログを取得して原因解析
3. ソース修正 → GitHub push → GitHub Actions ビルド完了待機
4. vast.ai インスタンスを新イメージで再起動 (destroy+recreate)
5. 起動後にentrypoint.shが動いていなければ手動起動
6. 上記を3時間ループ
7. 最後にすべての vast.ai インスタンスを終了して終了する。
============================================================
"""

import os
import sys
import time
import json
import re
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = "https://frorit-2022.softether.net:18004"
S3_BUCKET    = "mix3"
S3_ACCESS_KEY= "mioroot"
S3_SECRET_KEY= "Yakrty1484!#"

GH_WORKFLOW  = "build-dok-ea5.yml"
GH_REPO      = "frtggroup/fx-llm"
IMAGE        = "frtgroup/fx-ea5:latest"

POLL_INTERVAL   = 60   # 秒
STALL_THRESHOLD = 180  # 秒 — この間ログ更新なし → 停止とみなす
MAX_BID_PER_HOUR = 0.50  # $/hr — これを超える bid では新インスタンスを作成しない
DURATION_SECONDS = 5 * 3600 # 5 hours

VAST_SSH_KEY = str(Path.home() / ".ssh/google_compute_engine")

# ── グローバル変数 (動的更新) ─────────────────────────────────────────────────
VAST_INSTANCE_ID = None
VAST_SSH_HOST    = None
VAST_SSH_PORT    = None

# 修正ヒント: エラーパターン → 対処関数名
ERROR_HINTS = [
    (r"CUDA out of memory",          "fix_oom"),
    (r"OOM",                         "fix_oom"),
    (r"RuntimeError.*size mismatch", "fix_size_mismatch"),
    (r"ValueError.*Sample larger",   "fix_sample_larger"),
    (r"KeyError",                    "fix_keyerror"),
    (r"stop\.flag",                  "fix_stop_flag"),
]

# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ssh(cmd: str, timeout: int = 30) -> tuple:
    """vast.ai インスタンスに SSH して cmd を実行。(exit_code, stdout+stderr)"""
    if not (VAST_SSH_HOST and VAST_SSH_PORT):
        return -1, "SSH設定未初期化 (インスタンスが未起動?)"
    full = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=yes",
        "-i", VAST_SSH_KEY,
        "-p", str(VAST_SSH_PORT),
        f"root@{VAST_SSH_HOST}",
        cmd,
    ]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout + 5)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "SSH timeout"
    except Exception as e:
        return -1, str(e)


def s3_client():
    import boto3, urllib3
    urllib3.disable_warnings()
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        verify=False,
    )


# ── vastai インスタンス管理 ───────────────────────────────────────────────────
def get_current_instance() -> dict | None:
    """vastai show instances --raw から現在のインスタンス情報を取得
    複数ある場合は一番安いものを残し、他は削除する"""
    r = subprocess.run(
        "vastai show instances --raw",
        shell=True, capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    try:
        instances = json.loads(r.stdout)
        if instances:
            valid_instances = []
            for inst in instances:
                to_delete = inst.get("id")
                # is_bid が True (Interruptible) でない仮想マシンは削除
                if not inst.get("is_bid", False):
                    if to_delete:
                        log(f"[DESTROY] 非InterruptibleなVMを検出。削除します: ID={to_delete}")
                        subprocess.run(f"vastai destroy instance {to_delete}", shell=True)
                else:
                    valid_instances.append(inst)

            if not valid_instances:
                return None

            # dph_total で昇順にソート（一番安いのが先頭）
            valid_instances.sort(key=lambda x: x.get("dph_total", 999))
            
            # 2つ目以降のインスタンスがあれば削除
            for inst in valid_instances[1:]:
                to_delete = inst.get("id")
                price = inst.get("dph_total", 0)
                if to_delete:
                    log(f"[DESTROY] 複数インスタンス起動を検出。高い方を削除します: ID={to_delete} (${price:.3f}/hr)")
                    subprocess.run(f"vastai destroy instance {to_delete}", shell=True)
            
            return valid_instances[0]
    except Exception:
        pass
    return None

def destroy_all_instances():
    """すべての vastai インスタンスを削除する"""
    log("[DESTROY] 終了処理: すべての vast.ai インスタンスを削除します")
    r = subprocess.run(
        "vastai show instances --raw",
        shell=True, capture_output=True, text=True
    )
    if r.returncode == 0:
        try:
            instances = json.loads(r.stdout)
            for inst in instances:
                inst_id = inst.get("id")
                if inst_id:
                    log(f"[DESTROY] インスタンス {inst_id} を削除中...")
                    subprocess.run(f"vastai destroy instance {inst_id}", shell=True)
        except Exception as e:
            log(f"[DESTROY] エラー: {e}")

def update_ssh_from_instance(inst: dict) -> bool:
    """インスタンス情報からグローバルSSH設定を更新
    ssh_host が null の場合は public_ipaddr + ports["22/tcp"] を使う"""
    global VAST_INSTANCE_ID, VAST_SSH_HOST, VAST_SSH_PORT
    try:
        VAST_INSTANCE_ID = inst["id"]
        ssh_host = inst.get("ssh_host") or ""
        ssh_port = inst.get("ssh_port") or 0
        if not ssh_host:
            ssh_host = inst.get("public_ipaddr", "")
            ports = inst.get("ports", {})
            tcp22 = ports.get("22/tcp", [])
            if tcp22:
                ssh_port = int(tcp22[0].get("HostPort", 0))
        VAST_SSH_HOST = ssh_host
        VAST_SSH_PORT = ssh_port
        return bool(VAST_SSH_HOST and VAST_SSH_PORT)
    except Exception:
        return False


def wait_for_instance_ssh(max_tries: int = 40) -> bool:
    """インスタンスのSSHポートが開くまで待機してSSH情報を更新"""
    loading_streak = 0
    OUTBID_TRIES = 20   # 20回 × 30秒 = 10分
    for i in range(1, max_tries + 1):
        time.sleep(30)
        inst = get_current_instance()
        if inst is None:
            loading_streak += 1
            log(f"  SSHポート待機中 [{i}/{max_tries}] (インスタンス未検出) streak={loading_streak}")
        else:
            status    = inst.get("actual_status", "loading")
            ssh_host  = inst.get("ssh_host", "")
            ssh_port  = inst.get("ssh_port", 0)
            log(f"  SSHポート待機中 [{i}/{max_tries}] status={status} {ssh_host}:{ssh_port}")
            if ssh_host and ssh_port and status == "running":
                update_ssh_from_instance(inst)
                log(f"[VAST] SSH接続可能: {VAST_SSH_HOST}:{VAST_SSH_PORT}")
                return True
            if status == "loading":
                loading_streak += 1
            else:
                loading_streak = 0

        # 10分間 loading/なし → outbid 対処
        if loading_streak >= OUTBID_TRIES:
            log(f"[OUTBID] {loading_streak}回連続loading → 入札価格引き上げを試みる")
            if inst and VAST_INSTANCE_ID:
                recovered = try_raise_bid(VAST_INSTANCE_ID)
                if recovered:
                    loading_streak = 0
                    continue
            log("[OUTBID] bid引き上げ失敗 → 新インスタンスを作成")
            return False  
    return False


def try_raise_bid(instance_id: int) -> bool:
    """現在のインスタンスの bid を最安値まで段階的に引き上げ"""
    _, min_dph = find_cheapest_h200_offer()
    if not min_dph:
        return False

    inst = get_current_instance()
    if not inst:
        return False
    current_bid = inst.get("dph_total", 0.0)

    for step in range(1, 6):
        new_bid = round(current_bid + step * 0.02, 3)
        if new_bid > min_dph + 0.20:
            break
        log(f"[BID] bid引き上げ試行: ${current_bid:.3f} → ${new_bid:.3f} (最安値=${min_dph:.3f})")
        r = subprocess.run(
            f"vastai change bid {instance_id} --price {new_bid}",
            shell=True, capture_output=True, text=True
        )
        out = (r.stdout + r.stderr).strip()
        log(f"[BID] change bid: {out[:80]}")
        time.sleep(30)
        inst = get_current_instance()
        if inst and inst.get("actual_status") == "running":
            update_ssh_from_instance(inst)
            log(f"[BID] bid引き上げ成功! ${new_bid:.3f}/hr で稼働")
            return True

    log(f"[BID] bid引き上げ上限到達 → 新インスタンス作成へ")
    return False


def find_cheapest_h200_offer() -> tuple:
    all_offers = []
    for gpu in ("H200_NVL", "H100_NVL"):
        r = subprocess.run(
            f'vastai search offers "gpu_name={gpu} num_gpus=1 rentable=True" --interruptible --order dph_base --raw',
            shell=True, capture_output=True, text=True
        )
        try:
            offers = json.loads(r.stdout)
            all_offers.extend(offers)
        except Exception:
            pass
    if not all_offers:
        return None, 0.0
    cheapest = min(all_offers, key=lambda o: o.get("dph_base", 999))
    log(f"[OFFER] 最安値: {cheapest.get('gpu_name')} ${cheapest.get('dph_base', 0):.3f}/hr @ {cheapest.get('geolocation','?')}")
    return cheapest["id"], cheapest.get("dph_base", 0.0)


def create_new_instance(offer_id: int, bid: float) -> int | None:
    cmd = (
        f"vastai create instance {offer_id} "
        f"--image {IMAGE} --disk 50 "
        f"--bid_price {bid:.3f} --raw"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log(f"[VAST] create output: {r.stdout.strip()[:200]}")
    try:
        result = json.loads(r.stdout)
        new_id = result.get("new_contract") or result.get("id")
        if new_id:
            return int(new_id)
    except Exception:
        pass
    return None


# ── S3 ───────────────────────────────────────────────────────────────────────
def get_log_from_s3() -> str | None:
    try:
        c = s3_client()
        resp = c.list_objects_v2(Bucket=S3_BUCKET, Prefix="log/")
        keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".log")]
        if not keys:
            return None
        combined = []
        for key in keys:
            try:
                obj = c.get_object(Bucket=S3_BUCKET, Key=key)
                text = obj["Body"].read().decode("utf-8", errors="replace")
                combined.append(f"=== {key} ===\n{text[-8000:]}")
            except Exception:
                pass
        return "\n\n".join(combined) if combined else None
    except Exception as e:
        log(f"[WARN] S3 log 取得失敗: {e}")
        return None


def get_s3_object_count() -> int:
    try:
        c = s3_client()
        resp = c.list_objects_v2(Bucket=S3_BUCKET)
        return resp.get("KeyCount", 0)
    except Exception:
        return -1


def get_log_last_modified_age() -> float:
    try:
        c = s3_client()
        resp = c.list_objects_v2(Bucket=S3_BUCKET, Prefix="log/")
        objs = resp.get("Contents", [])
        if not objs:
            return float("inf")
        latest = max(objs, key=lambda o: o["LastModified"])
        age = (datetime.now(timezone.utc) - latest["LastModified"]).total_seconds()
        return age
    except Exception:
        return float("inf")


# ── 学習状態確認 ──────────────────────────────────────────────────────────────
def is_training_alive() -> tuple:
    code, out = ssh("pgrep -af '[r]un_train' | head -5", timeout=20)
    if code == 0 and "run_train" in out:
        return True, out
    return False, out


def ensure_training_running() -> bool:
    alive, detail = is_training_alive()
    if alive:
        log(f"[OK] 学習プロセス確認済み: {detail[:60]}")
        return True
    log("[!] 学習未起動 → entrypoint.sh を起動")
    code, out = ssh(
        "rm -f /workspace/stop.flag; "
        "nohup bash /workspace/entrypoint.sh >> /var/log/entrypoint.log 2>&1 &",
        timeout=30
    )
    log(f"[BOOT] entrypoint起動: exit={code} {out[:80]}")
    return code == 0


# ── 修正ロジック ──────────────────────────────────────────────────────────────
def analyze_log(log_text: str) -> str | None:
    for pattern, fix_fn in ERROR_HINTS:
        if re.search(pattern, log_text, re.IGNORECASE):
            return fix_fn
    return None


def fix_oom() -> bool:
    path = Path("f:/FX/fx-ea5/run_train.py")
    if not path.exists(): return False
    text = path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"(\"xlarge\":\s*\{[^}]*?\"par\":\s*)(\d+)",
        lambda m: m.group(1) + str(max(20, int(m.group(2)) - 8)),
        text,
        count=1,
    )
    if new_text == text:
        log("[WARN] fix_oom: パターン未検出 → スキップ")
        return False
    path.write_text(new_text, encoding="utf-8")
    log("[FIX] fix_oom: H200 xlarge par を削減")
    return True


def fix_sample_larger() -> bool:
    path = Path("f:/FX/fx-ea5/run_train.py")
    if not path.exists(): return False
    text = path.read_text(encoding="utf-8")
    if "size_targets" not in text:
        log("[WARN] fix_sample_larger: size_targets 未検出")
        return False
    new_text = re.sub(
        r"(size_targets\s*=\s*\[.*?\])",
        "size_targets = []  # fix: was crashing random.sample",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    log("[FIX] fix_sample_larger: size_targets をクリア")
    return True


def fix_stop_flag() -> bool:
    code, out = ssh("rm -f /workspace/stop.flag && echo OK")
    log(f"[FIX] stop.flag 削除: {out}")
    return code == 0


def fix_generic_restart() -> bool:
    fix_stop_flag()
    ssh("pkill -f run_train.py || true")
    return True


FIXERS = {
    "fix_oom":          fix_oom,
    "fix_size_mismatch": fix_generic_restart,
    "fix_sample_larger": fix_sample_larger,
    "fix_keyerror":     fix_generic_restart,
    "fix_stop_flag":    fix_stop_flag,
}


# ── GitHub Actions ────────────────────────────────────────────────────────────
def git_push_and_build(message: str) -> bool:
    cmds = [
        "git -C f:/FX add fx-ea5/run_train.py DOK/entrypoint_ea5.sh selfheal_monitor.py",
        f'git -C f:/FX commit -m "{message}" --allow-empty',
        "git -C f:/FX push origin master",
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="f:/FX")
        if r.returncode != 0:
            log(f"[ERROR] git コマンド失敗: {cmd}\n{r.stderr}")
            return False
        log(f"[GIT] OK: {cmd[:60]}")
    return True


def wait_for_build(timeout_min: int = 30) -> bool:
    import urllib.request
    log(f"[BUILD] GitHub Actions ビルド完了待機 (最大 {timeout_min}分)...")
    deadline = time.time() + timeout_min * 60
    time.sleep(30)
    url = f"https://api.github.com/repos/{GH_REPO}/actions/runs?per_page=3"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            runs = data.get("workflow_runs", [])
            ea5_runs = [r for r in runs if GH_WORKFLOW in r.get("path", "")]
            if not ea5_runs:
                ea5_runs = runs
            if ea5_runs:
                run = ea5_runs[0]
                status     = run.get("status", "")
                conclusion = run.get("conclusion") or ""
                log(f"[BUILD] status={status} conclusion={conclusion}")
                if status == "completed":
                    return conclusion == "success"
        except Exception as e:
            log(f"[WARN] ビルド確認失敗: {e}")
        time.sleep(30)
    log("[ERROR] ビルドタイムアウト")
    return False


# ── vast.ai 再起動 ────────────────────────────────────────────────────────────
def restart_vast_instance() -> bool:
    global VAST_INSTANCE_ID

    if VAST_INSTANCE_ID:
        log(f"[VAST] インスタンス {VAST_INSTANCE_ID} を削除中...")
        r = subprocess.run(
            f"vastai destroy instance {VAST_INSTANCE_ID}",
            shell=True, capture_output=True, text=True
        )
        out = (r.stdout + r.stderr).strip()
        log(f"[VAST] destroy: {out[:120]}")
        time.sleep(15)

    log("[VAST] 最安H200 NVL offer 検索中...")
    offer_id, dph = find_cheapest_h200_offer()
    if not offer_id:
        log("[ERROR] H200 NVL offer が見つかりません → 60秒後にリトライ")
        time.sleep(60)
        offer_id, dph = find_cheapest_h200_offer()
        if not offer_id:
            return False

    bid = round(dph + 0.10, 3)
    log(f"[VAST] 最安offer: {offer_id} (${dph:.3f}/hr) → bid=${bid:.3f}")

    if bid > MAX_BID_PER_HOUR:
        log(f"[SKIP] bid=${bid:.3f} > 上限${MAX_BID_PER_HOUR:.2f} → インスタンス作成をスキップ (60秒後に再確認)")
        time.sleep(60)
        return restart_vast_instance()

    new_id = create_new_instance(offer_id, bid)
    if new_id:
        VAST_INSTANCE_ID = new_id
        log(f"[VAST] 新インスタンス起動 ID={new_id} (offer={offer_id})")
    else:
        log("[WARN] 新インスタンスIDが取得できませんでした")

    result = wait_for_instance_ssh(max_tries=40)
    if not result:
        log("[VAST] SSH待機失敗 (outbid?) → 新インスタンスで再試行")
        return restart_vast_instance()
    return result


def verify_training_resumed(retries: int = 8) -> bool:
    for i in range(retries):
        alive, detail = is_training_alive()
        if alive:
            log(f"[OK] 学習再開確認: {detail[:80]}")
            return True
        log(f"[WAIT] 学習未確認 ({i+1}/{retries}): {detail[:60]}")
        time.sleep(30)
    return False


# ── メインループ ──────────────────────────────────────────────────────────────
def main():
    global VAST_INSTANCE_ID, VAST_SSH_HOST, VAST_SSH_PORT

    log("=" * 60)
    log("FX-EA5 自己修復モニター(3時間リミット付き) 起動")
    log(f"  S3              : {S3_ENDPOINT}/{S3_BUCKET}/log/")
    log(f"  ポーリング間隔   : {POLL_INTERVAL}秒")
    log("=" * 60)

    start_time = time.time()

    inst = get_current_instance()
    if inst:
        update_ssh_from_instance(inst)
        log(f"[INIT] 現在のインスタンス: ID={VAST_INSTANCE_ID} "
            f"({VAST_SSH_HOST}:{VAST_SSH_PORT}) status={inst.get('actual_status')}")
        if inst.get("actual_status") == "running":
            log("[INIT] インスタンス稼働中 → 学習プロセス確認...")
            time.sleep(10)
            ensure_training_running()
        elif inst.get("actual_status") in ("exited", "stopped"):
            log(f"[INIT] インスタンスが {inst.get('actual_status')} → 再起動")
            restart_vast_instance()
            time.sleep(60)
            ensure_training_running()
    else:
        log("[INIT] 実行中インスタンスなし → 新規作成")
        restart_vast_instance()
        time.sleep(60)
        ensure_training_running()

    obj_count = get_s3_object_count()
    log(f"[INIT] S3 mix3 objects: {obj_count}")

    heal_count = 0
    consecutive_failures = 0
    loading_since = None  

    while True:
        try:
            if time.time() - start_time >= DURATION_SECONDS:
                log(f"[FINISH] {DURATION_SECONDS/3600}時間が経過しました。全インスタンスを終了します。")
                destroy_all_instances()
                break

            inst = get_current_instance()
            inst_status = inst.get("actual_status", "unknown") if inst else "none"
            if inst:
                update_ssh_from_instance(inst)

            alive, detail = is_training_alive()
            log_age = get_log_last_modified_age()
            obj_count = get_s3_object_count()
            log(f"[CHECK] inst={inst_status} alive={alive} log_age={log_age:.0f}s heals={heal_count}")
            if inst_status == "loading":
                if loading_since is None:
                    loading_since = time.time()
                loading_elapsed = time.time() - loading_since
                log(f"[WAIT] インスタンスloading中 ({loading_elapsed:.0f}秒) → 待機")

                if loading_elapsed > 600:
                    log(f"[OUTBID] loading {loading_elapsed:.0f}秒超 → bid引き上げ試行")
                    if VAST_INSTANCE_ID and inst:
                        recovered = try_raise_bid(VAST_INSTANCE_ID)
                        if recovered:
                            loading_since = None
                            time.sleep(POLL_INTERVAL)
                            continue
                    log("[OUTBID] bid引き上げ失敗 → 新インスタンス作成")
                    heal_count += 1
                    restart_vast_instance()
                    loading_since = None
                    time.sleep(60)
                    ensure_training_running()

                time.sleep(POLL_INTERVAL)
                continue
            else:
                loading_since = None

            if inst_status in ("exited", "stopped", "none"):
                log(f"[!] インスタンス状態異常: {inst_status} → 再起動")
                heal_count += 1
                restart_vast_instance()
                time.sleep(60)
                ensure_training_running()
                time.sleep(POLL_INTERVAL)
                continue

            stalled = False
            
            if alive and log_age > STALL_THRESHOLD and log_age != float("inf"):
                log(f"[WARN] プロセス稼働中だがS3ログ遅延 ({log_age:.0f}s) → ローカルログ確認")
                code_stat, out_stat = ssh("stat -c %Y /workspace/train_run.log 2>/dev/null")
                if code_stat == 0 and out_stat.strip().isdigit():
                    local_age = time.time() - int(out_stat.strip())
                    if local_age > STALL_THRESHOLD:
                        log(f"[ERROR] ローカルログも更新停止 ({local_age:.0f}s) → ハングと判定")
                        alive = False
                        stalled = True
                    else:
                        log(f"[INFO] ローカルログは更新中 (age: {local_age:.0f}s) → S3遅延のみ様子見")
                else:
                    log("[WARN] ローカルログ日時取得失敗 → ハングと判定")
                    alive = False
                    stalled = True
            
            if alive:
                consecutive_failures = 0
                time.sleep(POLL_INTERVAL)
                continue

            if not alive and not stalled and log_age <= STALL_THRESHOLD:
                # この条件は S3が早いがプロセスが死んだ場合など
                pass

            reason = "プロセス停止" if not alive else f"ログ更新停滞 ({log_age:.0f}s)"
            log(f"[!] 学習異常検出: {reason}")
            heal_count += 1

            log_text = get_log_from_s3() or ""
            if not log_text.strip():
                log("[WARN] S3ログが空のため、直接SSHでログを取得します")
                _, out_train = ssh("tail -n 200 /workspace/train_run.log 2>/dev/null")
                _, out_s3err = ssh("tail -n 50 /var/log/s3_upload.log 2>/dev/null")
                log_text = f"=== SSH /workspace/train_run.log ===\n{out_train}\n=== SSH /var/log/s3_upload.log ===\n{out_s3err}"
                log(f"[ANALYZE] SSHログプレビュー: {log_text[-300:]}")
            
            fix_fn_name = analyze_log(log_text)
            log(f"[ANALYZE] 検出エラー: {fix_fn_name or '不明 → 汎用再起動'}")

            code_changed = False
            if fix_fn_name and fix_fn_name in FIXERS:
                fixer = FIXERS[fix_fn_name]
                try:
                    changed = fixer()
                    code_changed = changed and fix_fn_name in ("fix_oom", "fix_sample_larger")
                except Exception as e:
                    log(f"[WARN] fixer 失敗: {e}")

            if code_changed:
                msg = f"selfheal: {fix_fn_name} (heal #{heal_count})"
                pushed = git_push_and_build(msg)
                if pushed:
                    build_ok = wait_for_build(timeout_min=35)
                    if not build_ok:
                        log("[ERROR] ビルド失敗 → 現行イメージで再起動")
                else:
                    log("[WARN] push 失敗 → 現行イメージで再起動")
                restart_vast_instance()
                time.sleep(60)
                ensure_training_running()
            else:
                fix_stop_flag()
                code, out = ssh(
                    "rm -f /workspace/stop.flag; "
                    "pkill -f run_train.py 2>/dev/null || true; sleep 3; "
                    "nohup bash /workspace/entrypoint.sh >> /var/log/entrypoint.log 2>&1 &",
                    timeout=30
                )
                log(f"[FIX] コンテナ内再起動: exit={code} {out[:80]}")

            time.sleep(60)
            recovered = verify_training_resumed(retries=8)
            if recovered:
                log(f"[OK] 回復完了 (heal #{heal_count})")
                consecutive_failures = 0
            else:
                log(f"[WARN] 回復確認できず (heal #{heal_count})")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    log("[ERROR] 連続3回回復失敗 → インスタンス強制再起動")
                    restart_vast_instance()
                    time.sleep(60)
                    ensure_training_running()
                    consecutive_failures = 0

        except KeyboardInterrupt:
            log("中断されました")
            destroy_all_instances()
            sys.exit(0)
        except Exception as e:
            log(f"[ERROR] メインループ例外: {e}")
            time.sleep(POLL_INTERVAL)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
