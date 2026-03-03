#!/usr/bin/env python3
"""
selfheal_monitor.py  —  FX-EA5 自己修復モニター
============================================================
1. 60秒ごとに vast.ai インスタンスの学習状態を確認
2. 学習停止を検出 → S3からログを取得して原因解析
3. ソース修正 → GitHub push → GitHub Actions ビルド完了待機
4. vast.ai インスタンスを新イメージで再起動
5. 上記を無限ループ
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
VAST_INSTANCE_ID = 32301153
VAST_SSH_KEY     = str(Path.home() / ".ssh/google_compute_engine")
VAST_SSH_HOST    = "93.91.156.87"
VAST_SSH_PORT    = 58238

S3_ENDPOINT  = "https://frorit-2022.softether.net:18004"
S3_BUCKET    = "mix3"
S3_ACCESS_KEY= "mioroot"
S3_SECRET_KEY= "Yakrty1484!#"

GH_WORKFLOW  = "build-dok-ea5.yml"
GH_REPO      = "frtggroup/fx-llm"

POLL_INTERVAL   = 60   # 秒
STALL_THRESHOLD = 180  # 秒 — この間ログ更新なし → 停止とみなす

LOG_S3_KEY  = "log/train_run_{node}.log"   # {node} はホスト名

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


def ssh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """vast.ai インスタンスに SSH して cmd を実行。(exit_code, stdout+stderr)"""
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


def get_log_from_s3(node_id: str = "") -> str | None:
    """S3 から最新のトレーニングログを取得"""
    try:
        c = s3_client()
        # 全ノードのログを結合して返す
        resp = c.list_objects_v2(Bucket=S3_BUCKET, Prefix="log/")
        keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".log")]
        if not keys:
            return None
        combined = []
        for key in keys:
            try:
                obj = c.get_object(Bucket=S3_BUCKET, Key=key)
                text = obj["Body"].read().decode("utf-8", errors="replace")
                combined.append(f"=== {key} ===\n{text[-8000:]}")  # 末尾8KB
            except Exception:
                pass
        return "\n\n".join(combined) if combined else None
    except Exception as e:
        log(f"[WARN] S3 log 取得失敗: {e}")
        return None


def get_log_from_ssh() -> str | None:
    """SSH で直接ログを取得 (S3 が空の場合のフォールバック)"""
    code, out = ssh("tail -200 /workspace/train_run.log 2>/dev/null || echo ''")
    return out if out else None


def is_training_alive() -> tuple[bool, str]:
    """
    run_train.py プロセスが生きているか確認。
    Returns (alive: bool, detail: str)
    """
    code, out = ssh("pgrep -af run_train.py | head -5", timeout=20)
    if code == 0 and "run_train" in out:
        return True, out
    return False, out


def get_log_last_modified_age() -> float:
    """S3 ログの最終更新からの経過秒数 (取得失敗時は inf)"""
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


# ── 修正ロジック ──────────────────────────────────────────────────────────────
def analyze_log(log_text: str) -> str | None:
    """ログからエラーパターンを検出して対処関数名を返す"""
    for pattern, fix_fn in ERROR_HINTS:
        if re.search(pattern, log_text, re.IGNORECASE):
            return fix_fn
    return None


def fix_oom() -> bool:
    """CUDA OOM: run_train.py の MAX_PARALLEL を下げる"""
    path = Path("f:/FX/fx-ea5/run_train.py")
    text = path.read_text(encoding="utf-8")
    # H200 tier の par 値を 39 → 30 に下げる
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
    """ValueError: Sample larger than population → size_targets をリセット"""
    path = Path("f:/FX/fx-ea5/run_train.py")
    text = path.read_text(encoding="utf-8")
    # size_targets を N_GROUPS 以下に制限する箇所を確認
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
    """stop.flag が残っていれば削除 (SSH経由)"""
    code, out = ssh("rm -f /workspace/stop.flag && echo OK")
    log(f"[FIX] stop.flag 削除: {out}")
    return code == 0


def fix_generic_restart() -> bool:
    """汎用: stop.flag 削除 + プロセス強制終了 (コンテナ外からリセット)"""
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
    """変更をコミット・プッシュして GitHub Actions ビルドをトリガー"""
    cmds = [
        "git -C f:/FX add fx-ea5/run_train.py DOK/entrypoint_ea5.sh",
        f'git -C f:/FX commit -m "{message}" --allow-empty',
        "git -C f:/FX push origin master",
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="f:/FX")
        if r.returncode != 0:
            log(f"[ERROR] git コマンド失敗: {cmd}\n{r.stderr}")
            return False
        log(f"[GIT] {cmd.split()[0:3]} → OK")
    return True


def wait_for_build(timeout_min: int = 30) -> bool:
    """GitHub Actions ビルド完了を待機。成功なら True"""
    log(f"[BUILD] GitHub Actions ビルド完了待機 (最大 {timeout_min}分)...")
    deadline = time.time() + timeout_min * 60
    time.sleep(30)  # ビルド開始まで少し待つ
    while time.time() < deadline:
        r = subprocess.run(
            f"gh run list --repo {GH_REPO} --workflow {GH_WORKFLOW} --limit 1 --json status,conclusion,headSha",
            shell=True, capture_output=True, text=True
        )
        if r.returncode != 0:
            log(f"[WARN] gh run list 失敗: {r.stderr.strip()}")
            time.sleep(30)
            continue
        try:
            data = json.loads(r.stdout)
            if not data:
                time.sleep(30)
                continue
            run = data[0]
            status     = run.get("status", "")
            conclusion = run.get("conclusion", "")
            log(f"[BUILD] status={status} conclusion={conclusion}")
            if status == "completed":
                return conclusion == "success"
        except Exception as e:
            log(f"[WARN] JSON parse 失敗: {e}")
        time.sleep(30)
    log("[ERROR] ビルドタイムアウト")
    return False


# ── vast.ai 再起動 ────────────────────────────────────────────────────────────
def restart_vast_instance() -> bool:
    """vast.ai インスタンスを停止→起動して新イメージを適用"""
    log(f"[VAST] インスタンス {VAST_INSTANCE_ID} を再起動...")
    for action in ("stop", "start"):
        r = subprocess.run(
            f"vastai {action} instance {VAST_INSTANCE_ID}",
            shell=True, capture_output=True, text=True
        )
        if r.returncode != 0:
            log(f"[ERROR] vastai {action} 失敗: {r.stderr.strip()}")
            return False
        log(f"[VAST] {action} → OK")
        if action == "stop":
            log("[VAST] 停止完了待機 (30秒)...")
            time.sleep(30)
    log("[VAST] 起動完了待機 (60秒)...")
    time.sleep(60)
    return True


def verify_training_resumed(retries: int = 10) -> bool:
    """学習が再開されたか確認 (最大 retries × 30秒)"""
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
    log("=" * 60)
    log("FX-EA5 自己修復モニター 起動")
    log(f"  対象インスタンス : {VAST_INSTANCE_ID} ({VAST_SSH_HOST}:{VAST_SSH_PORT})")
    log(f"  S3              : {S3_ENDPOINT}/{S3_BUCKET}/log/")
    log(f"  ポーリング間隔   : {POLL_INTERVAL}秒")
    log("=" * 60)

    heal_count = 0
    consecutive_failures = 0

    while True:
        try:
            # ── 1. 学習プロセス確認 ───────────────────────────────────────
            alive, detail = is_training_alive()
            log_age = get_log_last_modified_age()
            log(f"[CHECK] alive={alive}  log_age={log_age:.0f}s  heals={heal_count}")

            stalled = log_age > STALL_THRESHOLD and log_age != float("inf")

            if alive and not stalled:
                consecutive_failures = 0
                time.sleep(POLL_INTERVAL)
                continue

            # ── 2. 停止/停滞を検出 ───────────────────────────────────────
            reason = "プロセス停止" if not alive else f"ログ更新停滞 ({log_age:.0f}s)"
            log(f"[!] 学習異常検出: {reason}")
            heal_count += 1

            # ── 3. ログ取得・解析 ─────────────────────────────────────────
            log_text = get_log_from_s3() or get_log_from_ssh() or ""
            fix_fn_name = analyze_log(log_text)
            log(f"[ANALYZE] 検出エラー: {fix_fn_name or '不明 → 汎用再起動'}")

            # ── 4. ソース修正 (コード変更が必要な場合のみ) ────────────────
            code_changed = False
            if fix_fn_name and fix_fn_name in FIXERS:
                fixer = FIXERS[fix_fn_name]
                try:
                    changed = fixer()
                    code_changed = changed and fix_fn_name in ("fix_oom", "fix_sample_larger")
                except Exception as e:
                    log(f"[WARN] fixer 失敗: {e}")

            # ── 5. GitHub push → ビルド (コード変更時のみ) ────────────────
            if code_changed:
                msg = f"selfheal: {fix_fn_name} (heal #{heal_count})"
                pushed = git_push_and_build(msg)
                if pushed:
                    build_ok = wait_for_build(timeout_min=35)
                    if not build_ok:
                        log("[ERROR] ビルド失敗 → 再起動のみ試みる")
                else:
                    log("[WARN] push 失敗 → 現行イメージで再起動")
                # vast.ai 再起動 (新イメージ反映)
                restart_vast_instance()
            else:
                # コード変更なし → コンテナ内リカバリを試みる
                # stop.flag 削除 + プロセス再起動
                fix_stop_flag()
                code, out = ssh("rm -f /workspace/stop.flag; "
                                "pkill -f run_train.py; sleep 3; "
                                "nohup /opt/conda/bin/python "
                                "/workspace/fx-ea5/run_train.py "
                                ">> /workspace/train_run.log 2>&1 &")
                log(f"[FIX] コンテナ内再起動: exit={code} {out[:80]}")

            # ── 6. 回復確認 ───────────────────────────────────────────────
            time.sleep(60)
            recovered = verify_training_resumed(retries=8)
            if recovered:
                log(f"[OK] 回復完了 (heal #{heal_count})")
                consecutive_failures = 0
            else:
                log(f"[WARN] 回復確認できず (heal #{heal_count})")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    log("[ERROR] 連続3回回復失敗 → vast.ai インスタンス強制再起動")
                    restart_vast_instance()
                    consecutive_failures = 0

        except KeyboardInterrupt:
            log("中断されました")
            sys.exit(0)
        except Exception as e:
            log(f"[ERROR] メインループ例外: {e}")
            time.sleep(POLL_INTERVAL)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
