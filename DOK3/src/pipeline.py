"""
FX LLM H100 パイプライン
起動直後に自動実行:
  1. データセット生成 (直近1年をテスト保留、制限なし)
  2. H100 ファインチューニング
  3. バックテスト + HTML レポート生成

コンテナ起動時に entrypoint.sh から呼ばれる。
"""
import sys, os, subprocess, json, time
from pathlib import Path

STOP_FLAG = Path('/workspace/stop.flag')

sys.path.insert(0, '/workspace/ai_ea')
sys.path.insert(0, '/workspace/src')

WORKSPACE     = Path('/workspace')
PROGRESS_JSON = WORKSPACE / 'progress.json'
OUTPUT_DIR    = WORKSPACE / 'output'
DATA_DIR      = WORKSPACE / 'data'

# CSVファイルを自動検出 (USDJPY_M1*.csv)
def find_csv() -> Path:
    for pattern in ['USDJPY_M1*.csv', 'usdjpy*.csv', '*.csv']:
        files = sorted(DATA_DIR.glob(pattern))
        if files:
            return files[-1]  # 最新
    return None


def write_progress(patch: dict) -> None:
    try:
        cur = {}
        if PROGRESS_JSON.exists():
            cur = json.loads(PROGRESS_JSON.read_text())
        cur.update(patch)
        PROGRESS_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    except Exception:
        pass


def run_step(label: str, cmd: list, env: dict = None) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"  STEP: {label}", flush=True)
    print(f"{'='*60}", flush=True)
    write_progress({'message': f'STEP: {label}'})
    merged = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, env=merged)
    if proc.returncode != 0:
        msg = f"[PIPELINE] {label} 失敗 (exit={proc.returncode})"
        print(msg, flush=True)
        write_progress({'phase': 'error', 'error': msg})
        return False
    return True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_id',   type=str,   default='Qwen/Qwen3-8B')
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--batch',      type=int,   default=8)
    p.add_argument('--grad_accum', type=int,   default=8)
    p.add_argument('--lora_r',     type=int,   default=64)
    p.add_argument('--lora_alpha', type=int,   default=128)
    p.add_argument('--max_length', type=int,   default=1024)
    p.add_argument('--lr',         type=float, default=5e-5)
    p.add_argument('--seq_len',    type=int,   default=20)
    p.add_argument('--tp',         type=float, default=1.5)
    p.add_argument('--sl',         type=float, default=1.0)
    p.add_argument('--skip_dataset', action='store_true',
                   help='データセット生成をスキップ (既存 JSONL を使用)')
    p.add_argument('--skip_train',   action='store_true',
                   help='訓練をスキップ (既存アダプターでバックテストのみ)')
    p.add_argument('--resume',       action='store_true')
    p.add_argument('--v2',           action='store_true',
                   help='v2プロンプト+v2データセットで訓練・評価する')
    p.add_argument('--cot',          action='store_true',
                   help='v2+CoTラベルを使用する (--v2 も自動ON)')
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / 'reports').mkdir(exist_ok=True)

    write_progress({
        'phase': 'loading',
        'message': 'パイプライン開始',
        'total_epochs': args.epochs,
        'epoch': 0, 'train_loss': 0.0, 'accuracy': 0.0, 'best_acc': 0.0,
        'gpu_pct': 0, 'vram_used_gb': 0.0, 'vram_total_gb': 80.0,
        'elapsed_sec': 0.0, 'eta_sec': -1, 'epoch_log': [], 'batch_log': [],
    })

    t_start = time.time()

    use_v2 = args.v2 or args.cot

    # ── STEP 1: データセット生成 ───────────────────────────────────────────
    if not args.skip_dataset:
        csv_path = find_csv()
        if csv_path is None:
            msg = (f"[ERROR] データCSVが見つかりません。\n"
                   f"  /workspace/data/ に USDJPY_M1*.csv をアップロードしてください。\n"
                   f"  例: scp USDJPY_M1_*.csv root@<DOK_IP>:/workspace/data/")
            print(msg, flush=True)
            write_progress({'phase': 'error', 'error': msg})
            sys.exit(1)

        print(f"  CSVファイル: {csv_path}", flush=True)
        write_progress({
            'phase': 'loading',
            'message': f'データセット生成中... ({csv_path.name})',
        })

        prep_script = '/workspace/src/dataset_prep_v2.py' if use_v2 \
                      else '/workspace/src/dataset_prep.py'
        prep_cmd = [
            sys.executable, prep_script,
            '--csv',      str(csv_path),
            '--seq_len',  str(args.seq_len),
            '--tp',       str(args.tp),
            '--sl',       str(args.sl),
            '--out_dir',  str(OUTPUT_DIR),
        ]
        if use_v2 and args.cot:
            prep_cmd.append('--cot')

        label = 'データセット生成 v2 (CoT)' if args.cot else \
                'データセット生成 v2'       if use_v2 else \
                'データセット生成'
        ok = run_step(label, prep_cmd)
        if not ok:
            sys.exit(1)
    else:
        print("  データセット生成: スキップ", flush=True)

    # ── STEP 2: H100 ファインチューニング ─────────────────────────────────
    if not args.skip_train:
        # v2 ではデータファイル名が llm_train_v2[_cot].jsonl
        cot_suffix = '_cot' if args.cot else ''
        train_jsonl = str(OUTPUT_DIR / f'llm_train_v2{cot_suffix}.jsonl') if use_v2 \
                      else str(OUTPUT_DIR / 'llm_train.jsonl')
        adapter_out = str(OUTPUT_DIR / 'llm_adapter_best_v2') if use_v2 \
                      else str(OUTPUT_DIR / 'llm_adapter_best')

        train_cmd = [
            sys.executable, '/workspace/src/train_h100.py',
            '--model_id',   args.model_id,
            '--epochs',     str(args.epochs),
            '--batch',      str(args.batch),
            '--grad_accum', str(args.grad_accum),
            '--lora_r',     str(args.lora_r),
            '--lora_alpha', str(args.lora_alpha),
            '--max_length', '1536' if use_v2 else str(args.max_length),
            '--lr',         str(args.lr),
            '--train_jsonl', train_jsonl,
            '--adapter_out', adapter_out,
        ]
        if args.resume:
            train_cmd.append('--resume')

        ok = run_step(f'H100 ファインチューニング ({args.model_id})', train_cmd)
        if not ok:
            sys.exit(1)
    else:
        print("  訓練: スキップ", flush=True)

    # 停止フラグが残っていれば削除（訓練スクリプト内で削除されるはずだが念のため）
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
        print("  [INFO] 停止フラグをクリア → バックテスト・完了処理を続行", flush=True)
        write_progress({'stop_requested': False, 'message': '停止後バックテスト実行中...'})

    # ── STEP 3: バックテスト + レポート ───────────────────────────────────
    if use_v2:
        bt_cmd = [
            sys.executable, '/workspace/src/backtest_report_v2.py',
            '--tp', str(args.tp),
            '--sl', str(args.sl),
        ]
        if args.cot:
            bt_cmd.append('--cot')
        ok = run_step('バックテスト + HTML レポート生成 (v2)', bt_cmd)
    else:
        ok = run_step('バックテスト + HTML レポート生成', [
            sys.executable, '/workspace/src/backtest_report.py',
            '--tp', str(args.tp),
            '--sl', str(args.sl),
        ])
    if not ok:
        sys.exit(1)

    # ── STEP 4: MT5シグナルCSV生成 ────────────────────────────────────────
    ok = run_step('MT5シグナルCSV生成', [
        sys.executable, '/workspace/src/signal_export.py',
        '--tp', str(args.tp),
        '--sl', str(args.sl),
    ])
    if not ok:
        print("[WARN] MT5シグナル生成に失敗しましたが続行します", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"  パイプライン完了！  {elapsed/60:.1f}分", flush=True)
    print(f"  ダッシュボード:     http://<DOK_IP>:7860", flush=True)
    print(f"  レポートDL:         http://<DOK_IP>:7860/download/report", flush=True)
    print(f"  モデルDL:           http://<DOK_IP>:7860/download/adapter", flush=True)
    print(f"  MT5シグナルDL:      http://<DOK_IP>:7860/download/mt5signals", flush=True)
    print(f"  MT5 EA DL:          http://<DOK_IP>:7860/download/mt5ea", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
