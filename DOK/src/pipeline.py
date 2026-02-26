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

        ok = run_step('データセット生成', [
            sys.executable, '/workspace/src/dataset_prep.py',
            '--csv',      str(csv_path),
            '--seq_len',  str(args.seq_len),
            '--tp',       str(args.tp),
            '--sl',       str(args.sl),
            '--out_dir',  str(OUTPUT_DIR),
        ])
        if not ok:
            sys.exit(1)
    else:
        print("  データセット生成: スキップ", flush=True)

    # ── STEP 2: H100 ファインチューニング ─────────────────────────────────
    if not args.skip_train:
        train_cmd = [
            sys.executable, '/workspace/src/train_h100.py',
            '--model_id',   args.model_id,
            '--epochs',     str(args.epochs),
            '--batch',      str(args.batch),
            '--grad_accum', str(args.grad_accum),
            '--lora_r',     str(args.lora_r),
            '--lora_alpha', str(args.lora_alpha),
            '--max_length', str(args.max_length),
            '--lr',         str(args.lr),
        ]
        if args.resume:
            train_cmd.append('--resume')

        ok = run_step(f'H100 ファインチューニング ({args.model_id})', train_cmd)
        if not ok:
            sys.exit(1)
    else:
        print("  訓練: スキップ", flush=True)

    # ── STEP 3: バックテスト + レポート ───────────────────────────────────
    ok = run_step('バックテスト + HTML レポート生成', [
        sys.executable, '/workspace/src/backtest_report.py',
        '--tp', str(args.tp),
        '--sl', str(args.sl),
    ])
    if not ok:
        sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"  パイプライン完了！  {elapsed/60:.1f}分", flush=True)
    print(f"  ダッシュボード: http://<DOK_IP>:7860", flush=True)
    print(f"  レポートDL:     http://<DOK_IP>:7860/download/report", flush=True)
    print(f"  モデルDL:       http://<DOK_IP>:7860/download/adapter", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
