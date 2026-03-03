"""
LLM ワンコマンド実行スクリプト

使用方法:
    py -3.14 llm_run.py setup      # パッケージインストール
    py -3.14 llm_run.py data       # データセット生成
    py -3.14 llm_run.py train      # LoRA ファインチューニング
    py -3.14 llm_run.py backtest   # バックテスト実行
    py -3.14 llm_run.py all        # setup → data → train → backtest を一括実行

オプション例:
    py -3.14 llm_run.py train --epochs 3 --batch 2 --grad_accum 16
    py -3.14 llm_run.py data  --seq_len 20 --tp 1.5 --sl 1.0 --max_train 8000
    py -3.14 llm_run.py all   --epochs 3 --max_train 8000
"""
import subprocess, sys, time, argparse, json
from pathlib import Path

PY      = sys.executable
HERE    = Path(__file__).parent
PARENT  = HERE.parent


# ──────────────────────────────────────────────────────────────────────────
# STEP 0: パッケージインストール
# ──────────────────────────────────────────────────────────────────────────
REQUIRED_PKGS = [
    'transformers',
    'peft',
    'accelerate',
    'datasets',
]

def cmd_setup(args):
    print("="*55)
    print("STEP 0: パッケージインストール")
    print("="*55)
    for pkg in REQUIRED_PKGS:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  [OK] {pkg} インストール済み")
        except ImportError:
            print(f"  [INSTALL] {pkg} インストール中...")
            r = subprocess.run(
                [PY, '-m', 'pip', 'install', '--upgrade', pkg],
                capture_output=False
            )
            if r.returncode != 0:
                print(f"  [ERROR] {pkg} のインストールに失敗しました")
                sys.exit(1)
            print(f"  [OK] {pkg} インストール完了")

    # sentencepiece (Qwen tokenizer に必要な場合)
    try:
        import sentencepiece
        print("  [OK] sentencepiece 利用可能")
    except ImportError:
        print("  [INSTALL] sentencepiece インストール中...")
        subprocess.run([PY, '-m', 'pip', 'install', 'sentencepiece'],
                       capture_output=False)

    print("\n  セットアップ完了!")
    print("  次: py -3.14 llm_run.py data")


# ──────────────────────────────────────────────────────────────────────────
# STEP 1: データセット生成
# ──────────────────────────────────────────────────────────────────────────
def cmd_data(args):
    print("="*55)
    print("STEP 1: LLM データセット生成")
    print("="*55)

    extra_args = [
        '--seq_len',   str(args.seq_len),
        '--tp',        str(args.tp),
        '--sl',        str(args.sl),
        '--forward',   str(args.forward),
    ]
    if args.max_train > 0:
        extra_args += ['--max_train', str(args.max_train)]

    t0 = time.time()
    r  = subprocess.run(
        [PY, str(HERE / 'llm_dataset.py')] + extra_args,
        cwd=str(PARENT)
    )
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"\n[ERROR] データ生成に失敗しました (exit={r.returncode})")
        sys.exit(1)

    train_jsonl = HERE / 'llm_train.jsonl'
    test_jsonl  = HERE / 'llm_test.jsonl'
    if train_jsonl.exists():
        n_tr = sum(1 for _ in open(train_jsonl, encoding='utf-8'))
        n_te = sum(1 for _ in open(test_jsonl,  encoding='utf-8'))
        print(f"\n  生成完了: 訓練={n_tr:,}  テスト={n_te:,}  ({elapsed:.0f}秒)")
    print("  次: py -3.14 llm_run.py train")


# ──────────────────────────────────────────────────────────────────────────
# STEP 2: LoRA ファインチューニング
# ──────────────────────────────────────────────────────────────────────────
def cmd_train(args):
    print("="*55)
    print("STEP 2: LLM LoRA ファインチューニング")
    print(f"  モデル : Qwen2.5-0.5B-Instruct (VRAM ~2-3GB)")
    print(f"  epochs : {args.epochs}")
    print(f"  batch  : {args.batch}  grad_accum: {args.grad_accum}")
    print(f"  実効batch: {args.batch * args.grad_accum}")
    print(f"  lora_r : {args.lora_r}  lora_alpha: {args.lora_alpha}")
    print("="*55)

    train_args = [
        '--epochs',      str(args.epochs),
        '--batch',       str(args.batch),
        '--grad_accum',  str(args.grad_accum),
        '--lr',          str(args.lr),
        '--wd',          str(args.wd),
        '--lora_r',      str(args.lora_r),
        '--lora_alpha',  str(args.lora_alpha),
        '--max_length',  str(args.max_length),
        '--seed',        str(args.seed),
        '--throttle',      str(args.throttle),
        '--vram_fraction', str(args.vram_fraction),
    ]
    if args.max_train > 0:
        train_args += ['--max_train', str(args.max_train)]
    if args.resume:
        train_args.append('--resume')

    t0 = time.time()
    r  = subprocess.run(
        [PY, str(HERE / 'llm_train.py')] + train_args,
        cwd=str(PARENT)
    )
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"\n[ERROR] 学習に失敗しました (exit={r.returncode})")
        sys.exit(1)

    result_json = HERE / 'llm_train_result.json'
    if result_json.exists():
        res = json.loads(result_json.read_text())
        print(f"\n  学習完了: {elapsed/60:.1f}分")
        print(f"  最良精度: {res.get('best_accuracy', '?'):.4f}")
    print("  次: py -3.14 llm_run.py backtest")


# ──────────────────────────────────────────────────────────────────────────
# STEP 3: バックテスト
# ──────────────────────────────────────────────────────────────────────────
def cmd_backtest(args):
    print("="*55)
    print("STEP 3: LLM バックテスト実行")
    print("="*55)

    bt_args = [
        '--batch',   str(args.infer_batch),
        '--tp',      str(args.tp),
        '--sl',      str(args.sl),
        '--seq_len', str(args.seq_len),
        '--forward', str(args.forward),
    ]

    t0 = time.time()
    r  = subprocess.run(
        [PY, str(HERE / 'llm_backtest.py')] + bt_args,
        cwd=str(PARENT)
    )
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"\n[ERROR] バックテストに失敗しました (exit={r.returncode})")
        sys.exit(1)

    result_json = HERE / 'llm_backtest_result.json'
    if result_json.exists():
        res = json.loads(result_json.read_text())
        print(f"\n  バックテスト完了: {elapsed/60:.1f}分")
        print(f"  PF    : {res.get('pf', 0):.4f}")
        print(f"  取引数: {res.get('trades', 0)}")
        print(f"  勝率  : {res.get('win_rate', 0):.1%}")
        print(f"  純損益: {res.get('net_pnl', 0):.4f}")
        print(f"  精度  : {res.get('accuracy', 0):.4f}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description='LLM FX シグナル予測 ワンコマンド実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('command',
                   choices=['setup', 'data', 'train', 'backtest', 'all'],
                   help='実行ステップ')

    # data / train 共通
    g = p.add_argument_group('データ設定')
    g.add_argument('--seq_len',    type=int,   default=20,   help='シーケンス長')
    g.add_argument('--tp',         type=float, default=1.5,  help='TP ATR倍率')
    g.add_argument('--sl',         type=float, default=1.0,  help='SL ATR倍率')
    g.add_argument('--forward',    type=int,   default=20,   help='ラベル先読みバー数')
    g.add_argument('--max_train',  type=int,   default=0,
                   help='訓練サンプル上限 (0=全部、VRAM不足なら8000程度推奨)')

    # train
    g2 = p.add_argument_group('学習設定')
    g2.add_argument('--epochs',     type=int,   default=5)
    g2.add_argument('--batch',      type=int,   default=4,
                    help='バッチサイズ (VRAM不足なら2に下げる)')
    g2.add_argument('--grad_accum', type=int,   default=8,
                    help='gradient accumulation (実効batch=batch×grad_accum)')
    g2.add_argument('--lr',         type=float, default=2e-4)
    g2.add_argument('--wd',         type=float, default=1e-2)
    g2.add_argument('--lora_r',     type=int,   default=16)
    g2.add_argument('--lora_alpha', type=int,   default=32)
    g2.add_argument('--max_length', type=int,   default=512)
    g2.add_argument('--seed',       type=int,   default=42)
    g2.add_argument('--throttle',      type=float, default=0.25,
                    help='GPU制限 (0.25→約80%%, 0.5→約67%%, 0=制限なし)')
    g2.add_argument('--vram_fraction', type=float, default=0.65,
                    help='VRAM上限割合 (0.65=65%%=約7.2GB) PCハング防止')
    g2.add_argument('--resume',        action='store_true',
                    help='チェックポイントから再開')

    # backtest
    g3 = p.add_argument_group('バックテスト設定')
    g3.add_argument('--infer_batch', type=int, default=16,
                    help='推論バッチサイズ')

    return p


def main():
    p    = build_parser()
    args = p.parse_args()

    print("\n" + "="*55)
    print("  LLM FX シグナル予測システム")
    print("  Qwen2.5-1.5B-Instruct + LoRA")
    print("="*55 + "\n")

    if args.command == 'setup':
        cmd_setup(args)

    elif args.command == 'data':
        cmd_data(args)

    elif args.command == 'train':
        cmd_train(args)

    elif args.command == 'backtest':
        cmd_backtest(args)

    elif args.command == 'all':
        total_t0 = time.time()
        print("[ALL] setup → data → train → backtest を順番に実行します\n")
        cmd_setup(args)
        print()
        cmd_data(args)
        print()
        cmd_train(args)
        print()
        cmd_backtest(args)
        print(f"\n[ALL 完了] 総実行時間: {(time.time()-total_t0)/60:.1f}分")


if __name__ == '__main__':
    main()
