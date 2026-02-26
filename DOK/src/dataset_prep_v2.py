"""
データセット生成 v2 (H100 DOK 版)
- 改善版プロンプト: 自然言語ナラティブ / サマリー / 強弱集計 / 変化記述 / CoT
- 出力先: /workspace/output/llm_train_v2.jsonl, llm_test_v2.jsonl
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, '/workspace/ai_ea')
WORKSPACE = Path('/workspace')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',      required=True)
    p.add_argument('--seq_len',  type=int,   default=20)
    p.add_argument('--tp',       type=float, default=1.5)
    p.add_argument('--sl',       type=float, default=1.0)
    p.add_argument('--forward',  type=int,   default=20)
    p.add_argument('--out_dir',  type=str,   default=str(WORKSPACE / 'output'))
    p.add_argument('--cot',      action='store_true', help='Chain-of-Thoughtラベル')
    args = p.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix     = '_cot' if args.cot else ''
    TRAIN_JSONL = OUT_DIR / f'llm_train_v2{suffix}.jsonl'
    TEST_JSONL  = OUT_DIR / f'llm_test_v2{suffix}.jsonl'

    from llm_dataset_v2 import build_llm_dataset_v2, save_jsonl
    print('=== データセット生成 v2 ===', flush=True)
    print(f'  CSV     : {args.csv}', flush=True)
    print(f'  seq_len : {args.seq_len}', flush=True)
    print(f'  TP/SL   : {args.tp} / {args.sl}', flush=True)
    print(f'  CoT     : {args.cot}', flush=True)

    train_s, test_s = build_llm_dataset_v2(
        csv_path     = args.csv,
        seq_len      = args.seq_len,
        tp_atr       = args.tp,
        sl_atr       = args.sl,
        forward_bars = args.forward,
        max_samples  = 0,
        use_cot      = args.cot,
    )

    save_jsonl(train_s, TRAIN_JSONL)
    save_jsonl(test_s,  TEST_JSONL)

    print(f'\n  訓練: {len(train_s):,} 件 → {TRAIN_JSONL}', flush=True)
    print(f'  テスト: {len(test_s):,} 件 → {TEST_JSONL}', flush=True)

    # サンプル表示
    print('\n--- v2 プロンプト例 ---', flush=True)
    print(train_s[0]['prompt'], flush=True)
    print(f'Label: {train_s[0]["label"]}', flush=True)


if __name__ == '__main__':
    main()
