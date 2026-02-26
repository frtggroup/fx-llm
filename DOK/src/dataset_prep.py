"""
データセット生成 (H100 DOK 版)
- max_train 制限なし (全件使用)
- 直近 1 年はテスト用に確保
- 出力先: /workspace/output/llm_train.jsonl, llm_test.jsonl
"""
import sys, json, time, argparse
from pathlib import Path
from datetime import timedelta

import numpy as np

sys.path.insert(0, '/workspace/ai_ea')

WORKSPACE = Path('/workspace')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',     required=True, help='USDJPY M1 CSVパス')
    p.add_argument('--seq_len', type=int,   default=20)
    p.add_argument('--tp',      type=float, default=1.5)
    p.add_argument('--sl',      type=float, default=1.0)
    p.add_argument('--forward', type=int,   default=20)
    p.add_argument('--out_dir', type=str,   default=str(WORKSPACE / 'output'))
    args = p.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_JSONL = OUT_DIR / 'llm_train.jsonl'
    TEST_JSONL  = OUT_DIR / 'llm_test.jsonl'

    from llm_dataset import build_llm_dataset, save_jsonl
    print("=== データセット生成 (制限なし) ===", flush=True)
    print(f"  CSV     : {args.csv}", flush=True)
    print(f"  seq_len : {args.seq_len}", flush=True)
    print(f"  TP/SL   : {args.tp} / {args.sl}", flush=True)

    train_s, test_s = build_llm_dataset(
        csv_path     = args.csv,
        seq_len      = args.seq_len,
        tp_atr       = args.tp,
        sl_atr       = args.sl,
        forward_bars = args.forward,
        max_samples  = 0,   # 制限なし
        seed         = 42,
    )

    save_jsonl(train_s, TRAIN_JSONL)
    save_jsonl(test_s,  TEST_JSONL)

    print(f"\n  訓練データ: {len(train_s):,} 件 → {TRAIN_JSONL}", flush=True)
    print(f"  テストデータ: {len(test_s):,} 件 → {TEST_JSONL}", flush=True)
    print("  データセット生成完了！", flush=True)


if __name__ == '__main__':
    main()
