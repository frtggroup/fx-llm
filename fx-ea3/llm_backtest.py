"""
LLM バックテスト v1
ファインチューン済み Qwen2.5-1.5B の推論結果で BUY/SELL/HOLD シグナルを生成し
既存 NN バックテストと同じロジックで PF / 勝率 / 取引数を算出

使用方法:
    py -3.14 llm_backtest.py [オプション]
"""
import sys, json, time, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR     = Path(__file__).parent
ADAPTER_DIR = OUT_DIR / 'llm_adapter_best'
TEST_JSONL  = OUT_DIR / 'llm_test.jsonl'
RESULT_JSON = OUT_DIR / 'llm_backtest_result.json'
DATA_PATH   = OUT_DIR.parent / 'USDJPY_M1_202301012206_202602250650.csv'

SPREAD = 0.003
MODEL_LABEL_NAMES = ['HOLD', 'BUY', 'SELL']


# ──────────────────────────────────────────────────────────────────────────
# テストデータ再構築 (df_te + X_te 同期)
# ──────────────────────────────────────────────────────────────────────────
def load_test_ohlcv(seq_len: int = 20, tp_atr: float = 1.5,
                    sl_atr: float = 1.0, forward_bars: int = 20):
    from datetime import timedelta
    from features import load_data, add_indicators, make_labels, FEATURE_COLS
    from llm_dataset import build_llm_dataset

    df = load_data(str(DATA_PATH), timeframe='H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    test_start = df.index[-1] - timedelta(days=365)
    df_te = df[df.index >= test_start].copy()
    return df_te


# ──────────────────────────────────────────────────────────────────────────
# LLM 推論 (バッチ)
# ──────────────────────────────────────────────────────────────────────────
def load_model(adapter_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  アダプター読み込み: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ベースモデル読み込み
    from llm_train import MODEL_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    )
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    model.config.use_cache = True
    print(f"  デバイス: {device}")
    return model, tokenizer, device


def predict_batch(model, tokenizer, samples: list, device,
                  batch_size: int = 16) -> np.ndarray:
    """
    samples: [{'prompt': str, 'label': str}, ...]
    Returns: np.ndarray [N] int (0=HOLD, 1=BUY, 2=SELL)
    """
    import torch
    from llm_train import make_chat_prompt, LABEL_NAMES

    # ラベルトークン ID
    label_tids = [tokenizer.encode(n, add_special_tokens=False)[0]
                  for n in LABEL_NAMES]
    lbl_tensor = torch.tensor(label_tids, device=device)

    preds = []
    n = len(samples)

    for start in range(0, n, batch_size):
        batch = samples[start: start + batch_size]

        # チャットテンプレート適用
        texts = [make_chat_prompt(s['prompt'], tokenizer) for s in batch]
        enc   = tokenizer(texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=1024)
        input_ids  = enc['input_ids'].to(device)
        attn_masks = enc['attention_mask'].to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(input_ids=input_ids, attention_mask=attn_masks)

        # 最後の非パディング位置のロジット
        last_pos = attn_masks.sum(dim=1) - 1
        for b in range(len(batch)):
            lpos  = last_pos[b].item()
            lgt   = out.logits[b, lpos, lbl_tensor]
            pred  = int(lgt.argmax().item())
            preds.append(pred)

        if start % (batch_size * 20) == 0:
            pct = (start + len(batch)) / n * 100
            print(f"  推論中... {pct:.0f}% ({start+len(batch):,}/{n:,})", end='\r')

    print(f"  推論完了: {n:,} サンプル              ")
    return np.array(preds, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────
# バックテスト (既存 NN と同じロジック)
# ──────────────────────────────────────────────────────────────────────────
def run_backtest(preds: np.ndarray, df_te,
                 seq_len: int, tp_mult: float, sl_mult: float,
                 threshold_idx: int = None) -> dict:
    """
    preds   : [N] 0=HOLD, 1=BUY, 2=SELL (確信度なしの場合)
    threshold_idx: None の場合 preds を直接シグナルとして使用
    """
    close = df_te['close'].values
    atr   = df_te['atr14'].values
    high  = df_te['high'].values
    low   = df_te['low'].values
    n     = len(preds)

    HOLD_BARS = 48
    trades = []
    pos    = None

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close):
            break

        c = close[bi]; a = atr[bi]

        # ポジション管理
        if pos:
            hi = high[bi]; lo = low[bi]
            age = i - pos['i0']
            pnl = None
            if pos['side'] == 1:
                if lo <= pos['sl']:    pnl = pos['sl'] - pos['entry'] - SPREAD
                elif hi >= pos['tp']:  pnl = pos['tp'] - pos['entry'] - SPREAD
            else:
                if hi >= pos['sl']:    pnl = pos['entry'] - pos['sl'] - SPREAD
                elif lo <= pos['tp']:  pnl = pos['entry'] - pos['tp'] - SPREAD
            if pnl is None and age >= HOLD_BARS:
                pnl = (c - pos['entry']) * pos['side'] - SPREAD
            if pnl is not None:
                trades.append({'pnl': pnl, 'side': pos['side']})
                pos = None

        # エントリー
        if pos is None:
            cls = int(preds[i])
            if cls == 1:    # BUY
                entry = c + SPREAD
                pos   = {'side':  1, 'entry': entry,
                         'tp': entry + tp_mult * a,
                         'sl': entry - sl_mult * a, 'i0': i}
            elif cls == 2:  # SELL
                entry = c - SPREAD
                pos   = {'side': -1, 'entry': entry,
                         'tp': entry - tp_mult * a,
                         'sl': entry + sl_mult * a, 'i0': i}

    MIN_TRADES = 50
    if len(trades) < MIN_TRADES:
        return {'pf': 0.0, 'trades': len(trades), 'win_rate': 0.0,
                'net_pnl': 0.0, 'note': f'取引数 {len(trades)} < {MIN_TRADES}'}

    pnl  = np.array([t['pnl'] for t in trades])
    gp   = float(pnl[pnl > 0].sum())
    gl   = float(abs(pnl[pnl < 0].sum()))
    return {
        'pf':       round(gp / max(gl, 1e-9), 4),
        'trades':   len(trades),
        'win_rate': round(float((pnl > 0).mean()), 4),
        'net_pnl':  round(float(pnl.sum()), 4),
        'gross_profit': round(gp, 4),
        'gross_loss':   round(gl, 4),
    }


# ──────────────────────────────────────────────────────────────────────────
# 精度比較表示
# ──────────────────────────────────────────────────────────────────────────
def compare_with_nn(llm_result: dict) -> None:
    nn_best = OUT_DIR / 'best_result.json'
    if not nn_best.exists():
        print("  [INFO] NN ベスト結果 (best_result.json) が見つかりません")
        return

    nn = json.loads(nn_best.read_text())
    print("\n" + "="*50)
    print("  比較: LLM vs NN (ベスト)")
    print("="*50)
    print(f"  {'指標':<15} {'LLM':>10} {'NN Best':>10}")
    print("-"*50)
    for key, label in [('pf', 'PF'), ('trades', '取引数'),
                        ('win_rate', '勝率'), ('net_pnl', '純損益')]:
        lv = llm_result.get(key, '-')
        nv = nn.get(key, '-')
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:<15} {fmt(lv):>10} {fmt(nv):>10}")
    print("="*50)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter',    type=str,   default=str(ADAPTER_DIR))
    p.add_argument('--batch',      type=int,   default=16,
                   help='推論バッチサイズ')
    p.add_argument('--tp',         type=float, default=1.5)
    p.add_argument('--sl',         type=float, default=1.0)
    p.add_argument('--seq_len',    type=int,   default=20)
    p.add_argument('--forward',    type=int,   default=20)
    return p.parse_args()


def main():
    args = parse_args()
    print("=== LLM バックテスト ===")
    t0 = time.time()

    # テストデータ読み込み
    print("  テストデータ読み込み中...")
    df_te = load_test_ohlcv(seq_len=args.seq_len, tp_atr=args.tp,
                            sl_atr=args.sl, forward_bars=args.forward)

    # テスト JSONL 読み込み
    if not TEST_JSONL.exists():
        print(f"[ERROR] {TEST_JSONL} が見つかりません。llm_dataset.py を実行してください")
        sys.exit(1)
    test_samples = []
    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    print(f"  テストサンプル: {len(test_samples):,}")

    # モデル読み込み
    model, tokenizer, device = load_model(Path(args.adapter))

    # 推論
    print("  推論開始...")
    preds = predict_batch(model, tokenizer, test_samples, device,
                          batch_size=args.batch)

    # シグナル分布
    for i, name in enumerate(['HOLD', 'BUY', 'SELL']):
        cnt = int((preds == i).sum())
        print(f"    {name}: {cnt:,} ({cnt/len(preds)*100:.1f}%)")

    # 正解率
    true_labels = np.array([{'HOLD':0,'BUY':1,'SELL':2}[s['label']]
                             for s in test_samples])
    acc = float((preds == true_labels).mean())
    print(f"  分類精度: {acc:.4f}")

    # バックテスト
    print("\n  バックテスト実行中...")
    result = run_backtest(preds, df_te, args.seq_len, args.tp, args.sl)

    print(f"\n=== LLM バックテスト結果 ===")
    print(f"  PF       : {result['pf']}")
    print(f"  取引数   : {result['trades']}")
    print(f"  勝率     : {result.get('win_rate', 0):.1%}")
    print(f"  純損益   : {result.get('net_pnl', 0):.4f}")
    print(f"  分類精度 : {acc:.4f}")
    print(f"  実行時間 : {(time.time()-t0)/60:.1f}分")

    result['accuracy'] = round(acc, 4)
    result['elapsed_min'] = round((time.time() - t0) / 60, 1)
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  結果保存: {RESULT_JSON}")

    compare_with_nn(result)
    return result


if __name__ == '__main__':
    main()
