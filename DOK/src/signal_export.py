"""
MT5バックテスト用 シグナルCSV生成
訓練済みモデルをテスト期間（直近1年）に適用し、
MT5 EAが読み込める形式のCSVを出力する

出力先: /workspace/reports/mt5_signals_YYYYMMDD.csv
列    : datetime, signal, confidence, open, high, low, close, atr,
        tp_price, sl_price, true_label
"""
import sys, json, re, time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, '/workspace/ai_ea')
sys.path.insert(0, '/workspace/src')

WORKSPACE    = Path('/workspace')
ADAPTER_DIR  = WORKSPACE / 'output' / 'llm_adapter_best'
TEST_JSONL   = WORKSPACE / 'output' / 'llm_test.jsonl'
REPORT_DIR   = WORKSPACE / 'reports'
PROGRESS_JSON = WORKSPACE / 'progress.json'

LABEL_NAMES  = ['HOLD', 'BUY', 'SELL']
TP_ATR       = 1.5
SL_ATR       = 1.0
BATCH_SIZE   = 32


def update_progress(patch: dict) -> None:
    try:
        cur = {}
        if PROGRESS_JSON.exists():
            cur = json.loads(PROGRESS_JSON.read_text())
        cur.update(patch)
        PROGRESS_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    except Exception:
        pass


def load_test_samples() -> list:
    if not TEST_JSONL.exists():
        raise FileNotFoundError(f"テストデータが見つかりません: {TEST_JSONL}")
    samples = []
    with open(TEST_JSONL, encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"  テストサンプル: {len(samples):,} 件", flush=True)
    return samples


def extract_timestamp(prompt: str) -> str | None:
    """プロンプト先頭行から '2025-02-25 10:00' 形式の日時を抽出"""
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', prompt)
    return m.group(1) if m else None


def extract_atr(prompt: str) -> float:
    """プロンプトから ATR/price 値を抽出 (ATR/price=0.003 など)"""
    m = re.search(r'ATR/price=([\d.]+)', prompt)
    return float(m.group(1)) if m else 0.0


def load_h1_prices(csv_path: Path) -> dict:
    """H1 OHLCV を datetime文字列 → dict で返す"""
    import pandas as pd
    from features import load_data, add_indicators, FEATURE_COLS

    print(f"  H1価格データ読込中: {csv_path.name}", flush=True)
    df = load_data(str(csv_path), timeframe='H1')
    df = add_indicators(df)
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)

    atr_col = 'atr14' if 'atr14' in df.columns else None
    price_map = {}
    for ts, row in df.iterrows():
        key = ts.strftime('%Y-%m-%d %H:%M')
        close = float(row['close'])
        atr   = float(row[atr_col]) if atr_col else close * 0.003
        price_map[key] = {
            'open':  float(row['open']),
            'high':  float(row['high']),
            'low':   float(row['low']),
            'close': close,
            'atr':   atr,
        }
    print(f"  H1バー数: {len(price_map):,}", flush=True)
    return price_map


def run_inference(samples: list) -> tuple[list, list]:
    """推論実行 → (pred_labels, confidences) を返す"""
    import torch
    from unsloth import FastLanguageModel

    print(f"  モデル読込: {ADAPTER_DIR}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = str(ADAPTER_DIR),
        max_seq_length    = 1024,
        load_in_4bit      = True,
        dtype             = None,
        trust_remote_code = True,
    )
    FastLanguageModel.for_inference(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  デバイス: {device}", flush=True)

    label_tids = [tokenizer.encode(n, add_special_tokens=False)[0]
                  for n in LABEL_NAMES]
    lbl_tensor = torch.tensor(label_tids, device=device)

    preds, confs = [], []
    n = len(samples)
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        batch = samples[start: start + BATCH_SIZE]
        texts = []
        for s in batch:
            msgs = [
                {"role": "system",
                 "content": ("You are a professional FX trading signal analyst. "
                             "Analyze the market data and respond with exactly one word: "
                             "BUY, SELL, or HOLD.")},
                {"role": "user", "content": s['prompt']},
            ]
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        enc       = tokenizer(texts, return_tensors='pt', padding=True,
                              truncation=True, max_length=1024)
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                                  enabled=(device.type == 'cuda')):
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        last_pos = attn_mask.sum(dim=1) - 1
        for b in range(len(batch)):
            logits = out.logits[b, last_pos[b], lbl_tensor]
            probs  = torch.softmax(logits.float(), dim=0).cpu().numpy()
            idx    = int(np.argmax(probs))
            preds.append(LABEL_NAMES[idx])
            confs.append(float(probs[idx]))

        done = min(start + BATCH_SIZE, n)
        if done % 500 == 0 or done == n:
            elapsed = time.time() - t0
            print(f"  推論: {done:,}/{n:,}  ({elapsed:.0f}s)", flush=True)
            update_progress({'message': f'MT5シグナル生成中: {done:,}/{n:,}'})

    return preds, confs


def find_csv() -> Path | None:
    data_dir = WORKSPACE / 'data'
    for pat in ['USDJPY_M1*.csv', '*.csv']:
        files = sorted(data_dir.glob(pat))
        if files:
            return files[-1]
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tp',        type=float, default=1.5)
    p.add_argument('--sl',        type=float, default=1.0)
    p.add_argument('--min_conf',  type=float, default=0.0,
                   help='この値未満の確信度のシグナルはHOLDに変換')
    args = p.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    update_progress({'message': 'MT5シグナルCSV生成開始'})

    print("\n=== MT5シグナルCSV生成 ===", flush=True)

    samples = load_test_samples()

    csv_path = find_csv()
    price_map = load_h1_prices(csv_path) if csv_path else {}
    if not price_map:
        print("  [WARN] 価格データなし。close=0 で出力します", flush=True)

    preds, confs = run_inference(samples)

    # CSV 生成
    rows = []
    no_price = 0
    for i, (s, pred, conf) in enumerate(zip(samples, preds, confs)):
        ts  = extract_timestamp(s['prompt'])
        if ts is None:
            continue

        # 確信度フィルタ
        sig = pred if conf >= args.min_conf else 'HOLD'

        pr = price_map.get(ts, {})
        close = pr.get('close', 0.0)
        atr   = pr.get('atr',   close * 0.003 if close else 0.0)

        if close == 0.0:
            no_price += 1

        tp_price = (close + atr * args.tp) if sig == 'BUY'  else \
                   (close - atr * args.tp) if sig == 'SELL' else 0.0
        sl_price = (close - atr * args.sl) if sig == 'BUY'  else \
                   (close + atr * args.sl) if sig == 'SELL' else 0.0

        rows.append({
            'datetime':   ts,
            'signal':     sig,
            'confidence': f'{conf:.4f}',
            'open':       f"{pr.get('open',  0.0):.5f}",
            'high':       f"{pr.get('high',  0.0):.5f}",
            'low':        f"{pr.get('low',   0.0):.5f}",
            'close':      f'{close:.5f}',
            'atr':        f'{atr:.5f}',
            'tp_price':   f'{tp_price:.5f}',
            'sl_price':   f'{sl_price:.5f}',
            'true_label': s.get('label', ''),
        })

    if no_price > 0:
        print(f"  [WARN] 価格が見つからなかった行: {no_price:,}", flush=True)

    # 統計
    total   = len(rows)
    n_buy   = sum(1 for r in rows if r['signal'] == 'BUY')
    n_sell  = sum(1 for r in rows if r['signal'] == 'SELL')
    n_hold  = sum(1 for r in rows if r['signal'] == 'HOLD')
    avg_conf_act = (
        np.mean([float(r['confidence']) for r in rows if r['signal'] != 'HOLD'])
        if (n_buy + n_sell) > 0 else 0.0
    )

    print(f"\n  シグナル統計:", flush=True)
    print(f"    総行数  : {total:,}", flush=True)
    print(f"    BUY     : {n_buy:,}  ({n_buy/total*100:.1f}%)", flush=True)
    print(f"    SELL    : {n_sell:,}  ({n_sell/total*100:.1f}%)", flush=True)
    print(f"    HOLD    : {n_hold:,}  ({n_hold/total*100:.1f}%)", flush=True)
    print(f"    平均信頼度(BUY+SELL): {avg_conf_act:.3f}", flush=True)

    # 書き出し
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M')
    out_csv = REPORT_DIR / f'mt5_signals_{ts_str}.csv'

    header  = 'datetime,signal,confidence,open,high,low,close,atr,tp_price,sl_price,true_label\n'
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(header)
        for r in rows:
            f.write(','.join([
                r['datetime'], r['signal'], r['confidence'],
                r['open'], r['high'], r['low'], r['close'],
                r['atr'], r['tp_price'], r['sl_price'], r['true_label'],
            ]) + '\n')

    print(f"\n  保存完了: {out_csv}  ({len(rows):,} rows)", flush=True)

    update_progress({
        'message': f'MT5シグナルCSV生成完了: {len(rows):,}件',
        'mt5_signal_file': str(out_csv),
        'mt5_stats': {
            'total': total, 'buy': n_buy, 'sell': n_sell, 'hold': n_hold,
            'avg_conf': round(avg_conf_act, 4),
        },
    })


if __name__ == '__main__':
    main()
