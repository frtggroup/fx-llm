"""
MT5 vs Python PF 差の診断スクリプト
"""
import sys, json
from pathlib import Path
from datetime import timedelta
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import onnxruntime as ort

NORM_JSON = Path(__file__).parent.parent / 'NN' / 'norm_params.json'
ONNX_PATH = Path(__file__).parent.parent / 'NN' / 'fx_model.onnx'
CSV_DATA  = Path(__file__).parent.parent / 'USDJPY_M1_202301020700_202602262003.csv'
SPREAD    = 0.005

from features import load_data, add_indicators, FEATURE_COLS

def main():
    norm = json.load(open(NORM_JSON))
    feat_indices = norm['feat_indices']
    seq_len      = norm['seq_len']
    threshold    = norm['threshold']
    tp_mult      = norm['tp_atr']
    sl_mult      = norm['sl_atr']
    hold_bars    = norm['hold_bars']

    print('=== モデル設定 ===')
    print(f'  seq_len={seq_len}  threshold={threshold}  tp={tp_mult}  sl={sl_mult}  hold={hold_bars}')
    print(f'  features={len(feat_indices)}  idx={feat_indices}')

    print('\nデータ読み込み中...')
    df = load_data(str(CSV_DATA), 'H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], float('nan'), inplace=True)
    df.dropna(inplace=True)

    last       = df.index[-1]
    test_start = last - timedelta(days=365)
    df_te      = df[df.index >= test_start].copy()

    print(f'\n=== テスト期間 ===')
    print(f'  開始: {test_start.strftime("%Y.%m.%d %H:%M")}  (MT5設定: この日から開始)')
    print(f'  終了: {last.strftime("%Y.%m.%d %H:%M")}')
    print(f'  バー数: {len(df_te)}本')

    # ONNX推論
    sess      = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    feat_all  = df[FEATURE_COLS].values[:, feat_indices].astype(np.float32)
    close_arr = df_te['close'].values
    open_arr  = df_te['open'].values
    high_arr  = df_te['high'].values
    low_arr   = df_te['low'].values
    atr_arr   = df_te['atr_14'].values
    dates     = df_te.index

    # テスト期間の開始インデックス (df全体の中での位置)
    te_start_idx = df.index.get_loc(df_te.index[0])
    n_te = len(df_te)

    trades = []
    in_pos = False; side = 0; entry = 0; tp_price = 0; sl_price = 0; entry_i = 0

    for i in range(n_te):
        abs_i = te_start_idx + i
        if abs_i < seq_len:
            continue

        # 推論
        x = feat_all[abs_i - seq_len + 1: abs_i + 1][np.newaxis]
        probs = sess.run(None, {'input': x})[0][0]
        p_hold, p_buy, p_sell = probs

        # ポジション管理
        if in_pos:
            hi = high_arr[i]; lo = low_arr[i]
            age = i - entry_i; pnl = None
            if side == 1:
                if lo <= sl_price:  pnl = sl_price - entry - SPREAD
                elif hi >= tp_price: pnl = tp_price - entry - SPREAD
            else:
                if hi >= sl_price:  pnl = entry - sl_price - SPREAD
                elif lo <= tp_price: pnl = entry - tp_price - SPREAD
            if pnl is None and age > hold_bars:
                pnl = (open_arr[i] - entry) * side - SPREAD
            if pnl is not None:
                trades.append({'pnl': pnl, 'side': side, 'date': str(dates[i].date())})
                in_pos = False

        # エントリー
        if not in_pos:
            sig = 0
            if p_buy > threshold and p_buy > p_sell and p_buy > p_hold:
                sig = 1
            elif p_sell > threshold and p_sell > p_buy and p_sell > p_hold:
                sig = 2
            if sig and i + 1 < n_te:
                a = atr_arr[i]
                nxt = open_arr[i + 1]
                side = sig
                if side == 1:
                    entry = nxt + SPREAD
                    tp_price = entry + tp_mult * a
                    sl_price = entry - sl_mult * a
                else:
                    side = -1
                    entry = nxt - SPREAD
                    tp_price = entry - tp_mult * a
                    sl_price = entry + sl_mult * a
                entry_i = i; in_pos = True

    if not trades:
        print('\n[WARNING] 取引なし - 閾値やモデルを確認してください')
        return

    wins  = [t['pnl'] for t in trades if t['pnl'] > 0]
    loses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    gross_p = sum(wins)
    gross_l = abs(sum(loses))
    pf      = gross_p / (gross_l + 1e-9)

    print(f'\n=== Python バックテスト結果 ===')
    print(f'  取引数  : {len(trades)}')
    print(f'  勝率    : {len(wins)/len(trades)*100:.1f}%')
    print(f'  PF      : {pf:.4f}')
    print(f'  純益    : {sum(t["pnl"] for t in trades):+.5f}')

    print(f'\n=== MT5 vs Python 比較ポイント ===')
    print(f'  MT5テスト開始日 → {test_start.strftime("%Y.%m.%d")} に設定してください')
    print(f'  MT5スプレッド  → 固定5ポイント (0.5pips)')
    print(f'  MT5モード      → 1分OHLCまたはEvery Tick推奨')
    print(f'  入力パラメータ → InpThreshold/TpAtr/SlAtr/MaxHoldBars はすべて -1 (JSON読込)')

if __name__ == '__main__':
    main()
