"""
MT5 feat_debug.csv vs Python 詳細比較 (第2版)
- 実際の数値を出力して根本原因を特定
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FEAT_DEBUG_CSV = Path(r"C:\Users\yu\AppData\Roaming\MetaQuotes\Terminal\Common\Files\feat_debug.csv")
NN_DIR         = Path(__file__).parent.parent / "NN"
NORM_JSON      = NN_DIR / "norm_params.json"
M1_CSV         = next(Path(__file__).parent.parent.glob("USDJPY_M1_*.csv"), None)

from features import load_data, add_indicators, FEATURE_COLS

def main():
    norm         = json.load(open(NORM_JSON, encoding='utf-8'))
    feat_indices = norm.get('feat_indices') or list(range(70))

    # ── MT5 CSV ─────────────────────────────────────────────────────
    mt5_df = pd.read_csv(str(FEAT_DEBUG_CSV))
    mt5_df['datetime'] = pd.to_datetime(mt5_df['datetime'], format='%Y.%m.%d %H:%M')
    mt5_df.set_index('datetime', inplace=True)
    mt5_feat_cols = [c for c in mt5_df.columns if c.startswith('feat')]
    mt5_indices   = [int(c.split('idx')[1].rstrip(')')) for c in mt5_feat_cols]

    # ── Python ──────────────────────────────────────────────────────
    print("Python 特徴量計算中...")
    df = load_data(str(M1_CSV), 'H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], float('nan'), inplace=True)

    common = mt5_df.index.intersection(df.index)
    print(f"共通バー: {len(common)}  ({common[0]} 〜 {common[-1]})")
    print()

    # ── OHLC 比較 (M1→H1リサンプルとMT5のH1が一致しているか) ──────
    print("="*60)
    print("【OHLC 直接比較】 先頭5バー")
    print("="*60)
    print(f"{'datetime':22}  {'py_close':>10}  MT5_c_ema8  MT5_c_ema200")
    for t in common[:5]:
        pv_close = df.loc[t, 'close']
        mt5_c8   = mt5_df.loc[t, mt5_feat_cols[mt5_indices.index(0)]]
        mt5_c200 = mt5_df.loc[t, mt5_feat_cols[mt5_indices.index(3)]]
        py_c8    = df.loc[t, FEATURE_COLS[0]]
        py_c200  = df.loc[t, FEATURE_COLS[3]]
        print(f"{str(t):22}  {pv_close:10.5f}  MT5:{mt5_c8:8.4f} Py:{py_c8:8.4f}  |  MT5:{mt5_c200:8.4f} Py:{py_c200:8.4f}")

    print()

    # ── close と ATR を推定してEMA自体のズレを計算 ──────────────────
    # c_ema8 = (close - ema8) / atr  →  ema8 = close - c_ema8*atr
    # c_ema8 - c_ema200 = (ema200 - ema8) / atr  →  ema200 - ema8 は直接比較可能
    print("="*60)
    print("【EMA 差分比較】 c_ema8 - c_ema200 (= (ema200-ema8)/atr)")
    print("  これが一致 → ATRは同じ → EMAsの差がズレの原因")
    print("="*60)
    print(f"{'datetime':22}  {'MT5_diff':>10}  {'Py_diff':>10}  {'error':>10}")
    diffs = []
    for t in common[:20]:
        fi0   = mt5_indices.index(0)
        fi3   = mt5_indices.index(3)
        mt5_diff = float(mt5_df.loc[t, mt5_feat_cols[fi3]]) - float(mt5_df.loc[t, mt5_feat_cols[fi0]])
        py_diff  = df.loc[t, FEATURE_COLS[3]] - df.loc[t, FEATURE_COLS[0]]
        diffs.append(abs(mt5_diff - py_diff))
        print(f"{str(t):22}  {mt5_diff:10.4f}  {py_diff:10.4f}  {mt5_diff-py_diff:10.4f}")
    print(f"  mean_abs_diff: {np.mean(diffs):.4f}  (小さければEMAは実質一致)")

    # ── ATR 比較 ─────────────────────────────────────────────────────
    # atr_ratio = atr14 / atr100  →  atr_ratio MT5 vs Py
    print()
    print("="*60)
    print("【ATR/atr_ratio 比較 (idx24)】 先頭10バー")
    print("="*60)
    fi24 = mt5_indices.index(24)
    for t in common[:10]:
        mt5_v = float(mt5_df.loc[t, mt5_feat_cols[fi24]])
        py_v  = float(df.loc[t, FEATURE_COLS[24]])
        print(f"{str(t):22}  MT5:{mt5_v:8.4f}  Py:{py_v:8.4f}  diff:{mt5_v-py_v:8.4f}")

    # ── Ichimoku 詳細 ────────────────────────────────────────────────
    print()
    print("="*60)
    print("【一目均衡表 ichi_cloud_thick (idx56)】 先頭10バー")
    print("="*60)
    fi56 = mt5_indices.index(56)
    fi55 = mt5_indices.index(55)
    fi54 = mt5_indices.index(54)
    for t in common[:10]:
        mt56 = float(mt5_df.loc[t, mt5_feat_cols[fi56]])
        py56 = float(df.loc[t, FEATURE_COLS[56]])
        mt55 = float(mt5_df.loc[t, mt5_feat_cols[fi55]])
        py55 = float(df.loc[t, FEATURE_COLS[55]])
        mt54 = float(mt5_df.loc[t, mt5_feat_cols[fi54]])
        py54 = float(df.loc[t, FEATURE_COLS[54]])
        print(f"{str(t):22}  tk:{mt54:7.4f}/{py54:7.4f}  pos:{mt55:7.4f}/{py55:7.4f}  thick:{mt56:7.4f}/{py56:7.4f}")

    # ── lower_w (idx33) 詳細 ────────────────────────────────────────
    print()
    print("="*60)
    print("【lower_w (idx33)】 先頭10バー")
    print("  lower_w = (min(open,close) - low) / atr")
    print("="*60)
    fi33 = mt5_indices.index(33)
    for t in common[:10]:
        mt33 = float(mt5_df.loc[t, mt5_feat_cols[fi33]])
        py33 = float(df.loc[t, FEATURE_COLS[33]])
        py_lo  = float(df.loc[t, 'low'])
        py_op  = float(df.loc[t, 'open'])
        py_cl  = float(df.loc[t, 'close'])
        py_atr = float(df.loc[t, 'atr14'])
        calc   = (min(py_op, py_cl) - py_lo) / py_atr if py_atr > 0 else 0
        print(f"{str(t):22}  MT5:{mt33:7.4f}  Py:{py33:7.4f}  calc:{calc:7.4f}  low:{py_lo:.3f}  op:{py_op:.3f}  cl:{py_cl:.3f}  atr:{py_atr:.4f}")

    # ── trend_consistency (idx10) 詳細 ──────────────────────────────
    print()
    print("="*60)
    print("【trend_consistency (idx10)】 先頭10バー")
    print("="*60)
    fi10 = mt5_indices.index(10)
    for t in common[:10]:
        mt10 = float(mt5_df.loc[t, mt5_feat_cols[fi10]])
        py10 = float(df.loc[t, FEATURE_COLS[10]])
        print(f"{str(t):22}  MT5:{mt10:7.4f}  Py:{py10:7.4f}  diff:{mt10-py10:7.4f}")

    # ── stoch_k (idx15) ──────────────────────────────────────────────
    print()
    print("="*60)
    print("【stoch_k (idx15)】 先頭10バー")
    print("  stoch_k = (close - lowest_low_14) / (highest_high_14 - lowest_low_14)")
    print("="*60)
    fi15 = mt5_indices.index(15)
    for t in common[:10]:
        mt15 = float(mt5_df.loc[t, mt5_feat_cols[fi15]])
        py15 = float(df.loc[t, FEATURE_COLS[15]])
        print(f"{str(t):22}  MT5:{mt15:7.4f}  Py:{py15:7.4f}  diff:{mt15-py15:7.4f}")

    # ── h4_trend (idx50) ─────────────────────────────────────────────
    print()
    print("="*60)
    print("【h4_trend (idx50)】 先頭10バー")
    print("="*60)
    fi50 = mt5_indices.index(50)
    for t in common[:10]:
        mt50 = float(mt5_df.loc[t, mt5_feat_cols[fi50]])
        py50 = float(df.loc[t, FEATURE_COLS[50]])
        print(f"{str(t):22}  MT5:{mt50:7.4f}  Py:{py50:7.4f}  diff:{mt50-py50:7.4f}")

    # ── rsi14 (idx11) ─────────────────────────────────────────────────
    print()
    print("="*60)
    print("【rsi14 (idx11)】 先頭10バー")
    print("  rsi14 = (rsi - 50) / 50")
    print("="*60)
    fi11 = mt5_indices.index(11)
    for t in common[:10]:
        mt11 = float(mt5_df.loc[t, mt5_feat_cols[fi11]])
        py11 = float(df.loc[t, FEATURE_COLS[11]])
        print(f"{str(t):22}  MT5:{mt11:7.4f}  Py:{py11:7.4f}  diff:{mt11-py11:7.4f}")

    print()
    print("="*60)
    print("【M1 CSV データ確認】 2025-02-25 22:00〜23:00 のM1バー数")
    # M1データで H1の2025-02-25 23:00 バー (22:00〜23:00) を確認
    raw_m1 = pd.read_csv(str(M1_CSV), sep='\t')
    raw_m1.columns = [c.strip('<>') for c in raw_m1.columns]
    raw_m1['dt'] = pd.to_datetime(raw_m1['DATE'] + ' ' + raw_m1['TIME'])
    mask = (raw_m1['dt'] >= '2025-02-25 22:00') & (raw_m1['dt'] < '2025-02-25 23:00')
    sub = raw_m1[mask]
    print(f"  2025-02-25 22:00〜23:00 のM1バー数: {len(sub)}")
    if len(sub) > 0:
        print(f"  最初: {sub.iloc[0]['dt']}  最後: {sub.iloc[-1]['dt']}")
        print(f"  open:{float(sub.iloc[0]['OPEN']):.3f}  high:{float(sub['HIGH'].max()):.3f}  low:{float(sub['LOW'].min()):.3f}  close:{float(sub.iloc[-1]['CLOSE']):.3f}")
        print(f"  → Python H1 close(22:00): {df.loc['2025-02-25 22:00', 'close']:.3f}" if '2025-02-25 22:00' in df.index else "  → Python H1 22:00: N/A")
        print(f"  → Python H1 close(23:00): {df.loc['2025-02-25 23:00', 'close']:.3f}" if '2025-02-25 23:00' in df.index else "  → Python H1 23:00: N/A")

if __name__ == '__main__':
    main()
