"""
H1終値の直接比較 - MT5のret1から逆算して価格レベルを確認
"""
import sys, re
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import load_data, FEATURE_COLS

CSV_DATA = Path(__file__).parent.parent / "USDJPY_M1_202301020700_202602262003.csv"
MT5_CSV  = Path(__file__).parent / "feat_debug_mt5.csv"

def main():
    # MT5 CSV
    mt5 = pd.read_csv(MT5_CSV)
    mt5["datetime"] = pd.to_datetime(mt5["datetime"].str.replace(".", "-", regex=False))
    mt5.set_index("datetime", inplace=True)

    # feat_indices from header
    feat_map = {}
    for col in mt5.columns:
        m = re.search(r"idx(\d+)", col)
        if m:
            feat_map[int(m.group(1))] = col

    # Python H1 data
    df = load_data(str(CSV_DATA), timeframe="H1")

    common_idx = df.index.intersection(mt5.index)
    print(f"共通バー: {len(common_idx)}")
    print()

    # Python H1 close一覧 (先頭15バー近辺)
    # MT5 feat_debug の範囲に合わせる
    start_dt = common_idx[0]
    py_window = df.loc[start_dt - pd.Timedelta(hours=5) : start_dt + pd.Timedelta(hours=10)]

    print("=== Python H1 close (MT5 debug 開始前後) ===")
    print(f"  {'datetime':<25} {'close':>10}  {'ret1_py':>10}")
    prev_c = None
    for dt, row in py_window.iterrows():
        c = row['close']
        r1 = (c / prev_c - 1) if prev_c else float('nan')
        marker = " <<<" if dt in common_idx else ""
        print(f"  {str(dt):<25} {c:>10.5f}  {r1:>10.6f}{marker}")
        prev_c = c

    print()

    # ret1からMT5の前バー終値を逆算
    print("=== MT5 ret1から逆算: MT5の前バー終値 vs Python ===")
    print(f"  {'datetime':<25} {'Py_close':>10}  {'MT5_ret1':>10}  {'MT5_prev(推算)':>14}  {'Py_prev':>10}  {'price_diff':>10}")
    ret1_col = feat_map.get(34, None)

    py_arr = df['close']
    for i, dt in enumerate(common_idx[:15]):
        py_c  = py_arr[dt]
        py_pos = df.index.get_loc(dt)
        py_c_prev = py_arr.iloc[py_pos - 1] if py_pos > 0 else float('nan')

        mt5_ret1 = float(mt5.loc[dt, ret1_col]) if ret1_col else float('nan')
        mt5_prev = py_c / (1 + mt5_ret1) if (not np.isnan(mt5_ret1) and mt5_ret1 != -1) else float('nan')

        price_diff = mt5_prev - py_c_prev

        print(f"  {str(dt):<25} {py_c:>10.5f}  {mt5_ret1:>10.6f}  {mt5_prev:>14.5f}  {py_c_prev:>10.5f}  {price_diff:>10.5f}")

    # M1生データの特定時刻を確認
    print()
    print("=== M1生データ (2025-01-31 22:50〜23:05) ===")
    raw = pd.read_csv(str(CSV_DATA), sep='\t')
    raw.columns = [c.strip('<>').lower() for c in raw.columns]
    raw['dt'] = pd.to_datetime(raw['date'].str.replace('.', '-', regex=False) + ' ' + raw['time'])
    mask = (raw['dt'] >= '2025-01-31 22:50') & (raw['dt'] <= '2025-01-31 23:05')
    sub = raw[mask]
    print(f"  {'datetime':<25} {'open':>10}  {'high':>10}  {'low':>10}  {'close':>10}  {'tickvol':>8}")
    for _, row in sub.iterrows():
        print(f"  {str(row['dt']):<25} {row['open']:>10.5f}  {row['high']:>10.5f}  {row['low']:>10.5f}  {row['close']:>10.5f}  {int(row['tickvol']):>8d}")

if __name__ == "__main__":
    main()
