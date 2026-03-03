"""
MT5とPythonのデータ・特徴量の詳細比較
"""
import sys, json, re
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import load_data, add_indicators, FEATURE_COLS

CSV_DATA  = Path(__file__).parent.parent / "USDJPY_M1_202301020700_202602262003.csv"
MT5_CSV   = Path(__file__).parent / "feat_debug_mt5.csv"
NORM_JSON = Path(__file__).parent.parent / "NN" / "norm_params.json"

def main():
    # ── MT5 CSV ────────────────────────────────────────────────────────────
    mt5_raw = pd.read_csv(MT5_CSV)
    mt5_raw["datetime"] = pd.to_datetime(
        mt5_raw["datetime"].str.replace(".", "-", regex=False)
    )
    mt5_raw.set_index("datetime", inplace=True)
    print(f"MT5: {len(mt5_raw)}行  {mt5_raw.index[0]} ~ {mt5_raw.index[-1]}")

    feat_cols = {}
    for col in mt5_raw.columns:
        m = re.search(r"idx(\d+)", col)
        if m:
            feat_cols[int(m.group(1))] = col
    print(f"MT5 feat_indices: {sorted(feat_cols.keys())}")

    # ── Python特徴量 ───────────────────────────────────────────────────────
    print("Python計算中...")
    df = load_data(str(CSV_DATA), timeframe="H1")
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    common_idx = df.index.intersection(mt5_raw.index)
    print(f"共通バー: {len(common_idx)}")
    if len(common_idx) == 0:
        print(f"Python: {df.index[:3].tolist()}")
        print(f"MT5:    {mt5_raw.index[:3].tolist()}")
        return

    py_all  = df[FEATURE_COLS].values
    mt5_sub = mt5_raw.loc[common_idx]

    # ── 最初の5バーの生のクローズ価格比較 ────────────────────────────────
    print("\n=== 最初5バーの Python close価格 (H1 resample) ===")
    for dt in common_idx[:5]:
        pos = df.index.get_loc(dt)
        c   = df["close"].iloc[pos]
        print(f"  {dt}  close={c:.5f}")

    # ── 特徴量比較 ────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  {'feature_col':<22} {'idx':>4}  {'MT5':>10}  {'Python':>10}  {'diff':>10}  rel   status")
    print("-"*72)

    results = []
    for fi, col in sorted(feat_cols.items()):
        if fi >= len(FEATURE_COLS):
            continue
        fname    = FEATURE_COLS[fi]
        mt5_vals = mt5_sub[col].values.astype(float)
        py_vals  = py_all[df.index.get_indexer(common_idx), fi]

        diff  = mt5_vals - py_vals
        rmse  = float(np.sqrt(np.mean(diff**2)))
        scale = float(np.std(py_vals)) + 1e-9
        rel   = rmse / scale
        status = "OK  " if rel < 0.01 else ("WARN" if rel < 0.05 else "NG!!")

        print(f"  {fname:<22} [{fi:>2}]  {mt5_vals[0]:>10.5f}  {py_vals[0]:>10.5f}  {diff[0]:>10.5f}  {rel:>5.3f}  {status}")
        results.append(dict(name=fname, fi=fi, rel=rel, mt5=mt5_vals, py=py_vals))

    # ── NG特徴量の詳細 ───────────────────────────────────────────────────
    ng = [r for r in results if r["rel"] >= 0.05]
    if ng:
        print(f"\nNG/WARN: {len(ng)}個  最悪→ {max(ng, key=lambda x: x['rel'])['name']}")
        worst = max(ng, key=lambda x: x["rel"])
        print(f"\n[{worst['fi']}] {worst['name']}  先頭10バー:")
        print(f"  {'datetime':<26} {'MT5':>10} {'Python':>10}  {'diff':>10}")
        for i, dt in enumerate(common_idx[:10]):
            print(f"  {str(dt):<26} {worst['mt5'][i]:>10.5f} {worst['py'][i]:>10.5f}  {worst['mt5'][i]-worst['py'][i]:>10.5f}")

    # ── MACD詳細 ─────────────────────────────────────────────────────────
    print("\n=== MACD_hist [13] 先頭10バー ===")
    r13 = next((r for r in results if r["fi"] == 13), None)
    if r13:
        print(f"  {'datetime':<26} {'MT5':>10} {'Python':>10}  {'diff':>10}")
        for i, dt in enumerate(common_idx[:10]):
            print(f"  {str(dt):<26} {r13['mt5'][i]:>10.5f} {r13['py'][i]:>10.5f}  {r13['mt5'][i]-r13['py'][i]:>10.5f}")

    # 中間値（バー100付近）も確認
    print("\n=== MACD_hist [13] バー100〜110 ===")
    if r13 and len(common_idx) > 110:
        print(f"  {'datetime':<26} {'MT5':>10} {'Python':>10}  {'diff':>10}")
        for i in range(100, 110):
            dt = common_idx[i]
            print(f"  {str(dt):<26} {r13['mt5'][i]:>10.5f} {r13['py'][i]:>10.5f}  {r13['mt5'][i]-r13['py'][i]:>10.5f}")

    # ── rel_err ランキング ────────────────────────────────────────────────
    results.sort(key=lambda x: x["rel"], reverse=True)
    print("\n不一致ランキング:")
    for r in results:
        bar = "#" * min(40, int(r["rel"] * 20))
        print(f"  [{r['fi']:>2}] {r['name']:<22}  rel={r['rel']:.4f}  {bar}")

    print("="*72)

if __name__ == "__main__":
    main()
