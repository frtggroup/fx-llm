"""
Python vs MT5 特徴量比較スクリプト
====================================
MT5 feat_debug.csv のヘッダー列名 (feat0(idx4) など) から
インデックスを自動抽出して Python 計算値と比較する。
"""
import pandas as pd
import numpy as np
import re, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from features import load_data, add_indicators, FEATURE_COLS

CSV_DATA = Path(__file__).parent.parent / "USDJPY_M1_202301020700_202602262003.csv"
MT5_CSV  = Path(__file__).parent / "feat_debug_mt5.csv"
NORM_JSON = Path(__file__).parent.parent / "NN" / "norm_params.json"

def main():
    if not MT5_CSV.exists():
        print(f"[ERROR] ファイルなし: {MT5_CSV}")
        return

    # ── MT5 CSV 読み込み ──────────────────────────────────────────────────────
    mt5_raw = pd.read_csv(MT5_CSV)
    mt5_raw["datetime"] = pd.to_datetime(mt5_raw["datetime"].str.replace(".", "-", regex=False).str.replace(
        r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})", r"\1-\2-\3 \4:\5:00", regex=True))
    mt5_raw.set_index("datetime", inplace=True)
    print(f"MT5ログ: {len(mt5_raw)}行  {mt5_raw.index[0]} ~ {mt5_raw.index[-1]}")

    # MT5列から feat_index を抽出 (例: "feat0(idx4)" → 4)
    feat_cols = {}  # {feat_index: column_name}
    for col in mt5_raw.columns:
        m = re.search(r"idx(\d+)", col)
        if m:
            feat_cols[int(m.group(1))] = col
    prob_cols = [c for c in ["p_hold","p_buy","p_sell"] if c in mt5_raw.columns]
    print(f"MT5特徴量インデックス: {sorted(feat_cols.keys())}")

    # ── Python 特徴量計算 ─────────────────────────────────────────────────────
    print("Python特徴量計算中...")
    df = load_data(str(CSV_DATA), timeframe="H1")
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 共通インデックス
    common_idx = df.index.intersection(mt5_raw.index)
    if len(common_idx) == 0:
        print("[ERROR] 共通バーなし - タイムゾーン確認")
        print(f"  Python: {df.index[:3].tolist()}")
        print(f"  MT5:    {mt5_raw.index[:3].tolist()}")
        return
    print(f"共通バー: {len(common_idx)}本")

    py_all = df.loc[common_idx, FEATURE_COLS].values   # [N, 70]
    mt5_sub = mt5_raw.loc[common_idx]

    # ── 各特徴量を比較 ────────────────────────────────────────────────────────
    print()
    print("="*72)
    print(f"  特徴量比較 (MT5 vs Python)  バー数:{len(common_idx)}")
    print("="*72)
    print(f"  {'feature_col':<22} {'idx':>4}  {'mean|diff|':>10}  {'max|diff|':>10}  {'rmse':>8}  {'rel':>6}  status")
    print("-"*72)

    results = []
    for fi, col in sorted(feat_cols.items()):
        if fi >= len(FEATURE_COLS):
            continue
        fname = FEATURE_COLS[fi]
        mt5_vals = mt5_sub[col].values.astype(float)
        py_vals  = py_all[:, fi]

        diff      = mt5_vals - py_vals
        mean_diff = float(np.mean(np.abs(diff)))
        max_diff  = float(np.max(np.abs(diff)))
        rmse      = float(np.sqrt(np.mean(diff**2)))
        scale     = float(np.std(py_vals))
        rel       = rmse / (scale + 1e-9)
        status    = "OK  " if rel < 0.01 else ("WARN" if rel < 0.05 else "NG!!")

        print(f"  {fname:<22} [{fi:>2}]  {mean_diff:>10.5f}  {max_diff:>10.5f}  {rmse:>8.5f}  {rel:>6.3f}  {status}")
        results.append({"name": fname, "idx": fi, "mean_diff": mean_diff,
                        "max_diff": max_diff, "rmse": rmse, "rel": rel,
                        "mt5": mt5_vals, "py": py_vals})

    # ── 上位不一致 ────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["rel"], reverse=True)
    print()
    print("不一致ランキング (rel_err = RMSE / std):")
    for r in results:
        bar = "#" * min(40, int(r["rel"] * 20))
        print(f"  [{r['idx']:>2}] {r['name']:<22}  rel={r['rel']:.4f}  {bar}")

    # ── NG特徴量の実際の値を数件表示 ─────────────────────────────────────────
    ng_feats = [r for r in results if r["rel"] >= 0.05]
    if ng_feats:
        print()
        print("NG/WARN特徴量の先頭10バーの実際の値:")
        for r in ng_feats[:3]:
            print(f"\n  [{r['idx']}] {r['name']}")
            print(f"  {'datetime':<25} {'MT5':>12} {'Python':>12} {'diff':>12}")
            for i, dt in enumerate(common_idx[:10]):
                print(f"  {str(dt):<25} {r['mt5'][i]:>12.6f} {r['py'][i]:>12.6f} {r['mt5'][i]-r['py'][i]:>12.6f}")

    # ── 確率比較 ──────────────────────────────────────────────────────────────
    if prob_cols:
        norm = json.loads(NORM_JSON.read_text()) if NORM_JSON.exists() else {}
        feat_indices = norm.get("feat_indices", list(feat_cols.keys()))
        seq_len = norm.get("seq_len", 5)
        onnx_path = NORM_JSON.parent / "fx_model.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                feat_sel = df[FEATURE_COLS].values[:, feat_indices].astype(np.float32)
                print()
                print(f"推論確率比較 (threshold比較用)  seq_len={seq_len}")
                print(f"  {'datetime':<25} {'MT5_buy':>8} {'Py_buy':>8} {'MT5_sel':>8} {'Py_sel':>8}  signal_match")
                n_match = 0
                for i, dt in enumerate(common_idx[:30]):
                    pos = df.index.get_loc(dt)
                    if pos < seq_len:
                        continue
                    x = feat_sel[pos-seq_len+1:pos+1][np.newaxis]
                    py_p = sess.run(None, {"input": x})[0][0]
                    mt5_row = mt5_sub.loc[dt]
                    m_buy  = float(mt5_row.get("p_buy",  0))
                    m_sell = float(mt5_row.get("p_sell", 0))
                    # 閾値0.42でのシグナル一致確認
                    thr = norm.get("threshold", 0.42)
                    def sig(buy, sell, thr):
                        if buy > thr and buy > sell: return "BUY "
                        if sell > thr and sell > buy: return "SELL"
                        return "HOLD"
                    s_mt5 = sig(m_buy, m_sell, thr)
                    s_py  = sig(py_p[1], py_p[2], thr)
                    match = "OK" if s_mt5 == s_py else "NG"
                    if match == "OK": n_match += 1
                    print(f"  {str(dt):<25} {m_buy:>8.4f} {py_p[1]:>8.4f} {m_sell:>8.4f} {py_p[2]:>8.4f}  {s_mt5}/{s_py} {match}")
                print(f"  シグナル一致率: {n_match}/30")
            except Exception as e:
                print(f"確率比較スキップ: {e}")

    print()
    print("="*72)

if __name__ == "__main__":
    main()
