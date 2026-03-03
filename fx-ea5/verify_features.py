"""
MT5 feat_debug.csv vs Python features 整合性チェック
Common\Files の feat_debug.csv (現在の39特徴量モデル) と比較する
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
    if not FEAT_DEBUG_CSV.exists():
        print(f"ERROR: {FEAT_DEBUG_CSV} が見つかりません")
        return
    if M1_CSV is None:
        print("ERROR: USDJPY_M1_*.csv が見つかりません")
        return

    norm = json.load(open(NORM_JSON, encoding='utf-8'))
    feat_indices = norm.get('feat_indices') or list(range(70))

    print(f"モデル: {len(feat_indices)}特徴量  indices={feat_indices}")
    print(f"CSV: {FEAT_DEBUG_CSV.name}")
    print(f"M1データ: {M1_CSV.name}")

    # ── MT5 CSV 読み込み ─────────────────────────────────────────────
    mt5_df = pd.read_csv(str(FEAT_DEBUG_CSV))
    mt5_df['datetime'] = pd.to_datetime(mt5_df['datetime'], format='%Y.%m.%d %H:%M')
    mt5_df.set_index('datetime', inplace=True)
    print(f"\nMT5 CSV: {len(mt5_df)}行  {mt5_df.index[0]} ～ {mt5_df.index[-1]}")

    # 列名からインデックスを抽出: "feat0(idx5)" → 5
    mt5_feat_cols = [c for c in mt5_df.columns if c.startswith('feat')]
    mt5_indices   = [int(c.split('idx')[1].rstrip(')')) for c in mt5_feat_cols]
    print(f"MT5インデックス数: {len(mt5_indices)}  {mt5_indices}")

    if sorted(mt5_indices) != sorted(feat_indices):
        print(f"\n[WARNING] feat_indices不一致!")
        print(f"  norm_params: {sorted(feat_indices)}")
        print(f"  CSV       : {sorted(mt5_indices)}")

    # ── Python 特徴量計算 ────────────────────────────────────────────
    print("\nPython 特徴量計算中...")
    df = load_data(str(M1_CSV), 'H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], float('nan'), inplace=True)

    # ── 比較 ─────────────────────────────────────────────────────────
    common_times = mt5_df.index.intersection(df.index)
    print(f"\n共通バー数: {len(common_times)}")
    if len(common_times) == 0:
        print("ERROR: 共通バーなし - データソース/タイムゾーンを確認してください")
        return

    results = []
    for idx_pos, fi in enumerate(mt5_indices):
        if fi >= len(FEATURE_COLS):
            continue
        fname    = FEATURE_COLS[fi]
        mt5_col  = mt5_feat_cols[idx_pos]
        mt5_vals = mt5_df.loc[common_times, mt5_col].values.astype(float)
        py_vals  = df.loc[common_times, fname].values.astype(float)

        valid    = ~(np.isnan(mt5_vals) | np.isnan(py_vals))
        if valid.sum() == 0:
            continue
        mv = mt5_vals[valid]
        pv = py_vals[valid]
        denom    = np.abs(pv) + 1e-9
        rel_err  = np.abs(mv - pv) / denom
        max_err  = rel_err.max()
        mean_err = rel_err.mean()
        abs_err  = np.abs(mv - pv).mean()

        results.append({
            'fi':      fi,
            'name':    fname,
            'max_rel': max_err,
            'mean_rel':mean_err,
            'mean_abs':abs_err,
            'ok':      max_err < 0.01,
        })

    # ── 結果表示 ─────────────────────────────────────────────────────
    bad  = [r for r in results if not r['ok']]
    good = [r for r in results if r['ok']]
    print(f"\n{'='*60}")
    print(f"✅ 一致 (相対誤差<1%): {len(good)}特徴量")
    print(f"❌ 不一致 (相対誤差≥1%): {len(bad)}特徴量")
    print(f"{'='*60}")

    if bad:
        bad_sorted = sorted(bad, key=lambda r: -r['max_rel'])
        print(f"\n{'idx':>4}  {'特徴量名':25}  {'max_rel':>10}  {'mean_rel':>10}  {'mean_abs':>10}")
        print("-"*70)
        for r in bad_sorted:
            flag = "🔴" if r['max_rel'] > 0.1 else "🟡"
            print(f"{r['fi']:>4}  {r['name']:25}  {r['max_rel']:10.4f}  {r['mean_rel']:10.4f}  {r['mean_abs']:10.6f}  {flag}")

    # ── サンプル比較 (最もエラーが大きい特徴量) ─────────────────────
    if bad:
        worst = sorted(bad, key=lambda r: -r['max_rel'])[:3]
        for r in worst:
            fi      = r['fi']
            fname   = r['name']
            mt5_col = mt5_feat_cols[mt5_indices.index(fi)]
            mt5_v   = mt5_df.loc[common_times, mt5_col].values[:5].astype(float)
            py_v    = df.loc[common_times, fname].values[:5].astype(float)
            times5  = [str(t) for t in common_times[:5]]
            print(f"\n── idx{fi} {fname} (先頭5行) ──")
            print(f"  {'datetime':20}  {'MT5':>12}  {'Python':>12}  {'diff':>10}")
            for t, m, p in zip(times5, mt5_v, py_v):
                print(f"  {t:20}  {m:12.6f}  {p:12.6f}  {m-p:10.6f}")

    # ── 推論確率の比較 ───────────────────────────────────────────────
    if 'p_hold' in mt5_df.columns:
        print(f"\n── 推論確率 (先頭5行) ──")
        for col in ['p_hold','p_buy','p_sell']:
            vals = mt5_df[col].values[:5]
            print(f"  {col}: {[f'{v:.4f}' for v in vals]}")

    print(f"\n{'='*60}")
    if not bad:
        print("🎉 全特徴量一致! PF差の原因はロジック(スプレッド・保有期間等)を確認してください")
    else:
        print(f"上記 {len(bad)} 特徴量に不一致があります。FX_AI_EA.mq5 を修正してください。")

if __name__ == '__main__':
    main()
