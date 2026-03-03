"""
feat/feat_lagged.py — 遅延リターン / ボリューム / モメンタム / ボラティリティ
対応 BASE_FEATURE_COLS index: 127-229 (実質的な計算部分)
"""
import pandas as pd


def add_lagged(c: pd.Series, vol: pd.Series) -> dict:
    out = {}

    # ret1_p, ret5_p, vol_p (p = 5,10,...,50)
    for p in range(5, 51, 5):
        out[f'ret1_{p}'] = c.pct_change(1).shift(p)
        out[f'ret5_{p}'] = c.pct_change(5).shift(p)
        out[f'vol_{p}']  = vol.rolling(p).mean()

    # momentum_p, volatility_p, price_diff_p (p = 5,10,...,115)
    for p in range(5, 120, 5):
        out[f'momentum_{p}']   = c.diff(p)
        out[f'volatility_{p}'] = c.rolling(p).std()
        out[f'price_diff_{p}'] = c - c.shift(p)

    return out
