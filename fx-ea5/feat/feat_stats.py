"""
feat/feat_stats.py — Rolling 統計 / Z-score 特徴量
対応 BASE_FEATURE_COLS index: 84-98
"""
import pandas as pd


def add_stats(c: pd.Series) -> dict:
    out = {}
    for p in [10, 20, 30]:
        mu  = c.rolling(p).mean()
        std = c.rolling(p).std()
        out[f'roll_mean_{p}'] = mu
        out[f'roll_std_{p}']  = std
        out[f'roll_skew_{p}'] = c.rolling(p).skew()
        out[f'roll_kurt_{p}'] = c.rolling(p).kurt()
        out[f'zscore_{p}']    = (c - mu) / (std + 1e-9)
    return out
