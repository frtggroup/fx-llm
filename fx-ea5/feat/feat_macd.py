"""
feat/feat_macd.py — MACD 系特徴量
対応 BASE_FEATURE_COLS index: 20-28
"""
import pandas as pd


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def add_macd(c: pd.Series) -> dict:
    out = {}
    for fast, slow, sig in [(12, 26, 9), (6, 19, 9), (5, 34, 5)]:
        line = _ema(c, fast) - _ema(c, slow)
        signal = _ema(line, sig)
        out[f'macd_{fast}_{slow}_{sig}']     = line
        out[f'macdsig_{fast}_{slow}_{sig}']  = signal
        out[f'macdhist_{fast}_{slow}_{sig}'] = line - signal
    return out
