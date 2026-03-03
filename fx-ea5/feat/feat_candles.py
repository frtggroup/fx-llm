"""
feat/feat_candles.py — ローソク足パターン特徴量
対応 BASE_FEATURE_COLS index: 99-107
"""
import pandas as pd


def add_candles(c: pd.Series, h: pd.Series, lo: pd.Series,
                o: pd.Series, tr: pd.Series) -> dict:
    out = {}
    body    = c - o
    upper_w = h - c.clip(lower=o)
    lower_w = c.clip(upper=o) - lo

    out['body']    = body
    out['upper_w'] = upper_w
    out['lower_w'] = lower_w
    out['tr']      = tr     # atr_adxモジュールから渡される

    out['is_doji']       = (body.abs() < tr * 0.1).astype(float)
    out['is_bull_engulf']= (
        (body > 0) & (body.shift(1) < 0) &
        (o <= c.shift(1)) & (c >= o.shift(1))
    ).astype(float)
    out['is_bear_engulf']= (
        (body < 0) & (body.shift(1) > 0) &
        (o >= c.shift(1)) & (c <= o.shift(1))
    ).astype(float)
    out['is_hammer']     = (
        (lower_w > body.abs() * 2) & (upper_w < body.abs() * 0.5)
    ).astype(float)
    out['is_inv_hammer'] = (
        (upper_w > body.abs() * 2) & (lower_w < body.abs() * 0.5)
    ).astype(float)
    return out
