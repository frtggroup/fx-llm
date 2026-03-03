"""
feat/feat_trend.py — トレンド系特徴量 (SMA / EMA / HMA / WMA)
対応 BASE_FEATURE_COLS index: 0-19
"""
import numpy as np
import pandas as pd


def _sma(s, p):
    return s.rolling(p).mean()

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _wma(s, p):
    w = np.arange(1, p + 1)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _hma(s, p):
    half = int(p / 2)
    sq   = int(np.sqrt(p))
    return _wma(_wma(s, half) * 2 - _wma(s, p), sq)


def add_trend(c: pd.Series) -> dict:
    """
    Args:
        c: close price series
    Returns:
        dict {feature_name: pd.Series}
    """
    out = {}
    for p in [5, 10, 20, 50, 100, 200]:
        out[f'sma_{p}'] = _sma(c, p)
        out[f'ema_{p}'] = _ema(c, p)
        if p <= 50:
            out[f'hma_{p}'] = _hma(c, p)
            out[f'wma_{p}'] = _wma(c, p)
    return out
