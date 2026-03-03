"""
feat/feat_osc.py — RSI / Stochastic / BB / KC 系特徴量
対応 BASE_FEATURE_COLS index: 29-70
"""
import numpy as np
import pandas as pd


def _rma(s, period):
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def add_rsi(c: pd.Series) -> dict:
    out = {}
    for p in [7, 14, 21, 28]:
        d = c.diff()
        g = _rma(d.clip(lower=0), p)
        l = _rma((-d).clip(lower=0), p)
        out[f'rsi_{p}'] = (100 - 100 / (1 + g / (l + 1e-9))) / 100.0
    return out


def add_stoch(c: pd.Series, h: pd.Series, lo: pd.Series) -> dict:
    out = {}
    for k_p, d_p in [(14, 3), (9, 3), (21, 5)]:
        lo_k = lo.rolling(k_p).min()
        hi_k = h.rolling(k_p).max()
        st_k = (c - lo_k) / (hi_k - lo_k + 1e-9)
        st_d = st_k.rolling(d_p).mean()
        out[f'stoch_k_{k_p}_{d_p}'] = st_k
        out[f'stoch_d_{k_p}_{d_p}'] = st_d
    return out


def add_bb(c: pd.Series) -> dict:
    out = {}
    for p in [20, 50]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std(ddof=0)
        up  = mid + 2 * std
        dn  = mid - 2 * std
        out[f'bb_pos_{p}']   = (c - dn) / (up - dn + 1e-9)
        out[f'bb_width_{p}'] = (up - dn) / (mid + 1e-9)
    return out


def add_kc(c: pd.Series, atr14: pd.Series) -> dict:
    kc_m = _ema(c, 20)
    kc_u = kc_m + 1.5 * atr14
    kc_d = kc_m - 1.5 * atr14
    return {
        'kc_pos_20':   (c - kc_d) / (kc_u - kc_d + 1e-9),
        'kc_width_20': (kc_u - kc_d) / (c + 1e-9),
    }
