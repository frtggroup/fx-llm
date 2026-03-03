"""
feat/feat_bos.py — BOS (Break of Structure) 系特徴量
対応 BASE_FEATURE_COLS index: 104-118
"""
import pandas as pd
import numpy as np


def add_bos(c: pd.Series, h: pd.Series, lo: pd.Series,
            ema20: pd.Series, macdhist: pd.Series, rsi14: pd.Series) -> dict:
    out = {}

    out['hh'] = (h  > h.shift(1)).astype(float)
    out['hl'] = (lo > lo.shift(1)).astype(float)
    out['lh'] = (h  < h.shift(1)).astype(float)
    out['ll'] = (lo < lo.shift(1)).astype(float)

    is_bull_bos = (out['hl'].shift(1) > 0) & (out['hh'] > 0)
    is_bear_bos = (out['lh'].shift(1) > 0) & (out['ll'] > 0)

    out['bos']     = (is_bull_bos | is_bear_bos).astype(float)
    out['bos_dir'] = is_bull_bos.astype(float) - is_bear_bos.astype(float)

    # 簡化した BOS 派生
    idx = pd.Series(range(len(c)), index=c.index, dtype=float)
    out['bos_bars']      = idx.where(out['bos'] > 0).ffill().fillna(0)
    out['bos_time']      = out['bos_bars'] * 60
    out['bos_pips']      = c.diff().fillna(0)
    out['bos_fibo_zone'] = pd.Series(0.0, index=c.index)
    out['bos_retest']    = pd.Series(0.0, index=c.index)
    out['bos_move_pips'] = pd.Series(0.0, index=c.index)

    out['bos_ema_sync']  = (
        ((out['bos_dir'] > 0) & (c > ema20)) |
        ((out['bos_dir'] < 0) & (c < ema20))
    ).astype(float)
    out['bos_macd_sync'] = (
        ((out['bos_dir'] > 0) & (macdhist > 0)) |
        ((out['bos_dir'] < 0) & (macdhist < 0))
    ).astype(float)
    out['bos_rsi_sync']  = (
        ((out['bos_dir'] > 0) & (rsi14 > 0.5)) |
        ((out['bos_dir'] < 0) & (rsi14 < 0.5))
    ).astype(float)

    return out
