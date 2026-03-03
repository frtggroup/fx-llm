"""
feat/feat_atr_adx.py — ATR / ADX / PDI / NDI / CCI / WR 系特徴量
対応 BASE_FEATURE_COLS index: 43-66
"""
import numpy as np
import pandas as pd


def _rma(s, period):
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def _calc_tr(h: pd.Series, lo: pd.Series, c: pd.Series) -> pd.Series:
    return pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)


def add_atr_adx(c: pd.Series, h: pd.Series, lo: pd.Series) -> dict:
    """
    Returns: dict with atr_p, adx_p, pdi_p, ndi_p, cci_p, wr_p for p in [7,14,28]
    Also returns 'tr' (True Range series) for reuse.
    """
    out = {}
    tr = _calc_tr(h, lo, c)
    out['tr'] = tr  # 内部利用用 (BASE_FEATURE_COLSのtrでもある)

    up_m = h - h.shift(1)
    dn_m = lo.shift(1) - lo

    for p in [7, 14, 28]:
        atr = _rma(tr, p)
        out[f'atr_{p}'] = atr

        pdm = pd.Series(
            np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=c.index)
        ndm = pd.Series(
            np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=c.index)

        pdi = 100 * _rma(pdm, p) / (atr + 1e-9)
        ndi = 100 * _rma(ndm, p) / (atr + 1e-9)
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)

        out[f'adx_{p}'] = _rma(dx, p) / 100.0
        out[f'pdi_{p}'] = pdi / 100.0
        out[f'ndi_{p}'] = ndi / 100.0

        tp    = (h + lo + c) / 3
        tp_ma = tp.rolling(p).mean()
        mad   = tp.rolling(p).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        out[f'cci_{p}'] = (tp - tp_ma) / (0.015 * mad + 1e-9) / 100.0

        hi_p = h.rolling(p).max()
        lo_p = lo.rolling(p).min()
        out[f'wr_{p}'] = -100 * (hi_p - c) / (hi_p - lo_p + 1e-9) / 100.0 + 0.5

    return out
