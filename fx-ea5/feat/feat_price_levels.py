"""
feat/feat_price_levels.py — PSAR / Pivot / Donchian / 乖離率 / 一目均衡表
対応 BASE_FEATURE_COLS index: 61-83
"""
import pandas as pd


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def add_price_levels(c: pd.Series, h: pd.Series, lo: pd.Series,
                     atr14: pd.Series, sma_dict: dict, ema_dict: dict) -> dict:
    """
    Args:
        sma_dict: {'sma_20': Series, 'sma_50': Series, 'sma_200': Series, ...}
        ema_dict: {'ema_20': Series, ...}
    """
    out = {}

    # PSAR (簡易 rolling mean 近似)
    out['psar'] = c.rolling(2).mean()

    # Pivot Points (前足ベース)
    pp = (h.shift(1) + lo.shift(1) + c.shift(1)) / 3
    out['pivot'] = pp
    out['r1']    = 2 * pp - lo.shift(1)
    out['r2']    = pp + (h.shift(1) - lo.shift(1))
    out['s1']    = 2 * pp - h.shift(1)
    out['s2']    = pp - (h.shift(1) - lo.shift(1))

    # Donchian Channel (20)
    dh = h.rolling(20).max()
    dl = lo.rolling(20).min()
    out['donchian_pos_20']   = (c - dl) / (dh - dl + 1e-9)
    out['donchian_width_20'] = (dh - dl) / (c + 1e-9)

    # Keltner Channel (20) ※ atr14 を使用
    kc_m = _ema(c, 20)
    kc_u = kc_m + 1.5 * atr14
    kc_d = kc_m - 1.5 * atr14
    out['kc_pos_20']   = (c - kc_d) / (kc_u - kc_d + 1e-9)
    out['kc_width_20'] = (kc_u - kc_d) / (c + 1e-9)

    # 乖離率
    for p in [20, 50, 200]:
        if f'sma_{p}' in sma_dict:
            out[f'diff_sma_{p}'] = (c - sma_dict[f'sma_{p}']) / (c + 1e-9)
        if f'ema_{p}' in ema_dict:
            out[f'diff_ema_{p}'] = (c - ema_dict[f'ema_{p}']) / (c + 1e-9)

    # 一目均衡表
    tk  = (h.rolling(9).max()  + lo.rolling(9).min())  / 2
    kj  = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    sa  = (tk + kj) / 2
    sb  = (h.rolling(52).max() + lo.rolling(52).min()) / 2
    top = pd.concat([sa, sb], axis=1).max(axis=1)
    bot = pd.concat([sa, sb], axis=1).min(axis=1)

    out['ichi_tenkan']     = tk
    out['ichi_kijun']      = kj
    out['ichi_senkou_a']   = sa
    out['ichi_senkou_b']   = sb
    out['ichi_cloud_pos']  = (c - bot) / (top - bot + 1e-9)
    out['ichi_cloud_width']= (top - bot) / (c + 1e-9)
    out['ichi_tk_cross']   = tk - kj

    return out
