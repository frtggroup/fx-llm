"""
feat/__init__.py
全特徴量モジュールを統合し、add_all_indicators() を提供する。

features.py の add_indicators() はここに委譲される:
    from feat import add_all_indicators
    df = add_all_indicators(df)
"""
import numpy as np
import pandas as pd

from .feat_trend       import add_trend
from .feat_macd        import add_macd
from .feat_osc         import add_rsi, add_stoch, add_bb
from .feat_atr_adx     import add_atr_adx
from .feat_price_levels import add_price_levels
from .feat_stats       import add_stats
from .feat_candles     import add_candles
from .feat_bos         import add_bos
from .feat_session     import add_session
from .feat_lagged      import add_lagged


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    features.py の BASE_FEATURE_COLS 700個を全て計算して df に追加し返す。
    各モジュールが担当グループの特徴量を計算する。

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume)
            index: DatetimeIndex
    Returns:
        df に全特徴量列が追加されたもの
    """
    c   = df['close']
    h   = df['high']
    lo  = df['low']
    o   = df['open']
    vol = df['volume']

    out: dict[str, pd.Series] = {}

    # ── 1. トレンド系 ──────────────────────────────────────────────────────────
    out.update(add_trend(c))

    # ── 2. MACD ────────────────────────────────────────────────────────────────
    out.update(add_macd(c))

    # ── 3. RSI / Stochastic / BB ───────────────────────────────────────────────
    out.update(add_rsi(c))
    out.update(add_stoch(c, h, lo))
    out.update(add_bb(c))

    # ── 4. ATR / ADX / PDI / NDI / CCI / WR ───────────────────────────────────
    atr_adx = add_atr_adx(c, h, lo)
    out.update(atr_adx)
    tr   = atr_adx['tr']
    atr14 = atr_adx['atr_14']

    # ── 5. 価格レベル (KC込) ───────────────────────────────────────────────────
    sma_dict = {k: v for k, v in out.items() if k.startswith('sma_')}
    ema_dict = {k: v for k, v in out.items() if k.startswith('ema_')}
    out.update(add_price_levels(c, h, lo, atr14, sma_dict, ema_dict))

    # ── 6. Rolling 統計 ────────────────────────────────────────────────────────
    out.update(add_stats(c))

    # ── 7. ローソク足 ──────────────────────────────────────────────────────────
    candle_out = add_candles(c, h, lo, o, tr)
    out.update(candle_out)

    # ── 8. BOS ─────────────────────────────────────────────────────────────────
    out.update(add_bos(
        c, h, lo,
        ema20    = out.get('ema_20', c),
        macdhist = out.get('macdhist_12_26_9', pd.Series(0.0, index=c.index)),
        rsi14    = out.get('rsi_14', pd.Series(0.5, index=c.index)),
    ))

    # ── 9. セッション ──────────────────────────────────────────────────────────
    out.update(add_session(df.index))

    # ── 10. 遅延/モメンタム ────────────────────────────────────────────────────
    out.update(add_lagged(c, vol))

    # ── DataFrame に書き込む (BASE_FEATURE_COLS 順に inf/nan→0, 一括 concat) ─────
    import importlib
    _feat_mod = importlib.import_module('features')
    _cols = _feat_mod.BASE_FEATURE_COLS

    new_cols: dict[str, pd.Series] = {}
    for k in _cols:
        if k in out:
            s = out[k]
            if isinstance(s, pd.Series):
                new_cols[k] = s.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            else:
                new_cols[k] = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        else:
            new_cols[k] = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

    # 既存列と新列を一括結合して再割当 (フラグメント化回避)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df
