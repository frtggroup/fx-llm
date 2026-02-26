"""
LLM データセット v2 — 改善版プロンプト生成
改善点:
  1. 自然言語ナラティブ（数値羅列→文章）
  2. 冒頭に総合サマリー行
  3. 強気/弱気シグナル数 + 対立検知
  4. 過去20本の変化を物語で記述（seq活用）
  5. 市場レジーム分類（トレンド/レンジ/高ボラ）
  6. RSIを0-100スケールに変換
  7. Chain-of-Thought ラベル（オプション）
"""
import sys, json, time, re
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (load_data, add_indicators, make_labels,
                      FEATURE_COLS, N_FEATURES)

DATA_PATH   = Path(__file__).parent.parent / 'USDJPY_M1_202301012206_202602250650.csv'
OUT_DIR     = Path(__file__).parent
TRAIN_JSONL = OUT_DIR / 'llm_train_v2.jsonl'
TEST_JSONL  = OUT_DIR / 'llm_test_v2.jsonl'

_IDX = {col: i for i, col in enumerate(FEATURE_COLS)}

def _f(feat: np.ndarray, col: str, default: float = 0.0) -> float:
    idx = _IDX.get(col)
    return float(feat[idx]) if idx is not None else default

def _r(v: float, d: int = 2) -> str:
    """有効数字 d 桁に丸めて文字列化"""
    if v == 0:
        return '0'
    from math import log10, floor
    mag = floor(log10(abs(v)))
    factor = 10 ** (d - 1 - mag)
    return str(round(v * factor) / factor)


# ──────────────────────────────────────────────────────────────────────────────
# 市場レジーム判定
# ──────────────────────────────────────────────────────────────────────────────
def classify_regime(feat: np.ndarray, seq: np.ndarray) -> str:
    adx     = _f(feat, 'adx')
    atr_pct = _f(feat, 'atr_pct50')
    squeeze = _f(feat, 'vol_squeeze')
    tc      = _f(feat, 'trend_consistency')

    if atr_pct > 0.75:
        return 'high-volatility'
    if squeeze > 0 or atr_pct < 0.25:
        return 'ranging/squeeze'
    if adx > 0.22 and abs(tc) > 0.4:
        return 'trending'
    return 'ranging'


# ──────────────────────────────────────────────────────────────────────────────
# 強気 / 弱気 シグナル集計
# ──────────────────────────────────────────────────────────────────────────────
def count_signals(feat: np.ndarray) -> tuple[list, list]:
    """(bull_reasons, bear_reasons) を返す"""
    bull, bear = [], []

    # トレンド
    e8_21 = _f(feat, 'ema8_21'); e21_55 = _f(feat, 'ema21_55'); e55_200 = _f(feat, 'ema55_200')
    if e8_21 > 0.05 and e21_55 > 0 and e55_200 > 0:
        bull.append('EMA fully aligned bullish')
    elif e8_21 < -0.05 and e21_55 < 0 and e55_200 < 0:
        bear.append('EMA fully aligned bearish')

    if _f(feat, 'c_ema200') > 0.2:
        bull.append('price above EMA200')
    elif _f(feat, 'c_ema200') < -0.2:
        bear.append('price below EMA200')

    # モメンタム
    rsi = _f(feat, 'rsi14') * 100
    if 40 < rsi < 65:
        bull.append(f'RSI neutral-bullish ({rsi:.0f})')
    elif rsi > 70:
        bear.append(f'RSI overbought ({rsi:.0f}) — potential reversal')
    elif rsi < 30:
        bull.append(f'RSI oversold ({rsi:.0f}) — potential reversal')
    elif rsi < 40:
        bear.append(f'RSI weak ({rsi:.0f})')

    macd_h = _f(feat, 'macd_hist'); macd_sl = _f(feat, 'macd_slope')
    if macd_h > 0 and macd_sl > 0:
        bull.append('MACD histogram positive and expanding')
    elif macd_h > 0:
        bull.append('MACD histogram positive')
    elif macd_h < 0 and macd_sl < 0:
        bear.append('MACD histogram negative and expanding')
    elif macd_h < 0:
        bear.append('MACD histogram negative')

    stoch_k = _f(feat, 'stoch_k'); stoch_d = _f(feat, 'stoch_d')
    if stoch_k > stoch_d + 0.05 and stoch_k < 0.8:
        bull.append('Stochastic K crossed above D (not overbought)')
    elif stoch_k < stoch_d - 0.05 and stoch_k > 0.2:
        bear.append('Stochastic K crossed below D (not oversold)')
    elif stoch_k > 0.85:
        bear.append('Stochastic overbought')
    elif stoch_k < 0.15:
        bull.append('Stochastic oversold')

    # 構造
    bb_pos = _f(feat, 'bb_pos')
    if bb_pos < 0.2:
        bull.append('price near lower Bollinger Band')
    elif bb_pos > 0.8:
        bear.append('price near upper Bollinger Band')

    don_pos = _f(feat, 'donchian_pos')
    if don_pos > 0.85:
        bull.append('price at 20-bar high (Donchian breakout)')
    elif don_pos < 0.15:
        bear.append('price at 20-bar low (Donchian breakdown)')

    h4 = _f(feat, 'h4_trend')
    if h4 > 0.4:
        bull.append('H4 trend bullish')
    elif h4 < -0.4:
        bear.append('H4 trend bearish')

    # ロウソク足パターン
    if _f(feat, 'engulf_bull') > 0.5:
        bull.append('bullish engulfing candle')
    if _f(feat, 'engulf_bear') > 0.5:
        bear.append('bearish engulfing candle')
    if _f(feat, 'pin_bull') > 0.5:
        bull.append('hammer/bullish pin bar')
    if _f(feat, 'pin_bear') > 0.5:
        bear.append('shooting star/bearish pin bar')

    # 出来高
    if _f(feat, 'obv_slope') > 0.15:
        bull.append('OBV rising (volume confirming up move)')
    elif _f(feat, 'obv_slope') < -0.15:
        bear.append('OBV falling (volume confirming down move)')

    return bull, bear


# ──────────────────────────────────────────────────────────────────────────────
# 過去変化のナラティブ（seq使用）
# ──────────────────────────────────────────────────────────────────────────────
def describe_recent_change(seq: np.ndarray) -> str:
    if seq is None or len(seq) < 4:
        return ''

    n = len(seq)
    mid = n // 2
    recent  = seq[mid:]   # 後半
    earlier = seq[:mid]   # 前半

    lines = []
    rsi_e = np.mean(earlier[:, _IDX['rsi14']]) * 100
    rsi_r = np.mean(recent[:,  _IDX['rsi14']]) * 100
    rsi_d = rsi_r - rsi_e
    if abs(rsi_d) > 5:
        direction = 'risen' if rsi_d > 0 else 'fallen'
        lines.append(f'RSI has {direction} {abs(rsi_d):.0f} points over the past {n//2} hours'
                     f' (from {rsi_e:.0f} to {rsi_r:.0f})')

    macd_e = np.mean(earlier[:, _IDX['macd_hist']])
    macd_r = np.mean(recent[:,  _IDX['macd_hist']])
    if macd_e * macd_r < 0:
        cross = 'turned positive' if macd_r > 0 else 'turned negative'
        lines.append(f'MACD histogram {cross} during this period')
    elif macd_r > macd_e + 0.05:
        lines.append('MACD momentum building (histogram expanding)')
    elif macd_r < macd_e - 0.05:
        lines.append('MACD momentum fading (histogram contracting)')

    atr_e = np.mean(earlier[:, _IDX['atr_pct50']])
    atr_r = np.mean(recent[:,  _IDX['atr_pct50']])
    if atr_r > atr_e + 0.15:
        lines.append('volatility expanding — market becoming more active')
    elif atr_r < atr_e - 0.15:
        lines.append('volatility contracting — market quieting down')

    # ブレイクアウト検出
    don_vals = seq[:, _IDX['donchian_pos']]
    if don_vals[-1] > 0.9 and don_vals[0] < 0.6:
        lines.append('price broke to a new 20-bar high during this period')
    elif don_vals[-1] < 0.1 and don_vals[0] > 0.4:
        lines.append('price broke to a new 20-bar low during this period')

    if not lines:
        return ''
    return 'Recent changes: ' + '; '.join(lines) + '.'


# ──────────────────────────────────────────────────────────────────────────────
# メインプロンプト生成 v2
# ──────────────────────────────────────────────────────────────────────────────
def bar_to_text_v2(feat: np.ndarray, seq: np.ndarray,
                   timestamp: pd.Timestamp = None,
                   use_cot: bool = False) -> str:
    lines = []

    # ── ヘッダー ──────────────────────────────────────────────────────────────
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M') if timestamp is not None else 'unknown'
    lines.append(f'USDJPY H1 — {ts_str}')

    # ── レジーム ──────────────────────────────────────────────────────────────
    regime = classify_regime(feat, seq)
    adx    = _f(feat, 'adx')
    atr_pct = _f(feat, 'atr_pct50')

    regime_desc = {
        'trending':         f'trending market (ADX={adx:.2f}, directional bias strong)',
        'ranging/squeeze':  f'ranging/squeeze market (ADX={adx:.2f}, volatility low)',
        'high-volatility':  f'high-volatility environment (ATR in top {atr_pct*100:.0f}%ile)',
        'ranging':          f'ranging market (ADX={adx:.2f}, no clear direction)',
    }.get(regime, regime)
    lines.append(f'Market regime: {regime_desc}')

    # ── 強弱集計 ──────────────────────────────────────────────────────────────
    bull, bear = count_signals(feat)
    nb, ns = len(bull), len(bear)
    if nb > ns + 1:
        bias = f'BULLISH BIAS ({nb} bullish vs {ns} bearish signals)'
    elif ns > nb + 1:
        bias = f'BEARISH BIAS ({ns} bearish vs {nb} bullish signals)'
    elif nb == 0 and ns == 0:
        bias = 'NEUTRAL (no strong signals)'
    else:
        bias = f'MIXED/CONFLICT ({nb} bullish vs {ns} bearish — unclear direction)'

    lines.append(f'Signal balance: {bias}')
    if bull:
        lines.append('  Bullish: ' + ' | '.join(bull))
    if bear:
        lines.append('  Bearish: ' + ' | '.join(bear))

    # ── 過去変化 ──────────────────────────────────────────────────────────────
    recent_str = describe_recent_change(seq)
    if recent_str:
        lines.append(recent_str)

    # ── トレンド詳細 ──────────────────────────────────────────────────────────
    c200    = _f(feat, 'c_ema200')
    slope   = _f(feat, 'ema200_slope')
    tc      = _f(feat, 'trend_consistency')
    h4      = _f(feat, 'h4_trend')
    ichi_cp = _f(feat, 'ichi_cloud_pos')
    ichi_tk = _f(feat, 'ichi_tk_diff')

    above200  = f"{'above' if c200 > 0 else 'below'} EMA200 by {abs(c200):.2f}ATR"
    slope_str = 'rising' if slope > 0.1 else ('falling' if slope < -0.1 else 'flat')
    tc_str    = f'consistent {tc*100:.0f}% of last 8 bars'
    h4_str    = 'bullish' if h4 > 0.3 else ('bearish' if h4 < -0.3 else 'sideways')
    cloud_str = 'above cloud' if ichi_cp > 0.5 else ('below cloud' if ichi_cp < -0.5 else 'in cloud')

    lines.append(
        f'Trend: price {above200}, EMA200 {slope_str}, '
        f'trend {tc_str}. H4: {h4_str}. Ichimoku: {cloud_str}, '
        f"TK {'bullish' if ichi_tk > 0 else 'bearish'}."
    )

    # ── モメンタム詳細 ────────────────────────────────────────────────────────
    rsi14  = _f(feat, 'rsi14') * 100   # 0-100スケール
    rsi28  = _f(feat, 'rsi28') * 100
    rslope = _f(feat, 'rsi_slope')
    macd_h = _f(feat, 'macd_hist')
    macd_sl= _f(feat, 'macd_slope')
    stoch_k= _f(feat, 'stoch_k') * 100
    wr14   = _f(feat, 'wr14')

    rsi_state  = 'overbought' if rsi14 > 70 else ('oversold' if rsi14 < 30 else 'neutral')
    rsi_dir    = 'rising' if rslope > 0.01 else ('falling' if rslope < -0.01 else 'flat')
    macd_state = ('positive+expanding' if macd_h > 0 and macd_sl > 0
                  else 'positive' if macd_h > 0
                  else 'negative+expanding' if macd_h < 0 and macd_sl < 0
                  else 'negative')
    stoch_state= 'overbought' if stoch_k > 80 else ('oversold' if stoch_k < 20 else 'neutral')

    lines.append(
        f'Momentum: RSI14={rsi14:.0f} ({rsi_state}, {rsi_dir}), '
        f'RSI28={rsi28:.0f}, MACD={macd_state}, '
        f'Stoch={stoch_k:.0f} ({stoch_state}), '
        f"WR14={'overbought' if wr14 > 0.3 else ('oversold' if wr14 < -0.3 else 'neutral')}."
    )

    # ── ボラティリティ ────────────────────────────────────────────────────────
    bb_pos  = _f(feat, 'bb_pos')
    bb_w    = _f(feat, 'bb_width')
    atr_r   = _f(feat, 'atr_ratio')
    squeeze = _f(feat, 'vol_squeeze')
    sq_cnt  = _f(feat, 'bb_squeeze_cnt')

    bb_zone  = 'upper band' if bb_pos > 0.8 else ('lower band' if bb_pos < 0.2 else 'middle zone')
    vol_env  = f"SQUEEZE active ({sq_cnt*100:.0f}% of last 20 bars)" if squeeze > 0 else 'no squeeze'
    atr_level= 'elevated' if atr_pct > 0.7 else ('low' if atr_pct < 0.3 else 'normal')

    lines.append(
        f'Volatility: BB position {bb_zone} ({bb_pos:.2f}), '
        f'{vol_env}, ATR {atr_level} (pct={atr_pct:.2f}), '
        f'ATR/price={atr_r:.4f}.'
    )

    # ── ローソク足 ────────────────────────────────────────────────────────────
    body   = _f(feat, 'body')
    ret1   = _f(feat, 'ret1')
    ret5   = _f(feat, 'ret5')
    accel  = _f(feat, 'ret_accel')
    up_cnt = int(_f(feat, 'consec_up') * 8)
    dn_cnt = int(_f(feat, 'consec_dn') * 8)

    patterns = []
    if _f(feat, 'engulf_bull') > 0.5: patterns.append('bullish-engulfing')
    if _f(feat, 'engulf_bear') > 0.5: patterns.append('bearish-engulfing')
    if _f(feat, 'pin_bull')    > 0.5: patterns.append('hammer')
    if _f(feat, 'pin_bear')    > 0.5: patterns.append('shooting-star')
    if _f(feat, 'is_doji')     > 0.5: patterns.append('doji')
    pat_str = ', '.join(patterns) if patterns else 'no notable pattern'

    streak = (f'{up_cnt} consecutive up bars' if up_cnt > 1
              else f'{dn_cnt} consecutive down bars' if dn_cnt > 1
              else 'mixed direction')
    accel_str = 'accelerating' if accel > 0.001 else ('decelerating' if accel < -0.001 else 'steady')
    body_dir  = 'bullish' if body > 0 else 'bearish'

    lines.append(
        f'Price action: {body_dir} candle ({body:+.2f}ATR), '
        f'ret1={ret1*100:+.3f}%, ret5={ret5*100:+.3f}%, '
        f'{streak}, momentum {accel_str}. '
        f'Pattern: {pat_str}.'
    )

    # ── 構造 ──────────────────────────────────────────────────────────────────
    don_pos  = _f(feat, 'donchian_pos')
    sw_hi    = _f(feat, 'swing_hi_dist')
    sw_lo    = _f(feat, 'swing_lo_dist')
    round_d  = _f(feat, 'round_dist')
    wk_pos   = _f(feat, 'weekly_pos')
    vwap_pos = _f(feat, 'price_vs_vwap')
    gap      = _f(feat, 'gap_open')

    don_str  = 'near 20-bar high' if don_pos > 0.8 else ('near 20-bar low' if don_pos < 0.2 else 'mid-range')
    vwap_str = 'above VWAP' if vwap_pos > 0.1 else ('below VWAP' if vwap_pos < -0.1 else 'at VWAP')
    round_str= f'near round number ({round_d:.2f}ATR away)' if round_d < 0.5 else 'away from round number'
    gap_str  = f'gap open {gap:+.2f}ATR' if abs(gap) > 0.1 else 'no gap'
    wk_str   = 'upper weekly range' if wk_pos > 0.7 else ('lower weekly range' if wk_pos < 0.3 else 'mid weekly range')

    lines.append(
        f'Structure: Donchian {don_str}, {vwap_str}, {wk_str}, '
        f'resistance {sw_hi:.2f}ATR / support {sw_lo:.2f}ATR above/below, '
        f'{round_str}, {gap_str}.'
    )

    # ── 出来高 ────────────────────────────────────────────────────────────────
    vol_r   = _f(feat, 'vol_ratio')
    obv_sl  = _f(feat, 'obv_slope')
    vol_tr  = _f(feat, 'vol_trend')

    vol_str = f'volume {vol_r:.1f}x average'
    obv_str = 'OBV rising' if obv_sl > 0.1 else ('OBV falling' if obv_sl < -0.1 else 'OBV flat')
    vtr_str = 'volume expanding' if vol_tr > 0.2 else ('volume contracting' if vol_tr < -0.2 else 'volume stable')

    lines.append(f'Volume: {vol_str}, {obv_str}, {vtr_str}.')

    # ── セッション ────────────────────────────────────────────────────────────
    hour_sin = _f(feat, 'hour_sin'); hour_cos = _f(feat, 'hour_cos')
    dow_sin  = _f(feat, 'dow_sin');  dow_cos  = _f(feat, 'dow_cos')
    hour = int(round(np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi))) % 24
    dow  = int(round(np.arctan2(dow_sin,  dow_cos)  *  5 / (2 * np.pi))) % 5
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    sessions = []
    if _f(feat, 'is_tokyo')  > 0.5: sessions.append('Tokyo')
    if _f(feat, 'is_london') > 0.5: sessions.append('London')
    if _f(feat, 'is_ny')     > 0.5: sessions.append('New York')
    if _f(feat, 'is_overlap')> 0.5: sessions.append('[overlap]')
    sess_str = ' + '.join(sessions) if sessions else 'off-hours'

    lines.append(f'Session: {dow_names[dow]} {hour:02d}:00 UTC — {sess_str}.')

    # ── CoT プレースホルダー ──────────────────────────────────────────────────
    if use_cot:
        lines.append('\nStep-by-step analysis:')
    else:
        lines.append('\nTrading signal:')

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CoT ラベル生成
# ──────────────────────────────────────────────────────────────────────────────
def make_cot_label(feat: np.ndarray, label: str) -> str:
    """推論過程付きラベルを生成"""
    bull, bear = count_signals(feat)
    regime = classify_regime(feat, None)
    rsi14  = _f(feat, 'rsi14') * 100
    macd_h = _f(feat, 'macd_hist')
    adx    = _f(feat, 'adx')

    reasons = []

    if label == 'BUY':
        if bull:
            reasons.append(f"{len(bull)} bullish signals present ({bull[0]})")
        if rsi14 < 65:
            reasons.append(f"RSI at {rsi14:.0f} has room to rise")
        if macd_h > 0:
            reasons.append("MACD positive")
        if adx > 0.2:
            reasons.append(f"ADX {adx:.2f} confirms trend strength")
        reason_str = '; '.join(reasons) if reasons else 'bullish setup'
        return f'{reason_str}. → BUY'

    elif label == 'SELL':
        if bear:
            reasons.append(f"{len(bear)} bearish signals present ({bear[0]})")
        if rsi14 > 55:
            reasons.append(f"RSI at {rsi14:.0f} elevated")
        if macd_h < 0:
            reasons.append("MACD negative")
        reason_str = '; '.join(reasons) if reasons else 'bearish setup'
        return f'{reason_str}. → SELL'

    else:  # HOLD
        if len(bull) > 0 and len(bear) > 0:
            reasons.append(f"conflicting signals ({len(bull)} bull vs {len(bear)} bear)")
        if regime in ('ranging', 'ranging/squeeze'):
            reasons.append(f"ranging market (ADX={adx:.2f})")
        reason_str = '; '.join(reasons) if reasons else 'no clear setup'
        return f'{reason_str}. → HOLD'


LABEL_MAP = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}


# ──────────────────────────────────────────────────────────────────────────────
# データセット生成 v2
# ──────────────────────────────────────────────────────────────────────────────
def build_llm_dataset_v2(
        csv_path:     str,
        seq_len:      int   = 20,
        tp_atr:       float = 1.5,
        sl_atr:       float = 1.0,
        forward_bars: int   = 20,
        max_samples:  int   = 0,
        seed:         int   = 42,
        use_cot:      bool  = False,
) -> tuple[list, list]:
    """
    Returns (train_samples, test_samples)
    use_cot=True の場合 label は 'Trend bullish, RSI rising. → BUY' 形式
    use_cot=False の場合 label は 'BUY' / 'SELL' / 'HOLD'
    """
    print('=== LLM データセット v2 生成 ===')
    print(f'  CoT モード: {use_cot}')
    t0 = time.time()

    df = load_data(csv_path, timeframe='H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    test_start = df.index[-1] - timedelta(days=365)
    df_tr = df[df.index < test_start].copy()
    df_te = df[df.index >= test_start].copy()

    def df_to_samples(df_part: pd.DataFrame, label_tag: str = '') -> list:
        feat_arr   = df_part[FEATURE_COLS].values.astype(np.float32)
        labels     = make_labels(df_part, tp_atr=tp_atr, sl_atr=sl_atr,
                                  forward_bars=forward_bars)
        timestamps = df_part.index
        n          = len(feat_arr)
        samples    = []

        for i in range(seq_len, n - forward_bars - 1):
            seq  = feat_arr[i - seq_len: i]
            feat = feat_arr[i - 1]
            ts   = timestamps[i - 1]
            lbl_str = LABEL_MAP[labels[i - 1]]

            prompt = bar_to_text_v2(feat, seq, timestamp=ts, use_cot=use_cot)
            final_label = make_cot_label(feat, lbl_str) if use_cot else lbl_str

            samples.append({'prompt': prompt, 'label': final_label})

        counts = {v: sum(1 for s in samples
                         if (s['label'].endswith(v) if use_cot else s['label'] == v))
                  for v in LABEL_MAP.values()}
        print(f'  {label_tag}: {len(samples):,} samples | '
              + ' | '.join(f'{k}:{v:,}' for k, v in counts.items()))
        return samples

    train_samples = df_to_samples(df_tr, 'TRAIN')
    test_samples  = df_to_samples(df_te, 'TEST')

    if max_samples > 0 and len(train_samples) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_samples), max_samples, replace=False)
        idx.sort()
        train_samples = [train_samples[i] for i in idx]
        print(f'  訓練データを {max_samples:,} に削減')

    print(f'  生成完了: {time.time()-t0:.1f}秒')
    return train_samples, test_samples


def save_jsonl(samples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'  保存: {path}  ({len(samples):,} samples)')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seq_len',   type=int,   default=20)
    p.add_argument('--tp',        type=float, default=1.5)
    p.add_argument('--sl',        type=float, default=1.0)
    p.add_argument('--forward',   type=int,   default=20)
    p.add_argument('--max_train', type=int,   default=0)
    p.add_argument('--cot',       action='store_true', help='Chain-of-Thoughtラベル使用')
    p.add_argument('--seed',      type=int,   default=42)
    args = p.parse_args()

    suffix = '_cot' if args.cot else ''
    train_path = OUT_DIR / f'llm_train_v2{suffix}.jsonl'
    test_path  = OUT_DIR / f'llm_test_v2{suffix}.jsonl'

    train_s, test_s = build_llm_dataset_v2(
        str(DATA_PATH),
        seq_len=args.seq_len, tp_atr=args.tp, sl_atr=args.sl,
        forward_bars=args.forward, max_samples=args.max_train,
        seed=args.seed, use_cot=args.cot,
    )
    save_jsonl(train_s, train_path)
    save_jsonl(test_s,  test_path)

    print('\n--- サンプルプロンプト (先頭1件) ---')
    print(train_s[0]['prompt'])
    print(f'Label: {train_s[0]["label"]}')
