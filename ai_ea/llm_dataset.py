"""
LLM用データセット生成 v1
70次元特徴量 → 英語テキスト記述に変換し JSONL 形式で保存

生成されるサンプル例:
  prompt: "USDJPY H1 Trade Signal\n[TREND] bullish aligned..."
  label : "BUY" / "SELL" / "HOLD"
"""
import sys, json, time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (load_data, add_indicators, make_labels,
                      FEATURE_COLS, N_FEATURES)

DATA_PATH  = Path(__file__).parent.parent / 'USDJPY_M1_202301012206_202602250650.csv'
OUT_DIR    = Path(__file__).parent
TRAIN_JSONL = OUT_DIR / 'llm_train.jsonl'
TEST_JSONL  = OUT_DIR / 'llm_test.jsonl'


# ──────────────────────────────────────────────────────────────────────────
# 特徴量インデックス
# ──────────────────────────────────────────────────────────────────────────
_IDX = {col: i for i, col in enumerate(FEATURE_COLS)}


def _f(feat: np.ndarray, col: str, default: float = 0.0) -> float:
    idx = _IDX.get(col)
    return float(feat[idx]) if idx is not None else default


def _cat(val: float, lo: float, hi: float,
         labels=('low', 'mid', 'high')) -> str:
    if val < lo:   return labels[0]
    if val > hi:   return labels[2]
    return labels[1]


# ──────────────────────────────────────────────────────────────────────────
# 単バー → テキスト変換
# ──────────────────────────────────────────────────────────────────────────
def bar_to_text(feat: np.ndarray, seq: np.ndarray,
                timestamp: pd.Timestamp = None) -> str:
    """
    feat  : 最新バーの特徴量ベクトル [70]
    seq   : シーケンス全体 [seq_len, 70] (トレンド要約用)
    """
    lines = []

    # ── ヘッダー ──────────────────────────────────────────────────────────
    if timestamp is not None:
        lines.append(f"USDJPY H1 Trade Signal - {timestamp.strftime('%Y-%m-%d %H:%M')}")
    else:
        lines.append("USDJPY H1 Trade Signal")

    # ── [TREND] ─────────────────────────────────────────────────────────
    c200 = _f(feat, 'c_ema200')
    e8_21   = _f(feat, 'ema8_21')
    e21_55  = _f(feat, 'ema21_55')
    e55_200 = _f(feat, 'ema55_200')
    adx     = _f(feat, 'adx')
    slope200= _f(feat, 'ema200_slope')
    tc      = _f(feat, 'trend_consistency')  # -1~1

    # EMA整列
    if e8_21 > 0.05 and e21_55 > 0 and e55_200 > 0:
        align_str = "bullish aligned (8>21>55>200)"
    elif e8_21 < -0.05 and e21_55 < 0 and e55_200 < 0:
        align_str = "bearish aligned (8<21<55<200)"
    else:
        align_str = "mixed/transitional"

    above200 = "above" if c200 > 0 else "below"
    adx_str  = "strong" if adx > 0.25 else ("moderate" if adx > 0.15 else "weak")
    slope_str= "rising" if slope200 > 0.1 else ("falling" if slope200 < -0.1 else "flat")
    tc_str   = "consistent" if tc > 0.5 else ("mixed" if tc > -0.5 else "inconsistent")

    lines.append(f"[TREND] EMA: {align_str}, price {above200} EMA200 "
                 f"({c200:+.2f}ATR), ADX: {adx_str} ({adx:.2f}), "
                 f"EMA200 slope: {slope_str}, consistency: {tc_str}")

    # H4/Ichimoku
    h4      = _f(feat, 'h4_trend')
    ichi_tk = _f(feat, 'ichi_tk_diff')
    ichi_cp = _f(feat, 'ichi_cloud_pos')
    h4_str  = "uptrend" if h4 > 0.3 else ("downtrend" if h4 < -0.3 else "sideways")
    cloud_str = "above cloud" if ichi_cp > 0.5 else ("below cloud" if ichi_cp < -0.5 else "in cloud")
    tk_str  = "TK bullish" if ichi_tk > 0 else "TK bearish"
    lines.append(f"[ICHIMOKU] H4: {h4_str}, {cloud_str}, {tk_str}")

    # ── [MOMENTUM] ──────────────────────────────────────────────────────
    rsi14   = _f(feat, 'rsi14')
    rsi28   = _f(feat, 'rsi28')
    rslope  = _f(feat, 'rsi_slope')
    macd_h  = _f(feat, 'macd_hist')
    macd_sl = _f(feat, 'macd_slope')
    stoch_k = _f(feat, 'stoch_k')
    stoch_d = _f(feat, 'stoch_d')
    wr14    = _f(feat, 'wr14')  # 0.5オフセット済み (-0.5~0.5)

    rsi_str  = "overbought" if rsi14 > 0.70 else ("oversold" if rsi14 < 0.30 else "neutral")
    rsi28_str= "overbought" if rsi28 > 0.70 else ("oversold" if rsi28 < 0.30 else "neutral")
    rs_str   = "rising" if rslope > 0.01 else ("falling" if rslope < -0.01 else "flat")
    macd_str = ("positive+expanding" if macd_h > 0.1 and macd_sl > 0
                else "positive" if macd_h > 0
                else "negative-expanding" if macd_h < -0.1 and macd_sl < 0
                else "negative")
    stoch_zone = "overbought" if stoch_k > 0.8 else ("oversold" if stoch_k < 0.2 else "neutral")
    stoch_cross= "K>D bullish" if stoch_k > stoch_d + 0.05 else ("K<D bearish" if stoch_k < stoch_d - 0.05 else "flat")
    wr_str   = "overbought" if wr14 > 0.3 else ("oversold" if wr14 < -0.3 else "neutral")

    lines.append(f"[MOMENTUM] RSI14: {rsi_str} ({rsi14:.2f}, {rs_str}), "
                 f"RSI28: {rsi28_str}, MACD: {macd_str}, "
                 f"Stoch: {stoch_zone} {stoch_cross}, WR: {wr_str}")

    # CCI / ROC
    cci  = _f(feat, 'cci14')
    roc  = _f(feat, 'roc20')
    cci_str = "overbought" if cci > 1.0 else ("oversold" if cci < -1.0 else "neutral")
    roc_str = f"{'up' if roc > 0 else 'down'} {abs(roc)*100:.2f}% (20H)"
    lines.append(f"[ROC/CCI] ROC20: {roc_str}, CCI: {cci_str} ({cci:.2f})")

    # ── [VOLATILITY] ────────────────────────────────────────────────────
    bb_pos  = _f(feat, 'bb_pos')
    bb_w    = _f(feat, 'bb_width')
    atr_r   = _f(feat, 'atr_ratio')
    kc_pos  = _f(feat, 'kc_pos')
    squeeze = _f(feat, 'vol_squeeze')
    sq_cnt  = _f(feat, 'bb_squeeze_cnt')
    atr_pct = _f(feat, 'atr_pct50')

    bb_zone  = "upper band" if bb_pos > 0.8 else ("lower band" if bb_pos < 0.2 else "middle zone")
    kc_zone  = "upper KC" if kc_pos > 0.8 else ("lower KC" if kc_pos < 0.2 else "mid KC")
    sq_str   = f"SQUEEZE ({sq_cnt*100:.0f}%)" if squeeze > 0 else "no squeeze"
    atr_str  = "high ATR" if atr_pct > 0.7 else ("low ATR" if atr_pct < 0.3 else "normal ATR")

    lines.append(f"[VOLATILITY] BB: {bb_zone} ({bb_pos:.2f}), KC: {kc_zone}, "
                 f"{sq_str}, {atr_str} (pct50={atr_pct:.2f}), ATR/price={atr_r:.3f}")

    # ── [CANDLE] ────────────────────────────────────────────────────────
    body     = _f(feat, 'body')
    upper_w  = _f(feat, 'upper_w')
    lower_w  = _f(feat, 'lower_w')
    ret1     = _f(feat, 'ret1')
    ret5     = _f(feat, 'ret5')
    consec_u = _f(feat, 'consec_up')
    consec_d = _f(feat, 'consec_dn')
    engulf_b = _f(feat, 'engulf_bull')
    engulf_s = _f(feat, 'engulf_bear')
    pin_b    = _f(feat, 'pin_bull')
    pin_s    = _f(feat, 'pin_bear')
    doji     = _f(feat, 'is_doji')
    accel    = _f(feat, 'ret_accel')

    body_str = f"{'bullish' if body > 0 else 'bearish'} ({body:+.2f}ATR)"
    wick_str = f"upper={upper_w:.2f} lower={lower_w:.2f}"

    patterns = []
    if engulf_b > 0.5: patterns.append("bullish-engulfing")
    if engulf_s > 0.5: patterns.append("bearish-engulfing")
    if pin_b > 0.5:    patterns.append("hammer/pin-bull")
    if pin_s > 0.5:    patterns.append("shooting-star/pin-bear")
    if doji > 0.5:     patterns.append("doji")
    pat_str = ", ".join(patterns) if patterns else "no pattern"

    streak = (f"{int(consec_u*8)} up bars" if consec_u > 0.1
              else f"{int(consec_d*8)} down bars" if consec_d > 0.1
              else "flat")
    accel_str = "accelerating" if accel > 0.001 else ("decelerating" if accel < -0.001 else "steady")

    lines.append(f"[CANDLE] Body: {body_str}, wicks: {wick_str}, "
                 f"ret1={ret1*100:+.3f}%, ret5={ret5*100:+.3f}%, "
                 f"streak: {streak}, momentum: {accel_str}, pattern: {pat_str}")

    # ── [STRUCTURE] ─────────────────────────────────────────────────────
    don_pos  = _f(feat, 'donchian_pos')
    sw_hi    = _f(feat, 'swing_hi_dist')
    sw_lo    = _f(feat, 'swing_lo_dist')
    round_d  = _f(feat, 'round_dist')
    wk_pos   = _f(feat, 'weekly_pos')
    day_pos  = _f(feat, 'daily_range_pos')
    gap      = _f(feat, 'gap_open')
    vwap_pos = _f(feat, 'price_vs_vwap')

    don_str  = "near top" if don_pos > 0.8 else ("near bottom" if don_pos < 0.2 else "mid-range")
    wk_str   = "upper weekly" if wk_pos > 0.7 else ("lower weekly" if wk_pos < 0.3 else "mid weekly")
    day_str  = "upper daily" if day_pos > 0.7 else ("lower daily" if day_pos < 0.3 else "mid daily")
    round_str= "near round" if round_d < 0.3 else ("away from round" if round_d > 1.0 else "mid")
    vwap_str = "above VWAP" if vwap_pos > 0.1 else ("below VWAP" if vwap_pos < -0.1 else "at VWAP")
    gap_str  = f"gap {gap:+.2f}ATR" if abs(gap) > 0.1 else "no gap"

    lines.append(f"[STRUCTURE] Donchian: {don_str} ({don_pos:.2f}), "
                 f"{wk_str}, {day_str}, {vwap_str}, "
                 f"resistance {sw_hi:.2f}ATR / support {sw_lo:.2f}ATR, "
                 f"{round_str}, {gap_str}")

    # ── [VOLUME] ────────────────────────────────────────────────────────
    vol_r   = _f(feat, 'vol_ratio')
    obv_sl  = _f(feat, 'obv_slope')
    vol_tr  = _f(feat, 'vol_trend')

    vol_str = "high vol" if vol_r > 1.5 else ("low vol" if vol_r < 0.5 else "normal vol")
    obv_str = "OBV rising" if obv_sl > 0.1 else ("OBV falling" if obv_sl < -0.1 else "OBV flat")
    vtr_str = "vol expanding" if vol_tr > 0.2 else ("vol contracting" if vol_tr < -0.2 else "vol stable")

    lines.append(f"[VOLUME] {vol_str} ({vol_r:.2f}x avg), {obv_str}, {vtr_str}")

    # ── [SESSION] ───────────────────────────────────────────────────────
    hour_sin = _f(feat, 'hour_sin')
    hour_cos = _f(feat, 'hour_cos')
    dow_sin  = _f(feat, 'dow_sin')
    dow_cos  = _f(feat, 'dow_cos')
    is_tok   = _f(feat, 'is_tokyo')
    is_lon   = _f(feat, 'is_london')
    is_ny    = _f(feat, 'is_ny')
    is_ovl   = _f(feat, 'is_overlap')

    hour = int(round(np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi))) % 24
    dow  = int(round(np.arctan2(dow_sin,  dow_cos)  *  5 / (2 * np.pi))) % 5
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    sessions = []
    if is_tok > 0.5: sessions.append("Tokyo")
    if is_lon > 0.5: sessions.append("London")
    if is_ny  > 0.5: sessions.append("NY")
    if is_ovl > 0.5: sessions.append("[overlap]")
    sess_str = "+".join(sessions) if sessions else "off-hours"

    lines.append(f"[SESSION] {dow_names[dow]} {hour:02d}:00, {sess_str}")

    # ── [TREND HISTORY] (シーケンス全体の要約) ──────────────────────────
    if seq is not None and len(seq) > 2:
        rsi_vals = seq[:, _IDX['rsi14']]
        macd_vals= seq[:, _IDX['macd_hist']]
        bb_vals  = seq[:, _IDX['bb_pos']]

        rsi_trend  = "rising" if rsi_vals[-1] > rsi_vals[0] + 0.05 else ("falling" if rsi_vals[-1] < rsi_vals[0] - 0.05 else "flat")
        macd_trend = "rising" if macd_vals[-1] > macd_vals[0] else "falling"
        bb_trend   = "expanding" if bb_vals[-1] > bb_vals[0] + 0.1 else ("contracting" if bb_vals[-1] < bb_vals[0] - 0.1 else "stable")
        n_bars = len(seq)

        lines.append(f"[HISTORY {n_bars}H] RSI: {rsi_trend}, MACD hist: {macd_trend}, BB: {bb_trend}")

    lines.append("\nSignal:")
    return "\n".join(lines)


LABEL_MAP = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}


# ──────────────────────────────────────────────────────────────────────────
# データセット生成
# ──────────────────────────────────────────────────────────────────────────
def build_llm_dataset(
        csv_path: str,
        seq_len: int      = 20,
        tp_atr: float     = 1.5,
        sl_atr: float     = 1.0,
        forward_bars: int = 20,
        timeframe: str    = 'H1',
        max_samples: int  = 0,
        seed: int         = 42,
) -> tuple[list, list]:
    """
    Returns (train_samples, test_samples) where each sample = {'prompt': str, 'label': str}
    """
    print("=== LLM データセット生成 ===")
    t0 = time.time()

    from features import build_dataset
    df = load_data(csv_path, timeframe=timeframe)
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    test_start = df.index[-1] - timedelta(days=365)
    df_tr = df[df.index < test_start].copy()
    df_te = df[df.index >= test_start].copy()

    def df_to_samples(df_part: pd.DataFrame, label: str = '') -> list:
        feat_arr = df_part[FEATURE_COLS].values.astype(np.float32)
        labels   = make_labels(df_part, tp_atr=tp_atr, sl_atr=sl_atr,
                               forward_bars=forward_bars)
        timestamps = df_part.index

        n       = len(feat_arr)
        valid_s = seq_len
        valid_e = n - forward_bars - 1
        samples = []

        for i in range(valid_s, valid_e):
            seq  = feat_arr[i - seq_len: i]   # [seq_len, 70]
            feat = feat_arr[i - 1]             # 最新バー
            ts   = timestamps[i - 1]
            lbl  = LABEL_MAP[labels[i - 1]]
            prompt = bar_to_text(feat, seq, timestamp=ts)
            samples.append({'prompt': prompt, 'label': lbl})

        # クラスバランス情報
        counts = {v: sum(1 for s in samples if s['label'] == v) for v in LABEL_MAP.values()}
        print(f"  {label}: {len(samples):,} samples | "
              + " | ".join(f"{k}:{v:,}" for k, v in counts.items()))
        return samples

    train_samples = df_to_samples(df_tr, 'TRAIN')
    test_samples  = df_to_samples(df_te, 'TEST')

    # 訓練データを最大 max_samples に制限
    if max_samples > 0 and len(train_samples) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_samples), max_samples, replace=False)
        idx.sort()
        train_samples = [train_samples[i] for i in idx]
        print(f"  訓練データを {max_samples:,} に削減 (ランダムサンプリング)")

    print(f"  生成完了: {time.time()-t0:.1f}秒")
    return train_samples, test_samples


def save_jsonl(samples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"  保存: {path}  ({len(samples):,} samples)")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seq_len',   type=int,   default=20)
    p.add_argument('--tp',        type=float, default=1.5)
    p.add_argument('--sl',        type=float, default=1.0)
    p.add_argument('--forward',   type=int,   default=20)
    p.add_argument('--max_train', type=int,   default=0,
                   help='訓練サンプル上限 (0=全部)')
    p.add_argument('--seed',      type=int,   default=42)
    args = p.parse_args()

    train_s, test_s = build_llm_dataset(
        str(DATA_PATH),
        seq_len=args.seq_len, tp_atr=args.tp, sl_atr=args.sl,
        forward_bars=args.forward, max_samples=args.max_train, seed=args.seed,
    )
    save_jsonl(train_s, TRAIN_JSONL)
    save_jsonl(test_s,  TEST_JSONL)

    # サンプル表示
    print("\n--- サンプルプロンプト (先頭1件) ---")
    s = train_s[0]
    print(s['prompt'])
    print(f"Label: {s['label']}")
