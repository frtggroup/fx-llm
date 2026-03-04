"""
特徴量エンジニアリング v5 - 700超の基底特徴量
各特徴量に過去24ステップの1本差分を付加し、最終的に17500次元程度になりますが、
必要に応じて削ってください。
"""
import numpy as np
import pandas as pd
from pathlib import Path

# 基底特徴量リストを作成
BASE_FEATURE_COLS = []

# トレンド系 (16)
for p in [5, 10, 20, 50, 100, 200]:
    BASE_FEATURE_COLS.extend([f"sma_{p}", f"ema_{p}"])
    if p <= 50:
        BASE_FEATURE_COLS.extend([f"hma_{p}", f"wma_{p}"])

# MACD (9)
for fast, slow, sig in [(12,26,9), (6,19,9), (5,34,5)]:
    BASE_FEATURE_COLS.extend([f"macd_{fast}_{slow}_{sig}", f"macdsig_{fast}_{slow}_{sig}", f"macdhist_{fast}_{slow}_{sig}"])

# RSI (4)
for p in [7, 14, 21, 28]:
    BASE_FEATURE_COLS.extend([f"rsi_{p}"])

# Stochastic (6)
for k, d in [(14,3), (9,3), (21,5)]:
    BASE_FEATURE_COLS.extend([f"stoch_k_{k}_{d}", f"stoch_d_{k}_{d}"])

# BB (4)
for p in [20, 50]:
    BASE_FEATURE_COLS.extend([f"bb_pos_{p}", f"bb_width_{p}"])

# ATR, ADX, PDI, NDI, CCI, WR (18)
for p in [7, 14, 28]:
    BASE_FEATURE_COLS.extend([f"atr_{p}", f"adx_{p}", f"pdi_{p}", f"ndi_{p}"])
    BASE_FEATURE_COLS.extend([f"cci_{p}", f"wr_{p}"])

# その他の指標群 (10)
BASE_FEATURE_COLS.extend([
    "psar",
    "pivot", "r1", "r2", "s1", "s2",
    "donchian_pos_20", "donchian_width_20",
    "kc_pos_20", "kc_width_20"
])

# 乖離率 (6)
for p in [20, 50, 200]:
    BASE_FEATURE_COLS.extend([f"diff_sma_{p}", f"diff_ema_{p}"])

# 一目均衡表 (7)
BASE_FEATURE_COLS.extend([
    "ichi_tenkan", "ichi_kijun", "ichi_senkou_a", "ichi_senkou_b",
    "ichi_cloud_pos", "ichi_cloud_width", "ichi_tk_cross"
])

# 統計 (15)
for p in [10, 20, 30]:
    BASE_FEATURE_COLS.extend([
        f"roll_mean_{p}", f"roll_std_{p}", f"roll_skew_{p}", f"roll_kurt_{p}",
        f"zscore_{p}"
    ])

# ローソク足 (9)
BASE_FEATURE_COLS.extend([
    "body", "upper_w", "lower_w", "tr", "is_doji", "is_bull_engulf", "is_bear_engulf",
    "is_hammer", "is_inv_hammer"
])

# BOS & 構造 (15)
BASE_FEATURE_COLS.extend([
    "hh", "hl", "lh", "ll",
    "bos", "bos_dir", "bos_bars", "bos_time", "bos_pips",
    "bos_fibo_zone", "bos_retest", "bos_move_pips",
    "bos_ema_sync", "bos_macd_sync", "bos_rsi_sync"
])

# セッション・時間 (8)
BASE_FEATURE_COLS.extend([
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_tokyo", "is_london", "is_ny", "is_overlap"
])

# 追加の遅延/差分特徴
for p in range(5, 51, 5):
    BASE_FEATURE_COLS.extend([f"ret1_{p}", f"ret5_{p}", f"vol_{p}"])

for p in range(5, 120, 5):
    BASE_FEATURE_COLS.extend([f"momentum_{p}", f"volatility_{p}", f"price_diff_{p}"])

# ※ extra_random_feat は削除 — 実在する特徴量のみ使用

N_GROUPS = len(BASE_FEATURE_COLS)  # 230

# 拡張特徴量リスト 
_N_DIFFS = 24
FEATURE_COLS = []
for _col in BASE_FEATURE_COLS:
    FEATURE_COLS.append(_col)
    for _k in range(1, _N_DIFFS + 1):
        FEATURE_COLS.append(f'{_col}_d{_k}')

N_FEATURES = len(FEATURE_COLS)
FEATURE_GROUPS = {
    i: list(range(i * (_N_DIFFS + 1), (i + 1) * (_N_DIFFS + 1)))
    for i in range(N_GROUPS)
}

def expand_groups(group_indices) -> list[int]:
    cols = []
    for g in sorted(int(g) for g in group_indices if 0 <= int(g) < N_GROUPS):
        cols.extend(FEATURE_GROUPS[g])
    return cols

def load_data(csv_path: str, timeframe: str = 'H1') -> pd.DataFrame:
    print(f"  CSV読込: {Path(csv_path).name}")
    df = pd.read_csv(csv_path, sep='\t')
    df.columns = [c.strip('<>') for c in df.columns]
    df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df.set_index('datetime', inplace=True)
    vol_col = 'TICKVOL' if 'TICKVOL' in df.columns else ('VOL' if 'VOL' in df.columns else 'REAL_VOLUME')
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', vol_col]].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    fname_upper = Path(csv_path).stem.upper()
    already_h1 = ('_H1' in fname_upper or fname_upper.endswith('H1'))

    tf_map = {'M1': None, 'H1': '1h', 'H4': '4h', 'D1': '1D'}
    rule = tf_map.get(timeframe.upper())
    if rule and not already_h1:
        print(f"  {timeframe}へリサンプル中...")
        df = df.resample(rule).agg(
            open=('open', 'first'), high=('high', 'max'),
            low=('low', 'min'),    close=('close', 'last'),
            volume=('volume', 'sum')
        ).dropna()
        df = df[df.index.dayofweek < 5]
    elif already_h1:
        df = df[df.index.dayofweek < 5]
        print(f"  H1 CSV を直接使用 (リサンプルなし)")

    print(f"  {timeframe}: {len(df):,}本  {df.index[0]} ～ {df.index[-1]}")
    return df

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _rma(s, period): return s.ewm(alpha=1.0 / period, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()
def _wma(s, p):
    weights = np.arange(1, p + 1)
    return s.rolling(p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def _hma(s, p):
    half_length = int(p / 2)
    sqrt_length = int(np.sqrt(p))
    wmaf = _wma(s, half_length)
    wmas = _wma(s, p)
    return _wma(wmaf * 2 - wmas, sqrt_length)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    全特徴量を計算して df に追加する。
    実装は feat/ パッケージの各モジュールに分割されている。
    """
    from feat import add_all_indicators
    df = add_all_indicators(df)

    # 差分展開 (d1 〜 d24) — 一括 concat でフラグメント化を回避
    diff1 = df[BASE_FEATURE_COLS].diff(1)
    diff_frames = []
    for k in range(1, _N_DIFFS + 1):
        shifted = diff1.shift(k - 1).fillna(0).astype(np.float32)
        shifted.columns = [f'{col}_d{k}' for col in BASE_FEATURE_COLS]
        diff_frames.append(shifted)
    df = pd.concat([df] + diff_frames, axis=1)

    return df

def make_labels(df: pd.DataFrame, tp_atr=1.5, sl_atr=1.0, forward_bars=20) -> np.ndarray:
    n = len(df)
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    atr = df.get('atr_14', df['high'] - df['low']).values.astype(np.float64)

    valid = n - forward_bars - 1
    entry = close[:valid].reshape(-1, 1)
    a = atr[:valid].reshape(-1, 1)
    tp_dist = tp_atr * a; sl_dist = sl_atr * a

    tp_l = entry + tp_dist; sl_l = entry - sl_dist
    tp_s = entry - tp_dist; sl_s = entry + sl_dist

    H = np.empty((valid, forward_bars), dtype=np.float64)
    L = np.empty((valid, forward_bars), dtype=np.float64)
    for j in range(forward_bars):
        H[:, j] = high[j+1: j+1+valid]
        L[:, j] = low[j+1: j+1+valid]

    def _first(mask):
        any_hit = mask.any(axis=1)
        return np.where(any_hit, np.argmax(mask, axis=1), forward_bars)

    ltp = _first(H >= tp_l); lsl = _first(L <= sl_l)
    stp = _first(L <= tp_s); ssl = _first(H >= sl_s)
    long_ok = ltp < lsl; short_ok = stp < ssl
    both = long_ok & short_ok

    lbl = np.where(long_ok & ~short_ok, 1,
            np.where(short_ok & ~long_ok, 2,
            np.where(both & (ltp <= stp), 1,
            np.where(both & (stp < ltp), 2, 0))))

    labels = np.zeros(n, dtype=np.int64)
    labels[:valid] = lbl
    return labels

def build_dataset(df, seq_len=20, tp_atr=1.5, sl_atr=1.0, forward_bars=20, label_fn=None, feat_indices=None, feat_precomputed=None) -> tuple:
    labels = make_labels(df, tp_atr=tp_atr, sl_atr=sl_atr, forward_bars=forward_bars)

    # feat_precomputed: worker_init で事前計算済みのfloat32配列を使う（毎試行の910MB再確保を回避）
    if feat_precomputed is not None:
        feat = feat_precomputed
    else:
        feat = df[FEATURE_COLS].values.astype(np.float32)
    if feat_indices is not None:
        feat = feat[:, feat_indices]
    n_feat = feat.shape[1]

    n = len(feat)
    valid_start = seq_len
    valid_end = n - forward_bars - 1
    n_samples = valid_end - valid_start

    print(f"  サンプル数: {n_samples:,}  特徴量: {n_feat}  "
          f"(HOLD:{(labels[:valid_end]==0).sum():,} "
          f"BUY:{(labels[:valid_end]==1).sum():,} "
          f"SELL:{(labels[:valid_end]==2).sum():,})")

    stride = feat.strides
    all_shape = (n - seq_len + 1, seq_len, n_feat)
    all_stride = (stride[0], stride[0], stride[1])
    all_windows = np.lib.stride_tricks.as_strided(feat, shape=all_shape, strides=all_stride)
    
    X = all_windows[valid_start - seq_len: valid_end - seq_len].copy()
    y = labels[valid_start - 1: valid_end - 1].copy().astype(np.int64)

    return X, y, labels
