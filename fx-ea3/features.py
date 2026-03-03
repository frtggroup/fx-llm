"""
特徴量エンジニアリング v4 - 1750次元 (70基底 × 25列)
逐次差分拡張: 各特徴量に過去24ステップの1本差分を付加

レイアウト (グループ単位):
  グループ i = [col_i, col_i_d1, col_i_d2, ..., col_i_d24]
  → col_i_dk = f[t-(k-1)] - f[t-k]  (k=1: 最新, k=24: 24本前)

FEATURE_GROUPS[i] = [i*25, i*25+1, ..., i*25+24]  (25列のカラムインデックス)
N_GROUPS = 70   (基底特徴量の数 = 選択時の単位)
N_FEATURES = 1750  (実際の入力次元)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ─── 基底特徴量リスト (70次元) ────────────────────────────────────────────────
BASE_FEATURE_COLS = [
    # ── トレンド (11) ────────────────────────────────────────────────────
    'c_ema8', 'c_ema21', 'c_ema55', 'c_ema200',
    'ema8_21', 'ema21_55', 'ema55_200',
    'adx', 'pdi_ndi',
    'ema200_slope', 'trend_consistency',

    # ── モメンタム (11) ──────────────────────────────────────────────────
    'rsi14', 'rsi28',
    'macd_hist', 'macd_signal',
    'stoch_k', 'stoch_d', 'stoch_kd_diff',
    'wr14', 'roc20',
    'rsi_slope',
    'macd_slope',

    # ── ボラティリティ (9) ───────────────────────────────────────────────
    'bb_pos', 'bb_width',
    'atr_ratio', 'atr5_14_ratio',
    'kc_pos', 'hv20', 'vol_squeeze',
    'atr_pct50',
    'bb_squeeze_cnt',

    # ── 価格アクション・ローソク足パターン (15) ──────────────────────────
    'body', 'upper_w', 'lower_w',
    'ret1', 'ret5', 'ret20',
    'close_pct_range',
    'consec_up', 'consec_dn',
    'ret_accel',
    'engulf_bull', 'engulf_bear',
    'pin_bull', 'pin_bear',
    'is_doji',

    # ── サポレジ・構造 (8) ──────────────────────────────────────────────
    'donchian_pos',
    'swing_hi_dist', 'swing_lo_dist',
    'round_dist',
    'h4_trend', 'daily_range_pos',
    'weekly_pos',
    'gap_open',

    # ── 一目均衡表 (3) ──────────────────────────────────────────────────
    'ichi_tk_diff',
    'ichi_cloud_pos',
    'ichi_cloud_thick',

    # ── 出来高 (5) ──────────────────────────────────────────────────────
    'vol_ratio',
    'obv_slope',
    'vol_trend',
    'price_vs_vwap',
    'cci14',

    # ── セッション (8) ──────────────────────────────────────────────────
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'is_tokyo', 'is_london', 'is_ny', 'is_overlap',
]

N_GROUPS = len(BASE_FEATURE_COLS)  # 70

# ─── 拡張特徴量リスト (1750次元) ─────────────────────────────────────────────
# レイアウト: [col, col_d1, ..., col_d24, col2, col2_d1, ..., col2_d24, ...]
_N_DIFFS = 24
FEATURE_COLS = []
for _col in BASE_FEATURE_COLS:
    FEATURE_COLS.append(_col)
    for _k in range(1, _N_DIFFS + 1):
        FEATURE_COLS.append(f'{_col}_d{_k}')

N_FEATURES = len(FEATURE_COLS)  # 1750
assert N_FEATURES == N_GROUPS * (_N_DIFFS + 1), f"N_FEATURES={N_FEATURES}"

# ─── グループマップ: グループインデックス → カラムインデックスリスト (25個) ────
# FEATURE_GROUPS[i] = [i*25, i*25+1, ..., i*25+24]
FEATURE_GROUPS: dict[int, list[int]] = {
    i: list(range(i * (_N_DIFFS + 1), (i + 1) * (_N_DIFFS + 1)))
    for i in range(N_GROUPS)
}


def expand_groups(group_indices) -> list[int]:
    """グループインデックスリスト → カラムインデックスリストに展開 (各25列)"""
    cols = []
    for g in sorted(int(g) for g in group_indices if 0 <= int(g) < N_GROUPS):
        cols.extend(FEATURE_GROUPS[g])
    return cols


# ─── データ読み込み ───────────────────────────────────────────────────────────
def load_data(csv_path: str, timeframe: str = 'H1') -> pd.DataFrame:
    """MetaTrader CSV → 指定タイムフレームにリサンプル"""
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


# ─── ユーティリティ ────────────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


# ─── インジケータ計算 ──────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, lo, o = df['close'], df['high'], df['low'], df['open']
    vol = df['volume']

    # ── EMA ───────────────────────────────────────────────────────────────
    ema8   = _ema(c, 8);   ema21  = _ema(c, 21)
    ema55  = _ema(c, 55);  ema200 = _ema(c, 200)
    df['ema8']  = ema8;  df['ema21']  = ema21
    df['ema55'] = ema55; df['ema200'] = ema200

    # ── ATR ───────────────────────────────────────────────────────────────
    atr14 = _atr(df, 14); atr5 = _atr(df, 5)
    df['atr14'] = atr14;  df['atr5'] = atr5

    # ── ADX / DMI ─────────────────────────────────────────────────────────
    tr_s = pd.concat([h - lo,
                      (h - c.shift(1)).abs(),
                      (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    up_m = h - h.shift(1);  dn_m = lo.shift(1) - lo
    pdm  = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    ndm  = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr14s = _rma(tr_s, 14)
    pdi = 100 * _rma(pd.Series(pdm, index=df.index), 14) / (atr14s + 1e-9)
    ndi = 100 * _rma(pd.Series(ndm, index=df.index), 14) / (atr14s + 1e-9)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    adx = _rma(dx, 14)
    df['adx']     = adx / 100.0
    df['pdi_ndi'] = (pdi - ndi) / 100.0

    # ── RSI ───────────────────────────────────────────────────────────────
    def _rsi(period):
        d = c.diff()
        g = _rma(d.clip(lower=0), period)
        l = _rma((-d).clip(lower=0), period)
        return (100 - 100 / (1 + g / (l + 1e-9))) / 100.0
    rsi14 = _rsi(14); rsi28 = _rsi(28)
    df['rsi14'] = rsi14; df['rsi28'] = rsi28
    df['rsi_slope'] = rsi14.diff(3).fillna(0)

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_line = _ema(c, 12) - _ema(c, 26)
    macd_sig  = _ema(macd_line, 9)
    macd_hist = (macd_line - macd_sig) / (atr14 + 1e-9)
    df['macd_hist']   = macd_hist
    df['macd_signal'] = macd_sig / (atr14 + 1e-9)
    df['macd_slope']  = macd_hist.diff(3).fillna(0)

    # ── Stochastic ────────────────────────────────────────────────────────
    lo14 = lo.rolling(14).min(); hi14 = h.rolling(14).max()
    k  = (c - lo14) / (hi14 - lo14 + 1e-9)
    sk = k.rolling(3).mean(); sd = sk.rolling(3).mean()
    df['stoch_k']      = sk
    df['stoch_d']      = sd
    df['stoch_kd_diff'] = sk - sd

    # ── Williams %R ───────────────────────────────────────────────────────
    df['wr14'] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-9) / 100.0 + 0.5

    # ── ROC ───────────────────────────────────────────────────────────────
    df['roc20'] = c.pct_change(20)

    # ── CCI ───────────────────────────────────────────────────────────────
    tp   = (h + lo + c) / 3
    tp20 = tp.rolling(20).mean()
    mad  = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['cci14'] = (tp - tp20) / (0.015 * mad + 1e-9) / 100.0

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std(ddof=0)
    bb_up  = bb_mid + 2*bb_std;   bb_lo  = bb_mid - 2*bb_std
    df['bb_pos']   = (c - bb_lo) / (bb_up - bb_lo + 1e-9)
    df['bb_width'] = (bb_up - bb_lo) / (bb_mid + 1e-9)

    # ── Keltner Channel ───────────────────────────────────────────────────
    kc_mid = _ema(c, 20)
    kc_up  = kc_mid + 1.5 * atr14; kc_lo = kc_mid - 1.5 * atr14
    df['kc_pos'] = (c - kc_lo) / (kc_up - kc_lo + 1e-9)

    # ── Volatility Squeeze ────────────────────────────────────────────────
    squeeze = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(float)
    df['vol_squeeze']   = squeeze
    df['bb_squeeze_cnt'] = squeeze.rolling(20).sum() / 20.0

    # ── ATR 関連 ──────────────────────────────────────────────────────────
    df['atr_ratio']     = atr14 / (c + 1e-9) * 100.0
    df['atr5_14_ratio'] = atr5 / (atr14 + 1e-9)
    df['hv20']          = c.pct_change().rolling(20).std() * np.sqrt(252 * 24) * 100
    df['atr_pct50'] = atr14.rolling(50).rank(pct=True).fillna(0.5)

    # ── EMA スロープ・整合性 ─────────────────────────────────────────────
    df['ema200_slope']     = ema200.diff(3) / (atr14 * 3 + 1e-9)
    above = (ema8 > ema21).astype(float)
    df['trend_consistency'] = above.rolling(8).mean() * 2 - 1

    # ── EMA 相対位置 ──────────────────────────────────────────────────────
    df['c_ema8']    = (c - ema8)   / (atr14 + 1e-9)
    df['c_ema21']   = (c - ema21)  / (atr14 + 1e-9)
    df['c_ema55']   = (c - ema55)  / (atr14 + 1e-9)
    df['c_ema200']  = (c - ema200) / (atr14 + 1e-9)
    df['ema8_21']   = (ema8  - ema21)  / (atr14 + 1e-9)
    df['ema21_55']  = (ema21 - ema55)  / (atr14 + 1e-9)
    df['ema55_200'] = (ema55 - ema200) / (atr14 + 1e-9)

    # ── 価格アクション ────────────────────────────────────────────────────
    body_raw  = c - o
    range_raw = h - lo + 1e-9
    df['body']    = body_raw  / (atr14 + 1e-9)
    df['upper_w'] = (h - c.clip(lower=o))  / (atr14 + 1e-9)
    df['lower_w'] = (c.clip(upper=o) - lo) / (atr14 + 1e-9)
    df['ret1']    = c.pct_change(1)
    df['ret5']    = c.pct_change(5)
    df['ret20']   = c.pct_change(20)
    df['ret_accel'] = c.pct_change(1).diff(1).fillna(0)

    range_h = h.rolling(24).max(); range_l = lo.rolling(24).min()
    df['close_pct_range'] = (c - range_l) / (range_h - range_l + 1e-9)

    up = (c > c.shift(1)).astype(int); dn = (c < c.shift(1)).astype(int)
    def _streak(s):
        streak = s * 0
        for i in range(1, len(s)):
            streak.iloc[i] = (streak.iloc[i-1] + s.iloc[i]) * s.iloc[i]
        return streak.clip(0, 8) / 8.0
    df['consec_up'] = _streak(up); df['consec_dn'] = _streak(dn)

    # ── ローソク足パターン ────────────────────────────────────────────────
    body_abs = body_raw.abs()
    df['engulf_bull'] = (
        (body_raw > 0) & (body_raw.shift(1) < 0) &
        (o <= c.shift(1)) & (c >= o.shift(1))
    ).astype(float)
    df['engulf_bear'] = (
        (body_raw < 0) & (body_raw.shift(1) > 0) &
        (o >= c.shift(1)) & (c <= o.shift(1))
    ).astype(float)
    df['pin_bull'] = (
        (df['lower_w'] * atr14 > body_abs * 2) &
        (df['upper_w'] * atr14 < body_abs * 0.5)
    ).astype(float)
    df['pin_bear'] = (
        (df['upper_w'] * atr14 > body_abs * 2) &
        (df['lower_w'] * atr14 < body_abs * 0.5)
    ).astype(float)
    df['is_doji'] = (body_abs < range_raw * 0.1).astype(float)

    # ── サポレジ・構造 ────────────────────────────────────────────────────
    don_hi = h.rolling(20).max(); don_lo = lo.rolling(20).min()
    df['donchian_pos']  = (c - don_lo) / (don_hi - don_lo + 1e-9)
    swing_hi = h.rolling(5).max(); swing_lo = lo.rolling(5).min()
    df['swing_hi_dist'] = (swing_hi - c) / (atr14 + 1e-9)
    df['swing_lo_dist'] = (c - swing_lo) / (atr14 + 1e-9)
    round_level = (c / 1.0).round(0) * 1.0
    df['round_dist']    = (c - round_level).abs() / (atr14 + 1e-9)
    ema84 = _ema(c, 84)
    df['h4_trend']      = (c - ema84) / (atr14 + 1e-9)
    day_hi = h.rolling(24, min_periods=1).max()
    day_lo = lo.rolling(24, min_periods=1).min()
    df['daily_range_pos'] = (c - day_lo) / (day_hi - day_lo + 1e-9)
    wk_hi = h.rolling(168, min_periods=20).max()
    wk_lo = lo.rolling(168, min_periods=20).min()
    df['weekly_pos']    = (c - wk_lo) / (wk_hi - wk_lo + 1e-9)
    df['gap_open']      = (o - c.shift(1)) / (atr14 + 1e-9)

    # ── 一目均衡表 ────────────────────────────────────────────────────────
    tk = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kj = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = (tk + kj) / 2
    span_b = (h.rolling(52).max() + lo.rolling(52).min()) / 2
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    df['ichi_tk_diff']     = (tk - kj) / (atr14 + 1e-9)
    df['ichi_cloud_pos']   = (c - (cloud_top + cloud_bot) / 2) / (atr14 + 1e-9)
    df['ichi_cloud_thick'] = (cloud_top - cloud_bot) / (atr14 + 1e-9)

    # ── 出来高 ────────────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20).mean()
    obv = (np.sign(c.diff()) * vol).fillna(0).cumsum()
    df['vol_ratio']    = vol / (vol_ma20 + 1e-9)
    df['obv_slope']    = obv.diff(5) / (vol_ma20 * 5 + 1e-9)
    df['vol_trend']    = (vol / (vol_ma20 + 1e-9)).diff(5).fillna(0)
    vwap = (c * vol).rolling(24).sum() / (vol.rolling(24).sum() + 1e-9)
    df['price_vs_vwap'] = (c - vwap) / (atr14 + 1e-9)

    # ── セッション ────────────────────────────────────────────────────────
    hour = df.index.hour; dow = df.index.dayofweek
    df['hour_sin']  = np.sin(2*np.pi * hour / 24.0)
    df['hour_cos']  = np.cos(2*np.pi * hour / 24.0)
    df['dow_sin']   = np.sin(2*np.pi * dow  / 5.0)
    df['dow_cos']   = np.cos(2*np.pi * dow  / 5.0)
    df['is_tokyo']  = ((hour >= 0)  & (hour <  9)).astype(float)
    df['is_london'] = ((hour >= 7)  & (hour < 16)).astype(float)
    df['is_ny']     = ((hour >= 13) & (hour < 22)).astype(float)
    df['is_overlap']= ((hour >= 13) & (hour < 16)).astype(float)

    # ── 逐次差分拡張 (Sequential Diffs) ─────────────────────────────────
    # col_dk[t] = col[t-(k-1)] - col[t-k]  (k=1: 最新の1本差分, k=24: 24本前)
    # 実装: diff(1) を shift(k-1) することで各時点の「速度」を過去に遡って取得
    diff1 = df[BASE_FEATURE_COLS].diff(1)   # shape: (N, 70)  各時点の1本差分
    for k in range(1, _N_DIFFS + 1):
        shifted = diff1.shift(k - 1)         # k=1: shift(0), k=24: shift(23)
        for col in BASE_FEATURE_COLS:
            df[f'{col}_d{k}'] = shifted[col]

    return df


# ─── ラベル生成 ───────────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame,
                tp_atr: float   = 1.5,
                sl_atr: float   = 1.0,
                forward_bars: int = 20) -> np.ndarray:
    """Triple Barrier ラベル生成 (vectorized): 0=HOLD, 1=BUY, 2=SELL"""
    n     = len(df)
    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    atr   = df['atr14'].values.astype(np.float64)

    valid   = n - forward_bars - 1
    entry   = close[:valid].reshape(-1, 1)
    a       = atr[:valid].reshape(-1, 1)
    tp_dist = tp_atr * a;  sl_dist = sl_atr * a

    tp_l = entry + tp_dist; sl_l = entry - sl_dist
    tp_s = entry - tp_dist; sl_s = entry + sl_dist

    H = np.empty((valid, forward_bars), dtype=np.float64)
    L = np.empty((valid, forward_bars), dtype=np.float64)
    for j in range(forward_bars):
        H[:, j] = high[j+1: j+1+valid]
        L[:, j] = low[j+1:  j+1+valid]

    def _first(mask):
        any_hit = mask.any(axis=1)
        return np.where(any_hit, np.argmax(mask, axis=1), forward_bars)

    ltp = _first(H >= tp_l); lsl = _first(L <= sl_l)
    stp = _first(L <= tp_s); ssl = _first(H >= sl_s)
    long_ok  = ltp < lsl;    short_ok = stp < ssl
    both     = long_ok & short_ok

    lbl = np.where(long_ok  & ~short_ok, 1,
          np.where(short_ok & ~long_ok,  2,
          np.where(both & (ltp <= stp),  1,
          np.where(both & (stp  < ltp),  2, 0))))

    labels = np.zeros(n, dtype=np.int64)
    labels[:valid] = lbl
    return labels


# ─── データセット構築 ──────────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame,
                  seq_len: int      = 20,
                  tp_atr: float     = 1.5,
                  sl_atr: float     = 1.0,
                  forward_bars: int = 20,
                  label_fn=None,
                  feat_indices=None) -> tuple:
    """X [N, seq_len, n_feat], y [N] を返す。
    feat_indices: カラムインデックスのリスト (None=全1750列)
                  ※ expand_groups() で展開済みのカラムインデックスを渡す
    """
    import hashlib

    cache_dir  = Path(__file__).parent.parent / 'label_cache'
    cache_dir.mkdir(exist_ok=True)
    cache_key  = f"tp{tp_atr:.3f}_sl{sl_atr:.3f}_fwd{forward_bars}_n{len(df)}"
    cache_file = cache_dir / f"labels_{hashlib.md5(cache_key.encode()).hexdigest()}.npy"

    if cache_file.exists():
        try:
            labels = np.load(str(cache_file))
        except Exception:
            cache_file.unlink(missing_ok=True)
            labels = make_labels(df, tp_atr=tp_atr, sl_atr=sl_atr,
                                 forward_bars=forward_bars)
            np.save(str(cache_file), labels)
    else:
        labels = make_labels(df, tp_atr=tp_atr, sl_atr=sl_atr,
                             forward_bars=forward_bars)
        try:
            np.save(str(cache_file), labels)
        except Exception:
            pass

    feat = df[FEATURE_COLS].values.astype(np.float32)
    if feat_indices is not None:
        feat = feat[:, feat_indices]
    n_feat = feat.shape[1]

    n           = len(feat)
    valid_start = seq_len
    valid_end   = n - forward_bars - 1
    n_samples   = valid_end - valid_start

    print(f"  サンプル数: {n_samples:,}  特徴量: {n_feat}  "
          f"(HOLD:{(labels[:valid_end]==0).sum():,} "
          f"BUY:{(labels[:valid_end]==1).sum():,} "
          f"SELL:{(labels[:valid_end]==2).sum():,})")

    stride     = feat.strides
    all_shape  = (n - seq_len + 1, seq_len, n_feat)
    all_stride = (stride[0], stride[0], stride[1])
    all_windows = np.lib.stride_tricks.as_strided(feat, shape=all_shape,
                                                   strides=all_stride)
    w_start = valid_start - seq_len
    w_end   = valid_end   - seq_len
    X = all_windows[w_start: w_end].copy()
    y = labels[valid_start - 1: valid_end - 1].copy().astype(np.int64)

    return X, y, labels
