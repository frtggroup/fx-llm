"""
特徴量エンジニアリング v3 - 70次元 (USDJPY H1/H4)

実際のトレーダーが重視する情報を網羅:
  1. トレンド      : EMA整列・ADX・200EMAスロープ・一目均衡表
  2. モメンタム    : RSI(2期間)・MACD・Stoch・Williams%R・CCI・加速度
  3. ボラティリティ: ATR・BB・Keltner・Squeeze・ATRパーセンタイル
  4. 価格アクション: ローソク形状・ピンバー・エンゲルフィング・十字線
  5. サポレジ構造  : Donchian・スウィング・丸数字・週次レンジ
  6. 出来高        : Vol比率・OBVスロープ・出来高トレンド・VWAP
  7. セッション    : 東京/ロンドン/NY + 曜日・時間 (サイクル符号化)
  8. マルチタイム  : H4近似・日次レンジ位置
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ─── 特徴量リスト (70次元) ────────────────────────────────────────────────
FEATURE_COLS = [
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
    'rsi_slope',    # RSIの変化率 (勢い加速/減速)
    'macd_slope',   # MACDヒストの変化率

    # ── ボラティリティ (9) ───────────────────────────────────────────────
    'bb_pos', 'bb_width',
    'atr_ratio', 'atr5_14_ratio',
    'kc_pos', 'hv20', 'vol_squeeze',
    'atr_pct50',    # ATRの50本パーセンタイル順位 (0-1)
    'bb_squeeze_cnt', # スクイーズ継続本数 (圧縮→爆発前兆)

    # ── 価格アクション・ローソク足パターン (13) ──────────────────────────
    'body', 'upper_w', 'lower_w',
    'ret1', 'ret5', 'ret20',
    'close_pct_range',
    'consec_up', 'consec_dn',
    'ret_accel',    # リターン加速度 (ret1 - prev_ret1)
    'engulf_bull',  # 強気エンゲルフィング
    'engulf_bear',  # 弱気エンゲルフィング
    'pin_bull',     # ハンマー/強気ピンバー
    'pin_bear',     # 逆ハンマー/射撃の星
    'is_doji',      # 十字線 (実体が小さい迷いのローソク)

    # ── サポレジ・構造 (8) ──────────────────────────────────────────────
    'donchian_pos',
    'swing_hi_dist', 'swing_lo_dist',
    'round_dist',
    'h4_trend', 'daily_range_pos',
    'weekly_pos',   # 週次レンジ内位置 (168本)
    'gap_open',     # 始値と前終値のギャップ

    # ── 一目均衡表 (3) ──────────────────────────────────────────────────
    'ichi_tk_diff',     # 転換線 - 基準線 (方向)
    'ichi_cloud_pos',   # 価格 vs 雲の中点
    'ichi_cloud_thick', # 雲の厚み

    # ── 出来高 (5) ──────────────────────────────────────────────────────
    'vol_ratio',    # 出来高 vs 20MA
    'obv_slope',    # OBVスロープ
    'vol_trend',    # 出来高5本トレンド
    'price_vs_vwap', # 価格 vs VWAP近似
    'cci14',        # CCI14 (平均乖離)

    # ── セッション (8) ──────────────────────────────────────────────────
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'is_tokyo', 'is_london', 'is_ny', 'is_overlap',
]
N_FEATURES = len(FEATURE_COLS)  # 70

assert N_FEATURES == 70, f"N_FEATURES={N_FEATURES} (期待値70)"


# ─── データ読み込み ───────────────────────────────────────────────────────
def load_data(csv_path: str, timeframe: str = 'H1') -> pd.DataFrame:
    """MetaTrader CSV → 指定タイムフレームにリサンプル (H1 CSV も直接読み込み可)

    MT5 から直接エクスポートした H1 CSV (M1 の代わり) を使うと
    バックテストの H1 バーと完全に一致する精度で学習できます。
    H1 CSV を使う場合は timeframe='H1' のまま csv_path に H1 CSV を指定してください。
    ファイル名に 'H1' が含まれていれば自動でリサンプルをスキップします。
    """
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

    # H1 CSV が直接与えられた場合はリサンプルをスキップ
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
        df = df[df.index.dayofweek < 5]  # weekdays only (Mon=0,...,Fri=4)
    elif already_h1:
        # H1 CSV 直接使用: 週末バーのみ除去
        df = df[df.index.dayofweek < 5]
        print(f"  H1 CSV を直接使用 (リサンプルなし)")

    print(f"  {timeframe}: {len(df):,}本  {df.index[0]} ～ {df.index[-1]}")
    return df


# ─── ユーティリティ ────────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing"""
    return s.ewm(alpha=1.0 / period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


# ─── インジケータ計算 ──────────────────────────────────────────────────────
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
    df['rsi_slope'] = rsi14.diff(3).fillna(0)   # 3本RSI変化率

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
    df['cci14'] = (tp - tp20) / (0.015 * mad + 1e-9) / 100.0  # スケール調整

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std(ddof=0)  # MT5 iBands は母標準偏差(ddof=0)
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
    df['bb_squeeze_cnt'] = squeeze.rolling(20).sum() / 20.0  # 20本中の圧縮割合

    # ── ATR 関連 ──────────────────────────────────────────────────────────
    df['atr_ratio']     = atr14 / (c + 1e-9) * 100.0
    df['atr5_14_ratio'] = atr5 / (atr14 + 1e-9)
    df['hv20']          = c.pct_change().rolling(20).std() * np.sqrt(252 * 24) * 100

    # ATR パーセンタイル (50本中の順位)
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
    df['ret_accel'] = c.pct_change(1).diff(1).fillna(0)  # リターン加速度

    range_h = h.rolling(24).max(); range_l = lo.rolling(24).min()
    df['close_pct_range'] = (c - range_l) / (range_h - range_l + 1e-9)

    # 連続上昇/下降
    up = (c > c.shift(1)).astype(int); dn = (c < c.shift(1)).astype(int)
    def _streak(s):
        streak = s * 0
        for i in range(1, len(s)):
            streak.iloc[i] = (streak.iloc[i-1] + s.iloc[i]) * s.iloc[i]
        return streak.clip(0, 8) / 8.0
    df['consec_up'] = _streak(up); df['consec_dn'] = _streak(dn)

    # ── ローソク足パターン ────────────────────────────────────────────────
    body_abs = body_raw.abs()
    # エンゲルフィング: 前本を包む + 逆方向
    df['engulf_bull'] = (
        (body_raw > 0) &                   # 陽線
        (body_raw.shift(1) < 0) &          # 前本陰線
        (o <= c.shift(1)) &                # 安値が前終値以下
        (c >= o.shift(1))                  # 高値が前始値以上
    ).astype(float)
    df['engulf_bear'] = (
        (body_raw < 0) &
        (body_raw.shift(1) > 0) &
        (o >= c.shift(1)) &
        (c <= o.shift(1))
    ).astype(float)
    # ピンバー: ひげが実体の2倍以上
    df['pin_bull'] = (
        (df['lower_w'] * atr14 > body_abs * 2) &   # 下ひげ長い
        (df['upper_w'] * atr14 < body_abs * 0.5)   # 上ひげ短い
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
    # 週次レンジ内位置 (168本 = 7日 × 24時間)
    wk_hi = h.rolling(168, min_periods=20).max()
    wk_lo = lo.rolling(168, min_periods=20).min()
    df['weekly_pos']    = (c - wk_lo) / (wk_hi - wk_lo + 1e-9)
    # 始値ギャップ
    df['gap_open']      = (o - c.shift(1)) / (atr14 + 1e-9)

    # ── 一目均衡表 ────────────────────────────────────────────────────────
    # 転換線 (9本): 最高値+最安値 / 2
    tk = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    # 基準線 (26本)
    kj = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    # 先行スパンA: 転換+基準 / 2
    span_a = (tk + kj) / 2
    # 先行スパンB: 52本
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
    df['vol_trend']    = (vol / (vol_ma20 + 1e-9)).diff(5).fillna(0)  # 出来高加速度
    # VWAP 近似 (直近24本の出来高加重平均)
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

    return df


# ─── ラベル生成 ───────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame,
                tp_atr: float   = 1.5,
                sl_atr: float   = 1.0,
                forward_bars: int = 20) -> np.ndarray:
    """
    Triple Barrier ラベル生成 (vectorized)
    0=HOLD, 1=BUY, 2=SELL
    """
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


# ─── 方向性ラベル (よりシンプル・ノイズに強い) ─────────────────────────────
def make_labels_directional(df: pd.DataFrame,
                             tp_atr: float = 1.5,
                             sl_atr: float = 1.0,
                             forward_bars: int = 20) -> np.ndarray:
    """
    forward_bars 後の終値方向でラベル生成。Triple Barrier より安定。
    tp_atr × ATR 以上の上昇 → BUY(1)
    tp_atr × ATR 以上の下降 → SELL(2)
    それ以外              → HOLD(0)
    """
    n     = len(df)
    close = df['close'].values.astype(np.float64)
    atr   = df['atr14'].values.astype(np.float64)
    valid = n - forward_bars - 1

    labels = np.zeros(n, dtype=np.int64)
    for i in range(valid):
        fut   = close[i + forward_bars]
        th    = tp_atr * atr[i]
        diff  = fut - close[i]
        if diff > th:
            labels[i] = 1
        elif diff < -th:
            labels[i] = 2
    return labels


# ─── データセット構築 ──────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame,
                  seq_len: int      = 20,
                  tp_atr: float     = 1.5,
                  sl_atr: float     = 1.0,
                  forward_bars: int = 20,
                  label_fn=None,
                  feat_indices=None) -> tuple:
    """X [N, seq_len, n_feat], y [N] を返す。
    feat_indices: 使う特徴量のインデックスリスト (None=全次元)

    最適化:
    - ラベルはディスクキャッシュ (tp/sl/forward が同じなら再計算なし)
    - シーケンス構築は numpy stride_tricks でベクトル化 (Pythonループなし)
    """
    import hashlib, pickle

    # ── ラベルキャッシュ (tp/sl/forward が同じなら再計算スキップ) ────────────
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

    # ── 特徴量配列 ────────────────────────────────────────────────────────
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

    # ── ベクトル化シーケンス構築 (stride_tricks, Pythonループなし) ──────────
    # feat の shape: (N, n_feat)
    # 目的: X[i] = feat[valid_start+i-seq_len : valid_start+i]  i=0..n_samples-1
    # → sliding_window_view で (N-seq_len+1, seq_len, n_feat) を作り範囲スライス
    stride     = feat.strides                           # (bytes_per_row, bytes_per_elem)
    all_shape  = (n - seq_len + 1, seq_len, n_feat)
    all_stride = (stride[0], stride[0], stride[1])
    all_windows = np.lib.stride_tricks.as_strided(feat, shape=all_shape,
                                                   strides=all_stride)
    # valid_start - seq_len = 0 (seq_len == valid_start), 通常 seq_len == valid_start
    w_start = valid_start - seq_len   # 0 の場合が多い
    w_end   = valid_end   - seq_len   # = n_samples (w_start==0 のとき)
    X = all_windows[w_start: w_end].copy()   # .copy() で連続メモリに変換

    y = labels[valid_start - 1: valid_end - 1].copy().astype(np.int64)

    return X, y, labels
