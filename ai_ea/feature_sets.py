"""
FX AI EA - AIが設計した特徴量セット 100種
各セットはトレーダーの視点・戦略に基づいて厳選。
ランダムサーチ時に FEATURE_SETS からランダムに選択される。

全70特徴量 (インデックス):
トレンド(0-10):  c_ema8,c_ema21,c_ema55,c_ema200,ema8_21,ema21_55,ema55_200,adx,pdi_ndi,ema200_slope,trend_consistency
モメンタム(11-21): rsi14,rsi28,macd_hist,macd_signal,stoch_k,stoch_d,stoch_kd_diff,wr14,roc20,rsi_slope,macd_slope
ボラティリティ(22-30): bb_pos,bb_width,atr_ratio,atr5_14_ratio,kc_pos,hv20,vol_squeeze,atr_pct50,bb_squeeze_cnt
価格アクション(31-45): body,upper_w,lower_w,ret1,ret5,ret20,close_pct_range,consec_up,consec_dn,ret_accel,engulf_bull,engulf_bear,pin_bull,pin_bear,is_doji
サポレジ(46-53): donchian_pos,swing_hi_dist,swing_lo_dist,round_dist,h4_trend,daily_range_pos,weekly_pos,gap_open
一目(54-56): ichi_tk_diff,ichi_cloud_pos,ichi_cloud_thick
出来高(57-61): vol_ratio,obv_slope,vol_trend,price_vs_vwap,cci14
セッション(62-69): hour_sin,hour_cos,dow_sin,dow_cos,is_tokyo,is_london,is_ny,is_overlap
"""

# ── インデックス定義 ────────────────────────────────────────────────────────
TREND       = list(range(0, 11))    # トレンド
MOMENTUM    = list(range(11, 22))   # モメンタム
VOLATILITY  = list(range(22, 31))   # ボラティリティ
PRICE_ACT   = list(range(31, 46))   # 価格アクション・ローソク足
STRUCTURE   = list(range(46, 54))   # サポレジ・構造
ICHIMOKU    = list(range(54, 57))   # 一目均衡表
VOLUME      = list(range(57, 62))   # 出来高
SESSION     = list(range(62, 70))   # セッション

ALL_FEAT    = list(range(70))       # 全70


def _s(*groups):
    """複数グループをマージして sorted リストを返す"""
    out = set()
    for g in groups:
        out.update(g)
    return sorted(out)


FEATURE_SETS = [
    # ════════════════════════════════════════════════════════════════
    # [1-10] トレンドフォロー系
    # ════════════════════════════════════════════════════════════════
    # 1: 純粋トレンド (EMA配列 + ADX)
    _s(TREND),
    # 2: トレンド + モメンタム確認
    _s(TREND, MOMENTUM),
    # 3: トレンド + ボラ (ATR でトレンド強度補正)
    _s(TREND, VOLATILITY),
    # 4: トレンド + 一目均衡表
    _s(TREND, ICHIMOKU),
    # 5: EMAクロスのみ (最シンプル)
    [4, 5, 6, 7, 8, 9],  # ema8_21, ema21_55, ema55_200, adx, pdi_ndi, ema200_slope
    # 6: トレンド + セッション (時間帯トレンド)
    _s(TREND, SESSION),
    # 7: トレンド + サポレジ
    _s(TREND, STRUCTURE),
    # 8: EMA + ADX + ATR (トレンド強度 + ボラ)
    [0,1,2,3,4,5,6,7,8,22,23,24,25],
    # 9: 一目フルセット + トレンド
    _s(TREND, ICHIMOKU, STRUCTURE),
    # 10: トレンド + 出来高 (出来高でトレンド確認)
    _s(TREND, VOLUME),

    # ════════════════════════════════════════════════════════════════
    # [11-20] モメンタム・オシレーター系
    # ════════════════════════════════════════════════════════════════
    # 11: 純粋モメンタム
    _s(MOMENTUM),
    # 12: RSI特化 (2本RSI + 傾き)
    [11, 12, 20],  # rsi14, rsi28, rsi_slope
    # 13: MACD特化 (ヒスト + シグナル + 傾き)
    [13, 14, 21],  # macd_hist, macd_signal, macd_slope
    # 14: ストキャス + RSI (過買売)
    [11, 15, 16, 17],  # rsi14, stoch_k, stoch_d, stoch_kd_diff
    # 15: モメンタム + ボラティリティ
    _s(MOMENTUM, VOLATILITY),
    # 16: モメンタム + 価格アクション
    _s(MOMENTUM, PRICE_ACT),
    # 17: モメンタム + セッション
    _s(MOMENTUM, SESSION),
    # 18: RSI + MACD + BB (標準3点セット)
    [11, 13, 14, 20, 21, 22, 23],
    # 19: WilliamsR + CCI + ROC
    [18, 19, 61],  # wr14, roc20, cci14
    # 20: モメンタム + サポレジ
    _s(MOMENTUM, STRUCTURE),

    # ════════════════════════════════════════════════════════════════
    # [21-30] ボラティリティ・レンジ系
    # ════════════════════════════════════════════════════════════════
    # 21: 純粋ボラティリティ
    _s(VOLATILITY),
    # 22: BB特化 (位置 + 幅 + スクイーズ)
    [22, 23, 30],  # bb_pos, bb_width, bb_squeeze_cnt
    # 23: ATR特化
    [24, 25, 29],  # atr_ratio, atr5_14_ratio, atr_pct50
    # 24: ボラ + モメンタム (ブレイクアウト)
    _s(VOLATILITY, MOMENTUM),
    # 25: ボラ + 価格アクション
    _s(VOLATILITY, PRICE_ACT),
    # 26: ボラ + トレンド (トレンド中のボラ変化)
    _s(VOLATILITY, TREND),
    # 27: スクイーズ戦略 (圧縮 → 爆発)
    [22, 23, 29, 30, 7, 8, 13],  # bb+squeeze+adx+macd
    # 28: ボラ + サポレジ
    _s(VOLATILITY, STRUCTURE),
    # 29: ボラ + 出来高
    _s(VOLATILITY, VOLUME),
    # 30: ボラ + セッション
    _s(VOLATILITY, SESSION),

    # ════════════════════════════════════════════════════════════════
    # [31-40] 価格アクション特化系
    # ════════════════════════════════════════════════════════════════
    # 31: ローソク足パターンのみ
    [41, 42, 43, 44, 45],  # engulf+pin+doji
    # 32: 価格アクション + ボラ
    _s(PRICE_ACT, VOLATILITY),
    # 33: 価格アクション + トレンド
    _s(PRICE_ACT, TREND),
    # 34: リターン系のみ
    [34, 35, 36, 37, 40],  # ret1,ret5,ret20,close_pct_range,ret_accel
    # 35: ローソク足形状 (実体・ひげ)
    [31, 32, 33, 34, 38, 39],  # body,upper_w,lower_w,ret1,consec_up,consec_dn
    # 36: 価格アクション + サポレジ
    _s(PRICE_ACT, STRUCTURE),
    # 37: 価格アクション + モメンタム
    _s(PRICE_ACT, MOMENTUM),
    # 38: 価格アクション + 出来高
    _s(PRICE_ACT, VOLUME),
    # 39: ブレイクアウト特化
    [22, 23, 29, 30, 46, 47, 48, 33, 34, 35],  # BB+squeeze+donchian+ret
    # 40: 価格アクション + セッション
    _s(PRICE_ACT, SESSION),

    # ════════════════════════════════════════════════════════════════
    # [41-50] サポレジ・構造系
    # ════════════════════════════════════════════════════════════════
    # 41: 純粋サポレジ
    _s(STRUCTURE),
    # 42: サポレジ + トレンド
    _s(STRUCTURE, TREND),
    # 43: サポレジ + モメンタム
    _s(STRUCTURE, MOMENTUM),
    # 44: サポレジ + 価格アクション
    _s(STRUCTURE, PRICE_ACT),
    # 45: 一目 + サポレジ
    _s(ICHIMOKU, STRUCTURE),
    # 46: サポレジ + セッション
    _s(STRUCTURE, SESSION),
    # 47: サポレジ + 出来高
    _s(STRUCTURE, VOLUME),
    # 48: ドンチャン + スウィング + ラウンド
    [46, 47, 48, 49],
    # 49: H4トレンド + 週次レンジ
    [50, 51, 52, 53],  # h4_trend, daily_range_pos, weekly_pos, gap_open
    # 50: 全構造系 + ボラ
    _s(STRUCTURE, ICHIMOKU, VOLATILITY),

    # ════════════════════════════════════════════════════════════════
    # [51-60] 複合戦略系 (実トレーダー視点)
    # ════════════════════════════════════════════════════════════════
    # 51: スキャルピング (短期モメンタム + 時間帯)
    [11, 13, 14, 15, 16, 17, 22, 31, 32, 33, 62, 63, 64, 65, 66, 67, 68, 69],
    # 52: デイトレード (トレンド + モメンタム + セッション)
    _s(TREND, MOMENTUM, SESSION),
    # 53: スイングトレード (トレンド + 構造 + 一目)
    _s(TREND, STRUCTURE, ICHIMOKU),
    # 54: 逆張り (オシレーター + BB + ローソク)
    [11, 12, 15, 16, 18, 22, 23, 41, 42, 43, 44, 45],
    # 55: ブレイクアウト (ボラ + 構造 + 出来高)
    _s(VOLATILITY, STRUCTURE, VOLUME),
    # 56: トレンドフォロー完全版
    _s(TREND, MOMENTUM, VOLATILITY),
    # 57: マルチタイムフレーム重視
    [0,1,2,3,4,5,6,7,8,9,10,50,51,54,55,56],  # EMA+ADX+H4+Ichimoku
    # 58: 出来高分析特化
    _s(VOLUME, MOMENTUM, TREND),
    # 59: ボラとモメンタムの組合せ (ATR比率でフィルタ)
    _s(VOLATILITY, MOMENTUM, PRICE_ACT),
    # 60: セッション × モメンタム
    _s(SESSION, MOMENTUM, TREND),

    # ════════════════════════════════════════════════════════════════
    # [61-70] 精鋭・絞り込みセット
    # ════════════════════════════════════════════════════════════════
    # 61: 最小有効セット想定 (RSI+MACD+BB+ATR+EMA)
    [4, 5, 7, 11, 13, 22, 23, 24],
    # 62: 有名なシステムトレード指標
    [7, 8, 11, 13, 22, 23, 46, 57, 58],  # ADX+RSI+MACD+BB+Donchian+OBV
    # 63: 転換点検出特化
    [11, 15, 16, 22, 41, 42, 43, 44, 45, 54],  # RSI+Stoch+BB+パターン+一目
    # 64: トレンド確認 + 出来高増加
    [4, 5, 7, 8, 57, 58, 59, 60],  # EMAクロス+ADX+出来高系
    # 65: 価格位置まとめ
    [0, 1, 2, 3, 22, 46, 54, 55, 51, 52],  # EMA比+BB位置+構造位置
    # 66: 短期9特徴量
    [11, 13, 22, 24, 33, 34, 37, 38, 41],
    # 67: 中期15特徴量
    [4, 5, 7, 11, 13, 22, 24, 33, 34, 46, 54, 57, 62, 63, 64],
    # 68: 長期25特徴量
    _s([0,1,2,3,4,5,6,7,8], [11,13,22,24], [33,34,35], [46,54,57,62,63]),
    # 69: 40特徴量バランス型
    _s(TREND, MOMENTUM[:5], VOLATILITY[:5], PRICE_ACT[:8], STRUCTURE, ICHIMOKU),
    # 70: 全特徴量
    ALL_FEAT,

    # ════════════════════════════════════════════════════════════════
    # [71-80] 実験的・変則セット
    # ════════════════════════════════════════════════════════════════
    # 71: セッションのみ (時間帯だけで判断できるか)
    _s(SESSION, MOMENTUM, PRICE_ACT[:5]),
    # 72: 一目完全 + 価格アクション
    _s(ICHIMOKU, PRICE_ACT, TREND[:4]),
    # 73: ローソク足 + 時間帯 (最シンプルな現実)
    _s(PRICE_ACT[4:10], SESSION, [22, 24]),
    # 74: OBV + 価格 + RSI
    [11, 20, 57, 58, 59, 33, 34, 35],
    # 75: VWAP系 (機関投資家の参照点)
    [60, 11, 13, 22, 34, 35, 46, 58],  # VWAP+RSI+MACD+BB+ret1+ret5+Donchian+OBV
    # 76: 加速度・変化率特化
    [19, 20, 21, 36, 39],  # roc20,rsi_slope,macd_slope,close_pct_range,ret_accel
    # 77: 週次・日次構造
    [51, 52, 53, 0, 1, 2, 3, 4, 5, 62, 63, 64, 65],  # daily+weekly+EMA+曜日
    # 78: スクイーズ直前特化 (爆発前の準備検出)
    [22, 23, 24, 25, 26, 27, 28, 29, 30, 7, 8, 57],
    # 79: 強気/弱気パターン特化
    [41, 42, 43, 44, 45, 37, 38, 11, 15, 22],  # engulf+pin+doji+RSI+stoch+BB
    # 80: マクロ + ミクロ (全レンジ)
    _s(TREND[:5], MOMENTUM[:4], VOLATILITY[:4], PRICE_ACT[:4],
       STRUCTURE[:4], ICHIMOKU, VOLUME[:3], SESSION[:4]),

    # ════════════════════════════════════════════════════════════════
    # [81-90] サイズ別バランスセット
    # ════════════════════════════════════════════════════════════════
    # 81: 極小 (5特徴量)
    [4, 11, 22, 33, 62],  # EMAクロス+RSI+BB+ret1+時間
    # 82: 超小 (10特徴量)
    [4, 5, 7, 11, 13, 22, 24, 33, 46, 62],
    # 83: 小 (15特徴量)
    [4, 5, 7, 8, 11, 13, 22, 23, 24, 33, 34, 46, 54, 62, 63],
    # 84: 中小 (20特徴量)
    _s([4,5,7,8,9], [11,13,20], [22,23,24], [33,34,41], [46,54], [62,63]),
    # 85: 中 (30特徴量)
    _s([0,1,4,5,7,8,9,10], [11,13,15,20,21], [22,23,24,29],
       [33,34,35,41,42], [46,50,54], [57,62,63,64]),
    # 86: 中大 (40特徴量)
    _s(TREND, [11,12,13,14,15,20,21], [22,23,24,25,29,30],
       [33,34,35,41,42,43], STRUCTURE[:6], ICHIMOKU),
    # 87: 大 (50特徴量)
    _s(TREND, MOMENTUM, VOLATILITY, PRICE_ACT[:10], STRUCTURE, ICHIMOKU),
    # 88: 超大 (60特徴量) - ボリューム除く
    _s(TREND, MOMENTUM, VOLATILITY, PRICE_ACT, STRUCTURE, ICHIMOKU, SESSION),
    # 89: 全特徴量 - セッション除く
    _s(TREND, MOMENTUM, VOLATILITY, PRICE_ACT, STRUCTURE, ICHIMOKU, VOLUME),
    # 90: 全特徴量 - 価格アクション除く
    _s(TREND, MOMENTUM, VOLATILITY, STRUCTURE, ICHIMOKU, VOLUME, SESSION),

    # ════════════════════════════════════════════════════════════════
    # [91-100] ハイブリッド・特化セット
    # ════════════════════════════════════════════════════════════════
    # 91: トレンド + 一目 + 出来高 (三位一体)
    _s(TREND, ICHIMOKU, VOLUME),
    # 92: オシレーター5種 + 時間帯
    [11, 12, 13, 15, 18, 19, 20, 21, 61, 62, 63, 64, 65],
    # 93: 価格アクション + 構造 + 出来高
    _s(PRICE_ACT, STRUCTURE, VOLUME),
    # 94: 短期反転 (RSI極値 + ローソク + BB)
    [11, 12, 20, 22, 23, 30, 41, 42, 43, 44, 45],
    # 95: トレンド確認 + 逆張りオシレーター
    [4, 5, 7, 8, 9, 11, 12, 15, 16, 22, 23, 29, 46],
    # 96: 全カテゴリ最重要1-2個ずつ
    [4, 7, 11, 13, 22, 24, 33, 41, 46, 54, 57, 62, 63],
    # 97: 月曜・金曜特化 (週初・週末パターン)
    _s(SESSION, [33,34,35,36,37,38,39], MOMENTUM[:5]),
    # 98: ロンドン・NY時間特化
    _s([66, 67, 68, 69], TREND[:5], MOMENTUM[:5], VOLATILITY[:3]),
    # 99: 東京時間特化
    _s([66, 69], TREND[:4], MOMENTUM[:3], STRUCTURE[:4], PRICE_ACT[:5]),
    # 100: 全特徴量 (再掲・出現確率向上用)
    ALL_FEAT,
]

# 重複除去・ソート
FEATURE_SETS = [sorted(set(s)) for s in FEATURE_SETS]

assert len(FEATURE_SETS) == 100, f"セット数={len(FEATURE_SETS)}"
