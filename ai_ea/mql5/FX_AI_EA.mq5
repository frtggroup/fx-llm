//+------------------------------------------------------------------+
//|  FX_AI_EA.mq5  ─  AI FX Expert Advisor (ONNX統合版)             |
//|  対応: USDJPY H1  /  MQL5 Build 3370+                           |
//|                                                                  |
//|  使い方:                                                          |
//|    1. fx_model_best.onnx と norm_params_best.json を              |
//|       MQL5/Files/ フォルダにコピー                                |
//|    2. EAをチャートにアタッチ                                       |
//|    3. 入力パラメータでThresholdやRiskPctを調整                     |
//+------------------------------------------------------------------+
#property copyright "FX AI EA"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//──────────────────────────────────────────────────────────────────
// 入力パラメータ
//──────────────────────────────────────────────────────────────────
input group "=== モデルファイル ==="
input string   InpModelFile  = "fx_model.onnx";        // ONNXモデル
input string   InpNormFile   = "norm_params.json";     // 正規化パラメータ

input group "=== エントリー設定 ==="
input double   InpThreshold  = 0.42;   // エントリー確率閾値 (norm_params未記載時のフォールバック)
input double   InpTpAtr      = 1.6;    // TP倍率 (ATR×)
input double   InpSlAtr      = 1.4;    // SL倍率 (ATR×)
input int      InpMaxHoldBars= 10;     // 最大保有バー数

input group "=== リスク管理 ==="
input double   InpRiskPct    = 1.0;    // リスク率 (%) ※LotSize計算に使用
input int      InpMagic      = 20260226;

input group "=== フィルター ==="
input bool     InpTimeFilter = false;  // 時間フィルター
input int      InpStartHour  = 7;      // 開始時間(UTC)
input int      InpEndHour    = 21;     // 終了時間(UTC)

//──────────────────────────────────────────────────────────────────
// グローバル変数
//──────────────────────────────────────────────────────────────────
long     g_model = INVALID_HANDLE;
CTrade   g_trade;

// 正規化パラメータ
double   g_mean[70];
double   g_std[70];
int      g_feat_idx[];   // 使用特徴量インデックス (feat_indices)
int      g_n_feat = 70;  // 選択特徴量数
int      g_seq_len = 20; // シーケンス長

// norm_params.json から読み込む取引パラメータ (未記載時はInpXxxの値を使用)
double   g_threshold  = -1.0;  // -1 = norm_params未記載 → InpThreshold使用
double   g_tp_atr     = -1.0;
double   g_sl_atr     = -1.0;
int      g_hold_bars  = -1;

// インジケータハンドル
int h_ema8, h_ema21, h_ema55, h_ema200;
int h_atr14, h_atr5;
int h_adx;
int h_rsi14, h_rsi28;
int h_macd;
int h_stoch;
int h_wpr;
int h_cci20;
int h_bb;
int h_ichi;

datetime g_last_bar = 0;
int      g_pos_bars = 0;

#define N_ALL 70   // 全特徴量数

//──────────────────────────────────────────────────────────────────
// OnInit
//──────────────────────────────────────────────────────────────────
int OnInit()
{
   // ── インジケータハンドル作成 ─────────────────────────────────
   h_ema8   = iMA(_Symbol, PERIOD_H1,   8, 0, MODE_EMA, PRICE_CLOSE);
   h_ema21  = iMA(_Symbol, PERIOD_H1,  21, 0, MODE_EMA, PRICE_CLOSE);
   h_ema55  = iMA(_Symbol, PERIOD_H1,  55, 0, MODE_EMA, PRICE_CLOSE);
   h_ema200 = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
   h_atr14  = iATR(_Symbol, PERIOD_H1, 14);
   h_atr5   = iATR(_Symbol, PERIOD_H1,  5);
   h_adx    = iADXWilder(_Symbol, PERIOD_H1, 14);
   h_rsi14  = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   h_rsi28  = iRSI(_Symbol, PERIOD_H1, 28, PRICE_CLOSE);
   h_macd   = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
   h_stoch  = iStochastic(_Symbol, PERIOD_H1, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   h_wpr    = iWPR(_Symbol, PERIOD_H1, 14);
   h_cci20  = iCCI(_Symbol, PERIOD_H1, 20, PRICE_TYPICAL);  // Python features.py と同じ20期間
   h_bb     = iBands(_Symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
   h_ichi   = iIchimoku(_Symbol, PERIOD_H1, 9, 26, 52);

   if(h_ema8==INVALID_HANDLE || h_atr14==INVALID_HANDLE ||
      h_adx==INVALID_HANDLE  || h_rsi14==INVALID_HANDLE)
   {
      Print("[EA] インジケータハンドル作成失敗");
      return INIT_FAILED;
   }

   // ── 正規化パラメータ読み込み ─────────────────────────────────
   if(!LoadNormParams(InpNormFile))
   {
      Print("[EA] norm_params 読み込み失敗: ", InpNormFile,
            " → MQL5/Files/ に配置してください");
      return INIT_FAILED;
   }

   // ── ONNX モデル読み込み ──────────────────────────────────────
   // まず MQL5\Files\ を試み、なければ共通フォルダを試みる
   g_model = OnnxCreate(InpModelFile, 0);
   if(g_model == INVALID_HANDLE)
      g_model = OnnxCreate(InpModelFile, ONNX_COMMON_FOLDER);
   if(g_model == INVALID_HANDLE)
   {
      Print("[EA] ONNX モデル読み込み失敗: ", InpModelFile,
            " → MQL5/Files/ に配置してください");
      return INIT_FAILED;
   }

   // 入出力シェイプ設定
   ulong in_shape[]  = {1, (ulong)g_seq_len, (ulong)g_n_feat};
   ulong out_shape[] = {1, 3};
   if(!OnnxSetInputShape(g_model, 0, in_shape) ||
      !OnnxSetOutputShape(g_model, 0, out_shape))
   {
      Print("[EA] ONNX シェイプ設定失敗  seq=", g_seq_len, " feat=", g_n_feat);
      OnnxRelease(g_model);
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(20);

   Print("[EA] 初期化完了  seq=", g_seq_len, " feat=", g_n_feat,
         " threshold=", InpThreshold);
   return INIT_SUCCEEDED;
}

//──────────────────────────────────────────────────────────────────
// OnDeinit
//──────────────────────────────────────────────────────────────────
void OnDeinit(const int reason)
{
   if(g_model != INVALID_HANDLE) OnnxRelease(g_model);
   IndicatorRelease(h_ema8);   IndicatorRelease(h_ema21);
   IndicatorRelease(h_ema55);  IndicatorRelease(h_ema200);
   IndicatorRelease(h_atr14);  IndicatorRelease(h_atr5);
   IndicatorRelease(h_adx);
   IndicatorRelease(h_rsi14);  IndicatorRelease(h_rsi28);
   IndicatorRelease(h_macd);   IndicatorRelease(h_stoch);
   IndicatorRelease(h_wpr);    IndicatorRelease(h_cci20);
   IndicatorRelease(h_bb);     IndicatorRelease(h_ichi);
}

//──────────────────────────────────────────────────────────────────
// OnTick
//──────────────────────────────────────────────────────────────────
void OnTick()
{
   // バー確認 (H1 新バーのみ処理)
   datetime cur_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(cur_bar == g_last_bar) return;
   g_last_bar = cur_bar;

   // 時間フィルター
   if(InpTimeFilter)
   {
      MqlDateTime _t; TimeToStruct(TimeCurrent(), _t); int h = _t.hour;
      if(h < InpStartHour || h >= InpEndHour) return;
   }

   // ── ポジション管理 (最大保有バー) ──────────────────────────
   ManagePosition();

   // ── 特徴量計算 + 推論 ──────────────────────────────────────
   float probs[3];
   if(!RunInference(probs)) return;

   float p_hold = probs[0];
   float p_buy  = probs[1];
   float p_sell = probs[2];

   // デバッグ: 100バーに1回確率を出力
   static int dbg_cnt = 0;
   if(++dbg_cnt % 100 == 0)
      Print("[PROB] hold=", DoubleToString(p_hold,3),
            " buy=",  DoubleToString(p_buy,3),
            " sell=", DoubleToString(p_sell,3),
            " max=",  DoubleToString(MathMax(p_buy,p_sell),3));

   // ── ポジション有無を確認 ────────────────────────────────────
   bool has_pos = HasPosition();
   if(has_pos) return;  // 1ポジション管理

   // ── エントリー判断 ─────────────────────────────────────────
   double atr = GetATR14(1);
   if(atr <= 0) return;

   // norm_params.json のパラメータを優先、未記載ならInpXxxを使用
   double use_threshold = (g_threshold > 0) ? g_threshold : InpThreshold;
   double use_tp_atr    = (g_tp_atr    > 0) ? g_tp_atr    : InpTpAtr;
   double use_sl_atr    = (g_sl_atr    > 0) ? g_sl_atr    : InpSlAtr;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double lot = LotSize();

   if(p_buy > use_threshold && p_buy > p_sell && p_buy > p_hold)
   {
      double sl = ask - use_sl_atr * atr;
      double tp = ask + use_tp_atr * atr;
      sl = NormalizeDouble(sl, _Digits);
      tp = NormalizeDouble(tp, _Digits);
      if(g_trade.Buy(lot, _Symbol, ask, sl, tp,
                     StringFormat("AI BUY p=%.3f", p_buy)))
      {
         g_pos_bars = 0;
         Print("[EA] BUY  lot=", lot, " sl=", sl, " tp=", tp,
               " p_buy=", p_buy);
      }
   }
   else if(p_sell > use_threshold && p_sell > p_buy && p_sell > p_hold)
   {
      double sl = bid + use_sl_atr * atr;
      double tp = bid - use_tp_atr * atr;
      sl = NormalizeDouble(sl, _Digits);
      tp = NormalizeDouble(tp, _Digits);
      if(g_trade.Sell(lot, _Symbol, bid, sl, tp,
                      StringFormat("AI SELL p=%.3f", p_sell)))
      {
         g_pos_bars = 0;
         Print("[EA] SELL lot=", lot, " sl=", sl, " tp=", tp,
               " p_sell=", p_sell);
      }
   }
}

//──────────────────────────────────────────────────────────────────
// ポジション管理 (最大保有バー超過で決済)
//──────────────────────────────────────────────────────────────────
void ManagePosition()
{
   if(!HasPosition()) { g_pos_bars = 0; return; }
   g_pos_bars++;
   int use_hold = (g_hold_bars > 0) ? g_hold_bars : InpMaxHoldBars;
   if(g_pos_bars >= use_hold)
   {
      for(int i = PositionsTotal()-1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) &&
            PositionGetInteger(POSITION_MAGIC) == InpMagic)
         {
            g_trade.PositionClose(ticket);
            Print("[EA] 最大保有バー超過 → 決済");
         }
      }
      g_pos_bars = 0;
   }
}

bool HasPosition()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong tk = PositionGetTicket(i);
      if(PositionSelectByTicket(tk) &&
         PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   }
   return false;
}

//──────────────────────────────────────────────────────────────────
// ONNX 推論
//──────────────────────────────────────────────────────────────────
bool RunInference(float &probs[])
{
   int need = MathMax(g_seq_len + 250, 300);  // インジケータウォームアップ分 (最低300本)

   // ── インジケータバッファ取得 ────────────────────────────────
   double ema8_buf[],  ema21_buf[], ema55_buf[], ema200_buf[];
   double atr14_buf[], atr5_buf[];
   double adx_buf[],   pdi_buf[],   ndi_buf[];
   double rsi14_buf[], rsi28_buf[];
   double macd_hist[], macd_sig[];
   double sk_buf[],    sd_buf[];
   double wpr_buf[];
   double cci_buf[];
   double bb_up[],     bb_mid[],    bb_lo[];
   double ichi_tk[],   ichi_kj[],   ichi_spa[], ichi_spb[];
   double close_buf[], high_buf[],  low_buf[],  open_buf[], vol_buf[];
   long   vol_buf_tick[];

   if(CopyBuffer(h_ema8,   0, 1, need, ema8_buf)  < need) return false;
   if(CopyBuffer(h_ema21,  0, 1, need, ema21_buf) < need) return false;
   if(CopyBuffer(h_ema55,  0, 1, need, ema55_buf) < need) return false;
   if(CopyBuffer(h_ema200, 0, 1, need, ema200_buf)< need) return false;
   if(CopyBuffer(h_atr14,  0, 1, need, atr14_buf) < need) return false;
   if(CopyBuffer(h_atr5,   0, 1, need, atr5_buf)  < need) return false;
   if(CopyBuffer(h_adx, 0, 1, need, adx_buf) < need) return false;  // ADX
   if(CopyBuffer(h_adx, 1, 1, need, pdi_buf) < need) return false;  // +DI
   if(CopyBuffer(h_adx, 2, 1, need, ndi_buf) < need) return false;  // -DI
   if(CopyBuffer(h_rsi14,  0, 1, need, rsi14_buf) < need) return false;
   if(CopyBuffer(h_rsi28,  0, 1, need, rsi28_buf) < need) return false;
   if(CopyBuffer(h_macd, 0, 1, need, macd_hist) < need) return false; // MACD hist
   if(CopyBuffer(h_macd, 1, 1, need, macd_sig)  < need) return false; // Signal
   if(CopyBuffer(h_stoch, 0, 1, need, sk_buf) < need) return false;
   if(CopyBuffer(h_stoch, 1, 1, need, sd_buf) < need) return false;
   if(CopyBuffer(h_wpr,   0, 1, need, wpr_buf)   < need) return false;
   if(CopyBuffer(h_cci20, 0, 1, need, cci_buf)   < need) return false;
   if(CopyBuffer(h_bb, 1, 1, need, bb_up)  < need) return false;
   if(CopyBuffer(h_bb, 0, 1, need, bb_mid) < need) return false;
   if(CopyBuffer(h_bb, 2, 1, need, bb_lo)  < need) return false;
   if(CopyBuffer(h_ichi, 0, 1, need, ichi_tk)  < need) return false; // 転換線
   if(CopyBuffer(h_ichi, 1, 1, need, ichi_kj)  < need) return false; // 基準線
   if(CopyBuffer(h_ichi, 2, 1, need, ichi_spa) < need) return false; // スパンA
   if(CopyBuffer(h_ichi, 3, 1, need, ichi_spb) < need) return false; // スパンB
   if(CopyClose(_Symbol, PERIOD_H1, 1, need, close_buf) < need) return false;
   if(CopyHigh (_Symbol, PERIOD_H1, 1, need, high_buf)  < need) return false;
   if(CopyLow  (_Symbol, PERIOD_H1, 1, need, low_buf)   < need) return false;
   if(CopyOpen (_Symbol, PERIOD_H1, 1, need, open_buf)  < need) return false;
   if(CopyTickVolume(_Symbol, PERIOD_H1, 1, need, vol_buf_tick) < need) return false;
   // TickVolumeをdoubleに変換
   ArrayResize(vol_buf, need);
   for(int i=0; i<need; i++) vol_buf[i] = (double)vol_buf_tick[i];

   // ── 全70特徴量を seq_len バー分計算 ──────────────────────────
   // 配列インデックス規則: [0]=最新バー (1本前), [need-1]=最古バー
   // seq_len 本を [0]...[seq_len-1] で使用

   // 入力テンソル [1, seq_len, g_n_feat]
   float input_data[];
   ArrayResize(input_data, g_seq_len * g_n_feat);

   for(int t = 0; t < g_seq_len; t++)
   {
      // t=0が最新, t=seq_len-1が最古 → モデルは時系列順(古→新)
      // 配列は [0]=最新なのでインデックスは t
      int i = t;  // [0]=最新

      double c   = close_buf[i];
      double hh  = high_buf[i];
      double lo  = low_buf[i];
      double op  = open_buf[i];
      double vol = vol_buf[i];
      double atr = atr14_buf[i];
      if(atr < 1e-10) atr = 1e-10;

      double feat[N_ALL];

      // ── [0-3] c_ema8, c_ema21, c_ema55, c_ema200 ───────────
      feat[0]  = (c - ema8_buf[i])   / atr;
      feat[1]  = (c - ema21_buf[i])  / atr;
      feat[2]  = (c - ema55_buf[i])  / atr;
      feat[3]  = (c - ema200_buf[i]) / atr;

      // ── [4-6] ema8_21, ema21_55, ema55_200 ─────────────────
      feat[4]  = (ema8_buf[i]  - ema21_buf[i])  / atr;
      feat[5]  = (ema21_buf[i] - ema55_buf[i])  / atr;
      feat[6]  = (ema55_buf[i] - ema200_buf[i]) / atr;

      // ── [7-8] adx, pdi_ndi ──────────────────────────────────
      feat[7]  = adx_buf[i] / 100.0;
      feat[8]  = (pdi_buf[i] - ndi_buf[i]) / 100.0;

      // ── [9] ema200_slope ────────────────────────────────────
      int i3 = MathMin(i+3, need-1);
      feat[9]  = (ema200_buf[i] - ema200_buf[i3]) / (atr * 3.0 + 1e-9);

      // ── [10] trend_consistency (EMA8>EMA21 の8本中の割合*2-1) ─
      {
         double tc = 0;
         for(int k=0; k<8; k++) tc += (ema8_buf[MathMin(i+k,need-1)] > ema21_buf[MathMin(i+k,need-1)]) ? 1.0 : 0.0;
         feat[10] = tc / 8.0 * 2.0 - 1.0;
      }

      // ── [11-12] rsi14, rsi28 ────────────────────────────────
      feat[11] = rsi14_buf[i] / 100.0;
      feat[12] = rsi28_buf[i] / 100.0;

      // ── [13-14] macd_hist, macd_signal ──────────────────────
      // iMACD buffer[0]=main_line(ema12-ema26), buffer[1]=signal
      // Pythonのmacd_hist = main_line - signal → buffer[0]-buffer[1]
      feat[13] = (macd_hist[i] - macd_sig[i]) / (atr + 1e-9);
      feat[14] = macd_sig[i]                  / (atr + 1e-9);

      // ── [15-17] stoch_k, stoch_d, stoch_kd_diff ─────────────
      feat[15] = sk_buf[i] / 100.0;
      feat[16] = sd_buf[i] / 100.0;
      feat[17] = (sk_buf[i] - sd_buf[i]) / 100.0;

      // ── [18] wr14 (WPR は -100～0 → -0.5～+0.5 に変換) ──────
      feat[18] = wpr_buf[i] / 100.0 + 0.5;

      // ── [19] roc20 ──────────────────────────────────────────
      {
         int i20 = MathMin(i+20, need-1);
         feat[19] = (close_buf[i20] > 0) ? (c / close_buf[i20] - 1.0) : 0.0;
      }

      // ── [20] rsi_slope (RSI 3本変化) ────────────────────────
      feat[20] = (rsi14_buf[i] - rsi14_buf[MathMin(i+3,need-1)]) / 100.0;

      // ── [21] macd_slope (MACD 3本変化) ──────────────────────
      feat[21] = (macd_hist[i] - macd_hist[MathMin(i+3,need-1)]) / (atr + 1e-9);

      // ── BB関連 ─────────────────────────────────────────────
      double bb_rng = bb_up[i] - bb_lo[i];
      if(bb_rng < 1e-10) bb_rng = 1e-10;
      feat[22] = (c - bb_lo[i]) / bb_rng;       // [22] bb_pos
      feat[23] = bb_rng / (bb_mid[i] + 1e-9);   // [23] bb_width

      // ── [24-25] atr_ratio, atr5_14_ratio ────────────────────
      feat[24] = atr / (c + 1e-9) * 100.0;
      feat[25] = atr5_buf[i] / (atr + 1e-9);

      // ── [26] kc_pos (ケルトナーチャネル位置) ────────────────
      {
         double kc_mid = ema20_at(close_buf, i, need);
         double kc_up  = kc_mid + 1.5 * atr;
         double kc_lo  = kc_mid - 1.5 * atr;
         feat[26] = (c - kc_lo) / (kc_up - kc_lo + 1e-9);

         // ── [28] vol_squeeze (BB inside KC) ────────────────
         feat[28] = ((bb_up[i] < kc_up) && (bb_lo[i] > kc_lo)) ? 1.0 : 0.0;
      }

      // ── [27] hv20 (実現ボラ年率) ────────────────────────────
      {
         double sum2 = 0;
         for(int k=1; k<=20; k++){
            int k1 = MathMin(i+k,   need-1);
            int k2 = MathMin(i+k+1, need-1);
            if(close_buf[k2] > 0){
               double r = MathLog(close_buf[k1] / close_buf[k2]);
               sum2 += r*r;
            }
         }
         feat[27] = MathSqrt(sum2/20.0) * MathSqrt(252.0*24.0) * 100.0;
      }

      // ── [29] atr_pct50 (ATR の50本パーセンタイル) ───────────
      {
         double atr_arr[50];
         for(int k=0; k<50; k++) atr_arr[k] = atr14_buf[MathMin(i+k,need-1)];
         double rank = 0;
         for(int k=0; k<50; k++) if(atr_arr[k] <= atr) rank++;
         feat[29] = rank / 50.0;
      }

      // ── [30] bb_squeeze_cnt (20本中スクイーズ割合) ──────────
      {
         double sq_cnt = 0;
         for(int k=0; k<20; k++){
            int ki = MathMin(i+k, need-1);
            double kc_m = ema20_at(close_buf, ki, need);
            double kc_u = kc_m + 1.5 * atr14_buf[ki];
            double kc_l = kc_m - 1.5 * atr14_buf[ki];
            if(bb_up[ki] < kc_u && bb_lo[ki] > kc_l) sq_cnt++;
         }
         feat[30] = sq_cnt / 20.0;
      }

      // ── 価格アクション ──────────────────────────────────────
      double body_raw  = c - op;
      double range_raw = hh - lo + 1e-9;
      double body_abs  = MathAbs(body_raw);
      double upper_w   = (hh - MathMax(c, op)) / (atr + 1e-9);
      double lower_w   = (MathMin(c, op) - lo) / (atr + 1e-9);

      feat[31] = body_raw / (atr + 1e-9);      // [31] body
      feat[32] = upper_w;                       // [32] upper_w
      feat[33] = lower_w;                       // [33] lower_w

      // [34] ret1, [35] ret5, [36] ret20
      feat[34] = (close_buf[MathMin(i+1,need-1)] > 0) ? (c / close_buf[MathMin(i+1,need-1)] - 1.0) : 0.0;
      feat[35] = (close_buf[MathMin(i+5,need-1)] > 0) ? (c / close_buf[MathMin(i+5,need-1)] - 1.0) : 0.0;
      feat[36] = (close_buf[MathMin(i+20,need-1)]> 0) ? (c / close_buf[MathMin(i+20,need-1)]- 1.0) : 0.0;

      // [37] close_pct_range (24本レンジ内位置)
      {
         double rhi = high_buf[i],  rlo = low_buf[i];
         for(int k=0; k<24; k++){
            int ki = MathMin(i+k, need-1);
            rhi = MathMax(rhi, high_buf[ki]);
            rlo = MathMin(rlo, low_buf[ki]);
         }
         feat[37] = (c - rlo) / (rhi - rlo + 1e-9);
      }

      // [38-39] consec_up, consec_dn (連続上昇・下降 / 8)
      {
         double cu = 0, cd = 0;
         for(int k=0; k<8; k++){
            int k1 = MathMin(i+k,   need-1);
            int k2 = MathMin(i+k+1, need-1);
            if(close_buf[k1] > close_buf[k2]) cu++;
            else break;
         }
         for(int k=0; k<8; k++){
            int k1 = MathMin(i+k,   need-1);
            int k2 = MathMin(i+k+1, need-1);
            if(close_buf[k1] < close_buf[k2]) cd++;
            else break;
         }
         feat[38] = cu / 8.0;
         feat[39] = cd / 8.0;
      }

      // [40] ret_accel
      {
         double r1  = (close_buf[MathMin(i+1,need-1)] > 0) ? (c / close_buf[MathMin(i+1,need-1)] - 1.0) : 0.0;
         double r1p = (close_buf[MathMin(i+2,need-1)] > 0) ? (close_buf[MathMin(i+1,need-1)] / close_buf[MathMin(i+2,need-1)] - 1.0) : 0.0;
         feat[40] = r1 - r1p;
      }

      // ローソク足パターン
      int ip1 = MathMin(i+1, need-1);
      double prev_c = close_buf[ip1]; double prev_o = open_buf[ip1];
      double prev_body = prev_c - prev_o;

      // [41] engulf_bull
      feat[41] = (body_raw > 0 && prev_body < 0 && op <= prev_c && c >= prev_o) ? 1.0 : 0.0;
      // [42] engulf_bear
      feat[42] = (body_raw < 0 && prev_body > 0 && op >= prev_c && c <= prev_o) ? 1.0 : 0.0;
      // [43] pin_bull (下ひげ長い)
      feat[43] = (lower_w * atr > body_abs * 2 && upper_w * atr < body_abs * 0.5) ? 1.0 : 0.0;
      // [44] pin_bear (上ひげ長い)
      feat[44] = (upper_w * atr > body_abs * 2 && lower_w * atr < body_abs * 0.5) ? 1.0 : 0.0;
      // [45] is_doji
      feat[45] = (body_abs < range_raw * 0.1) ? 1.0 : 0.0;

      // ── サポレジ・構造 ──────────────────────────────────────
      // [46] donchian_pos (20本)
      {
         double dhi = high_buf[i], dlo = low_buf[i];
         for(int k=0; k<20; k++){
            int ki = MathMin(i+k, need-1);
            dhi = MathMax(dhi, high_buf[ki]);
            dlo = MathMin(dlo, low_buf[ki]);
         }
         feat[46] = (c - dlo) / (dhi - dlo + 1e-9);
      }
      // [47] swing_hi_dist, [48] swing_lo_dist (5本スウィング)
      {
         double shi = high_buf[i], slo = low_buf[i];
         for(int k=0; k<5; k++){
            int ki = MathMin(i+k, need-1);
            shi = MathMax(shi, high_buf[ki]);
            slo = MathMin(slo, low_buf[ki]);
         }
         feat[47] = (shi - c) / (atr + 1e-9);
         feat[48] = (c - slo) / (atr + 1e-9);
      }
      // [49] round_dist (丸数字からの距離)
      {
         double rounded = MathRound(c);
         feat[49] = MathAbs(c - rounded) / (atr + 1e-9);
      }
      // [50] h4_trend (EMA84近似)
      {
         double e84 = ema84_at(close_buf, i, need);
         feat[50] = (c - e84) / (atr + 1e-9);
      }
      // [51] daily_range_pos (24本)
      {
         double dhi = high_buf[i], dlo = low_buf[i];
         for(int k=0; k<24; k++){
            int ki = MathMin(i+k, need-1);
            dhi = MathMax(dhi, high_buf[ki]);
            dlo = MathMin(dlo, low_buf[ki]);
         }
         feat[51] = (c - dlo) / (dhi - dlo + 1e-9);
      }
      // [52] weekly_pos (168本)
      {
         double whi = high_buf[i], wlo = low_buf[i];
         int wn = MathMin(168, need-i);
         for(int k=0; k<wn; k++){
            int ki = i+k;
            whi = MathMax(whi, high_buf[ki]);
            wlo = MathMin(wlo, low_buf[ki]);
         }
         feat[52] = (c - wlo) / (whi - wlo + 1e-9);
      }
      // [53] gap_open
      feat[53] = (op - close_buf[MathMin(i+1,need-1)]) / (atr + 1e-9);

      // ── 一目均衡表 ──────────────────────────────────────────
      // MT5のiIchimoku buffer[2/3](スパンA/B)は26本先行シフト済みのためPythonと不一致
      // → Pythonと同じ「シフトなし」の現在値をtk/kjおよびhigh/lowから直接計算
      double span_a_cur = (ichi_tk[i] + ichi_kj[i]) / 2.0;
      {
         // span_b_cur: 52本high/lowの中値 (Python: h.rolling(52).max/min / 2)
         double hi52 = high_buf[i], lo52 = low_buf[i];
         for(int k = 1; k < 52; k++){
            int ki = MathMin(i + k, need - 1);
            hi52 = MathMax(hi52, high_buf[ki]);
            lo52 = MathMin(lo52, low_buf[ki]);
         }
         double span_b_cur = (hi52 + lo52) / 2.0;
         double cloud_top = MathMax(span_a_cur, span_b_cur);
         double cloud_bot = MathMin(span_a_cur, span_b_cur);
         feat[54] = (ichi_tk[i] - ichi_kj[i])               / (atr + 1e-9); // ichi_tk_diff
         feat[55] = (c - (cloud_top + cloud_bot) / 2.0)      / (atr + 1e-9); // ichi_cloud_pos
         feat[56] = (cloud_top - cloud_bot)                  / (atr + 1e-9); // ichi_cloud_thick
      }

      // ── 出来高 ─────────────────────────────────────────────
      double vol_ma20 = 0;
      for(int k=0; k<20; k++) vol_ma20 += vol_buf[MathMin(i+k,need-1)];
      vol_ma20 /= 20.0;
      if(vol_ma20 < 1) vol_ma20 = 1;

      feat[57] = vol / vol_ma20;  // vol_ratio

      // [58] obv_slope (5本OBV変化)
      {
         double obv = 0;
         for(int k=0; k<5; k++){
            int k1 = MathMin(i+k,   need-1);
            int k2 = MathMin(i+k+1, need-1);
            obv += (close_buf[k1] >= close_buf[k2] ? 1.0 : -1.0) * vol_buf[k1];
         }
         feat[58] = obv / (vol_ma20 * 5.0 + 1e-9);
      }
      // [59] vol_trend (vol ratio 5本変化)
      {
         double vr0 = vol / vol_ma20;
         double vm5 = 0;
         for(int k=5; k<10; k++) vm5 += vol_buf[MathMin(i+k,need-1)];
         vm5 /= 5.0;
         double vol_ma20_5 = 0;
         for(int k=5; k<25; k++) vol_ma20_5 += vol_buf[MathMin(i+k,need-1)];
         vol_ma20_5 /= 20.0;
         if(vol_ma20_5 < 1) vol_ma20_5 = 1;
         feat[59] = vr0 - vm5 / vol_ma20_5;
      }
      // [60] price_vs_vwap (24本VWAP近似)
      {
         double vwap_num = 0, vwap_den = 0;
         for(int k=0; k<24; k++){
            int ki = MathMin(i+k, need-1);
            vwap_num += close_buf[ki] * vol_buf[ki];
            vwap_den += vol_buf[ki];
         }
         double vwap = (vwap_den > 0) ? vwap_num / vwap_den : c;
         feat[60] = (c - vwap) / (atr + 1e-9);
      }
      // [61] cci14
      feat[61] = cci_buf[i] / 100.0;

      // ── セッション ──────────────────────────────────────────
      datetime dt = iTime(_Symbol, PERIOD_H1, i+1); // 1本前バーの時刻
      MqlDateTime mdt;
      TimeToStruct(dt, mdt);
      int hr  = mdt.hour;
      int dow = mdt.day_of_week;

      feat[62] = MathSin(2.0 * M_PI * hr  / 24.0);  // hour_sin
      feat[63] = MathCos(2.0 * M_PI * hr  / 24.0);  // hour_cos
      feat[64] = MathSin(2.0 * M_PI * dow /  5.0);  // dow_sin
      feat[65] = MathCos(2.0 * M_PI * dow /  5.0);  // dow_cos
      feat[66] = (hr >=  0 && hr <  9) ? 1.0 : 0.0; // is_tokyo
      feat[67] = (hr >=  7 && hr < 16) ? 1.0 : 0.0; // is_london
      feat[68] = (hr >= 13 && hr < 22) ? 1.0 : 0.0; // is_ny
      feat[69] = (hr >= 13 && hr < 16) ? 1.0 : 0.0; // is_overlap

      // ── 特徴量選択 → 入力テンソルに書き込み ─────────────────
      // 正規化はモデル内部 (FXPredictorWithNorm) で実施済み → EA側では生値を渡す
      // テンソル順序: [seq_len-1-t] (古→新の時系列順)
      int row = g_seq_len - 1 - t;
      for(int f = 0; f < g_n_feat; f++)
      {
         int fi = g_feat_idx[f];  // 70次元中のインデックス
         input_data[row * g_n_feat + f] = (float)feat[fi];
      }
   }

   // ── ONNX 推論 ───────────────────────────────────────────────
   float output_data[3];
   if(!OnnxRun(g_model, ONNX_DEFAULT, input_data, output_data))
   {
      Print("[EA] OnnxRun 失敗");
      return false;
   }

   // モデル出力は既にSoftmax確率 → そのままコピー
   probs[0] = output_data[0];  // P(HOLD)
   probs[1] = output_data[1];  // P(BUY)
   probs[2] = output_data[2];  // P(SELL)

   return true;
}

//──────────────────────────────────────────────────────────────────
// EMA20 (インデックスi から計算) ─ ケルトナー用
//──────────────────────────────────────────────────────────────────
double ema20_at(const double &src[], int i, int total)
{
   double alpha = 2.0 / 21.0;
   double ema   = src[MathMin(i+19, total-1)];
   for(int k = MathMin(i+19, total-1); k >= i; k--)
      ema = alpha * src[k] + (1.0 - alpha) * ema;
   return ema;
}

// EMA84 (H4トレンド近似)
double ema84_at(const double &src[], int i, int total)
{
   double alpha = 2.0 / 85.0;
   double ema   = src[MathMin(i+83, total-1)];
   for(int k = MathMin(i+83, total-1); k >= i; k--)
      ema = alpha * src[k] + (1.0 - alpha) * ema;
   return ema;
}

//──────────────────────────────────────────────────────────────────
// ATR14 (直近1本目) 取得
//──────────────────────────────────────────────────────────────────
double GetATR14(int shift = 1)
{
   double buf[1];
   if(CopyBuffer(h_atr14, 0, shift, 1, buf) < 1) return 0;
   return buf[0];
}

//──────────────────────────────────────────────────────────────────
// ロットサイズ計算 (MQL5 LotSize 相当)
//──────────────────────────────────────────────────────────────────
double LotSize()
{
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   int magnification;
   string currency = AccountInfoString(ACCOUNT_CURRENCY);
   magnification = (currency == "JPY") ? 10000 : 100;

   double lot = MathCeil(free_margin * InpRiskPct / magnification) / 100.0;
   lot = lot - 0.01;

   if(lot < 0.01) lot = 0.01;
   if(lot > 1.0)  lot = MathCeil(lot);
   if(lot > 20.0) lot = 20.0;

   // ブローカーの最小ロット・ステップに合わせる
   double lot_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lot = MathMax(lot_min, MathRound(lot / lot_step) * lot_step);

   return lot;
}

//──────────────────────────────────────────────────────────────────
// norm_params.json 読み込み
//──────────────────────────────────────────────────────────────────
bool LoadNormParams(string filename)
{
   // まず通常の MQL5\Files\ を試み、なければ共通フォルダ (FILE_COMMON) を試みる
   int fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(fh == INVALID_HANDLE)
   {
      Print("[EA] ファイルを開けません: ", filename);
      return false;
   }

   string content = "";
   while(!FileIsEnding(fh))
      content += FileReadString(fh);
   FileClose(fh);

   // ── JSON から mean 配列を抽出 ──────────────────────────────
   if(!ParseJsonArray(content, "\"mean\"", g_mean, 70)) return false;
   if(!ParseJsonArray(content, "\"std\"",  g_std,  70)) return false;

   // ── seq_len 抽出 ────────────────────────────────────────────
   {
      string key = "\"seq_len\"";
      int pos = StringFind(content, key);
      if(pos >= 0)
      {
         pos += StringLen(key);
         int colon = StringFind(content, ":", pos);
         int next_comma = StringFind(content, ",", colon);
         int next_brace = StringFind(content, "}", colon);
         int end = (next_comma < next_brace && next_comma >= 0) ? next_comma : next_brace;
         string val = StringSubstr(content, colon+1, end - colon - 1);
         StringTrimLeft(val); StringTrimRight(val);
         g_seq_len = (int)StringToInteger(val);
      }
   }

   // ── feat_indices 抽出 ──────────────────────────────────────
   {
      string key = "\"feat_indices\"";
      int pos = StringFind(content, key);
      if(pos >= 0)
      {
         int start = StringFind(content, "[", pos);
         int end   = StringFind(content, "]", start);
         if(start >= 0 && end > start)
         {
            string arr_str = StringSubstr(content, start+1, end - start - 1);
            string parts[];
            int n = StringSplit(arr_str, ',', parts);
            ArrayResize(g_feat_idx, n);
            g_n_feat = n;
            for(int i=0; i<n; i++)
            {
               StringTrimLeft(parts[i]); StringTrimRight(parts[i]);
               g_feat_idx[i] = (int)StringToInteger(parts[i]);
            }
         }
      }
      else
      {
         // feat_indices が null の場合は全70特徴量を使用
         ArrayResize(g_feat_idx, 70);
         g_n_feat = 70;
         for(int i=0; i<70; i++) g_feat_idx[i] = i;
      }
   }

   // ── 取引パラメータ読み込み (threshold / tp_atr / sl_atr / hold_bars) ──
   // 記載がある場合のみ上書き。なければ Inp*** のデフォルト値を使用
   g_threshold = ParseJsonDouble(content, "\"threshold\"", g_threshold);
   g_tp_atr    = ParseJsonDouble(content, "\"tp_atr\"",    g_tp_atr);
   g_sl_atr    = ParseJsonDouble(content, "\"sl_atr\"",    g_sl_atr);
   {
      double hb = ParseJsonDouble(content, "\"hold_bars\"", -1.0);
      if(hb > 0) g_hold_bars = (int)hb;
   }

   string thr_src  = (g_threshold > 0) ? "norm_params" : "InpThreshold";
   string tp_src   = (g_tp_atr    > 0) ? "norm_params" : "InpTpAtr";
   double thr_use  = (g_threshold > 0) ? g_threshold : InpThreshold;
   double tp_use   = (g_tp_atr    > 0) ? g_tp_atr    : InpTpAtr;
   double sl_use   = (g_sl_atr    > 0) ? g_sl_atr    : InpSlAtr;
   int    hb_use   = (g_hold_bars > 0) ? g_hold_bars  : InpMaxHoldBars;

   Print("[EA] norm_params読込完了  feat=", g_n_feat, " seq=", g_seq_len,
         "  threshold=", thr_use, "(", thr_src, ")",
         "  tp=", tp_use, "  sl=", sl_use, "  hold=", hb_use);
   return true;
}

//──────────────────────────────────────────────────────────────────
// JSON 配列パーサ (単純な数値配列を抽出)
//──────────────────────────────────────────────────────────────────
bool ParseJsonArray(string &content, string key, double &out[], int max_size)
{
   int pos = StringFind(content, key);
   if(pos < 0) { Print("[EA] JSON キー未発見: ", key); return false; }

   int start = StringFind(content, "[", pos);
   int end   = StringFind(content, "]", start);
   if(start < 0 || end <= start) { Print("[EA] JSON 配列パース失敗: ", key); return false; }

   string arr_str = StringSubstr(content, start+1, end - start - 1);
   string parts[];
   int n = StringSplit(arr_str, ',', parts);
   n = MathMin(n, max_size);
   ArrayResize(out, n);
   for(int i=0; i<n; i++)
   {
      StringTrimLeft(parts[i]); StringTrimRight(parts[i]);
      out[i] = StringToDouble(parts[i]);
   }
   return true;
}

//──────────────────────────────────────────────────────────────────
// JSON スカラー値パーサ (数値のみ対応)
//──────────────────────────────────────────────────────────────────
double ParseJsonDouble(const string &content, string key, double default_val)
{
   int pos = StringFind(content, key);
   if(pos < 0) return default_val;
   int colon = StringFind(content, ":", pos);
   if(colon < 0) return default_val;
   int nc  = StringFind(content, ",", colon);
   int nb  = StringFind(content, "}", colon);
   int end = (nc >= 0 && (nb < 0 || nc < nb)) ? nc : nb;
   if(end < 0) return default_val;
   string val = StringSubstr(content, colon + 1, end - colon - 1);
   StringTrimLeft(val);
   StringTrimRight(val);
   if(StringLen(val) == 0) return default_val;
   return StringToDouble(val);
}
