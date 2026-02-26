//+------------------------------------------------------------------+
//|  FX_AI_EA.mq5  ─  AI FX Expert Advisor (ONNX統合版)             |
//|  対応: USDJPY H1  /  MQL5 Build 3370+                           |
//|                                                                  |
//|  【インジケータ完全自前計算版】                                    |
//|  MT5組み込みインジケータを一切使わず、Python features.py と        |
//|  完全に同じ計算式 (ewm(adjust=False)) で全指標を自前計算する。     |
//+------------------------------------------------------------------+
#property copyright "FX AI EA"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//──────────────────────────────────────────────────────────────────
// 入力パラメータ
//──────────────────────────────────────────────────────────────────
input group "=== モデルファイル (Common\\Files\\ に置く) ==="
input string   InpModelFile  = "fx_model.onnx";        // ONNX モデル
input string   InpNormFile   = "norm_params.json";     // 正規化パラメータ

input group "=== エントリー設定 (-1 = JSONから自動読込) ==="
input double   InpThreshold  = -1;    // エントリー確率閾値 (-1=JSON)
input double   InpTpAtr      = -1;    // TP倍率 ATR× (-1=JSON)
input double   InpSlAtr      = -1;    // SL倍率 ATR× (-1=JSON)
input int      InpMaxHoldBars= -1;    // 最大保有バー数 (-1=JSON)

input group "=== リスク管理 ==="
input double   InpRiskPct    = 1.0;    // リスク率 (%)
input int      InpMagic      = 20260226;

input group "=== フィルター ==="
input bool     InpTimeFilter = false;  // 時間フィルター
input int      InpStartHour  = 7;      // 開始時間(UTC)
input int      InpEndHour    = 21;     // 終了時間(UTC)

input group "=== 診断 ==="
input bool     InpDebugLog   = false;  // 特徴量CSVログ出力
input int      InpDebugBars  = 200;    // ログ出力バー数

//──────────────────────────────────────────────────────────────────
// グローバル変数
//──────────────────────────────────────────────────────────────────
long     g_model = INVALID_HANDLE;
CTrade   g_trade;

double   g_mean[];
double   g_std[];
int      g_feat_idx[];
int      g_n_feat  = 15;
int      g_seq_len = 10;

double   g_threshold = -1.0;
double   g_tp_atr    = -1.0;
double   g_sl_atr    = -1.0;
int      g_hold_bars = -1;

datetime g_last_bar     = 0;
int      g_pos_bars     = 0;
int      g_debug_handle = INVALID_HANDLE;
int      g_debug_count  = 0;

#define N_ALL  70    // 全特徴量数
#define N_WARM 5000  // EMAウォームアップバー数 (EMA200収束に1000本、余裕をもって5000本)

//──────────────────────────────────────────────────────────────────
// OnInit
//──────────────────────────────────────────────────────────────────
int OnInit()
{
   // norm_params.json 読み込み
   if(!LoadNormParams(InpNormFile))
   {
      Print("[EA] norm_params 読み込み失敗: ", InpNormFile);
      return INIT_FAILED;
   }

   // ONNX モデル読み込み  ─  Common\Files\ のみ使用
   g_model = OnnxCreate(InpModelFile, ONNX_COMMON_FOLDER);
   if(g_model == INVALID_HANDLE)
   {
      Print("[EA] ONNX モデル読み込み失敗: Common\\Files\\", InpModelFile);
      return INIT_FAILED;
   }

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

   if(InpDebugLog)
   {
      g_debug_handle = FileOpen("feat_debug.csv",
                                FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(g_debug_handle != INVALID_HANDLE)
      {
         string hdr = "datetime";
         for(int f = 0; f < g_n_feat; f++)
            hdr += StringFormat(",feat%d(idx%d)", f, g_feat_idx[f]);
         hdr += ",p_hold,p_buy,p_sell";
         FileWriteString(g_debug_handle, hdr + "\n");
         Print("[EA] 診断ログ開始: feat_debug.csv");
      }
   }

   Print("[EA] 初期化完了 (自前計算版)  seq=", g_seq_len, " feat=", g_n_feat);
   return INIT_SUCCEEDED;
}

//──────────────────────────────────────────────────────────────────
// OnDeinit
//──────────────────────────────────────────────────────────────────
void OnDeinit(const int reason)
{
   if(g_model != INVALID_HANDLE) OnnxRelease(g_model);
   if(g_debug_handle != INVALID_HANDLE)
   {
      FileClose(g_debug_handle);
      g_debug_handle = INVALID_HANDLE;
   }
}

//──────────────────────────────────────────────────────────────────
// OnTick
//──────────────────────────────────────────────────────────────────
void OnTick()
{
   datetime cur_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(cur_bar == g_last_bar) return;
   g_last_bar = cur_bar;

   if(InpTimeFilter)
   {
      MqlDateTime _t; TimeToStruct(TimeCurrent(), _t); int h = _t.hour;
      if(h < InpStartHour || h >= InpEndHour) return;
   }

   ManagePosition();

   float probs[3];
   if(!RunInference(probs)) return;

   float p_hold = probs[0], p_buy = probs[1], p_sell = probs[2];

   static int dbg_cnt = 0;
   if(++dbg_cnt % 100 == 0)
      Print("[PROB] hold=", DoubleToString(p_hold,3),
            " buy=",  DoubleToString(p_buy,3),
            " sell=", DoubleToString(p_sell,3));

   if(HasPosition()) return;

   double atr = GetATR14();
   if(atr <= 0) return;

   // 入力値 > 0 なら入力値（最適化用）、-1 なら JSON 値、どちらもなければ安全なデフォルト
   double use_threshold = (InpThreshold > 0) ? InpThreshold : ((g_threshold > 0) ? g_threshold : 0.40);
   double use_tp_atr    = (InpTpAtr    > 0) ? InpTpAtr    : ((g_tp_atr    > 0) ? g_tp_atr    : 2.0);
   double use_sl_atr    = (InpSlAtr    > 0) ? InpSlAtr    : ((g_sl_atr    > 0) ? g_sl_atr    : 1.0);

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double lot = LotSize();

   if(p_buy > use_threshold && p_buy > p_sell && p_buy > p_hold)
   {
      double sl = NormalizeDouble(ask - use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(ask + use_tp_atr * atr, _Digits);
      if(g_trade.Buy(lot, _Symbol, ask, sl, tp,
                     StringFormat("AI BUY p=%.3f", p_buy)))
      {
         g_pos_bars = 0;
         Print("[EA] BUY  lot=", lot, " sl=", sl, " tp=", tp);
      }
   }
   else if(p_sell > use_threshold && p_sell > p_buy && p_sell > p_hold)
   {
      double sl = NormalizeDouble(bid + use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(bid - use_tp_atr * atr, _Digits);
      if(g_trade.Sell(lot, _Symbol, bid, sl, tp,
                      StringFormat("AI SELL p=%.3f", p_sell)))
      {
         g_pos_bars = 0;
         Print("[EA] SELL lot=", lot, " sl=", sl, " tp=", tp);
      }
   }
}

//──────────────────────────────────────────────────────────────────
// ポジション管理
//──────────────────────────────────────────────────────────────────
void ManagePosition()
{
   if(!HasPosition()) { g_pos_bars = 0; return; }
   g_pos_bars++;
   int use_hold = (InpMaxHoldBars > 0) ? InpMaxHoldBars : ((g_hold_bars > 0) ? g_hold_bars : 20);
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

//══════════════════════════════════════════════════════════════════
// 自前インジケータ計算ヘルパー
// 配列規則: [0]=最新バー, [n-1]=最古バー (as-series順)
// 計算方向: 最古(n-1) → 最新(0)  ← Python chronological と同一
//══════════════════════════════════════════════════════════════════

//── EMA: Python ewm(span=N, adjust=False) と完全一致 ────────────
// alpha = 2/(span+1), 初期値 = src[n-1] (最古バーの値)
void _Ema(const double &src[], double &dst[], int n, int span)
{
   double alpha = 2.0 / (span + 1.0);
   double em = src[n-1];
   for(int i = n-1; i >= 0; i--)
   {
      em    = alpha * src[i] + (1.0 - alpha) * em;
      dst[i] = em;
   }
}

//── RMA: Python ewm(alpha=1/period, adjust=False) と完全一致 ────
// Wilder's Smoothing  alpha = 1/period
void _Rma(const double &src[], double &dst[], int n, int period)
{
   double alpha = 1.0 / (double)period;
   double rm = src[n-1];
   for(int i = n-1; i >= 0; i--)
   {
      rm    = alpha * src[i] + (1.0 - alpha) * rm;
      dst[i] = rm;
   }
}

//── True Range ────────────────────────────────────────────────────
// tr[n-1] = h[n-1]-l[n-1]  (最古バー: 前バー終値なし)
// tr[i]   = max(h-l, |h-c_prev|, |l-c_prev|)  c_prev=c[i+1]
void _CalcTR(const double &h[], const double &l[], const double &c[],
             double &tr[], int n)
{
   tr[n-1] = h[n-1] - l[n-1];
   for(int i = n-2; i >= 0; i--)
   {
      double hl  = h[i] - l[i];
      double hpc = MathAbs(h[i] - c[i+1]);
      double lpc = MathAbs(l[i] - c[i+1]);
      tr[i] = MathMax(MathMax(hl, hpc), lpc);
   }
}

//══════════════════════════════════════════════════════════════════
// ONNX 推論 (全インジケータ自前計算)
//══════════════════════════════════════════════════════════════════
bool RunInference(float &probs[])
{
   int n_total = N_WARM + g_seq_len;

   // ── H1 OHLCV 取得 ────────────────────────────────────────────
   MqlRates rates[];
   ArraySetAsSeries(rates, true);  // [0]=最新
   int got = CopyRates(_Symbol, PERIOD_H1, 1, n_total, rates);
   if(got < n_total)
   {
      Print("[EA] CopyRates不足: ", got, "/", n_total);
      return false;
   }

   // ── 配列抽出 ─────────────────────────────────────────────────
   double c[], hh[], ll[], op[], vol[];
   ArrayResize(c,   n_total); ArrayResize(hh, n_total);
   ArrayResize(ll,  n_total); ArrayResize(op, n_total);
   ArrayResize(vol, n_total);
   for(int i = 0; i < n_total; i++)
   {
      c[i]   = rates[i].close;
      hh[i]  = rates[i].high;
      ll[i]  = rates[i].low;
      op[i]  = rates[i].open;
      vol[i] = (double)rates[i].tick_volume;
   }

   // ── インジケータ配列確保 ──────────────────────────────────────
   double tr_a[],  atr14[], atr5[];
   double ema8[],  ema21[], ema55[], ema200[], ema12[], ema26[], ema84[], ema20[];
   double macd_line[], sig9[], macd_h[];
   double pdm[],   ndm[],   pdm_r[], ndm_r[];
   double pdi_a[], ndi_a[], dx_a[], adx_a[];
   double gain14[], loss14[], ag14[], al14[], rsi14_a[];
   double gain28[], loss28[], ag28[], al28[], rsi28_a[];

   ArrayResize(tr_a,    n_total); ArrayResize(atr14,   n_total); ArrayResize(atr5,    n_total);
   ArrayResize(ema8,    n_total); ArrayResize(ema21,   n_total); ArrayResize(ema55,   n_total);
   ArrayResize(ema200,  n_total); ArrayResize(ema12,   n_total); ArrayResize(ema26,   n_total);
   ArrayResize(ema84,   n_total); ArrayResize(ema20,   n_total);
   ArrayResize(macd_line,n_total);ArrayResize(sig9,    n_total);
   ArrayResize(macd_h,  n_total);
   ArrayResize(pdm,     n_total); ArrayResize(ndm,     n_total);
   ArrayResize(pdm_r,   n_total); ArrayResize(ndm_r,   n_total);
   ArrayResize(pdi_a,   n_total); ArrayResize(ndi_a,   n_total);
   ArrayResize(dx_a,    n_total); ArrayResize(adx_a,   n_total);
   ArrayResize(gain14,  n_total); ArrayResize(loss14,  n_total);
   ArrayResize(ag14,    n_total); ArrayResize(al14,    n_total); ArrayResize(rsi14_a, n_total);
   ArrayResize(gain28,  n_total); ArrayResize(loss28,  n_total);
   ArrayResize(ag28,    n_total); ArrayResize(al28,    n_total); ArrayResize(rsi28_a, n_total);

   // ── TR / ATR ─────────────────────────────────────────────────
   _CalcTR(hh, ll, c, tr_a, n_total);
   _Rma(tr_a, atr14, n_total, 14);
   _Rma(tr_a, atr5,  n_total, 5);

   // ── EMA 各種 ─────────────────────────────────────────────────
   _Ema(c, ema8,   n_total, 8);
   _Ema(c, ema21,  n_total, 21);
   _Ema(c, ema55,  n_total, 55);
   _Ema(c, ema200, n_total, 200);
   _Ema(c, ema12,  n_total, 12);
   _Ema(c, ema26,  n_total, 26);
   _Ema(c, ema84,  n_total, 84);
   _Ema(c, ema20,  n_total, 20);

   // ── MACD ─────────────────────────────────────────────────────
   for(int i = 0; i < n_total; i++) macd_line[i] = ema12[i] - ema26[i];
   _Ema(macd_line, sig9, n_total, 9);
   for(int i = 0; i < n_total; i++)
      macd_h[i] = (macd_line[i] - sig9[i]) / (atr14[i] + 1e-9);

   // ── ADX / PDI / NDI ──────────────────────────────────────────
   // up_m = h[i] - h[i+1],  dn_m = l[i+1] - l[i]  (i+1 = 1本前)
   pdm[n_total-1] = 0.0; ndm[n_total-1] = 0.0;
   for(int i = n_total-2; i >= 0; i--)
   {
      double up_m = hh[i] - hh[i+1];
      double dn_m = ll[i+1] - ll[i];
      pdm[i] = (up_m > dn_m && up_m > 0) ? up_m : 0.0;
      ndm[i] = (dn_m > up_m && dn_m > 0) ? dn_m : 0.0;
   }
   _Rma(pdm, pdm_r, n_total, 14);
   _Rma(ndm, ndm_r, n_total, 14);
   for(int i = 0; i < n_total; i++)
   {
      double pa  = atr14[i] + 1e-9;
      pdi_a[i] = 100.0 * pdm_r[i] / pa;
      ndi_a[i] = 100.0 * ndm_r[i] / pa;
      dx_a[i]  = 100.0 * MathAbs(pdi_a[i] - ndi_a[i]) / (pdi_a[i] + ndi_a[i] + 1e-9);
   }
   _Rma(dx_a, adx_a, n_total, 14);

   // ── RSI14 / RSI28 ─────────────────────────────────────────────
   gain14[n_total-1] = 0.0; loss14[n_total-1] = 0.0;
   gain28[n_total-1] = 0.0; loss28[n_total-1] = 0.0;
   for(int i = n_total-2; i >= 0; i--)
   {
      double d    = c[i] - c[i+1];
      gain14[i]  = (d > 0) ? d : 0.0;
      loss14[i]  = (d < 0) ? -d : 0.0;
      gain28[i]  = gain14[i];
      loss28[i]  = loss14[i];
   }
   _Rma(gain14, ag14, n_total, 14); _Rma(loss14, al14, n_total, 14);
   _Rma(gain28, ag28, n_total, 28); _Rma(loss28, al28, n_total, 28);
   for(int i = 0; i < n_total; i++)
   {
      double rs14  = ag14[i] / (al14[i] + 1e-9);
      rsi14_a[i]  = (100.0 - 100.0 / (1.0 + rs14)) / 100.0;
      double rs28  = ag28[i] / (al28[i] + 1e-9);
      rsi28_a[i]  = (100.0 - 100.0 / (1.0 + rs28)) / 100.0;
   }

   // ── 入力テンソル構築 (seq_len バー) ──────────────────────────
   float input_data[];
   ArrayResize(input_data, g_seq_len * g_n_feat);

   for(int t = 0; t < g_seq_len; t++)
   {
      int i = t;  // [0]=最新バー, [seq_len-1]=最古バー
      double cv = c[i], hv = hh[i], lv = ll[i], ov = op[i], vv = vol[i];
      double atr = atr14[i];
      if(atr < 1e-10) atr = 1e-10;

      double feat[N_ALL];
      ArrayInitialize(feat, 0.0);

      // ── [0-3] c_ema8, c_ema21, c_ema55, c_ema200 ────────────
      feat[0] = (cv - ema8[i])   / atr;
      feat[1] = (cv - ema21[i])  / atr;
      feat[2] = (cv - ema55[i])  / atr;
      feat[3] = (cv - ema200[i]) / atr;

      // ── [4-6] ema8_21, ema21_55, ema55_200 ──────────────────
      feat[4] = (ema8[i]  - ema21[i])  / atr;
      feat[5] = (ema21[i] - ema55[i])  / atr;
      feat[6] = (ema55[i] - ema200[i]) / atr;

      // ── [7-8] adx, pdi_ndi ───────────────────────────────────
      feat[7] = adx_a[i] / 100.0;
      feat[8] = (pdi_a[i] - ndi_a[i]) / 100.0;

      // ── [9] ema200_slope ─────────────────────────────────────
      feat[9] = (ema200[i] - ema200[MathMin(i+3,n_total-1)]) / (atr*3.0 + 1e-9);

      // ── [10] trend_consistency ───────────────────────────────
      {
         double tc = 0;
         for(int k = 0; k < 8; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            if(ema8[ki] > ema21[ki]) tc++;
         }
         feat[10] = tc / 8.0 * 2.0 - 1.0;
      }

      // ── [11-12] rsi14, rsi28 ─────────────────────────────────
      feat[11] = rsi14_a[i];
      feat[12] = rsi28_a[i];

      // ── [13-14] macd_hist, macd_signal ───────────────────────
      feat[13] = macd_h[i];
      feat[14] = sig9[i] / (atr + 1e-9);

      // ── [15-17] stoch_k, stoch_d, stoch_kd_diff (14/3/3) ────
      // Python: lo14=lo.rolling(14).min, hi14=h.rolling(14).max
      //         k=(c-lo14)/(hi14-lo14), sk=k.rolling(3).mean, sd=sk.rolling(3).mean
      {
         // fast_k for bars i, i+1, i+2  (for smoothing)
         double fk[5];
         for(int kb = 0; kb < 5; kb++)
         {
            int ib = MathMin(i+kb, n_total-1);
            double lo14 = ll[ib], hi14 = hh[ib];
            for(int j = 0; j < 14; j++)
            {
               int kj = MathMin(ib+j, n_total-1);
               lo14 = MathMin(lo14, ll[kj]);
               hi14 = MathMax(hi14, hh[kj]);
            }
            fk[kb] = (c[ib] - lo14) / (hi14 - lo14 + 1e-9);
         }
         double sk0 = (fk[0]+fk[1]+fk[2]) / 3.0;
         double sk1 = (fk[1]+fk[2]+fk[3]) / 3.0;
         double sk2 = (fk[2]+fk[3]+fk[4]) / 3.0;
         feat[15] = sk0;
         feat[16] = (sk0+sk1+sk2) / 3.0;  // sd
         feat[17] = feat[15] - feat[16];
      }

      // ── [18] wr14 ────────────────────────────────────────────
      {
         double lo14 = lv, hi14 = hv;
         for(int k = 0; k < 14; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            lo14 = MathMin(lo14, ll[ki]);
            hi14 = MathMax(hi14, hh[ki]);
         }
         feat[18] = -(hi14 - cv) / (hi14 - lo14 + 1e-9) + 0.5;
      }

      // ── [19] roc20 ───────────────────────────────────────────
      {
         int i20 = MathMin(i+20, n_total-1);
         feat[19] = (c[i20] > 0) ? (cv / c[i20] - 1.0) : 0.0;
      }

      // ── [20] rsi_slope (3本差) ───────────────────────────────
      feat[20] = rsi14_a[i] - rsi14_a[MathMin(i+3, n_total-1)];

      // ── [21] macd_slope (3本差) ──────────────────────────────
      feat[21] = macd_h[i] - macd_h[MathMin(i+3, n_total-1)];

      // ── [22-23] bb_pos, bb_width (SMA20, 母標準偏差 ddof=0) ──
      {
         double sma = 0;
         for(int k = 0; k < 20; k++) sma += c[MathMin(i+k, n_total-1)];
         sma /= 20.0;
         double var = 0;
         for(int k = 0; k < 20; k++)
         {
            double d = c[MathMin(i+k, n_total-1)] - sma;
            var += d * d;
         }
         double std20 = MathSqrt(var / 20.0);
         double bb_up  = sma + 2.0 * std20;
         double bb_lo  = sma - 2.0 * std20;
         double bb_rng = bb_up - bb_lo + 1e-9;
         feat[22] = (cv - bb_lo) / bb_rng;
         feat[23] = bb_rng / (sma + 1e-9);
      }

      // ── [24-25] atr_ratio, atr5_14_ratio ────────────────────
      feat[24] = atr / (cv + 1e-9) * 100.0;
      feat[25] = atr5[i] / (atr + 1e-9);

      // ── [26] kc_pos (EMA20 + 1.5*ATR) ── Python: _ema(c,20) と同一
      double kc_mid = ema20[i];
      double kc_up  = kc_mid + 1.5 * atr;
      double kc_lo  = kc_mid - 1.5 * atr;
      feat[26] = (cv - kc_lo) / (kc_up - kc_lo + 1e-9);

      // ── [27] hv20 (実現ボラ年率) ── Python: pct_change.rolling(20).std(ddof=1)
      {
         double rets[20]; double sum1 = 0;
         for(int k = 0; k < 20; k++)
         {
            int k1 = MathMin(i+k,   n_total-1);
            int k2 = MathMin(i+k+1, n_total-1);
            rets[k] = (c[k2] > 0) ? (c[k1] - c[k2]) / c[k2] : 0.0;
            sum1 += rets[k];
         }
         double mean = sum1 / 20.0;
         double sum2 = 0;
         for(int k = 0; k < 20; k++) { double d = rets[k] - mean; sum2 += d * d; }
         feat[27] = MathSqrt(sum2 / 19.0) * MathSqrt(252.0 * 24.0) * 100.0;  // ddof=1
      }

      // ── [28] vol_squeeze ─────────────────────────────────────
      {
         double sma = 0;
         for(int k = 0; k < 20; k++) sma += c[MathMin(i+k, n_total-1)];
         sma /= 20.0;
         double var = 0;
         for(int k = 0; k < 20; k++) { double d=c[MathMin(i+k,n_total-1)]-sma; var+=d*d; }
         double std20 = MathSqrt(var/20.0);
         double bb_up2 = sma + 2.0*std20; double bb_lo2 = sma - 2.0*std20;
         feat[28] = (bb_up2 < kc_up && bb_lo2 > kc_lo) ? 1.0 : 0.0;
      }

      // ── [29] atr_pct50 ───────────────────────────────────────
      {
         double rank = 0;
         for(int k = 0; k < 50; k++)
            if(atr14[MathMin(i+k,n_total-1)] <= atr) rank++;
         feat[29] = rank / 50.0;
      }

      // ── [30] bb_squeeze_cnt ──────────────────────────────────
      // (簡略版: sqz_cnt/20  現時点のsqueeze値をローリング)
      {
         double sq_cnt = 0;
         for(int k = 0; k < 20; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            double sma_k = 0;
            for(int j = 0; j < 20; j++) sma_k += c[MathMin(ki+j,n_total-1)];
            sma_k /= 20.0;
            double var_k = 0;
            for(int j = 0; j < 20; j++) { double d=c[MathMin(ki+j,n_total-1)]-sma_k; var_k+=d*d; }
            double std_k = MathSqrt(var_k/20.0);
            double bb_u = sma_k+2.0*std_k, bb_l = sma_k-2.0*std_k;
            double ku = ema20[ki]+1.5*atr14[ki], kl = ema20[ki]-1.5*atr14[ki];
            if(bb_u < ku && bb_l > kl) sq_cnt++;
         }
         feat[30] = sq_cnt / 20.0;
      }

      // ── [31-33] body, upper_w, lower_w ──────────────────────
      double body_raw = cv - ov;
      double range_raw= hv - lv + 1e-9;
      double body_abs = MathAbs(body_raw);
      double upper_w  = (hv - MathMax(cv, ov)) / (atr + 1e-9);
      double lower_w  = (MathMin(cv, ov) - lv)  / (atr + 1e-9);
      feat[31] = body_raw / (atr + 1e-9);
      feat[32] = upper_w;
      feat[33] = lower_w;

      // ── [34-36] ret1, ret5, ret20 ────────────────────────────
      feat[34] = (c[MathMin(i+1,n_total-1)]>0) ? (cv/c[MathMin(i+1,n_total-1)]-1.0) : 0.0;
      feat[35] = (c[MathMin(i+5,n_total-1)]>0) ? (cv/c[MathMin(i+5,n_total-1)]-1.0) : 0.0;
      feat[36] = (c[MathMin(i+20,n_total-1)]>0)? (cv/c[MathMin(i+20,n_total-1)]-1.0): 0.0;

      // ── [37] close_pct_range (24本) ──────────────────────────
      {
         double rhi = hv, rlo = lv;
         for(int k = 0; k < 24; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            rhi = MathMax(rhi, hh[ki]); rlo = MathMin(rlo, ll[ki]);
         }
         feat[37] = (cv - rlo) / (rhi - rlo + 1e-9);
      }

      // ── [38-39] consec_up, consec_dn ─────────────────────────
      {
         double cu = 0, cd = 0;
         for(int k = 0; k < 8; k++)
         {
            int k1=MathMin(i+k,n_total-1), k2=MathMin(i+k+1,n_total-1);
            if(c[k1] > c[k2]) cu++; else break;
         }
         for(int k = 0; k < 8; k++)
         {
            int k1=MathMin(i+k,n_total-1), k2=MathMin(i+k+1,n_total-1);
            if(c[k1] < c[k2]) cd++; else break;
         }
         feat[38] = cu / 8.0;
         feat[39] = cd / 8.0;
      }

      // ── [40] ret_accel ───────────────────────────────────────
      {
         double r1  = (c[MathMin(i+1,n_total-1)]>0) ? (cv/c[MathMin(i+1,n_total-1)]-1.0)  : 0.0;
         double r1p = (c[MathMin(i+2,n_total-1)]>0) ? (c[MathMin(i+1,n_total-1)]/c[MathMin(i+2,n_total-1)]-1.0) : 0.0;
         feat[40] = r1 - r1p;
      }

      // ── [41-45] ローソク足パターン ──────────────────────────
      {
         int ip1 = MathMin(i+1, n_total-1);
         double pc = c[ip1], po = op[ip1];
         double pb = pc - po;
         feat[41] = (body_raw>0 && pb<0 && ov<=pc && cv>=po) ? 1.0 : 0.0;
         feat[42] = (body_raw<0 && pb>0 && ov>=pc && cv<=po) ? 1.0 : 0.0;
         feat[43] = (lower_w*atr > body_abs*2 && upper_w*atr < body_abs*0.5) ? 1.0 : 0.0;
         feat[44] = (upper_w*atr > body_abs*2 && lower_w*atr < body_abs*0.5) ? 1.0 : 0.0;
         feat[45] = (body_abs < range_raw * 0.1) ? 1.0 : 0.0;
      }

      // ── [46] donchian_pos (20本) ─────────────────────────────
      {
         double dhi = hv, dlo = lv;
         for(int k = 0; k < 20; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            dhi = MathMax(dhi, hh[ki]); dlo = MathMin(dlo, ll[ki]);
         }
         feat[46] = (cv - dlo) / (dhi - dlo + 1e-9);
      }

      // ── [47-48] swing_hi_dist, swing_lo_dist (5本) ──────────
      {
         double shi = hv, slo = lv;
         for(int k = 0; k < 5; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            shi = MathMax(shi, hh[ki]); slo = MathMin(slo, ll[ki]);
         }
         feat[47] = (shi - cv) / (atr + 1e-9);
         feat[48] = (cv - slo) / (atr + 1e-9);
      }

      // ── [49] round_dist (1円単位) ────────────────────────────
      {
         double rl = MathRound(cv / 1.0) * 1.0;
         feat[49] = MathAbs(cv - rl) / (atr + 1e-9);
      }

      // ── [50] h4_trend (EMA84) ────────────────────────────────
      feat[50] = (cv - ema84[i]) / (atr + 1e-9);

      // ── [51] daily_range_pos (24本) ──────────────────────────
      {
         double dhi2 = hv, dlo2 = lv;
         for(int k = 0; k < 24; k++)
         {
            int ki = MathMin(i+k, n_total-1);
            dhi2 = MathMax(dhi2, hh[ki]); dlo2 = MathMin(dlo2, ll[ki]);
         }
         feat[51] = (cv - dlo2) / (dhi2 - dlo2 + 1e-9);
      }

      // ── [52] weekly_pos (168本) ──────────────────────────────
      {
         int wn = MathMin(168, n_total-i);
         double whi = hv, wlo = lv;
         for(int k = 0; k < wn; k++)
         {
            int ki = i+k;
            whi = MathMax(whi, hh[ki]); wlo = MathMin(wlo, ll[ki]);
         }
         feat[52] = (cv - wlo) / (whi - wlo + 1e-9);
      }

      // ── [53] gap_open ────────────────────────────────────────
      feat[53] = (ov - c[MathMin(i+1, n_total-1)]) / (atr + 1e-9);

      // ── [54-56] 一目均衡表 ───────────────────────────────────
      {
         double hi9=hv, lo9=lv;
         for(int k=0; k<9;  k++) { int ki=MathMin(i+k,n_total-1); hi9=MathMax(hi9,hh[ki]); lo9=MathMin(lo9,ll[ki]); }
         double tenkan = (hi9+lo9)/2.0;

         double hi26=hv, lo26=lv;
         for(int k=0; k<26; k++) { int ki=MathMin(i+k,n_total-1); hi26=MathMax(hi26,hh[ki]); lo26=MathMin(lo26,ll[ki]); }
         double kijun  = (hi26+lo26)/2.0;

         double spa = (tenkan+kijun)/2.0;

         double hi52=hv, lo52=lv;
         for(int k=0; k<52; k++) { int ki=MathMin(i+k,n_total-1); hi52=MathMax(hi52,hh[ki]); lo52=MathMin(lo52,ll[ki]); }
         double spb  = (hi52+lo52)/2.0;

         double ctop = MathMax(spa, spb); double cbot = MathMin(spa, spb);
         feat[54] = (tenkan - kijun)             / (atr + 1e-9);
         feat[55] = (cv - (ctop+cbot)/2.0)       / (atr + 1e-9);
         feat[56] = (ctop - cbot)                / (atr + 1e-9);
      }

      // ── [57-60] 出来高特徴量 ─────────────────────────────────
      {
         double vm20 = 0;
         for(int k=0; k<20; k++) vm20 += vol[MathMin(i+k,n_total-1)];
         vm20 /= 20.0;
         if(vm20 < 1) vm20 = 1;

         feat[57] = vv / vm20;

         double obv5 = 0;
         for(int k=0; k<5; k++)
         {
            int k1=MathMin(i+k,n_total-1), k2=MathMin(i+k+1,n_total-1);
            obv5 += (c[k1]>=c[k2]?1.0:-1.0)*vol[k1];
         }
         feat[58] = obv5 / (vm20*5.0 + 1e-9);

         // vol_trend = (vol/vol_ma20).diff(5) = 現在の比率 - 5本前の比率
         // vol_ma20[5本前] = bars i+5 から i+24 の20本平均
         double vm20_5 = 0;
         for(int k=5; k<25; k++) vm20_5 += vol[MathMin(i+k,n_total-1)];
         vm20_5 /= 20.0;
         if(vm20_5 < 1) vm20_5 = 1;
         double vol_5ago = vol[MathMin(i+5, n_total-1)];
         feat[59] = vv/vm20 - vol_5ago/vm20_5;

         double vn=0, vd=0;
         for(int k=0; k<24; k++) { int ki=MathMin(i+k,n_total-1); vn+=c[ki]*vol[ki]; vd+=vol[ki]; }
         double vwap = (vd>0) ? vn/vd : cv;
         feat[60] = (cv - vwap) / (atr + 1e-9);
      }

      // ── [61] CCI20 ───────────────────────────────────────────
      {
         double tp_m = 0;
         for(int k=0; k<20; k++)
         {
            int ki=MathMin(i+k,n_total-1);
            tp_m += (hh[ki]+ll[ki]+c[ki])/3.0;
         }
         tp_m /= 20.0;
         double mad = 0;
         for(int k=0; k<20; k++)
         {
            int ki=MathMin(i+k,n_total-1);
            mad += MathAbs((hh[ki]+ll[ki]+c[ki])/3.0 - tp_m);
         }
         mad /= 20.0;
         double tp_now = (hv+lv+cv)/3.0;
         feat[61] = (tp_now - tp_m) / (0.015*mad + 1e-9) / 100.0;
      }

      // ── [62-69] セッション ───────────────────────────────────
      {
         MqlDateTime mdt;
         TimeToStruct(rates[i].time, mdt);
         int hr      = mdt.hour;
         int dow_mql = mdt.day_of_week;
         // Python dayofweek: Mon=0,...,Fri=4,Sat=5,Sun=6
         int dow_py  = (dow_mql == 0) ? 6 : (dow_mql - 1);

         feat[62] = MathSin(2.0 * M_PI * hr     / 24.0);
         feat[63] = MathCos(2.0 * M_PI * hr     / 24.0);
         feat[64] = MathSin(2.0 * M_PI * dow_py / 5.0);
         feat[65] = MathCos(2.0 * M_PI * dow_py / 5.0);
         feat[66] = (hr >= 0  && hr < 9)  ? 1.0 : 0.0;
         feat[67] = (hr >= 7  && hr < 16) ? 1.0 : 0.0;
         feat[68] = (hr >= 13 && hr < 22) ? 1.0 : 0.0;
         feat[69] = (hr >= 13 && hr < 16) ? 1.0 : 0.0;
      }

      // ── 入力テンソルへ書き込み (古→新 の順: row = seq_len-1-t) ─
      int row = g_seq_len - 1 - t;
      for(int f = 0; f < g_n_feat; f++)
      {
         int fi = g_feat_idx[f];
         input_data[row * g_n_feat + f] = (float)feat[fi];
      }
   }  // end seq_len loop

   // ── ONNX 推論 ─────────────────────────────────────────────────
   float output_data[3];
   if(!OnnxRun(g_model, ONNX_DEFAULT, input_data, output_data))
   {
      Print("[EA] OnnxRun 失敗");
      return false;
   }
   probs[0] = output_data[0];
   probs[1] = output_data[1];
   probs[2] = output_data[2];

   // ── 診断ログ ────────────────────────────────────────────────
   if(InpDebugLog && g_debug_handle != INVALID_HANDLE && g_debug_count < InpDebugBars)
   {
      datetime bar_time = rates[0].time;
      string line = TimeToString(bar_time, TIME_DATE | TIME_MINUTES);
      int last_row = g_seq_len - 1;
      for(int f = 0; f < g_n_feat; f++)
         line += StringFormat(",%.6f", input_data[last_row * g_n_feat + f]);
      line += StringFormat(",%.6f,%.6f,%.6f", probs[0], probs[1], probs[2]);
      FileWriteString(g_debug_handle, line + "\n");
      FileFlush(g_debug_handle);
      g_debug_count++;
      if(g_debug_count >= InpDebugBars)
         Print("[EA] 診断ログ完了: ", InpDebugBars, "バー出力");
   }

   return true;
}

//──────────────────────────────────────────────────────────────────
// ATR14 取得 (CopyRatesから自前計算)
//──────────────────────────────────────────────────────────────────
double GetATR14()
{
   int n = 300;
   MqlRates rt[];
   ArraySetAsSeries(rt, true);
   if(CopyRates(_Symbol, PERIOD_H1, 1, n, rt) < n) return 0;
   double c_a[], h_a[], l_a[], tr_a[], atr_a[];
   ArrayResize(c_a, n); ArrayResize(h_a, n); ArrayResize(l_a, n);
   ArrayResize(tr_a, n); ArrayResize(atr_a, n);
   for(int i = 0; i < n; i++) { c_a[i]=rt[i].close; h_a[i]=rt[i].high; l_a[i]=rt[i].low; }
   _CalcTR(h_a, l_a, c_a, tr_a, n);
   _Rma(tr_a, atr_a, n, 14);
   return atr_a[0];
}

//──────────────────────────────────────────────────────────────────
// ロットサイズ計算
//──────────────────────────────────────────────────────────────────
double LotSize()
{
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   string currency    = AccountInfoString(ACCOUNT_CURRENCY);
   int magnification  = (currency == "JPY") ? 10000 : 100;

   double lot = MathCeil(free_margin * InpRiskPct / magnification) / 100.0;
   lot = lot - 0.01;
   if(lot < 0.01) lot = 0.01;
   if(lot > 1.0)  lot = MathCeil(lot);
   if(lot > 20.0) lot = 20.0;

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
   // norm_params.json は Common\Files\ のみ使用
   int fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(fh == INVALID_HANDLE)
   {
      Print("[EA] ファイルを開けません: Common\\Files\\", filename);
      return false;
   }
   string content = "";
   while(!FileIsEnding(fh)) content += FileReadString(fh);
   FileClose(fh);

   if(!ParseJsonArray(content, "\"mean\"", g_mean, 70)) return false;
   if(!ParseJsonArray(content, "\"std\"",  g_std,  70)) return false;

   // seq_len
   {
      string key = "\"seq_len\"";
      int pos = StringFind(content, key);
      if(pos >= 0)
      {
         pos += StringLen(key);
         int colon = StringFind(content, ":", pos);
         int nc    = StringFind(content, ",", colon);
         int nb    = StringFind(content, "}", colon);
         int end   = (nc < nb && nc >= 0) ? nc : nb;
         string val = StringSubstr(content, colon+1, end-colon-1);
         StringTrimLeft(val); StringTrimRight(val);
         g_seq_len = (int)StringToInteger(val);
      }
   }

   // feat_indices
   {
      string key = "\"feat_indices\"";
      int pos = StringFind(content, key);
      if(pos >= 0)
      {
         // キーが見つかった場合、値が null かどうかを確認する
         int colon = StringFind(content, ":", pos);
         string after = StringSubstr(content, colon+1, 10);
         StringTrimLeft(after);
         bool is_null = (StringFind(after, "null") == 0);

         int start = StringFind(content, "[", pos);
         int end   = (start >= 0) ? StringFind(content, "]", start) : -1;
         if(!is_null && start >= 0 && end > start)
         {
            string arr_str = StringSubstr(content, start+1, end-start-1);
            string parts[];
            int n = StringSplit(arr_str, ',', parts);
            ArrayResize(g_feat_idx, n);
            g_n_feat = n;
            for(int i = 0; i < n; i++)
            {
               StringTrimLeft(parts[i]); StringTrimRight(parts[i]);
               g_feat_idx[i] = (int)StringToInteger(parts[i]);
            }
         }
         else
         {
            // null → 全70特徴量使用
            ArrayResize(g_feat_idx, 70);
            g_n_feat = 70;
            for(int i = 0; i < 70; i++) g_feat_idx[i] = i;
         }
      }
      else
      {
         ArrayResize(g_feat_idx, 70);
         g_n_feat = 70;
         for(int i = 0; i < 70; i++) g_feat_idx[i] = i;
      }
   }

   g_threshold = ParseJsonDouble(content, "\"threshold\"", g_threshold);
   g_tp_atr    = ParseJsonDouble(content, "\"tp_atr\"",    g_tp_atr);
   g_sl_atr    = ParseJsonDouble(content, "\"sl_atr\"",    g_sl_atr);
   {
      double hb = ParseJsonDouble(content, "\"hold_bars\"", -1.0);
      if(hb > 0) g_hold_bars = (int)hb;
   }

   Print("[EA] norm_params読込完了  feat=", g_n_feat, " seq=", g_seq_len,
         "  threshold=", g_threshold, "  tp=", g_tp_atr, "  sl=", g_sl_atr, "  hold=", g_hold_bars,
         "  (入力値>0の場合は入力値が優先)");
   return true;
}

//──────────────────────────────────────────────────────────────────
// JSON パーサ
//──────────────────────────────────────────────────────────────────
bool ParseJsonArray(string &content, string key, double &out[], int max_size)
{
   int pos = StringFind(content, key);
   if(pos < 0) { Print("[EA] JSON キー未発見: ", key); return false; }
   int start = StringFind(content, "[", pos);
   int end   = StringFind(content, "]", start);
   if(start < 0 || end <= start) return false;
   string arr_str = StringSubstr(content, start+1, end-start-1);
   string parts[];
   int n = StringSplit(arr_str, ',', parts);
   n = MathMin(n, max_size);
   ArrayResize(out, n);
   for(int i = 0; i < n; i++)
   {
      StringTrimLeft(parts[i]); StringTrimRight(parts[i]);
      out[i] = StringToDouble(parts[i]);
   }
   return true;
}

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
   string val = StringSubstr(content, colon+1, end-colon-1);
   StringTrimLeft(val); StringTrimRight(val);
   if(StringLen(val) == 0) return default_val;
   return StringToDouble(val);
}
