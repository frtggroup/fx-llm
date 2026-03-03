import sys, os
from pathlib import Path
import features as fx

OUT_MQ5 = Path('f:/FX/fx-ea5/AI_EA_ONNX_v3.mq5')

# The base features length
N_BASE = len(fx.BASE_FEATURE_COLS)  # 230
N_DIFFS = fx._N_DIFFS  # 24

# Create includes
includes = "\n".join([f'#include "feat/feat_{i:03d}.mqh"' for i in range(N_BASE)])

# Create base feature array calls
base_calls = []
for i in range(N_BASE):
    base_calls.append(f"       base_arr[bar][{i}] = CalcFeat_{i:03d}(c, hh, ll, op, vol, n_total, bar);")
base_calls_str = "\n".join(base_calls)

mq5_template = f"""//+------------------------------------------------------------------+
//|  AI_EA_ONNX_v3.mq5  ─  AI FX Expert Advisor (ONNX統合版)             |
//|  【インジケータ完全自前計算版・Diff機能内蔵】                      |
//|  MT5組み込みインジケータを一切使わず、Python features.py と完全に  |
//|  同じ計算式 (自前 mqh ファイル群) を使用します。                   |
//+------------------------------------------------------------------+
#property copyright "FX AI EA"
#property version   "3.00"
#property strict

#include <Trade\\Trade.mqh>

{includes}

//──────────────────────────────────────────────────────────────────
// 入力パラメータ
//──────────────────────────────────────────────────────────────────
input group "=== モデルファイル (Common\\\\Files\\\\ に置く) ==="
input string   InpModelFile  = "fx_model.onnx";        // ONNX モデル
input string   InpNormFile   = "norm_params.json";     // 正規化パラメータ

input group "=== エントリー設定 (-1 = JSONから自動読込) ==="
input double   InpThreshold  = -1;    // エントリー確率閾値 (-1=JSON)
input double   InpTpAtr      = -1;    // TP倍率 ATR× (-1=JSON)
input double   InpSlAtr      = -1;    // SL倍率 ATR× (-1=JSON)
input int      InpMaxHoldBars= -1;    // 最大保有バー数 (-1=JSON)

input group "=== リスク管理 ==="
input double   InpRiskPct    = 1.0;    // リスク率 (%)
input int      InpMagic      = 20260303;

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
int      g_n_feat  = {N_BASE * (N_DIFFS + 1)};
int      g_seq_len = 20;

double   g_threshold = -1.0;
double   g_tp_atr    = -1.0;
double   g_sl_atr    = -1.0;
int      g_hold_bars = -1;

datetime g_last_bar     = 0;
int      g_pos_bars     = 0;
int      g_debug_handle = INVALID_HANDLE;
int      g_debug_count  = 0;

#define N_WARM 1000  // EMAウォームアップバー数

//──────────────────────────────────────────────────────────────────
// bar_time helper functions required by feat_XXX.mqh
//──────────────────────────────────────────────────────────────────
datetime GetBarTime(int i)
{{
   return iTime(_Symbol, PERIOD_H1, i);
}}

//──────────────────────────────────────────────────────────────────
// OnInit
//──────────────────────────────────────────────────────────────────
int OnInit()
{{
   if(!LoadNormParams(InpNormFile))
   {{
      Print("[EA] norm_params 読み込み失敗: ", InpNormFile);
      return INIT_FAILED;
   }}

   g_model = OnnxCreate(InpModelFile, ONNX_COMMON_FOLDER);
   if(g_model == INVALID_HANDLE)
   {{
      Print("[EA] ONNX モデル読み込み失敗: Common\\\\Files\\\\", InpModelFile);
      return INIT_FAILED;
   }}

   ulong in_shape[]  = {{1, (ulong)g_seq_len, (ulong)g_n_feat}};
   ulong out_shape[] = {{1, 3}};
   if(!OnnxSetInputShape(g_model, 0, in_shape) ||
      !OnnxSetOutputShape(g_model, 0, out_shape))
   {{
      Print("[EA] ONNX シェイプ設定失敗  seq=", g_seq_len, " feat=", g_n_feat);
      OnnxRelease(g_model);
      return INIT_FAILED;
   }}

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(20);

   if(InpDebugLog)
   {{
      g_debug_handle = FileOpen("feat_debug_v3.csv", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(g_debug_handle != INVALID_HANDLE)
      {{
         string hdr = "datetime";
         for(int f = 0; f < g_n_feat; f++) hdr += StringFormat(",feat%d(idx%d)", f, g_feat_idx[f]);
         hdr += ",p_hold,p_buy,p_sell";
         FileWriteString(g_debug_handle, hdr + "\\n");
      }}
   }}

   Print("[EA] 初期化完了 v3 (230Base + Diffs)  seq=", g_seq_len, " feat=", g_n_feat);
   return INIT_SUCCEEDED;
}}

void OnDeinit(const int reason)
{{
   if(g_model != INVALID_HANDLE) OnnxRelease(g_model);
   if(g_debug_handle != INVALID_HANDLE) FileClose(g_debug_handle);
}}

void OnTick()
{{
   datetime cur_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(cur_bar == g_last_bar) return;
   g_last_bar = cur_bar;

   if(InpTimeFilter)
   {{
      MqlDateTime _t; TimeToStruct(TimeCurrent(), _t); int h = _t.hour;
      if(h < InpStartHour || h >= InpEndHour) return;
   }}

   ManagePosition();

   float probs[3];
   if(!RunInference(probs)) return;

   float p_hold = probs[0], p_buy = probs[1], p_sell = probs[2];

   static int dbg_cnt = 0;
   if(++dbg_cnt % 100 == 0)
      Print("[PROB] hold=", DoubleToString(p_hold,3), " buy=", DoubleToString(p_buy,3), " sell=", DoubleToString(p_sell,3));

   if(HasPosition()) return;

   double atr = GetATR14();
   if(atr <= 0) return;

   double use_threshold = (InpThreshold > 0) ? InpThreshold : ((g_threshold > 0) ? g_threshold : 0.40);
   double use_tp_atr    = (InpTpAtr    > 0) ? InpTpAtr    : ((g_tp_atr    > 0) ? g_tp_atr    : 2.0);
   double use_sl_atr    = (InpSlAtr    > 0) ? InpSlAtr    : ((g_sl_atr    > 0) ? g_sl_atr    : 1.0);

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double lot = LotSize();

   if(p_buy > use_threshold && p_buy > p_sell && p_buy > p_hold)
   {{
      double sl = NormalizeDouble(ask - use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(ask + use_tp_atr * atr, _Digits);
      if(g_trade.Buy(lot, _Symbol, ask, sl, tp, StringFormat("AI BUY p=%.3f", p_buy)))
         {{ g_pos_bars = 0; Print("[EA] BUY  lot=", lot, " sl=", sl, " tp=", tp); }}
   }}
   else if(p_sell > use_threshold && p_sell > p_buy && p_sell > p_hold)
   {{
      double sl = NormalizeDouble(bid + use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(bid - use_tp_atr * atr, _Digits);
      if(g_trade.Sell(lot, _Symbol, bid, sl, tp, StringFormat("AI SELL p=%.3f", p_sell)))
         {{ g_pos_bars = 0; Print("[EA] SELL lot=", lot, " sl=", sl, " tp=", tp); }}
   }}
}}

void ManagePosition()
{{
   if(!HasPosition()) {{ g_pos_bars = 0; return; }}
   g_pos_bars++;
   int use_hold = (InpMaxHoldBars > 0) ? InpMaxHoldBars : ((g_hold_bars > 0) ? g_hold_bars : 20);
   if(g_pos_bars >= use_hold)
   {{
      for(int i = PositionsTotal()-1; i >= 0; i--)
      {{
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         {{
            g_trade.PositionClose(ticket);
            Print("[EA] 最大保有バー超過 → 決済");
         }}
      }}
      g_pos_bars = 0;
   }}
}}

bool HasPosition()
{{
   for(int i = 0; i < PositionsTotal(); i++)
   {{
      ulong tk = PositionGetTicket(i);
      if(PositionSelectByTicket(tk) && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   }}
   return false;
}}

//══════════════════════════════════════════════════════════════════
// ONNX 推論 (Base {N_BASE} features + {N_DIFFS} diffs = {N_BASE*(N_DIFFS+1)})
//══════════════════════════════════════════════════════════════════
bool RunInference(float &probs[])
{{
   int n_total = N_WARM + g_seq_len + {N_DIFFS};

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int got = CopyRates(_Symbol, PERIOD_H1, 0, n_total, rates);
   if(got < n_total)
   {{
      Print("[EA] CopyRates不足: ", got, "/", n_total);
      return false;
   }}

   double c[], hh[], ll[], op[], vol[];
   ArrayResize(c, n_total); ArrayResize(hh, n_total);
   ArrayResize(ll, n_total); ArrayResize(op, n_total); ArrayResize(vol, n_total);
   for(int i = 0; i < n_total; i++)
   {{
      c[i]   = rates[i].close;
      hh[i]  = rates[i].high;
      ll[i]  = rates[i].low;
      op[i]  = rates[i].open;
      vol[i] = (double)rates[i].tick_volume;
   }}

   // ── {N_BASE} Base features calculated for seq_len + diff offset limits
   // base_arr[t][f] where t = 0 to seq_len + 24 - 1
   int max_t = g_seq_len + {N_DIFFS};
   double base_arr[];
   ArrayResize(base_arr, max_t * {N_BASE});

   for(int bar = 0; bar < max_t; bar++)
   {{
      // To mimic 2D array behaviour: base_arr[bar * N_BASE + f]
{base_calls_str.replace('base_arr[bar][', 'base_arr[bar * ' + str(N_BASE) + ' + ')}
   }}

   // ── Input tensor construction
   float input_data[];
   ArrayResize(input_data, g_seq_len * g_n_feat);

   float all_scaled[];
   ArrayResize(all_scaled, {N_BASE*(N_DIFFS+1)});

   for(int t = 0; t < g_seq_len; t++)
   {{
      int row = g_seq_len - 1 - t; // Old to new order

      // 1. Base features
      for(int f=0; f<{N_BASE}; f++)
      {{
         all_scaled[f] = (float)base_arr[t * {N_BASE} + f];
      }}

      // 2. Diff features (d1 to d24)
      int idx = {N_BASE};
      for(int k=1; k<={N_DIFFS}; k++)
      {{
         for(int f=0; f<{N_BASE}; f++)
         {{
            // d(k) = base_arr[t + k - 1][f] - base_arr[t + k][f]
            float val = (float)(base_arr[(t + k - 1) * {N_BASE} + f] - base_arr[(t + k) * {N_BASE} + f]);
            all_scaled[idx++] = val;
         }}
      }}

      // Apply index mapping and normalization
      for(int f = 0; f < g_n_feat; f++)
      {{
         int fi = g_feat_idx[f];
         float val = all_scaled[fi];
         val = (float)((val - g_mean[fi]) / (g_std[fi] + 1e-9));
         input_data[row * g_n_feat + f] = val;
      }}
   }}

   float output_data[3];
   if(!OnnxRun(g_model, ONNX_DEFAULT, input_data, output_data))
   {{
      Print("[EA] OnnxRun 失敗");
      return false;
   }}
   probs[0] = output_data[0];
   probs[1] = output_data[1];
   probs[2] = output_data[2];

   if(InpDebugLog && g_debug_handle != INVALID_HANDLE && g_debug_count < InpDebugBars)
   {{
      string line = TimeToString(rates[0].time, TIME_DATE | TIME_MINUTES);
      int last_row = g_seq_len - 1;
      for(int f = 0; f < g_n_feat; f++) line += StringFormat(",%.6f", input_data[last_row * g_n_feat + f]);
      line += StringFormat(",%.6f,%.6f,%.6f", probs[0], probs[1], probs[2]);
      FileWriteString(g_debug_handle, line + "\\n");
      FileFlush(g_debug_handle);
      g_debug_count++;
   }}

   return true;
}}

//──────────────────────────────────────────────────────────────────
// ATR14 取得 (Simple)
//──────────────────────────────────────────────────────────────────
double GetATR14()
{{
   int n = 300;
   MqlRates rt[];
   ArraySetAsSeries(rt, true);
   if(CopyRates(_Symbol, PERIOD_H1, 0, n, rt) < n) return 0;
   
   double tr_a[], atr_a[]; ArrayResize(tr_a, n); ArrayResize(atr_a, n);
   tr_a[n-1] = rt[n-1].high - rt[n-1].low;
   for(int i=n-2; i>=0; i--) {{
      double hl = rt[i].high - rt[i].low;
      double hpc= MathAbs(rt[i].high - rt[i+1].close);
      double lpc= MathAbs(rt[i].low  - rt[i+1].close);
      tr_a[i] = MathMax(hl, MathMax(hpc, lpc));
   }}
   double alpha = 1.0/14.0;
   atr_a[n-1] = tr_a[n-1];
   for(int i=n-2; i>=0; i--) atr_a[i] = alpha*tr_a[i] + (1-alpha)*atr_a[i+1];
   return atr_a[0];
}}

double LotSize()
{{
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   string currency    = AccountInfoString(ACCOUNT_CURRENCY);
   int magnification  = (StringFind(currency, "JPY") >= 0) ? 10000 : 100;
   double lot = MathCeil(free_margin * InpRiskPct / magnification) / 100.0;
   lot = lot - 0.01; if(lot < 0.01) lot = 0.01; if(lot > 20.0) lot = 20.0;
   double ls = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   return MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), MathRound(lot / ls) * ls);
}}

bool LoadNormParams(string filename)
{{
   int fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(fh == INVALID_HANDLE) return false;
   string content = ""; while(!FileIsEnding(fh)) content += FileReadString(fh); FileClose(fh);

   int TOTAL_ALL = {N_BASE * (N_DIFFS + 1)};
   if(!ParseJsonArray(content, "\\"mean\\"", g_mean, TOTAL_ALL)) return false;
   if(!ParseJsonArray(content, "\\"std\\"",  g_std,  TOTAL_ALL)) return false;

   g_seq_len = (int)ParseJsonDouble(content, "\\"seq_len\\"", g_seq_len);

   string key = "\\"feat_indices\\"";
   int pos = StringFind(content, key);
   if(pos >= 0)
   {{
      int colon = StringFind(content, ":", pos);
      string after = StringSubstr(content, colon+1, 10);
      StringTrimLeft(after);
      if(StringFind(after, "null") == 0)
      {{
         ArrayResize(g_feat_idx, TOTAL_ALL); g_n_feat = TOTAL_ALL;
         for(int i = 0; i < TOTAL_ALL; i++) g_feat_idx[i] = i;
      }}
      else
      {{
         int start = StringFind(content, "[", pos), end = StringFind(content, "]", start);
         if(start >= 0 && end > start)
         {{
            string parts[]; int n = StringSplit(StringSubstr(content, start+1, end-start-1), ',', parts);
            ArrayResize(g_feat_idx, n); g_n_feat = n;
            for(int i=0; i<n; i++) {{ StringTrimLeft(parts[i]); g_feat_idx[i] = (int)StringToInteger(parts[i]); }}
         }}
      }}
   }}
   else
   {{
      ArrayResize(g_feat_idx, TOTAL_ALL); g_n_feat = TOTAL_ALL;
      for(int i = 0; i < TOTAL_ALL; i++) g_feat_idx[i] = i;
   }}

   g_threshold = ParseJsonDouble(content, "\\"threshold\\"", g_threshold);
   g_tp_atr    = ParseJsonDouble(content, "\\"tp_atr\\"",    g_tp_atr);
   g_sl_atr    = ParseJsonDouble(content, "\\"sl_atr\\"",    g_sl_atr);
   double hb   = ParseJsonDouble(content, "\\"hold_bars\\"", -1.0);
   if(hb > 0) g_hold_bars = (int)hb;

   Print("[EA] norm_params読込完了  feat=", g_n_feat, " seq=", g_seq_len, " thresh=", g_threshold);
   return true;
}}

bool ParseJsonArray(string &content, string key, double &out[], int max_size)
{{
   int pos = StringFind(content, key); if(pos < 0) return false;
   int start = StringFind(content, "[", pos), end = StringFind(content, "]", start);
   if(start < 0 || end <= start) return false;
   string parts[]; int n = StringSplit(StringSubstr(content, start+1, end-start-1), ',', parts);
   n = MathMin(n, max_size); ArrayResize(out, n);
   for(int i=0; i<n; i++) {{ StringTrimLeft(parts[i]); out[i] = StringToDouble(parts[i]); }}
   return true;
}}

double ParseJsonDouble(const string &content, string key, double default_val)
{{
   int pos = StringFind(content, key); if(pos < 0) return default_val;
   int colon = StringFind(content, ":", pos), nc = StringFind(content, ",", colon), nb = StringFind(content, "}}", colon);
   int end = (nc >= 0 && (nb < 0 || nc < nb)) ? nc : nb;
   if(end < 0) return default_val;
   string val = StringSubstr(content, colon+1, end-colon-1); StringTrimLeft(val); StringTrimRight(val);
   return (StringLen(val) > 0) ? StringToDouble(val) : default_val;
}}
"""

OUT_MQ5.write_text(mq5_template, encoding='utf-8')
print("Generated AI_EA_ONNX_v3.mq5 successfully.")
