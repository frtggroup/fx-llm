//+------------------------------------------------------------------+
//|  AI_EA_ONNX_v3.mq5  ─  AI FX Expert Advisor (ONNX統合版)             |
//|  【インジケータ完全自前計算版・Diff機能内蔵】                      |
//|  MT5組み込みインジケータを一切使わず、Python features.py と完全に  |
//|  同じ計算式 (自前 mqh ファイル群) を使用します。                   |
//+------------------------------------------------------------------+
#property copyright "FX AI EA"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>

#include "feat/feat_000.mqh"
#include "feat/feat_001.mqh"
#include "feat/feat_002.mqh"
#include "feat/feat_003.mqh"
#include "feat/feat_004.mqh"
#include "feat/feat_005.mqh"
#include "feat/feat_006.mqh"
#include "feat/feat_007.mqh"
#include "feat/feat_008.mqh"
#include "feat/feat_009.mqh"
#include "feat/feat_010.mqh"
#include "feat/feat_011.mqh"
#include "feat/feat_012.mqh"
#include "feat/feat_013.mqh"
#include "feat/feat_014.mqh"
#include "feat/feat_015.mqh"
#include "feat/feat_016.mqh"
#include "feat/feat_017.mqh"
#include "feat/feat_018.mqh"
#include "feat/feat_019.mqh"
#include "feat/feat_020.mqh"
#include "feat/feat_021.mqh"
#include "feat/feat_022.mqh"
#include "feat/feat_023.mqh"
#include "feat/feat_024.mqh"
#include "feat/feat_025.mqh"
#include "feat/feat_026.mqh"
#include "feat/feat_027.mqh"
#include "feat/feat_028.mqh"
#include "feat/feat_029.mqh"
#include "feat/feat_030.mqh"
#include "feat/feat_031.mqh"
#include "feat/feat_032.mqh"
#include "feat/feat_033.mqh"
#include "feat/feat_034.mqh"
#include "feat/feat_035.mqh"
#include "feat/feat_036.mqh"
#include "feat/feat_037.mqh"
#include "feat/feat_038.mqh"
#include "feat/feat_039.mqh"
#include "feat/feat_040.mqh"
#include "feat/feat_041.mqh"
#include "feat/feat_042.mqh"
#include "feat/feat_043.mqh"
#include "feat/feat_044.mqh"
#include "feat/feat_045.mqh"
#include "feat/feat_046.mqh"
#include "feat/feat_047.mqh"
#include "feat/feat_048.mqh"
#include "feat/feat_049.mqh"
#include "feat/feat_050.mqh"
#include "feat/feat_051.mqh"
#include "feat/feat_052.mqh"
#include "feat/feat_053.mqh"
#include "feat/feat_054.mqh"
#include "feat/feat_055.mqh"
#include "feat/feat_056.mqh"
#include "feat/feat_057.mqh"
#include "feat/feat_058.mqh"
#include "feat/feat_059.mqh"
#include "feat/feat_060.mqh"
#include "feat/feat_061.mqh"
#include "feat/feat_062.mqh"
#include "feat/feat_063.mqh"
#include "feat/feat_064.mqh"
#include "feat/feat_065.mqh"
#include "feat/feat_066.mqh"
#include "feat/feat_067.mqh"
#include "feat/feat_068.mqh"
#include "feat/feat_069.mqh"
#include "feat/feat_070.mqh"
#include "feat/feat_071.mqh"
#include "feat/feat_072.mqh"
#include "feat/feat_073.mqh"
#include "feat/feat_074.mqh"
#include "feat/feat_075.mqh"
#include "feat/feat_076.mqh"
#include "feat/feat_077.mqh"
#include "feat/feat_078.mqh"
#include "feat/feat_079.mqh"
#include "feat/feat_080.mqh"
#include "feat/feat_081.mqh"
#include "feat/feat_082.mqh"
#include "feat/feat_083.mqh"
#include "feat/feat_084.mqh"
#include "feat/feat_085.mqh"
#include "feat/feat_086.mqh"
#include "feat/feat_087.mqh"
#include "feat/feat_088.mqh"
#include "feat/feat_089.mqh"
#include "feat/feat_090.mqh"
#include "feat/feat_091.mqh"
#include "feat/feat_092.mqh"
#include "feat/feat_093.mqh"
#include "feat/feat_094.mqh"
#include "feat/feat_095.mqh"
#include "feat/feat_096.mqh"
#include "feat/feat_097.mqh"
#include "feat/feat_098.mqh"
#include "feat/feat_099.mqh"
#include "feat/feat_100.mqh"
#include "feat/feat_101.mqh"
#include "feat/feat_102.mqh"
#include "feat/feat_103.mqh"
#include "feat/feat_104.mqh"
#include "feat/feat_105.mqh"
#include "feat/feat_106.mqh"
#include "feat/feat_107.mqh"
#include "feat/feat_108.mqh"
#include "feat/feat_109.mqh"
#include "feat/feat_110.mqh"
#include "feat/feat_111.mqh"
#include "feat/feat_112.mqh"
#include "feat/feat_113.mqh"
#include "feat/feat_114.mqh"
#include "feat/feat_115.mqh"
#include "feat/feat_116.mqh"
#include "feat/feat_117.mqh"
#include "feat/feat_118.mqh"
#include "feat/feat_119.mqh"
#include "feat/feat_120.mqh"
#include "feat/feat_121.mqh"
#include "feat/feat_122.mqh"
#include "feat/feat_123.mqh"
#include "feat/feat_124.mqh"
#include "feat/feat_125.mqh"
#include "feat/feat_126.mqh"
#include "feat/feat_127.mqh"
#include "feat/feat_128.mqh"
#include "feat/feat_129.mqh"
#include "feat/feat_130.mqh"
#include "feat/feat_131.mqh"
#include "feat/feat_132.mqh"
#include "feat/feat_133.mqh"
#include "feat/feat_134.mqh"
#include "feat/feat_135.mqh"
#include "feat/feat_136.mqh"
#include "feat/feat_137.mqh"
#include "feat/feat_138.mqh"
#include "feat/feat_139.mqh"
#include "feat/feat_140.mqh"
#include "feat/feat_141.mqh"
#include "feat/feat_142.mqh"
#include "feat/feat_143.mqh"
#include "feat/feat_144.mqh"
#include "feat/feat_145.mqh"
#include "feat/feat_146.mqh"
#include "feat/feat_147.mqh"
#include "feat/feat_148.mqh"
#include "feat/feat_149.mqh"
#include "feat/feat_150.mqh"
#include "feat/feat_151.mqh"
#include "feat/feat_152.mqh"
#include "feat/feat_153.mqh"
#include "feat/feat_154.mqh"
#include "feat/feat_155.mqh"
#include "feat/feat_156.mqh"
#include "feat/feat_157.mqh"
#include "feat/feat_158.mqh"
#include "feat/feat_159.mqh"
#include "feat/feat_160.mqh"
#include "feat/feat_161.mqh"
#include "feat/feat_162.mqh"
#include "feat/feat_163.mqh"
#include "feat/feat_164.mqh"
#include "feat/feat_165.mqh"
#include "feat/feat_166.mqh"
#include "feat/feat_167.mqh"
#include "feat/feat_168.mqh"
#include "feat/feat_169.mqh"
#include "feat/feat_170.mqh"
#include "feat/feat_171.mqh"
#include "feat/feat_172.mqh"
#include "feat/feat_173.mqh"
#include "feat/feat_174.mqh"
#include "feat/feat_175.mqh"
#include "feat/feat_176.mqh"
#include "feat/feat_177.mqh"
#include "feat/feat_178.mqh"
#include "feat/feat_179.mqh"
#include "feat/feat_180.mqh"
#include "feat/feat_181.mqh"
#include "feat/feat_182.mqh"
#include "feat/feat_183.mqh"
#include "feat/feat_184.mqh"
#include "feat/feat_185.mqh"
#include "feat/feat_186.mqh"
#include "feat/feat_187.mqh"
#include "feat/feat_188.mqh"
#include "feat/feat_189.mqh"
#include "feat/feat_190.mqh"
#include "feat/feat_191.mqh"
#include "feat/feat_192.mqh"
#include "feat/feat_193.mqh"
#include "feat/feat_194.mqh"
#include "feat/feat_195.mqh"
#include "feat/feat_196.mqh"
#include "feat/feat_197.mqh"
#include "feat/feat_198.mqh"
#include "feat/feat_199.mqh"
#include "feat/feat_200.mqh"
#include "feat/feat_201.mqh"
#include "feat/feat_202.mqh"
#include "feat/feat_203.mqh"
#include "feat/feat_204.mqh"
#include "feat/feat_205.mqh"
#include "feat/feat_206.mqh"
#include "feat/feat_207.mqh"
#include "feat/feat_208.mqh"
#include "feat/feat_209.mqh"
#include "feat/feat_210.mqh"
#include "feat/feat_211.mqh"
#include "feat/feat_212.mqh"
#include "feat/feat_213.mqh"
#include "feat/feat_214.mqh"
#include "feat/feat_215.mqh"
#include "feat/feat_216.mqh"
#include "feat/feat_217.mqh"
#include "feat/feat_218.mqh"
#include "feat/feat_219.mqh"
#include "feat/feat_220.mqh"
#include "feat/feat_221.mqh"
#include "feat/feat_222.mqh"
#include "feat/feat_223.mqh"
#include "feat/feat_224.mqh"
#include "feat/feat_225.mqh"
#include "feat/feat_226.mqh"
#include "feat/feat_227.mqh"
#include "feat/feat_228.mqh"
#include "feat/feat_229.mqh"

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
int      g_n_feat  = 5750;
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
{
   return iTime(_Symbol, PERIOD_H1, i);
}

//──────────────────────────────────────────────────────────────────
// OnInit
//──────────────────────────────────────────────────────────────────
int OnInit()
{
   if(!LoadNormParams(InpNormFile))
   {
      Print("[EA] norm_params 読み込み失敗: ", InpNormFile);
      return INIT_FAILED;
   }

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
      g_debug_handle = FileOpen("feat_debug_v3.csv", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(g_debug_handle != INVALID_HANDLE)
      {
         string hdr = "datetime";
         for(int f = 0; f < g_n_feat; f++) hdr += StringFormat(",feat%d(idx%d)", f, g_feat_idx[f]);
         hdr += ",p_hold,p_buy,p_sell";
         FileWriteString(g_debug_handle, hdr + "\n");
      }
   }

   Print("[EA] 初期化完了 v3 (230Base + Diffs)  seq=", g_seq_len, " feat=", g_n_feat);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_model != INVALID_HANDLE) OnnxRelease(g_model);
   if(g_debug_handle != INVALID_HANDLE) FileClose(g_debug_handle);
}

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
   {
      double sl = NormalizeDouble(ask - use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(ask + use_tp_atr * atr, _Digits);
      if(g_trade.Buy(lot, _Symbol, ask, sl, tp, StringFormat("AI BUY p=%.3f", p_buy)))
         { g_pos_bars = 0; Print("[EA] BUY  lot=", lot, " sl=", sl, " tp=", tp); }
   }
   else if(p_sell > use_threshold && p_sell > p_buy && p_sell > p_hold)
   {
      double sl = NormalizeDouble(bid + use_sl_atr * atr, _Digits);
      double tp = NormalizeDouble(bid - use_tp_atr * atr, _Digits);
      if(g_trade.Sell(lot, _Symbol, bid, sl, tp, StringFormat("AI SELL p=%.3f", p_sell)))
         { g_pos_bars = 0; Print("[EA] SELL lot=", lot, " sl=", sl, " tp=", tp); }
   }
}

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
         if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagic)
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
      if(PositionSelectByTicket(tk) && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
   }
   return false;
}

//══════════════════════════════════════════════════════════════════
// ONNX 推論 (Base 230 features + 24 diffs = 5750)
//══════════════════════════════════════════════════════════════════
bool RunInference(float &probs[])
{
   int n_total = N_WARM + g_seq_len + 24;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int got = CopyRates(_Symbol, PERIOD_H1, 0, n_total, rates);
   if(got < n_total)
   {
      Print("[EA] CopyRates不足: ", got, "/", n_total);
      return false;
   }

   double c[], hh[], ll[], op[], vol[];
   ArrayResize(c, n_total); ArrayResize(hh, n_total);
   ArrayResize(ll, n_total); ArrayResize(op, n_total); ArrayResize(vol, n_total);
   for(int i = 0; i < n_total; i++)
   {
      c[i]   = rates[i].close;
      hh[i]  = rates[i].high;
      ll[i]  = rates[i].low;
      op[i]  = rates[i].open;
      vol[i] = (double)rates[i].tick_volume;
   }

   // ── 230 Base features calculated for seq_len + diff offset limits
   // base_arr[t][f] where t = 0 to seq_len + 24 - 1
   int max_t = g_seq_len + 24;
   double base_arr[];
   ArrayResize(base_arr, max_t * 230);

   for(int bar = 0; bar < max_t; bar++)
   {
      // To mimic 2D array behaviour: base_arr[bar * N_BASE + f]
       base_arr[bar * 230 + 0] = CalcFeat_000(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 1] = CalcFeat_001(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 2] = CalcFeat_002(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 3] = CalcFeat_003(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 4] = CalcFeat_004(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 5] = CalcFeat_005(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 6] = CalcFeat_006(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 7] = CalcFeat_007(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 8] = CalcFeat_008(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 9] = CalcFeat_009(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 10] = CalcFeat_010(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 11] = CalcFeat_011(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 12] = CalcFeat_012(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 13] = CalcFeat_013(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 14] = CalcFeat_014(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 15] = CalcFeat_015(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 16] = CalcFeat_016(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 17] = CalcFeat_017(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 18] = CalcFeat_018(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 19] = CalcFeat_019(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 20] = CalcFeat_020(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 21] = CalcFeat_021(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 22] = CalcFeat_022(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 23] = CalcFeat_023(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 24] = CalcFeat_024(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 25] = CalcFeat_025(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 26] = CalcFeat_026(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 27] = CalcFeat_027(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 28] = CalcFeat_028(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 29] = CalcFeat_029(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 30] = CalcFeat_030(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 31] = CalcFeat_031(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 32] = CalcFeat_032(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 33] = CalcFeat_033(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 34] = CalcFeat_034(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 35] = CalcFeat_035(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 36] = CalcFeat_036(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 37] = CalcFeat_037(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 38] = CalcFeat_038(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 39] = CalcFeat_039(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 40] = CalcFeat_040(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 41] = CalcFeat_041(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 42] = CalcFeat_042(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 43] = CalcFeat_043(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 44] = CalcFeat_044(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 45] = CalcFeat_045(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 46] = CalcFeat_046(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 47] = CalcFeat_047(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 48] = CalcFeat_048(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 49] = CalcFeat_049(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 50] = CalcFeat_050(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 51] = CalcFeat_051(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 52] = CalcFeat_052(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 53] = CalcFeat_053(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 54] = CalcFeat_054(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 55] = CalcFeat_055(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 56] = CalcFeat_056(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 57] = CalcFeat_057(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 58] = CalcFeat_058(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 59] = CalcFeat_059(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 60] = CalcFeat_060(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 61] = CalcFeat_061(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 62] = CalcFeat_062(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 63] = CalcFeat_063(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 64] = CalcFeat_064(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 65] = CalcFeat_065(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 66] = CalcFeat_066(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 67] = CalcFeat_067(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 68] = CalcFeat_068(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 69] = CalcFeat_069(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 70] = CalcFeat_070(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 71] = CalcFeat_071(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 72] = CalcFeat_072(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 73] = CalcFeat_073(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 74] = CalcFeat_074(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 75] = CalcFeat_075(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 76] = CalcFeat_076(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 77] = CalcFeat_077(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 78] = CalcFeat_078(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 79] = CalcFeat_079(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 80] = CalcFeat_080(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 81] = CalcFeat_081(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 82] = CalcFeat_082(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 83] = CalcFeat_083(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 84] = CalcFeat_084(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 85] = CalcFeat_085(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 86] = CalcFeat_086(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 87] = CalcFeat_087(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 88] = CalcFeat_088(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 89] = CalcFeat_089(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 90] = CalcFeat_090(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 91] = CalcFeat_091(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 92] = CalcFeat_092(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 93] = CalcFeat_093(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 94] = CalcFeat_094(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 95] = CalcFeat_095(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 96] = CalcFeat_096(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 97] = CalcFeat_097(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 98] = CalcFeat_098(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 99] = CalcFeat_099(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 100] = CalcFeat_100(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 101] = CalcFeat_101(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 102] = CalcFeat_102(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 103] = CalcFeat_103(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 104] = CalcFeat_104(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 105] = CalcFeat_105(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 106] = CalcFeat_106(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 107] = CalcFeat_107(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 108] = CalcFeat_108(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 109] = CalcFeat_109(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 110] = CalcFeat_110(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 111] = CalcFeat_111(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 112] = CalcFeat_112(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 113] = CalcFeat_113(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 114] = CalcFeat_114(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 115] = CalcFeat_115(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 116] = CalcFeat_116(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 117] = CalcFeat_117(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 118] = CalcFeat_118(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 119] = CalcFeat_119(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 120] = CalcFeat_120(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 121] = CalcFeat_121(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 122] = CalcFeat_122(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 123] = CalcFeat_123(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 124] = CalcFeat_124(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 125] = CalcFeat_125(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 126] = CalcFeat_126(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 127] = CalcFeat_127(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 128] = CalcFeat_128(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 129] = CalcFeat_129(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 130] = CalcFeat_130(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 131] = CalcFeat_131(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 132] = CalcFeat_132(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 133] = CalcFeat_133(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 134] = CalcFeat_134(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 135] = CalcFeat_135(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 136] = CalcFeat_136(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 137] = CalcFeat_137(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 138] = CalcFeat_138(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 139] = CalcFeat_139(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 140] = CalcFeat_140(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 141] = CalcFeat_141(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 142] = CalcFeat_142(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 143] = CalcFeat_143(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 144] = CalcFeat_144(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 145] = CalcFeat_145(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 146] = CalcFeat_146(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 147] = CalcFeat_147(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 148] = CalcFeat_148(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 149] = CalcFeat_149(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 150] = CalcFeat_150(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 151] = CalcFeat_151(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 152] = CalcFeat_152(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 153] = CalcFeat_153(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 154] = CalcFeat_154(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 155] = CalcFeat_155(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 156] = CalcFeat_156(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 157] = CalcFeat_157(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 158] = CalcFeat_158(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 159] = CalcFeat_159(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 160] = CalcFeat_160(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 161] = CalcFeat_161(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 162] = CalcFeat_162(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 163] = CalcFeat_163(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 164] = CalcFeat_164(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 165] = CalcFeat_165(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 166] = CalcFeat_166(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 167] = CalcFeat_167(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 168] = CalcFeat_168(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 169] = CalcFeat_169(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 170] = CalcFeat_170(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 171] = CalcFeat_171(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 172] = CalcFeat_172(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 173] = CalcFeat_173(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 174] = CalcFeat_174(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 175] = CalcFeat_175(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 176] = CalcFeat_176(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 177] = CalcFeat_177(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 178] = CalcFeat_178(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 179] = CalcFeat_179(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 180] = CalcFeat_180(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 181] = CalcFeat_181(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 182] = CalcFeat_182(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 183] = CalcFeat_183(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 184] = CalcFeat_184(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 185] = CalcFeat_185(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 186] = CalcFeat_186(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 187] = CalcFeat_187(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 188] = CalcFeat_188(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 189] = CalcFeat_189(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 190] = CalcFeat_190(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 191] = CalcFeat_191(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 192] = CalcFeat_192(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 193] = CalcFeat_193(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 194] = CalcFeat_194(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 195] = CalcFeat_195(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 196] = CalcFeat_196(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 197] = CalcFeat_197(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 198] = CalcFeat_198(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 199] = CalcFeat_199(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 200] = CalcFeat_200(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 201] = CalcFeat_201(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 202] = CalcFeat_202(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 203] = CalcFeat_203(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 204] = CalcFeat_204(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 205] = CalcFeat_205(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 206] = CalcFeat_206(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 207] = CalcFeat_207(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 208] = CalcFeat_208(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 209] = CalcFeat_209(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 210] = CalcFeat_210(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 211] = CalcFeat_211(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 212] = CalcFeat_212(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 213] = CalcFeat_213(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 214] = CalcFeat_214(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 215] = CalcFeat_215(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 216] = CalcFeat_216(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 217] = CalcFeat_217(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 218] = CalcFeat_218(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 219] = CalcFeat_219(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 220] = CalcFeat_220(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 221] = CalcFeat_221(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 222] = CalcFeat_222(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 223] = CalcFeat_223(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 224] = CalcFeat_224(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 225] = CalcFeat_225(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 226] = CalcFeat_226(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 227] = CalcFeat_227(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 228] = CalcFeat_228(c, hh, ll, op, vol, n_total, bar);
       base_arr[bar * 230 + 229] = CalcFeat_229(c, hh, ll, op, vol, n_total, bar);
   }

   // ── Input tensor construction
   float input_data[];
   ArrayResize(input_data, g_seq_len * g_n_feat);

   float all_scaled[];
   ArrayResize(all_scaled, 5750);

   for(int t = 0; t < g_seq_len; t++)
   {
      int row = g_seq_len - 1 - t; // Old to new order

      // 1. Base features
      for(int f=0; f<230; f++)
      {
         all_scaled[f] = (float)base_arr[t * 230 + f];
      }

      // 2. Diff features (d1 to d24)
      int idx = 230;
      for(int k=1; k<=24; k++)
      {
         for(int f=0; f<230; f++)
         {
            // d(k) = base_arr[t + k - 1][f] - base_arr[t + k][f]
            float val = (float)(base_arr[(t + k - 1) * 230 + f] - base_arr[(t + k) * 230 + f]);
            all_scaled[idx++] = val;
         }
      }

      // Apply index mapping and normalization
      for(int f = 0; f < g_n_feat; f++)
      {
         int fi = g_feat_idx[f];
         float val = all_scaled[fi];
         val = (float)((val - g_mean[fi]) / (g_std[fi] + 1e-9));
         input_data[row * g_n_feat + f] = val;
      }
   }

   float output_data[3];
   if(!OnnxRun(g_model, ONNX_DEFAULT, input_data, output_data))
   {
      Print("[EA] OnnxRun 失敗");
      return false;
   }
   probs[0] = output_data[0];
   probs[1] = output_data[1];
   probs[2] = output_data[2];

   if(InpDebugLog && g_debug_handle != INVALID_HANDLE && g_debug_count < InpDebugBars)
   {
      string line = TimeToString(rates[0].time, TIME_DATE | TIME_MINUTES);
      int last_row = g_seq_len - 1;
      for(int f = 0; f < g_n_feat; f++) line += StringFormat(",%.6f", input_data[last_row * g_n_feat + f]);
      line += StringFormat(",%.6f,%.6f,%.6f", probs[0], probs[1], probs[2]);
      FileWriteString(g_debug_handle, line + "\n");
      FileFlush(g_debug_handle);
      g_debug_count++;
   }

   return true;
}

//──────────────────────────────────────────────────────────────────
// ATR14 取得 (Simple)
//──────────────────────────────────────────────────────────────────
double GetATR14()
{
   int n = 300;
   MqlRates rt[];
   ArraySetAsSeries(rt, true);
   if(CopyRates(_Symbol, PERIOD_H1, 0, n, rt) < n) return 0;
   
   double tr_a[], atr_a[]; ArrayResize(tr_a, n); ArrayResize(atr_a, n);
   tr_a[n-1] = rt[n-1].high - rt[n-1].low;
   for(int i=n-2; i>=0; i--) {
      double hl = rt[i].high - rt[i].low;
      double hpc= MathAbs(rt[i].high - rt[i+1].close);
      double lpc= MathAbs(rt[i].low  - rt[i+1].close);
      tr_a[i] = MathMax(hl, MathMax(hpc, lpc));
   }
   double alpha = 1.0/14.0;
   atr_a[n-1] = tr_a[n-1];
   for(int i=n-2; i>=0; i--) atr_a[i] = alpha*tr_a[i] + (1-alpha)*atr_a[i+1];
   return atr_a[0];
}

double LotSize()
{
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   string currency    = AccountInfoString(ACCOUNT_CURRENCY);
   int magnification  = (StringFind(currency, "JPY") >= 0) ? 10000 : 100;
   double lot = MathCeil(free_margin * InpRiskPct / magnification) / 100.0;
   lot = lot - 0.01; if(lot < 0.01) lot = 0.01; if(lot > 20.0) lot = 20.0;
   double ls = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   return MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), MathRound(lot / ls) * ls);
}

bool LoadNormParams(string filename)
{
   int fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(fh == INVALID_HANDLE) return false;
   string content = ""; while(!FileIsEnding(fh)) content += FileReadString(fh); FileClose(fh);

   int TOTAL_ALL = 5750;
   if(!ParseJsonArray(content, "\"mean\"", g_mean, TOTAL_ALL)) return false;
   if(!ParseJsonArray(content, "\"std\"",  g_std,  TOTAL_ALL)) return false;

   g_seq_len = (int)ParseJsonDouble(content, "\"seq_len\"", g_seq_len);

   string key = "\"feat_indices\"";
   int pos = StringFind(content, key);
   if(pos >= 0)
   {
      int colon = StringFind(content, ":", pos);
      string after = StringSubstr(content, colon+1, 10);
      StringTrimLeft(after);
      if(StringFind(after, "null") == 0)
      {
         ArrayResize(g_feat_idx, TOTAL_ALL); g_n_feat = TOTAL_ALL;
         for(int i = 0; i < TOTAL_ALL; i++) g_feat_idx[i] = i;
      }
      else
      {
         int start = StringFind(content, "[", pos), end = StringFind(content, "]", start);
         if(start >= 0 && end > start)
         {
            string parts[]; int n = StringSplit(StringSubstr(content, start+1, end-start-1), ',', parts);
            ArrayResize(g_feat_idx, n); g_n_feat = n;
            for(int i=0; i<n; i++) { StringTrimLeft(parts[i]); g_feat_idx[i] = (int)StringToInteger(parts[i]); }
         }
      }
   }
   else
   {
      ArrayResize(g_feat_idx, TOTAL_ALL); g_n_feat = TOTAL_ALL;
      for(int i = 0; i < TOTAL_ALL; i++) g_feat_idx[i] = i;
   }

   g_threshold = ParseJsonDouble(content, "\"threshold\"", g_threshold);
   g_tp_atr    = ParseJsonDouble(content, "\"tp_atr\"",    g_tp_atr);
   g_sl_atr    = ParseJsonDouble(content, "\"sl_atr\"",    g_sl_atr);
   double hb   = ParseJsonDouble(content, "\"hold_bars\"", -1.0);
   if(hb > 0) g_hold_bars = (int)hb;

   Print("[EA] norm_params読込完了  feat=", g_n_feat, " seq=", g_seq_len, " thresh=", g_threshold);
   return true;
}

bool ParseJsonArray(string &content, string key, double &out[], int max_size)
{
   int pos = StringFind(content, key); if(pos < 0) return false;
   int start = StringFind(content, "[", pos), end = StringFind(content, "]", start);
   if(start < 0 || end <= start) return false;
   string parts[]; int n = StringSplit(StringSubstr(content, start+1, end-start-1), ',', parts);
   n = MathMin(n, max_size); ArrayResize(out, n);
   for(int i=0; i<n; i++) { StringTrimLeft(parts[i]); out[i] = StringToDouble(parts[i]); }
   return true;
}

double ParseJsonDouble(const string &content, string key, double default_val)
{
   int pos = StringFind(content, key); if(pos < 0) return default_val;
   int colon = StringFind(content, ":", pos), nc = StringFind(content, ",", colon), nb = StringFind(content, "}", colon);
   int end = (nc >= 0 && (nb < 0 || nc < nb)) ? nc : nb;
   if(end < 0) return default_val;
   string val = StringSubstr(content, colon+1, end-colon-1); StringTrimLeft(val); StringTrimRight(val);
   return (StringLen(val) > 0) ? StringToDouble(val) : default_val;
}
