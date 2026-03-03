//+------------------------------------------------------------------+
//|  LLM_Signal_EA.mq5                                               |
//|  FX LLM ファインチューニング済みモデルのシグナルを使った MT5 EA    |
//|                                                                  |
//|  使い方:                                                          |
//|  1. mt5_signals_YYYYMMDD.csv を MQL5/Files/ フォルダに置く        |
//|  2. このファイルを MQL5/Experts/ フォルダに置いてコンパイル        |
//|  3. USDJPY H1 チャートに適用                                       |
//|  4. ストラテジーテスターで                                          |
//|       銘柄: USDJPY  時間足: H1                                     |
//|       期間: シグナルCSVの期間 (直近1年) に合わせる                  |
//+------------------------------------------------------------------+
#property copyright "FX LLM Fine-tuning DOK"
#property version   "1.00"
#property strict

//--- 入力パラメータ
input string   SignalFile    = "mt5_signals.csv";  // シグナルCSVファイル名 (MQL5/Files/内)
input double   LotSize       = 0.1;                // 取引ロット数
input double   MinConfidence = 0.0;                // 最低信頼度 (0.0=全シグナル使用)
input bool     UseCSVTP_SL   = true;               // CSVのTP/SL価格を使う
input int      FixedTP_Pips  = 150;                // UseCSVTP_SL=false時の固定TP(pips)
input int      FixedSL_Pips  = 100;                // UseCSVTP_SL=false時の固定SL(pips)
input int      MaxPositions  = 1;                  // 同時保有ポジション数上限
input bool     CloseOnOpposite = true;             // 逆シグナルで既存ポジションをクローズ
input bool     PrintSignals  = true;               // シグナル読込ログ表示

//--- シグナル構造体
struct SignalRow {
   datetime  dt;
   string    signal;
   double    confidence;
   double    open_price;
   double    high_price;
   double    low_price;
   double    close_price;
   double    atr;
   double    tp_price;
   double    sl_price;
   string    true_label;
};

//--- グローバル変数
SignalRow  g_signals[];
int        g_signal_count = 0;
bool       g_loaded       = false;
datetime   g_last_bar     = 0;

//--- バックテスト統計
int        g_trades_total  = 0;
int        g_trades_win    = 0;
double     g_gross_profit  = 0.0;
double     g_gross_loss    = 0.0;


//+------------------------------------------------------------------+
//| Expert 初期化                                                      |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== LLM Signal EA 初期化 ===");
   Print("シグナルファイル: ", SignalFile);

   if (!LoadSignals()) {
      Alert("シグナルCSVの読み込みに失敗しました: ", SignalFile);
      return INIT_FAILED;
   }

   Print("シグナル読込完了: ", g_signal_count, " 件");
   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert 終了                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   int total = g_trades_win + (g_trades_total - g_trades_win);
   double net = g_gross_profit + g_gross_loss;  // g_gross_loss は負値
   double pf  = (g_gross_loss != 0.0) ? g_gross_profit / MathAbs(g_gross_loss) : 0.0;
   double wr  = (g_trades_total > 0) ? (double)g_trades_win / g_trades_total * 100.0 : 0.0;

   Print("=== バックテスト結果サマリー ===");
   Print("  総取引数  : ", g_trades_total);
   Print("  勝ち      : ", g_trades_win,  " (", DoubleToString(wr,1), "%)");
   Print("  純損益    : ", DoubleToString(net, 5));
   Print("  プロフィットファクター: ", DoubleToString(pf, 3));
}


//+------------------------------------------------------------------+
//| Tick 処理                                                          |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   if (bar_time == g_last_bar) return;
   g_last_bar = bar_time;

   OnNewBar(bar_time);
}


//+------------------------------------------------------------------+
//| 新バー処理                                                          |
//+------------------------------------------------------------------+
void OnNewBar(datetime bar_time)
{
   // 現在バー時刻に合致するシグナルを検索
   SignalRow sig;
   if (!FindSignal(bar_time, sig)) return;

   // 信頼度フィルタ
   if (sig.confidence < MinConfidence) return;
   if (sig.signal == "HOLD") return;

   // ポジション確認
   int pos_count = CountPositions();

   // 逆シグナルで既存ポジションをクローズ
   if (CloseOnOpposite) {
      CloseOppositePositions(sig.signal);
   }

   // 新規ポジション
   if (pos_count < MaxPositions) {
      OpenPosition(sig);
   }
}


//+------------------------------------------------------------------+
//| ポジション開く                                                      |
//+------------------------------------------------------------------+
void OpenPosition(const SignalRow &sig)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   double tp, sl, price;
   ENUM_ORDER_TYPE order_type;

   if (sig.signal == "BUY") {
      order_type = ORDER_TYPE_BUY;
      price = ask;
      if (UseCSVTP_SL && sig.tp_price > 0.0) {
         tp = sig.tp_price;
         sl = sig.sl_price;
      } else {
         tp = NormalizeDouble(price + FixedTP_Pips * point * 10, digits);
         sl = NormalizeDouble(price - FixedSL_Pips * point * 10, digits);
      }
   } else if (sig.signal == "SELL") {
      order_type = ORDER_TYPE_SELL;
      price = bid;
      if (UseCSVTP_SL && sig.tp_price > 0.0) {
         tp = sig.tp_price;
         sl = sig.sl_price;
      } else {
         tp = NormalizeDouble(price - FixedTP_Pips * point * 10, digits);
         sl = NormalizeDouble(price + FixedSL_Pips * point * 10, digits);
      }
   } else {
      return;
   }

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};

   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = LotSize;
   req.type      = order_type;
   req.price     = price;
   req.tp        = tp;
   req.sl        = sl;
   req.deviation = 10;
   req.magic     = 20250101;
   req.comment   = StringFormat("LLM_%s_conf%.2f", sig.signal, sig.confidence);

   if (OrderSend(req, res)) {
      if (PrintSignals)
         Print(TimeToString(sig.dt), " | ", sig.signal,
               " conf=", DoubleToString(sig.confidence, 3),
               " tp=", DoubleToString(tp, digits),
               " sl=", DoubleToString(sl, digits));
   } else {
      Print("OrderSend エラー: ", res.retcode, " at ", TimeToString(sig.dt));
   }
}


//+------------------------------------------------------------------+
//| 逆シグナルのポジションをクローズ                                    |
//+------------------------------------------------------------------+
void CloseOppositePositions(const string &signal)
{
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (!PositionSelectByTicket(ticket)) continue;
      if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if (PositionGetInteger(POSITION_MAGIC) != 20250101) continue;

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      bool should_close = (signal == "BUY"  && ptype == POSITION_TYPE_SELL) ||
                          (signal == "SELL" && ptype == POSITION_TYPE_BUY);
      if (!should_close) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action    = TRADE_ACTION_DEAL;
      req.position  = ticket;
      req.symbol    = _Symbol;
      req.volume    = PositionGetDouble(POSITION_VOLUME);
      req.type      = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      req.price     = (ptype == POSITION_TYPE_BUY)
                        ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                        : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      req.deviation = 10;
      req.magic     = 20250101;
      OrderSend(req, res);
   }
}


//+------------------------------------------------------------------+
//| ポジション数カウント                                                |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for (int i = 0; i < PositionsTotal(); i++) {
      ulong ticket = PositionGetTicket(i);
      if (!PositionSelectByTicket(ticket)) continue;
      if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if (PositionGetInteger(POSITION_MAGIC) != 20250101) continue;
      count++;
   }
   return count;
}


//+------------------------------------------------------------------+
//| シグナル検索 (二分探索)                                             |
//+------------------------------------------------------------------+
bool FindSignal(datetime bar_time, SignalRow &out)
{
   // H1バーのxx:00:00 に正規化
   datetime target = bar_time - bar_time % 3600;

   int lo = 0, hi = g_signal_count - 1;
   while (lo <= hi) {
      int mid = (lo + hi) / 2;
      if (g_signals[mid].dt == target) {
         out = g_signals[mid];
         return true;
      } else if (g_signals[mid].dt < target) {
         lo = mid + 1;
      } else {
         hi = mid - 1;
      }
   }
   return false;
}


//+------------------------------------------------------------------+
//| CSVシグナル読込                                                     |
//+------------------------------------------------------------------+
bool LoadSignals()
{
   int fh = FileOpen(SignalFile, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if (fh == INVALID_HANDLE) {
      // ファイル名にパスがない場合はFiles/直下を試す
      Print("ファイルが見つかりません: ", SignalFile);
      Print("MQL5/Files/ フォルダに置いてください");
      return false;
   }

   // ヘッダー行をスキップ
   string header = FileReadString(fh);
   if (StringFind(header, "datetime") < 0) {
      // 1列目だけ読んだ場合は残りの列もスキップ
      while (!FileIsLineEnding(fh) && !FileIsEnding(fh))
         FileReadString(fh);
   }

   int count = 0;
   ArrayResize(g_signals, 0);

   while (!FileIsEnding(fh)) {
      string dt_str   = FileReadString(fh); if (FileIsEnding(fh) && dt_str == "") break;
      string sig_str  = FileReadString(fh);
      string conf_str = FileReadString(fh);
      string open_s   = FileReadString(fh);
      string high_s   = FileReadString(fh);
      string low_s    = FileReadString(fh);
      string close_s  = FileReadString(fh);
      string atr_s    = FileReadString(fh);
      string tp_s     = FileReadString(fh);
      string sl_s     = FileReadString(fh);
      string true_lbl = FileReadString(fh);

      if (dt_str == "" || dt_str == "datetime") continue;

      SignalRow row;
      // "2025-02-25 10:00" → datetime
      row.dt          = StringToTime(dt_str);
      row.signal      = sig_str;
      row.confidence  = StringToDouble(conf_str);
      row.open_price  = StringToDouble(open_s);
      row.high_price  = StringToDouble(high_s);
      row.low_price   = StringToDouble(low_s);
      row.close_price = StringToDouble(close_s);
      row.atr         = StringToDouble(atr_s);
      row.tp_price    = StringToDouble(tp_s);
      row.sl_price    = StringToDouble(sl_s);
      row.true_label  = true_lbl;

      ArrayResize(g_signals, count + 1);
      g_signals[count] = row;
      count++;
   }

   FileClose(fh);
   g_signal_count = count;
   g_loaded       = (count > 0);

   if (PrintSignals && count > 0) {
      Print("読込: ", count, " シグナル / 先頭: ",
            TimeToString(g_signals[0].dt), " / 末尾: ",
            TimeToString(g_signals[count-1].dt));
   }

   return g_loaded;
}


//+------------------------------------------------------------------+
//| 約定イベント (勝敗カウント)                                         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest     &req,
                        const MqlTradeResult      &res)
{
   if (trans.type != TRADE_TRANSACTION_DEAL_ADD) return;

   ulong deal = trans.deal;
   if (!HistoryDealSelect(deal)) return;
   if (HistoryDealGetInteger(deal, DEAL_MAGIC) != 20250101) return;

   ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal, DEAL_ENTRY);
   if (entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) return;

   double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
   g_trades_total++;
   if (profit > 0.0) {
      g_trades_win++;
      g_gross_profit += profit;
   } else {
      g_gross_loss += profit;
   }
}
