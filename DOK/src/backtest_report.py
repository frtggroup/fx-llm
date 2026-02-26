"""
LLM バックテスト + HTML レポート生成
訓練完了後に pipeline.py から呼ばれる

生成するレポート:
  - エクイティカーブ
  - 月次リターン表
  - シグナル分布 (BUY/SELL/HOLD)
  - 損益分布ヒストグラム
  - 訓練メトリクス (Loss/Accuracy)
  - 主要成績サマリー
"""
import sys, json, time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, '/workspace/ai_ea')
sys.path.insert(0, '/workspace/src')

WORKSPACE     = Path('/workspace')
ADAPTER_DIR   = WORKSPACE / 'output' / 'llm_adapter_best'
TEST_JSONL    = WORKSPACE / 'output' / 'llm_test.jsonl'
TRAIN_RESULT  = WORKSPACE / 'output' / 'llm_train_result.json'
PROGRESS_JSON = WORKSPACE / 'progress.json'
REPORT_DIR    = WORKSPACE / 'reports'
DATA_PATH     = WORKSPACE / 'data' / 'USDJPY_M1.csv'

SPREAD       = 0.003
LABEL_NAMES  = ['HOLD', 'BUY', 'SELL']
HOLD_BARS    = 48


# ──────────────────────────────────────────────────────────────────────────────
# 進捗更新
# ──────────────────────────────────────────────────────────────────────────────
def update_progress(patch: dict) -> None:
    try:
        cur = {}
        if PROGRESS_JSON.exists():
            cur = json.loads(PROGRESS_JSON.read_text())
        cur.update(patch)
        PROGRESS_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# モデル推論
# ──────────────────────────────────────────────────────────────────────────────
def load_model_for_inference(adapter_dir: Path):
    import torch
    from unsloth import FastLanguageModel
    print(f"  アダプター読み込み: {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = str(adapter_dir),
        max_seq_length    = 1024,
        load_in_4bit      = True,
        dtype             = None,
        trust_remote_code = True,
    )
    FastLanguageModel.for_inference(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  デバイス: {device}")
    return model, tokenizer, device


def make_chat_prompt(prompt_text: str, tokenizer) -> str:
    messages = [
        {"role": "system",
         "content": ("You are a professional FX trading signal analyst. "
                     "Analyze the market data and respond with exactly one word: "
                     "BUY, SELL, or HOLD.")},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def predict_batch(model, tokenizer, samples: list, device, batch_size: int = 32):
    import torch
    label_tids = [tokenizer.encode(n, add_special_tokens=False)[0]
                  for n in LABEL_NAMES]
    lbl_tensor = torch.tensor(label_tids, device=device)
    preds = []
    n = len(samples)
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch  = samples[start: start + batch_size]
        texts  = [make_chat_prompt(s['prompt'], tokenizer) for s in batch]
        enc    = tokenizer(texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=1024)
        input_ids  = enc['input_ids'].to(device)
        attn_masks = enc['attention_mask'].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                                  enabled=(device.type=='cuda')):
            out = model(input_ids=input_ids, attention_mask=attn_masks)
        last_pos = attn_masks.sum(dim=1) - 1
        for b in range(len(batch)):
            lpos = last_pos[b].item()
            lgt  = out.logits[b, lpos, lbl_tensor]
            preds.append(int(lgt.argmax().item()))
        pct = (start + len(batch)) / n * 100
        elapsed = time.time() - t0
        eta = elapsed / max(start + len(batch), 1) * (n - start - len(batch))
        print(f"  推論 {pct:.0f}%  ({start+len(batch):,}/{n:,})  "
              f"残り {eta:.0f}s", end='\r', flush=True)
        update_progress({
            'phase': 'backtest',
            'message': f'バックテスト推論中... {pct:.0f}% ({start+len(batch):,}/{n:,})',
        })
    print(f"  推論完了: {n:,} サンプル  {time.time()-t0:.1f}s")
    return np.array(preds, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# バックテスト
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(preds: np.ndarray, df_te,
                 seq_len: int = 20,
                 tp_mult: float = 1.5,
                 sl_mult: float = 1.0) -> dict:
    close  = df_te['close'].values
    atr    = df_te['atr14'].values
    high   = df_te['high'].values
    low    = df_te['low'].values
    dates  = df_te.index
    n      = len(preds)
    trades = []
    pos    = None

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close):
            break
        c = close[bi]; a = atr[bi]

        if pos:
            hi = high[bi]; lo = low[bi]
            age = i - pos['i0']
            pnl = None
            if pos['side'] == 1:
                if lo <= pos['sl']:   pnl = pos['sl'] - pos['entry'] - SPREAD
                elif hi >= pos['tp']: pnl = pos['tp'] - pos['entry'] - SPREAD
            else:
                if hi >= pos['sl']:   pnl = pos['entry'] - pos['sl'] - SPREAD
                elif lo <= pos['tp']: pnl = pos['entry'] - pos['tp'] - SPREAD
            if pnl is None and age >= HOLD_BARS:
                pnl = (c - pos['entry']) * pos['side'] - SPREAD
            if pnl is not None:
                trades.append({
                    'pnl': pnl, 'side': pos['side'],
                    'entry_i': pos['i0'], 'exit_i': i,
                    'entry_date': str(dates[seq_len - 1 + pos['i0']])[:10],
                    'exit_date':  str(dates[bi])[:10],
                })
                pos = None

        if pos is None:
            cls = int(preds[i])
            if cls == 1:
                entry = c + SPREAD
                pos = {'side': 1, 'entry': entry,
                       'tp': entry + tp_mult * a,
                       'sl': entry - sl_mult * a, 'i0': i}
            elif cls == 2:
                entry = c - SPREAD
                pos = {'side': -1, 'entry': entry,
                       'tp': entry - tp_mult * a,
                       'sl': entry + sl_mult * a, 'i0': i}

    if len(trades) < 10:
        return {'pf': 0.0, 'trades': len(trades), 'win_rate': 0.0,
                'net_pnl': 0.0, 'note': f'取引数不足: {len(trades)}'}

    pnl_arr  = np.array([t['pnl'] for t in trades])
    gp       = float(pnl_arr[pnl_arr > 0].sum())
    gl       = float(abs(pnl_arr[pnl_arr < 0].sum()))
    equity   = np.cumsum(pnl_arr).tolist()
    dd       = _max_drawdown(equity)

    return {
        'pf':           round(gp / max(gl, 1e-9), 4),
        'trades':       len(trades),
        'win_rate':     round(float((pnl_arr > 0).mean()), 4),
        'net_pnl':      round(float(pnl_arr.sum()), 4),
        'gross_profit': round(gp, 4),
        'gross_loss':   round(gl, 4),
        'max_drawdown': round(dd, 4),
        'avg_pnl':      round(float(pnl_arr.mean()), 6),
        'trades_list':  trades,
        'equity_curve': equity,
        'pnl_list':     pnl_arr.tolist(),
    }


def _max_drawdown(equity: list) -> float:
    if not equity:
        return 0.0
    peak = -float('inf'); dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = max(dd, peak - v)
    return dd


def monthly_returns(trades: list) -> dict:
    monthly = {}
    for t in trades:
        key = t['exit_date'][:7]  # YYYY-MM
        monthly[key] = monthly.get(key, 0.0) + t['pnl']
    return {k: round(v, 4) for k, v in sorted(monthly.items())}


# ──────────────────────────────────────────────────────────────────────────────
# HTML レポート生成
# ──────────────────────────────────────────────────────────────────────────────
def generate_html_report(bt_result: dict, train_result: dict,
                          preds: np.ndarray, test_samples: list,
                          epoch_log: list) -> str:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pf       = bt_result.get('pf', 0)
    trades   = bt_result.get('trades', 0)
    wr       = bt_result.get('win_rate', 0)
    net_pnl  = bt_result.get('net_pnl', 0)
    gp       = bt_result.get('gross_profit', 0)
    gl       = bt_result.get('gross_loss', 0)
    mdd      = bt_result.get('max_drawdown', 0)
    best_acc = train_result.get('best_accuracy', 0)
    model_id = train_result.get('model_id', '-')
    lr       = train_result.get('lora_r', '-')
    eff_b    = train_result.get('effective_batch', '-')
    tr_min   = train_result.get('total_min', 0)
    eq_curve = bt_result.get('equity_curve', [])
    pnl_list = bt_result.get('pnl_list', [])
    monthly  = monthly_returns(bt_result.get('trades_list', []))

    # シグナル分布
    sig_counts = {n: int((preds == i).sum()) for i, n in enumerate(LABEL_NAMES)}
    true_labels = np.array([{'HOLD': 0, 'BUY': 1, 'SELL': 2}[s['label']]
                              for s in test_samples])
    acc = float((preds == true_labels).mean())

    # LoSSチャートデータ
    ep_nums  = [r['epoch']      for r in epoch_log]
    ep_tl    = [r['train_loss'] for r in epoch_log]
    ep_vl    = [r.get('val_loss', None) for r in epoch_log]
    ep_acc   = [r.get('acc', 0) * 100   for r in epoch_log]

    # 月次リターン
    mo_keys = list(monthly.keys())
    mo_vals = list(monthly.values())
    mo_colors = ['rgba(63,185,80,0.8)' if v >= 0 else 'rgba(244,67,54,0.8)'
                 for v in mo_vals]

    # PnL ヒストグラム (20bin)
    if pnl_list:
        pnl_arr = np.array(pnl_list)
        hist, edges = np.histogram(pnl_arr, bins=25)
        hist_labels = [f'{(edges[i]+edges[i+1])/2:.4f}' for i in range(len(hist))]
        hist_colors = ['rgba(63,185,80,0.7)' if (edges[i]+edges[i+1])/2 >= 0
                       else 'rgba(244,67,54,0.7)' for i in range(len(hist))]
    else:
        hist, hist_labels, hist_colors = [], [], []

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>FX LLM バックテストレポート — {ts}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:24px}}
h1{{font-size:1.4em;color:#58a6ff;margin-bottom:6px}}
.ts{{font-size:0.78em;color:#8b949e;margin-bottom:20px}}
.grid{{display:grid;gap:14px;margin-bottom:14px}}
.g2{{grid-template-columns:1fr 1fr}}
.g3{{grid-template-columns:repeat(3,1fr)}}
.g4{{grid-template-columns:repeat(4,1fr)}}
.g6{{grid-template-columns:repeat(6,1fr)}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.card h2{{font-size:0.72em;color:#8b949e;margin-bottom:10px;text-transform:uppercase;letter-spacing:.8px}}
.kpi{{text-align:center;padding:12px}}
.kpi .v{{font-size:2em;font-weight:700}}
.kpi .l{{font-size:0.72em;color:#8b949e;margin-top:3px}}
table{{width:100%;border-collapse:collapse;font-size:0.8em}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e}}td:first-child,th:first-child{{text-align:left}}
tr:hover{{background:#21262d}}
.chart-wrap{{position:relative;height:200px}}
.footer{{text-align:center;font-size:0.72em;color:#484f58;margin-top:20px}}
</style>
</head>
<body>
<h1>📊 FX LLM バックテスト レポート</h1>
<div class="ts">生成日時: {ts} &nbsp;|&nbsp; モデル: {model_id} &nbsp;|&nbsp; LoRA rank: {lr} &nbsp;|&nbsp; 実効バッチ: {eff_b}</div>

<!-- KPI サマリー -->
<div class="grid g6" style="margin-bottom:14px">
  <div class="card kpi"><div class="v" style="color:{'#3fb950' if pf>=1.3 else '#f44336'}">{pf:.3f}</div><div class="l">プロフィットファクター</div></div>
  <div class="card kpi"><div class="v" style="color:#ffa657">{wr*100:.1f}%</div><div class="l">勝率</div></div>
  <div class="card kpi"><div class="v" style="color:#79c0ff">{trades:,}</div><div class="l">総取引数</div></div>
  <div class="card kpi"><div class="v" style="color:{'#3fb950' if net_pnl>=0 else '#f44336'}">{net_pnl:.4f}</div><div class="l">純損益 (pip)</div></div>
  <div class="card kpi"><div class="v" style="color:#f44336">{mdd:.4f}</div><div class="l">最大ドローダウン</div></div>
  <div class="card kpi"><div class="v" style="color:#3fb950">{best_acc*100:.2f}%</div><div class="l">分類精度 (Best)</div></div>
</div>

<div class="grid g2">
  <!-- エクイティカーブ -->
  <div class="card">
    <h2>エクイティカーブ</h2>
    <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
  </div>
  <!-- 月次リターン -->
  <div class="card">
    <h2>月次リターン</h2>
    <div class="chart-wrap"><canvas id="moChart"></canvas></div>
  </div>
</div>

<div class="grid g3" style="margin-top:14px">
  <!-- 訓練Loss -->
  <div class="card">
    <h2>訓練 Loss / Accuracy</h2>
    <div class="chart-wrap"><canvas id="trainChart"></canvas></div>
  </div>
  <!-- PnL 分布 -->
  <div class="card">
    <h2>損益分布</h2>
    <div class="chart-wrap"><canvas id="histChart"></canvas></div>
  </div>
  <!-- シグナル分布 -->
  <div class="card">
    <h2>シグナル分布</h2>
    <div class="chart-wrap"><canvas id="sigChart"></canvas></div>
  </div>
</div>

<!-- 訓練サマリー -->
<div class="card" style="margin-top:14px">
  <h2>訓練サマリー</h2>
  <table>
    <thead><tr><th>指標</th><th>値</th><th>指標</th><th>値</th></tr></thead>
    <tbody>
      <tr><td>モデル</td><td>{model_id}</td><td>訓練時間</td><td>{tr_min:.1f} 分</td></tr>
      <tr><td>LoRA rank</td><td>{lr}</td><td>分類精度(Best)</td><td>{best_acc*100:.2f}%</td></tr>
      <tr><td>実効バッチ</td><td>{eff_b}</td><td>推論精度</td><td>{acc*100:.2f}%</td></tr>
      <tr><td>総粗利</td><td>{gp:.4f}</td><td>総粗損</td><td>{gl:.4f}</td></tr>
    </tbody>
  </table>
</div>

<!-- 月次詳細テーブル -->
<div class="card" style="margin-top:14px">
  <h2>月次損益詳細</h2>
  <table>
    <thead><tr><th>年月</th><th>損益 (pip)</th><th>累計</th></tr></thead>
    <tbody>
{''.join(f'<tr><td>{k}</td><td style="color:{\'#3fb950\' if v>=0 else \'#f44336\'}">{v:+.4f}</td><td>{sum(mo_vals[:i+1]):.4f}</td></tr>' for i,(k,v) in enumerate(zip(mo_keys,mo_vals)))}
    </tbody>
  </table>
</div>

<div class="footer">FX LLM バックテストレポート &nbsp;|&nbsp; 生成: {ts}</div>

<script>
const eq = {json.dumps(eq_curve)};
const moKeys = {json.dumps(mo_keys)};
const moVals = {json.dumps(mo_vals)};
const moColors = {json.dumps(mo_colors)};
const epNums = {json.dumps(ep_nums)};
const epTl   = {json.dumps(ep_tl)};
const epVl   = {json.dumps(ep_vl)};
const epAcc  = {json.dumps(ep_acc)};
const hist   = {json.dumps(hist.tolist() if len(hist) else [])};
const histLbl= {json.dumps(hist_labels)};
const histCol= {json.dumps(hist_colors)};
const sigLabels = {json.dumps(list(sig_counts.keys()))};
const sigVals   = {json.dumps(list(sig_counts.values()))};

const dark = {{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{{legend:{{labels:{{color:'#e6edf3',font:{{size:10}}}}}}}},
  scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
          y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}};

// エクイティカーブ
new Chart(document.getElementById('eqChart').getContext('2d'),{{
  type:'line',
  data:{{labels:eq.map((_,i)=>i),datasets:[{{
    label:'Equity (pip)',data:eq,
    borderColor:'#58a6ff',backgroundColor:'#58a6ff14',
    borderWidth:2,pointRadius:0,fill:true,tension:.2
  }}]}},
  options:{{...dark,plugins:{{...dark.plugins,
    annotation:{{annotations:[{{type:'line',yMin:0,yMax:0,borderColor:'#484f58',borderWidth:1}}]}}
  }}}}
}});

// 月次リターン
new Chart(document.getElementById('moChart').getContext('2d'),{{
  type:'bar',
  data:{{labels:moKeys,datasets:[{{label:'月次損益',data:moVals,
    backgroundColor:moColors,borderWidth:0}}]}},
  options:dark
}});

// 訓練チャート
new Chart(document.getElementById('trainChart').getContext('2d'),{{
  type:'line',
  data:{{labels:epNums,datasets:[
    {{label:'Train Loss',data:epTl,borderColor:'#f0883e',borderWidth:2,pointRadius:3,tension:.3,yAxisID:'yL',spanGaps:true}},
    {{label:'Val Loss',  data:epVl,borderColor:'#79c0ff',borderWidth:2,pointRadius:3,tension:.3,yAxisID:'yL',spanGaps:false}},
    {{label:'Acc %',     data:epAcc,borderColor:'#3fb950',borderWidth:2,pointRadius:3,tension:.3,yAxisID:'yA',spanGaps:false}},
  ]}},
  options:{{...dark,scales:{{
    x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}},title:{{display:true,text:'Epoch',color:'#8b949e'}}}},
    yL:{{type:'linear',position:'left',ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
    yA:{{type:'linear',position:'right',min:0,max:100,
         ticks:{{color:'#3fb950',callback:v=>v+'%'}},grid:{{drawOnChartArea:false}}}},
  }}}}
}});

// 損益ヒスト
if(hist.length){{
  new Chart(document.getElementById('histChart').getContext('2d'),{{
    type:'bar',
    data:{{labels:histLbl,datasets:[{{label:'頻度',data:hist,backgroundColor:histCol,borderWidth:0}}]}},
    options:{{...dark,plugins:{{...dark.plugins,tooltip:{{callbacks:{{
      title: items=>'PnL ≈ '+items[0].label,
      label: item=>'取引数: '+item.raw
    }}}}}}}}
  }});
}}

// シグナル分布
new Chart(document.getElementById('sigChart').getContext('2d'),{{
  type:'doughnut',
  data:{{labels:sigLabels,datasets:[{{
    data:sigVals,
    backgroundColor:['#8b949e','#3fb950','#f44336'],
    borderWidth:2,borderColor:'#0d1117'
  }}]}},
  options:{{...dark,cutout:'55%',plugins:{{...dark.plugins,
    legend:{{position:'bottom',labels:{{color:'#e6edf3',font:{{size:11}}}}}}
  }}}}
}});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
# メイン実行
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=== LLM バックテスト + レポート生成 ===", flush=True)
    update_progress({'phase': 'backtest', 'message': 'バックテスト準備中...'})
    t0 = time.time()

    # テスト JSONL
    if not TEST_JSONL.exists():
        print(f"[ERROR] {TEST_JSONL} が見つかりません")
        update_progress({'phase': 'error', 'error': f'{TEST_JSONL} が見つかりません'})
        return None

    test_samples = []
    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    print(f"  テストサンプル: {len(test_samples):,}", flush=True)

    # テスト OHLCV データ (直近1年)
    print("  テストデータ読み込み中...", flush=True)
    from features import load_data, add_indicators
    df = load_data(str(DATA_PATH), timeframe='H1')
    df = add_indicators(df)
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)
    test_start = df.index[-1] - timedelta(days=365)
    df_te = df[df.index >= test_start].copy()
    print(f"  テスト期間: {df_te.index[0].date()} ~ {df_te.index[-1].date()}", flush=True)

    # モデル推論
    model, tokenizer, device = load_model_for_inference(ADAPTER_DIR)
    preds = predict_batch(model, tokenizer, test_samples, device, batch_size=32)

    # 精度
    true_labels = np.array([{'HOLD': 0, 'BUY': 1, 'SELL': 2}[s['label']]
                              for s in test_samples])
    acc = float((preds == true_labels).mean())
    print(f"  分類精度: {acc:.4f}", flush=True)

    for i, name in enumerate(LABEL_NAMES):
        cnt = int((preds == i).sum())
        print(f"    {name}: {cnt:,} ({cnt/len(preds)*100:.1f}%)", flush=True)

    # バックテスト
    update_progress({'message': 'バックテスト計算中...'})
    bt = run_backtest(preds, df_te)
    bt['accuracy'] = round(acc, 4)
    bt['elapsed_min'] = round((time.time() - t0) / 60, 1)

    print(f"\n=== バックテスト結果 ===", flush=True)
    print(f"  PF        : {bt['pf']}", flush=True)
    print(f"  取引数    : {bt['trades']}", flush=True)
    print(f"  勝率      : {bt.get('win_rate',0):.1%}", flush=True)
    print(f"  純損益    : {bt.get('net_pnl',0):.4f}", flush=True)
    print(f"  最大DD    : {bt.get('max_drawdown',0):.4f}", flush=True)
    print(f"  精度      : {acc:.4f}", flush=True)

    # 訓練結果
    train_result = {}
    if TRAIN_RESULT.exists():
        train_result = json.loads(TRAIN_RESULT.read_text())

    # epoch_log を progress.json から取得
    epoch_log = []
    try:
        prog = json.loads(PROGRESS_JSON.read_text())
        epoch_log = prog.get('epoch_log', [])
    except Exception:
        pass

    # HTML レポート生成
    update_progress({'message': 'HTMLレポート生成中...'})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORT_DIR / f'backtest_report_{ts_tag}.html'

    bt_for_html = {k: v for k, v in bt.items()
                   if k not in ('trades_list', 'equity_curve', 'pnl_list')}
    html = generate_html_report(bt, train_result, preds, test_samples, epoch_log)
    report_path.write_text(html, encoding='utf-8')
    print(f"  レポート保存: {report_path}", flush=True)

    # progress.json に結果を反映
    bt_clean = {k: v for k, v in bt.items()
                if not isinstance(v, (list,)) or k in ('equity_curve',)}
    bt_clean['equity_curve'] = bt.get('equity_curve', [])[:500]  # 先頭500点に間引き

    update_progress({
        'phase': 'complete',
        'backtest_result': {
            'pf':         bt.get('pf'),
            'trades':     bt.get('trades'),
            'win_rate':   bt.get('win_rate'),
            'net_pnl':    bt.get('net_pnl'),
            'max_drawdown': bt.get('max_drawdown'),
            'accuracy':   acc,
        },
        'report_ready': True,
        'report_path':  str(report_path),
        'message': (f'全工程完了！  PF={bt["pf"]:.3f}  '
                    f'勝率={bt.get("win_rate",0):.1%}  '
                    f'取引数={bt["trades"]}'),
    })
    return bt


if __name__ == '__main__':
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument('--tp', type=float, default=1.5)
    _p.add_argument('--sl', type=float, default=1.0)
    _args = _p.parse_args()
    # tp/sl は run_backtest に渡す (将来拡張用)
    main()
