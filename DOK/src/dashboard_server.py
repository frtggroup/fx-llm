"""
FX LLM ファインチューニング ダッシュボード
Flask サーバー — 外部公開 (0.0.0.0:7860)

エンドポイント:
  GET /              → リアルタイムダッシュボード HTML
  GET /api/status    → progress.json をそのまま返す JSON
  GET /api/logs      → epoch_log + batch_log
  GET /download/report → バックテストHTMLレポートダウンロード
  GET /download/adapter → アダプター tar.gz ダウンロード
"""
import json, os, time, tarfile, io
from pathlib import Path
from flask import Flask, jsonify, send_file, Response, make_response
from flask_cors import CORS

WORKSPACE     = Path('/workspace')
PROGRESS_JSON = WORKSPACE / 'progress.json'
REPORT_DIR    = WORKSPACE / 'reports'
ADAPTER_DIR   = WORKSPACE / 'output' / 'llm_adapter_best'

app = Flask(__name__)
CORS(app)


def read_progress() -> dict:
    try:
        return json.loads(PROGRESS_JSON.read_text())
    except Exception:
        return {'phase': 'waiting', 'message': '起動中 / データ待機中...'}


# ──────────────────────────────────────────────────────────────────────────────
# メインダッシュボード HTML (SSE ポーリング方式)
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX LLM H100 ダッシュボード</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:16px;min-height:100vh}
h1{font-size:1.3em;margin-bottom:14px;color:#58a6ff;display:flex;align-items:center;gap:10px}
.badge{display:inline-block;padding:3px 12px;border-radius:10px;font-size:0.78em;font-weight:bold;border:1px solid}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card h2{font-size:0.72em;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.8px}
.big{font-size:2em;font-weight:700;color:#e6edf3}
.sub{font-size:0.75em;color:#8b949e;margin-top:3px}
.bar-wrap{background:#21262d;border-radius:5px;height:12px;overflow:hidden;margin-top:6px}
.bar{height:100%;border-radius:5px;transition:width .4s}
.lrow{display:flex;justify-content:space-between;font-size:0.73em;color:#8b949e;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:0.8em}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e}td:first-child,th:first-child{text-align:center}
tr:hover{background:#21262d}
.msg{background:#161b22;border-left:3px solid #58a6ff;padding:8px 14px;border-radius:4px;
     font-size:0.82em;color:#8b949e;margin-bottom:10px;min-height:28px}
.btn{display:inline-block;padding:8px 18px;border-radius:6px;font-size:0.85em;font-weight:600;
     cursor:pointer;text-decoration:none;border:none;margin:4px 4px 4px 0}
.btn-blue{background:#1f6feb;color:#fff}.btn-blue:hover{background:#388bfd}
.btn-green{background:#238636;color:#fff}.btn-green:hover{background:#2ea043}
.btn-gray{background:#21262d;color:#e6edf3;border:1px solid #30363d}.btn-gray:hover{background:#30363d}
.dl-section{margin-top:10px;padding-top:10px;border-top:1px solid #30363d}
.stat-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:8px}
.stat-item{text-align:center}
.stat-val{font-size:1.5em;font-weight:700}
.stat-lbl{font-size:0.7em;color:#8b949e}
#live-dot{width:8px;height:8px;border-radius:50%;background:#3fb950;display:inline-block;
          animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<h1>
  <span id="live-dot"></span>
  FX LLM H100 ファインチューニング ダッシュボード
  <span id="phase-badge" class="badge" style="background:#58a6ff22;color:#58a6ff;border-color:#58a6ff66">待機中</span>
</h1>

<div id="msg" class="msg">起動中...</div>

<!-- メトリクス 4列 -->
<div class="grid4" id="metrics-grid">
  <div class="card"><h2>エポック</h2>
    <div class="big"><span id="m-epoch">0</span><span style="font-size:.45em;color:#8b949e"> / <span id="m-total-ep">-</span></span></div>
    <div class="bar-wrap"><div id="bar-ep" class="bar" style="background:#58a6ff;width:0%"></div></div>
    <div class="lrow"><span id="m-ep-pct">0%</span><span>経過 <span id="m-elapsed">--:--:--</span></span><span>残り <span id="m-eta">--:--:--</span></span></div>
  </div>
  <div class="card"><h2>Train Loss</h2>
    <div class="big" id="m-tloss" style="color:#f0883e">-.----</div>
    <div class="sub">Val Loss: <span id="m-vloss" style="color:#79c0ff">-.----</span></div>
    <div class="sub">LR: <span id="m-lr">-.--e-00</span></div>
  </div>
  <div class="card"><h2>Accuracy</h2>
    <div class="big" id="m-acc" style="color:#3fb950">--%</div>
    <div class="sub">Best: <span id="m-best" style="color:#ffa657">--%</span></div>
  </div>
  <div class="card"><h2>GPU / VRAM (H100)</h2>
    <div class="lrow" style="margin-bottom:3px"><span>GPU</span><span id="m-gpu-pct" style="color:#3fb950;font-weight:700">0%</span></div>
    <div class="bar-wrap"><div id="bar-gpu" class="bar" style="background:#3fb950;width:0%"></div></div>
    <div class="lrow" style="margin:5px 0 3px"><span>VRAM</span><span id="m-vram" style="color:#79c0ff">0 / 80 GB</span></div>
    <div class="bar-wrap"><div id="bar-vram" class="bar" style="background:#2196f3;width:0%"></div></div>
  </div>
</div>

<!-- グラフ -->
<div class="card" style="margin-bottom:12px">
  <h2>Loss / Accuracy チャート (リアルタイム)</h2>
  <div id="chart-placeholder" style="color:#8b949e;font-size:0.82em;padding:20px;text-align:center">訓練開始後に表示されます</div>
  <div id="chart-wrap" style="display:none;position:relative;height:220px">
    <canvas id="lossChart"></canvas>
  </div>
</div>

<!-- エポックログ + バックテスト並列 -->
<div class="grid2">
  <div class="card">
    <h2>エポック履歴</h2>
    <table>
      <thead><tr><th>Ep</th><th>Train</th><th>Val</th><th>Acc</th><th>時間</th></tr></thead>
      <tbody id="epoch-tbody"><tr><td colspan="5" style="text-align:center;color:#8b949e">待機中</td></tr></tbody>
    </table>
  </div>
  <div class="card" id="bt-card">
    <h2>バックテスト結果</h2>
    <div id="bt-waiting" style="color:#8b949e;font-size:0.82em;padding:10px">訓練完了後に実行されます</div>
    <div id="bt-results" style="display:none">
      <div class="stat-row" id="bt-stats"></div>
      <div class="dl-section">
        <a id="btn-report" class="btn btn-green" href="/download/report" target="_blank">📊 HTMLレポートDL</a>
        <a id="btn-adapter" class="btn btn-blue" href="/download/adapter" target="_blank">💾 モデルDL</a>
      </div>
    </div>
  </div>
</div>

<!-- MT5 セクション -->
<div class="card" style="margin-top:12px" id="mt5-card">
  <h2>MT5 バックテスト連携</h2>
  <div id="mt5-waiting" style="color:#8b949e;font-size:0.82em;padding:10px">
    訓練・バックテスト完了後にシグナルCSVが自動生成されます
  </div>
  <div id="mt5-ready" style="display:none">
    <div class="stat-row" id="mt5-stats"></div>
    <div style="font-size:0.78em;color:#8b949e;margin:8px 0 12px">
      ① CSVをMT5の <code style="color:#79c0ff">MQL5/Files/</code> フォルダにコピー<br>
      ② EAをコンパイルして同フォルダの任意のチャートに適用<br>
      ③ MT5ストラテジーテスターで <b>USDJPY H1</b> 期間を合わせてバックテスト実行
    </div>
    <a class="btn btn-green" href="/download/mt5signals" target="_blank">📥 シグナルCSVダウンロード</a>
    <a class="btn btn-blue"  href="/download/mt5ea"      target="_blank">🤖 MT5 EAダウンロード (.mq5)</a>
  </div>
</div>

<div style="text-align:right;font-size:0.7em;color:#484f58;margin-top:10px">
  最終更新: <span id="last-update">-</span> &nbsp;|&nbsp; 3秒ポーリング
</div>

<script>
let lossChart = null;

function fmt(sec) {
  if (sec < 0 || sec > 604800) return '--:--:--';
  const s = Math.floor(sec), h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
  return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(ss).padStart(2,'0');
}

function phaseStyle(ph) {
  const m = {
    loading:   ['モデル読込中', '#2196f3'],
    tokenizing:['トークナイズ中','#9c27b0'],
    training:  ['訓練中',       '#3fb950'],
    evaluating:['評価中',       '#ff9800'],
    done:      ['訓練完了',     '#009688'],
    backtest:  ['バックテスト実行中','#ff9800'],
    complete:  ['全工程完了！', '#ffa657'],
    error:     ['エラー',       '#f44336'],
    waiting:   ['待機中',       '#8b949e'],
  };
  return m[ph] || [ph, '#607d8b'];
}

function pct(a, b) { return b > 0 ? Math.min(100, Math.round(a/b*100)) : 0; }

function updateChart(batchLog) {
  if (!batchLog || batchLog.length === 0) return;
  document.getElementById('chart-placeholder').style.display = 'none';
  document.getElementById('chart-wrap').style.display = 'block';

  // 間引き (最大 800点)
  let sampled = batchLog;
  if (batchLog.length > 800) {
    const step = Math.ceil(batchLog.length / 800);
    sampled = batchLog.filter((_,i) => i % step === 0);
  }
  const labels    = sampled.map(r => r.step);
  const trainLoss = sampled.map(r => r.train_loss ?? null);
  const valLoss   = sampled.map(r => r.val_loss   ?? null);
  const accData   = sampled.map(r => r.acc        ?? null);

  const cfg = {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Train Loss', data:trainLoss, borderColor:'#f0883e',
          backgroundColor:'transparent', borderWidth:1.5, tension:.3,
          pointRadius:0, yAxisID:'yLoss', spanGaps:true },
        { label:'Val Loss', data:valLoss, borderColor:'#79c0ff',
          backgroundColor:'#79c0ff14', borderWidth:2, tension:.3,
          pointRadius:3, pointHoverRadius:6, yAxisID:'yLoss', spanGaps:false },
        { label:'Accuracy %', data:accData, borderColor:'#3fb950',
          backgroundColor:'#3fb95014', borderWidth:2, tension:.3,
          pointRadius:3, pointHoverRadius:6, yAxisID:'yAcc', spanGaps:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#e6edf3',font:{size:11},usePointStyle:true,boxWidth:10}},
        tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
          callbacks:{
            title: items => 'Step '+items[0].label,
            label: item => {
              if(item.raw==null) return null;
              return ' '+item.dataset.label+': '+(item.datasetIndex===2 ? item.raw+'%' : item.raw);
            }
          }
        },
      },
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'},
           title:{display:true,text:'Step',color:'#8b949e'}},
        yLoss:{type:'linear',position:'left',ticks:{color:'#8b949e'},grid:{color:'#21262d'},
               title:{display:true,text:'Loss',color:'#8b949e'}},
        yAcc:{type:'linear',position:'right',min:0,max:100,
              ticks:{color:'#3fb950',callback:v=>v+'%'},grid:{drawOnChartArea:false},
              title:{display:true,text:'Accuracy',color:'#3fb950'}},
      }
    }
  };

  if (lossChart) {
    lossChart.data = cfg.data;
    lossChart.update('none');
  } else {
    lossChart = new Chart(document.getElementById('lossChart').getContext('2d'), cfg);
  }
}

function updateEpochTable(log) {
  if (!log || log.length === 0) return;
  const tbody = document.getElementById('epoch-tbody');
  tbody.innerHTML = [...log].reverse().slice(0,15).map(r => {
    const acc = r.acc != null ? (r.acc*100).toFixed(2)+'%' : '-';
    const best = r.is_best ? ' ★' : '';
    return `<tr>
      <td>${r.epoch}</td>
      <td>${r.train_loss?.toFixed(4)??'-'}</td>
      <td>${r.val_loss?.toFixed(4)??'-'}</td>
      <td style="color:${r.is_best?'#ffa657':'inherit'}">${acc}${best}</td>
      <td>${fmt(r.elapsed??0)}</td>
    </tr>`;
  }).join('');
}

function updateBacktest(bt) {
  if (!bt) return;
  document.getElementById('bt-waiting').style.display = 'none';
  document.getElementById('bt-results').style.display = 'block';
  const stats = document.getElementById('bt-stats');
  const items = [
    ['PF',      bt.pf?.toFixed(3)??'-',    '#ffa657'],
    ['勝率',    bt.win_rate!=null?(bt.win_rate*100).toFixed(1)+'%':'-', '#3fb950'],
    ['取引数',  bt.trades??'-',             '#79c0ff'],
    ['純損益',  bt.net_pnl?.toFixed(4)??'-','#e6edf3'],
    ['分類精度',bt.accuracy!=null?(bt.accuracy*100).toFixed(2)+'%':'-','#3fb950'],
  ];
  stats.innerHTML = items.map(([l,v,c]) =>
    `<div class="stat-item"><div class="stat-val" style="color:${c}">${v}</div><div class="stat-lbl">${l}</div></div>`
  ).join('');
}

async function poll() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();

    // フェーズバッジ
    const [phLabel, phColor] = phaseStyle(d.phase);
    const badge = document.getElementById('phase-badge');
    badge.textContent = phLabel;
    badge.style.cssText = `background:${phColor}22;color:${phColor};border-color:${phColor}66`;

    // ライブドット色
    const dot = document.getElementById('live-dot');
    dot.style.background = d.phase === 'error' ? '#f44336' :
                           d.phase === 'complete' ? '#ffa657' : '#3fb950';

    // メッセージ
    document.getElementById('msg').textContent = d.message || '';

    // エポック
    document.getElementById('m-epoch').textContent    = d.epoch ?? 0;
    document.getElementById('m-total-ep').textContent = d.total_epochs ?? '-';
    const epP = pct(d.epoch??0, d.total_epochs??1);
    document.getElementById('m-ep-pct').textContent   = epP+'%';
    document.getElementById('bar-ep').style.width     = epP+'%';
    document.getElementById('m-elapsed').textContent  = fmt(d.elapsed_sec??-1);
    document.getElementById('m-eta').textContent      = fmt(d.eta_sec??-1);

    // Loss
    document.getElementById('m-tloss').textContent = d.train_loss?.toFixed(4) ?? '-.----';
    document.getElementById('m-vloss').textContent = d.val_loss?.toFixed(4)   ?? '-.----';
    document.getElementById('m-lr').textContent    = d.lr != null ? d.lr.toExponential(2) : '-.--e-00';

    // Accuracy
    const accV = d.accuracy ?? 0;
    const bestV= d.best_acc ?? 0;
    document.getElementById('m-acc').textContent  = (accV*100).toFixed(2)+'%';
    document.getElementById('m-best').textContent = (bestV*100).toFixed(2)+'%';

    // GPU/VRAM
    const gpuP  = Math.min(100, d.gpu_pct ?? 0);
    const vramU = d.vram_used_gb  ?? 0;
    const vramT = d.vram_total_gb ?? 80;
    const vramP = pct(vramU, vramT);
    document.getElementById('m-gpu-pct').textContent = gpuP+'%';
    document.getElementById('m-gpu-pct').style.color = gpuP>90?'#f44336':gpuP>75?'#ff9800':'#3fb950';
    document.getElementById('bar-gpu').style.width   = gpuP+'%';
    document.getElementById('bar-gpu').style.background = gpuP>90?'#f44336':gpuP>75?'#ff9800':'#3fb950';
    document.getElementById('m-vram').textContent    = `${vramU.toFixed(1)} / ${vramT.toFixed(0)} GB`;
    document.getElementById('bar-vram').style.width  = vramP+'%';

    // グラフ
    if (d.batch_log) updateChart(d.batch_log);

    // エポックログ
    if (d.epoch_log) updateEpochTable(d.epoch_log);

    // バックテスト
    if (d.backtest_result) updateBacktest(d.backtest_result);

    // MT5シグナル
    if (d.mt5_stats) updateMT5(d.mt5_stats);

    document.getElementById('last-update').textContent = new Date().toLocaleTimeString('ja-JP');
  } catch(e) {
    console.warn('poll error', e);
  }
}

function updateMT5(st) {
  if (!st) return;
  document.getElementById('mt5-waiting').style.display = 'none';
  document.getElementById('mt5-ready').style.display   = 'block';
  const items = [
    ['総シグナル', st.total ?? '-',                             '#e6edf3'],
    ['BUY',        st.buy   ?? '-',                             '#3fb950'],
    ['SELL',       st.sell  ?? '-',                             '#f85149'],
    ['HOLD',       st.hold  ?? '-',                             '#8b949e'],
    ['平均信頼度', st.avg_conf != null ? (st.avg_conf*100).toFixed(1)+'%' : '-', '#ffa657'],
  ];
  document.getElementById('mt5-stats').innerHTML = items.map(([l,v,c]) =>
    `<div class="stat-item"><div class="stat-val" style="color:${c};font-size:1.1em">${v}</div><div class="stat-lbl">${l}</div></div>`
  ).join('');
}

poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return DASHBOARD_HTML


@app.route('/api/status')
def api_status():
    return jsonify(read_progress())


@app.route('/api/logs')
def api_logs():
    d = read_progress()
    return jsonify({
        'epoch_log': d.get('epoch_log', []),
        'batch_log': d.get('batch_log', []),
    })


@app.route('/download/report')
def download_report():
    html_files = sorted(REPORT_DIR.glob('backtest_report_*.html'))
    if not html_files:
        return Response('レポートがまだ生成されていません。訓練完了後に自動生成されます。',
                        mimetype='text/plain; charset=utf-8', status=404)
    latest = html_files[-1]
    return send_file(
        str(latest),
        mimetype='text/html',
        as_attachment=True,
        download_name=latest.name,
    )


@app.route('/download/adapter')
def download_adapter():
    if not ADAPTER_DIR.exists():
        return Response('モデルアダプターがまだ生成されていません。',
                        mimetype='text/plain; charset=utf-8', status=404)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        tar.add(str(ADAPTER_DIR), arcname='llm_adapter_best')
    buf.seek(0)
    resp = make_response(buf.read())
    resp.headers['Content-Type'] = 'application/gzip'
    resp.headers['Content-Disposition'] = 'attachment; filename=llm_adapter_best.tar.gz'
    return resp


@app.route('/download/mt5signals')
def download_mt5signals():
    csv_files = sorted(REPORT_DIR.glob('mt5_signals_*.csv'))
    if not csv_files:
        return Response('MT5シグナルCSVがまだ生成されていません。訓練完了後に自動生成されます。',
                        mimetype='text/plain; charset=utf-8', status=404)
    latest = csv_files[-1]
    return send_file(
        str(latest),
        mimetype='text/csv',
        as_attachment=True,
        download_name=latest.name,
    )


@app.route('/download/mt5ea')
def download_mt5ea():
    ea_path = WORKSPACE / 'mql5' / 'LLM_Signal_EA.mq5'
    if not ea_path.exists():
        return Response('MT5 EAファイルが見つかりません。',
                        mimetype='text/plain; charset=utf-8', status=404)
    return send_file(
        str(ea_path),
        mimetype='text/plain',
        as_attachment=True,
        download_name='LLM_Signal_EA.mq5',
    )


def run():
    print(f"  ダッシュボード起動: http://0.0.0.0:7860", flush=True)
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)


if __name__ == '__main__':
    run()
