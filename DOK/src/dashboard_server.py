"""
FX LLM ファインチューニング ダッシュボード v2
Flask サーバー — 外部公開 (0.0.0.0:7860)

エンドポイント:
  GET  /              → リアルタイムダッシュボード HTML (local_monitor と同一UI)
  GET  /api/status    → progress.json をそのまま返す JSON
  GET  /api/logs      → epoch_log + batch_log
  POST /api/stop      → 学習停止フラグファイルを生成
  GET  /download/report    → バックテスト HTML レポート
  GET  /download/adapter   → アダプター tar.gz
  GET  /download/mt5signals → MT5 シグナル CSV
  GET  /download/mt5ea     → MT5 EA .mq5
"""
import json, os, time, tarfile, io
from pathlib import Path
from flask import Flask, jsonify, send_file, Response, make_response, request
from flask_cors import CORS

WORKSPACE     = Path('/workspace')
PROGRESS_JSON = WORKSPACE / 'progress.json'
REPORT_DIR    = WORKSPACE / 'reports'
ADAPTER_DIR   = WORKSPACE / 'output' / 'llm_adapter_best'
STOP_FLAG     = WORKSPACE / 'stop.flag'

app = Flask(__name__)
CORS(app)


def read_progress() -> dict:
    try:
        return json.loads(PROGRESS_JSON.read_text())
    except Exception:
        return {'phase': 'waiting', 'message': '起動中 / データ待機中...'}


# ──────────────────────────────────────────────────────────────────────────────
# ダッシュボード HTML — local_monitor.html と同一 UI
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
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:16px}

.header{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.header h1{font-size:1.25em;color:#58a6ff;flex:1}
.live-dot{width:9px;height:9px;border-radius:50%;background:#3fb950;animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.badge{padding:3px 12px;border-radius:12px;font-size:.78em;font-weight:700;border:1px solid}
.badge-train{background:#3fb95022;color:#3fb950;border-color:#3fb95066}
.badge-done {background:#ffa65722;color:#ffa657;border-color:#ffa65766}
.badge-error{background:#f4433622;color:#f44336;border-color:#f4433666}
.badge-wait {background:#8b949e22;color:#8b949e;border-color:#8b949e66}

.toolbar{display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.btn{display:inline-block;padding:6px 14px;border-radius:6px;font-size:.82em;font-weight:600;
  cursor:pointer;border:none;text-decoration:none}
.btn-blue {background:#1f6feb;color:#fff}.btn-blue:hover {background:#388bfd}
.btn-green{background:#238636;color:#fff}.btn-green:hover{background:#2ea043}
.btn-red  {background:#b91c1c;color:#fff}.btn-red:hover  {background:#dc2626}
.btn-gray {background:#21262d;color:#e6edf3;border:1px solid #30363d}.btn-gray:hover{background:#30363d}
.btn-orange{background:#9a3412;color:#fff}.btn-orange:hover{background:#c2410c}

.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
@media(max-width:900px){.grid4{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}

.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card h2{font-size:.7em;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.8px}
.big{font-size:2em;font-weight:700}
.sub{font-size:.75em;color:#8b949e;margin-top:3px}
.bar-wrap{background:#21262d;border-radius:4px;height:10px;overflow:hidden;margin-top:6px}
.bar{height:100%;border-radius:4px;transition:width .5s}
.lrow{display:flex;justify-content:space-between;font-size:.72em;color:#8b949e;margin-top:3px}
.msg{background:#161b22;border-left:3px solid #58a6ff;padding:8px 14px;border-radius:4px;
  font-size:.82em;color:#c9d1d9;margin-bottom:12px;min-height:28px}

table{width:100%;border-collapse:collapse;font-size:.8em}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}td:first-child,th:first-child{text-align:center}
tr:hover td{background:#1c2128}

.dl-section{margin-top:12px;padding-top:10px;border-top:1px solid #30363d;display:flex;gap:8px;flex-wrap:wrap}
.stat-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px}
.stat-item{text-align:center}
.stat-val{font-size:1.4em;font-weight:700}
.stat-lbl{font-size:.7em;color:#8b949e}
.footer{text-align:right;font-size:.7em;color:#484f58;margin-top:12px}

/* 停止確認ダイアログ */
#stop-modal{display:none;position:fixed;inset:0;background:#00000088;z-index:999;
  align-items:center;justify-content:center}
#stop-modal.show{display:flex}
#stop-box{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px;
  max-width:400px;width:90%;text-align:center}
#stop-box h3{color:#f85149;margin-bottom:12px;font-size:1.1em}
#stop-box p{color:#8b949e;font-size:.85em;margin-bottom:20px;line-height:1.6}
#stop-box .btn-row{display:flex;gap:10px;justify-content:center}
</style>
</head>
<body>

<div class="header">
  <span class="live-dot" id="dot"></span>
  <h1>FX LLM H100 ダッシュボード</h1>
  <span class="badge badge-wait" id="phase-badge">待機中</span>
</div>

<div class="toolbar">
  <button class="btn btn-red" onclick="openStopModal()" id="stop-btn">⏹ 学習停止</button>
  <span style="font-size:.75em;color:#8b949e" id="stop-status"></span>
  <span style="flex:1"></span>
  <span style="font-size:.75em;color:#8b949e">3秒ポーリング中</span>
</div>

<div class="msg" id="msg">起動中...</div>

<!-- メトリクス 4列 -->
<div class="grid4">
  <div class="card">
    <h2>全体進捗</h2>
    <div class="big"><span id="m-ep">-</span><span style="font-size:.4em;color:#8b949e"> / <span id="m-ep-total">-</span> ep</span></div>
    <div style="font-size:.72em;color:#8b949e;margin:3px 0 2px">全体</div>
    <div class="bar-wrap"><div id="bar-ep" class="bar" style="background:#58a6ff;width:0%"></div></div>
    <div class="lrow"><span id="m-ep-pct">0%</span><span>経過 <span id="m-elapsed">--:--:--</span></span><span>残り <span id="m-eta">--:--:--</span></span></div>
    <!-- エポック内バッチ進捗 -->
    <div style="font-size:.72em;color:#8b949e;margin:6px 0 2px">
      エポック内 <span id="m-bat-cur">-</span> / <span id="m-bat-total">-</span> batch
      &nbsp;<span id="m-bat-pct" style="color:#58a6ff;font-weight:700">0%</span>
    </div>
    <div class="bar-wrap"><div id="bar-bat" class="bar" style="background:#388bfd;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>Loss</h2>
    <div class="big" id="m-tloss" style="color:#f0883e">-.----</div>
    <div class="sub">Val Loss: <span id="m-vloss" style="color:#79c0ff">-.----</span></div>
    <div class="sub">LR: <span id="m-lr">-</span></div>
  </div>

  <div class="card">
    <h2>Accuracy</h2>
    <div class="big" id="m-acc" style="color:#3fb950">--%</div>
    <div class="sub">★ Best: <span id="m-best" style="color:#ffa657">--%</span></div>
    <div class="bar-wrap"><div id="bar-acc" class="bar" style="background:#3fb950;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>GPU / VRAM (H100)</h2>
    <div class="lrow" style="margin-bottom:2px"><span>GPU使用率</span><span id="m-gpu" style="font-weight:700">0%</span></div>
    <div class="bar-wrap"><div id="bar-gpu" class="bar" style="background:#3fb950;width:0%"></div></div>
    <div class="lrow" style="margin:5px 0 2px"><span>VRAM</span><span id="m-vram" style="color:#79c0ff">0 / 80 GB</span></div>
    <div class="bar-wrap"><div id="bar-vram" class="bar" style="background:#2196f3;width:0%"></div></div>
  </div>
</div>

<!-- チャート -->
<div class="card" style="margin-bottom:12px">
  <h2>Loss / Accuracy リアルタイムチャート</h2>
  <div id="chart-ph" style="color:#8b949e;font-size:.82em;padding:20px;text-align:center">訓練開始後に表示されます</div>
  <div id="chart-wrap" style="display:none;position:relative;height:240px">
    <canvas id="mainChart"></canvas>
  </div>
</div>

<!-- 中間評価ポイント -->
<div class="card" style="margin-bottom:12px">
  <h2>中間評価ポイント（100step毎）— 最新20件</h2>
  <div style="overflow-x:auto">
    <table id="eval-table">
      <thead>
        <tr><th>Step</th><th>Ep</th><th>Train Loss</th><th>Val Loss</th><th>Accuracy</th><th>Δ Acc</th></tr>
      </thead>
      <tbody id="eval-tbody">
        <tr><td colspan="6" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- エポックログ + バックテスト -->
<div class="grid2">
  <div class="card">
    <h2>エポック履歴</h2>
    <table>
      <thead><tr><th>Ep</th><th>Train Loss</th><th>Val Loss</th><th>Accuracy</th><th>時間</th></tr></thead>
      <tbody id="ep-tbody">
        <tr><td colspan="5" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card" id="bt-card">
    <h2>バックテスト結果</h2>
    <div id="bt-wait" style="color:#8b949e;font-size:.82em;padding:8px">訓練完了後に自動実行されます</div>
    <div id="bt-result" style="display:none">
      <div class="stat-row" id="bt-stats"></div>
      <div class="dl-section" id="dl-links"></div>
    </div>
  </div>
</div>

<!-- MT5 セクション -->
<div class="card" style="margin-top:12px" id="mt5-card">
  <h2>MT5 バックテスト連携</h2>
  <div id="mt5-waiting" style="color:#8b949e;font-size:.82em;padding:10px">訓練・バックテスト完了後にシグナルCSVが自動生成されます</div>
  <div id="mt5-ready" style="display:none">
    <div class="stat-row" id="mt5-stats"></div>
    <div style="font-size:.78em;color:#8b949e;margin:8px 0 12px">
      ① CSVをMT5の <code style="color:#79c0ff">MQL5/Files/</code> フォルダにコピー<br>
      ② EAをコンパイルして同フォルダの任意のチャートに適用<br>
      ③ MT5ストラテジーテスターで <b>USDJPY H1</b> 期間を合わせてバックテスト実行
    </div>
    <a class="btn btn-green" href="/download/mt5signals" target="_blank">📥 シグナルCSVダウンロード</a>
    <a class="btn btn-blue"  href="/download/mt5ea"      target="_blank">🤖 MT5 EAダウンロード (.mq5)</a>
  </div>
</div>

<div class="footer">
  最終更新: <span id="last-upd">-</span> &nbsp;|&nbsp;
  取得回数: <span id="poll-cnt">0</span> &nbsp;|&nbsp;
  エラー: <span id="err-cnt">0</span>
</div>

<!-- 停止確認ダイアログ -->
<div id="stop-modal">
  <div id="stop-box">
    <h3>⏹ 学習を停止しますか？</h3>
    <p>現在のエポックが終わり次第、<br>学習を停止して<b>バックテスト・モデル保存</b>を自動実行します。<br><br>停止後も再開はできません。</p>
    <div class="btn-row">
      <button class="btn btn-red"  onclick="confirmStop()">停止する</button>
      <button class="btn btn-gray" onclick="closeStopModal()">キャンセル</button>
    </div>
  </div>
</div>

<script>
let chart     = null;
let pollCount = 0;
let errCount  = 0;
let stopRequested = false;

function fmtSec(s) {
  if (s == null || s < 0 || s > 604800) return '--:--:--';
  s = Math.floor(s);
  const h = Math.floor(s/3600), m = Math.floor(s%3600/60), ss = s%60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}
function pct(a, b) { return b > 0 ? Math.min(100, Math.round(a/b*100)) : 0; }

// ─── 停止ボタン ─────────────────────────────────────────────────────────────
function openStopModal()  { document.getElementById('stop-modal').classList.add('show'); }
function closeStopModal() { document.getElementById('stop-modal').classList.remove('show'); }

async function confirmStop() {
  closeStopModal();
  try {
    const res = await fetch('/api/stop', { method: 'POST' });
    const d   = await res.json();
    stopRequested = true;
    document.getElementById('stop-status').textContent = '停止リクエスト送信済み — 現エポック終了後に停止します';
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
  } catch(e) {
    document.getElementById('stop-status').textContent = '停止リクエスト失敗: ' + e.message;
  }
}

// ─── フェーズバッジ ──────────────────────────────────────────────────────────
function applyPhase(phase) {
  const badge = document.getElementById('phase-badge');
  const dot   = document.getElementById('dot');
  const map = {
    loading:    ['モデル読込中',    'badge-train', '#2196f3'],
    tokenizing: ['トークナイズ中',  'badge-train', '#9c27b0'],
    training:   ['訓練中',          'badge-train', '#3fb950'],
    evaluating: ['評価中',          'badge-train', '#ff9800'],
    stopping:   ['停止処理中',      'badge-train', '#f85149'],
    done:       ['訓練完了',        'badge-done',  '#009688'],
    backtest:   ['バックテスト中',  'badge-train', '#ff9800'],
    complete:   ['全工程完了！',    'badge-done',  '#ffa657'],
    error:      ['エラー',          'badge-error', '#f44336'],
    waiting:    ['待機中',          'badge-wait',  '#8b949e'],
  };
  const [label, cls, color] = map[phase] || [phase, 'badge-wait', '#607d8b'];
  badge.textContent = label;
  badge.className   = 'badge ' + cls;
  dot.style.background = color;
}

// ─── チャート ────────────────────────────────────────────────────────────────
function updateChart(batchLog) {
  if (!batchLog || !batchLog.length) return;
  document.getElementById('chart-ph').style.display   = 'none';
  document.getElementById('chart-wrap').style.display = 'block';
  let data = batchLog;
  if (data.length > 1000) {
    const step = Math.ceil(data.length / 1000);
    data = data.filter((_, i) => i % step === 0);
  }
  const cfg = {
    type: 'line',
    data: {
      labels: data.map(r => r.step),
      datasets: [
        { label:'Train Loss', data:data.map(r=>r.train_loss??null),
          borderColor:'#f0883e', backgroundColor:'transparent',
          borderWidth:1.5, tension:.3, pointRadius:0, yAxisID:'yL', spanGaps:true },
        { label:'Val Loss', data:data.map(r=>r.val_loss??null),
          borderColor:'#79c0ff', backgroundColor:'#79c0ff18',
          borderWidth:2, tension:.3, pointRadius:3, pointHoverRadius:6, yAxisID:'yL', spanGaps:false },
        { label:'Accuracy %', data:data.map(r=>r.acc??null),
          borderColor:'#3fb950', backgroundColor:'#3fb95018',
          borderWidth:2, tension:.3, pointRadius:3, pointHoverRadius:6, yAxisID:'yA', spanGaps:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#e6edf3',font:{size:11},usePointStyle:true,boxWidth:10}},
        tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
          callbacks:{
            title: items => 'Step ' + items[0].label,
            label: item => item.raw == null ? null :
              ' ' + item.dataset.label + ': ' +
              (item.datasetIndex === 2 ? item.raw + '%' : item.raw.toFixed(4))
          }
        },
      },
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:10},grid:{color:'#21262d'}},
        yL:{type:'linear',position:'left',ticks:{color:'#8b949e'},grid:{color:'#21262d'},
            title:{display:true,text:'Loss',color:'#8b949e'}},
        yA:{type:'linear',position:'right',min:0,max:100,
            ticks:{color:'#3fb950',callback:v=>v+'%'},grid:{drawOnChartArea:false},
            title:{display:true,text:'Accuracy',color:'#3fb950'}},
      }
    }
  };
  if (chart) { chart.data = cfg.data; chart.update('none'); }
  else { chart = new Chart(document.getElementById('mainChart').getContext('2d'), cfg); }
}

// ─── 中間評価テーブル ────────────────────────────────────────────────────────
function updateEvalTable(batchLog, totalBatches) {
  if (!batchLog || !batchLog.length) return;
  const evalPts = batchLog.filter(r => r.acc != null);
  if (!evalPts.length) return;
  const recent = [...evalPts].slice(-20).reverse();
  const tbody  = document.getElementById('eval-tbody');
  tbody.innerHTML = recent.map((r, i) => {
    const epNum = totalBatches > 0 ? Math.floor((r.step-1)/totalBatches)+1 : '-';
    const next  = recent[i+1];
    let deltaHtml = '-';
    if (next && next.acc != null) {
      const d = r.acc - next.acc;
      const c = d>0?'#3fb950':d<0?'#f85149':'#8b949e';
      deltaHtml = `<span style="color:${c}">${d>0?'+':''}${d.toFixed(1)}%</span>`;
    }
    const acc    = r.acc ?? 0;
    const aColor = acc>=50?'#3fb950':acc>=40?'#ffa657':acc>=33?'#e6edf3':'#f85149';
    const vl     = r.val_loss ?? 9;
    const vlColor= vl<1.0?'#3fb950':vl<1.2?'#79c0ff':vl<1.5?'#ffa657':'#f0883e';
    return `<tr style="${i===0?'background:#1c2128':''}">
      <td style="color:#8b949e">${r.step}</td>
      <td style="color:#58a6ff">${epNum}</td>
      <td style="color:#f0883e">${r.train_loss?.toFixed(4)??'-'}</td>
      <td style="color:${vlColor}">${r.val_loss?.toFixed(4)??'-'}</td>
      <td style="color:${aColor};font-weight:${acc>=40?'700':'400'}">${acc.toFixed(1)}%</td>
      <td>${deltaHtml}</td>
    </tr>`;
  }).join('');
}

// ─── エポックテーブル ────────────────────────────────────────────────────────
function updateEpochTable(log) {
  if (!log || !log.length) return;
  const tbody = document.getElementById('ep-tbody');
  tbody.innerHTML = [...log].reverse().map(r => {
    const acc  = r.acc != null ? (r.acc*100).toFixed(2)+'%' : (r.acc??'-')+'%';
    const best = r.is_best ? ' ★' : '';
    return `<tr>
      <td>${r.epoch}</td>
      <td style="color:#f0883e">${r.train_loss?.toFixed(4)??'-'}</td>
      <td style="color:#79c0ff">${r.val_loss?.toFixed(4)??'-'}</td>
      <td style="color:${r.is_best?'#ffa657':'#3fb950'}">${acc}${best}</td>
      <td>${fmtSec(r.elapsed)}</td>
    </tr>`;
  }).join('');
}

// ─── バックテスト ────────────────────────────────────────────────────────────
function updateBacktest(bt) {
  if (!bt) return;
  document.getElementById('bt-wait').style.display   = 'none';
  document.getElementById('bt-result').style.display = 'block';
  const items = [
    ['PF',    bt.pf?.toFixed(3)??'-',                               '#ffa657'],
    ['勝率',  bt.win_rate!=null?(bt.win_rate*100).toFixed(1)+'%':'-','#3fb950'],
    ['取引数',bt.trades??'-',                                         '#79c0ff'],
    ['純損益',bt.net_pnl?.toFixed(4)??'-',                           '#e6edf3'],
    ['精度',  bt.accuracy!=null?(bt.accuracy*100).toFixed(2)+'%':'-','#3fb950'],
  ];
  document.getElementById('bt-stats').innerHTML = items.map(([l,v,c]) =>
    `<div class="stat-item"><div class="stat-val" style="color:${c}">${v}</div><div class="stat-lbl">${l}</div></div>`
  ).join('');
  document.getElementById('dl-links').innerHTML = `
    <a class="btn btn-green" href="/download/report"     target="_blank">📊 HTMLレポート</a>
    <a class="btn btn-blue"  href="/download/adapter"    target="_blank">💾 モデルDL</a>
    <a class="btn btn-gray"  href="/download/mt5signals" target="_blank">📥 MT5シグナル</a>
    <a class="btn btn-gray"  href="/download/mt5ea"      target="_blank">🤖 MT5 EA</a>
  `;
}

// ─── MT5 ─────────────────────────────────────────────────────────────────────
function updateMT5(st) {
  if (!st) return;
  document.getElementById('mt5-waiting').style.display = 'none';
  document.getElementById('mt5-ready').style.display   = 'block';
  const items = [
    ['総シグナル', st.total??'-',                              '#e6edf3'],
    ['BUY',        st.buy??'-',                                '#3fb950'],
    ['SELL',       st.sell??'-',                               '#f85149'],
    ['HOLD',       st.hold??'-',                               '#8b949e'],
    ['平均信頼度', st.avg_conf!=null?(st.avg_conf*100).toFixed(1)+'%':'-','#ffa657'],
  ];
  document.getElementById('mt5-stats').innerHTML = items.map(([l,v,c]) =>
    `<div class="stat-item"><div class="stat-val" style="color:${c};font-size:1.1em">${v}</div><div class="stat-lbl">${l}</div></div>`
  ).join('');
}

// ─── ポーリング ──────────────────────────────────────────────────────────────
async function poll() {
  try {
    const res = await fetch('/api/status', { cache:'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    pollCount++;
    document.getElementById('poll-cnt').textContent = pollCount;
    document.getElementById('last-upd').textContent = new Date().toLocaleTimeString('ja-JP');

    applyPhase(d.phase ?? 'waiting');
    document.getElementById('msg').textContent = d.message ?? '';

    // 停止フラグ確認
    if (d.stop_requested && !stopRequested) {
      stopRequested = true;
      document.getElementById('stop-btn').disabled = true;
      document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
      document.getElementById('stop-status').textContent = '停止リクエスト受付済み';
    }

    const ep  = d.epoch ?? 0;
    const tot = d.total_epochs ?? 10;
    const bat = d.batch ?? 0;
    const totB= d.total_batches ?? 1;
    const epGlobal = (ep-1 + bat/totB) / tot;
    const epP = Math.min(100, Math.round(epGlobal * 100));
    document.getElementById('m-ep').textContent       = ep;
    document.getElementById('m-ep-total').textContent = tot;
    document.getElementById('m-ep-pct').textContent   = epP + '%';
    document.getElementById('bar-ep').style.width     = epP + '%';
    document.getElementById('m-elapsed').textContent  = fmtSec(d.elapsed_sec);
    document.getElementById('m-eta').textContent      = fmtSec(d.eta_sec);

    const batP = Math.min(100, Math.round(bat/totB*100));
    document.getElementById('m-bat-cur').textContent   = bat.toLocaleString();
    document.getElementById('m-bat-total').textContent = totB.toLocaleString();
    document.getElementById('m-bat-pct').textContent   = batP + '%';
    document.getElementById('bar-bat').style.width     = batP + '%';
    const batColor = batP>80?'#3fb950':batP>50?'#388bfd':'#58a6ff';
    document.getElementById('bar-bat').style.background  = batColor;
    document.getElementById('m-bat-pct').style.color     = batColor;

    document.getElementById('m-tloss').textContent = d.train_loss?.toFixed(4) ?? '-.----';
    document.getElementById('m-vloss').textContent = d.val_loss?.toFixed(4)   ?? '-.----';
    document.getElementById('m-lr').textContent    = d.lr!=null ? d.lr.toExponential(2) : '-';

    const accV  = (d.accuracy  ?? 0) * 100;
    const bestV = (d.best_acc  ?? 0) * 100;
    document.getElementById('m-acc').textContent  = accV.toFixed(2)  + '%';
    document.getElementById('m-best').textContent = bestV.toFixed(2) + '%';
    document.getElementById('bar-acc').style.width = Math.min(100, accV) + '%';
    const accColor = accV>=50?'#3fb950':accV>=35?'#ffa657':'#f0883e';
    document.getElementById('m-acc').style.color        = accColor;
    document.getElementById('bar-acc').style.background = accColor;

    const gpuP  = d.gpu_pct  ?? 0;
    const vramU = d.vram_used_gb  ?? 0;
    const vramT = d.vram_total_gb ?? 80;
    const vramP = pct(vramU, vramT);
    const gpuColor = gpuP>90?'#f44336':gpuP>75?'#ff9800':'#3fb950';
    document.getElementById('m-gpu').textContent        = gpuP + '%';
    document.getElementById('m-gpu').style.color        = gpuColor;
    document.getElementById('bar-gpu').style.width      = gpuP + '%';
    document.getElementById('bar-gpu').style.background = gpuColor;
    document.getElementById('m-vram').textContent       = `${vramU.toFixed(1)} / ${vramT.toFixed(0)} GB`;
    document.getElementById('bar-vram').style.width     = vramP + '%';

    if (d.batch_log) updateChart(d.batch_log);
    if (d.batch_log) updateEvalTable(d.batch_log, d.total_batches ?? 1628);
    if (d.epoch_log) updateEpochTable(d.epoch_log);
    if (d.backtest_result) updateBacktest(d.backtest_result);
    if (d.mt5_stats)       updateMT5(d.mt5_stats);
  } catch(e) {
    errCount++;
    document.getElementById('err-cnt').textContent = errCount;
    document.getElementById('msg').textContent = '⚠ 取得エラー: ' + e.message;
  }
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


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """学習停止フラグを立てる。train_h100.py がエポック終了後に検知して停止する。"""
    try:
        STOP_FLAG.write_text('stop')
        # progress.json にも記録
        d = read_progress()
        d['stop_requested'] = True
        d['message'] = '⏹ 停止リクエスト受付 — 現エポック終了後に停止してバックテスト実行します'
        PROGRESS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return jsonify({'ok': True, 'message': '停止フラグを設定しました'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/download/report')
def download_report():
    html_files = sorted(REPORT_DIR.glob('backtest_report_*.html'))
    if not html_files:
        return Response('レポートがまだ生成されていません。',
                        mimetype='text/plain; charset=utf-8', status=404)
    latest = html_files[-1]
    return send_file(str(latest), mimetype='text/html',
                     as_attachment=True, download_name=latest.name)


@app.route('/download/adapter')
def download_adapter():
    # v2 アダプターが存在すればそちらを優先
    for cand in [WORKSPACE/'output'/'llm_adapter_best_v2', ADAPTER_DIR]:
        if cand.exists():
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w:gz') as tar:
                tar.add(str(cand), arcname=cand.name)
            buf.seek(0)
            resp = make_response(buf.read())
            resp.headers['Content-Type'] = 'application/gzip'
            resp.headers['Content-Disposition'] = f'attachment; filename={cand.name}.tar.gz'
            return resp
    return Response('モデルアダプターがまだ生成されていません。',
                    mimetype='text/plain; charset=utf-8', status=404)


@app.route('/download/mt5signals')
def download_mt5signals():
    csv_files = sorted(REPORT_DIR.glob('mt5_signals_*.csv'))
    if not csv_files:
        return Response('MT5シグナルCSVがまだ生成されていません。',
                        mimetype='text/plain; charset=utf-8', status=404)
    latest = csv_files[-1]
    return send_file(str(latest), mimetype='text/csv',
                     as_attachment=True, download_name=latest.name)


@app.route('/download/mt5ea')
def download_mt5ea():
    ea_path = WORKSPACE / 'mql5' / 'LLM_Signal_EA.mq5'
    if not ea_path.exists():
        return Response('MT5 EAファイルが見つかりません。',
                        mimetype='text/plain; charset=utf-8', status=404)
    return send_file(str(ea_path), mimetype='text/plain',
                     as_attachment=True, download_name='LLM_Signal_EA.mq5')


def run():
    print(f"  ダッシュボード起動: http://0.0.0.0:7860", flush=True)
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)


if __name__ == '__main__':
    run()
