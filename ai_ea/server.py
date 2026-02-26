"""
FX AI EA ランダムサーチ ダッシュボードサーバー
Sakura DOK / H100 対応  ─  FastAPI  port 7860

エンドポイント:
  GET  /                    → ダッシュボード HTML
  GET  /api/status          → progress.json + GPU 情報
  GET  /api/top10           → TOP10 モデルメタ情報
  POST /api/stop            → 学習停止フラグ
  GET  /download/model/{n}  → rank N の ONNX + norm_params.json (zip)
  GET  /download/results    → all_results.json
  GET  /download/best       → best ONNX + norm_params.json (zip)
  GET  /download/log        → 学習ログ
  GET  /health              → ヘルスチェック
"""
import io, json, os, zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse

# ── パス設定 ─────────────────────────────────────────────────────────────────
WORKSPACE     = Path('/workspace')
AI_EA_DIR     = WORKSPACE / 'ai_ea'
PROGRESS_JSON = AI_EA_DIR / 'progress.json'
ALL_RESULTS   = AI_EA_DIR / 'all_results.json'
TOP10_DIR     = AI_EA_DIR / 'top10'
BEST_ONNX     = AI_EA_DIR / 'fx_model_best.onnx'
BEST_NORM     = AI_EA_DIR / 'norm_params_best.json'
STOP_FLAG     = WORKSPACE / 'stop.flag'
LOG_FILE      = WORKSPACE / 'train_run.log'

app = FastAPI(title="FX AI EA Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── ユーティリティ ─────────────────────────────────────────────────────────────

def _read_progress() -> dict:
    try:
        return json.loads(PROGRESS_JSON.read_text(encoding='utf-8'))
    except Exception:
        return {'phase': 'waiting', 'message': '起動中 / データ待機中...'}


def _gpu_stats() -> dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        n = pynvml.nvmlDeviceGetName(h)
        if isinstance(n, bytes):
            n = n.decode()
        return {
            'gpu_pct':      u.gpu,
            'vram_used_gb': round(m.used  / 1e9, 1),
            'vram_total_gb':round(m.total / 1e9, 1),
            'gpu_name':     n,
        }
    except Exception:
        return {'gpu_pct': 0, 'vram_used_gb': 0, 'vram_total_gb': 80, 'gpu_name': 'H100'}


def _get_top10() -> list:
    try:
        results = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
        valid   = [r for r in results
                   if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
        top10   = sorted(valid, key=lambda x: -x['pf'])[:10]
        for i, r in enumerate(top10):
            rank = i + 1
            r['rank']      = rank
            r['has_model'] = (TOP10_DIR / f'rank_{rank:02d}' / 'fx_model.onnx').exists()
        return top10
    except Exception:
        return []


# ── API エンドポイント ─────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
def index():
    return HTMLResponse(DASHBOARD_HTML)


@app.get('/api/status')
def api_status():
    st = _read_progress()
    st.update(_gpu_stats())
    st['server_time']     = datetime.now().isoformat()
    st['stop_requested']  = STOP_FLAG.exists()
    return JSONResponse(st)


@app.get('/api/top10')
def api_top10():
    return _get_top10()


@app.post('/api/stop')
def api_stop():
    STOP_FLAG.write_text('stop')
    d = _read_progress()
    d['stop_requested'] = True
    d['message'] = '⏹ 停止リクエスト受付 — 現在の試行終了後に停止します'
    try:
        tmp = PROGRESS_JSON.with_suffix('.tmp')
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(PROGRESS_JSON)
    except Exception:
        pass
    return {'ok': True}


@app.get('/download/model/{rank}')
def download_model(rank: int):
    model_dir = TOP10_DIR / f'rank_{rank:02d}'
    if not model_dir.exists():
        raise HTTPException(404, f'rank {rank} のモデルがまだ生成されていません')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(model_dir.iterdir()):
            zf.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename=fx_ea_rank{rank:02d}.zip'})


@app.get('/download/best')
def download_best():
    files = []
    for f in [BEST_ONNX, BEST_NORM]:
        if f.exists():
            files.append(f)
    if not files:
        raise HTTPException(404, 'ベストモデルがまだ生成されていません')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=fx_ea_best.zip'})


@app.get('/download/results')
def download_results():
    if not ALL_RESULTS.exists():
        raise HTTPException(404, '結果ファイルが見つかりません')
    return FileResponse(str(ALL_RESULTS), filename='all_results.json')


@app.get('/download/log')
def download_log():
    if not LOG_FILE.exists():
        raise HTTPException(404, 'ログファイルが見つかりません')
    return FileResponse(str(LOG_FILE), filename='train_run.log')


@app.get('/health')
def health():
    return {'ok': True, 'time': datetime.now().isoformat()}


# ── ダッシュボード HTML ────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX AI EA H100 ダッシュボード</title>
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
.badge-goal {background:#f0883e22;color:#f0883e;border-color:#f0883e66}

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

.param-wrap{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.ptag{background:#1f3a5f;color:#79c0ff;border-radius:4px;padding:2px 8px;font-size:.72em}

table{width:100%;border-collapse:collapse;font-size:.8em}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}
td:first-child,th:first-child{text-align:center}
tr:hover td{background:#1c2128}

.dl-section{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}
.stat-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}
.stat-item{text-align:center}
.stat-val{font-size:1.4em;font-weight:700}
.stat-lbl{font-size:.7em;color:#8b949e}
.footer{text-align:right;font-size:.7em;color:#484f58;margin-top:12px}

#stop-modal{display:none;position:fixed;inset:0;background:#00000088;z-index:999;
  align-items:center;justify-content:center}
#stop-modal.show{display:flex}
#stop-box{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px;
  max-width:400px;width:90%;text-align:center}
#stop-box h3{color:#f85149;margin-bottom:12px;font-size:1.1em}
#stop-box p{color:#8b949e;font-size:.85em;margin-bottom:20px;line-height:1.6}
#stop-box .btn-row{display:flex;gap:10px;justify-content:center}

.rank-1{color:#ffd700;font-weight:700}
.rank-2{color:#c0c0c0;font-weight:700}
.rank-3{color:#cd7f32;font-weight:700}
</style>
</head>
<body>

<div class="header">
  <span class="live-dot" id="dot"></span>
  <h1>FX AI EA H100 ランダムサーチ ダッシュボード</h1>
  <span class="badge badge-wait" id="phase-badge">待機中</span>
</div>

<div class="toolbar">
  <button class="btn btn-red"  onclick="openStopModal()" id="stop-btn">⏹ 学習停止</button>
  <a class="btn btn-gray" href="/download/best"    target="_blank">💾 ベストモデル DL</a>
  <a class="btn btn-gray" href="/download/results" target="_blank">📊 全結果 JSON</a>
  <a class="btn btn-gray" href="/download/log"     target="_blank">📋 ログ</a>
  <span style="flex:1"></span>
  <span style="font-size:.75em;color:#8b949e" id="stop-status"></span>
  <span style="font-size:.75em;color:#8b949e">3秒ポーリング中</span>
</div>

<div class="msg" id="msg">起動中...</div>

<!-- 4列メトリクス -->
<div class="grid4">
  <div class="card">
    <h2>試行進捗</h2>
    <div class="big"><span id="m-trial" style="color:#58a6ff">-</span>
      <span style="font-size:.4em;color:#8b949e"> 試行目</span>
    </div>
    <div class="sub" id="m-elapsed-str">経過: --:--:--</div>
    <div class="bar-wrap"><div id="bar-trial" class="bar" style="background:#58a6ff;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>最良 PF / 目標 2.0</h2>
    <div class="big" id="m-pf" style="color:#ef4444">0.0000</div>
    <div class="sub">目標: 2.0 &nbsp;|&nbsp; <span id="m-pf-pct" style="color:#8b949e">0%</span></div>
    <div class="bar-wrap"><div id="bar-pf" class="bar" style="background:#ef4444;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>現在エポック</h2>
    <div class="big"><span id="m-ep" style="color:#818cf8">-</span>
      <span style="font-size:.4em;color:#8b949e"> / <span id="m-ep-total">-</span></span>
    </div>
    <div class="sub">
      Loss: <span id="m-tloss" style="color:#f0883e">-.----</span>
      / <span id="m-vloss" style="color:#79c0ff">-.----</span>
      &nbsp; Acc: <span id="m-acc" style="color:#3fb950">--%</span>
    </div>
    <div class="bar-wrap"><div id="bar-ep" class="bar" style="background:#818cf8;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>GPU / VRAM (H100)</h2>
    <div class="lrow" style="margin-bottom:2px"><span>GPU使用率</span>
      <span id="m-gpu" style="font-weight:700;color:#3fb950">0%</span></div>
    <div class="bar-wrap"><div id="bar-gpu" class="bar" style="background:#3fb950;width:0%"></div></div>
    <div class="lrow" style="margin:5px 0 2px"><span>VRAM</span>
      <span id="m-vram" style="color:#79c0ff">0 / 80 GB</span></div>
    <div class="bar-wrap"><div id="bar-vram" class="bar" style="background:#2196f3;width:0%"></div></div>
    <div class="sub" id="m-gpu-name" style="margin-top:4px">H100 SXM5</div>
  </div>
</div>

<!-- 現在パラメータ -->
<div class="card" style="margin-bottom:12px">
  <h2>現在の学習パラメータ</h2>
  <div class="param-wrap" id="param-tags">
    <span class="ptag" style="color:#8b949e">待機中...</span>
  </div>
</div>

<!-- チャート 2列 -->
<div class="grid2">
  <div class="card">
    <h2>Loss / Accuracy チャート (現在試行)</h2>
    <div id="chart-ph" style="color:#8b949e;font-size:.82em;padding:20px;text-align:center">
      訓練開始後に表示されます
    </div>
    <div id="chart-wrap" style="display:none;position:relative;height:220px">
      <canvas id="mainChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>試行別 PF チャート (直近50件)</h2>
    <div id="pf-chart-ph" style="color:#8b949e;font-size:.82em;padding:20px;text-align:center">
      試行完了後に表示されます
    </div>
    <div id="pf-chart-wrap" style="display:none;position:relative;height:220px">
      <canvas id="pfChart"></canvas>
    </div>
  </div>
</div>

<!-- TOP 10 テーブル -->
<div class="card" style="margin-bottom:12px">
  <h2>TOP 10 モデル (PF降順) — ダウンロード可能</h2>
  <div style="overflow-x:auto">
    <table id="top10-table">
      <thead>
        <tr>
          <th>Rank</th><th>Trial#</th><th>PF</th><th>取引数</th><th>勝率</th>
          <th>Arch</th><th>Hidden</th><th>Feat#</th><th>TP/SL</th><th>DL</th>
        </tr>
      </thead>
      <tbody id="top10-tbody">
        <tr><td colspan="10" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>
  <div class="dl-section">
    <a class="btn btn-green" href="/download/best" target="_blank">💾 ベストモデル一括DL</a>
    <a class="btn btn-gray"  href="/download/results" target="_blank">📊 全試行結果 JSON</a>
  </div>
</div>

<!-- エポックログ -->
<div class="card" style="margin-bottom:12px">
  <h2>エポック履歴 (現在試行・最新20件)</h2>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr><th>Ep</th><th>Train Loss</th><th>Val Loss</th><th>Gap</th><th>Accuracy</th></tr>
      </thead>
      <tbody id="ep-tbody">
        <tr><td colspan="5" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
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
    <p>現在の試行が終わり次第、<br>ランダムサーチを停止します。<br><br>停止後もダッシュボードは継続表示されます。</p>
    <div class="btn-row">
      <button class="btn btn-red"  onclick="confirmStop()">停止する</button>
      <button class="btn btn-gray" onclick="closeStopModal()">キャンセル</button>
    </div>
  </div>
</div>

<script>
let lossChart   = null;
let pfChart     = null;
let pollCount   = 0;
let errCount    = 0;
let stopReq     = false;
let top10Timer  = 0;

function fmtSec(s) {
  if (s == null || s < 0) return '--:--:--';
  s = Math.floor(s);
  const h = Math.floor(s/3600), m = Math.floor(s%3600/60), ss = s%60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}
function pct(a,b){ return b>0?Math.min(100,Math.round(a/b*100)):0; }

// ─── 停止ボタン ───────────────────────────────────────────────────────────────
function openStopModal()  { document.getElementById('stop-modal').classList.add('show'); }
function closeStopModal() { document.getElementById('stop-modal').classList.remove('show'); }
async function confirmStop() {
  closeStopModal();
  try {
    await fetch('/api/stop', {method:'POST'});
    stopReq = true;
    document.getElementById('stop-btn').disabled    = true;
    document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
    document.getElementById('stop-status').textContent = '停止リクエスト送信済み';
  } catch(e) {
    document.getElementById('stop-status').textContent = '失敗: ' + e.message;
  }
}

// ─── フェーズバッジ ───────────────────────────────────────────────────────────
function applyPhase(phase) {
  const badge = document.getElementById('phase-badge');
  const dot   = document.getElementById('dot');
  const map = {
    training:   ['ランダムサーチ中', 'badge-train', '#3fb950'],
    trial_done: ['試行完了',         'badge-train', '#58a6ff'],
    done:       ['目標達成！',        'badge-goal',  '#f0883e'],
    complete:   ['全工程完了！',      'badge-done',  '#ffa657'],
    error:      ['エラー',            'badge-error', '#f44336'],
    waiting:    ['待機中',            'badge-wait',  '#8b949e'],
  };
  const [label, cls, color] = map[phase] || [phase, 'badge-wait', '#607d8b'];
  badge.textContent    = label;
  badge.className      = 'badge ' + cls;
  dot.style.background = color;
}

// ─── 損失チャート ─────────────────────────────────────────────────────────────
function updateLossChart(epochLog) {
  if (!epochLog || !epochLog.length) return;
  document.getElementById('chart-ph').style.display   = 'none';
  document.getElementById('chart-wrap').style.display = 'block';
  const labels = epochLog.map(r => r.epoch);
  const cfg = {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Train Loss', data: epochLog.map(r=>r.train_loss??null),
          borderColor:'#f0883e', backgroundColor:'transparent',
          borderWidth:1.5, tension:.3, pointRadius:0, yAxisID:'yL', spanGaps:true },
        { label:'Val Loss',   data: epochLog.map(r=>r.val_loss??null),
          borderColor:'#79c0ff', backgroundColor:'#79c0ff18',
          borderWidth:2,   tension:.3, pointRadius:2, yAxisID:'yL', spanGaps:false },
        { label:'Accuracy %', data: epochLog.map(r=>(r.acc??null)*100),
          borderColor:'#3fb950', backgroundColor:'#3fb95018',
          borderWidth:2,   tension:.3, pointRadius:2, yAxisID:'yA', spanGaps:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#e6edf3',font:{size:11},usePointStyle:true,boxWidth:10}},
        tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
          callbacks:{
            title: items => 'Ep ' + items[0].label,
            label: item => item.raw == null ? null :
              ' ' + item.dataset.label + ': ' +
              (item.datasetIndex===2 ? item.raw.toFixed(1)+'%' : item.raw.toFixed(4))
          }
        }
      },
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:10},grid:{color:'#21262d'}},
        yL:{type:'linear',position:'left', ticks:{color:'#8b949e'},grid:{color:'#21262d'},
            title:{display:true,text:'Loss',color:'#8b949e'}},
        yA:{type:'linear',position:'right',min:0,max:100,
            ticks:{color:'#3fb950',callback:v=>v+'%'},grid:{drawOnChartArea:false},
            title:{display:true,text:'Accuracy',color:'#3fb950'}},
      }
    }
  };
  if (lossChart) { lossChart.data = cfg.data; lossChart.update('none'); }
  else { lossChart = new Chart(document.getElementById('mainChart').getContext('2d'), cfg); }
}

// ─── PFチャート ───────────────────────────────────────────────────────────────
function updatePFChart(trialResults) {
  if (!trialResults || !trialResults.length) return;
  document.getElementById('pf-chart-ph').style.display   = 'none';
  document.getElementById('pf-chart-wrap').style.display = 'block';
  const recent = trialResults.slice(-50);
  const labels = recent.map(r => '#'+r.trial);
  const pfs    = recent.map(r => r.pf ?? 0);
  const bgColors  = pfs.map(v => v>=2.0?'#f0883e80':v>=1.5?'#3fb95080':v>=1.2?'#ffa65780':'#58a6ff40');
  const bdrColors = pfs.map(v => v>=2.0?'#f0883e'  :v>=1.5?'#3fb950'  :v>=1.2?'#ffa657'  :'#58a6ff');
  const cfg = {
    type: 'bar',
    data: {
      labels,
      datasets:[{
        label:'PF', data:pfs,
        backgroundColor:bgColors, borderColor:bdrColors, borderWidth:1,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{
        legend:{display:false},
        tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
          callbacks:{
            title: items => '試行 ' + items[0].label,
            label: item => ` PF: ${item.raw.toFixed(4)}`
          }
        },
        annotation: { annotations: {
          target: { type:'line', yMin:2.0, yMax:2.0,
            borderColor:'#f0883e', borderWidth:2, borderDash:[6,4],
            label:{display:true,content:'目標 2.0',color:'#f0883e',font:{size:11}} }
        }}
      },
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:15},grid:{color:'#21262d'},
           maxBarThickness:20},
        y:{min:0, ticks:{color:'#8b949e'},grid:{color:'#21262d'},
           title:{display:true,text:'Profit Factor',color:'#8b949e'}},
      }
    }
  };
  if (pfChart) { pfChart.data = cfg.data; pfChart.update('none'); }
  else { pfChart = new Chart(document.getElementById('pfChart').getContext('2d'), cfg); }
}

// ─── エポックテーブル ─────────────────────────────────────────────────────────
function updateEpTable(log) {
  if (!log || !log.length) return;
  const tbody = document.getElementById('ep-tbody');
  tbody.innerHTML = [...log].slice(-20).reverse().map(r => {
    const gap   = (r.val_loss - r.train_loss).toFixed(4);
    const gapC  = (r.val_loss - r.train_loss) > 0.15 ? '#f85149' : '#3fb950';
    const accPct= ((r.acc??0)*100).toFixed(2);
    const accC  = (r.acc??0)>=0.45?'#3fb950':(r.acc??0)>=0.35?'#ffa657':'#f0883e';
    return `<tr>
      <td>${r.epoch}</td>
      <td style="color:#f0883e">${r.train_loss?.toFixed(4)??'-'}</td>
      <td style="color:#79c0ff">${r.val_loss?.toFixed(4)??'-'}</td>
      <td style="color:${gapC}">${gap}</td>
      <td style="color:${accC}">${accPct}%</td>
    </tr>`;
  }).join('');
}

// ─── TOP 10 テーブル ──────────────────────────────────────────────────────────
async function updateTop10() {
  try {
    const res = await fetch('/api/top10', {cache:'no-store'});
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('top10-tbody');
    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#8b949e">まだ有効な試行がありません (取引数200以上 PF>0)</td></tr>';
      return;
    }
    tbody.innerHTML = data.map(r => {
      const rankCls = r.rank===1?'rank-1':r.rank===2?'rank-2':r.rank===3?'rank-3':'';
      const pfColor = r.pf>=2.0?'#f0883e':r.pf>=1.5?'#3fb950':r.pf>=1.2?'#ffa657':'#79c0ff';
      const dlBtn   = r.has_model
        ? `<a class="btn btn-green" href="/download/model/${r.rank}" target="_blank">📥 DL</a>`
        : `<span style="color:#8b949e;font-size:.75em">生成中...</span>`;
      return `<tr>
        <td class="${rankCls}">${r.rank===1?'🥇':r.rank===2?'🥈':r.rank===3?'🥉':'#'+r.rank}</td>
        <td style="color:#8b949e">#${r.trial??'-'}</td>
        <td style="color:${pfColor};font-weight:700">${(r.pf??0).toFixed(4)}</td>
        <td style="color:#e6edf3">${r.trades??'-'}</td>
        <td style="color:#3fb950">${((r.win_rate??0)*100).toFixed(1)}%</td>
        <td style="color:#79c0ff">${r.arch??'-'}</td>
        <td>${r.hidden??'-'}×${r.layers??'-'}</td>
        <td>${r.n_features??'-'}</td>
        <td style="color:#8b949e">${r.tp??'-'}/${r.sl??'-'}</td>
        <td>${dlBtn}</td>
      </tr>`;
    }).join('');
  } catch(e) { /* silent */ }
}

// ─── メインポーリング ─────────────────────────────────────────────────────────
async function poll() {
  try {
    const res = await fetch('/api/status', {cache:'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    pollCount++;
    document.getElementById('poll-cnt').textContent = pollCount;
    document.getElementById('last-upd').textContent = new Date().toLocaleTimeString('ja-JP');

    applyPhase(d.phase ?? 'waiting');
    document.getElementById('msg').textContent = d.message ?? '';

    if (d.stop_requested && !stopReq) {
      stopReq = true;
      document.getElementById('stop-btn').disabled    = true;
      document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
      document.getElementById('stop-status').textContent = '停止リクエスト受付済み';
    }

    // 試行進捗
    const trial = d.trial ?? 0;
    document.getElementById('m-trial').textContent = trial;
    const trialBar = Math.min(100, (trial / 9999) * 100);
    document.getElementById('bar-trial').style.width = trialBar + '%';
    document.getElementById('m-elapsed-str').textContent = '経過: ' + fmtSec(d.elapsed_sec);

    // 最良 PF
    const bpf   = d.best_pf ?? 0;
    const pfPct = Math.min(100, Math.round(bpf / 2.0 * 100));
    const pfColor = bpf >= 2.0 ? '#f0883e' : bpf >= 1.5 ? '#3fb950' : bpf >= 1.2 ? '#ffa657' : '#f85149';
    document.getElementById('m-pf').textContent       = bpf.toFixed(4);
    document.getElementById('m-pf').style.color       = pfColor;
    document.getElementById('m-pf-pct').textContent   = pfPct + '%';
    document.getElementById('bar-pf').style.width     = pfPct + '%';
    document.getElementById('bar-pf').style.background = pfColor;

    // 現在エポック
    const ep  = d.epoch ?? 0;
    const tot = d.total_epochs ?? 800;
    const epP = pct(ep, tot);
    document.getElementById('m-ep').textContent       = ep;
    document.getElementById('m-ep-total').textContent = tot;
    document.getElementById('bar-ep').style.width     = epP + '%';
    const tl = d.train_loss ?? 0, vl = d.val_loss ?? 0, acc = d.accuracy ?? 0;
    document.getElementById('m-tloss').textContent = tl.toFixed(4);
    document.getElementById('m-vloss').textContent = vl.toFixed(4);
    const accColor = acc>=0.45?'#3fb950':acc>=0.35?'#ffa657':'#f0883e';
    document.getElementById('m-acc').textContent   = (acc*100).toFixed(1) + '%';
    document.getElementById('m-acc').style.color   = accColor;

    // GPU
    const gpuP  = d.gpu_pct  ?? 0;
    const vramU = d.vram_used_gb  ?? 0;
    const vramT = d.vram_total_gb ?? 80;
    const vramP = pct(vramU, vramT);
    const gpuC  = gpuP>90?'#f44336':gpuP>75?'#3fb950':'#58a6ff';
    document.getElementById('m-gpu').textContent         = gpuP + '%';
    document.getElementById('m-gpu').style.color         = gpuC;
    document.getElementById('bar-gpu').style.width       = gpuP + '%';
    document.getElementById('bar-gpu').style.background  = gpuC;
    document.getElementById('m-vram').textContent        = `${vramU.toFixed(1)} / ${vramT.toFixed(0)} GB`;
    document.getElementById('bar-vram').style.width      = vramP + '%';
    document.getElementById('m-gpu-name').textContent    = d.gpu_name ?? 'H100 SXM5';

    // パラメータタグ
    const p = d.current_params ?? {};
    const skipKeys = new Set(['seed','timeframe','label_type','start_time','total_trials']);
    const tags = Object.entries(p)
      .filter(([k]) => !skipKeys.has(k))
      .map(([k,v]) => `<span class="ptag">${k}: ${v}</span>`)
      .join('');
    document.getElementById('param-tags').innerHTML = tags || '<span class="ptag" style="color:#8b949e">待機中...</span>';

    // チャート
    if (d.epoch_log)    updateLossChart(d.epoch_log);
    if (d.trial_results) updatePFChart(d.trial_results);
    if (d.epoch_log)    updateEpTable(d.epoch_log);

  } catch(e) {
    errCount++;
    document.getElementById('err-cnt').textContent = errCount;
    document.getElementById('msg').textContent = '⚠ 取得エラー: ' + e.message;
  }

  // TOP 10 は 10 秒ごとに更新
  top10Timer += 3;
  if (top10Timer >= 10) {
    top10Timer = 0;
    updateTop10();
  }
}

// 初回即実行
poll();
updateTop10();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860, log_level='info')
