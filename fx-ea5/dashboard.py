"""
学習進捗 HTML ダッシュボード生成モジュール
train.py / run_train.py から呼ばれ dashboard.html を書き換える
ブラウザで開くと自動リフレッシュして最新状態が見える

Docker / Sakura 環境では progress.json も同時に書き出す
→ server.py (FastAPI) が /api/status で返す
"""
import json
import time
from pathlib import Path
from datetime import datetime

OUT_DIR       = Path(__file__).parent
HTML_PATH     = OUT_DIR / 'dashboard.html'
PROGRESS_JSON = OUT_DIR / 'progress.json'


def _write_progress_json(status: dict) -> None:
    """FastAPI ダッシュボードサーバー向けに progress.json をアトミック書き込み"""
    data = {
        'phase':          status.get('phase', 'waiting'),
        'trial':          status.get('trial', 0),
        'total_trials':   status.get('total_trials', 9999),
        'best_pf':        status.get('best_pf', 0.0),
        'target_pf':      status.get('target_pf', 2.0),
        'current_params': status.get('current_params', {}),
        'epoch':          status.get('epoch', 0),
        'total_epochs':   status.get('total_epochs', 800),
        'train_loss':     status.get('train_loss', 0.0),
        'val_loss':       status.get('val_loss', 0.0),
        'accuracy':       status.get('accuracy', 0.0),
        'epoch_log':      status.get('epoch_log', []),
        'trial_results':  status.get('trial_results', []),
        'start_time':     status.get('start_time', time.time()),
        'elapsed_sec':    time.time() - status.get('start_time', time.time()),
        'message':        status.get('message', ''),
        'stop_requested': status.get('stop_requested', False),
        'updated_at':     datetime.now().isoformat(),
    }
    try:
        tmp = PROGRESS_JSON.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(PROGRESS_JSON)
    except Exception:
        pass


def update_dashboard(status: dict) -> None:
    """
    status dict のキー:
      phase          : 'training' | 'trial_done' | 'done'
      trial          : 現在の試行番号 (1始まり)
      total_trials   : 総試行数
      best_pf        : これまでの最良PF
      target_pf      : 目標PF
      current_params : 現在の学習パラメータ dict
      epoch          : 現在エポック
      total_epochs   : 総エポック数
      train_loss     : float
      val_loss       : float
      accuracy       : float
      epoch_log      : [{'epoch', 'train_loss', 'val_loss', 'acc'}, ...]
      trial_results  : [{'trial', 'pf', 'trades', 'win_rate', ...}, ...]
      start_time     : Unix timestamp (float)
      message        : 任意メッセージ
    """
    # progress.json を書く (Docker / FastAPI サーバー向け)
    _write_progress_json(status)
    # dashboard.html を書く (ローカル確認用)
    html = _build_html(status)
    # アトミック書き込み: 一時ファイルに書いてからリネーム → ブラウザが空ページを読まない
    tmp = HTML_PATH.with_suffix('.tmp')
    tmp.write_text(html, encoding='utf-8')
    tmp.replace(HTML_PATH)


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_html(st: dict) -> str:
    phase        = st.get('phase', 'training')
    trial        = st.get('trial', 1)
    total_trials = st.get('total_trials', 40)
    best_pf      = st.get('best_pf', 0.0)
    target_pf    = st.get('target_pf', 1.5)
    epoch        = st.get('epoch', 0)
    total_epochs = st.get('total_epochs', 80)
    train_loss   = st.get('train_loss', 0.0)
    val_loss     = st.get('val_loss', 0.0)
    accuracy     = st.get('accuracy', 0.0)
    epoch_log    = st.get('epoch_log', [])
    trial_results= st.get('trial_results', [])
    params       = st.get('current_params', {})
    start_time   = st.get('start_time', time.time())
    message      = st.get('message', '')

    elapsed      = time.time() - start_time
    elapsed_str  = _fmt_time(elapsed)
    now_str      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    trial_pct    = int(trial / max(total_trials, 1) * 100)
    epoch_pct    = int(epoch / max(total_epochs, 1) * 100)
    pf_pct       = min(int(best_pf / target_pf * 100), 100)

    pf_color     = '#22c55e' if best_pf >= target_pf else ('#f59e0b' if best_pf >= 1.2 else '#ef4444')
    phase_label  = {'training': '学習中', 'trial_done': '試行完了', 'done': '完了！'}.get(phase, phase)
    phase_dot    = 'dot-green' if phase == 'done' else 'dot-blue'

    # エポックチャート用データ
    ep_labels  = json.dumps([x['epoch'] for x in epoch_log])
    ep_train   = json.dumps([round(x['train_loss'], 5) for x in epoch_log])
    ep_val     = json.dumps([round(x['val_loss'], 5) for x in epoch_log])
    ep_acc     = json.dumps([round(x['acc'], 4) for x in epoch_log])

    # 試行PFチャート用データ
    tr_labels  = json.dumps([f"#{x['trial']}" for x in trial_results])
    tr_pf      = json.dumps([round(x.get('pf', 0), 4) for x in trial_results])

    # 試行結果テーブル
    sorted_results = sorted(trial_results, key=lambda x: -x.get('pf', 0))
    table_rows = ''
    for r in sorted_results[:20]:
        pf_v  = r.get('pf', 0)
        badge = 'badge-green' if pf_v >= 1.5 else ('badge-yellow' if pf_v >= 1.2 else 'badge-red')
        star  = '⭐ ' if pf_v == best_pf and best_pf > 0 else ''
        table_rows += f"""
        <tr>
          <td class="td-center">#{r.get('trial','')}</td>
          <td class="td-center"><span class="badge {badge}">{star}{pf_v:.4f}</span></td>
          <td class="td-center">{r.get('trades','')}</td>
          <td class="td-center">{float(r.get('win_rate',0)):.1%}</td>
          <td class="td-center">{r.get('epochs','')}</td>
          <td class="td-center">{r.get('hidden','')}/{r.get('layers','')}</td>
          <td class="td-center">{r.get('tp','')}/{r.get('sl','')}</td>
          <td class="td-center">{r.get('threshold','')}</td>
        </tr>"""

    # パラメータ表示
    param_html = ''.join(
        f'<span class="param-tag">{k}: {v}</span>'
        for k, v in params.items()
        if k not in ('seed',)
    )

    refresh_tag = '' if phase == 'done' else '<meta http-equiv="refresh" content="2">'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  {refresh_tag}
  <title>FX AI EA - 学習ダッシュボード</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', 'Noto Sans JP', sans-serif;
      background: #0f172a; color: #e2e8f0;
      padding: 20px; min-height: 100vh;
    }}
    h1 {{ font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
    .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 20px; }}

    .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }}
    @media (max-width: 900px) {{ .grid-4 {{ grid-template-columns: repeat(2,1fr); }} .grid-2 {{ grid-template-columns: 1fr; }} }}

    .card {{
      background: #1e293b; border-radius: 12px; padding: 18px;
      border: 1px solid #334155;
    }}
    .card-title {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }}
    .card-value {{ font-size: 2rem; font-weight: 700; line-height: 1; }}
    .card-sub {{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }}

    .status-bar {{
      display: flex; align-items: center; gap: 10px;
      background: #1e293b; border-radius: 12px; padding: 14px 18px;
      border: 1px solid #334155; margin-bottom: 20px;
    }}
    .dot-blue {{ width:10px; height:10px; border-radius:50%; background:#3b82f6;
                  box-shadow: 0 0 8px #3b82f6; animation: pulse 1.5s infinite; }}
    .dot-green {{ width:10px; height:10px; border-radius:50%; background:#22c55e; }}
    @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:.4;}} }}
    .status-label {{ font-weight: 600; font-size: 0.95rem; }}
    .status-time {{ margin-left: auto; color: #64748b; font-size: 0.8rem; }}

    .prog-wrap {{ margin-top: 8px; }}
    .prog-label {{ display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:4px; }}
    .prog-bg {{ background:#0f172a; border-radius:999px; height:8px; overflow:hidden; }}
    .prog-fill {{ height:100%; border-radius:999px; transition: width .4s ease; }}
    .prog-blue  {{ background: linear-gradient(90deg,#3b82f6,#818cf8); }}
    .prog-green {{ background: linear-gradient(90deg,#22c55e,#86efac); }}
    .prog-amber {{ background: linear-gradient(90deg,#f59e0b,#fcd34d); }}

    .param-tag {{
      display:inline-block; background:#1e3a5f; color:#93c5fd;
      border-radius:6px; padding:2px 8px; font-size:0.75rem; margin:2px;
    }}

    .chart-card {{ background:#1e293b; border-radius:12px; padding:18px; border:1px solid #334155; }}
    .chart-title {{ font-size:0.85rem; font-weight:600; color:#cbd5e1; margin-bottom:12px; }}
    canvas {{ max-height: 220px; }}

    table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
    th {{ background:#0f172a; color:#64748b; padding:8px 10px; text-align:center;
          font-weight:600; text-transform:uppercase; font-size:0.72rem; letter-spacing:.05em; }}
    tr:nth-child(odd) td {{ background:#243044; }}
    td {{ padding:7px 10px; border-bottom:1px solid #1e293b; }}
    .td-center {{ text-align:center; }}

    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.78rem; }}
    .badge-green {{ background:#14532d; color:#86efac; }}
    .badge-yellow {{ background:#451a03; color:#fcd34d; }}
    .badge-red {{ background:#3f0d0d; color:#fca5a5; }}

    .msg {{ background:#172554; border:1px solid #1d4ed8; border-radius:8px;
            padding:10px 14px; font-size:0.85rem; color:#bfdbfe; margin-bottom:16px; }}

    .refresh-note {{ text-align:right; font-size:0.72rem; color:#334155; margin-top:20px; }}
  </style>
</head>
<body>
  <h1>🤖 FX AI EA &nbsp; 学習ダッシュボード</h1>
  <div class="subtitle">USDJPY M1 &nbsp;|&nbsp; LSTM + CUDA &nbsp;|&nbsp; GTX 1080 Ti &nbsp;|&nbsp; 最終更新: {now_str}</div>

  <div class="status-bar">
    <div class="{phase_dot}"></div>
    <span class="status-label">{phase_label}</span>
    <span style="color:#94a3b8; font-size:0.85rem;">試行 {trial}/{total_trials} &nbsp;|&nbsp; エポック {epoch}/{total_epochs}</span>
    {f'<span class="msg" style="margin:0 0 0 10px; padding:4px 10px;">{message}</span>' if message else ''}
    <span class="status-time">経過: {elapsed_str}</span>
  </div>

  <!-- KPI カード -->
  <div class="grid-4">
    <div class="card">
      <div class="card-title">最良 PF</div>
      <div class="card-value" style="color:{pf_color}">{best_pf:.4f}</div>
      <div class="prog-wrap">
        <div class="prog-label"><span>0</span><span>目標 {target_pf}</span></div>
        <div class="prog-bg"><div class="prog-fill prog-green" style="width:{pf_pct}%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">試行進捗</div>
      <div class="card-value" style="color:#3b82f6">{trial}<span style="font-size:1rem;color:#64748b">/{total_trials}</span></div>
      <div class="prog-wrap">
        <div class="prog-bg"><div class="prog-fill prog-blue" style="width:{trial_pct}%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">エポック進捗</div>
      <div class="card-value" style="color:#818cf8">{epoch}<span style="font-size:1rem;color:#64748b">/{total_epochs}</span></div>
      <div class="prog-wrap">
        <div class="prog-bg"><div class="prog-fill prog-amber" style="width:{epoch_pct}%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Val Loss / Gap / Acc</div>
      <div class="card-value" style="color:#f472b6">{val_loss:.4f}</div>
      <div class="card-sub">
        Gap: <span style="color:{'#ef4444' if val_loss-train_loss > 0.15 else '#22c55e'}">{val_loss-train_loss:+.4f}</span>
        &nbsp; Acc: {accuracy:.1%} &nbsp; Train: {train_loss:.4f}
      </div>
    </div>
  </div>

  <!-- 現在パラメータ -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">現在の学習パラメータ</div>
    <div style="margin-top:6px">{param_html}</div>
  </div>

  <!-- チャート 2列 -->
  <div class="grid-2">
    <div class="chart-card">
      <div class="chart-title">📉 Loss 推移 (現在試行)</div>
      <canvas id="lossChart"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">📊 試行別 PF</div>
      <canvas id="pfChart"></canvas>
    </div>
  </div>

  <!-- 試行結果テーブル -->
  <div class="card">
    <div class="chart-title" style="margin-bottom:12px">🏆 試行結果 (PF降順 上位20件)</div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>#</th><th>PF</th><th>取引数</th><th>勝率</th>
          <th>Epochs</th><th>Hidden/Layers</th><th>TP/SL</th><th>閾値</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="refresh-note">{'自動更新: 2秒' if phase != 'done' else '学習完了 - 自動更新停止'}</div>

  <script>
  const chartDefaults = {{
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#334155' }} }},
    }}
  }};

  // Loss チャート
  new Chart(document.getElementById('lossChart'), {{
    type: 'line',
    data: {{
      labels: {ep_labels},
      datasets: [
        {{ label: 'Train Loss', data: {ep_train}, borderColor: '#3b82f6', backgroundColor: '#3b82f620',
           tension: 0.3, pointRadius: 2, fill: true }},
        {{ label: 'Val Loss',   data: {ep_val},   borderColor: '#f472b6', backgroundColor: '#f472b620',
           tension: 0.3, pointRadius: 2, fill: true }},
      ]
    }},
    options: {{ ...chartDefaults, animation: false,
      plugins: {{ ...chartDefaults.plugins,
        annotation: {{ annotations: {{}} }}
      }}
    }}
  }});

  // PF チャート
  const pfData = {tr_pf};
  new Chart(document.getElementById('pfChart'), {{
    type: 'bar',
    data: {{
      labels: {tr_labels},
      datasets: [{{
        label: 'PF',
        data: pfData,
        backgroundColor: pfData.map(v => v >= 1.5 ? '#22c55e80' : v >= 1.2 ? '#f59e0b80' : '#ef444480'),
        borderColor:     pfData.map(v => v >= 1.5 ? '#22c55e'  : v >= 1.2 ? '#f59e0b'  : '#ef4444'),
        borderWidth: 1,
      }}]
    }},
    options: {{
      ...chartDefaults, animation: false,
      plugins: {{ ...chartDefaults.plugins }},
      scales: {{
        ...chartDefaults.scales,
        y: {{ ...chartDefaults.scales.y,
          min: 0,
          grid: {{ color: '#334155' }},
          ticks: {{ color: '#64748b' }}
        }}
      }}
    }}
  }});
  </script>
</body>
</html>"""
