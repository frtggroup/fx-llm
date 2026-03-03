"""
LLM 学習進捗ダッシュボード
llm_train.py から呼ばれ llm_dashboard.html を更新する
ブラウザで開くと 5 秒ごとに自動リフレッシュ
"""
import json, time
from pathlib import Path
from datetime import datetime

OUT_DIR   = Path(__file__).parent
HTML_PATH = OUT_DIR / 'llm_dashboard.html'


def update(status: dict) -> None:
    """
    status キー:
      phase         : 'loading' | 'tokenizing' | 'training' | 'evaluating' | 'done' | 'error'
      epoch         : 現在エポック (1始まり)
      total_epochs  : 総エポック数
      batch         : 現在バッチ番号
      total_batches : 1エポックのバッチ総数
      train_loss    : float
      val_loss      : float (エポック末のみ)
      accuracy      : float
      gpu_pct       : GPU使用率 (0-100)
      vram_used_gb  : VRAM使用量 GB
      vram_total_gb : VRAM総量 GB
      lr            : 現在学習率
      elapsed_sec   : 経過秒
      eta_sec       : 残り予想秒
      epoch_log     : [{'epoch', 'train_loss', 'val_loss', 'acc', 'elapsed'}, ...]
      best_acc      : 最良 accuracy
      message       : 任意メッセージ
      error         : エラーメッセージ (phase='error' 時)
    """
    html = _build_html(status)
    tmp = HTML_PATH.with_suffix('.tmp')
    tmp.write_text(html, encoding='utf-8')
    tmp.replace(HTML_PATH)


def _fmt(sec: float) -> str:
    if sec < 0 or sec > 86400 * 7:
        return '--:--:--'
    s = int(sec)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_html(st: dict) -> str:
    phase        = st.get('phase', 'loading')
    epoch        = st.get('epoch', 0)
    total_epochs = st.get('total_epochs', 1)
    batch        = st.get('batch', 0)
    total_batches= st.get('total_batches', 1)
    train_loss   = st.get('train_loss', 0.0)
    val_loss     = st.get('val_loss', 0.0)
    acc          = st.get('accuracy', 0.0)
    best_acc     = st.get('best_acc', 0.0)
    gpu_pct      = st.get('gpu_pct', 0)
    vram_used    = st.get('vram_used_gb', 0.0)
    vram_total   = st.get('vram_total_gb', 11.0)
    lr           = st.get('lr', 0.0)
    elapsed      = st.get('elapsed_sec', 0.0)
    eta          = st.get('eta_sec', -1)
    epoch_log    = st.get('epoch_log', [])
    message      = st.get('message', '')
    error        = st.get('error', '')

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # エポック進捗
    ep_pct   = int(epoch / max(total_epochs, 1) * 100)
    # バッチ進捗
    bt_pct   = int(batch / max(total_batches, 1) * 100)
    # GPU カラー
    gpu_color = ('#4caf50' if gpu_pct <= 80 else '#ff9800' if gpu_pct <= 90 else '#f44336')
    # VRAM パーセント
    vram_pct  = int(vram_used / max(vram_total, 1) * 100)

    phase_labels = {
        'loading':    ('モデル読み込み中', '#2196f3'),
        'tokenizing': ('トークナイズ中',   '#9c27b0'),
        'training':   ('訓練中',           '#4caf50'),
        'evaluating': ('評価中',           '#ff9800'),
        'done':       ('完了',             '#009688'),
        'error':      ('エラー',           '#f44336'),
    }
    phase_label, phase_color = phase_labels.get(phase, (phase, '#607d8b'))

    # エポックログテーブル行
    log_rows = ''
    for r in reversed(epoch_log[-10:]):
        ep   = r.get('epoch', '')
        tl   = f"{r.get('train_loss', 0):.4f}"
        vl   = f"{r.get('val_loss',   0):.4f}" if r.get('val_loss') else '-'
        a    = f"{r.get('acc', 0)*100:.2f}%"
        el   = _fmt(r.get('elapsed', 0))
        best = ' ★' if r.get('is_best') else ''
        log_rows += f'<tr><td>{ep}</td><td>{tl}</td><td>{vl}</td><td>{a}{best}</td><td>{el}</td></tr>\n'

    if not log_rows:
        log_rows = '<tr><td colspan="5" style="text-align:center;color:#888">訓練開始待ち</td></tr>'

    # Chart.js 用データ
    batch_log = st.get('batch_log', [])

    # 間引き (最大 600 点)
    if len(batch_log) > 600:
        s = max(1, len(batch_log) // 600)
        sampled = batch_log[::s]
    else:
        sampled = batch_log

    b_labels     = json.dumps([r['step'] for r in sampled])
    b_train_loss = json.dumps([r.get('train_loss') for r in sampled])
    b_val_loss   = json.dumps([r.get('val_loss')   for r in sampled])  # None はそのまま
    b_acc        = json.dumps([r.get('acc')        for r in sampled])  # None はそのまま

    has_chart = 'true' if batch_log else 'false'

    error_block = ''
    if error:
        error_block = f'<div class="error-box"><b>エラー:</b> {error}</div>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>LLM 学習ダッシュボード</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  h1   {{ font-size: 1.4em; margin-bottom: 16px; color: #58a6ff; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .card h2 {{ font-size: 0.85em; color: #8b949e; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
  .big {{ font-size: 2.2em; font-weight: bold; color: #e6edf3; }}
  .sub {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
  .phase-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 12px;
    background: {phase_color}22; color: {phase_color};
    border: 1px solid {phase_color}66; font-weight: bold; font-size: 0.9em;
  }}
  .bar-wrap {{ background: #21262d; border-radius: 6px; height: 14px; overflow: hidden; margin-top: 6px; }}
  .bar      {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
  .label-row {{ display: flex; justify-content: space-between; font-size: 0.78em; color: #8b949e; margin-top: 3px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; text-align: right; }}
  td:first-child, th:first-child {{ text-align: center; }}
  tr:hover {{ background: #21262d; }}
  .msg {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 8px 14px;
           border-radius: 4px; font-size: 0.85em; color: #8b949e; margin-bottom: 12px; }}
  .error-box {{ background: #f4433622; border: 1px solid #f44336; border-radius: 6px;
                 padding: 10px 14px; color: #f44336; margin-bottom: 12px; font-size: 0.9em; }}
  .footer {{ text-align: right; font-size: 0.72em; color: #484f58; margin-top: 16px; }}
  .gpu-bar {{ background: {gpu_color}; }}
  .vram-bar{{ background: #2196f3; }}
  .ep-bar  {{ background: #58a6ff; }}
  .bt-bar  {{ background: #3fb950; }}
</style>
</head>
<body>

<h1>LLM ファインチューニング ダッシュボード
  <span class="phase-badge">{phase_label}</span>
</h1>

{error_block}

<div class="msg">{message if message else '&nbsp;'}</div>

<div class="grid">

  <!-- エポック進捗 -->
  <div class="card">
    <h2>エポック進捗</h2>
    <div class="big">{epoch} <span style="font-size:0.5em;color:#8b949e">/ {total_epochs}</span></div>
    <div class="bar-wrap"><div class="bar ep-bar" style="width:{ep_pct}%"></div></div>
    <div class="label-row"><span>{ep_pct}%</span><span>経過 {_fmt(elapsed)}</span><span>残り {_fmt(eta)}</span></div>
  </div>

  <!-- バッチ進捗 -->
  <div class="card">
    <h2>バッチ進捗 (現エポック)</h2>
    <div class="big">{batch:,} <span style="font-size:0.5em;color:#8b949e">/ {total_batches:,}</span></div>
    <div class="bar-wrap"><div class="bar bt-bar" style="width:{bt_pct}%"></div></div>
    <div class="label-row"><span>{bt_pct}%</span><span>lr = {lr:.2e}</span></div>
  </div>

  <!-- Loss / Accuracy -->
  <div class="card">
    <h2>Loss &amp; Accuracy</h2>
    <div style="display:flex;gap:24px;margin-bottom:8px">
      <div>
        <div style="font-size:0.78em;color:#8b949e">Train Loss</div>
        <div style="font-size:1.6em;font-weight:bold;color:#f0883e">{train_loss:.4f}</div>
      </div>
      <div>
        <div style="font-size:0.78em;color:#8b949e">Val Loss</div>
        <div style="font-size:1.6em;font-weight:bold;color:#79c0ff">{val_loss:.4f}</div>
      </div>
      <div>
        <div style="font-size:0.78em;color:#8b949e">Accuracy</div>
        <div style="font-size:1.6em;font-weight:bold;color:#3fb950">{acc*100:.2f}%</div>
      </div>
    </div>
    <div class="sub">Best Accuracy: <b style="color:#ffa657">{best_acc*100:.2f}%</b></div>
  </div>

  <!-- GPU / VRAM -->
  <div class="card">
    <h2>GPU / VRAM (GTX 1080 Ti)</h2>
    <div style="margin-bottom:10px">
      <div class="label-row" style="margin-bottom:3px"><span>GPU 使用率</span><span style="color:{gpu_color};font-weight:bold">{gpu_pct}%</span></div>
      <div class="bar-wrap"><div class="bar gpu-bar" style="width:{gpu_pct}%"></div></div>
    </div>
    <div>
      <div class="label-row" style="margin-bottom:3px"><span>VRAM</span><span style="color:#79c0ff">{vram_used:.1f} / {vram_total:.1f} GB</span></div>
      <div class="bar-wrap"><div class="bar vram-bar" style="width:{vram_pct}%"></div></div>
    </div>
  </div>

</div>

<!-- グラフ -->
<div class="card" style="margin-bottom:16px">
  <h2>Train Loss / Val Loss / Accuracy (バッチごとリアルタイム)</h2>
  <div id="no-chart" style="color:#8b949e;font-size:0.85em;padding:20px;text-align:center">
    訓練開始後にグラフが表示されます
  </div>
  <div id="chart-wrap" style="display:none">
    <canvas id="lossChart" height="110"></canvas>
  </div>
</div>

<!-- エポックログ -->
<div class="card">
  <h2>エポック履歴</h2>
  <table>
    <thead><tr><th>Ep</th><th>Train Loss</th><th>Val Loss</th><th>Accuracy</th><th>経過時間</th></tr></thead>
    <tbody>{log_rows}</tbody>
  </table>
</div>

<div class="footer">
  最終更新: {now_str} &nbsp;|&nbsp; 5秒ごと自動更新 &nbsp;|&nbsp;
  <span id="hb-badge" style="padding:2px 10px;border-radius:10px;font-size:0.85em;font-weight:bold"></span>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
(function() {{
  // ── ハートビート (HTML の最終更新時刻で稼働判定) ───────────────
  const badge = document.getElementById('hb-badge');
  const lastMod = new Date(document.lastModified);
  const ageSec  = Math.floor((Date.now() - lastMod) / 1000);
  if (ageSec < 30) {{
    badge.textContent = '● 稼働中 (' + ageSec + '秒前更新)';
    badge.style.cssText = 'background:#3fb95022;color:#3fb950;border:1px solid #3fb95066';
  }} else {{
    const m = Math.floor(ageSec / 60), s = ageSec % 60;
    badge.textContent = '■ 停止中 (' + m + '分' + s + '秒前)';
    badge.style.cssText = 'background:#f4433622;color:#f44336;border:1px solid #f4433666';
  }}

  if (!{has_chart}) return;
  document.getElementById('no-chart').style.display = 'none';
  document.getElementById('chart-wrap').style.display = 'block';

  const labels     = {b_labels};
  const trainLoss  = {b_train_loss};
  const valLoss    = {b_val_loss};
  const accData    = {b_acc};

  new Chart(document.getElementById('lossChart').getContext('2d'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{
          label: 'Train Loss',
          data: trainLoss,
          borderColor: '#f0883e',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          tension: 0.3,
          pointRadius: 0,
          yAxisID: 'yLoss',
          spanGaps: true,
        }},
        {{
          label: 'Val Loss',
          data: valLoss,
          borderColor: '#79c0ff',
          backgroundColor: '#79c0ff18',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          yAxisID: 'yLoss',
          spanGaps: false,
        }},
        {{
          label: 'Accuracy %',
          data: accData,
          borderColor: '#3fb950',
          backgroundColor: '#3fb95018',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          yAxisID: 'yAcc',
          spanGaps: false,
        }},
      ]
    }},
    options: {{
      responsive: true,
      animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          labels: {{ color: '#e6edf3', font: {{ size: 11 }}, usePointStyle: true, boxWidth: 12 }}
        }},
        tooltip: {{
          backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
          callbacks: {{
            title: items => 'Step ' + items[0].label,
            label: item => {{
              const v = item.raw;
              if (v === null || v === undefined) return null;
              const suffix = item.datasetIndex === 2 ? '%' : '';
              return ` ${{item.dataset.label}}: ${{v}}${{suffix}}`;
            }}
          }}
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#8b949e', maxTicksLimit: 8 }},
          grid:  {{ color: '#21262d' }},
          title: {{ display: true, text: 'Step', color: '#8b949e' }},
        }},
        yLoss: {{
          type: 'linear', position: 'left',
          ticks: {{ color: '#8b949e' }},
          grid:  {{ color: '#21262d' }},
          title: {{ display: true, text: 'Loss', color: '#8b949e' }},
        }},
        yAcc: {{
          type: 'linear', position: 'right',
          min: 0, max: 100,
          ticks: {{ color: '#3fb950', callback: v => v + '%' }},
          grid:  {{ drawOnChartArea: false }},
          title: {{ display: true, text: 'Accuracy', color: '#3fb950' }},
        }},
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""
    return html


# 初期 HTML を生成しておく
if __name__ == '__main__':
    update({
        'phase': 'loading', 'epoch': 0, 'total_epochs': 5,
        'batch': 0, 'total_batches': 3255,
        'message': 'ダッシュボード初期化完了。llm_train.py を実行してください。',
    })
    print(f"Dashboard: {HTML_PATH}")
