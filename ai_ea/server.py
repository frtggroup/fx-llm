"""
FX AI EA 並列ランダムサーチ ダッシュボードサーバー v2
GTX 1080 Ti / ローカル対応  ─  FastAPI  port 8080

エンドポイント:
  GET  /                      → ダッシュボード HTML
  GET  /api/status            → progress.json + GPU 情報
  GET  /api/top100            → TOP100 モデルメタ情報
  POST /api/stop              → 学習停止フラグ
  GET  /report/<trial_no>     → バックテスト資産曲線 HTML
  GET  /download/model/<rank> → rank N の ONNX + norm_params.json (zip)
  GET  /download/results      → all_results.json
  GET  /download/best         → best ONNX + norm_params.json (zip)
  GET  /download/log          → 学習ログ
  GET  /health                → ヘルスチェック
"""
import io, json, os, threading, zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse, PlainTextResponse

WORKSPACE     = Path('/workspace')
AI_EA_DIR     = WORKSPACE / 'ai_ea'
PROGRESS_JSON = AI_EA_DIR / 'progress.json'
ALL_RESULTS   = AI_EA_DIR / 'all_results.json'
TOP_DIR       = AI_EA_DIR / 'top100'
TRIALS_DIR    = AI_EA_DIR / 'trials'
BEST_ONNX     = AI_EA_DIR / 'fx_model_best.onnx'
BEST_NORM     = AI_EA_DIR / 'norm_params_best.json'
STOP_FLAG     = WORKSPACE / 'stop.flag'
LOG_FILE      = WORKSPACE / 'train_run.log'

app = FastAPI(title="FX AI EA Dashboard v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event('startup')
def _startup_backfill():
    """起動時にバックグラウンドでバックフィルを実行"""
    def _run():
        import time as _time
        _time.sleep(5)          # run_train.py のデータ復元を待つ
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'backfill_top100', str(AI_EA_DIR / 'backfill_top100.py'))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        except Exception as e:
            print(f'[startup_backfill] エラー: {e}')
    threading.Thread(target=_run, daemon=True).start()


def _read_progress() -> dict:
    try:
        return json.loads(PROGRESS_JSON.read_text(encoding='utf-8'))
    except Exception:
        return {'phase': 'waiting', 'message': '起動中...', 'completed_count': 0}


def _gpu_stats() -> dict:
    try:
        from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
                            nvmlDeviceGetName)
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        u = nvmlDeviceGetUtilizationRates(h)
        m = nvmlDeviceGetMemoryInfo(h)
        n = nvmlDeviceGetName(h)
        if isinstance(n, bytes):
            n = n.decode()
        return {
            'gpu_pct':      u.gpu,
            'vram_used_gb': round(m.used  / 1e9, 1),
            'vram_total_gb':round(m.total / 1e9, 1),
            'gpu_name':     n,
        }
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                prop  = torch.cuda.get_device_properties(0)
                total = round(prop.total_memory / 1e9, 1)
                used  = round((torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / 1e9, 1)
                name  = prop.name
                return {'gpu_pct': 0, 'vram_used_gb': used, 'vram_total_gb': total, 'gpu_name': name}
        except Exception:
            pass
        return {'gpu_pct': 0, 'vram_used_gb': 0, 'vram_total_gb': 11, 'gpu_name': 'GTX 1080 Ti'}


def _find_model_dir(rank: int, trial_no: int) -> Path | None:
    """rank_XXX → TRIALS_DIR/trial_XXXXXX の順でモデルディレクトリを探す"""
    rank_dir = TOP_DIR / f'rank_{rank:03d}'
    if (rank_dir / 'fx_model.onnx').exists():
        return rank_dir
    # TRIALS_DIR にフォールバック (rebuild_top_n がキャッシュミスした場合)
    trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
    if (trial_dir / 'fx_model.onnx').exists():
        return trial_dir
    return None


def _get_top_n(n: int = 100) -> list:
    try:
        results = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
        valid   = [r for r in results
                   if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
        top     = sorted(valid, key=lambda x: -x['pf'])[:n]
        for i, r in enumerate(top):
            rank     = i + 1
            trial_no = r.get('trial', 0)
            r['rank']       = rank
            rank_dir        = TOP_DIR / f'rank_{rank:03d}'
            model_dir       = _find_model_dir(rank, trial_no)
            r['has_model']  = model_dir is not None
            r['has_report'] = (rank_dir / 'report.html').exists() or (
                model_dir is not None and (model_dir / 'report.html').exists())
            # 特徴量重要度: all_results.json → rank_dir/result.json の順に取得
            imp = r.get('feature_importance')
            if not imp:
                for res_f in [rank_dir / 'result.json',
                              TRIALS_DIR / f'trial_{trial_no:06d}' / 'last_result.json']:
                    if res_f.exists():
                        try:
                            rd = json.loads(res_f.read_text(encoding='utf-8'))
                            imp = rd.get('feature_importance', [])
                            if imp:
                                break
                        except Exception:
                            pass
            r['feature_importance'] = imp or []
        return top
    except Exception:
        return []


@app.get('/', response_class=HTMLResponse)
def index():
    return HTMLResponse(DASHBOARD_HTML)


@app.get('/api/status')
def api_status():
    st = _read_progress()
    gpu = _gpu_stats()
    # progress.json にない場合だけ上書き
    for k, v in gpu.items():
        st.setdefault(k, v)
    st['server_time']    = datetime.now().isoformat()
    st['stop_requested'] = STOP_FLAG.exists()
    return JSONResponse(st)


@app.get('/api/top100')
def api_top100():
    return _get_top_n(100)


@app.post('/api/backfill')
def api_backfill():
    """不足データ（feature_importance 等）をバックグラウンドで補完する"""
    def _run():
        try:
            import importlib.util, sys as _sys
            spec = importlib.util.spec_from_file_location(
                'backfill_top100', str(AI_EA_DIR / 'backfill_top100.py'))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        except Exception as e:
            print(f'[backfill] エラー: {e}')
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {'ok': True, 'message': 'バックフィルをバックグラウンドで開始しました'}


@app.post('/api/stop')
def api_stop():
    STOP_FLAG.write_text('stop')
    d = _read_progress()
    d['stop_requested'] = True
    d['message'] = '⏹ 停止リクエスト受付 — 実行中の試行終了後に停止します'
    try:
        tmp = PROGRESS_JSON.with_suffix('.tmp')
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(PROGRESS_JSON)
    except Exception:
        pass
    return {'ok': True}


@app.get('/report/{trial_no}')
def get_report(trial_no: int):
    # TOP_DIR からレポートを探す
    for rank_dir in sorted(TOP_DIR.glob('rank_*')):
        res_f = rank_dir / 'result.json'
        if res_f.exists():
            try:
                r = json.loads(res_f.read_text(encoding='utf-8'))
                if r.get('trial') == trial_no:
                    rp = rank_dir / 'report.html'
                    if rp.exists():
                        return HTMLResponse(rp.read_text(encoding='utf-8'))
            except Exception:
                pass
    # trials ディレクトリからも探す
    trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
    rp = trial_dir / 'report.html'
    if rp.exists():
        return HTMLResponse(rp.read_text(encoding='utf-8'))
    raise HTTPException(404, f'試行 #{trial_no} のレポートがまだ生成されていません')


@app.get('/download/model/{rank}')
def download_model(rank: int):
    # top100 テーブルから trial_no を逆引き
    trial_no = 0
    try:
        top = _get_top_n(100)
        for r in top:
            if r.get('rank') == rank:
                trial_no = r.get('trial', 0)
                break
    except Exception:
        pass
    model_dir = _find_model_dir(rank, trial_no)
    if model_dir is None:
        raise HTTPException(404, f'rank {rank} のモデルがまだ生成されていません (ONNX未出力 or PF<1.2)')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(model_dir.iterdir()):
            if f.suffix in ('.onnx', '.json'):
                zf.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename=fx_ea_rank{rank:03d}.zip'})


@app.get('/download/best')
def download_best():
    files = [f for f in [BEST_ONNX, BEST_NORM] if f.exists()]
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


@app.get('/download/checkpoint')
def download_checkpoint():
    """チェックポイント一式 (all_results + best model + top100) を zip でダウンロード"""
    ckpt = WORKSPACE / 'data' / 'checkpoint'
    if not ckpt.exists() or not (ckpt / 'all_results.json').exists():
        raise HTTPException(404, 'チェックポイントがまだ作成されていません (10分ごとに自動保存)')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in ckpt.rglob('*'):
            if f.is_file():
                zf.write(f, f.relative_to(ckpt))
    buf.seek(0)
    fname = f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    return StreamingResponse(buf, media_type='application/zip',
                             headers={'Content-Disposition': f'attachment; filename={fname}'})


@app.get('/api/checkpoint_status')
def checkpoint_status():
    """チェックポイントのメタ情報を返す"""
    meta_path = WORKSPACE / 'data' / 'checkpoint' / 'meta.json'
    if not meta_path.exists():
        return {'exists': False}
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        return {'exists': True, **meta}
    except Exception:
        return {'exists': False}


@app.get('/api/trial_log/{trial_no}')
def trial_log(trial_no: int, lines: int = 100):
    """試行ログの末尾 lines 行を返す"""
    log_path = TRIALS_DIR / f'trial_{trial_no:06d}' / 'train.log'
    if not log_path.exists():
        raise HTTPException(404, f'試行#{trial_no} のログが見つかりません')
    text = log_path.read_text(encoding='utf-8', errors='replace')
    tail = '\n'.join(text.splitlines()[-lines:])
    return PlainTextResponse(tail)


@app.get('/health')
def health():
    return {'ok': True, 'time': datetime.now().isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# ダッシュボード HTML
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX AI EA 並列ランダムサーチ</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:16px}
.header{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.header h1{font-size:1.2em;color:#58a6ff;flex:1}
.live-dot{width:9px;height:9px;border-radius:50%;background:#3fb950;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.badge{padding:3px 12px;border-radius:12px;font-size:.78em;font-weight:700;border:1px solid}
.badge-train{background:#3fb95022;color:#3fb950;border-color:#3fb95066}
.badge-done {background:#ffa65722;color:#ffa657;border-color:#ffa65766}
.badge-error{background:#f4433622;color:#f44336;border-color:#f4433666}
.badge-wait {background:#8b949e22;color:#8b949e;border-color:#8b949e66}

.toolbar{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.btn{display:inline-block;padding:6px 14px;border-radius:6px;font-size:.82em;font-weight:600;
  cursor:pointer;border:none;text-decoration:none}
.btn-blue {background:#1f6feb;color:#fff}.btn-blue:hover {background:#388bfd}
.btn-green{background:#238636;color:#fff}.btn-green:hover{background:#2ea043}
.btn-red  {background:#b91c1c;color:#fff}.btn-red:hover  {background:#dc2626}
.btn-gray {background:#21262d;color:#e6edf3;border:1px solid #30363d}.btn-gray:hover{background:#30363d}
.btn-sm   {padding:3px 9px;font-size:.74em}

.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
@media(max-width:900px){.grid4{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card h2{font-size:.68em;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.8px}
.big{font-size:1.9em;font-weight:700}
.sub{font-size:.74em;color:#8b949e;margin-top:3px}
.bar-wrap{background:#21262d;border-radius:4px;height:8px;overflow:hidden;margin-top:6px}
.bar{height:100%;border-radius:4px;transition:width .5s}
.lrow{display:flex;justify-content:space-between;font-size:.71em;color:#8b949e;margin-top:3px}
.msg{background:#161b22;border-left:3px solid #58a6ff;padding:8px 14px;border-radius:4px;
  font-size:.8em;color:#c9d1d9;margin-bottom:12px;min-height:26px}

/* 並列試行カード */
.running-grid{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.trial-card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 14px;
  min-width:200px;flex:1}
.trial-card h3{font-size:.72em;color:#58a6ff;margin-bottom:6px}
.trial-badge{display:inline-block;background:#1f3a5f;color:#79c0ff;border-radius:4px;
  padding:1px 6px;font-size:.7em;margin-right:4px}

table{width:100%;border-collapse:collapse;font-size:.78em}
th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600;font-size:.72em;text-transform:uppercase}
td:first-child,th:first-child{text-align:center}
tr:hover td{background:#1c2128}

.footer{text-align:right;font-size:.7em;color:#484f58;margin-top:12px}
#stop-modal{display:none;position:fixed;inset:0;background:#00000088;z-index:999;
  align-items:center;justify-content:center}
#stop-modal.show{display:flex}
#stop-box{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px;
  max-width:380px;width:90%;text-align:center}
#stop-box h3{color:#f85149;margin-bottom:10px}
#stop-box p{color:#8b949e;font-size:.85em;margin-bottom:18px;line-height:1.6}
#stop-box .btn-row{display:flex;gap:10px;justify-content:center}
</style>
</head>
<body>

<div class="header">
  <span class="live-dot" id="dot"></span>
  <h1>FX AI EA 並列ランダムサーチ ダッシュボード
    <span id="gpu-name-badge" style="font-size:.55em;background:#21262d;border:1px solid #30363d;
      border-radius:6px;padding:2px 8px;vertical-align:middle;color:#79c0ff;font-weight:400;
      margin-left:10px">GPU: ...</span>
  </h1>
  <span class="badge badge-wait" id="phase-badge">待機中</span>
</div>

<!-- 稼働マシン一覧 -->
<div class="card" style="margin-bottom:12px" id="nodes-card">
  <h2>🖥 稼働マシン一覧</h2>
  <div style="overflow-x:auto">
    <table id="nodes-table">
      <thead>
        <tr>
          <th>GPU</th><th>ノードID</th><th>完了件数</th>
          <th>ベストPF</th><th>速度 (件/30分)</th><th>最終更新</th>
        </tr>
      </thead>
      <tbody id="nodes-tbody">
        <tr><td colspan="6" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="toolbar">
  <button class="btn btn-red" onclick="openStopModal()" id="stop-btn">⏹ 学習停止</button>
  <a class="btn btn-gray" href="/download/best"    target="_blank">💾 ベスト DL</a>
  <a class="btn btn-gray" href="/download/results" target="_blank">📊 全結果 JSON</a>
  <a class="btn btn-gray" href="/download/log"     target="_blank">📋 ログ</a>
  <button class="btn btn-gray" onclick="runBackfill()" id="backfill-btn" title="特徴量重要度など不足データを補完">🔄 データ補完</button>
  <span style="flex:1"></span>
  <span style="font-size:.74em;color:#8b949e" id="stop-status"></span>
</div>

<div class="msg" id="msg">起動中...</div>

<!-- 4列メトリクス -->
<div class="grid4">
  <div class="card">
    <h2>完了試行 / 実行中</h2>
    <div class="big">
      <span id="m-done" style="color:#58a6ff">0</span>
      <span style="font-size:.4em;color:#8b949e"> / </span>
      <span id="m-running" style="color:#3fb950;font-size:.7em">0</span>
      <span style="font-size:.3em;color:#8b949e">並列</span>
    </div>
    <div class="sub" id="m-elapsed-str">経過: --:--:--</div>
  </div>

  <div class="card">
    <h2>最良 PF</h2>
    <div class="big" id="m-pf" style="color:#f85149">0.0000</div>
    <div class="sub">
      SR: <span id="m-sr" style="color:#79c0ff">-</span> &nbsp;
      MaxDD: <span id="m-dd" style="color:#f85149">-</span>
    </div>
    <div class="bar-wrap"><div id="bar-pf" class="bar" style="background:#f85149;width:0%"></div></div>
    <div class="lrow"><span id="m-best-trial" style="color:#8b949e"></span><span id="m-pf-info" style="color:#8b949e"></span></div>
  </div>

  <div class="card">
    <h2>GPU 使用率</h2>
    <div class="big" id="m-gpu" style="color:#3fb950">0%</div>
    <div class="bar-wrap"><div id="bar-gpu" class="bar" style="background:#3fb950;width:0%"></div></div>
    <div class="lrow" style="margin-top:6px"><span>VRAM</span>
      <span id="m-vram" style="color:#79c0ff">0 / 11 GB</span></div>
    <div class="bar-wrap"><div id="bar-vram" class="bar" style="background:#2196f3;width:0%"></div></div>
  </div>

  <div class="card">
    <h2>TOP100 進捗</h2>
    <div class="big" id="m-top-n" style="color:#ffa657">0</div>
    <div class="sub" id="m-top-pf">TOP1 PF: -</div>
    <div class="bar-wrap"><div id="bar-top" class="bar" style="background:#ffa657;width:0%"></div></div>
  </div>
</div>

<!-- 実行中の並列試行 -->
<div class="card" style="margin-bottom:12px">
  <h2>実行中の並列試行</h2>
  <div class="running-grid" id="running-trials">
    <div style="color:#8b949e;font-size:.82em;padding:6px">待機中...</div>
  </div>
</div>

<!-- チャート 2列 -->
<div class="grid2">
  <div class="card">
    <h2>Loss / Accuracy チャート (最新試行)</h2>
    <div id="chart-ph" style="color:#8b949e;font-size:.82em;padding:16px;text-align:center">
      訓練開始後に表示</div>
    <div id="chart-wrap" style="display:none;position:relative;height:200px">
      <canvas id="mainChart"></canvas></div>
  </div>
  <div class="card">
    <h2>試行別 PF / SR チャート (直近100件)</h2>
    <div id="pf-ph" style="color:#8b949e;font-size:.82em;padding:16px;text-align:center">
      試行完了後に表示</div>
    <div id="pf-wrap" style="display:none;position:relative;height:200px">
      <canvas id="pfChart"></canvas></div>
  </div>
</div>

<!-- ベストモデル ダウンロードリンク -->
<div class="card" id="best-links-card" style="margin-bottom:12px;display:none">
  <h2>📥 ベストモデル ダウンロード</h2>
  <div id="best-links-body" style="font-size:.85em"></div>
</div>

<!-- TOP100 テーブル -->
<div class="card" style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <h2 style="margin:0">TOP 100 モデル (PF降順)</h2>
    <a class="btn btn-green btn-sm" href="/download/best" target="_blank">💾 ベスト一括DL</a>
  </div>
  <div style="overflow-x:auto;max-height:480px;overflow-y:auto">
    <table id="top100-table">
      <thead>
        <tr>
          <th>Rank</th><th>Trial#</th>
          <th>PF</th><th>純利益</th><th>SR</th><th>MaxDD</th>
          <th>取引</th><th>勝率</th><th>Arch</th><th>Hidden</th><th>Feat#</th>
          <th style="min-width:180px">重要特徴量 TOP10</th>
          <th>Report</th><th>DL</th>
        </tr>
      </thead>
      <tbody id="top100-tbody">
        <tr><td colspan="13" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- 最近完了 50件 (全ノード) -->
<div class="card" style="margin-bottom:12px">
  <h2>最近完了した試行 — 全ノード (最新50件)</h2>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr><th>#</th><th>PF</th><th>SR</th><th>MaxDD</th><th>純利益</th>
            <th>取引</th><th>勝率</th><th>Arch</th><th>ノード</th><th>時刻</th></tr>
      </thead>
      <tbody id="recent-tbody">
        <tr><td colspan="10" style="text-align:center;color:#8b949e">待機中</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="footer">
  最終更新: <span id="last-upd">-</span> &nbsp;|&nbsp;
  取得: <span id="poll-cnt">0</span> &nbsp;|&nbsp;
  エラー: <span id="err-cnt">0</span>
</div>

<!-- 停止確認 -->
<div id="stop-modal">
  <div id="stop-box">
    <h3>⏹ 学習を停止しますか？</h3>
    <p>実行中の全試行が終了次第、<br>ランダムサーチを停止します。<br><br>ダッシュボードは継続表示されます。</p>
    <div class="btn-row">
      <button class="btn btn-red"  onclick="confirmStop()">停止する</button>
      <button class="btn btn-gray" onclick="closeStopModal()">キャンセル</button>
    </div>
  </div>
</div>

<script>
let lossChart = null, pfChart = null;
let pollCount = 0, errCount = 0, stopReq = false;
let top100Timer = 0;

function fmtSec(s) {
  if (!s || s < 0) return '--:--:--';
  s = Math.floor(s);
  const h = Math.floor(s/3600), m = Math.floor(s%3600/60), ss = s%60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}
function pct(a,b){ return b>0?Math.min(100,Math.round(a/b*100)):0; }
function fmtN(v, d=4){ return v == null ? '-' : (+v).toFixed(d); }
function openStopModal()  { document.getElementById('stop-modal').classList.add('show'); }
function closeStopModal() { document.getElementById('stop-modal').classList.remove('show'); }
async function runBackfill() {
  const btn = document.getElementById('backfill-btn');
  btn.disabled = true;
  btn.textContent = '⏳ 補完中...';
  try {
    const res = await fetch('/api/backfill', {method:'POST'});
    const d   = await res.json();
    btn.textContent = '✅ 開始しました';
    setTimeout(() => { btn.disabled = false; btn.textContent = '🔄 データ補完'; }, 10000);
  } catch(e) {
    btn.textContent = '❌ 失敗: '+e.message;
    setTimeout(() => { btn.disabled = false; btn.textContent = '🔄 データ補完'; }, 5000);
  }
}
async function confirmStop() {
  closeStopModal();
  try {
    await fetch('/api/stop', {method:'POST'});
    stopReq = true;
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
    document.getElementById('stop-status').textContent = '停止リクエスト送信済み';
  } catch(e) { document.getElementById('stop-status').textContent = '失敗: '+e.message; }
}

function applyPhase(phase) {
  const badge = document.getElementById('phase-badge');
  const dot   = document.getElementById('dot');
  const map = {
    training:['並列サーチ中','badge-train','#3fb950'],
    trial_done:['試行完了','badge-train','#58a6ff'],
    done:['完了','badge-done','#ffa657'], waiting:['待機中','badge-wait','#8b949e'],
    error:['エラー','badge-error','#f44336'],
  };
  const [label, cls, color] = map[phase] || [phase,'badge-wait','#607d8b'];
  badge.textContent = label; badge.className = 'badge '+cls;
  dot.style.background = color;
}

function updateRunningTrials(runningList) {
  const wrap = document.getElementById('running-trials');
  if (!runningList || !runningList.length) {
    wrap.innerHTML = '<div style="color:#8b949e;font-size:.82em;padding:6px">試行なし</div>';
    return;
  }
  wrap.innerHTML = runningList.map(r => {
    const epPct = r.total_epochs > 0 ? Math.round(r.epoch/r.total_epochs*100) : 0;
    const vl    = r.val_loss ?? 0;
    const vlC   = vl<0.9?'#3fb950':vl<1.1?'#79c0ff':vl<1.3?'#ffa657':'#f0883e';
    return `<div class="trial-card">
      <h3>試行 #${r.trial}</h3>
      <span class="trial-badge">${r.arch}</span>
      <span class="trial-badge">h=${r.hidden}</span>
      <div style="font-size:.72em;margin-top:6px">
        Ep: ${r.epoch}/${r.total_epochs} (${epPct}%)<br>
        TrL: <span style="color:#f0883e">${fmtN(r.train_loss)}</span> &nbsp;
        VaL: <span style="color:${vlC}">${fmtN(r.val_loss)}</span><br>
        Acc: <span style="color:#3fb950">${((r.accuracy??0)*100).toFixed(1)}%</span>
        &nbsp; ${fmtSec(r.elapsed_sec)} 経過
      </div>
      <div class="bar-wrap" style="margin-top:5px">
        <div class="bar" style="background:#818cf8;width:${epPct}%"></div></div>
    </div>`;
  }).join('');
}

function updateLossChart(epochLog) {
  if (!epochLog || !epochLog.length) return;
  document.getElementById('chart-ph').style.display   = 'none';
  document.getElementById('chart-wrap').style.display = 'block';
  const cfg = {
    type: 'line',
    data: {
      labels: epochLog.map(r=>r.epoch),
      datasets: [
        {label:'Train Loss', data:epochLog.map(r=>r.train_loss??null),
         borderColor:'#f0883e',backgroundColor:'transparent',borderWidth:1.5,tension:.3,pointRadius:0,yAxisID:'yL',spanGaps:true},
        {label:'Val Loss',   data:epochLog.map(r=>r.val_loss??null),
         borderColor:'#79c0ff',backgroundColor:'#79c0ff18',borderWidth:2,tension:.3,pointRadius:2,yAxisID:'yL'},
        {label:'Acc %',      data:epochLog.map(r=>(r.acc??null)*100),
         borderColor:'#3fb950',backgroundColor:'#3fb95018',borderWidth:2,tension:.3,pointRadius:2,yAxisID:'yA'},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#e6edf3',font:{size:10},usePointStyle:true,boxWidth:8}},
        tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1}
      },
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
        yL:{type:'linear',position:'left',ticks:{color:'#8b949e'},grid:{color:'#21262d'},
            title:{display:true,text:'Loss',color:'#8b949e'}},
        yA:{type:'linear',position:'right',min:0,max:100,
            ticks:{color:'#3fb950',callback:v=>v+'%'},grid:{drawOnChartArea:false}},
      }
    }
  };
  if (lossChart) { lossChart.data = cfg.data; lossChart.update('none'); }
  else { lossChart = new Chart(document.getElementById('mainChart').getContext('2d'), cfg); }
}

function updatePFChart(trialResults) {
  if (!trialResults || !trialResults.length) return;
  document.getElementById('pf-ph').style.display  = 'none';
  document.getElementById('pf-wrap').style.display = 'block';
  const recent = trialResults.filter(r=>r.pf>0).slice(-100);
  const labels = recent.map(r=>'#'+r.trial);
  const pfs    = recent.map(r=>r.pf??0);
  const bgC    = pfs.map(v=>v>=2?'#f0883e80':v>=1.5?'#3fb95080':v>=1.2?'#ffa65780':'#58a6ff40');
  const bdC    = pfs.map(v=>v>=2?'#f0883e':v>=1.5?'#3fb950':v>=1.2?'#ffa657':'#58a6ff');
  const cfg = {
    type:'bar',
    data:{labels,datasets:[{label:'PF',data:pfs,backgroundColor:bgC,borderColor:bdC,borderWidth:1}]},
    options:{
      responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
        callbacks:{title:items=>'試行 '+items[0].label,label:item=>` PF: ${item.raw.toFixed(4)}`}}},
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:15},grid:{color:'#21262d'},maxBarThickness:18},
        y:{min:0,ticks:{color:'#8b949e'},grid:{color:'#21262d'},
           title:{display:true,text:'PF',color:'#8b949e'}},
      }
    }
  };
  if (pfChart){pfChart.data=cfg.data;pfChart.update('none');}
  else{pfChart=new Chart(document.getElementById('pfChart').getContext('2d'),cfg);}
}

function updateRecentTable(trialResults) {
  if (!trialResults || !trialResults.length) return;
  // 全ノードの結果を timestamp 降順で最新50件
  const recent = [...trialResults].filter(r=>r.trial)
    .sort((a,b)=>(b.timestamp||'').localeCompare(a.timestamp||'')).slice(0,50);
  const tbody  = document.getElementById('recent-tbody');
  tbody.innerHTML = recent.map(r => {
    const pf  = r.pf??0;
    const pfC = pf>=2?'#f0883e':pf>=1.5?'#3fb950':pf>=1.2?'#ffa657':'#8b949e';
    const sr  = r.sr??0;
    const srC = sr>=1?'#3fb950':sr>=0.5?'#ffa657':'#8b949e';
    const dd  = r.max_dd??0;
    const nid = (r.node_id||'').toUpperCase();
    const nidC = nid ? '#79c0ff' : '#8b949e';
    return `<tr>
      <td style="color:#8b949e">#${r.trial}</td>
      <td style="color:${pfC};font-weight:${pf>=1.2?'700':'400'}">${pf.toFixed(4)}</td>
      <td style="color:${srC}">${(sr).toFixed(3)}</td>
      <td style="color:#f85149">${(dd).toFixed(4)}</td>
      <td style="color:${(r.net_pnl??0)>=0?'#3fb950':'#f85149'}">${fmtN(r.net_pnl,3)}</td>
      <td>${r.trades??'-'}</td>
      <td style="color:#3fb950">${((r.win_rate??0)*100).toFixed(1)}%</td>
      <td style="color:#79c0ff">${r.arch??'-'}</td>
      <td style="color:${nidC};font-size:.75em">${nid||'-'}</td>
      <td style="color:#8b949e;font-size:.7em">${(r.timestamp??'').slice(5,16)}</td>
    </tr>`;
  }).join('');
}

async function updateTop100() {
  try {
    const res = await fetch('/api/top100',{cache:'no-store'});
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('top100-tbody');
    // TOP カード更新
    document.getElementById('m-top-n').textContent = data.length;
    document.getElementById('bar-top').style.width = Math.min(100, data.length) + '%';
    if (data.length) {
      const best = data[0];
      document.getElementById('m-top-pf').textContent = `TOP1 PF: ${(best.pf??0).toFixed(4)}`;
    }
    if (!data.length) {
      tbody.innerHTML='<tr><td colspan="14" style="text-align:center;color:#8b949e">まだ有効な試行がありません (取引数≥200)</td></tr>';
      return;
    }
      tbody.innerHTML = data.map(r => {
      const pf  = r.pf??0;
      const pfC = pf>=2?'#f0883e':pf>=1.5?'#3fb950':pf>=1.2?'#ffa657':'#79c0ff';
      const sr  = r.sr??0;
      const srC = sr>=1?'#3fb950':sr>=0.5?'#ffa657':'#8b949e';
      const rkMd= r.rank<=3?['🥇','🥈','🥉'][r.rank-1]:'#'+r.rank;
      const dlBtn = r.has_model
        ? `<a class="btn btn-green btn-sm" href="/download/model/${r.rank}" target="_blank">📥</a>`
        : `<span style="color:#484f58;font-size:.7em">-</span>`;
      const rpBtn = r.has_report
        ? `<a class="btn btn-blue btn-sm" href="/report/${r.trial}" target="_blank">📊</a>`
        : `<span style="color:#484f58;font-size:.7em">-</span>`;
      // 特徴量重要度TOP10
      const imp = (r.feature_importance || []).filter(fi => Array.isArray(fi) && fi.length >= 2 && fi[0] && typeof fi[0]==='string');
      const maxScore = imp.length && imp[0][1] > 0 ? imp[0][1] : 1;
      const impHtml = imp.length
        ? imp.slice(0,10).map((fi,idx) => {
            const fname = String(fi[0]);
            const score = Number(fi[1]) || 0;
            const pct = Math.round((score / maxScore) * 100);
            const col = pct>80?'#f0883e':pct>50?'#ffa657':'#79c0ff';
            const barW = Math.max(2, pct);
            return `<span style="display:inline-block;margin:1px 2px;padding:2px 6px;`
              +`border-radius:3px;background:#21262d;font-size:.68em;color:${col};`
              +`border-left:${barW/10+1}px solid ${col}" `
              +`title="${fname}: ${score.toFixed(5)} (${pct}%)">`
              +`<b>${idx+1}.</b>${fname}</span>`;
          }).join('')
        : r.has_model
          ? '<span style="color:#58a6ff;font-size:.75em">⏳ 解析中</span>'
          : '<span style="color:#484f58;font-size:.75em">—</span>';
      return `<tr>
        <td style="font-weight:${r.rank<=3?'700':'400'}">${rkMd}</td>
        <td style="color:#8b949e">#${r.trial??'-'}</td>
        <td style="color:${pfC};font-weight:700">${pf.toFixed(4)}</td>
        <td style="color:${(r.net_pnl??0)>=0?'#3fb950':'#f85149'}">${fmtN(r.net_pnl,3)}</td>
        <td style="color:${srC}">${fmtN(sr,3)}</td>
        <td style="color:#f85149">${fmtN(r.max_dd,4)}</td>
        <td>${r.trades??'-'}</td>
        <td style="color:#3fb950">${((r.win_rate??0)*100).toFixed(1)}%</td>
        <td style="color:#79c0ff;font-size:.8em">${r.arch??'-'}</td>
        <td style="font-size:.8em">${r.hidden??'-'}×${r.layers??1}</td>
        <td style="font-size:.8em">${r.n_features??'-'}</td>
        <td style="text-align:left;max-width:220px">${impHtml}</td>
        <td>${rpBtn}</td>
        <td>${dlBtn}</td>
      </tr>`;
    }).join('');
  } catch(e) { /* silent */ }
}

async function poll() {
  try {
    const res = await fetch('/api/status',{cache:'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    pollCount++;
    document.getElementById('poll-cnt').textContent = pollCount;
    document.getElementById('last-upd').textContent = new Date().toLocaleTimeString('ja-JP');

    applyPhase(d.phase??'waiting');
    document.getElementById('msg').textContent = d.message??'';

    if (d.stop_requested && !stopReq) {
      stopReq = true;
      document.getElementById('stop-btn').disabled = true;
      document.getElementById('stop-btn').textContent = '⏳ 停止待機中...';
    }

    document.getElementById('m-done').textContent    = d.completed_count??0;
    document.getElementById('m-running').textContent = d.running_count??0;
    document.getElementById('m-elapsed-str').textContent = '経過: '+fmtSec(d.elapsed_sec);

    // 最良 PF
    const bpf   = d.best_pf??0;
    const pfPct = Math.min(100, bpf/2*100);
    const pfC   = bpf>=2?'#f0883e':bpf>=1.5?'#3fb950':bpf>=1.2?'#ffa657':'#f85149';
    document.getElementById('m-pf').textContent       = bpf.toFixed(4);
    document.getElementById('m-pf').style.color       = pfC;
    document.getElementById('bar-pf').style.width     = pfPct+'%';
    document.getElementById('bar-pf').style.background= pfC;
    const tr = d.trial_results;
    if (tr && tr.length) {
      const best = [...tr].sort((a,b)=>(b.pf??0)-(a.pf??0))[0];
      document.getElementById('m-sr').textContent  = fmtN(best.sr??0, 3);
      document.getElementById('m-dd').textContent  = fmtN(best.max_dd??0, 4);
      document.getElementById('m-best-trial').textContent = `試行 #${best.trial}`;
      document.getElementById('m-pf-info').textContent    =
        `純損益: ${fmtN(best.net_pnl??0, 3)}`;
    }

    // GPU
    const gpuP  = d.gpu_pct??0;
    const vramU = d.vram_used_gb??0;
    const vramT = d.vram_total_gb??11;
    const gpuC  = gpuP>90?'#f44336':gpuP>75?'#3fb950':'#58a6ff';
    document.getElementById('m-gpu').textContent         = gpuP+'%';
    document.getElementById('m-gpu').style.color         = gpuC;
    document.getElementById('bar-gpu').style.width       = gpuP+'%';
    document.getElementById('bar-gpu').style.background  = gpuC;
    document.getElementById('m-vram').textContent        = `${vramU.toFixed(1)} / ${vramT.toFixed(0)} GB`;
    document.getElementById('bar-vram').style.width      = pct(vramU,vramT)+'%';

    // ベストモデル ダウンロードリンク (S3 / GDrive)
    if (d.best_links && Object.keys(d.best_links).length > 0) {
      const bl = d.best_links;
      const card = document.getElementById('best-links-card');
      const body = document.getElementById('best-links-body');
      const pfTxt  = bl.pf         ? ` (PF=${parseFloat(bl.pf).toFixed(4)})` : '';
      const updTxt = bl.updated_at ? ` — 更新: ${bl.updated_at}` : '';
      const storageBadge = bl.storage === 'S3'
        ? `<span style="background:#1f6feb;color:#fff;padding:1px 6px;border-radius:4px;font-size:.8em">S3</span>`
        : `<span style="background:#238636;color:#fff;padding:1px 6px;border-radius:4px;font-size:.8em">GDrive</span>`;
      let html = `<div style="color:#8b949e;margin-bottom:8px">
        ${storageBadge} &nbsp;ノード: <b style="color:#e3b341">${bl.node_id||'-'}</b>
        &nbsp;試行#${bl.trial||'-'}${pfTxt}${updTxt}
      </div>`;
      const fileLabels = {
        'fx_model_best.onnx':    ['🧠', 'ONNX モデル'],
        'norm_params_best.json': ['📐', '正規化パラメータ'],
        'best_result.json':      ['📊', 'ベスト結果 JSON'],
        'report.html':           ['📈', 'バックテストレポート'],
      };
      html += '<div style="display:flex;flex-wrap:wrap;gap:8px">';
      for (const [fname, [icon, label]] of Object.entries(fileLabels)) {
        if (bl[fname]) {
          html += `<a href="${bl[fname]}" target="_blank"
            style="display:inline-flex;align-items:center;gap:4px;padding:5px 10px;
                   background:#21262d;border:1px solid #30363d;border-radius:6px;
                   color:#58a6ff;text-decoration:none;font-size:.82em">
            ${icon} ${label}
          </a>`;
        }
      }
      html += '</div>';
      body.innerHTML = html;
      card.style.display = '';
    }

    // GPU名 & ノード情報
    const gpuLabel = d.gpu_name || (d.node_id ? d.node_id.toUpperCase() : null);
    if (gpuLabel) {
      document.getElementById('gpu-name-badge').textContent = gpuLabel;
    }
    if (d.nodes_summary) {
      const ns = d.nodes_summary;
      const rows = Object.entries(ns).map(([nid, info]) => {
        const pf   = info.best_pf || 0;
        const pfC  = pf >= 2 ? '#f0883e' : pf >= 1.5 ? '#3fb950' : pf >= 1.2 ? '#ffa657' : '#79c0ff';
        const rate = info.rate_30min || 0;
        const rateC = rate >= 20 ? '#3fb950' : rate >= 10 ? '#ffa657' : '#8b949e';
        return `<tr>
          <td style="color:#e3b341;font-weight:600">${info.gpu_name || '?'}</td>
          <td style="color:#79c0ff">${nid.toUpperCase()}</td>
          <td style="color:#58a6ff">${info.count}</td>
          <td style="color:${pfC};font-weight:600">${pf.toFixed(4)}</td>
          <td style="color:${rateC};font-weight:600">${rate.toFixed(1)}</td>
          <td style="color:#8b949e;font-size:.8em">${info.last_seen || '-'}</td>
        </tr>`;
      }).join('');
      document.getElementById('nodes-tbody').innerHTML = rows ||
        '<tr><td colspan="6" style="color:#8b949e;text-align:center">データなし</td></tr>';
    }

    // 並列試行状態
    updateRunningTrials(d.running_trials);

    // チャート
    if (d.epoch_log)     updateLossChart(d.epoch_log);
    if (d.trial_results) updatePFChart(d.trial_results);
    if (d.trial_results) updateRecentTable(d.trial_results);

  } catch(e) {
    errCount++;
    document.getElementById('err-cnt').textContent = errCount;
    document.getElementById('msg').textContent = '⚠ 取得エラー: '+e.message;
  }
  // TOP100 は 10秒ごと
  top100Timer += 1;
  if (top100Timer >= 10) { top100Timer = 0; updateTop100(); }
}

poll();
updateTop100();
setInterval(poll, 1000);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    import uvicorn
    _port = int(os.environ.get('DASHBOARD_PORT', '8080'))
    uvicorn.run(app, host='0.0.0.0', port=_port, log_level='info')
