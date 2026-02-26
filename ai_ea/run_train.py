"""
FX AI EA 自動トレーニング v7 - 並列ランダムサーチ
  ・VRAM / GPU 使用率を監視して動的に並列数を決定
  ・停止条件なし (stop.flag が置かれるまで無限継続)
  ・TOP100 モデル保存 + SR / DD / 資産曲線レポート
  ・H100_MODE=1 で大型モデル / 大バッチ
"""
import os, subprocess, sys, json, shutil, time, random, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feature_sets import FEATURE_SETS

PY        = sys.executable
TRAIN_PY  = Path(__file__).parent / 'train.py'
OUT_DIR   = Path(__file__).parent

_WORKSPACE    = Path('/workspace') if Path('/workspace').exists() else OUT_DIR.parent
STOP_FLAG     = _WORKSPACE / 'stop.flag'
TRIALS_DIR    = OUT_DIR / 'trials'
TOP_CACHE_DIR = OUT_DIR / 'top_cache'
TOP_DIR       = OUT_DIR / 'top100'
ALL_RESULTS   = OUT_DIR / 'all_results.json'
PROGRESS_JSON = OUT_DIR / 'progress.json'
BEST_ONNX     = OUT_DIR / 'fx_model_best.onnx'
BEST_NORM     = OUT_DIR / 'norm_params_best.json'
BEST_JSON     = OUT_DIR / 'best_result.json'

TOP_N         = 100
H100_MODE     = os.environ.get('H100_MODE', '0') == '1'
MAX_PARALLEL  = int(os.environ.get('MAX_PARALLEL', '3' if H100_MODE else '1'))
VRAM_PER_TRIAL= float(os.environ.get('VRAM_PER_TRIAL', '10'))   # GB

ARCHS = [
    'mlp', 'gru_attn', 'bigru', 'lstm_attn',
    'cnn', 'tcn', 'cnn_gru', 'transformer', 'resnet', 'inception',
]

HIDDEN_MAP_LOCAL = {
    'mlp':         [32, 64, 128, 256, 512],
    'gru_attn':    [64, 128, 256, 512],
    'bigru':       [64, 128, 256],
    'lstm_attn':   [64, 128, 256, 512],
    'cnn':         [64, 128, 256, 512],
    'tcn':         [64, 128, 256, 512],
    'cnn_gru':     [64, 128, 256],
    'transformer': [64, 128, 256],
    'resnet':      [64, 128, 256, 512],
    'inception':   [64, 128, 256],
}
HIDDEN_MAP_H100 = {
    'mlp':         [512, 1024, 2048, 4096],
    'gru_attn':    [256, 512, 1024, 2048],
    'bigru':       [256, 512, 1024],
    'lstm_attn':   [256, 512, 1024, 2048],
    'cnn':         [256, 512, 1024, 2048],
    'tcn':         [256, 512, 1024, 2048],
    'cnn_gru':     [256, 512, 1024],
    'transformer': [256, 512, 1024],
    'resnet':      [256, 512, 1024, 2048],
    'inception':   [256, 512, 1024],
}
HIDDEN_MAP     = HIDDEN_MAP_H100  if H100_MODE else HIDDEN_MAP_LOCAL
BATCH_CHOICES  = [4096, 8192, 16384, 32768] if H100_MODE else [512, 1024, 2048, 4096]
SEQ_CHOICES    = [10, 15, 20, 30, 40, 50]   if H100_MODE else [5, 8, 10, 15, 20]
EPOCH_COUNT    = 2000 if H100_MODE else 800
TRIAL_TIMEOUT  = 7200 if H100_MODE else 600


# ── ハイパーパラメータサンプリング ───────────────────────────────────────────
def sample_params(rng: random.Random) -> dict:
    arch    = rng.choice(ARCHS)
    hidden  = rng.choice(HIDDEN_MAP[arch])
    layers  = rng.choice([1, 2, 3] if arch not in ('mlp', 'gru_attn') else [1, 2])
    dropout = round(rng.uniform(0.3, 0.6), 1)
    lr      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3]
                         if H100_MODE else [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
    batch   = rng.choice(BATCH_CHOICES)
    tp      = round(rng.uniform(1.5, 3.5), 1)
    sl      = round(rng.uniform(0.5, 1.5), 1)
    fwd     = rng.choice([10, 15, 20, 25, 30])
    thr     = round(rng.uniform(0.33, 0.50), 2)
    seq_len = rng.choice(SEQ_CHOICES)
    sched   = rng.choice(['onecycle', 'cosine', 'cosine'])
    wd      = rng.choice([1e-3, 1e-2, 5e-2, 1e-1])
    tm      = rng.choice([0, 0, 0, 12, 18, 12])
    if rng.random() < 0.25:
        n_feat   = rng.randint(2, 70)
        feat_set = -1
    else:
        feat_set = rng.randint(0, len(FEATURE_SETS) - 1)
        n_feat   = len(FEATURE_SETS[feat_set])
    seed = rng.randint(0, 9999)
    return dict(
        arch=arch, hidden=hidden, layers=layers, dropout=dropout,
        lr=lr, batch=batch, tp=tp, sl=sl, forward=fwd,
        threshold=thr, seq_len=seq_len, scheduler=sched,
        wd=wd, train_months=tm, feat_set=feat_set, n_features=n_feat,
        seed=seed, timeframe='H1', epochs=EPOCH_COUNT,
        label_type='triple_barrier',
    )


# ── GPU 情報取得 ─────────────────────────────────────────────────────────────
def _gpu_info() -> dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        return {
            'free_gb':  m.free  / 1e9,
            'total_gb': m.total / 1e9,
            'used_gb':  m.used  / 1e9,
            'gpu_pct':  u.gpu,
            'mem_pct':  round(m.used / m.total * 100),
        }
    except Exception:
        return {'free_gb': 999, 'total_gb': 80, 'used_gb': 0, 'gpu_pct': 0, 'mem_pct': 0}


def get_max_parallel(n_running: int) -> int:
    """VRAM/GPU 使用率から動的に最大並列数を返す"""
    if not H100_MODE:
        return MAX_PARALLEL
    gi = _gpu_info()
    # VRAM 空きから枠を計算
    vram_slots = max(1, int(gi['free_gb'] / VRAM_PER_TRIAL))
    # GPU が高負荷なら維持
    if gi['gpu_pct'] > 92 and n_running > 0:
        return n_running
    return min(MAX_PARALLEL, vram_slots)


# ── TOP_N 管理 ────────────────────────────────────────────────────────────────
def save_trial_model(trial_no: int) -> None:
    """現在の ONNX と norm_params を top_cache に保存"""
    trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
    dest = TOP_CACHE_DIR / f'trial_{trial_no:06d}'
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ['fx_model.onnx', 'norm_params.json', 'report.html']:
        src = trial_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def rebuild_top_n(results: list) -> None:
    """all_results から TOP_N を計算して top100/rank_XXX/ を再構築"""
    valid = [r for r in results
             if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    top_n = sorted(valid, key=lambda x: -x['pf'])[:TOP_N]
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(top_n, 1):
        tno = r.get('trial', 0)
        src = TOP_CACHE_DIR / f'trial_{tno:06d}'
        dst = TOP_DIR / f'rank_{rank:03d}'
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            (dst / 'result.json').write_text(
                json.dumps(r, indent=2, ensure_ascii=False), encoding='utf-8')


# ── 集約 progress.json ────────────────────────────────────────────────────────
def write_progress(running: dict, results: list, best_pf: float, start: float) -> None:
    running_info = []
    gi = _gpu_info()
    for tno, info in list(running.items()):
        tp_file = info['trial_dir'] / 'trial_progress.json'
        tp = {}
        if tp_file.exists():
            try:
                tp = json.loads(tp_file.read_text(encoding='utf-8'))
            except Exception:
                pass
        running_info.append({
            'trial':       tno,
            'arch':        info['params'].get('arch', '?'),
            'hidden':      info['params'].get('hidden', '?'),
            'epoch':       tp.get('epoch', 0),
            'total_epochs':tp.get('total_epochs', EPOCH_COUNT),
            'train_loss':  tp.get('train_loss', 0.0),
            'val_loss':    tp.get('val_loss', 0.0),
            'accuracy':    tp.get('accuracy', 0.0),
            'phase':       tp.get('phase', 'running'),
            'elapsed_sec': round(time.time() - info['start_time'], 0),
        })

    # 最近の trial 結果 (epoch_log 用に最新 running trial の log を使う)
    epoch_log = []
    if running_info:
        latest = max(running_info, key=lambda x: x['trial'])
        tp_file = (TRIALS_DIR / f"trial_{latest['trial']:06d}" / 'trial_progress.json')
        if tp_file.exists():
            try:
                epoch_log = json.loads(tp_file.read_text(encoding='utf-8')).get('epoch_log', [])
            except Exception:
                pass

    progress = {
        'phase':           'training' if running else 'waiting',
        'completed_count': len(results),
        'running_count':   len(running),
        'running_trials':  running_info,
        'best_pf':         best_pf,
        'target_pf':       0,
        'epoch_log':       epoch_log,
        'trial_results':   results[-200:],
        'start_time':      start,
        'elapsed_sec':     time.time() - start,
        'gpu_pct':         gi['gpu_pct'],
        'vram_used_gb':    round(gi['used_gb'], 1),
        'vram_total_gb':   round(gi['total_gb'], 1),
        'message': (f"実行中: {len(running)}並列  完了: {len(results)}件  "
                    f"ベスト PF: {best_pf:.4f}  "
                    f"GPU: {gi['gpu_pct']}%  VRAM: {gi['used_gb']:.1f}/{gi['total_gb']:.0f}GB"),
    }
    try:
        tmp = PROGRESS_JSON.with_suffix('.tmp')
        tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(PROGRESS_JSON)
    except Exception as e:
        print(f"  [WARN] progress.json 書き込み失敗: {e}")


# ── 並列トレーナー ────────────────────────────────────────────────────────────
class ParallelTrainer:
    def __init__(self):
        self.running: dict = {}   # trial_no -> {proc, params, start_time, trial_dir, log_fh}
        self.lock = threading.Lock()

    def launch(self, trial_no: int, params: dict, best_pf: float, start_time: float):
        trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = [PY, str(TRAIN_PY),
               '--trial',        str(trial_no),
               '--total_trials', '99999',
               '--best_pf',      str(best_pf),
               '--start_time',   str(start_time),
               '--out_dir',      str(trial_dir),
               ]
        for k, v in params.items():
            cmd += [f'--{k}', str(v)]

        log_fh = open(trial_dir / 'train.log', 'w', encoding='utf-8', buffering=1)
        proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

        with self.lock:
            self.running[trial_no] = {
                'proc':       proc,
                'params':     params,
                'start_time': time.time(),
                'trial_dir':  trial_dir,
                'log_fh':     log_fh,
            }
        feat_info = (f"set#{params['feat_set']}"
                     if params.get('feat_set', -1) >= 0 else f"rand{params['n_features']}")
        print(f"  [LAUNCH] 試行#{trial_no:4d}  {params['arch']:12s}  "
              f"h={params['hidden']:4d}  feat={feat_info}  PID={proc.pid}")

    def poll_completed(self) -> list:
        """完了した試行のリストを返し running から削除"""
        done = []
        with self.lock:
            for tno in list(self.running.keys()):
                info = self.running[tno]
                if info['proc'].poll() is not None:
                    info['log_fh'].close()
                    done.append((tno, info))
                    del self.running[tno]
        return done

    def terminate_all(self):
        with self.lock:
            for info in self.running.values():
                try:
                    info['proc'].terminate()
                except Exception:
                    pass

    def __len__(self):
        return len(self.running)


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    TOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TOP_DIR.mkdir(parents=True, exist_ok=True)

    mode_str = f'H100 80GB  並列={MAX_PARALLEL}  VRAM/試行={VRAM_PER_TRIAL}GB' \
               if H100_MODE else 'GTX 1080 Ti  シングル'
    print('=' * 60)
    print(f'FX AI EA v7 - 並列ランダムサーチ [{mode_str}]')
    print(f'  TOP {TOP_N} 保存  タイムアウト {TRIAL_TIMEOUT//60}分  stop.flag: {STOP_FLAG}')
    print('=' * 60)

    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    rng      = random.Random()
    trainer  = ParallelTrainer()
    results  = []
    best_pf  = 0.0
    trial_no = 1
    start    = time.time()

    # 既存結果を引き継ぐ
    if ALL_RESULTS.exists():
        try:
            results  = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
            trial_no = max((r.get('trial', 0) for r in results), default=0) + 1
            valid    = [r for r in results if r.get('pf', 0) > 0]
            if valid:
                best_r  = max(valid, key=lambda r: r['pf'])
                best_pf = best_r['pf']
                print(f"  前回最良PF={best_pf:.4f}  完了{len(results)}件  次試行#{trial_no}")
        except Exception:
            pass

    write_progress(trainer.running, results, best_pf, start)

    # ── メインループ ────────────────────────────────────────────────────────
    while True:
        # stop.flag チェック
        if STOP_FLAG.exists():
            print(f"\n[STOP] stop.flag 検出 → 実行中の試行を待機して終了")
            trainer.terminate_all()
            break

        # ── 完了した試行を回収 ──────────────────────────────────────────────
        for tno, info in trainer.poll_completed():
            result_path = info['trial_dir'] / 'last_result.json'
            r = {}
            if result_path.exists():
                try:
                    r = json.loads(result_path.read_text(encoding='utf-8'))
                except Exception:
                    pass

            pf     = float(r.get('pf', 0.0))
            trades = int(r.get('trades', 0))
            sr     = float(r.get('sr', 0.0))
            max_dd = float(r.get('max_dd', 0.0))
            elapsed= round(time.time() - info['start_time'], 0)

            record = {
                'trial':     tno,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pf':        pf,
                'trades':    trades,
                'win_rate':  r.get('win_rate',      0.0),
                'net_pnl':   r.get('net_pnl',       0.0),
                'gross_profit': r.get('gross_profit', 0.0),
                'gross_loss':   r.get('gross_loss',   0.0),
                'sr':        sr,
                'max_dd':    max_dd,
                'elapsed_sec': elapsed,
                **{k: v for k, v in info['params'].items()},
            }
            results.append(record)
            results.sort(key=lambda x: x['trial'])

            # all_results.json アトミック書き込み
            try:
                tmp = ALL_RESULTS.with_suffix('.tmp')
                tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding='utf-8')
                tmp.replace(ALL_RESULTS)
            except Exception as e:
                print(f"  [WARN] 結果保存失敗: {e}")

            # TOP_N に入ったらモデルを保存して再構築
            if pf > 0 and trades >= 200:
                try:
                    save_trial_model(tno)
                    rebuild_top_n(results)
                except Exception as e:
                    print(f"  [WARN] TOP{TOP_N} 更新失敗: {e}")

            # ベスト更新
            if pf > best_pf:
                best_pf = pf
                for src, dst in [(info['trial_dir'] / 'fx_model.onnx',    BEST_ONNX),
                                  (info['trial_dir'] / 'norm_params.json', BEST_NORM)]:
                    if src.exists():
                        shutil.copy2(src, dst)
                BEST_JSON.write_text(
                    json.dumps({**info['params'], 'pf': best_pf,
                                'sr': sr, 'max_dd': max_dd, 'trial': tno},
                               indent=2, ensure_ascii=False), encoding='utf-8')
                print(f"  [BEST] 試行#{tno}  PF={pf:.4f}  SR={sr:.3f}  MaxDD={max_dd:.4f}")
            else:
                print(f"  [DONE] 試行#{tno:4d}  PF={pf:.4f}  SR={sr:.3f}  "
                      f"MaxDD={max_dd:.4f}  取引={trades}  "
                      f"{elapsed/60:.1f}分  (ベスト={best_pf:.4f})")

        # ── 新規試行を投入 ──────────────────────────────────────────────────
        max_par = get_max_parallel(len(trainer))
        while len(trainer) < max_par:
            if STOP_FLAG.exists():
                break
            p = sample_params(rng)
            trainer.launch(trial_no, p, best_pf, start)
            trial_no += 1
            time.sleep(2)   # 連続起動の間隔

        # ── 進捗 JSON 書き込み (5秒ごと) ───────────────────────────────────
        write_progress(trainer.running, results, best_pf, start)
        time.sleep(5)

    # ── 終了処理 ────────────────────────────────────────────────────────────
    write_progress({}, results, best_pf, start)
    print(f"\n完了  総試行: {len(results)}件  最良PF: {best_pf:.4f}")
    if BEST_ONNX.exists():
        shutil.copy2(BEST_ONNX, OUT_DIR / 'fx_model.onnx')
    if BEST_NORM.exists():
        shutil.copy2(BEST_NORM, OUT_DIR / 'norm_params.json')


if __name__ == '__main__':
    main()
