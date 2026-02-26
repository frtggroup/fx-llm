"""
FX AI EA 自動トレーニング v8 - ハイブリッド遺伝的アルゴリズム
  ・最初の 500 件: ランダムサーチ (探索フェーズ)
  ・501 件以降: 75% 遺伝的アルゴリズム (TOP 結果を交叉・突然変異) + 25% ランダム
  ・VRAM / GPU 使用率を監視して動的に並列数を決定
  ・停止条件なし (stop.flag が置かれるまで無限継続)
  ・TOP100 モデル保存 + SR / DD / 資産曲線レポート
"""
import os, subprocess, sys, json, shutil, time, random, threading, signal
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

# ── チェックポイント (停止→再開用) ─────────────────────────────────────────
# ローカル: /workspace/data/checkpoint/ に定期保存
# S3: 環境変数 S3_* が設定されていれば Sakura オブジェクトストレージにも保存
CHECKPOINT_DIR      = _WORKSPACE / 'data' / 'checkpoint'
CHECKPOINT_INTERVAL = 600   # 秒 (10分ごとに保存)
CHECKPOINT_EVERY_N  = 10    # 件 (10試行完了ごとに保存)

S3_ENDPOINT  = os.environ.get('S3_ENDPOINT',   '')   # 例: https://s3.isk01.sakurastorage.jp
S3_ACCESS_KEY= os.environ.get('S3_ACCESS_KEY',  '')
S3_SECRET_KEY= os.environ.get('S3_SECRET_KEY',  '')
S3_BUCKET    = os.environ.get('S3_BUCKET',      'fxea')
S3_PREFIX    = os.environ.get('S3_PREFIX',      'checkpoint')
S3_ENABLED   = bool(S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY)


def _s3_client():
    import boto3
    return boto3.client(
        's3',
        endpoint_url      = S3_ENDPOINT,
        aws_access_key_id = S3_ACCESS_KEY,
        aws_secret_access_key = S3_SECRET_KEY,
        region_name       = os.environ.get('S3_REGION', 'jp-north-1'),
    )


def s3_upload(local_path: Path, s3_key: str) -> bool:
    """ファイルを S3 にアップロード。失敗しても例外を投げず False を返す"""
    try:
        _s3_client().upload_file(str(local_path), S3_BUCKET,
                                 f'{S3_PREFIX}/{s3_key}')
        return True
    except Exception as e:
        print(f'  [S3] upload失敗 {s3_key}: {e}')
        return False


def s3_download(s3_key: str, local_path: Path) -> bool:
    """S3 からファイルをダウンロード。失敗したら False を返す"""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _s3_client().download_file(S3_BUCKET, f'{S3_PREFIX}/{s3_key}',
                                   str(local_path))
        return True
    except Exception as e:
        print(f'  [S3] download失敗 {s3_key}: {e}')
        return False


def s3_list_keys(prefix: str = '') -> list:
    """S3_PREFIX/prefix 以下のキー一覧を返す"""
    try:
        full_prefix = f'{S3_PREFIX}/{prefix}' if prefix else S3_PREFIX + '/'
        paginator = _s3_client().get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=full_prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys
    except Exception as e:
        print(f'  [S3] list失敗: {e}')
        return []

TOP_N              = 100
RANDOM_PHASE_LIMIT = 200    # この件数までは純ランダム、以降は GA 主体
GA_RATIO           = 0.75   # GA の割合 (残りはランダム)
GA_PARENT_POOL     = 20     # 親候補を上位何件から選ぶか
H100_MODE     = os.environ.get('H100_MODE', '0') == '1'
MAX_PARALLEL  = int(os.environ.get('MAX_PARALLEL', '3' if H100_MODE else '1'))
VRAM_PER_TRIAL= float(os.environ.get('VRAM_PER_TRIAL', '10'))   # GB

# ── フリーズ検知: GPU無使用タイムアウト ──────────────────────────────────────
# データロード・前処理フェーズに DATA_PREP_BUDGET 秒の猶予を与え、
# それ以降も GPU を使っていなければ強制終了
DATA_PREP_BUDGET  = 600    # 秒: データ準備の最大許容時間 (10分)
NO_GPU_TIMEOUT    = 900    # 秒: GPU使用なしでこれ以上→強制終了 (15分)
LAUNCH_INTERVAL   = 5      # 秒: 試行投入間隔 (CUDA初期化の重複を防ぐ)

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
    'mlp':         [512, 1024, 2048],
    'gru_attn':    [256, 512, 1024],
    'bigru':       [256, 512, 1024],
    'lstm_attn':   [256, 512, 1024],
    'cnn':         [256, 512, 1024],
    'tcn':         [256, 512, 1024],
    'cnn_gru':     [256, 512, 1024],
    'transformer': [256, 512, 1024],
    'resnet':      [256, 512, 1024, 2048],
    'inception':   [256, 512, 1024],
}
HIDDEN_MAP     = HIDDEN_MAP_H100  if H100_MODE else HIDDEN_MAP_LOCAL
# 並列3本 × 最大バッチを考慮: H100 80GB / 3 ≈ 26GB/試行
# 大モデル(h≥1024)では小バッチ、小モデルでは大バッチ
# H100: 小バッチで1エポックあたりのイテレーション数を増やしGPU稼働率を上げる
# データ13K件 / 512 = 25バッチ/ep → GPU稼働率 ~60-80%
BATCH_CHOICES  = [256, 512, 1024, 2048] if H100_MODE else [256, 512, 1024, 2048]
SEQ_CHOICES    = [10, 15, 20, 30, 40, 50]  if H100_MODE else [5, 8, 10, 15, 20]
EPOCH_COUNT    = 2000 if H100_MODE else 800
TRIAL_TIMEOUT  = 5400 if H100_MODE else 600   # 90分 (torch.compile考慮)


# ── ハイパーパラメータサンプリング ───────────────────────────────────────────
def sample_params(rng: random.Random) -> dict:
    arch    = rng.choice(ARCHS)
    hidden  = rng.choice(HIDDEN_MAP[arch])
    layers  = rng.choice([1, 2, 3] if arch not in ('mlp', 'gru_attn') else [1, 2])
    dropout = round(rng.uniform(0.3, 0.6), 1)
    lr      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3]
                         if H100_MODE else [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
    # 大モデルでは小バッチ強制 (CUDA OOM防止: 3並列 × 26GB/trial)
    if H100_MODE and hidden >= 1024:
        batch = rng.choice([256, 512, 1024])
    else:
        batch = rng.choice(BATCH_CHOICES)
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


# ── 遺伝的アルゴリズム ────────────────────────────────────────────────────────
def _apply_one_mutation(p: dict, key: str, rng: random.Random) -> None:
    """key に対応するパラメータを1つ変異させる (in-place)"""
    if key == 'arch':
        p['arch']    = rng.choice(ARCHS)
        p['hidden']  = rng.choice(HIDDEN_MAP[p['arch']])
    elif key == 'hidden':
        p['hidden']  = rng.choice(HIDDEN_MAP[p['arch']])
    elif key == 'layers':
        p['layers']  = rng.choice([1, 2, 3] if p['arch'] not in ('mlp','gru_attn') else [1,2])
    elif key == 'dropout':
        p['dropout'] = round(rng.uniform(0.2, 0.7), 1)
    elif key == 'lr':
        p['lr']      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3]
                                   if H100_MODE else [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
    elif key == 'batch':
        p['batch']   = (rng.choice([2048, 4096, 8192]) if H100_MODE and p['hidden'] >= 1024
                        else rng.choice(BATCH_CHOICES))
    elif key == 'tp':
        p['tp']      = round(rng.uniform(1.2, 4.0), 1)
    elif key == 'sl':
        p['sl']      = round(rng.uniform(0.5, 2.0), 1)
    elif key == 'forward':
        p['forward'] = rng.choice([10, 15, 20, 25, 30, 40])
    elif key == 'threshold':
        p['threshold'] = round(rng.uniform(0.33, 0.55), 2)
    elif key == 'seq_len':
        p['seq_len'] = rng.choice(SEQ_CHOICES)
    elif key == 'scheduler':
        p['sched']   = rng.choice(['onecycle', 'cosine'])
    elif key == 'wd':
        p['wd']      = rng.choice([1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
    elif key == 'train_months':
        p['train_months'] = rng.choice([0, 0, 12, 18, 24, 12])
    elif key == 'feat_set':
        # フィーチャーセットを変える (探索多様性向上)
        p['feat_set'] = rng.randint(0, 99)


def _mutate(params: dict, rng: random.Random) -> dict:
    """複数パラメータを変異させる (1〜3個をランダムに選択)"""
    p = dict(params)
    mut_keys = [
        'arch', 'hidden', 'layers', 'dropout', 'lr', 'batch',
        'tp', 'sl', 'forward', 'threshold', 'seq_len',
        'scheduler', 'wd', 'train_months', 'feat_set',
    ]
    # 変異数: 多様性のため1〜3個
    n_mut = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    chosen = rng.sample(mut_keys, n_mut)
    for key in chosen:
        _apply_one_mutation(p, key, rng)
    # arch/hidden の整合性を保証
    if p['hidden'] not in HIDDEN_MAP.get(p['arch'], [p['hidden']]):
        p['hidden'] = rng.choice(HIDDEN_MAP[p['arch']])
    p['seed'] = rng.randint(0, 9999)
    return p


def _crossover(p1: dict, p2: dict, rng: random.Random) -> dict:
    """2 つの親パラメータを 1 点交叉で混合"""
    keys = [
        'arch', 'hidden', 'layers', 'dropout', 'lr', 'batch',
        'tp', 'sl', 'forward', 'threshold', 'seq_len',
        'scheduler', 'wd', 'train_months', 'feat_set', 'n_features',
    ]
    child = dict(p1)
    for k in keys:
        if rng.random() < 0.5 and k in p2:
            child[k] = p2[k]
    # arch と hidden の組み合わせが崩れていたら修正
    if child['hidden'] not in HIDDEN_MAP.get(child['arch'], [child['hidden']]):
        child['hidden'] = rng.choice(HIDDEN_MAP[child['arch']])
    child['seed'] = rng.randint(0, 9999)
    child['epochs'] = EPOCH_COUNT
    child['timeframe'] = 'H1'
    child['label_type'] = 'triple_barrier'
    return child


def _tournament_select(pool: list, rng: random.Random, k: int = 4) -> dict:
    """トーナメント選択: pool から k 件を引いて PF 最大を返す"""
    candidates = rng.sample(pool, min(k, len(pool)))
    return max(candidates, key=lambda r: r['pf'])


def ga_sample(results: list, rng: random.Random) -> dict:
    """遺伝的アルゴリズムでパラメータを生成する"""
    valid = [r for r in results if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    if len(valid) < 2:
        return sample_params(rng)   # 候補不足ならランダムにフォールバック

    # ── 親プール: 上位 GA_PARENT_POOL 件 (多様性のため arch・feat_set が被らないよう調整) ──
    sorted_valid = sorted(valid, key=lambda x: -x['pf'])
    pool = []
    seen_arch_feat: set = set()
    for r in sorted_valid:
        key = (r.get('arch', '?'), r.get('feat_set', -1))
        if key not in seen_arch_feat or len(pool) < GA_PARENT_POOL // 2:
            pool.append(r)
            seen_arch_feat.add(key)
        if len(pool) >= GA_PARENT_POOL:
            break

    r_val = rng.random()
    if r_val < 0.5:
        # 交叉: 親 2 体を選んでパラメータを混合
        p1 = _tournament_select(pool, rng)
        p2 = _tournament_select(pool, rng)
        child = _crossover(p1, p2, rng)
    elif r_val < 0.85:
        # 突然変異: 親 1 体から複数パラメータを変える
        p1    = _tournament_select(pool, rng)
        child = _mutate(p1, rng)
    else:
        # 15%: 上位から親を選んでランダム大変異 (exploration)
        p1 = pool[0]  # best parent
        child = _mutate(p1, rng)
        # さらに追加で 1〜2 パラメータをランダムに再変異
        extra = rng.sample(['arch', 'tp', 'sl', 'threshold', 'feat_set', 'forward'], 2)
        for k in extra:
            _apply_one_mutation(child, k, rng)
        if child['hidden'] not in HIDDEN_MAP.get(child['arch'], [child['hidden']]):
            child['hidden'] = rng.choice(HIDDEN_MAP[child['arch']])

    return child


def next_params(results: list, rng: random.Random) -> tuple[dict, str]:
    """完了件数に応じて GA / ランダムを切り替えてパラメータと戦略名を返す"""
    n = len(results)
    if n < RANDOM_PHASE_LIMIT:
        return sample_params(rng), 'random'
    if rng.random() < GA_RATIO:
        return ga_sample(results, rng), 'GA'
    return sample_params(rng), 'random'


# ── GPU 情報取得 ─────────────────────────────────────────────────────────────
def _gpu_info() -> dict:
    try:
        # nvidia-ml-py (pynvml の後継パッケージ)
        from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates)
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        m = nvmlDeviceGetMemoryInfo(h)
        u = nvmlDeviceGetUtilizationRates(h)
        return {
            'free_gb':  m.free  / 1e9,
            'total_gb': m.total / 1e9,
            'used_gb':  m.used  / 1e9,
            'gpu_pct':  u.gpu,
            'mem_pct':  round(m.used / m.total * 100),
        }
    except Exception:
        return {'free_gb': 999, 'total_gb': 80, 'used_gb': 0, 'gpu_pct': 0, 'mem_pct': 0}


def get_gpu_compute_pids() -> set:
    """nvidia-smi で現在 GPU 計算を使用している PID セットを返す"""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        pids = set()
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids
    except Exception:
        return set()


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
    # VRAM不足でも最低1並列は保証 (フリーズ防止)
    return max(1, min(MAX_PARALLEL, vram_slots))


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
            'strategy':    info.get('strategy', 'random'),
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

    n_done    = len(results)
    search_phase = ('random' if n_done < RANDOM_PHASE_LIMIT
                    else f'GA {int(GA_RATIO*100)}% + random {int((1-GA_RATIO)*100)}%')
    progress = {
        'phase':           'training' if running else 'waiting',
        'search_phase':    search_phase,
        'completed_count': n_done,
        'random_phase_limit': RANDOM_PHASE_LIMIT,
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
        'message': (f"実行中: {len(running)}並列  完了: {n_done}件  "
                    f"ベスト PF: {best_pf:.4f}  [{search_phase}]  "
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

    def launch(self, trial_no: int, params: dict, best_pf: float, start_time: float,
               strategy: str = 'random'):
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
                'strategy':   strategy,
            }
        feat_info = (f"set#{params['feat_set']}"
                     if params.get('feat_set', -1) >= 0 else f"rand{params['n_features']}")
        tag = '🧬GA' if strategy == 'GA' else '🎲Rnd'
        print(f"  [LAUNCH] 試行#{trial_no:4d} {tag}  {params['arch']:12s}  "
              f"h={params['hidden']:4d}  feat={feat_info}  PID={proc.pid}")

    def poll_completed(self) -> list:
        """完了/タイムアウトした試行のリストを返し running から削除"""
        done = []
        gpu_pids = get_gpu_compute_pids()   # 現在GPU使用中のPIDセット
        now = time.time()

        with self.lock:
            for tno in list(self.running.keys()):
                info    = self.running[tno]
                elapsed = now - info['start_time']
                proc    = info['proc']

                if proc.poll() is None:   # まだ実行中
                    pid = proc.pid

                    # ── GPU使用中PIDの追跡 ──────────────────────────────
                    if pid in gpu_pids:
                        info['last_gpu_time'] = now   # GPUアクティブ時刻を更新

                    last_gpu = info.get('last_gpu_time')
                    since_gpu = (now - last_gpu) if last_gpu else elapsed

                    # ── GPUノーアクティビティウォッチドッグ ─────────────
                    # DATA_PREP_BUDGET 秒以内はデータ準備中として許容
                    # それ以降も GPU を使っていなければ強制終了
                    if elapsed > DATA_PREP_BUDGET and since_gpu > NO_GPU_TIMEOUT:
                        print(f"  [NO-GPU] 試行#{tno}  経過{elapsed/60:.1f}分"
                              f"  GPU無使用{since_gpu/60:.1f}分 → 強制終了")
                        try:
                            proc.terminate()
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()

                    # ── 全体タイムアウト ────────────────────────────────
                    elif elapsed > TRIAL_TIMEOUT:
                        print(f"  [TIMEOUT] 試行#{tno} ({elapsed/60:.0f}分超) → 強制終了")
                        try:
                            proc.terminate()
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()

                if proc.poll() is not None:
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


# ── チェックポイント保存・復元 ────────────────────────────────────────────────
def save_checkpoint(results: list, best_pf: float) -> None:
    """all_results + best model + top100 をローカル & S3 に保存"""
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        # all_results.json
        tmp = CHECKPOINT_DIR / 'all_results.json.tmp'
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(CHECKPOINT_DIR / 'all_results.json')
        # best model ファイル
        for src, name in [(BEST_ONNX, 'fx_model_best.onnx'),
                          (BEST_NORM, 'norm_params_best.json'),
                          (BEST_JSON, 'best_result.json')]:
            if src.exists():
                shutil.copy2(src, CHECKPOINT_DIR / name)
        # top100 ディレクトリ
        top_dst = CHECKPOINT_DIR / 'top100'
        if TOP_DIR.exists():
            if top_dst.exists():
                shutil.rmtree(top_dst)
            shutil.copytree(TOP_DIR, top_dst)
        # メタ情報
        meta = {'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'completed': len(results), 'best_pf': best_pf,
                's3': S3_ENABLED}
        (CHECKPOINT_DIR / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        print(f'  [CKPT] ローカル保存完了 ({len(results)}件 / bestPF={best_pf:.4f})')

        # S3 アップロード
        if S3_ENABLED:
            upload_files = ['all_results.json', 'meta.json',
                            'fx_model_best.onnx', 'norm_params_best.json', 'best_result.json']
            ok = 0
            for name in upload_files:
                p = CHECKPOINT_DIR / name
                if p.exists() and s3_upload(p, name):
                    ok += 1
            # top100 を S3 に同期
            top100_ok = 0
            if top_dst.exists():
                for f in top_dst.rglob('*'):
                    if f.is_file():
                        rel = f.relative_to(CHECKPOINT_DIR)
                        if s3_upload(f, str(rel).replace('\\', '/')):
                            top100_ok += 1
            print(f'  [S3]  アップロード完了 ({ok}/{len(upload_files)}件 + top100:{top100_ok}件) '
                  f'→ s3://{S3_BUCKET}/{S3_PREFIX}/')
        else:
            print(f'  [CKPT] S3未設定 → ローカルのみ保存 ({CHECKPOINT_DIR})')
    except Exception as e:
        print(f'  [CKPT] 保存失敗: {e}')


def restore_checkpoint() -> bool:
    """S3 → ローカル → 作業ディレクトリ の順にチェックポイントを復元"""
    # S3 から先にダウンロードを試みる
    if S3_ENABLED:
        print(f'  [S3]  チェックポイント確認中 s3://{S3_BUCKET}/{S3_PREFIX}/ ...')
        dl_files = ['all_results.json', 'meta.json',
                    'fx_model_best.onnx', 'norm_params_best.json', 'best_result.json']
        downloaded = 0
        for name in dl_files:
            if s3_download(name, CHECKPOINT_DIR / name):
                downloaded += 1
        # top100 は result.json のみダウンロード (ONNXは大きいので起動時はスキップ)
        top100_json_count = 0
        for key in s3_list_keys('top100'):
            if not key.endswith('result.json'):
                continue   # ONNX / norm_params / report.html はスキップ
            rel  = key[len(S3_PREFIX)+1:]
            dest = CHECKPOINT_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if s3_download(key[len(S3_PREFIX)+1:], dest):
                top100_json_count += 1
        if top100_json_count:
            print(f'  [S3]  top100 result.json {top100_json_count}件 取得 (ONNX はスキップ)')
        if downloaded == 0:
            print('  [S3]  チェックポイントなし')

    # ローカルから復元
    meta_path = CHECKPOINT_DIR / 'meta.json'
    ar_path   = CHECKPOINT_DIR / 'all_results.json'
    if not ar_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        print(f'  [CKPT] チェックポイント発見: {meta.get("saved_at","?")}  '
              f'{meta.get("completed","?")}件  bestPF={meta.get("best_pf","?")}')
        shutil.copy2(ar_path, ALL_RESULTS)
        for name, dst in [('fx_model_best.onnx',   BEST_ONNX),
                          ('norm_params_best.json', BEST_NORM),
                          ('best_result.json',       BEST_JSON)]:
            src = CHECKPOINT_DIR / name
            if src.exists():
                shutil.copy2(src, dst)
        top_src = CHECKPOINT_DIR / 'top100'
        if top_src.exists():
            if TOP_DIR.exists():
                shutil.rmtree(TOP_DIR)
            shutil.copytree(top_src, TOP_DIR)
        print('  [CKPT] 復元完了 → 前回の続きから再開します')
        return True
    except Exception as e:
        print(f'  [CKPT] 復元失敗: {e}')
        return False


# ── メイン ────────────────────────────────────────────────────────────────────
def _precache_data() -> bool:
    """データキャッシュを事前作成して全試行が即座に使えるようにする"""
    import pickle
    DATA_PATH = Path(os.environ.get('DATA_PATH', '/workspace/data/USDJPY_M1.csv'))
    cache_path = TRIALS_DIR.parent / 'df_cache_H1.pkl'
    if cache_path.exists():
        print(f"  [PRE-CACHE] キャッシュ既存: {cache_path}")
        return True
    if not DATA_PATH.exists():
        print(f"  [PRE-CACHE] データファイルなし: {DATA_PATH}")
        return False
    print(f"  [PRE-CACHE] データキャッシュを事前作成中... (初回のみ数分かかります)")
    try:
        import sys as _sys
        _sys.path.insert(0, str(TRAIN_PY.parent))
        from features import load_data, add_indicators
        import numpy as np
        from datetime import timedelta
        t0 = time.time()
        df = load_data(str(DATA_PATH), timeframe='H1')
        df = add_indicators(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        test_start = df.index[-1] - timedelta(days=365)
        df_tr = df[df.index < test_start].copy()
        df_te = df[df.index >= test_start].copy()
        tmp = cache_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump((df_tr, df_te), f)
        tmp.replace(cache_path)
        print(f"  [PRE-CACHE] 完了 {time.time()-t0:.1f}秒  "
              f"訓練:{len(df_tr):,}行  テスト:{len(df_te):,}行  → {cache_path}")
        return True
    except Exception as e:
        print(f"  [PRE-CACHE] 失敗 (訓練は続行): {e}")
        return False


def main():
    # SIGTERM (コンテナ停止時) を受け取ったら stop.flag を置いてgraceful shutdown
    def _sigterm_handler(signum, frame):
        print('\n[SIGNAL] SIGTERM 受信 → チェックポイント保存して停止します...')
        STOP_FLAG.touch()
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT,  _sigterm_handler)

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    TOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TOP_DIR.mkdir(parents=True, exist_ok=True)

    mode_str = f'H100 80GB  並列={MAX_PARALLEL}  VRAM/試行={VRAM_PER_TRIAL}GB' \
               if H100_MODE else 'GTX 1080 Ti  シングル'
    print('=' * 60)
    print(f'FX AI EA v8 - 並列ランダムサーチ [{mode_str}]')
    print(f'  TOP {TOP_N} 保存  タイムアウト {TRIAL_TIMEOUT//60}分  stop.flag: {STOP_FLAG}')
    print(f'  GPU無使用タイムアウト: {NO_GPU_TIMEOUT//60}分  データ準備猶予: {DATA_PREP_BUDGET//60}分')
    print('=' * 60)

    # ── S3 接続確認 ────────────────────────────────────────────────────────────
    print(f'  S3_ENABLED : {S3_ENABLED}')
    print(f'  S3_ENDPOINT: {S3_ENDPOINT or "(未設定)"}')
    print(f'  S3_BUCKET  : {S3_BUCKET}  PREFIX: {S3_PREFIX}')
    if S3_ENABLED:
        try:
            cl = _s3_client()
            cl.put_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/.ping', Body=b'ok')
            cl.delete_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/.ping')
            print('  [S3] 接続テスト OK ✅')
        except Exception as e:
            print(f'  [S3] 接続テスト 失敗 ❌: {e}')
    else:
        print('  [S3] 無効 (S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY を環境変数で設定してください)')

    # ── 起動時にデータキャッシュを事前作成 (全試行が即座に学習開始できる) ──
    _precache_data()

    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    rng      = random.Random()
    trainer  = ParallelTrainer()
    results  = []
    best_pf  = 0.0
    trial_no = 1
    start    = time.time()

    # ── チェックポイントから復元 (ディスクマウント時は自動継続) ──────────────
    if not ALL_RESULTS.exists():
        restore_checkpoint()

    # 既存結果を引き継ぐ
    if ALL_RESULTS.exists():
        try:
            raw = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
            # ── 重複排除: 同じ trial 番号は最初の1件のみ残す ──────────────
            seen: set = set()
            results = []
            for r in raw:
                tno_r = r.get('trial', -1)
                if tno_r not in seen:
                    seen.add(tno_r)
                    results.append(r)
            if len(raw) != len(results):
                print(f"  [DEDUP] 重複除去: {len(raw)} → {len(results)} 件")
                # クリーンなデータで上書き保存
                tmp = ALL_RESULTS.with_suffix('.tmp')
                tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding='utf-8')
                tmp.replace(ALL_RESULTS)
            trial_no = max((r.get('trial', 0) for r in results), default=0) + 1
            valid    = [r for r in results if r.get('pf', 0) > 0]
            if valid:
                best_r  = max(valid, key=lambda r: r['pf'])
                best_pf = best_r['pf']
                print(f"  前回最良PF={best_pf:.4f}  完了{len(results)}件  次試行#{trial_no}")
        except Exception:
            pass

    last_checkpoint        = time.time()
    completed_since_ckpt   = 0   # チェックポイント後の完了件数カウンタ

    write_progress(trainer.running, results, best_pf, start)

    # ── メインループ ────────────────────────────────────────────────────────
    while True:
        # stop.flag チェック
        if STOP_FLAG.exists():
            print(f"\n[STOP] stop.flag 検出 → 実行中の試行を待機して終了")
            trainer.terminate_all()
            break

        # ── 完了した試行を回収 ──────────────────────────────────────────────
        newly_done = trainer.poll_completed()
        completed_since_ckpt += len(newly_done)
        for tno, info in newly_done:
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
                'strategy':  info.get('strategy', 'random'),
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
            # 重複防止: 同じ trial_no がすでにあれば上書き、なければ追加
            existing_idx = next((i for i, r in enumerate(results) if r['trial'] == tno), None)
            if existing_idx is not None:
                results[existing_idx] = record
            else:
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

            # ベスト更新 (200取引以上のみ対象)
            if pf > best_pf and trades >= 200:
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
            p, strategy = next_params(results, rng)
            trainer.launch(trial_no, p, best_pf, start, strategy)
            trial_no += 1
            time.sleep(LAUNCH_INTERVAL)   # 連続起動の間隔 (CUDA初期化の重複を防ぐ)

        # ── 進捗 JSON 書き込み (5秒ごと) ───────────────────────────────────
        write_progress(trainer.running, results, best_pf, start)

        # ── チェックポイント保存: 10試行ごと or 10分ごと ────────────────────
        should_ckpt = (completed_since_ckpt >= CHECKPOINT_EVERY_N or
                       time.time() - last_checkpoint >= CHECKPOINT_INTERVAL)
        if should_ckpt:
            save_checkpoint(results, best_pf)
            last_checkpoint      = time.time()
            completed_since_ckpt = 0

        time.sleep(5)

    # ── 終了処理 ────────────────────────────────────────────────────────────
    write_progress({}, results, best_pf, start)
    save_checkpoint(results, best_pf)   # 停止時に必ずチェックポイント保存
    print(f"\n完了  総試行: {len(results)}件  最良PF: {best_pf:.4f}")
    if BEST_ONNX.exists():
        shutil.copy2(BEST_ONNX, OUT_DIR / 'fx_model.onnx')
    if BEST_NORM.exists():
        shutil.copy2(BEST_NORM, OUT_DIR / 'norm_params.json')

    # ── MT5 Common\Files へ自動コピー ────────────────────────────────────
    _appdata = Path(os.environ.get('APPDATA', ''))
    _common  = _appdata / 'MetaQuotes' / 'Terminal' / 'Common' / 'Files'
    if _common.exists():
        _copies = [
            (OUT_DIR / 'fx_model.onnx',    _common / 'fx_model.onnx'),
            (OUT_DIR / 'norm_params.json',  _common / 'norm_params.json'),
        ]
        for src, dst in _copies:
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  → Common\\Files\\ にコピー: {src.name}")
    else:
        print(f"  [skip] Common\\Files 未検出: {_common}")


if __name__ == '__main__':
    main()
