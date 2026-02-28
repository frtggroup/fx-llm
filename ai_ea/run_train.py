"""
FX AI EA 自動トレーニング v8 - ハイブリッド遺伝的アルゴリズム
  ・最初の 500 件: ランダムサーチ (探索フェーズ)
  ・501 件以降: 75% 遺伝的アルゴリズム (TOP 結果を交叉・突然変異) + 25% ランダム
  ・VRAM / GPU 使用率を監視して動的に並列数を決定
  ・停止条件なし (stop.flag が置かれるまで無限継続)
  ・TOP100 モデル保存 + SR / DD / 資産曲線レポート
"""
import os, subprocess, sys, json, shutil, time, random, threading, signal, platform
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
S3_PREFIX    = os.environ.get('S3_PREFIX',      'mix')   # 両ノード共有フォルダ
S3_ENABLED   = bool(S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY)

# ── Google Drive 共有ストレージ (S3 より優先) ─────────────────────────────────
import gdrive as _gdrive
GDRIVE_ENABLED = _gdrive.GDRIVE_ENABLED


def remote_upload(local_path: Path, rel_key: str) -> bool:
    """GDrive > S3 の優先順でアップロード"""
    if GDRIVE_ENABLED:
        return _gdrive.upload(local_path, rel_key)
    if S3_ENABLED:
        return s3_upload(local_path, rel_key)
    return False


def remote_download(rel_key: str, local_path: Path) -> bool:
    """GDrive > S3 の優先順でダウンロード"""
    if GDRIVE_ENABLED:
        return _gdrive.download(rel_key, local_path)
    if S3_ENABLED:
        return s3_download(rel_key, local_path)
    return False


def remote_list_node_keys(glob_prefix: str) -> list[str]:
    """全ノードの同種ファイル一覧 (GDrive > S3)"""
    if GDRIVE_ENABLED:
        return _gdrive.list_node_keys(glob_prefix)
    if S3_ENABLED:
        return s3_list_node_keys(glob_prefix)
    return []


def remote_list_top100_keys() -> list[str]:
    """top100_* 以下の全ファイル相対パス一覧 (GDrive > S3)"""
    if GDRIVE_ENABLED:
        return _gdrive.list_keys_recursive('top100_')
    if S3_ENABLED:
        raw = s3_list_keys('top100_')
        return [k[len(S3_PREFIX)+1:] for k in raw]
    return []


def remote_list_best_keys() -> list[str]:
    """best_* 以下の全ファイル相対パス一覧 (GDrive > S3)"""
    if GDRIVE_ENABLED:
        return _gdrive.list_keys_recursive('best_')
    if S3_ENABLED:
        return s3_list_node_keys('best_')
    return []


def REMOTE_ENABLED() -> bool:
    return GDRIVE_ENABLED or S3_ENABLED

# ── ノードID (GTX / H100 / CPU) ─────────────────────────────────────────────
# S3 上でノードごとにファイルを分離することで競合を回避する
def _detect_node_id() -> str:
    """デバイス名からノードIDを自動決定。環境変数 NODE_ID で上書き可能"""
    nid = os.environ.get('NODE_ID', '').strip()
    if nid:
        return nid.lower()
    # TPU 検出 (torch_xla が利用可能な場合)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        dev_str = str(xm.xla_device()).lower()
        tpu_type = os.environ.get('TPU_NAME', os.environ.get('TPU_ACCELERATOR_TYPE', 'tpu'))
        # tpu_type 例: 'v4-8', 'v5litepod-8', 'v6e-1', 'trillium'
        for ver in ('v6e', 'v5p', 'v5e', 'v5litepod', 'v4', 'v3', 'trillium'):
            if ver in tpu_type.lower():
                return f'tpu_{ver}'
        return 'tpu'
    except Exception:
        pass
    # GPU 検出
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if 'h200' in name:     return 'h200'
            if 'h100' in name:     return 'h100'
            if 'a100' in name:     return 'a100'
            if '3090' in name:     return 'rtx3090'
            if '4090' in name:     return 'rtx4090'
            if '1080' in name:     return 'gtx1080ti'
            return name.replace(' ', '_')[:12]
    except Exception:
        pass
    return 'h100' if os.environ.get('H100_MODE', '0') == '1' else 'gtx1080ti'


def _auto_gpu_config(node_id: str) -> tuple[str, float, float, int]:
    """実際のデバイスメモリを読み取り、最適な並列数と割当を自動計算する。

    ティア基準:
      tpu    : Google TPU (v3/v4/v5/Trillium) — HBM ≥ 16 GB/chip
      xlarge : VRAM 120 GB+  (H200 SXM5  141 GB)
      large  : VRAM  60 GB+  (H100 80 GB / A100 80 GB / H200 NVL 94 GB)
      medium : VRAM  30 GB+  (A100 40 GB)
      small  : VRAM  14 GB+  (RTX 3090/4090  24 GB)
      micro  : VRAM   0 GB+  (GTX 1080 Ti 11 GB / その他)

    Returns: (tier, total_mem_gb, vram_per_trial_gb, max_parallel)
    """
    # TPU 検出
    if node_id.startswith('tpu'):
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            # TPU v4/v5 は 1チップあたり 32 GB HBM
            # Trillium (v6e) は 1チップあたり 32 GB
            num_devices = int(os.environ.get('TPU_NUM_DEVICES', '4'))
            mem_per_chip = {
                'tpu_v3': 16.0, 'tpu_v4': 32.0,
                'tpu_v5e': 16.0, 'tpu_v5p': 95.0,
                'tpu_trillium': 32.0,
                'tpu_v6e': 32.0,   # Trillium (v6e) = 32 GB HBM/chip
            }.get(node_id, 32.0)
            total_gb = mem_per_chip * num_devices
            # TPU は並列度を 1チップ = 1試行 として計算
            vpt = mem_per_chip * 0.75
            par = max(1, num_devices)
            return 'tpu', total_gb, vpt, par
        except Exception:
            pass

    total_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass

    if total_gb <= 0:
        _fallback: dict[str, float] = {
            'h200': 141.0, 'h100': 80.0, 'a100': 80.0,
            'rtx4090': 24.0, 'rtx3090': 24.0, 'gtx1080ti': 11.0,
        }
        total_gb = _fallback.get(node_id, 11.0)

    if   total_gb >= 120: tier = 'xlarge'
    elif total_gb >=  60: tier = 'large'
    elif total_gb >=  30: tier = 'medium'
    elif total_gb >=  14: tier = 'small'
    else:                 tier = 'micro'

    vpt_map = {'xlarge': 12.0, 'large': 10.0, 'medium': 8.0, 'small': 7.0, 'micro': 8.0}
    vpt = vpt_map[tier]
    par = max(1, int(total_gb * 0.85 / vpt))

    return tier, total_gb, vpt, par


NODE_ID = _detect_node_id()   # このノードの識別子 (例: 'h100', 'gtx1080ti')
GPU_NAME = os.environ.get("GPU_NAME", NODE_ID.upper())  # GPU display name for dashboard
_GPU_TIER, _GPU_VRAM_GB, _VPT_DEFAULT, _PAR_DEFAULT = _auto_gpu_config(NODE_ID)


def _s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        's3',
        endpoint_url      = S3_ENDPOINT,
        aws_access_key_id = S3_ACCESS_KEY,
        aws_secret_access_key = S3_SECRET_KEY,
        region_name       = os.environ.get('S3_REGION', 'jp-north-1'),
        config            = Config(connect_timeout=10, read_timeout=60,
                                   retries={'max_attempts': 2}),
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


def s3_node_key(name: str) -> str:
    """このノード専用の S3 キーを返す (例: 'results_h100.json')"""
    return name.replace('NODE_ID', NODE_ID)


def s3_list_node_keys(glob_prefix: str) -> list[str]:
    """全ノードの同種ファイル一覧 (例: glob_prefix='results_') → ['results_h100.json', 'results_gtx1080ti.json']"""
    all_keys = s3_list_keys('')
    prefix_full = f'{S3_PREFIX}/{glob_prefix}'
    return [k[len(S3_PREFIX)+1:] for k in all_keys if k.startswith(prefix_full)]

TOP_N              = 100
RANDOM_PHASE_LIMIT = 30     # この件数までは純ランダム、以降は 10分交互モード
GA_PARENT_POOL     = 20     # 親候補を上位何件から選ぶか

# ── 10分交互モード ─────────────────────────────────────────────────────────
# ランダムサーチ 10分 → GA 10分 → ランダム 10分 → ... を繰り返す
MODE_SWITCH_SEC    = 600    # 10分 = 600秒
_mode_start_time   = 0.0    # main() で time.time() を設定

# 重要特徴量 GA フェーチャ: 上位モデルから重視特徴量を収集して GA に使用
IMP_FEAT_TOP_K     = 15     # 各モデルから取り出す重要特徴量数
IMP_FEAT_POOL_SIZE = 30     # 重要特徴量プールサイズ (多めに持つ)
_important_features: list[str] = []   # 集計した重要特徴量名リスト (更新される)
_important_scores: dict = {}          # 特徴量名 → 重要度スコア (重み付きサンプリング用)

# ── 2フェーズGA の割合 (GA_RATIO 内の内訳) ─────────────────────────────────
# GA_feat : feat_set のみ変えてアーキテクチャ固定で特徴量を探索
# GA_param: feat_set 固定でハイパラのみ微調整
# GA_cross: 2親の交叉 (多様性維持)
# 合計 = 1.0
GA_FEAT_RATIO  = 0.40   # 特徴量探索フェーズ
GA_PARAM_RATIO = 0.40   # パラメータチューニングフェーズ
GA_CROSS_RATIO = 0.20   # 交叉フェーズ
# large/xlarge/medium/tpu = H100_MODE (大モデル・長シーケンスを有効化)
# 環境変数 H100_MODE=1 で強制有効、H100_MODE=0 で強制無効も可能
_h100_env = os.environ.get('H100_MODE', '').strip()
H100_MODE = (
    (_h100_env == '1') or
    (_h100_env != '0' and _GPU_TIER in ('medium', 'large', 'xlarge', 'tpu'))
)
TPU_MODE = (_GPU_TIER == 'tpu')

# MAX_PARALLEL / VRAM_PER_TRIAL:
#   0 または未設定 → GPU VRAM から自動計算
#   1以上の数値   → その値を強制使用
def _resolve_int_env(key: str, default: int) -> int:
    v = os.environ.get(key, '0').strip()
    return int(v) if v not in ('0', '', 'auto') else default

def _resolve_float_env(key: str, default: float) -> float:
    v = os.environ.get(key, '0').strip()
    return float(v) if v not in ('0', '', 'auto') else default

MAX_PARALLEL   = _resolve_int_env('MAX_PARALLEL',   _PAR_DEFAULT)
VRAM_PER_TRIAL = _resolve_float_env('VRAM_PER_TRIAL', _VPT_DEFAULT)

# ── フリーズ検知: GPU無使用タイムアウト ──────────────────────────────────────
# データロード・前処理フェーズに DATA_PREP_BUDGET 秒の猶予を与え、
# それ以降も GPU を使っていなければ強制終了
DATA_PREP_BUDGET  = 600    # 秒: データ準備の最大許容時間 (10分)
NO_GPU_TIMEOUT    = 900    # 秒: GPU使用なしでこれ以上→強制終了 (15分)
LAUNCH_INTERVAL   = 1      # 秒: 試行投入間隔 (CUDA初期化の重複を防ぐ)

ARCHS = [
    'mlp', 'gru_attn', 'bigru', 'lstm_attn',
    'cnn', 'tcn', 'cnn_gru', 'transformer', 'resnet', 'inception',
]

# ── GPU ティア別ハイパーパラメータ探索空間 ──────────────────────────────────
# micro  : GTX 1080 Ti  (11 GB)
# small  : RTX 3090/4090 (24 GB)
# medium : A100 40 GB
# large  : H100 80 GB / A100 80 GB / H200 NVL 94 GB
# xlarge : H200 SXM5 141 GB

_HIDDEN_MICRO = {
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
_HIDDEN_SMALL = {   # RTX 3090/4090: 中規模モデル
    'mlp':         [128, 256, 512, 1024],
    'gru_attn':    [128, 256, 512, 1024],
    'bigru':       [128, 256, 512],
    'lstm_attn':   [128, 256, 512, 1024],
    'cnn':         [128, 256, 512, 1024],
    'tcn':         [128, 256, 512, 1024],
    'cnn_gru':     [128, 256, 512],
    'transformer': [128, 256, 512],
    'resnet':      [128, 256, 512, 1024],
    'inception':   [128, 256, 512],
}
_HIDDEN_LARGE = {   # medium/large/xlarge: 大規模モデル
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

_TIER_HIDDEN_MAP = {
    'micro':  _HIDDEN_MICRO,
    'small':  _HIDDEN_SMALL,
    'medium': _HIDDEN_LARGE,
    'large':  _HIDDEN_LARGE,
    'xlarge': _HIDDEN_LARGE,
}
HIDDEN_MAP_LOCAL = _HIDDEN_MICRO   # 後方互換エイリアス
HIDDEN_MAP_H100  = _HIDDEN_LARGE   # 後方互換エイリアス
HIDDEN_MAP = _TIER_HIDDEN_MAP[_GPU_TIER]

# バッチサイズ: 大VRAM ほど大きいバッチを探索
_TIER_BATCH = {
    'micro':  [64, 128, 256, 512],
    'small':  [128, 256, 512, 1024],
    'medium': [256, 512, 1024, 2048],
    'large':  [256, 512, 1024, 2048],
    'xlarge': [512, 1024, 2048, 4096],
}
BATCH_CHOICES = _TIER_BATCH[_GPU_TIER]

# シーケンス長: 大VRAM ほど長いシーケンスを探索
_TIER_SEQ = {
    'micro':  [5, 8, 10, 15, 20],
    'small':  [8, 10, 15, 20, 30],
    'medium': [10, 15, 20, 30, 40],
    'large':  [10, 15, 20, 30, 40, 50],
    'xlarge': [15, 20, 30, 40, 50, 60],
}
SEQ_CHOICES = _TIER_SEQ[_GPU_TIER]

EPOCH_COUNT = 800   # GPU 共通

# 試行タイムアウト: 大モデルは学習に時間がかかる
_TIER_TIMEOUT = {
    'micro':   600,   # 10分
    'small':   900,   # 15分
    'medium': 1200,   # 20分
    'large':  1800,   # 30分
    'xlarge': 2400,   # 40分
}
TRIAL_TIMEOUT = _TIER_TIMEOUT[_GPU_TIER]


# ── ハイパーパラメータサンプリング ───────────────────────────────────────────
def sample_params(rng: random.Random) -> dict:
    arch    = rng.choice(ARCHS)
    hidden  = rng.choice(HIDDEN_MAP[arch])
    layers  = rng.choice([1, 2, 3] if arch not in ('mlp', 'gru_attn') else [1, 2])
    dropout = round(rng.uniform(0.3, 0.6), 1)
    lr      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3])
    # 大モデル(hidden≥1024)は CUDA OOM 防止で小バッチ上限を設ける
    # 上限はティアの BATCH_CHOICES 最大値の半分
    if hidden >= 1024 and len(BATCH_CHOICES) > 1:
        safe_batches = BATCH_CHOICES[:max(1, len(BATCH_CHOICES) - 1)]
        batch = rng.choice(safe_batches)
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
        p['lr']      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3])
    elif key == 'batch':
        if p['hidden'] >= 1024 and len(BATCH_CHOICES) > 1:
            p['batch'] = rng.choice(BATCH_CHOICES[:max(1, len(BATCH_CHOICES) - 1)])
        else:
            p['batch'] = rng.choice(BATCH_CHOICES)
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
    # ハイパーパラメータのみコピー (trial/pf 等の結果メタデータは除外)
    _hp_keys = ('arch', 'hidden', 'layers', 'dropout', 'lr', 'batch',
                'tp', 'sl', 'forward', 'threshold', 'seq_len',
                'scheduler', 'sched', 'wd', 'train_months', 'feat_set', 'n_features',
                'seed', 'epochs', 'timeframe', 'label_type')
    p = {k: v for k, v in params.items() if k in _hp_keys}
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
    # ハイパーパラメータのみコピー (trial/pf/strategy 等の結果メタデータは除外)
    child = {k: p1[k] for k in keys if k in p1}
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


def _build_parent_pool(results: list) -> list:
    """有効な結果から親プールを構築 (arch×feat_set の多様性を確保)"""
    valid = [r for r in results if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    sorted_valid = sorted(valid, key=lambda x: -x['pf'])
    pool: list = []
    seen_arch_feat: set = set()
    for r in sorted_valid:
        key = (r.get('arch', '?'), r.get('feat_set', -1))
        if key not in seen_arch_feat or len(pool) < GA_PARENT_POOL // 2:
            pool.append(r)
            seen_arch_feat.add(key)
        if len(pool) >= GA_PARENT_POOL:
            break
    return pool


_GA_HP_KEYS = ('arch', 'hidden', 'layers', 'dropout', 'lr', 'batch',
               'tp', 'sl', 'forward', 'threshold', 'seq_len',
               'scheduler', 'sched', 'wd', 'train_months', 'feat_set', 'n_features',
               'seed', 'epochs', 'timeframe', 'label_type')


def ga_feat_explore(results: list, rng: random.Random) -> dict:
    """フェーズ1 ─ 特徴量探索
    親の arch/hidden/lr/dropout 等を固定し feat_set だけを変える。
    重要特徴量プールがあれば、重複度の高い feat_set を優先して選ぶ。
    """
    pool = _build_parent_pool(results)
    if not pool:
        return sample_params(rng)

    parent = _tournament_select(pool, rng)
    child  = {k: parent[k] for k in _GA_HP_KEYS if k in parent}
    orig_feat = parent.get('feat_set', -1)

    # 重要特徴量と重複度が高い feat_set を優先
    if _important_features:
        from features import FEATURE_COLS
        imp_set = set(_important_features)
        # 各 feat_set の重要特徴量重複スコアを計算
        scores = []
        for fi, fset in enumerate(FEATURE_SETS):
            if fi == orig_feat:
                scores.append(0.0)   # 親と同じは除外
                continue
            feat_names = set(FEATURE_COLS[j] for j in fset if j < len(FEATURE_COLS))
            overlap = len(feat_names & imp_set)
            scores.append(float(overlap) + 0.1)   # 0.1 はゼロ重みを防ぐ
        total = sum(scores)
        if total > 0:
            weights = [s / total for s in scores]
            new_feat = rng.choices(range(len(FEATURE_SETS)), weights=weights)[0]
        else:
            new_feat = rng.randint(0, len(FEATURE_SETS) - 1)
    else:
        # 重要特徴量未集計の場合はランダム
        new_feat = orig_feat
        for _ in range(3):
            new_feat = rng.randint(0, len(FEATURE_SETS) - 1)
            if new_feat != orig_feat:
                break

    child['feat_set']   = new_feat
    child['n_features'] = len(FEATURE_SETS[new_feat])
    child['seed']       = rng.randint(0, 9999)
    return child


def ga_param_tune(results: list, rng: random.Random) -> dict:
    """フェーズ2 ─ パラメータチューニング
    親の feat_set を固定し、lr/dropout/tp/sl/threshold 等のハイパラのみ変える。
    良い特徴量セットを保持したまま細かい最適化を行う。
    """
    pool = _build_parent_pool(results)
    if not pool:
        return sample_params(rng)

    parent = _tournament_select(pool, rng)
    child  = {k: parent[k] for k in _GA_HP_KEYS if k in parent}

    # feat_set・arch は固定、ハイパラのみ 1〜2 個変更
    tune_keys = ['lr', 'dropout', 'tp', 'sl', 'threshold',
                 'batch', 'wd', 'forward', 'seq_len', 'layers',
                 'train_months', 'scheduler']
    n_mut  = rng.choices([1, 2, 3], weights=[0.45, 0.40, 0.15])[0]
    chosen = rng.sample(tune_keys, min(n_mut, len(tune_keys)))
    for key in chosen:
        _apply_one_mutation(child, key, rng)

    # arch と hidden の組み合わせ整合性を保証
    if child.get('hidden') not in HIDDEN_MAP.get(child.get('arch', ''), [child.get('hidden')]):
        child['hidden'] = rng.choice(HIDDEN_MAP[child['arch']])
    child['seed'] = rng.randint(0, 9999)
    return child


def ga_sample(results: list, rng: random.Random) -> tuple[dict, str]:
    """2フェーズGA: 特徴量探索 → パラメータチューニング → 交叉 の3サブ戦略を返す"""
    valid = [r for r in results if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    if len(valid) < 2:
        return sample_params(rng), 'random'

    r_val = rng.random()
    if r_val < GA_FEAT_RATIO:
        # フェーズ1: 特徴量探索 (feat_set だけ変える)
        return ga_feat_explore(results, rng), 'GA_feat'
    elif r_val < GA_FEAT_RATIO + GA_PARAM_RATIO:
        # フェーズ2: パラメータチューニング (feat_set 固定でハイパラ変更)
        return ga_param_tune(results, rng), 'GA_param'
    else:
        # 交叉: 2 親から多様性を生成
        pool = _build_parent_pool(results)
        if len(pool) < 2:
            return sample_params(rng), 'random'
        p1 = _tournament_select(pool, rng)
        p2 = _tournament_select(pool, rng)
        return _crossover(p1, p2, rng), 'GA_cross'


def _current_mode() -> str:
    """10分ごとに 'random' / 'ga' を交互に返す"""
    if _mode_start_time <= 0:
        return 'random'
    elapsed = time.time() - _mode_start_time
    cycle   = int(elapsed // MODE_SWITCH_SEC)
    return 'ga' if cycle % 2 == 1 else 'random'


def _update_important_features(results: list) -> None:
    """上位モデルの feature_importance から重要特徴量プールを更新。
    PF 重み付きスコアを集計し、多様性のあるプールを構築する。
    """
    global _important_features, _important_scores
    from collections import defaultdict
    # PF > 0.9 のモデルを最大30件対象（PF閾値は緩め）
    valid = [r for r in results
             if r.get('pf', 0) > 0.9 and r.get('trades', 0) >= 200
             and r.get('feature_importance')]
    if not valid:
        # PF > 0 でも試みる (序盤用フォールバック)
        valid = [r for r in results
                 if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200
                 and r.get('feature_importance')]
    top30 = sorted(valid, key=lambda x: -x['pf'])[:30]
    if not top30:
        return

    # PF で重み付けしてスコアを集計
    scores: dict = defaultdict(float)
    for rank_i, r in enumerate(top30):
        pf_weight = r['pf'] ** 2   # PF が高いほど重みを大きく
        for fname, score in (r.get('feature_importance') or [])[:IMP_FEAT_TOP_K]:
            if isinstance(fname, str):
                scores[fname] += score * pf_weight

    if not scores:
        return

    # スコア降順でソート
    sorted_feats = sorted(scores.items(), key=lambda x: -x[1])
    _important_features = [f for f, _ in sorted_feats[:IMP_FEAT_POOL_SIZE]]
    _important_scores   = {f: s for f, s in sorted_feats[:IMP_FEAT_POOL_SIZE]}


def _ga_sample_with_important_features(results: list, rng: random.Random) -> tuple[dict, str]:
    """重要特徴量プールを使って特徴量セットを構築し GA パラメータと組み合わせる。

    3つのモード:
      A (60%) imp_core   : 重要度上位を必ず含み、残りは重み付きサンプリング
      B (25%) imp_wide   : 重要特徴量 + 多めのランダム追加 (多様性重視)
      C (15%) imp_exploit: 最重要特徴量のみ絞り込み (特化型)
    各モードに対して GA_param チューニングも適用する。
    """
    global _important_features, _important_scores
    from features import FEATURE_COLS, N_FEATURES

    if len(_important_features) < 5:
        return ga_sample(results, rng)

    # 重要特徴量のインデックスマップ
    imp_idx  = [FEATURE_COLS.index(f) for f in _important_features if f in FEATURE_COLS]
    all_idx  = list(range(N_FEATURES))
    non_imp  = [i for i in all_idx if i not in imp_idx]

    mode_r = rng.random()
    if mode_r < 0.60:
        # モード A: コア重要特徴量 (上位 10〜15) + 重み付き追加
        core_lo  = min(10, len(imp_idx))
        core_hi  = min(15, len(imp_idx))
        core_n   = rng.randint(core_lo, max(core_lo, core_hi))
        core_idx = imp_idx[:core_n]
        if non_imp:
            extra_lo = min(5, len(non_imp))
            extra_hi = min(20, len(non_imp))
            extra_n  = rng.randint(extra_lo, max(extra_lo, extra_hi))
            extra    = rng.sample(non_imp, extra_n)
        else:
            extra = []
        feat_idx = sorted(set(core_idx + extra))
        mode_tag = 'imp_core'
    elif mode_r < 0.85:
        # モード B: 重要特徴量から重み付きサンプリング + 多めランダム
        weights   = [_important_scores.get(FEATURE_COLS[i], 0.001) for i in imp_idx]
        total_w   = sum(weights) or 1.0
        weights   = [w / total_w for w in weights]
        k_lo      = max(1, min(5, len(imp_idx) // 2))
        k_imp     = rng.randint(k_lo, max(k_lo, len(imp_idx)))
        # 重みに基づくサンプリング
        chosen_imp = []
        pool_copy  = list(zip(imp_idx, weights))
        for _ in range(k_imp):
            if not pool_copy:
                break
            ws = [w for _, w in pool_copy]
            pick = rng.choices(range(len(pool_copy)), weights=ws)[0]
            chosen_imp.append(pool_copy[pick][0])
            pool_copy.pop(pick)
        if non_imp:
            extra_lo = min(5, len(non_imp))
            extra_hi = min(30, len(non_imp))
            extra_n  = rng.randint(extra_lo, max(extra_lo, extra_hi))
            extra    = rng.sample(non_imp, extra_n)
        else:
            extra = []
        feat_idx = sorted(set(chosen_imp + extra))
        mode_tag = 'imp_wide'
    else:
        # モード C: 最重要特徴量のみ (絞り込み・特化)
        top_lo   = min(5, len(imp_idx))
        top_n    = rng.randint(top_lo, max(top_lo, min(12, len(imp_idx))))
        feat_idx = sorted(imp_idx[:top_n])
        mode_tag = 'imp_exploit'

    # GA パラメータも合わせて取得
    p, strategy = ga_sample(results, rng)
    p['feat_indices'] = feat_idx
    p.pop('feat_set',   None)   # feat_set は feat_indices で上書き
    p.pop('n_features', None)
    return p, f'{strategy}_{mode_tag}'


def next_params(results: list, rng: random.Random) -> tuple[dict, str]:
    """10分ごとにランダム↔GAを切り替えてパラメータと戦略名を返す"""
    n = len(results)
    if n < RANDOM_PHASE_LIMIT:
        return sample_params(rng), 'random'
    mode = _current_mode()
    if mode == 'ga':
        return _ga_sample_with_important_features(results, rng)
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
        pass
    # pynvml 失敗時 → torch から VRAM を取得
    try:
        import torch
        if torch.cuda.is_available():
            prop  = torch.cuda.get_device_properties(0)
            total = prop.total_memory / 1e9
            used  = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / 1e9
            return {'free_gb': total - used, 'total_gb': total, 'used_gb': used,
                    'gpu_pct': 0, 'mem_pct': round(used / max(total, 1) * 100)}
    except Exception:
        pass
    fallback_total = _GPU_VRAM_GB if _GPU_VRAM_GB > 0 else 11.0
    return {'free_gb': fallback_total, 'total_gb': fallback_total, 'used_gb': 0,
            'gpu_pct': 0, 'mem_pct': 0}


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
    total_gb = max(gi['total_gb'], 1)
    used_gb  = gi['used_gb']
    mem_pct  = used_gb / total_gb * 100

    # VRAM使用率が90%超なら新規起動を抑制 (OOM防止)
    if mem_pct > 90 and n_running > 0:
        return n_running  # 現状維持、追加起動しない
    # VRAM使用率が85%超なら保守的に並列数制限
    if mem_pct > 85:
        return max(1, min(n_running + 1, MAX_PARALLEL))
    # VRAM 空きから枠を計算
    vram_slots = max(1, int(gi['free_gb'] / VRAM_PER_TRIAL))
    # GPU が高負荷なら維持
    if gi['gpu_pct'] > 92 and n_running > 0:
        return n_running
    # VRAM不足でも最低1並列は保証 (フリーズ防止)
    return max(1, min(MAX_PARALLEL, vram_slots))


# ── TOP_N 管理 ────────────────────────────────────────────────────────────────
def save_trial_model(trial_no: int) -> None:
    """現在の ONNX と norm_params を top_cache に保存 (ノードIDプレフィックス付き)"""
    trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
    # キャッシュキー: trial_{node_id}_{trial_no} で全ノード間でユニーク
    cache_key = f'trial_{NODE_ID}_{trial_no:06d}'
    dest = TOP_CACHE_DIR / cache_key
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ['fx_model.onnx', 'norm_params.json', 'report.html']:
        src = trial_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def rebuild_top_n(results: list) -> None:
    """全ノードの results から TOP_N を計算して top100/rank_XXX/ を再構築"""
    valid = [r for r in results
             if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    top_n = sorted(valid, key=lambda x: -x['pf'])[:TOP_N]
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(top_n, 1):
        tno  = r.get('trial', 0)
        nid  = r.get('node_id', NODE_ID)
        # ノードIDつきキャッシュキーで検索 (旧形式にもフォールバック)
        src = TOP_CACHE_DIR / f'trial_{nid}_{tno:06d}'
        if not src.exists():
            src = TOP_CACHE_DIR / f'trial_{tno:06d}'   # 旧形式フォールバック
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
    if n_done < RANDOM_PHASE_LIMIT:
        search_phase = 'random (初期フェーズ)'
    else:
        mode    = _current_mode()
        elapsed = time.time() - _mode_start_time if _mode_start_time > 0 else 0
        cycle   = int(elapsed // MODE_SWITCH_SEC)
        remain  = int(MODE_SWITCH_SEC - (elapsed % MODE_SWITCH_SEC))
        imp_tag = f' 🎯重要特徴量{len(_important_features)}個活用' if _important_features else ''
        if mode == 'ga':
            search_phase = f'🔍 GA モード (残{remain}秒→ランダム切替){imp_tag}'
        else:
            search_phase = f'🎲 ランダム モード (残{remain}秒→GA切替){imp_tag}'
    # 全ノードの結果集計
    nodes_summary = {}
    for r in results:
        nid = r.get('node_id', NODE_ID)
        if nid not in nodes_summary:
            nodes_summary[nid] = {'count': 0, 'best_pf': 0.0}
        nodes_summary[nid]['count'] += 1
        if r.get('pf', 0) > nodes_summary[nid]['best_pf'] and r.get('trades', 0) >= 200:
            nodes_summary[nid]['best_pf'] = r.get('pf', 0)

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
        'gpu_name':        GPU_NAME,
        'node_id':         NODE_ID,
        'nodes_summary':   nodes_summary,
        'important_features': _important_features[:10],   # 上位10件を表示
        'message': (f"[{NODE_ID.upper()}] 実行中: {len(running)}並列  完了: {n_done}件  "
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
        _TAG_MAP = {
            'GA_feat':  '🔍GA_feat ',   # 特徴量探索
            'GA_param': '🔧GA_param',   # パラメータ調整
            'GA_cross': '🧬GA_cross',   # 交叉
            'random':   '🎲Rnd     ',   # ランダム
        }
        tag = _TAG_MAP.get(strategy, f'?{strategy}')
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


# ── 常駐ワーカープール (ProcessPoolExecutor) ─────────────────────────────────
# サブプロセス起動コスト (~8秒/試行) を初回起動1回に圧縮する。
# workers stay alive → Python/torch/CUDA init は1ワーカー当たり1回のみ。
class WorkerPool:
    """ProcessPoolExecutor ベースの常駐ワーカープール。
    ParallelTrainer と同一インターフェースを持ち差し替え可能。
    """
    def __init__(self, max_workers: int, cache_pkl_path):
        import concurrent.futures as _cf
        import multiprocessing as _mp
        import sys as _sys
        _sys.path.insert(0, str(TRAIN_PY.parent))
        from train import worker_init  # noqa: 存在確認
        self._max_workers   = max_workers
        self._cache_path    = str(cache_pkl_path)
        self._futures: dict = {}   # trial_no -> Future
        self._meta:    dict = {}   # trial_no -> {params, strategy, start_time, trial_dir}
        self.lock           = threading.Lock()
        print(f"  [WorkerPool] {max_workers}ワーカー起動中... (初回のみ数秒かかります)")
        self._executor = _cf.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=_mp.get_context('spawn'),
            initializer=_worker_init_proxy,
            initargs=(str(TRAIN_PY.parent), self._cache_path),
        )
        # ウォームアップ: ワーカーを全部起こしてCUDAを初期化させる
        import concurrent.futures as _cf2
        warmup_futs = [self._executor.submit(_warmup_probe) for _ in range(max_workers)]
        _cf2.wait(warmup_futs, timeout=120)
        print(f"  [WorkerPool] 全{max_workers}ワーカー準備完了")

    def _restart_executor(self) -> None:
        """BrokenProcessPool 発生時に executor を再作成する"""
        import concurrent.futures as _cf
        import multiprocessing as _mp
        print(f"  [WorkerPool] executor 再起動中...")
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # 壊れた futures / meta を破棄
        self._futures.clear()
        self._meta.clear()
        self._executor = _cf.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=_mp.get_context('spawn'),
            initializer=_worker_init_proxy,
            initargs=(str(TRAIN_PY.parent), self._cache_path),
        )
        print(f"  [WorkerPool] executor 再起動完了 ({self._max_workers}ワーカー)")

    def launch(self, trial_no: int, params: dict, best_pf: float, start_time: float,
               strategy: str = 'random'):
        trial_dir = TRIALS_DIR / f'trial_{trial_no:06d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            future = self._executor.submit(
                _run_trial_proxy,
                str(TRAIN_PY.parent),
                trial_no, params, str(trial_dir), best_pf, start_time,
            )
        except Exception as e:
            # BrokenProcessPool や shutdown 後の submit → executor を再作成して再試行
            print(f"  [WorkerPool] submit失敗 ({type(e).__name__}: {e}) → executor再起動")
            self._restart_executor()
            future = self._executor.submit(
                _run_trial_proxy,
                str(TRAIN_PY.parent),
                trial_no, params, str(trial_dir), best_pf, start_time,
            )
        with self.lock:
            self._futures[trial_no] = future
            self._meta[trial_no] = {
                'params':     params,
                'strategy':   strategy,
                'start_time': time.time(),
                'trial_dir':  trial_dir,
            }
        feat_info = (f"set#{params['feat_set']}"
                     if params.get('feat_set', -1) >= 0 else f"rand{params['n_features']}")
        _TAG_MAP = {
            'GA_feat':  '🔍GA_feat ',
            'GA_param': '🔧GA_param',
            'GA_cross': '🧬GA_cross',
            'random':   '🎲Rnd     ',
        }
        tag = _TAG_MAP.get(strategy, f'?{strategy}')
        print(f"  [LAUNCH] 試行#{trial_no:4d} {tag}  {params['arch']:12s}  "
              f"h={params['hidden']:4d}  feat={feat_info}")

    def poll_completed(self) -> list:
        done = []
        now  = time.time()
        broken = False
        with self.lock:
            for tno in list(self._futures.keys()):
                future = self._futures[tno]
                meta   = self._meta[tno]
                elapsed = now - meta['start_time']

                if future.done():
                    # future が BrokenProcessPool で終了しているか確認
                    try:
                        exc = future.exception(timeout=0)
                        if exc is not None:
                            ename = type(exc).__name__
                            if 'BrokenProcessPool' in ename or 'broken' in str(exc).lower():
                                broken = True
                            print(f"  [WARN] 試行#{tno} 例外終了: {ename}: {exc}")
                    except Exception:
                        pass
                    done.append((tno, meta))
                    del self._futures[tno]
                    del self._meta[tno]
                elif elapsed > TRIAL_TIMEOUT:
                    future.cancel()
                    print(f"  [TIMEOUT] 試行#{tno} ({elapsed/60:.0f}分超) → スキップ")
                    done.append((tno, meta))
                    del self._futures[tno]
                    del self._meta[tno]
        if broken:
            print(f"  [WorkerPool] BrokenProcessPool 検出 → executor 再起動")
            self._restart_executor()
        return done

    def terminate_all(self):
        with self.lock:
            for f in self._futures.values():
                f.cancel()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    @property
    def running(self) -> dict:
        """write_progress との互換性のため _meta を running として公開"""
        return self._meta

    def __len__(self):
        return len(self._futures)

    @property
    def n_active_workers(self) -> int:
        """実際にワーカーで実行中の数 (キュー待ちを除く)"""
        return min(len(self._futures), self._max_workers)


def _worker_init_proxy(train_py_dir: str, cache_pkl_path: str) -> None:
    """spawn ワーカーの初期化 (pickleできる関数でなければならない)"""
    import sys as _sys
    _sys.path.insert(0, train_py_dir)
    from train import worker_init
    worker_init(cache_pkl_path)


def _warmup_probe() -> bool:
    """ワーカーが起動済みか確認するだけのダミータスク"""
    return True


def _run_trial_proxy(train_py_dir: str, trial_no: int, params: dict,
                     trial_dir_str: str, best_pf: float, start_time: float) -> dict:
    """spawn ワーカーで run_trial_worker を呼ぶプロキシ (pickleできる関数)"""
    import sys as _sys
    _sys.path.insert(0, train_py_dir)
    from train import run_trial_worker
    return run_trial_worker(trial_no, params, trial_dir_str, best_pf, start_time)


# ── チェックポイント保存・復元 ────────────────────────────────────────────────
# S3 mix フォルダ共有設計:
#   各ノードは自分のファイルだけを書き込む → 競合ゼロ
#   mix/results_<NODE_ID>.json    : このノードの全試行結果
#   mix/top100_<NODE_ID>/         : このノードの top100 モデル
#   mix/best_<NODE_ID>/           : このノードのベストモデル
#   mix/meta_<NODE_ID>.json       : このノードのメタ情報
#   読み込み時は全ノードのファイルをマージして統合 top100 を再構築する

def save_checkpoint(results: list, best_pf: float) -> None:
    """自ノードの結果を S3 mix/<NODE_ID>/* に保存 (競合なし)"""
    # 自ノードの結果のみ (node_id が自分 or node_id フィールドなし) を保存
    own = [r for r in results if r.get('node_id', NODE_ID) == NODE_ID]
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        own_key = f'results_{NODE_ID}.json'
        tmp = CHECKPOINT_DIR / f'{own_key}.tmp'
        tmp.write_text(json.dumps(own, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(CHECKPOINT_DIR / own_key)

        # best model ファイル
        for src, name in [(BEST_ONNX, f'best_{NODE_ID}/fx_model_best.onnx'),
                          (BEST_NORM, f'best_{NODE_ID}/norm_params_best.json'),
                          (BEST_JSON, f'best_{NODE_ID}/best_result.json')]:
            if src.exists():
                dst = CHECKPOINT_DIR / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # top100 (自ノード分のみ)
        top_dst = CHECKPOINT_DIR / f'top100_{NODE_ID}'
        if TOP_DIR.exists():
            if top_dst.exists():
                shutil.rmtree(top_dst)
            shutil.copytree(TOP_DIR, top_dst)

        meta = {'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'node_id': NODE_ID, 'completed': len(own), 'best_pf': best_pf}
        meta_name = f'meta_{NODE_ID}.json'
        (CHECKPOINT_DIR / meta_name).write_text(
            json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        print(f'  [CKPT] ローカル保存完了 node={NODE_ID} ({len(own)}件 / bestPF={best_pf:.4f})')

        if REMOTE_ENABLED():
            tag = 'GDrive' if GDRIVE_ENABLED else 'S3'
            # 軽量ファイル (結果JSON / best / meta) は同期でアップロード
            ok = 0
            if remote_upload(CHECKPOINT_DIR / own_key, own_key): ok += 1
            if remote_upload(CHECKPOINT_DIR / meta_name, meta_name): ok += 1
            for name in [f'best_{NODE_ID}/fx_model_best.onnx',
                         f'best_{NODE_ID}/norm_params_best.json',
                         f'best_{NODE_ID}/best_result.json']:
                p = CHECKPOINT_DIR / name
                if p.exists() and remote_upload(p, name): ok += 1

            # top100 (大量ファイル) はバックグラウンドスレッドで差分のみアップロード
            # → メインループをブロックしない
            def _upload_top100_bg(top_dst=top_dst):
                top100_ok = 0
                if not top_dst.exists():
                    return
                for f in top_dst.rglob('*'):
                    if not f.is_file():
                        continue
                    rel = f'top100_{NODE_ID}/{f.relative_to(top_dst)}'.replace('\\', '/')
                    if remote_upload(f, rel):
                        top100_ok += 1
                if top100_ok:
                    print(f'  [{tag}]  top100アップロード完了 ({top100_ok}件)')

            import threading
            threading.Thread(target=_upload_top100_bg, daemon=True).start()
            print(f'  [{tag}]  アップロード完了 node={NODE_ID} ({ok}件 + top100:BG)')
        else:
            print(f'  [CKPT] リモートストレージ未設定 → ローカルのみ保存 ({CHECKPOINT_DIR})')
    except Exception as e:
        print(f'  [CKPT] 保存失敗: {e}')


def _merge_results_files(result_files: list[Path]) -> list:
    """複数ノードの results_*.json を読み込んでマージ・重複排除"""
    merged: dict = {}  # key: (node_id, trial) → result
    for f in result_files:
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            for r in data:
                nid = r.get('node_id', NODE_ID)
                tno = r.get('trial', 0)
                key = (nid, tno)
                if key not in merged or r.get('pf', 0) > merged[key].get('pf', 0):
                    merged[key] = r
        except Exception as e:
            print(f'  [WARN] 結果ファイル読込失敗 {f}: {e}')
    return sorted(merged.values(), key=lambda x: x.get('trial', 0))


def fetch_other_nodes_results() -> list:
    """リモートから他ノードの結果をダウンロードして返す (ノンブロッキング用)"""
    if not REMOTE_ENABLED():
        return []
    other_files = remote_list_node_keys('results_')
    results_all = []
    for rel_key in other_files:
        if f'results_{NODE_ID}.json' == rel_key:
            continue
        local = CHECKPOINT_DIR / rel_key
        if remote_download(rel_key, local):
            try:
                data = json.loads(local.read_text(encoding='utf-8'))
                results_all.extend(data)
            except Exception:
                pass
    return results_all


def restore_checkpoint() -> bool:
    """S3 mix/ から全ノードのチェックポイントをダウンロードしてマージ復元"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if REMOTE_ENABLED():
        tag = 'GDrive' if GDRIVE_ENABLED else 'S3'
        print(f'  [{tag}]  チェックポイント確認中 ...')
        # 全ノードの results_*.json をダウンロード
        result_keys = remote_list_node_keys('results_')
        if not result_keys:
            print(f'  [{tag}]  チェックポイントなし (全ノード)')
        else:
            for rk in result_keys:
                remote_download(rk, CHECKPOINT_DIR / rk)
                print(f'  [{tag}]  取得: {rk}')
        # 全ノードの meta_*.json をダウンロード
        for mk in remote_list_node_keys('meta_'):
            remote_download(mk, CHECKPOINT_DIR / mk)
        # このノード + 他ノードの best モデルをダウンロード (サブフォルダ内ファイル)
        for bk in remote_list_best_keys():   # 例: best_h100/fx_model_best.onnx
            remote_download(bk, CHECKPOINT_DIR / bk)
        # 全ノードの top100 をダウンロード (ONNX含む全ファイル)
        top100_count = 0
        for rel in remote_list_top100_keys():   # 例: top100_h100/rank_001/fx_model.onnx
            dest = CHECKPOINT_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if remote_download(rel, dest):
                top100_count += 1
        if top100_count:
            print(f'  [{tag}]  top100 {top100_count}ファイル取得 (全ノード)')

    # ── 全ノードの results_*.json をマージ ───────────────────────────────────
    result_files = list(CHECKPOINT_DIR.glob('results_*.json'))
    if not result_files:
        return False
    try:
        merged = _merge_results_files(result_files)
        if not merged:
            return False
        # ノードID付与 (古いデータで欠落している場合の補完)
        own_key = f'results_{NODE_ID}.json'
        for r in merged:
            if 'node_id' not in r:
                # どのファイルから来たか特定
                r['node_id'] = NODE_ID  # デフォルト
        best_pf_all = max((r.get('pf', 0) for r in merged if r.get('trades', 0) >= 200), default=0.0)
        print(f'  [CKPT] 全ノードマージ: {len(merged)}件  bestPF={best_pf_all:.4f}  '
              f'ノード: {sorted({r.get("node_id","?") for r in merged})}')
        tmp = ALL_RESULTS.with_suffix('.tmp')
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(ALL_RESULTS)

        # このノードのベストモデルを復元
        best_dir = CHECKPOINT_DIR / f'best_{NODE_ID}'
        if best_dir.exists():
            for src_name, dst in [('fx_model_best.onnx',   BEST_ONNX),
                                   ('norm_params_best.json', BEST_NORM),
                                   ('best_result.json',       BEST_JSON)]:
                src = best_dir / src_name
                if src.exists():
                    shutil.copy2(src, dst)
        else:
            # 旧形式 (node_id なし) のフォールバック
            for src_name, dst in [('fx_model_best.onnx',   BEST_ONNX),
                                   ('norm_params_best.json', BEST_NORM),
                                   ('best_result.json',       BEST_JSON)]:
                src = CHECKPOINT_DIR / src_name
                if src.exists():
                    shutil.copy2(src, dst)

        # top100 を全ノード分まとめて合成 → TOP_DIR に展開
        _restore_merged_top100()
        print('  [CKPT] 復元完了 → 全ノードの結果を統合して再開します')

        # 復元後に特徴量重要度バックフィルをバックグラウンド実行
        def _run_backfill():
            import importlib.util, time as _t
            _t.sleep(2)
            try:
                spec = importlib.util.spec_from_file_location(
                    'backfill_top100', str(OUT_DIR / 'backfill_top100.py'))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.main()
            except Exception as _e:
                print(f'  [BACKFILL] エラー: {_e}')
        threading.Thread(target=_run_backfill, daemon=True).start()

        return True
    except Exception as e:
        print(f'  [CKPT] 復元失敗: {e}')
        return False


def _restore_merged_top100() -> None:
    """全ノードの top100_<NODE_ID>/ をマージして TOP_DIR に展開"""
    # ノードごとの top100 ディレクトリを収集
    top_dirs = [d for d in CHECKPOINT_DIR.iterdir()
                if d.is_dir() and d.name.startswith('top100_')]
    # 旧形式 (top100/) も考慮
    legacy = CHECKPOINT_DIR / 'top100'
    if legacy.exists() and legacy.is_dir():
        top_dirs.append(legacy)
    if not top_dirs:
        return

    # 全モデルを PF 降順でソートして上位 TOP_N を TOP_DIR に展開
    all_models: list[tuple[float, Path]] = []
    for td in top_dirs:
        for rank_dir in td.iterdir():
            if not rank_dir.is_dir():
                continue
            rf = rank_dir / 'result.json'
            if rf.exists():
                try:
                    r = json.loads(rf.read_text(encoding='utf-8'))
                    all_models.append((r.get('pf', 0), rank_dir))
                except Exception:
                    pass
    all_models.sort(key=lambda x: -x[0])

    TOP_DIR.mkdir(parents=True, exist_ok=True)
    TOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_restored = 0
    for rank_i, (pf, src_dir) in enumerate(all_models[:TOP_N], 1):
        rank_dst = TOP_DIR / f'rank_{rank_i:03d}'
        if rank_dst.exists():
            shutil.rmtree(rank_dst)
        shutil.copytree(src_dir, rank_dst)
        # top_cache にも trial 番号でコピー
        rf = rank_dst / 'result.json'
        if rf.exists():
            try:
                r = json.loads(rf.read_text(encoding='utf-8'))
                tno = r.get('trial', 0)
                nid = r.get('node_id', NODE_ID)
                if tno > 0:
                    cache_key = f'trial_{nid}_{tno:06d}'
                    cache_dst = TOP_CACHE_DIR / cache_key
                    if not cache_dst.exists():
                        cache_dst.mkdir(parents=True, exist_ok=True)
                        for fn in ['fx_model.onnx', 'norm_params.json', 'report.html']:
                            fs = rank_dst / fn
                            if fs.exists():
                                shutil.copy2(fs, cache_dst / fn)
                        cache_restored += 1
            except Exception:
                pass
    if cache_restored:
        print(f'  [CKPT] top_cache に {cache_restored}件 復元 (全ノード合算)')


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
    global _mode_start_time, _important_features

    # SIGTERM (コンテナ停止時) を受け取ったら stop.flag を置いてgraceful shutdown
    def _sigterm_handler(signum, frame):
        print('\n[SIGNAL] SIGTERM 受信 → チェックポイント保存して停止します...')
        STOP_FLAG.touch()
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT,  _sigterm_handler)

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    TOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TOP_DIR.mkdir(parents=True, exist_ok=True)

    gpu_name = NODE_ID.upper()
    print('=' * 60)
    storage_tag = 'GDrive' if GDRIVE_ENABLED else ('S3' if S3_ENABLED else 'ローカルのみ')
    print(f'FX AI EA v8 - 並列ランダムサーチ  ストレージ: {storage_tag}/{NODE_ID}')
    print(f'  GPU     : {gpu_name}  ({_GPU_VRAM_GB:.0f} GB)  tier={_GPU_TIER}')
    print(f'  並列数  : {MAX_PARALLEL}  VRAM/試行={VRAM_PER_TRIAL} GB  H100_MODE={H100_MODE}')
    print(f'  モデル  : hidden={[v for v in HIDDEN_MAP.get("mlp",[])]}  batch={BATCH_CHOICES}')
    print(f'  TOP {TOP_N} 保存  タイムアウト {TRIAL_TIMEOUT//60}分  stop.flag: {STOP_FLAG}')
    print('=' * 60)

    # ── ストレージ接続確認 ─────────────────────────────────────────────────────
    print(f'  GDRIVE_ENABLED: {GDRIVE_ENABLED}')
    print(f'  S3_ENABLED    : {S3_ENABLED}  (S3_ENDPOINT: {S3_ENDPOINT or "(未設定)"})')
    if GDRIVE_ENABLED:
        _gdrive.test_connection()
    elif S3_ENABLED:
        try:
            cl = _s3_client()
            cl.put_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/.ping', Body=b'ok')
            cl.delete_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/.ping')
            print('  [S3] 接続テスト OK ✅')
        except Exception as e:
            print(f'  [S3] 接続テスト 失敗 ❌: {e}')
    else:
        print('  [ストレージ] 未設定 → ローカルのみ (GDRIVE_FOLDER_ID + GDRIVE_CREDENTIALS_BASE64 を設定)')

    # ── 起動時にデータキャッシュを事前作成 (全試行が即座に学習開始できる) ──
    _precache_data()

    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    rng     = random.Random()
    # ────────────────────────────────────────────────────────────────────────
    # WorkerPool (ProcessPoolExecutor) vs ParallelTrainer (subprocess.Popen)
    #
    # Windows では ProcessPoolExecutor の future.cancel() は実行中のワーカーを
    # 止められない (Linux と異なり SIGKILL が届かない)。
    # タイムアウト検出後もワーカーがスロットを占有し続けるため、
    # 新規試行がキュー詰まり → 完了数が止まる (stuck-at-N 症状)。
    # → Windows ではサブプロセスモード (ParallelTrainer) を強制使用する。
    # Linux/Docker 環境では WorkerPool の起動コスト削減メリットを活かす。
    # ────────────────────────────────────────────────────────────────────────
    _cache_pkl  = TRIALS_DIR.parent / 'df_cache_H1.pkl'
    _on_windows = platform.system() == 'Windows'
    if _cache_pkl.exists() and not _on_windows:
        try:
            trainer = WorkerPool(MAX_PARALLEL, _cache_pkl)
        except Exception as _e:
            print(f"  [WARN] WorkerPool 初期化失敗 → subprocess フォールバック: {_e}")
            trainer = ParallelTrainer()
    else:
        if _on_windows:
            print("  [INFO] Windows 環境 → サブプロセスモード使用 "
                  "(WorkerPool は Linux/Docker 専用: proc.terminate() で確実タイムアウト)")
        else:
            print("  [INFO] キャッシュなし → subprocess モードで起動")
        trainer = ParallelTrainer()
    results  = []
    best_pf  = 0.0
    trial_no = 1
    start    = time.time()
    _mode_start_time = start   # 10分モード切り替えの基準時刻

    # ── チェックポイントから復元 (ディスクマウント時は自動継続) ──────────────
    if not ALL_RESULTS.exists():
        restore_checkpoint()

    # ── 他ノード結果の定期マージスレッド (5分ごと) ─────────────────────────
    _other_merge_stop = threading.Event()
    def _other_nodes_merge_loop():
        """バックグラウンドで他ノードの新着結果を取り込んでマージ"""
        while not _other_merge_stop.wait(300):   # 5分ごと
            if not REMOTE_ENABLED():
                continue
            try:
                other = fetch_other_nodes_results()
                if not other:
                    continue
                new_count = 0
                for r in other:
                    nid  = r.get('node_id', '?')
                    tno  = r.get('trial', 0)
                    key  = (nid, tno)
                    if not any(x.get('node_id') == nid and x.get('trial') == tno
                               for x in results):
                        results.append(r)
                        new_count += 1
                if new_count:
                    results.sort(key=lambda x: x.get('trial', 0))
                    print(f'  [SYNC] 他ノード結果を {new_count}件 取り込み '
                          f'(合計 {len(results)}件)')
                    try:
                        rebuild_top_n(results)
                    except Exception:
                        pass
            except Exception as e:
                print(f'  [SYNC] 他ノードマージ失敗: {e}')
    _sync_thread = threading.Thread(target=_other_nodes_merge_loop, daemon=True, name='NodeSync')
    _sync_thread.start()

    # 既存結果を引き継ぐ
    if ALL_RESULTS.exists():
        try:
            raw = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
            # ── 重複排除: (node_id, trial) でユニーク ───────────────────────
            seen_key: set = set()
            results = []
            for r in raw:
                nid_r = r.get('node_id', NODE_ID)
                tno_r = r.get('trial', -1)
                key_r = (nid_r, tno_r)
                if key_r not in seen_key:
                    seen_key.add(key_r)
                    results.append(r)

            if len(raw) != len(results):
                print(f"  [DEDUP] 重複除去: {len(raw)} → {len(results)} 件")
                tmp = ALL_RESULTS.with_suffix('.tmp')
                tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding='utf-8')
                tmp.replace(ALL_RESULTS)
            # 自ノードの最大 trial 番号から次の trial 番号を決定
            own_trials = [r.get('trial', 0) for r in results
                          if r.get('node_id', NODE_ID) == NODE_ID]
            trial_no = max(own_trials, default=0) + 1
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
    _loop_errors = 0
    while True:
      try:
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

            # ハイパーパラメータのみ展開 (trial/pf 等の結果メタデータは除外して上書きを防ぐ)
            _meta_keys = {'trial', 'pf', 'trades', 'sr', 'max_dd', 'win_rate',
                          'net_pnl', 'gross_profit', 'gross_loss', 'elapsed_sec',
                          'timestamp', 'strategy', 'node_id'}
            record = {
                **{k: v for k, v in info['params'].items() if k not in _meta_keys},
                'trial':     tno,
                'node_id':   NODE_ID,          # ノードID (マージ時の識別子)
                'gpu_name':  GPU_NAME,
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
                'feature_importance': r.get('feature_importance', []),
            }
            # 重複防止: 同じ (node_id, trial) がすでにあれば上書き、なければ追加
            existing_idx = next((i for i, r in enumerate(results)
                                 if r['trial'] == tno and r.get('node_id', NODE_ID) == NODE_ID), None)
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

            # 重要特徴量プールを更新 (5件ごと)
            if len(results) % 5 == 0:
                try:
                    _update_important_features(results)
                    if _important_features:
                        print(f"  [IMP] 重要特徴量TOP5: {_important_features[:5]}")
                except Exception as e:
                    print(f"  [WARN] 重要特徴量更新失敗: {e}")

            # 10分モード切り替えログ
            if _mode_start_time > 0:
                elapsed_total = time.time() - _mode_start_time
                cycle   = int(elapsed_total // MODE_SWITCH_SEC)
                remain  = int(MODE_SWITCH_SEC - (elapsed_total % MODE_SWITCH_SEC))
                if remain <= 5:   # 切り替え直前に通知
                    next_mode = 'GA' if cycle % 2 == 0 else 'ランダム'
                    print(f"  [MODE] まもなく {next_mode} モードに切り替え...")

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
                # ベスト更新時に重要特徴量プールも即時更新
                try:
                    _update_important_features(results)
                    if _important_features:
                        print(f"  [IMP] ベスト更新 → 重要特徴量更新: {_important_features[:5]}")
                except Exception:
                    pass
                print(f"  [BEST] 試行#{tno}  PF={pf:.4f}  SR={sr:.3f}  MaxDD={max_dd:.4f}")
            else:
                print(f"  [DONE] 試行#{tno:4d}  PF={pf:.4f}  SR={sr:.3f}  "
                      f"MaxDD={max_dd:.4f}  取引={trades}  "
                      f"{elapsed/60:.1f}分  (ベスト={best_pf:.4f})")

        # ── 新規試行を投入 ──────────────────────────────────────────────────
        n_active = trainer.n_active_workers if isinstance(trainer, WorkerPool) else len(trainer)
        max_par = get_max_parallel(n_active)
        # WorkerPool (ProcessPoolExecutor) の場合はダブルバッファリング:
        # ワーカー数×2 をキューに保持 → ワーカーが終わった瞬間に次ジョブ開始
        submit_limit = max_par * 2 if isinstance(trainer, WorkerPool) else max_par
        while len(trainer) < submit_limit:
            if STOP_FLAG.exists():
                break
            try:
                p, strategy = next_params(results, rng)
            except Exception as _np_exc:
                print(f"  [WARN] next_params 失敗: {_np_exc} → ランダムサンプルにフォールバック")
                p, strategy = sample_params(rng), 'random'
            try:
                trainer.launch(trial_no, p, best_pf, start, strategy)
            except Exception as _launch_exc:
                print(f"  [WARN] launch 失敗: {_launch_exc} → スキップ")
                time.sleep(5)
                break
            trial_no += 1
            if isinstance(trainer, WorkerPool):
                time.sleep(0.1)   # キュー投入は高速でOK (CUDA初期化済み)
            else:
                time.sleep(LAUNCH_INTERVAL)

        # ── 進捗 JSON 書き込み (5秒ごと) ───────────────────────────────────
        write_progress(trainer.running, results, best_pf, start)

        # ── チェックポイント保存: 10試行ごと or 10分ごと ────────────────────
        should_ckpt = (completed_since_ckpt >= CHECKPOINT_EVERY_N or
                       time.time() - last_checkpoint >= CHECKPOINT_INTERVAL)
        if should_ckpt:
            save_checkpoint(results, best_pf)
            last_checkpoint      = time.time()
            completed_since_ckpt = 0

        time.sleep(1 if isinstance(trainer, WorkerPool) else 5)
        _loop_errors = 0  # 正常ループが回ればエラーカウントリセット

      except KeyboardInterrupt:
          print(f"\n[STOP] KeyboardInterrupt → 終了処理")
          trainer.terminate_all()
          break
      except Exception as _loop_exc:
          import traceback as _tb
          _loop_errors += 1
          print(f"  [ERROR] メインループ例外 ({_loop_errors}回目): "
                f"{type(_loop_exc).__name__}: {_loop_exc}")
          _tb.print_exc()
          if _loop_errors >= 10:
              print(f"  [ERROR] 連続エラー10回 → 終了")
              trainer.terminate_all()
              break
          time.sleep(5)
          continue

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
