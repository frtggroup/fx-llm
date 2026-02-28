"""
CUDA GPU 学習スクリプト v3
- 70次元特徴量 × H1データ
- 10種アーキテクチャ対応
- 時間重み付け・過学習早期終了

使用方法:
    py -3.12 train.py [オプション]
"""
import os, sys, json, argparse, time, threading
from pathlib import Path
from datetime import timedelta

# ── 非同期ファイル書き込みエグゼキューター ─────────────────────────────────────
# GPU訓練ループをブロックしないよう、ファイルI/OはバックグラウンドスレッドでOK
_last_write_thread: threading.Thread | None = None

def _async_write(path: Path, text: str) -> None:
    """ノンブロッキング: バックグラウンドで tmp→rename アトミック書き込み"""
    global _last_write_thread
    def _do():
        try:
            tmp = path.with_suffix('.tmp')
            tmp.write_text(text, encoding='utf-8')
            tmp.replace(path)
        except Exception:
            pass
    # 前のスレッドが生きていてもスキップせず新スレッドで上書き
    _last_write_thread = threading.Thread(target=_do, daemon=True)
    _last_write_thread.start()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa (DataLoader は backtest で使用)

sys.path.insert(0, str(Path(__file__).parent))
from features import load_data, add_indicators, build_dataset, FEATURE_COLS, N_FEATURES
from feature_sets import FEATURE_SETS
import random as _random
from model    import build_model, FXPredictorWithNorm, export_onnx, verify_onnx, ARCH_MAP
from dashboard import update_dashboard

_DEFAULT_DATA = str(Path(__file__).parent.parent / 'USDJPY_M1_202301020700_202602262003.csv')
DATA_PATH = Path(os.environ.get('DATA_PATH', _DEFAULT_DATA))
OUT_DIR   = Path(__file__).parent
ONNX_PATH = OUT_DIR / 'fx_model.onnx'
NORM_PATH = OUT_DIR / 'norm_params.json'
SPREAD    = 0.005  # ラウンドトリップ 1.0pips = 0.010円
                  # エントリー時・エグジット時の2回引くため 0.01/2 = 0.005 に設定

# ── バックテスト 資金設定 ──────────────────────────────────────────────────
BT_CAPITAL  = float(os.environ.get('BT_CAPITAL',  '150000'))  # スタート資金 (円)
BT_LEVERAGE = float(os.environ.get('BT_LEVERAGE', '1000'))    # レバレッジ倍率
BT_RISK_PCT = float(os.environ.get('BT_RISK_PCT', '1.0'))     # リスク率 (%)


def _calc_lot(equity_yen: float, risk_pct: float = BT_RISK_PCT) -> float:
    """MQL5 LotSize() 相当: 口座残高の risk_pct % を使うロット数を返す
    JPY口座: magnification=10000
    例) 150,000円 × 1% → MathCeil(150000*1/10000)/100 - 0.01 = 0.14 lot
    """
    import math
    magnification = 10000  # JPY口座
    lot = math.ceil(equity_yen * risk_pct / magnification) / 100.0
    lot = lot - 0.01
    if lot < 0.01:
        lot = 0.01
    if lot > 1.0:
        lot = math.ceil(lot)
    if lot > 20.0:
        lot = 20.0
    return lot

_dash: dict = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=int,   default=500)
    p.add_argument('--hidden',    type=int,   default=24)
    p.add_argument('--layers',    type=int,   default=1)
    p.add_argument('--dropout',   type=float, default=0.5)
    p.add_argument('--lr',        type=float, default=5e-4)
    p.add_argument('--batch',     type=int,   default=128)
    p.add_argument('--tp',        type=float, default=1.5)
    p.add_argument('--sl',        type=float, default=1.0)
    p.add_argument('--forward',   type=int,   default=20)
    p.add_argument('--threshold', type=float, default=0.40)
    p.add_argument('--timeframe', type=str,   default='H1')
    p.add_argument('--seq_len',   type=int,   default=20,  help='シーケンス長')
    p.add_argument('--arch',       type=str,   default='gru_attn',
                   help=f'アーキテクチャ: {list(ARCH_MAP.keys())}')
    p.add_argument('--label_type',    type=str,   default='triple_barrier',
                   help='triple_barrier | directional')
    p.add_argument('--wd',            type=float, default=1e-2,  help='weight_decay')
    p.add_argument('--scheduler',     type=str,   default='onecycle',
                   help='onecycle | cosine | step')
    p.add_argument('--train_months',  type=int,   default=0,
                   help='訓練期間を直近N月に限定 (0=全期間)')
    p.add_argument('--feat_frac',     type=float, default=1.0,
                   help='(後方互換) 使用する特徴量の割合')
    p.add_argument('--n_features',    type=int,   default=0,
                   help='使用する特徴量数 (0=全70, 2-70で指定)')
    p.add_argument('--feat_set',      type=int,   default=-2,
                   help='feature_sets.py のセット番号 (0-99), -1=ランダム数, -2=未指定')
    p.add_argument('--seed',          type=int,   default=42)
    # run_train.py から渡される追加引数
    p.add_argument('--trial',       type=int,   default=1)
    p.add_argument('--total_trials',type=int,   default=1)
    p.add_argument('--best_pf',     type=float, default=0.0)
    p.add_argument('--start_time',  type=float, default=0.0)
    p.add_argument('--out_dir',     type=str,   default='',
                   help='出力ディレクトリ (並列モード用; 省略時は ai_ea/ 直下)')
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        xm.set_rng_state(torch.manual_seed(seed).get_state()[:])
    except Exception:
        pass


# ─── データ準備 ────────────────────────────────────────────────────────────
_DF_CACHE: dict = {}        # {timeframe: (df_tr, df_te)} プロセス内キャッシュ
_WORKER_STATE: dict = {}    # ProcessPoolExecutor ワーカーが保持する常駐データ


def prepare(args):
    print(f"\n=== データ準備 [{args.timeframe}] ===")
    t0 = time.time()

    # WorkerPool 常駐ワーカーならメモリ上のデータを直接使用 (ディスクI/Oゼロ)
    if _WORKER_STATE.get('df_tr') is not None:
        df_tr_raw = _WORKER_STATE['df_tr']
        df_te     = _WORKER_STATE['df_te']
        print(f"  [WORKER-CACHE HIT] {len(df_tr_raw):,} / {len(df_te):,}")
        # 後続の train_months フィルタに備えてコピーを返す
        df_tr = df_tr_raw
        # → 以下のキャッシュ処理はスキップして train_months フィルタへ直行
        _DF_CACHE[args.timeframe] = (df_tr_raw, df_te)  # 通常パスとの互換性

    # 同一プロセス内でのキャッシュ（シングルモード用）
    # 並列モードでは各サブプロセスが独自にキャッシュを持つ
    cache_key = args.timeframe
    if cache_key in _DF_CACHE:
        df_tr, df_te = _DF_CACHE[cache_key]
        print(f"  データキャッシュHIT ({len(df_tr):,} / {len(df_te):,})")
    else:
        # ディスクキャッシュ: 並列試行間で共有 (ファイルロックで競合防止)
        cache_path = OUT_DIR.parent / f'df_cache_{args.timeframe}.pkl'
        lock_path  = cache_path.with_suffix('.lock')
        import pickle, fcntl

        def _load_cache():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        def _build_and_save():
            df = load_data(str(DATA_PATH), timeframe=args.timeframe)
            df = add_indicators(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            test_start = df.index[-1] - timedelta(days=365)
            dtr = df[df.index < test_start].copy()
            dte = df[df.index >= test_start].copy()
            try:
                tmp = cache_path.with_suffix('.tmp')
                with open(tmp, 'wb') as f:
                    pickle.dump((dtr, dte), f)
                tmp.replace(cache_path)
                print(f"  ディスクキャッシュ保存: {cache_path}")
            except Exception as e:
                print(f"  キャッシュ保存スキップ: {e}")
            return dtr, dte

        if cache_path.exists():
            print(f"  ディスクキャッシュ読み込み中...")
            try:
                df_tr, df_te = _load_cache()
                print(f"  キャッシュ読み込み完了 ({len(df_tr):,} / {len(df_te):,})  {time.time()-t0:.1f}秒")
            except Exception:
                # 破損キャッシュ → 再構築
                print(f"  キャッシュ破損 → 再構築します")
                cache_path.unlink(missing_ok=True)
                df_tr, df_te = _build_and_save()
        else:
            # ファイルロックで1プロセスだけが構築し、他は待機して読む
            print(f"  キャッシュ未作成 → 排他ロックで構築中...")
            try:
                lock_fh = open(lock_path, 'w')
                fcntl.flock(lock_fh, fcntl.LOCK_EX)   # 排他ロック取得
                try:
                    if cache_path.exists():   # ロック待ち中に他が作成済み
                        df_tr, df_te = _load_cache()
                        print(f"  他プロセスが作成済みキャッシュを読み込み ({len(df_tr):,}/{len(df_te):,})")
                    else:
                        df_tr, df_te = _build_and_save()
                finally:
                    fcntl.flock(lock_fh, fcntl.LOCK_UN)
                    lock_fh.close()
            except (ImportError, AttributeError):
                # Windows など fcntl 非対応環境はロックなしで実行
                if cache_path.exists():
                    df_tr, df_te = _load_cache()
                else:
                    df_tr, df_te = _build_and_save()

        _DF_CACHE[cache_key] = (df_tr, df_te)

    # 訓練期間を最近N月に絞る (分布シフト対策)  ※元 df_tr をコピーして使う
    df_tr = df_tr.copy()
    tm = getattr(args, 'train_months', 0)
    if tm > 0:
        train_start = df_tr.index[-1] - timedelta(days=tm * 30)
        df_tr = df_tr[df_tr.index >= train_start].copy()
        print(f"  訓練期間を直近{tm}ヶ月に限定: {df_tr.index[0].date()} ～")
    print(f"  訓練: {len(df_tr):,}  テスト: {len(df_te):,}")

    print(f"  ラベル生成中... (triple_barrier)")
    seq_len   = args.seq_len

    # 特徴量セット選択 (優先順: feat_indices(直接指定) > feat_set > n_features > feat_frac > 全部)
    feat_set_id = getattr(args, 'feat_set', -2)
    n_feat_arg  = getattr(args, 'n_features', 0)
    feat_frac   = getattr(args, 'feat_frac', 1.0)
    feat_indices_direct = getattr(args, 'feat_indices', None)

    if feat_indices_direct and isinstance(feat_indices_direct, list):
        # GAが重要特徴量から直接指定したインデックスリスト
        feat_indices = sorted(int(i) for i in feat_indices_direct if 0 <= int(i) < N_FEATURES)
        print(f"  特徴量直指定: {len(feat_indices)}個 (重要特徴量GAモード)")
    elif 0 <= feat_set_id <= 99:
        # 設計済みセットを使用
        feat_indices = FEATURE_SETS[feat_set_id]
        print(f"  特徴量セット#{feat_set_id+1}: {len(feat_indices)}特徴量")
    elif n_feat_arg > 0 and n_feat_arg < N_FEATURES:
        # ランダム N 個
        n_sel = max(2, min(n_feat_arg, N_FEATURES))
        _random.seed(args.seed)
        feat_indices = sorted(_random.sample(range(N_FEATURES), n_sel))
        print(f"  特徴量ランダム: {n_sel}/{N_FEATURES} (seed={args.seed})")
    elif feat_frac < 1.0:
        n_sel = max(2, int(N_FEATURES * feat_frac))
        _random.seed(args.seed)
        feat_indices = sorted(_random.sample(range(N_FEATURES), n_sel))
        print(f"  特徴量サブセット: {n_sel}/{N_FEATURES} (frac={feat_frac:.0%})")
    else:
        feat_indices = None
        print(f"  特徴量: 全{N_FEATURES}次元")

    X_tr, y_tr, _ = build_dataset(df_tr, seq_len, args.tp, args.sl, args.forward,
                                   feat_indices=feat_indices)
    X_te, y_te, _ = build_dataset(df_te, seq_len, args.tp, args.sl, args.forward,
                                   feat_indices=feat_indices)

    # 正規化 (訓練データのみで計算)
    n_feat_actual = X_tr.shape[2]          # サブセット後の実際の特徴量数
    flat = X_tr.reshape(-1, n_feat_actual)
    mean = flat.mean(0).astype(np.float32)
    std  = flat.std(0).astype(np.float32)
    std[std < 1e-8] = 1.0

    with open(NORM_PATH, 'w', encoding='utf-8') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist(),
                   'feature_cols': (FEATURE_COLS if feat_indices is None
                                    else [FEATURE_COLS[i] for i in feat_indices]),
                   'feat_indices': feat_indices,
                   'timeframe': args.timeframe,
                   'seq_len': seq_len}, f, indent=2)

    print(f"  データ準備: {time.time()-t0:.1f}秒")
    return X_tr, y_tr, X_te, y_te, mean, std, df_te, seq_len, feat_indices


# ─── GPU 常駐データローダー ──────────────────────────────────────────────────
class GPULoader:
    """
    全データを GPU に常駐させてバッチを GPU 上で切り出す。
    CPU↔GPU 転送オーバーヘッドをゼロにして GPU 使用率を最大化。
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 device: torch.device, batch_size: int,
                 weights: np.ndarray = None, shuffle: bool = False):
        self.X   = torch.tensor(X, dtype=torch.float32, device=device)
        self.y   = torch.tensor(y, dtype=torch.long,    device=device)
        self.n   = len(y)
        self.bs  = batch_size
        self.dev = device
        self.shuf = shuffle
        # 重み付きサンプリング用 (GPU上で multinomial)
        if weights is not None:
            self.w = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            self.w = None

    def __len__(self):
        return max(1, (self.n + self.bs - 1) // self.bs)

    def __iter__(self):
        if self.w is not None:
            idx = torch.multinomial(self.w, self.n, replacement=True)
        elif self.shuf:
            idx = torch.randperm(self.n, device=self.dev)
        else:
            idx = torch.arange(self.n, device=self.dev)
        for start in range(0, self.n, self.bs):
            sl = idx[start:start + self.bs]
            yield self.X[sl], self.y[sl]


def make_loaders(X_tr, y_tr, X_te, y_te, args, device):
    counts = np.bincount(y_tr, minlength=3).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    n      = len(y_tr)
    tw     = 1.0 + np.linspace(0, 1, n)          # 新データ重視
    weights = (1.0 / counts)[y_tr] * tw

    tr_dl = GPULoader(X_tr, y_tr, device, args.batch, weights=weights)
    va_dl = GPULoader(X_te, y_te, device, args.batch, shuffle=False)
    print(f"  GPU常駐データ: 訓練{len(X_tr):,} テスト{len(X_te):,} "
          f"batch={args.batch} batches/ep={len(tr_dl)}")
    return tr_dl, va_dl


# ─── 学習 ────────────────────────────────────────────────────────────────────
def _detect_device():
    """実行デバイスを検出。TPU (XLA) → GPU (CUDA) → CPU の優先順で試みる。"""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm.xla_device(), 'xla'
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    return torch.device('cpu'), 'cpu'


def train(args, X_tr, y_tr, X_te, y_te, mean, std, n_feat=None):
    device, dev_type = _detect_device()
    print(f"\n=== 学習 [{device}] ===")

    # デバイス別の最適化設定
    is_h100    = False
    is_tpu     = (dev_type == 'xla')
    # TPU は BF16 ネイティブ。CUDA (CC9=H100/CC8=A100) も BF16 が高速
    amp_dtype  = torch.bfloat16 if is_tpu else torch.float16

    if dev_type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        cc       = torch.cuda.get_device_capability(0)
        print(f"  GPU: {gpu_name}  CC={cc[0]}.{cc[1]}")
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
        if cc[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
        if cc[0] >= 9:
            is_h100   = True
            amp_dtype = torch.bfloat16
            print(f"  H100 モード: BF16 + TF32 + torch.compile 有効")
        else:
            print(f"  FP16 AMP モード")
    elif is_tpu:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            tpu_info = xm.get_xla_supported_devices()
            print(f"  TPU モード: デバイス数={len(tpu_info)}  BF16 AMP 有効")
        except Exception:
            print(f"  TPU モード: BF16 AMP 有効")

    tr_dl, va_dl = make_loaders(X_tr, y_tr, X_te, y_te, args, device)

    # AMP 設定: TPU は BF16 (GradScaler 不要), CUDA FP16 は Scaler 使用
    use_amp    = True  # GPU/TPU ともに混合精度を使用
    use_scaler = (dev_type == 'cuda') and (amp_dtype == torch.float16)
    _amp_backend = dev_type if dev_type in ('cuda', 'xla') else 'cpu'
    scaler     = torch.amp.GradScaler('cuda', enabled=use_scaler) if dev_type == 'cuda' else None

    seq_len  = X_tr.shape[1]
    arch     = getattr(args, 'arch', 'gru_attn')
    n_in     = n_feat if n_feat is not None else N_FEATURES

    def _build_and_place():
        return build_model(arch, n_in, seq_len,
                           args.hidden, args.layers, args.dropout).to(device)

    try:
        model = _build_and_place()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        # OOM (CUDA) または XLA メモリ不足: バッチ半減・モデル縮小でリトライ
        print(f"  [OOM] モデル生成失敗 ({type(e).__name__}) → hidden/2, batch/2 でリトライ")
        if dev_type == 'cuda':
            torch.cuda.empty_cache()
        args.hidden = max(64, args.hidden // 2)
        args.batch  = max(256, args.batch  // 2)
        tr_dl, va_dl = make_loaders(X_tr, y_tr, X_te, y_te, args, device)
        model = _build_and_place()

    # H100: torch.compile で GPU カーネル効率を向上 (TPU は XLA が自動最適化するためスキップ)
    model_for_export = model
    _is_worker = bool(getattr(args, 'out_dir', ''))
    if is_h100 and not _is_worker and not is_tpu:
        compile_mode = 'max-autotune'
        try:
            compiled_model = torch.compile(model, mode=compile_mode)
            print(f"  torch.compile({compile_mode}) 有効 → ウォームアップ実行中...")
            _wup_x, _wup_y = next(iter(tr_dl))
            with torch.amp.autocast(_amp_backend, enabled=use_amp, dtype=amp_dtype):
                _ = compiled_model(_wup_x)
            del _wup_x, _wup_y
            torch.cuda.synchronize()
            print(f"  torch.compile ウォームアップ完了")
            model = compiled_model
        except Exception as e:
            print(f"  torch.compile スキップ: {e}")
    elif is_tpu:
        print(f"  XLA 自動最適化 (torch.compile スキップ)")
    elif is_h100 and _is_worker:
        print(f"  torch.compile スキップ (並列ワーカーモード: 探索優先)")

    n_params = sum(p.numel() for p in model.parameters())
    ratio    = len(X_tr) / max(n_params, 1)
    print(f"  arch={arch}  パラメータ数: {n_params:,}  サンプル/パラメータ比: {ratio:.1f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    wd        = getattr(args, 'wd', 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=wd)
    dtype_str = 'BF16' if amp_dtype == torch.bfloat16 else 'FP16'
    print(f"  AMP({dtype_str})={'ON' if use_amp else 'OFF'}  GradScaler={'ON' if use_scaler else 'OFF (BF16不要)'}")

    sched_name = getattr(args, 'scheduler', 'onecycle')
    if sched_name == 'cosine':
        # ウォームアップなし → val_loss 安定
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(tr_dl),
            eta_min=args.lr * 0.01,
        )
    elif sched_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 5), gamma=0.5,
        )
    else:  # onecycle
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 3,
            steps_per_epoch=len(tr_dl), epochs=args.epochs,
            pct_start=0.03, anneal_strategy='cos',
        )
    print(f"  scheduler={sched_name}")

    best_loss      = float('inf')
    best_state     = None
    warmup_end     = max(20, int(args.epochs * 0.05))  # 5% warmup
    patience       = max(30, int(args.epochs * 0.08))  # 8% → 800ep=64 (旧15%=120)
    overfit_pat    = 15                                 # 15ep連続で判定
    min_epochs     = warmup_end + 5
    no_imp         = 0
    overfit_count  = 0
    stop_reason    = ''
    recent_gaps    = []
    prev_v_loss    = float('inf')    # val_loss の前エポック値 (下降判定用)
    # バリデーション頻度: エポック数に応じて調整 (GPU稼働率向上)
    # 2000ep → 10ep毎 (200回), 800ep → 4ep毎
    val_every    = max(1, args.epochs // 200)
    # 進捗JSON書き込み頻度 (非同期だが頻度を下げることで辞書更新コストも削減)
    progress_every = max(5, args.epochs // 100)
    print(f"  early_stop: ep{min_epochs}～, patience={patience}, wd={wd}"
          f"  val_every={val_every}  progress_every={progress_every}")

    v_loss = float('inf')   # 最初のvalidationまでの初期値
    acc    = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        try:
            step_losses = [
                _train_step(model, xb, yb,
                            optimizer, criterion, scheduler, scaler,
                            use_amp, amp_dtype, use_scaler,
                            _amp_backend, is_tpu)
                for xb, yb in tr_dl
            ]
            t_loss = torch.stack(step_losses).mean().item()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if dev_type == 'cuda':
                torch.cuda.empty_cache()
            oom_msg = 'CUDA OOM' if dev_type == 'cuda' else 'XLA OOM'
            print(f"  [OOM] ep{epoch} {oom_msg} → 学習打ち切り: {e}")
            stop_reason = f'{oom_msg} (ep{epoch})'
            break

        # validation は val_every エポックごとに実行 (不要なGPU同期を削減)
        if epoch % val_every == 0 or epoch <= 5 or epoch == args.epochs:
            model.eval()
            v_loss_sum = torch.zeros(1, device=device)
            correct_sum = torch.zeros(1, device=device, dtype=torch.long)
            total_sum   = 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    with torch.amp.autocast(_amp_backend, enabled=use_amp, dtype=amp_dtype):
                        lo = model(xb)
                    v_loss_sum  += criterion(lo, yb).detach()
                    correct_sum += (lo.argmax(1) == yb).sum()
                    total_sum   += len(yb)
            v_loss = (v_loss_sum / len(va_dl)).item()   # ここで1回だけ同期
            acc    = correct_sum.item() / max(total_sum, 1)

        gap = v_loss - t_loss   # 過学習ギャップ

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Ep{epoch:4d}/{args.epochs}  "
                  f"tr={t_loss:.4f}  va={v_loss:.4f}  "
                  f"gap={gap:+.4f}  acc={acc:.3f}  lr={lr_now:.2e}")

        # 進捗更新 (progress_every エポックごと / 非同期書き込み)
        if epoch % progress_every == 0 or epoch <= 5:
            _dash['epoch']       = epoch
            _dash['total_epochs']= args.epochs
            _dash['train_loss']  = round(t_loss, 5)
            _dash['val_loss']    = round(v_loss, 5)
            _dash['accuracy']    = round(acc, 4)
            _dash.setdefault('epoch_log', []).append(
                {'epoch': epoch, 'train_loss': round(t_loss,6),
                 'val_loss': round(v_loss,6), 'acc': round(acc,4)})
            # 並列モード: trial_progress.json に非同期書き込み (GPUをブロックしない)
            _write_trial_progress(OUT_DIR, {
                'trial': args.trial, 'arch': getattr(args, 'arch', '?'),
                'hidden': getattr(args, 'hidden', 0),
                'epoch': epoch, 'total_epochs': args.epochs,
                'train_loss': round(t_loss, 5), 'val_loss': round(v_loss, 5),
                'accuracy': round(acc, 4), 'phase': 'training',
            })
            # シングルモード: HTML ダッシュボードも更新
            if not getattr(args, 'out_dir', ''):
                try: update_dashboard(_dash)
                except Exception: pass

        # ── val_loss 改善チェック ──────────────────────────────────────────
        if v_loss < best_loss - 1e-5:
            best_loss  = v_loss
            # GPU上に保持 (cpu().clone()はI/Oボトルネック → detach().clone()に変更)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        # ── 即時発散: val が爆発 かつ 下がっていない ────────────────────────
        # 意味のある改善 = 前エポック比で 1e-4 以上の減少 (微小ノイズを除外)
        val_falling = v_loss < prev_v_loss - 1e-4
        if epoch >= 10 and gap > 0.35 and v_loss > 1.30:
            stop_reason = f'即時発散 (ep{epoch}: gap={gap:+.4f}, val={v_loss:.4f})'
            break
        prev_v_loss = v_loss

        # ── 過学習検出 (min_epochs 経過後) ──────────────────────────────────
        if epoch >= min_epochs:
            recent_gaps.append(gap)
            if len(recent_gaps) > overfit_pat:
                recent_gaps.pop(0)

            # 条件1: gap が大きく持続 かつ val が意味ある改善なし
            if (len(recent_gaps) == overfit_pat
                    and all(g > 0.20 for g in recent_gaps)
                    and not val_falling):
                overfit_count += 1
                if overfit_count >= 3:
                    stop_reason = f'過学習検出 (gap={gap:+.4f} が {overfit_pat}ep継続)'
                    break
            else:
                overfit_count = 0

            # 条件2: val が改善なし かつ 過学習が顕著 (val下降中は免除)
            if gap > 0.25 and no_imp >= 20 and not val_falling:
                stop_reason = f'過学習早期終了 (gap={gap:+.4f}, no_imp={no_imp})'
                break

            # 条件3: patience 到達 (意味ある改善なし)
            if no_imp >= patience and not val_falling:
                stop_reason = f'改善なし早期終了 (patience={patience})'
                break

            # 条件4: patience の2倍を超えたら val_falling 免除で強制終了
            if no_imp >= patience * 2:
                stop_reason = f'改善なし強制終了 (no_imp={no_imp} >= patience*2={patience*2})'
                break

    if stop_reason:
        print(f"  [STOP] {stop_reason}  ep={epoch}  best_val={best_loss:.4f}")
    else:
        print(f"  [DONE] 全エポック完了  best_val={best_loss:.4f}")

    if best_state:
        # compiled model と元モデルの両方に best_state を反映
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass
        # ONNX export 用の元モデル (compile前) にも反映
        if model_for_export is not model:
            try:
                model_for_export.load_state_dict(best_state)
            except Exception:
                pass
    return model_for_export  # ONNX export には compile前モデルを返す


def _train_step(model, xb, yb, opt, crit, sched,
                scaler=None, use_amp=False,
                amp_dtype=torch.float16, use_scaler=False,
                amp_backend='cuda', is_tpu=False):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(amp_backend, enabled=use_amp, dtype=amp_dtype):
        loss = crit(model(xb), yb)
    if use_scaler and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if is_tpu:
            # TPU: XLA グラフをコンパイル・実行して同期
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                xm.optimizer_step(opt)
            except Exception:
                opt.step()
        else:
            opt.step()
    sched.step()
    return loss.detach()


# ─── バックテスト ─────────────────────────────────────────────────────────
def _generate_report(trades: list, equity: np.ndarray, dd: np.ndarray,
                     result: dict, out_dir: Path, trial_no: int) -> None:
    """資産曲線・DDチャートのスタンドアロン HTML を生成"""
    capital  = result.get('capital', BT_CAPITAL)
    lev      = result.get('leverage', BT_LEVERAGE)
    # 資産曲線・DDはすでに円ベース (ラベルは各取引の決済日付)
    eq_data    = [round(float(v), 0) for v in equity]
    dd_data    = [round(float(v), 0) for v in dd]
    trade_dates = [t.get('date', '') for t in trades]  # 取引日付ラベル
    # 日別損益 (円)
    daily: dict = {}
    for t in trades:
        d = t.get('date', '?')
        daily[d] = round(daily.get(d, 0.0) + t.get('pnl_yen', t['pnl']), 0)
    dl   = list(daily.keys())
    dv   = [round(daily[k], 0) for k in dl]
    dclr = [('rgba(63,185,80,.6)' if v >= 0 else 'rgba(248,81,73,.6)') for v in dv]
    pf_c    = '#f0883e' if result['pf'] >= 2 else '#3fb950' if result['pf'] >= 1.5 else '#ffa657' if result['pf'] >= 1.2 else '#f85149'
    net_pnl = result.get('net_pnl', 0)
    fin_eq  = result.get('final_equity', capital)
    ret_pct = result.get('return_pct', 0)
    mdd     = result.get('max_dd', 0)
    mdd_pct = result.get('max_dd_pct', 0)

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>Trial #{trial_no} Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:20px}}
h1{{color:#58a6ff;font-size:1.15em;margin-bottom:4px}}
.sub{{font-size:.72em;color:#8b949e;margin-bottom:14px}}
.stats{{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}}
.s{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px;text-align:center;min-width:100px}}
.sv{{font-size:1.45em;font-weight:700}}
.sl{{font-size:.7em;color:#8b949e;margin-top:3px}}
.cw{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:12px}}
.ct{{font-size:.7em;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}}
canvas{{max-height:250px}}
</style></head><body>
<h1>Trial #{trial_no} バックテストレポート</h1>
<div class="sub">資金: {capital:,.0f}円 / レバレッジ: {lev:.0f}倍 / リスク: {BT_RISK_PCT:.0f}%/取引 (動的ロット)</div>
<div class="stats">
  <div class="s"><div class="sv" style="color:{pf_c}">{result['pf']:.4f}</div><div class="sl">Profit Factor</div></div>
  <div class="s"><div class="sv" style="color:{'#3fb950' if net_pnl>=0 else '#f85149'}">{net_pnl:+,.0f}円</div><div class="sl">純利益</div></div>
  <div class="s"><div class="sv" style="color:{'#3fb950' if ret_pct>=0 else '#f85149'}">{ret_pct:+.1f}%</div><div class="sl">リターン</div></div>
  <div class="s"><div class="sv" style="color:#79c0ff">{result.get('sr',0):.3f}</div><div class="sl">Sharpe Ratio</div></div>
  <div class="s"><div class="sv" style="color:#f85149">{mdd:,.0f}円<br><span style="font-size:.55em">({mdd_pct:.1f}%)</span></div><div class="sl">最大 DD</div></div>
  <div class="s"><div class="sv" style="color:#58a6ff">{fin_eq:,.0f}円</div><div class="sl">最終資産</div></div>
  <div class="s"><div class="sv">{result['trades']}</div><div class="sl">取引数</div></div>
  <div class="s"><div class="sv" style="color:#3fb950">{result['win_rate']*100:.1f}%</div><div class="sl">勝率</div></div>
</div>
<div class="cw"><div class="ct">資産曲線 (累積損益)</div>
  <canvas id="eq"></canvas></div>
<div class="cw"><div class="ct">ドローダウン</div>
  <canvas id="dd"></canvas></div>
<div class="cw"><div class="ct">日別損益</div>
  <canvas id="dl"></canvas></div>
<script>
const eq={json.dumps(eq_data)};
const eql={json.dumps(trade_dates)};
const ddv={json.dumps(dd_data)};
const dlv={json.dumps(dv)};
const dll={json.dumps(dl)};
const dlc={json.dumps(dclr)};
function mc(id,lbl,data,color,fill,type='line'){{
  new Chart(document.getElementById(id),{{
    type,data:{{labels:lbl,datasets:[{{data,borderColor:color,
      backgroundColor:fill,borderWidth:type==='bar'?0:1.5,
      pointRadius:0,fill:type==='line',tension:.1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{{legend:{{display:false}},
               tooltip:{{callbacks:{{label:ctx=>ctx.parsed.y.toLocaleString('ja-JP')+'円'}}}}}},
      scales:{{x:{{ticks:{{color:'#8b949e',maxTicksLimit:14,maxRotation:45}},grid:{{color:'#21262d'}}}},
              y:{{ticks:{{color:'#8b949e',callback:v=>v.toLocaleString('ja-JP')+'円'}},grid:{{color:'#21262d'}}}}}}}}
  }});
}}
mc('eq',eql,eq,'#3fb950','#3fb95018');
mc('dd',eql,ddv,'#f85149','#f8514918');
mc('dl',dll,dlv,'rgba(0,0,0,0)','rgba(0,0,0,0)','bar');
// 日別バーは個別色
const dChart=Chart.getChart('dl');
if(dChart)dChart.data.datasets[0].backgroundColor=dlc,dChart.update('none');
</script>
</body></html>"""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'report.html').write_text(html, encoding='utf-8')
    except Exception as e:
        print(f"  [WARN] レポート生成失敗: {e}")


_BT_MODE: str | None = None  # キャッシュ: 'cpu' / 'cuda_iobind' / 'cuda'
# Windows では /tmp が存在しないため tempfile.gettempdir() を使用する
import tempfile as _tempfile
_BT_MODE_CACHE_FILE = Path(_tempfile.gettempdir()) / 'fx_bt_mode.txt'

def _bench_onnx_providers(onnx_path: str, full_data: np.ndarray) -> str:
    """CUDA利用可能時はCUDA vs CUDA+IOBindingのみ比較（CPUベンチはスキップ）。
    結果はインメモリ + ファイルキャッシュで再利用し、並列試行での再計測を防ぐ。"""
    global _BT_MODE
    if _BT_MODE is not None:
        return _BT_MODE

    # ── ファイルキャッシュ確認（並列サブプロセス間で共有）──────────────
    if _BT_MODE_CACHE_FILE.exists():
        try:
            cached = _BT_MODE_CACHE_FILE.read_text().strip()
            if cached in ('cpu', 'cuda', 'cuda_iobind'):
                _BT_MODE = cached
                print(f'  [BT-bench] キャッシュ使用: {_BT_MODE}')
                return _BT_MODE
        except Exception:
            pass

    import onnxruntime as ort
    import time

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available:
        _BT_MODE = 'cpu'
        print('  [BT-bench] CUDA unavailable → CPU')
        _BT_MODE_CACHE_FILE.write_text(_BT_MODE)
        return _BT_MODE

    # ── CUDA利用可能: CPUベンチはスキップしCUDA vs CUDA+IOBindのみ比較 ──
    # CUDAが常にCPUより高速なため、CPUの38秒ベンチを省略して高速化
    x = np.ascontiguousarray(full_data[:1024])  # 1024サンプルで十分
    cuda_pvd = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    results = {}

    # ── CUDA (通常バッチ転送) ───────────────────────────────────────────
    try:
        s = ort.InferenceSession(onnx_path, providers=cuda_pvd)
        s.run(None, {'input': x[:64]})   # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(3):
            s.run(None, {'input': x})[0]
        results['cuda'] = (time.perf_counter() - t0) / 3
    except Exception as e:
        print(f'  [BT-bench] CUDA error: {e}')
        results['cuda'] = float('inf')

    # ── CUDA + IOBinding (データをGPUメモリに常駐) ─────────────────────
    try:
        s2 = ort.InferenceSession(onnx_path, providers=cuda_pvd)
        io = s2.io_binding()
        x_ort   = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        out_ort = ort.OrtValue.ortvalue_from_shape_and_type((len(x), 3), np.float32, 'cuda', 0)
        io.bind_ortvalue_input('input', x_ort)
        io.bind_ortvalue_output('output', out_ort)
        s2.run_with_iobinding(io)   # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(3):
            s2.run_with_iobinding(io)
        _ = out_ort.numpy()
        results['cuda_iobind'] = (time.perf_counter() - t0) / 3
    except Exception as e:
        print(f'  [BT-bench] CUDA+IOBind error: {e}')
        results['cuda_iobind'] = float('inf')

    _BT_MODE = min(results, key=results.get)
    print(f'  [BT-bench] CUDA={results.get("cuda",99)*1000:.1f}ms  '
          f'CUDA+IOBind={results.get("cuda_iobind",99)*1000:.1f}ms  '
          f'→ {_BT_MODE} を採用')
    # ファイルキャッシュに保存（以降の全試行がスキップ可能）
    try:
        _BT_MODE_CACHE_FILE.write_text(_BT_MODE)
    except Exception:
        pass
    return _BT_MODE


def _run_inference(sess, x: np.ndarray, mode: str) -> np.ndarray:
    """採用モードに応じて全データ一括推論し確率配列を返す"""
    import onnxruntime as ort
    n = len(x)

    if mode == 'cuda_iobind':
        try:
            io = sess.io_binding()
            x_ort   = ort.OrtValue.ortvalue_from_numpy(
                          np.ascontiguousarray(x), 'cuda', 0)
            out_ort = ort.OrtValue.ortvalue_from_shape_and_type(
                          (n, 3), np.float32, 'cuda', 0)
            io.bind_ortvalue_input('input', x_ort)
            io.bind_ortvalue_output('output', out_ort)
            sess.run_with_iobinding(io)
            return out_ort.numpy()
        except Exception:
            pass   # フォールバック

    # CPU / CUDA 通常バッチ
    probs = np.zeros((n, 3), dtype=np.float32)
    bs = 4096 if mode == 'cpu' else 512
    for i in range(0, n, bs):
        probs[i:i+bs] = sess.run(None, {'input': x[i:i+bs]})[0]
    return probs


def calc_feature_importance(model, X_te: np.ndarray,
                            feat_indices: list | None,
                            n_samples: int = 100) -> list[tuple[str, float]]:
    """Permutation Importance: 全特徴量を1バッチで並列シャッフル推論 (高速版)。
    特徴量ごとのループをバッチ化してGPU転送回数を削減。
    Returns: [(feat_name, importance_score), ...] 降順ソート済み
    """
    import torch
    from features import FEATURE_COLS
    device = next(model.parameters()).device
    model.eval()
    n = min(n_samples, len(X_te))
    X_s = np.ascontiguousarray(X_te[:n])   # (n, seq, feat)
    n_feat = X_s.shape[2]

    rng = np.random.default_rng(42)
    # 全特徴量分のシャッフル済みデータを一括生成 (n_feat, n, seq, feat)
    X_all = np.stack([X_s.copy() for _ in range(n_feat)], axis=0)
    for i in range(n_feat):
        perm = rng.permutation(n)
        X_all[i, :, :, i] = X_all[i, perm, :, i]

    with torch.no_grad():
        # baseline
        xb = torch.from_numpy(X_s).to(device)
        base_probs = model(xb).float().cpu().numpy()
        base_ent = float(-np.mean(base_probs * np.log(base_probs + 1e-9)))

        # 全特徴量を結合してまとめて推論 (n_feat*n, seq, feat)
        X_flat = X_all.reshape(n_feat * n, X_s.shape[1], n_feat)
        bs = 512
        perm_probs_list = []
        for start in range(0, len(X_flat), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X_flat[start:start+bs])).to(device)
            perm_probs_list.append(model(xb).float().cpu().numpy())
        perm_probs_all = np.concatenate(perm_probs_list, axis=0).reshape(n_feat, n, 3)

    importances = []
    for i in range(n_feat):
        pp = perm_probs_all[i]
        perm_ent = float(-np.mean(pp * np.log(pp + 1e-9)))
        score = abs(perm_ent - base_ent)
        global_idx = feat_indices[i] if feat_indices else i
        fname = FEATURE_COLS[global_idx] if global_idx < len(FEATURE_COLS) else f'feat_{global_idx}'
        importances.append((fname, round(score, 6)))

    importances.sort(key=lambda x: -x[1])
    return importances


def backtest_torch(model, X_te, df_te, threshold, tp_mult, sl_mult,
                   seq_len=20, hold_bars=48, report_dir: Path = None, trial_no: int = 0):
    """PyTorchモデルで直接推論するバックテスト。
    ONNX エクスポート/ロードを省略して高速化。ゴミモデル1試行あたり約2秒節約。"""
    import torch
    device = next(model.parameters()).device
    model.to(device).eval()   # パラメータ + バッファを同じデバイスに揃える
    n = len(X_te)
    probs = np.zeros((n, 3), dtype=np.float32)
    bs = 1024
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(np.ascontiguousarray(X_te[i:i+bs])).to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.float16):
                out = model(xb)
            probs[i:i+bs] = out.float().cpu().numpy()
    close    = df_te['close'].values
    atr      = df_te['atr14'].values
    high     = df_te['high'].values
    low      = df_te['low'].values
    open_arr = df_te['open'].values
    dates    = df_te.index
    trades = _simulate_trades(probs, close, high, low, open_arr, atr,
                              n, seq_len, hold_bars, threshold, tp_mult, sl_mult, dates)
    return _backtest_evaluate(trades, dates, report_dir, trial_no)


def backtest(onnx_path, X_te, df_te, threshold, tp_mult, sl_mult,
             seq_len=20, hold_bars=48, report_dir: Path = None, trial_no: int = 0):
    import onnxruntime as ort
    mode = _bench_onnx_providers(str(onnx_path), X_te)
    pvd  = (['CUDAExecutionProvider', 'CPUExecutionProvider']
            if mode in ('cuda', 'cuda_iobind') else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(str(onnx_path), providers=pvd)
    close    = df_te['close'].values
    atr      = df_te['atr14'].values
    high     = df_te['high'].values
    low      = df_te['low'].values
    open_arr = df_te['open'].values
    dates    = df_te.index
    n        = len(X_te)

    # 全データ一括推論 (採用モードで最適化)
    probs = _run_inference(sess, X_te, mode)

    trades = _simulate_trades(
        probs, close, high, low, open_arr, atr,
        n, seq_len, hold_bars, threshold, tp_mult, sl_mult, dates
    )
    return _backtest_evaluate(trades, dates, report_dir, trial_no)


def _simulate_trades(probs, close, high, low, open_arr, atr,
                     n, seq_len, hold_bars, threshold, tp_mult, sl_mult, dates):
    """NumPy最適化版トレードシミュレーション。
    推論結果(probs)をもとにTP/SL/hold決済を処理し取引リストを返す。"""
    # シグナル検出をNumPyで一括計算
    cls_arr   = np.argmax(probs, axis=1).astype(np.int8)       # (n,)
    conf_arr  = probs[np.arange(n), cls_arr]                   # (n,)
    signal    = (conf_arr > threshold) & (cls_arr != 0)        # (n,) bool

    trades    = []
    in_pos    = False
    side      = 0
    entry     = 0.0
    tp_price  = 0.0
    sl_price  = 0.0
    entry_i   = 0

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close):
            break

        if in_pos:
            hi  = high[bi]
            lo  = low[bi]
            age = i - entry_i
            pnl = None
            if side == 1:
                if   lo  <= sl_price: pnl = sl_price - entry - SPREAD
                elif hi  >= tp_price: pnl = tp_price - entry - SPREAD
            else:
                if   hi  >= sl_price: pnl = entry - sl_price - SPREAD
                elif lo  <= tp_price: pnl = entry - tp_price - SPREAD
            if pnl is None and age > hold_bars:
                # age > hold_bars (= hold_bars+1本目の始値で決済) → MQL5 g_pos_bars >= hold_bars と同一
                pnl = (open_arr[bi] - entry) * side - SPREAD
            if pnl is not None:
                date_str = str(dates[bi].date()) if bi < len(dates) else str(bi)
                trades.append({'pnl': pnl, 'side': side,
                               'date': date_str, 'entry': entry})
                in_pos = False

        if not in_pos and signal[i]:
            next_bi = bi + 1
            if next_bi >= len(close):
                break
            a         = atr[bi]
            next_open = open_arr[next_bi]
            side      = int(cls_arr[i])   # 1=BUY, 2→-1=SELL
            if side == 1:
                entry    = next_open + SPREAD
                tp_price = entry + tp_mult * a
                sl_price = entry - sl_mult * a
            else:
                side     = -1
                entry    = next_open - SPREAD
                tp_price = entry - tp_mult * a
                sl_price = entry + sl_mult * a
            entry_i  = i
            in_pos   = True

    return trades


def _backtest_evaluate(trades, dates, report_dir, trial_no):

    MIN_TRADES = 200

    if len(trades) < MIN_TRADES:
        print(f"  [SKIP] 取引数 {len(trades)} < {MIN_TRADES} → PF=0 (除外)")
        return {'pf': 0.0, 'trades': len(trades), 'win_rate': 0.0,
                'gross_profit': 0.0, 'gross_loss': 0.0, 'net_pnl': 0.0,
                'sr': 0.0, 'max_dd': 0.0, 'max_dd_pct': 0.0,
                'final_equity': BT_CAPITAL, 'return_pct': 0.0}

    # ── 動的ロットサイズ計算 (MQL5 LotSize() 相当) ──────────────────────────
    # 各トレード時点の残高に基づいてロットを動的計算 → 複利効果
    cur_equity = BT_CAPITAL
    pnl_yen_list   = []
    equity_curve_list = [BT_CAPITAL]
    for t in trades:
        lot        = _calc_lot(cur_equity, BT_RISK_PCT)
        lot_units  = round(lot * 100_000)          # 単位数 (整数)
        pnl_yen    = t['pnl'] * lot_units          # 損益(円)
        margin     = t.get('entry', 150.0) * lot_units / BT_LEVERAGE
        cur_equity += pnl_yen
        pnl_yen_list.append(pnl_yen)
        equity_curve_list.append(cur_equity)
        t['pnl_yen']   = round(pnl_yen, 0)
        t['lot']       = lot
        t['lot_units'] = lot_units
        t['margin']    = round(margin, 0)
        t['equity']    = round(cur_equity, 0)

    pnl          = np.array(pnl_yen_list)
    equity_curve = np.array(equity_curve_list[1:])  # 各取引後の資産
    gp           = float(pnl[pnl>0].sum())
    gl           = float(abs(pnl[pnl<0].sum()))
    peak         = np.maximum.accumulate(equity_curve)
    dd           = equity_curve - peak
    max_dd       = float(dd.min())
    max_dd_pct   = float(max_dd / BT_CAPITAL * 100)

    # 日別 Sharpe Ratio: テスト期間の全営業日を対象にゼロ埋めして計算
    # 取引がない日を除外すると SR が過大評価されるため全日を使用
    daily_trade: dict = {}
    for t in trades:
        d = t.get('date', '0')
        daily_trade[d] = daily_trade.get(d, 0.0) + t['pnl_yen']
    # テスト期間の全ユニーク日付を取得してゼロ埋め
    all_dates = sorted({str(d.date()) for d in dates})
    dr = np.array([daily_trade.get(d, 0.0) for d in all_dates])
    sr = float((dr.mean() / dr.std()) * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0

    net_pnl_yen    = round(float(pnl.sum()), 0)
    final_equity   = round(float(equity_curve[-1]), 0)
    return_pct     = round((final_equity - BT_CAPITAL) / BT_CAPITAL * 100, 2)

    result = {
        'pf':           round(gp / max(gl, 1e-9), 4),
        'trades':       len(trades),
        'win_rate':     round(float((pnl>0).mean()), 4),
        'gross_profit': round(gp, 0),
        'gross_loss':   round(gl, 0),
        'net_pnl':      net_pnl_yen,
        'sr':           round(sr, 3),
        'max_dd':       round(max_dd, 0),
        'max_dd_pct':   round(max_dd_pct, 2),
        'final_equity': final_equity,
        'return_pct':   return_pct,
        'capital':      BT_CAPITAL,
        'risk_pct':     BT_RISK_PCT,
        'leverage':     BT_LEVERAGE,
    }
    print(f"  SR={sr:.3f}  MaxDD={max_dd:,.0f}円({max_dd_pct:.1f}%)  "
          f"NetPnL={net_pnl_yen:,.0f}円  最終資産={final_equity:,.0f}円  "
          f"リターン={return_pct:.1f}%")

    # 資産曲線 HTML レポート生成
    if report_dir is not None:
        _generate_report(trades, equity_curve, dd, result, Path(report_dir), trial_no)

    return result


def _write_trial_progress(out_dir: Path, data: dict) -> None:
    """並列モード: 試行固有の進捗を trial_progress.json に非同期で書く (GPU待機なし)"""
    try:
        _async_write(out_dir / 'trial_progress.json',
                     json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


# ─── メイン ──────────────────────────────────────────────────────────────────
def main():
    global OUT_DIR, ONNX_PATH, NORM_PATH
    args = parse_args()
    set_seed(args.seed)

    # --out_dir が指定されていれば出力先を切り替え (並列モード)
    parallel_mode = bool(getattr(args, 'out_dir', ''))
    if parallel_mode:
        OUT_DIR   = Path(args.out_dir)
        ONNX_PATH = OUT_DIR / 'fx_model.onnx'
        NORM_PATH = OUT_DIR / 'norm_params.json'
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ダッシュボード初期化
    st = args.start_time or time.time()

    if not parallel_mode:
        _all_res_path = OUT_DIR / 'all_results.json'
        _best_path    = OUT_DIR / 'best_result.json'
        prev_results  = json.loads(_all_res_path.read_text()) if _all_res_path.exists() else []
        best_pf_saved = (json.loads(_best_path.read_text()).get('pf', 0.0)
                         if _best_path.exists() else 0.0)
        best_pf_show  = max(args.best_pf, best_pf_saved)
        _dash.update({
            'phase': 'training',
            'trial': args.trial, 'total_trials': args.total_trials,
            'best_pf': best_pf_show, 'target_pf': 0,
            'current_params': {k: v for k, v in vars(args).items()
                               if k not in ('trial','total_trials','best_pf','start_time','out_dir')},
            'epoch': 0, 'total_epochs': args.epochs,
            'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
            'epoch_log': [], 'trial_results': prev_results,
            'start_time': st, 'message': f'試行{args.trial}: データ準備中...',
        })
        try: update_dashboard(_dash)
        except Exception: pass

    # 並列モードでも trial_progress.json に初期状態を書く
    _write_trial_progress(OUT_DIR, {
        'trial': args.trial, 'arch': getattr(args, 'arch', '?'),
        'hidden': getattr(args, 'hidden', 0),
        'epoch': 0, 'total_epochs': args.epochs,
        'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
        'phase': 'training',
    })

    X_tr, y_tr, X_te, y_te, mean, std, df_te, seq_len, feat_indices = prepare(args)
    n_feat = X_tr.shape[2]
    model = train(args, X_tr, y_tr, X_te, y_te, mean, std, n_feat=n_feat)

    wrapped = FXPredictorWithNorm(model, mean, std)

    print("\n=== バックテスト (テスト期間) ===")
    r = backtest_torch(wrapped, X_te, df_te, args.threshold, args.tp, args.sl,
                       seq_len=seq_len,
                       hold_bars=args.forward,
                       report_dir=OUT_DIR,
                       trial_no=args.trial)

    if r['trades'] >= 200:
        print(f"\n=== ONNX エクスポート (取引{r['trades']}件) ===")
        export_onnx(wrapped, seq_len, n_feat, str(ONNX_PATH), opset=12)
    print(f"  PF={r['pf']}  取引={r['trades']}  勝率={r['win_rate']:.1%}  "
          f"SR={r.get('sr',0):.3f}  MaxDD={r.get('max_dd',0):.4f}")

    # 特徴量重要度 (PF > 0 で取引があれば計算)
    feat_imp = []
    if r['trades'] >= 10:
        try:
            feat_imp = calc_feature_importance(wrapped, X_te, feat_indices, n_samples=300)
            top5 = ', '.join(f'{n}({s:.4f})' for n, s in feat_imp[:5])
            print(f"  特徴量重要度 TOP5: {top5}")
        except Exception as e:
            print(f"  [WARN] 特徴量重要度計算失敗: {e}")

    # 結果保存
    full = {**{k: v for k, v in vars(args).items() if k != 'out_dir'}, **r}
    full['feature_importance'] = feat_imp
    (OUT_DIR / 'last_result.json').write_text(
        json.dumps(full, indent=2, ensure_ascii=False), encoding='utf-8')

    # norm_params.json にEA用パラメータを追記
    # EA はこのファイルを読んで threshold/tp/sl/hold_bars を自動適用する
    try:
        np_data = json.loads(NORM_PATH.read_text(encoding='utf-8'))
        np_data['threshold'] = args.threshold
        np_data['tp_atr']    = args.tp
        np_data['sl_atr']    = args.sl
        np_data['hold_bars'] = args.forward
        NORM_PATH.write_text(json.dumps(np_data, indent=2, ensure_ascii=False),
                             encoding='utf-8')
        print(f"  norm_params更新: threshold={args.threshold} tp={args.tp} sl={args.sl} hold={args.forward}")
    except Exception as e:
        print(f"  [WARN] norm_params更新失敗: {e}")

    _write_trial_progress(OUT_DIR, {
        'trial': args.trial, 'arch': getattr(args, 'arch', '?'),
        'hidden': getattr(args, 'hidden', 0),
        'epoch': args.epochs, 'total_epochs': args.epochs,
        'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
        'phase': 'done', 'pf': r['pf'], 'sr': r.get('sr', 0), 'max_dd': r.get('max_dd', 0),
    })

    if not parallel_mode:
        _dash.update({
            'phase': 'trial_done',
            'message': (f"PF={r['pf']:.4f}  SR={r.get('sr',0):.3f}  "
                        f"MaxDD={r.get('max_dd',0):.4f}  取引={r['trades']}"),
        })
        try: update_dashboard(_dash)
        except Exception: pass

    return r['pf']


# ─── ProcessPoolExecutor ワーカー API ────────────────────────────────────────

def worker_init(cache_pkl_path: str) -> None:
    """ワーカープロセス初期化。Python起動 + torch import + CUDA初期化を1回だけ行う。
    以降は run_trial_worker() を何度呼んでもオーバーヘッドなし。
    """
    import pickle
    global _WORKER_STATE
    t0 = time.time()
    with open(cache_pkl_path, 'rb') as f:
        df_tr, df_te = pickle.load(f)
    _WORKER_STATE['df_tr'] = df_tr
    _WORKER_STATE['df_te'] = df_te
    # CUDA ウォームアップ (以後の試行でCUDA初期化コストが発生しないよう)
    if torch.cuda.is_available():
        _dummy = torch.zeros(1, device='cuda')
        del _dummy
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"  [WORKER pid={os.getpid()}] 初期化完了 {time.time()-t0:.1f}秒  "
          f"df_tr={len(df_tr):,}  df_te={len(df_te):,}", flush=True)


def run_trial_worker(trial_no: int, params: dict, trial_dir_str: str,
                     best_pf: float, start_time: float) -> dict:
    """ProcessPoolExecutor ワーカーで1試行を実行する。
    worker_init() が事前に呼ばれ _WORKER_STATE にデータが入っている前提。
    サブプロセス版と同一の last_result.json を出力する。
    """
    global OUT_DIR, ONNX_PATH, NORM_PATH

    trial_dir = Path(trial_dir_str)
    trial_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR   = trial_dir
    ONNX_PATH = trial_dir / 'fx_model.onnx'
    NORM_PATH = trial_dir / 'norm_params.json'

    # argparse 相当のオブジェクトを params から構築
    class _A:
        pass
    args = _A()
    _defaults = dict(
        epochs=800, seed=42, timeframe='H1', label_type='triple_barrier',
        train_months=0, feat_frac=1.0, n_features=0, feat_set=-2,
        scheduler='onecycle', wd=1e-2, seq_len=20, batch=256,
        lr=5e-4, dropout=0.5, hidden=64, layers=1, arch='gru_attn',
        tp=1.5, sl=1.0, forward=20, threshold=0.4,
    )
    for k, v in _defaults.items():
        setattr(args, k, v)
    for k, v in params.items():
        setattr(args, k, v)
    args.trial        = trial_no
    args.total_trials = 99999
    args.best_pf      = best_pf
    args.start_time   = start_time
    args.out_dir      = trial_dir_str

    set_seed(args.seed)
    _write_trial_progress(trial_dir, {
        'trial': trial_no, 'arch': getattr(args, 'arch', '?'),
        'hidden': getattr(args, 'hidden', 0),
        'epoch': 0, 'total_epochs': args.epochs,
        'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0, 'phase': 'training',
    })

    try:
        # prepare() は _WORKER_STATE を参照するのでディスクI/Oなし
        X_tr, y_tr, X_te, y_te, mean, std, df_te, seq_len, feat_indices = prepare(args)
        n_feat = X_tr.shape[2]

        model = train(args, X_tr, y_tr, X_te, y_te, mean, std, n_feat=n_feat)

        wrapped = FXPredictorWithNorm(model, mean, std)

        # ── PyTorch で直接バックテスト (ONNX エクスポート/ロード不要 → 2〜3秒節約) ──
        r = backtest_torch(wrapped, X_te, df_te, args.threshold, args.tp, args.sl,
                           seq_len=seq_len, hold_bars=args.forward,
                           report_dir=trial_dir, trial_no=trial_no)

        # ── ONNX エクスポート (採用基準 trades >= 200 のモデルのみ) ─────────────
        if r['trades'] >= 200:
            try:
                export_onnx(wrapped, seq_len, n_feat, str(ONNX_PATH), opset=12)
            except Exception as _e:
                print(f"  [WARN] ONNX エクスポート失敗: {_e}")

        # ── 特徴量重要度 (ONNX保存対象 trades>=200 のみ計算 / 軽量化)
        # trades < 200 のゴミモデルに時間を使わない → H100 速度改善
        feat_imp = []
        if r['trades'] >= 200:
            try:
                feat_imp = calc_feature_importance(wrapped, X_te, feat_indices, n_samples=100)
                top5 = ', '.join(f'{n}({s:.4f})' for n, s in feat_imp[:5])
                print(f"  特徴量重要度 TOP5: {top5}")
            except Exception as _e:
                print(f"  [WARN] 特徴量重要度計算失敗: {_e}")

        # norm_params に EA 用パラメータを追記
        try:
            nd = json.loads(NORM_PATH.read_text(encoding='utf-8'))
            nd.update({'threshold': args.threshold, 'tp_atr': args.tp,
                       'sl_atr': args.sl, 'hold_bars': args.forward})
            NORM_PATH.write_text(json.dumps(nd, indent=2, ensure_ascii=False),
                                 encoding='utf-8')
        except Exception:
            pass

        # last_result.json 保存 (メインプロセスが読み取る)
        full = {**{k: v for k, v in vars(args).items()
                   if k not in ('out_dir', 'total_trials', 'best_pf', 'start_time')}, **r}
        full['feature_importance'] = feat_imp
        (trial_dir / 'last_result.json').write_text(
            json.dumps(full, indent=2, ensure_ascii=False), encoding='utf-8')

        _write_trial_progress(trial_dir, {
            'trial': trial_no, 'arch': getattr(args, 'arch', '?'),
            'hidden': getattr(args, 'hidden', 0),
            'epoch': args.epochs, 'total_epochs': args.epochs,
            'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
            'phase': 'done', 'pf': r['pf'], 'sr': r.get('sr', 0),
            'max_dd': r.get('max_dd', 0),
        })
        print(f"  [WORKER #{trial_no}] PF={r['pf']:.4f}  取引={r['trades']}", flush=True)
        return r

    except Exception as e:
        import traceback
        print(f"  [WORKER ERROR] trial#{trial_no}: {e}\n{traceback.format_exc()}",
              flush=True)
        return {'pf': 0.0, 'trades': 0, 'error': str(e),
                'sr': 0.0, 'max_dd': 0.0, 'win_rate': 0.0,
                'net_pnl': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0}


if __name__ == '__main__':
    main()
