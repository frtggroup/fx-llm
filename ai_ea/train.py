"""
CUDA GPU 学習スクリプト v3
- 70次元特徴量 × H1データ
- 10種アーキテクチャ対応
- 時間重み付け・過学習早期終了

使用方法:
    py -3.12 train.py [オプション]
"""
import os, sys, json, argparse, time
from pathlib import Path
from datetime import timedelta

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

_DEFAULT_DATA = str(Path(__file__).parent.parent / 'USDJPY_M1_202301012206_202602250650.csv')
DATA_PATH = Path(os.environ.get('DATA_PATH', _DEFAULT_DATA))
OUT_DIR   = Path(__file__).parent
ONNX_PATH = OUT_DIR / 'fx_model.onnx'
NORM_PATH = OUT_DIR / 'norm_params.json'
SPREAD    = 0.003  # 0.3pips × 0.01 = 0.003円 (USDJPY H1 spread)

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


# ─── データ準備 ────────────────────────────────────────────────────────────
_DF_CACHE: dict = {}   # {timeframe: (df_tr, df_te)} プロセス内キャッシュ


def prepare(args):
    print(f"\n=== データ準備 [{args.timeframe}] ===")
    t0 = time.time()

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

    # 特徴量セット選択 (優先順: feat_set > n_features > feat_frac > 全部)
    feat_set_id = getattr(args, 'feat_set', -2)
    n_feat_arg  = getattr(args, 'n_features', 0)
    feat_frac   = getattr(args, 'feat_frac', 1.0)

    if 0 <= feat_set_id <= 99:
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
def train(args, X_tr, y_tr, X_te, y_te, mean, std, n_feat=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== 学習 [{device}] ===")

    # GPU 能力を検出して最適化を選択
    is_h100    = False
    amp_dtype  = torch.float16
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        cc       = torch.cuda.get_device_capability(0)
        print(f"  GPU: {gpu_name}  CC={cc[0]}.{cc[1]}")
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
        if cc[0] >= 8:          # A100 (8.0) / H100 (9.0)
            # TF32: デフォルトで有効だが明示的に設定
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
        if cc[0] >= 9:          # H100 SXM5 (9.0)
            is_h100   = True
            amp_dtype = torch.bfloat16   # H100 は BF16 が FP16 より高速
            print(f"  H100 モード: BF16 + TF32 + torch.compile 有効")
        else:
            print(f"  FP16 AMP モード")

    tr_dl, va_dl = make_loaders(X_tr, y_tr, X_te, y_te, args, device)

    seq_len  = X_tr.shape[1]
    arch     = getattr(args, 'arch', 'gru_attn')
    n_in     = n_feat if n_feat is not None else N_FEATURES
    try:
        model = build_model(
            arch, n_in, seq_len,
            args.hidden, args.layers, args.dropout,
        ).to(device)
    except torch.cuda.OutOfMemoryError:
        # OOM: バッチ半減・モデル縮小でリトライ
        print(f"  [OOM] モデル生成 OOM → hidden/2, batch/2 でリトライ")
        torch.cuda.empty_cache()
        args.hidden = max(64, args.hidden // 2)
        args.batch  = max(256, args.batch  // 2)
        tr_dl, va_dl = make_loaders(X_tr, y_tr, X_te, y_te, args, device)
        model = build_model(
            arch, n_in, seq_len,
            args.hidden, args.layers, args.dropout,
        ).to(device)

    # H100: torch.compile (並列ランダムサーチでは reduce-overhead を使用)
    # max-autotune は初回コンパイルに10-30分かかるため並列サーチには不向き
    if is_h100 and not getattr(args, 'out_dir', ''):
        try:
            model = torch.compile(model, mode='max-autotune')
            print("  torch.compile(max-autotune) 有効 [シングルモード]")
        except Exception as e:
            print(f"  torch.compile スキップ: {e}")

    n_params = sum(p.numel() for p in model.parameters() if not hasattr(p, '_is_param'))
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        pass
    ratio    = len(X_tr) / max(n_params, 1)
    print(f"  arch={arch}  パラメータ数: {n_params:,}  サンプル/パラメータ比: {ratio:.1f}")

    use_amp = (device.type == 'cuda')
    # BF16 は勾配スケーリング不要 (FP16 のみ必要)
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler     = torch.amp.GradScaler('cuda', enabled=use_scaler)

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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 2,
            steps_per_epoch=len(tr_dl), epochs=args.epochs,
            pct_start=0.03, anneal_strategy='linear',
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
    patience       = max(50, int(args.epochs * 0.15))  # 15% (緩和)
    overfit_pat    = 20                                 # 20ep連続で判定
    min_epochs     = warmup_end + 10
    no_imp         = 0
    overfit_count  = 0
    stop_reason    = ''
    recent_gaps    = []
    prev_v_loss    = float('inf')    # val_loss の前エポック値 (下降判定用)
    print(f"  early_stop: ep{min_epochs}～, patience={patience}, wd={wd}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        try:
            t_loss = sum(
                _train_step(model, xb, yb,
                            optimizer, criterion, scheduler, scaler,
                            use_amp, amp_dtype, use_scaler)
                for xb, yb in tr_dl
            ) / len(tr_dl)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [OOM] ep{epoch} CUDA OOM → 学習打ち切り")
            stop_reason = f'CUDA OOM (ep{epoch})'
            break

        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    lo = model(xb)
                v_loss += criterion(lo, yb).item()
                correct += (lo.argmax(1) == yb).sum().item()
                total   += len(yb)
        v_loss /= len(va_dl)
        acc     = correct / max(total, 1)
        gap     = v_loss - t_loss   # 過学習ギャップ

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Ep{epoch:4d}/{args.epochs}  "
                  f"tr={t_loss:.4f}  va={v_loss:.4f}  "
                  f"gap={gap:+.4f}  acc={acc:.3f}  lr={lr_now:.2e}")

        # 進捗更新 (3エポックごと)
        if epoch % 3 == 0 or epoch <= 5:
            _dash['epoch']       = epoch
            _dash['total_epochs']= args.epochs
            _dash['train_loss']  = round(t_loss, 5)
            _dash['val_loss']    = round(v_loss, 5)
            _dash['accuracy']    = round(acc, 4)
            _dash.setdefault('epoch_log', []).append(
                {'epoch': epoch, 'train_loss': round(t_loss,6),
                 'val_loss': round(v_loss,6), 'acc': round(acc,4)})
            # 並列モード: trial_progress.json に書く
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        # ── 即時発散: val が爆発 かつ 下がっていない ────────────────────────
        val_falling = v_loss < prev_v_loss          # 今エポックで val 下降中か
        if epoch >= 10 and gap > 0.35 and v_loss > 1.30 and not val_falling:
            stop_reason = f'即時発散 (ep{epoch}: gap={gap:+.4f}, val={v_loss:.4f})'
            break
        prev_v_loss = v_loss

        # ── 過学習検出 (min_epochs 経過後) ──────────────────────────────────
        if epoch >= min_epochs:
            recent_gaps.append(gap)
            if len(recent_gaps) > overfit_pat:
                recent_gaps.pop(0)

            # 条件1: gap が大きく持続 かつ val が下がっていない
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

            # 条件3: val_loss 改善なし上限 (両方下がってる場合は免除)
            if no_imp >= patience and not val_falling:
                stop_reason = f'改善なし早期終了 (patience={patience})'
                break

    if stop_reason:
        print(f"  [STOP] {stop_reason}  ep={epoch}  best_val={best_loss:.4f}")
    else:
        print(f"  [DONE] 全エポック完了  best_val={best_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def _train_step(model, xb, yb, opt, crit, sched,
                scaler=None, use_amp=False,
                amp_dtype=torch.float16, use_scaler=False):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
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
        opt.step()
    sched.step()
    return loss.item()


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

def _bench_onnx_providers(onnx_path: str, full_data: np.ndarray) -> str:
    """CPU / CUDA / CUDA+IOBinding の3パターンで全データ推論速度を比較し
    最速モードを返す。結果は _BT_MODE にキャッシュして再利用。"""
    global _BT_MODE
    if _BT_MODE is not None:
        return _BT_MODE

    import onnxruntime as ort
    import time

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available:
        _BT_MODE = 'cpu'
        print('  [BT-bench] CUDA unavailable → CPU')
        return _BT_MODE

    x = np.ascontiguousarray(full_data)   # 全データで計測（実運用と同条件）
    results = {}

    # ── CPU ────────────────────────────────────────────────────────────
    try:
        s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        s.run(None, {'input': x[:512]})   # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(3):
            s.run(None, {'input': x})
        results['cpu'] = (time.perf_counter() - t0) / 3
    except Exception as e:
        print(f'  [BT-bench] CPU error: {e}')
        results['cpu'] = float('inf')

    cuda_pvd = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # ── CUDA (通常バッチ転送) ───────────────────────────────────────────
    try:
        s = ort.InferenceSession(onnx_path, providers=cuda_pvd)
        s.run(None, {'input': x[:512]})
        t0 = time.perf_counter()
        for _ in range(3):
            bs = 512
            out = np.zeros((len(x), 3), dtype=np.float32)
            for i in range(0, len(x), bs):
                out[i:i+bs] = s.run(None, {'input': x[i:i+bs]})[0]
        results['cuda'] = (time.perf_counter() - t0) / 3
    except Exception as e:
        print(f'  [BT-bench] CUDA error: {e}')
        results['cuda'] = float('inf')

    # ── CUDA + IOBinding (データをGPUメモリに常駐) ─────────────────────
    try:
        import ctypes
        s = ort.InferenceSession(onnx_path, providers=cuda_pvd)
        io = s.io_binding()
        # 全データを一括でGPU常駐テンソルとしてバインド
        x_ort = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        out_shape = (len(x), 3)
        out_ort = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float32, 'cuda', 0)
        io.bind_ortvalue_input('input', x_ort)
        io.bind_ortvalue_output('output', out_ort)
        s.run_with_iobinding(io)  # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(3):
            s.run_with_iobinding(io)
        _ = out_ort.numpy()       # GPU→CPU (計測に含める)
        results['cuda_iobind'] = (time.perf_counter() - t0) / 3
    except Exception as e:
        print(f'  [BT-bench] CUDA+IOBind error: {e}')
        results['cuda_iobind'] = float('inf')

    _BT_MODE = min(results, key=results.get)
    print(f'  [BT-bench] CPU={results.get("cpu",99)*1000:.1f}ms  '
          f'CUDA={results.get("cuda",99)*1000:.1f}ms  '
          f'CUDA+IOBind={results.get("cuda_iobind",99)*1000:.1f}ms  '
          f'→ {_BT_MODE} を採用')
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


def backtest(onnx_path, X_te, df_te, threshold, tp_mult, sl_mult,
             seq_len=20, report_dir: Path = None, trial_no: int = 0):
    import onnxruntime as ort
    mode = _bench_onnx_providers(str(onnx_path), X_te)
    pvd  = (['CUDAExecutionProvider', 'CPUExecutionProvider']
            if mode in ('cuda', 'cuda_iobind') else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(str(onnx_path), providers=pvd)
    close = df_te['close'].values
    atr   = df_te['atr14'].values
    high  = df_te['high'].values
    low   = df_te['low'].values
    dates = df_te.index
    n     = len(X_te)

    # 全データ一括推論 (採用モードで最適化)
    probs = _run_inference(sess, X_te, mode)

    HOLD_BARS = 48  # 最大保有 (H1×48=2日)
    trades = []
    pos    = None

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close): break

        c = close[bi]; a = atr[bi]
        date_str = str(dates[bi].date()) if bi < len(dates) else str(bi)

        # ポジション管理
        if pos:
            hi = high[bi]; lo = low[bi]
            age = i - pos['i0']
            pnl = None
            if pos['side'] == 1:
                if lo <= pos['sl']:  pnl = pos['sl'] - pos['entry'] - SPREAD
                elif hi >= pos['tp']: pnl = pos['tp'] - pos['entry'] - SPREAD
            else:
                if hi >= pos['sl']:  pnl = pos['entry'] - pos['sl'] - SPREAD
                elif lo <= pos['tp']: pnl = pos['entry'] - pos['tp'] - SPREAD
            if pnl is None and age >= HOLD_BARS:
                pnl = (c - pos['entry']) * pos['side'] - SPREAD
            if pnl is not None:
                trades.append({'pnl': pnl, 'side': pos['side'],
                               'date': date_str, 'entry': pos['entry']})
                pos = None

        # エントリー
        if pos is None:
            p = probs[i]
            cls = int(np.argmax(p))
            if p[cls] >= threshold and cls != 0:
                if cls == 1:
                    entry = c + SPREAD
                    pos   = {'side': 1, 'entry': entry,
                             'tp': entry + tp_mult*a,
                             'sl': entry - sl_mult*a, 'i0': i}
                else:
                    entry = c - SPREAD
                    pos   = {'side': -1, 'entry': entry,
                             'tp': entry - tp_mult*a,
                             'sl': entry + sl_mult*a, 'i0': i}

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

    # 日別 Sharpe Ratio (円換算PnLベース、年率換算 √252)
    daily: dict = {}
    for t in trades:
        d = t.get('date', '0')
        daily[d] = daily.get(d, 0.0) + t['pnl_yen']
    dr = np.array(list(daily.values()))
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
    """並列モード: 試行固有の進捗を trial_progress.json に書く"""
    try:
        path = out_dir / 'trial_progress.json'
        tmp  = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
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
        'phase': 'preparing',
    })

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

    print("\n=== ONNX エクスポート ===")
    wrapped = FXPredictorWithNorm(model, mean, std)
    export_onnx(wrapped, seq_len, n_feat, str(ONNX_PATH), opset=12)
    verify_onnx(str(ONNX_PATH), seq_len, n_feat)

    print("\n=== バックテスト (テスト期間) ===")
    r = backtest(ONNX_PATH, X_te, df_te, args.threshold, args.tp, args.sl,
                 seq_len=seq_len,
                 report_dir=OUT_DIR,
                 trial_no=args.trial)
    print(f"  PF={r['pf']}  取引={r['trades']}  勝率={r['win_rate']:.1%}  "
          f"SR={r.get('sr',0):.3f}  MaxDD={r.get('max_dd',0):.4f}")

    # 結果保存
    full = {**{k: v for k, v in vars(args).items() if k != 'out_dir'}, **r}
    (OUT_DIR / 'last_result.json').write_text(
        json.dumps(full, indent=2, ensure_ascii=False), encoding='utf-8')

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


if __name__ == '__main__':
    main()
