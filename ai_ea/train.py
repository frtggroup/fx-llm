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
SPREAD    = 0.003  # 0.3pips × 0.01 = 0.003円

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
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── データ準備 ────────────────────────────────────────────────────────────
def prepare(args):
    print(f"\n=== データ準備 [{args.timeframe}] ===")
    t0 = time.time()

    df = load_data(str(DATA_PATH), timeframe=args.timeframe)
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 直近1年 = テスト
    test_start = df.index[-1] - timedelta(days=365)
    df_tr = df[df.index < test_start].copy()
    df_te = df[df.index >= test_start].copy()

    # 訓練期間を最近N月に絞る (分布シフト対策)
    tm = getattr(args, 'train_months', 0)
    if tm > 0:
        train_start = test_start - timedelta(days=tm * 30)
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
    model    = build_model(
        arch, n_in, seq_len,
        args.hidden, args.layers, args.dropout,
    ).to(device)

    # H100: torch.compile で kernel fusion を最大活用
    if is_h100:
        try:
            model = torch.compile(model, mode='max-autotune')
            print("  torch.compile(max-autotune) 有効")
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
        t_loss = sum(
            _train_step(model, xb, yb,
                        optimizer, criterion, scheduler, scaler,
                        use_amp, amp_dtype, use_scaler)
            for xb, yb in tr_dl
        ) / len(tr_dl)

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

        # ダッシュボード更新 (3エポックごと)
        if epoch % 3 == 0 or epoch <= 5:
            _dash['epoch']       = epoch
            _dash['total_epochs']= args.epochs
            _dash['train_loss']  = round(t_loss, 5)
            _dash['val_loss']    = round(v_loss, 5)
            _dash['accuracy']    = round(acc, 4)
            # trial_results は _dash に既に入っているので上書きしない
            _dash.setdefault('epoch_log', []).append(
                {'epoch': epoch, 'train_loss': round(t_loss,6),
                 'val_loss': round(v_loss,6), 'acc': round(acc,4)})
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
def backtest(onnx_path, X_te, df_te, threshold, tp_mult, sl_mult, seq_len=20):
    import onnxruntime as ort
    sess  = ort.InferenceSession(str(onnx_path),
                                  providers=['CPUExecutionProvider'])
    close = df_te['close'].values
    atr   = df_te['atr14'].values
    high  = df_te['high'].values
    low   = df_te['low'].values
    n     = len(X_te)

    # バッチ推論
    probs = np.zeros((n, 3), dtype=np.float32)
    bs    = 512
    for i in range(0, n, bs):
        probs[i:i+bs] = sess.run(None, {'input': X_te[i:i+bs]})[0]

    HOLD_BARS = 48  # 最大保有 (H1×48=2日)
    trades = []
    pos    = None

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close): break

        c = close[bi]; a = atr[bi]

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
                trades.append({'pnl': pnl, 'side': pos['side']})
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
                'gross_profit': 0.0, 'gross_loss': 0.0, 'net_pnl': 0.0}

    pnl  = np.array([t['pnl'] for t in trades])
    gp   = float(pnl[pnl>0].sum())
    gl   = float(abs(pnl[pnl<0].sum()))
    return {
        'pf':           round(gp / max(gl, 1e-9), 4),
        'trades':       len(trades),
        'win_rate':     round(float((pnl>0).mean()), 4),
        'gross_profit': round(gp, 4),
        'gross_loss':   round(gl, 4),
        'net_pnl':      round(float(pnl.sum()), 4),
    }


# ─── メイン ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ダッシュボード初期化 ── 既存の累積結果を引き継ぐ
    st = args.start_time or time.time()

    # 過去の試行結果を読み込む (train.py が直接起動された場合も表示が途切れないように)
    _all_res_path = OUT_DIR / 'all_results.json'
    _best_path    = OUT_DIR / 'best_result.json'
    prev_results  = json.loads(_all_res_path.read_text()) if _all_res_path.exists() else []
    best_pf_saved = (json.loads(_best_path.read_text()).get('pf', 0.0)
                     if _best_path.exists() else 0.0)
    best_pf_show  = max(args.best_pf, best_pf_saved)

    _dash.update({
        'phase': 'training',
        'trial': args.trial, 'total_trials': args.total_trials,
        'best_pf': best_pf_show, 'target_pf': 2.0,
        'current_params': {k: v for k, v in vars(args).items()
                           if k not in ('trial','total_trials','best_pf','start_time')},
        'epoch': 0, 'total_epochs': args.epochs,
        'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
        'epoch_log': [], 'trial_results': prev_results,
        'start_time': st, 'message': f'試行{args.trial}: データ準備中...',
    })
    try: update_dashboard(_dash)
    except Exception: pass

    X_tr, y_tr, X_te, y_te, mean, std, df_te, seq_len, feat_indices = prepare(args)
    n_feat = X_tr.shape[2]   # 実際の特徴量数 (サブセット後)
    model = train(args, X_tr, y_tr, X_te, y_te, mean, std, n_feat=n_feat)

    print("\n=== ONNX エクスポート ===")
    wrapped = FXPredictorWithNorm(model, mean, std)
    export_onnx(wrapped, seq_len, n_feat, str(ONNX_PATH), opset=12)
    verify_onnx(str(ONNX_PATH), seq_len, n_feat)

    print("\n=== バックテスト (テスト期間) ===")
    r = backtest(ONNX_PATH, X_te, df_te, args.threshold, args.tp, args.sl, seq_len)
    print(f"  PF={r['pf']}  取引={r['trades']}  勝率={r['win_rate']:.1%}  "
          f"純損益={r['net_pnl']:.4f}")

    # 結果保存
    full = {**vars(args), **r}
    (OUT_DIR / 'last_result.json').write_text(json.dumps(full, indent=2))

    # ダッシュボード更新
    _dash.update({
        'phase': 'trial_done',
        'message': f"PF={r['pf']:.4f} 取引={r['trades']} 勝率={r['win_rate']:.1%}",
    })
    try: update_dashboard(_dash)
    except Exception: pass

    return r['pf']


if __name__ == '__main__':
    pf = main()
    import sys; sys.exit(0 if pf >= 1.5 else 1)
