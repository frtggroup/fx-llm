"""
FX AI EA 自動トレーニング v6 - ランダムサーチ無限ループ
  ラベル: triple_barrier (何もしない / 買いで勝ち / 売りで勝ち)
  探索: NN構成・特徴量サブセット・全ハイパーパラメータをランダムサンプリング
  終了: PF >= 2.0 達成 または stop.flag / 手動停止

  H100_MODE=1 の環境変数で H100 向け大型モデル/バッチを使用
"""
import os, subprocess, sys, json, shutil, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dashboard import update_dashboard
from feature_sets import FEATURE_SETS

PY          = sys.executable
TRAIN_PY    = Path(__file__).parent / 'train.py'
OUT_DIR     = Path(__file__).parent
BEST_ONNX   = OUT_DIR / 'fx_model_best.onnx'
BEST_NORM   = OUT_DIR / 'norm_params_best.json'
BEST_JSON   = OUT_DIR / 'best_result.json'
TARGET_PF   = 2.0

# Docker 環境では /workspace/stop.flag を参照
_WORKSPACE  = Path('/workspace') if Path('/workspace').exists() else OUT_DIR.parent
STOP_FLAG   = _WORKSPACE / 'stop.flag'

# TOP10 モデル保存ディレクトリ
TOP10_CACHE = OUT_DIR / 'top10_cache'   # 全有効試行のモデルをキャッシュ
TOP10_DIR   = OUT_DIR / 'top10'         # rank_01〜rank_10 を再構築する先

H100_MODE   = os.environ.get('H100_MODE', '0') == '1'

ARCHS = [
    'mlp', 'gru_attn', 'bigru', 'lstm_attn',
    'cnn', 'tcn', 'cnn_gru', 'transformer', 'resnet', 'inception',
]

# ── GTX 1080 Ti 向け (デフォルト) ────────────────────────────────────────────
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
BATCH_CHOICES_LOCAL   = [512, 1024, 2048, 4096]
SEQ_LEN_CHOICES_LOCAL = [5, 8, 10, 15, 20]
EPOCH_COUNT_LOCAL     = 800
TRIAL_TIMEOUT_LOCAL   = 600   # 10分

# ── H100 SXM5 80GB 向け (大型・長時間) ──────────────────────────────────────
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
BATCH_CHOICES_H100   = [4096, 8192, 16384, 32768]
SEQ_LEN_CHOICES_H100 = [10, 15, 20, 30, 40, 50]
EPOCH_COUNT_H100     = 2000
TRIAL_TIMEOUT_H100   = 7200   # 2時間

# 実際に使う設定
HIDDEN_MAP     = HIDDEN_MAP_H100   if H100_MODE else HIDDEN_MAP_LOCAL
BATCH_CHOICES  = BATCH_CHOICES_H100  if H100_MODE else BATCH_CHOICES_LOCAL
SEQ_LEN_CHOICES= SEQ_LEN_CHOICES_H100 if H100_MODE else SEQ_LEN_CHOICES_LOCAL
EPOCH_COUNT    = EPOCH_COUNT_H100  if H100_MODE else EPOCH_COUNT_LOCAL
TRIAL_TIMEOUT  = TRIAL_TIMEOUT_H100 if H100_MODE else TRIAL_TIMEOUT_LOCAL


def sample_params(rng: random.Random) -> dict:
    """ランダムにハイパーパラメータをサンプリング"""
    arch    = rng.choice(ARCHS)
    hidden  = rng.choice(HIDDEN_MAP[arch])
    layers  = rng.choice([1, 2, 3] if arch not in ('mlp', 'gru_attn') else [1, 2])
    dropout = round(rng.uniform(0.3, 0.6), 1)
    lr      = rng.choice([1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3] if H100_MODE
                         else [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
    batch   = rng.choice(BATCH_CHOICES)
    tp      = round(rng.uniform(1.5, 3.5), 1)
    sl      = round(rng.uniform(0.5, 1.5), 1)
    fwd     = rng.choice([10, 15, 20, 25, 30])
    thr     = round(rng.uniform(0.33, 0.50), 2)
    seq_len = rng.choice(SEQ_LEN_CHOICES)
    sched   = rng.choice(['onecycle', 'cosine', 'cosine'])  # cosine を多く
    wd      = rng.choice([1e-3, 1e-2, 5e-2, 1e-1])
    # 直近データだけ使う確率 40%
    tm      = rng.choice([0, 0, 0, 12, 18, 12])
    # 特徴量セット: AIが設計した100セットからランダム選択
    # 25%の確率でランダムサブセット(2-70)も混ぜる
    if rng.random() < 0.25:
        n_feat   = rng.randint(2, 70)
        feat_set = -1
    else:
        feat_set = rng.randint(0, len(FEATURE_SETS) - 1)
        n_feat   = len(FEATURE_SETS[feat_set])
    seed    = rng.randint(0, 9999)

    return dict(
        arch=arch, hidden=hidden, layers=layers, dropout=dropout,
        lr=lr, batch=batch, tp=tp, sl=sl, forward=fwd,
        threshold=thr, seq_len=seq_len, scheduler=sched,
        wd=wd, train_months=tm,
        feat_set=feat_set,
        n_features=n_feat,
        seed=seed,
        timeframe='H1', epochs=EPOCH_COUNT,
        label_type='triple_barrier',
    )


def save_trial_model(trial_no: int) -> None:
    """現在の ONNX と norm_params を top10_cache に保存"""
    dest = TOP10_CACHE / f'trial_{trial_no:04d}'
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ['fx_model.onnx', 'norm_params.json']:
        src = OUT_DIR / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def rebuild_top10(results: list) -> None:
    """全試行結果から TOP10 を再計算して top10/rank_XX/ を再構築"""
    valid = [r for r in results
             if r.get('pf', 0) > 0 and r.get('trades', 0) >= 200]
    top10 = sorted(valid, key=lambda x: -x['pf'])[:10]
    TOP10_DIR.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(top10, 1):
        tno = r.get('trial', 0)
        src = TOP10_CACHE / f'trial_{tno:04d}'
        dst = TOP10_DIR / f'rank_{rank:02d}'
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            (dst / 'result.json').write_text(
                json.dumps(r, indent=2, ensure_ascii=False), encoding='utf-8')


def run_trial(params: dict, trial_no: int, best_pf: float, start_time: float) -> dict:
    cmd = [PY, str(TRAIN_PY),
           '--trial',        str(trial_no),
           '--total_trials', '9999',
           '--best_pf',      str(best_pf),
           '--start_time',   str(start_time),
           ]
    for k, v in params.items():
        cmd += [f'--{k}', str(v)]

    t0 = time.time()
    try:
        subprocess.run(cmd, timeout=TRIAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] 試行{trial_no} {TRIAL_TIMEOUT//60}分超過 → スキップ")
    print(f"  実行時間: {(time.time()-t0)/60:.1f}分")

    rp = OUT_DIR / 'last_result.json'
    if not rp.exists():
        return {'pf': 0.0, 'trades': 0, 'win_rate': 0.0}
    return json.loads(rp.read_text())


def main():
    best_pf   = 0.0
    best_p    = {}
    results   = []
    start     = time.time()
    trial_no  = 1
    rng       = random.Random()

    mode_str  = 'H100 80GB' if H100_MODE else 'GTX 1080 Ti'
    print('='*60)
    print(f'FX AI EA v6 - ランダムサーチ (PF>=2.0 まで) [{mode_str}]')
    print('  ラベル: triple_barrier')
    print(f'  タイムアウト: {TRIAL_TIMEOUT//60}分/試行  エポック: {EPOCH_COUNT}')
    print(f'  Python: {PY}')
    print(f'  stop.flag: {STOP_FLAG}')
    print('='*60)

    # 起動時に stop.flag をクリア
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    # 既存の結果を引き継ぐ
    ar = OUT_DIR / 'all_results.json'
    if ar.exists():
        try:
            results  = json.loads(ar.read_text(encoding='utf-8'))
            trial_no = len(results) + 1
            bests    = [r for r in results if r.get('pf', 0) > 0]
            if bests:
                best_r  = max(bests, key=lambda r: r['pf'])
                best_pf = best_r['pf']
                print(f"  前回最良PF={best_pf:.4f} を引き継ぎ (試行{trial_no-1}件完了)")
        except Exception:
            pass

    try:
        update_dashboard({
            'phase': 'training', 'trial': trial_no, 'total_trials': 9999,
            'best_pf': best_pf, 'target_pf': TARGET_PF,
            'current_params': {}, 'epoch': 0, 'total_epochs': EPOCH_COUNT,
            'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
            'epoch_log': [], 'trial_results': results[-50:],
            'start_time': start,
            'message': f'ランダムサーチ開始 [{mode_str}] (既存{trial_no-1}件引継ぎ)',
        })
    except Exception:
        pass

    while best_pf < TARGET_PF:
        # stop.flag チェック
        if STOP_FLAG.exists():
            print(f"\n[STOP] stop.flag 検出 → ループ終了")
            break

        p = sample_params(rng)

        print(f"\n{'='*60}")
        print(f"試行 #{trial_no}  (ベストPF={best_pf:.4f})")
        for k, v in p.items():
            print(f"  {k:14s} = {v}")
        print('='*60)

        feat_info = (f"set#{p['feat_set']}" if p['feat_set'] >= 0
                     else f"rand{p['n_features']}")
        try:
            update_dashboard({
                'phase': 'training', 'trial': trial_no, 'total_trials': 9999,
                'best_pf': best_pf, 'target_pf': TARGET_PF,
                'current_params': p, 'epoch': 0, 'total_epochs': p['epochs'],
                'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
                'epoch_log': [], 'trial_results': results[-50:],
                'start_time': start,
                'message': f'試行{trial_no}: {p["arch"]} feat={feat_info} sched={p["scheduler"]}',
            })
        except Exception:
            pass

        r = run_trial(p, trial_no, best_pf, start)
        pf = float(r.get('pf', 0.0))

        record = {
            'trial':        trial_no,
            'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
            'pf':           pf,
            'trades':       r.get('trades',       0),
            'win_rate':     r.get('win_rate',      0.0),
            'gross_profit': r.get('gross_profit',  0.0),
            'gross_loss':   r.get('gross_loss',    0.0),
            'net_pnl':      r.get('net_pnl',       0.0),
            **p,
        }
        results.append(record)

        # アトミック書き込み
        try:
            tmp = ar.with_suffix('.tmp')
            tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding='utf-8')
            tmp.replace(ar)
        except Exception as e:
            print(f"  [WARN] 結果保存失敗: {e}")

        # 有効試行のモデルを top10_cache に保存して TOP10 を再構築
        if pf > 0 and r.get('trades', 0) >= 200:
            try:
                save_trial_model(trial_no)
                rebuild_top10(results)
            except Exception as e:
                print(f"  [WARN] TOP10更新失敗: {e}")

        if pf > best_pf:
            best_pf = pf
            best_p  = p.copy()
            for src, dst in [(OUT_DIR/'fx_model.onnx',    BEST_ONNX),
                              (OUT_DIR/'norm_params.json', BEST_NORM)]:
                if src.exists():
                    shutil.copy2(src, dst)
            BEST_JSON.write_text(json.dumps({**best_p, 'pf': best_pf}, indent=2,
                                             ensure_ascii=False), encoding='utf-8')
            tag = ' [GOAL!!]' if best_pf >= TARGET_PF else ' [BEST]'
            print(f"  {tag} PF={best_pf:.4f}")

        try:
            update_dashboard({
                'phase': 'trial_done', 'trial': trial_no, 'total_trials': 9999,
                'best_pf': best_pf, 'target_pf': TARGET_PF,
                'current_params': p, 'epoch': p['epochs'], 'total_epochs': p['epochs'],
                'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
                'epoch_log': [], 'trial_results': results[-50:],
                'start_time': start,
                'message': f'試行{trial_no}完了: PF={pf:.4f}  ベスト: {best_pf:.4f}',
            })
        except Exception:
            pass

        trial_no += 1

        if best_pf >= TARGET_PF:
            print(f"\n[GOAL] PF={best_pf:.4f} 達成!")
            break

    # 最終モデルをコピー
    if BEST_ONNX.exists():
        shutil.copy2(BEST_ONNX, OUT_DIR/'fx_model.onnx')
    if BEST_NORM.exists():
        shutil.copy2(BEST_NORM, OUT_DIR/'norm_params.json')

    stopped = STOP_FLAG.exists()
    msg = (f'[GOAL] PF={best_pf:.4f} 達成!'
           if best_pf >= TARGET_PF
           else f'[STOP] PF={best_pf:.4f} (手動停止)' if stopped
           else f'最良PF={best_pf:.4f}')
    print(f"\n{msg}  パラメータ: {best_p}")

    try:
        update_dashboard({
            'phase': 'done', 'trial': trial_no - 1, 'total_trials': 9999,
            'best_pf': best_pf, 'target_pf': TARGET_PF,
            'current_params': best_p, 'epoch': 0, 'total_epochs': 0,
            'train_loss': 0.0, 'val_loss': 0.0, 'accuracy': 0.0,
            'epoch_log': [], 'trial_results': results[-50:],
            'start_time': start, 'message': msg,
        })
    except Exception:
        pass
    return best_pf


if __name__ == '__main__':
    main()
