"""
XLA グラフ事前コンパイル (TPU専用)

TPU起動時に全 (arch, hidden, layers) パターンを1フォワード+バックワード実行し
XLA_PERSISTENT_CACHE_PATH にコンパイル済みグラフをキャッシュする。
2回目以降の試行はキャッシュを再利用するため即座に学習開始できる。

Usage:
    python warmup_xla.py [--dry-run]

進捗は /workspace/xla_warmup_progress.json に保存され、
ダッシュボードの /api/status に反映される。
"""
import argparse, json, os, sys, time
from pathlib import Path

# ── 全パターン定義 (run_train.py の TPU tier と同一) ──────────────────────────
ARCHS = [
    'mlp', 'gru_attn', 'bigru', 'lstm_attn',
    'cnn', 'tcn', 'cnn_gru', 'transformer', 'resnet', 'inception',
]

_HIDDEN_LARGE = {
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

# layers の上限 (mlp/gru_attn は2まで)
_MAX_LAYERS = {
    'mlp': 2, 'gru_attn': 2,
}

N_FEATURES = 70
SEQ_LEN    = 60
BATCH      = 1024
N_CLASSES  = 3
DROPOUT    = 0.3

PROGRESS_PATH = Path('/workspace/xla_warmup_progress.json')


def _all_patterns():
    """全 (arch, hidden, layers) 組み合わせを返す"""
    patterns = []
    for arch in ARCHS:
        max_l = _MAX_LAYERS.get(arch, 3)
        for hidden in _HIDDEN_LARGE[arch]:
            for layers in range(1, max_l + 1):
                patterns.append((arch, hidden, layers))
    return patterns


def _load_progress(patterns):
    """進捗ファイルを読み込み、完了済みセットを返す"""
    if PROGRESS_PATH.exists():
        try:
            d = json.loads(PROGRESS_PATH.read_text(encoding='utf-8'))
            done = set(tuple(p) for p in d.get('completed_patterns', []))
            return done
        except Exception:
            pass
    return set()


def _save_progress(patterns, done_set, current=None):
    """進捗をJSONに書き込む (ダッシュボードが読む)"""
    data = {
        'phase':              'warmup',
        'warmup_total':       len(patterns),
        'warmup_done':        len(done_set),
        'warmup_current':     list(current) if current else None,
        'warmup_pct':         round(len(done_set) / max(len(patterns), 1) * 100, 1),
        'completed_patterns': [list(p) for p in done_set],
        'updated_at':         time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    tmp = PROGRESS_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(PROGRESS_PATH)


def warmup(dry_run=False):
    sys.path.insert(0, str(Path(__file__).parent))
    from model import build_model  # noqa

    # TPU 初期化
    import torch
    import torch._dynamo as _dynamo
    _dynamo.config.disable = True   # Inductor デッドロック防止

    import torch_xla.core.xla_model as xm  # type: ignore
    device = xm.xla_device()
    print(f"[WARMUP] デバイス: {device}", flush=True)

    patterns = _all_patterns()
    done_set = _load_progress(patterns)

    todo = [p for p in patterns if p not in done_set]
    print(f"[WARMUP] 全パターン: {len(patterns)}  完了済み: {len(done_set)}  残り: {len(todo)}", flush=True)

    if not todo:
        print("[WARMUP] 全パターンコンパイル済み → スキップ", flush=True)
        _save_progress(patterns, done_set, current=None)
        return

    criterion = torch.nn.CrossEntropyLoss()

    for i, (arch, hidden, layers) in enumerate(todo, 1):
        tag = f"{arch}/h{hidden}/L{layers}"
        total_remaining = len(todo)
        print(f"[WARMUP] ({len(done_set)+1}/{len(patterns)}) {tag} コンパイル中...", flush=True)
        _save_progress(patterns, done_set, current=(arch, hidden, layers))

        if dry_run:
            done_set.add((arch, hidden, layers))
            continue

        t0 = time.time()
        try:
            # モデル構築 (CPU → TPU)
            model = build_model(arch, N_FEATURES, SEQ_LEN, hidden, layers, DROPOUT, N_CLASSES)
            model = model.to(device).train()

            # ランダム入力 (shape は train.py と同じ)
            x_dummy = torch.randn(BATCH, SEQ_LEN, N_FEATURES, device=device, dtype=torch.bfloat16)
            y_dummy = torch.randint(0, N_CLASSES, (BATCH,), device=device)

            # forward + backward → XLA グラフコンパイル
            with torch.amp.autocast('xla', enabled=True, dtype=torch.bfloat16):
                logits = model(x_dummy)
                loss   = criterion(logits, y_dummy)
            loss.backward()

            # グラフを実行・キャッシュに保存
            xm.mark_step()

            elapsed = time.time() - t0
            print(f"[WARMUP] ✓ {tag}  {elapsed:.0f}秒", flush=True)

        except Exception as e:
            print(f"[WARMUP] ✗ {tag}  エラー: {e}", flush=True)

        done_set.add((arch, hidden, layers))
        _save_progress(patterns, done_set, current=None)

    _save_progress(patterns, done_set, current=None)
    print(f"[WARMUP] 完了! {len(done_set)}/{len(patterns)} パターンをキャッシュ", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='実際のコンパイルをスキップ (テスト用)')
    args = parser.parse_args()
    warmup(dry_run=args.dry_run)
