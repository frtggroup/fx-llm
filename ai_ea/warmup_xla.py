"""
XLA グラフ事前コンパイル (TPU専用)

TPU起動時に全 (arch, hidden, layers) パターンを1フォワード+バックワード実行し
XLA_PERSISTENT_CACHE_PATH にコンパイル済みグラフをキャッシュする。
2回目以降の試行はキャッシュを再利用するため即座に学習開始できる。

単一チップ:
    python warmup_xla.py

複数チップ (v6e-4 など): xmp.spawn で TPU_NUM_DEVICES 枚を自動並列化
    TPU_NUM_DEVICES=4 python warmup_xla.py

進捗は /workspace/xla_warmup_rank_{rank}.json に保存され、
ダッシュボードの /api/status がランクファイルを集計して表示する。
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

_MAX_LAYERS = {
    'mlp': 2, 'gru_attn': 2,
}

N_FEATURES = 70
SEQ_LEN    = 60
BATCH      = 1024
N_CLASSES  = 3
DROPOUT    = 0.3

WORKSPACE = Path('/workspace')


def _all_patterns():
    """全 (arch, hidden, layers) 組み合わせを返す"""
    patterns = []
    for arch in ARCHS:
        max_l = _MAX_LAYERS.get(arch, 3)
        for hidden in _HIDDEN_LARGE[arch]:
            for layers in range(1, max_l + 1):
                patterns.append((arch, hidden, layers))
    return patterns


def _rank_progress_path(rank: int) -> Path:
    return WORKSPACE / f'xla_warmup_rank_{rank}.json'


def _load_done(rank: int) -> set:
    """このランクの完了済みパターンを返す"""
    p = _rank_progress_path(rank)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            return set(tuple(x) for x in d.get('completed_patterns', []))
        except Exception:
            pass
    return set()


def _save_rank(rank: int, world_size: int, all_patterns: list,
               done_set: set, current=None):
    """ランク別進捗を書き込む (ダッシュボードが集計して読む)"""
    my_patterns = [p for i, p in enumerate(all_patterns) if i % world_size == rank]
    data = {
        'rank':               rank,
        'world_size':         world_size,
        'warmup_total':       len(all_patterns),   # 全体の合計 (集計用)
        'my_total':           len(my_patterns),
        'warmup_done':        len(done_set),
        'warmup_current':     list(current) if current else None,
        'warmup_pct':         round(len(done_set) / max(len(my_patterns), 1) * 100, 1),
        'completed_patterns': [list(p) for p in done_set],
        'updated_at':         time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    path = _rank_progress_path(rank)
    tmp  = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def _warmup_spawn_worker(rank, world_size, dry_run):
    """xmp.spawn(start_method='spawn') から呼ばれるトップレベル関数"""
    warmup(rank=rank, world_size=world_size, dry_run=dry_run)


def warmup(rank: int = 0, world_size: int = 1, dry_run: bool = False):
    sys.path.insert(0, str(Path(__file__).parent))
    from model import build_model  # noqa

    import torch
    import torch._dynamo as _dynamo
    _dynamo.config.disable = True   # Inductor デッドロック防止

    import torch_xla.core.xla_model as xm  # type: ignore
    device = xm.xla_device()
    print(f"[WARMUP rank={rank}] デバイス: {device}", flush=True)

    all_pats = _all_patterns()
    # 担当パターン: インターリーブ分割 (負荷均等)
    my_pats  = [p for i, p in enumerate(all_pats) if i % world_size == rank]
    done_set = _load_done(rank)
    todo     = [p for p in my_pats if p not in done_set]

    print(f"[WARMUP rank={rank}] 担当: {len(my_pats)}  完了済み: {len(done_set)}  残り: {len(todo)}", flush=True)

    if not todo:
        print(f"[WARMUP rank={rank}] 全パターンコンパイル済み → スキップ", flush=True)
        _save_rank(rank, world_size, all_pats, done_set)
        return

    criterion = torch.nn.CrossEntropyLoss()

    for arch, hidden, layers in todo:
        tag = f"{arch}/h{hidden}/L{layers}"
        print(f"[WARMUP rank={rank}] ({len(done_set)+1}/{len(my_pats)}) {tag} コンパイル中...", flush=True)
        _save_rank(rank, world_size, all_pats, done_set, current=(arch, hidden, layers))

        if dry_run:
            done_set.add((arch, hidden, layers))
            continue

        t0 = time.time()
        try:
            model = build_model(arch, N_FEATURES, SEQ_LEN, hidden, layers, DROPOUT, N_CLASSES)
            model = model.to(device).train()

            x_dummy = torch.randn(BATCH, SEQ_LEN, N_FEATURES, device=device, dtype=torch.bfloat16)
            y_dummy = torch.randint(0, N_CLASSES, (BATCH,), device=device)

            with torch.amp.autocast('xla', enabled=True, dtype=torch.bfloat16):
                logits = model(x_dummy)
                loss   = criterion(logits, y_dummy)
            loss.backward()
            xm.mark_step()

            elapsed = time.time() - t0
            print(f"[WARMUP rank={rank}] ✓ {tag}  {elapsed:.0f}秒", flush=True)

        except Exception as e:
            print(f"[WARMUP rank={rank}] ✗ {tag}  エラー: {e}", flush=True)

        done_set.add((arch, hidden, layers))
        _save_rank(rank, world_size, all_pats, done_set)

    _save_rank(rank, world_size, all_pats, done_set)
    print(f"[WARMUP rank={rank}] 完了! {len(done_set)}/{len(my_pats)} パターンをキャッシュ", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='実際のコンパイルをスキップ')
    args = parser.parse_args()

    n_dev = int(os.environ.get('TPU_NUM_DEVICES', '1'))

    if n_dev > 1:
        # xmp.spawn で全チップを並列コンパイル (PJRT デバイス割り当ては xmp が制御)
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
            print(f"[WARMUP] xmp.spawn: {n_dev} チップ並列コンパイル開始", flush=True)
            # nprocs=None: PJRT が利用可能な全チップを自動割り当て
            # start_method='spawn': PJRT 推奨。トップレベル関数が必要
            xmp.spawn(_warmup_spawn_worker,
                      args=(n_dev, args.dry_run),
                      nprocs=None,
                      start_method='spawn')
            print(f"[WARMUP] 全 {n_dev} チップ コンパイル完了", flush=True)
        except Exception as e:
            print(f"[WARMUP] xmp.spawn 失敗: {e} → シングルチップにフォールバック", flush=True)
            warmup(rank=0, world_size=1, dry_run=args.dry_run)
    else:
        warmup(rank=0, world_size=1, dry_run=args.dry_run)
