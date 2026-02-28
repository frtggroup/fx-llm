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
import argparse, json, os, subprocess, sys, time
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

            import torch.optim as _optim
            opt = _optim.Adam(model.parameters(), lr=1e-3)

            # 大型モデル(h>=1024)は1ステップのみ: 2ステップだとXLAウォッチドッグ(121秒)を超えてkillされる
            n_steps = 1 if hidden >= 1024 else 2
            for _step in range(n_steps):
                opt.zero_grad()
                with torch.amp.autocast('xla', enabled=True, dtype=torch.bfloat16):
                    logits = model(x_dummy)
                    loss   = criterion(logits, y_dummy)
                loss.backward()
                xm.optimizer_step(opt)
                xm.mark_step()

            elapsed = time.time() - t0
            print(f"[WARMUP rank={rank}] ✓ {tag}  {elapsed:.0f}秒", flush=True)

        except Exception as e:
            print(f"[WARMUP rank={rank}] ✗ {tag}  エラー: {e}", flush=True)

        done_set.add((arch, hidden, layers))
        _save_rank(rank, world_size, all_pats, done_set)

    _save_rank(rank, world_size, all_pats, done_set)
    print(f"[WARMUP rank={rank}] 完了! {len(done_set)}/{len(my_pats)} パターンをキャッシュ", flush=True)


def _worker_env(rank: int, n_dev: int) -> dict:
    """ワーカーサブプロセス用の環境変数を生成"""
    env = os.environ.copy()
    env['PJRT_LOCAL_PROCESS_RANK'] = str(rank)
    env['LOCAL_RANK']              = str(rank)
    env['TPU_VISIBLE_DEVICES']     = str(rank)
    env['TPU_NUM_DEVICES']         = '1'
    return env


def _run_rank_until_done(rank: int, n_dev: int, dry_run: bool, max_retries: int = 20):
    """
    1チップのwarmupワーカーを実行。クラッシュ(watchdog timeout等)したら
    進捗JSONから自動再開し、全パターン完了まで繰り返す。
    """
    all_pats = _all_patterns()
    my_pats  = [p for i, p in enumerate(all_pats) if i % n_dev == rank]
    cmd_base = [sys.executable, __file__,
                '--rank', str(rank),
                '--world-size', str(n_dev)]
    if dry_run:
        cmd_base.append('--dry-run')

    env = _worker_env(rank, n_dev)

    for attempt in range(1, max_retries + 1):
        # 残りパターン数を確認
        done = _load_done(rank)
        remaining = len(my_pats) - len(done)
        if remaining <= 0:
            print(f"[WARMUP] rank={rank} 全パターン完了", flush=True)
            return True

        print(f"[WARMUP] rank={rank} 起動 attempt={attempt}  残り={remaining}/{len(my_pats)}", flush=True)
        p = subprocess.Popen(cmd_base, env=env)
        p.wait()

        if p.returncode == 0:
            print(f"[WARMUP] rank={rank} 正常完了", flush=True)
            return True

        # クラッシュ — 進捗を再確認して続行するか判断
        done_after = _load_done(rank)
        print(f"[WARMUP] rank={rank} クラッシュ (exit={p.returncode}) "
              f"完了={len(done_after)}/{len(my_pats)}  再起動待機中...", flush=True)

        if len(done_after) >= len(my_pats):
            print(f"[WARMUP] rank={rank} 全パターン完了 (クラッシュ前に保存済み)", flush=True)
            return True

        # TPUデバイス解放を待つ
        time.sleep(5)

    print(f"[WARMUP] rank={rank} 最大リトライ({max_retries})超過", flush=True)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='実際のコンパイルをスキップ')
    parser.add_argument('--rank', type=int, default=None,
                        help='チップrank (サブプロセス用。省略時は親プロセスとして全チップを起動)')
    parser.add_argument('--world-size', type=int, default=None,
                        help='総チップ数 (サブプロセス用。パターン分割に使用)')
    args = parser.parse_args()

    n_dev = int(os.environ.get('TPU_NUM_DEVICES', '1'))

    # サブプロセスとして起動された場合 (--rank 指定あり)
    if args.rank is not None:
        ws = args.world_size if args.world_size is not None else n_dev
        warmup(rank=args.rank, world_size=ws, dry_run=args.dry_run)
        sys.exit(0)

    if n_dev > 1:
        # subprocess.Popen で各チップを独立プロセスとして並列起動
        # クラッシュ時は自動再起動して全パターン完了を保証
        print(f"[WARMUP] subprocess: {n_dev} チップ並列コンパイル開始 (自動再起動あり)", flush=True)

        import threading
        results = {}

        def _run_rank_thread(rank):
            results[rank] = _run_rank_until_done(rank, n_dev, args.dry_run)

        threads = [threading.Thread(target=_run_rank_thread, args=(r,), daemon=True)
                   for r in range(n_dev)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ok = all(results.get(r, False) for r in range(n_dev))
        print(f"[WARMUP] 全 {n_dev} チップ コンパイル完了 (success={ok})", flush=True)
    else:
        warmup(rank=0, world_size=1, dry_run=args.dry_run)
