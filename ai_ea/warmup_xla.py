"""
XLA グラフ事前コンパイル (TPU専用)

TPU起動時に全 (arch, hidden, layers, seq_len) パターンを1フォワード+バックワード実行し
XLA_PERSISTENT_CACHE_PATH にコンパイル済みグラフをキャッシュする。
2回目以降の試行はキャッシュを再利用するため即座に学習開始できる。

単一チップ:
    python warmup_xla.py

複数チップ (v6e-4 など): xmp.spawn で TPU_NUM_DEVICES 枚を自動並列化
    TPU_NUM_DEVICES=4 python warmup_xla.py

進捗は /workspace/xla_warmup_rank_{rank}.json に保存され、
ダッシュボードの /api/status がランクファイルを集計して表示する。
"""
import argparse, io, json, logging, os, subprocess, sys, time, concurrent.futures, zipfile
from pathlib import Path

# urllib3 の "Connection pool is full" 警告を抑制 (S3並列アップロード時の無害な情報ログ)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# ── S3 設定 (環境変数から取得) ────────────────────────────────────────────────
def _s3_client():
    """boto3 S3クライアントを生成。設定がなければ None を返す。"""
    endpoint = os.environ.get('S3_ENDPOINT', '')
    key      = os.environ.get('S3_ACCESS_KEY', '')
    secret   = os.environ.get('S3_SECRET_KEY', '')
    if not endpoint or not key:
        return None, None, None
    try:
        import boto3, urllib3
        import botocore.config
        urllib3.disable_warnings()
        bucket = os.environ.get('S3_BUCKET', 'fxea')
        prefix = os.environ.get('S3_PREFIX', 'mix') + '/xla_cache'
        s3 = boto3.client('s3', endpoint_url=endpoint,
                          aws_access_key_id=key, aws_secret_access_key=secret,
                          verify=False,
                          config=botocore.config.Config(max_pool_connections=50))
        return s3, bucket, prefix
    except Exception:
        return None, None, None


# ── S3 分散クレーム: 複数VM間で同じパターンを二重コンパイルしない ─────────────
_CLAIM_TTL = 3600  # 秒: この時間内に完了しないクレームはスタールとみなす


def _claim_prefix(xla_prefix: str) -> str:
    """XLAキャッシュprefixからクレームprefixを生成"""
    base = xla_prefix.rsplit('/xla_cache', 1)[0]
    return base + '/warmup_claims'


def _pattern_key(arch, hidden, layers, seq_len) -> str:
    return f"{arch}_h{hidden}_L{layers}_seq{seq_len}"


def _try_claim(s3, bucket: str, cprefix: str, pkey: str) -> bool:
    """IfNoneMatch="*" でアトミッククレーム。成功=True、既存クレームあり=False。
    MinIO/S3 の条件付きPUTを使うため、クレームは排他的に1VMのみが獲得できる。"""
    try:
        s3.put_object(
            Bucket=bucket,
            Key=f"{cprefix}/{pkey}.json",
            Body=json.dumps({"ts": time.time()}).encode(),
            IfNoneMatch="*",
        )
        return True
    except Exception as e:
        try:
            code = e.response['Error']['Code']  # type: ignore[attr-defined]
        except Exception:
            code = ''
        if code in ('PreconditionFailed', 'ConditionalRequestConflict'):
            return False  # 他VMがクレーム済み
        # 予期しないエラー: クレームなしで続行 (安全側: コンパイルを試みる)
        return True


def _claim_is_stale(s3, bucket: str, cprefix: str, pkey: str) -> bool:
    """クレームが TTL を超えていれば True (VMクラッシュ時の再クレーム用)"""
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"{cprefix}/{pkey}.json")
        data = json.loads(obj['Body'].read())
        return time.time() - data.get('ts', 0) > _CLAIM_TTL
    except Exception:
        return True  # 取得失敗 = 存在しないか壊れている → スタールとみなす


def _release_claim(s3, bucket: str, cprefix: str, pkey: str):
    """クレームを解放 (コンパイル完了後またはエラー時)"""
    try:
        s3.delete_object(Bucket=bucket, Key=f"{cprefix}/{pkey}.json")
    except Exception:
        pass


def _upload_new_cache_files(cache_dir: Path, known_files: set, rank: int) -> set:
    """キャッシュディレクトリに増えたファイルを即座にS3へアップロードする。
    戻り値: 更新後の既知ファイルセット"""
    if not cache_dir.exists():
        return known_files
    current = {f for f in cache_dir.rglob('*') if f.is_file() and not f.name.endswith('.uploading.zip')}
    new_files = current - known_files
    if not new_files:
        return current
    s3, bucket, prefix = _s3_client()
    if s3 is None:
        return current
    def _upload_task(f):
        rel = f.relative_to(cache_dir)
        s3_key = f'{prefix}/{rel}.zip'
        # tmpファイルに書いてからアップロード (BytesIOだと大ファイルでOOM)
        tmp_path = f.parent / (f.name + f'.rank{rank}.uploading.zip')
        max_retries = 5
        try:
            from boto3.s3.transfer import TransferConfig as _TC
        except ImportError:
            _TC = None
        # multipart_threshold=256MB: XLA zip は通常 <100MB なので単一PUT で IncompleteBody を回避
        _tcfg = _TC(multipart_threshold=256*1024*1024, max_concurrency=1) if _TC else None
        for attempt in range(max_retries):
            # リトライ前に前回の残骸を必ず削除 (ENOENT 対策)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                # ディレクトリが存在しない場合に備えて作成 (rglob でサブディレクトリ内ファイルも対象)
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(str(tmp_path), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                    zf.write(str(f), f.name)
                if not tmp_path.exists():
                    raise RuntimeError(f"zip作成後にtmpファイルが消えた: {tmp_path}")
                if _tcfg:
                    s3.upload_file(str(tmp_path), bucket, s3_key, Config=_tcfg)
                else:
                    s3.upload_file(str(tmp_path), bucket, s3_key)
                return True
            except Exception as e:
                import time
                if attempt < max_retries - 1:
                    sleep_sec = 2 ** attempt
                    print(f"[WARMUP rank={rank}] S3アップロード失敗 {f.name}: {e} → {sleep_sec}秒後にリトライ ({attempt+1}/{max_retries})", flush=True)
                    time.sleep(sleep_sec)
                else:
                    print(f"[WARMUP rank={rank}] S3アップロード 最終失敗 {f.name}: {e}", flush=True)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return False

    import concurrent.futures
    uploaded = 0
    # max_workers=2: BytesIO廃止でメモリ圧迫は解消済みだが並列数は控えめに (4rank×2=8並列)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = {executor.submit(_upload_task, f): f for f in new_files}
        for fut in concurrent.futures.as_completed(futs):
            if fut.result():
                uploaded += 1
    if uploaded:
        print(f"[WARMUP rank={rank}] S3アップ: {uploaded}件 → {prefix}/", flush=True)
    return current

# ── 全パターン定義 (run_train.py の _ARCHS_TPU と同一) ─────────────────────────
# TPU では RNN 系 (bigru/gru_attn/lstm_attn) を除外
# 理由: XLA は RNN の隠れ状態依存を逐次コンパイルするため 1ep に数百秒かかる
ARCHS = [
    'mlp', 'cnn', 'tcn', 'cnn_gru', 'transformer', 'resnet', 'inception',
]

_HIDDEN_LARGE = {
    'mlp':         [512, 1024, 2048],
    'cnn':         [256, 512, 1024],
    'tcn':         [256, 512, 1024],
    'cnn_gru':     [256, 512, 1024],
    'transformer': [256, 512, 1024],
    'resnet':      [256, 512, 1024, 2048],
    'inception':   [256, 512, 1024],
}

_MAX_LAYERS = {
    'mlp': 2,  # run_train.py: layers in [1,2] for mlp
}

N_FEATURES = 70
SEQ_LENS   = [15, 20, 30, 40, 50, 60]   # run_train.py の _TIER_SEQ['tpu'] と同一
BATCH      = 1024
N_CLASSES  = 3
DROPOUT    = 0.3

WORKSPACE = Path('/workspace')


def _all_patterns():
    """全 (arch, hidden, layers, seq_len) 組み合わせを返す"""
    patterns = []
    for arch in ARCHS:
        max_l = _MAX_LAYERS.get(arch, 3)
        for hidden in _HIDDEN_LARGE[arch]:
            for layers in range(1, max_l + 1):
                for seq_len in SEQ_LENS:
                    patterns.append((arch, hidden, layers, seq_len))
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
               done_set: set, current=None, claim_mode: bool = False):
    """ランク別進捗を書き込む (ダッシュボードが集計して読む)"""
    if claim_mode:
        # クレームモード: 全パターンが対象 (複数VMで動的割当)
        my_total = len(all_patterns)
    else:
        # 静的分割モード: このランクの担当パターン数
        my_total = len([p for i, p in enumerate(all_patterns) if i % world_size == rank])
    data = {
        'rank':               rank,
        'world_size':         world_size,
        'warmup_total':       len(all_patterns),   # 全体の合計 (集計用)
        'my_total':           my_total,
        'warmup_done':        len(done_set),
        'warmup_current':     list(current) if current else None,
        'warmup_pct':         round(len(done_set) / max(my_total, 1) * 100, 1),
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

    # torch_xla が LIBTPU_INIT_ARGS に非対応フラグ (xla_tpu_heartbeat_watchdog_timeout_ms 等) を
    # 追加してlibtpuがクラッシュするのを防ぐ。空文字列をセットしておくと setdefault が上書きしない。
    os.environ.setdefault('LIBTPU_INIT_ARGS', '')

    import torch_xla.core.xla_model as xm  # type: ignore
    device = xm.xla_device()
    print(f"[WARMUP rank={rank}] デバイス: {device}", flush=True)

    all_pats = _all_patterns()
    done_set = _load_done(rank)

    # S3クライアント取得 (クレームモード判定)
    s3c, bucket, xla_prefix = _s3_client()
    cprefix = _claim_prefix(xla_prefix) if s3c else None
    use_claims = s3c is not None

    if use_claims:
        # ── クレームモード: 複数VM間で動的にパターンを割り当て ────────────────
        # シャッフルで各VM/チップが異なる順序から試みる → 自然に分散
        import random
        todo = [p for p in all_pats if p not in done_set]
        random.shuffle(todo)
        print(f"[WARMUP rank={rank}] クレームモード(S3): 全{len(all_pats)}パターンから動的割当  "
              f"完了済み: {len(done_set)}  未処理候補: {len(todo)}", flush=True)
    else:
        # ── ローカルモード: 静的インターリーブ分割 (単一VM) ──────────────────
        my_pats  = [p for i, p in enumerate(all_pats) if i % world_size == rank]
        todo     = [p for p in my_pats if p not in done_set]
        print(f"[WARMUP rank={rank}] ローカルモード: 担当{len(my_pats)}  "
              f"完了済み: {len(done_set)}  残り: {len(todo)}", flush=True)

    if not todo:
        print(f"[WARMUP rank={rank}] 全パターンコンパイル済み → スキップ", flush=True)
        _save_rank(rank, world_size, all_pats, done_set, claim_mode=use_claims)
        return

    # XLAキャッシュディレクトリ (コンパイル結果の書き込み先)
    cache_dir = Path(os.environ.get('XLA_PERSISTENT_CACHE_PATH',
                     os.environ.get('XLA_CACHE_DIR', '/workspace/xla_cache')))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # コンパイル前の既知ファイルを記録 (新規ファイルのみ S3 アップロードするため)
    known_files = {f for f in cache_dir.rglob('*') if f.is_file()}

    # label_smoothing=0.1 は train.py と同一 (HLO グラフを一致させるため必須)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    import torch.optim as _optim

    # ── S3アップロードは独立スレッドで非同期実行 (TPUコンパイルをブロックしない) ──
    _s3_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="s3up")
    _pending_s3: list[concurrent.futures.Future] = []

    def _async_upload(snap_known: set) -> set:
        return _upload_new_cache_files(cache_dir, snap_known, rank)

    def _flush_pending_s3():
        nonlocal known_files
        done = [f for f in _pending_s3 if f.done()]
        for f in done:
            try:
                known_files = f.result()
            except Exception as e:
                print(f"[WARMUP rank={rank}] S3非同期アップロードエラー: {e}", flush=True)
            _pending_s3.remove(f)

    # ── 全パターンを起動直後にCPU並列でビルド ──────────────────────────────────
    _N_BUILD_WORKERS = min(4, max(1, len(todo)))

    def _build_model_cpu(pat):
        a, h, l, s = pat
        try:
            return build_model(a, N_FEATURES, s, h, l, DROPOUT, N_CLASSES)
        except Exception:
            return None

    print(f"[WARMUP rank={rank}] 全{len(todo)}パターンをCPU {_N_BUILD_WORKERS}並列でビルド開始...", flush=True)
    _build_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_N_BUILD_WORKERS, thread_name_prefix="cpu_build"
    )
    _model_cache: dict = {
        pat: _build_executor.submit(_build_model_cpu, pat) for pat in todo
    }

    # ── 1パス目: 未クレームパターンをコンパイル ──────────────────────────────
    skipped_claims: list = []  # 他VMがクレーム中でスキップしたパターン

    def _compile_pattern(arch, hidden, layers, seq_len):
        """1パターンをTPUコンパイル。成功時 True、失敗時 False を返す。
        train.py の _train_step と HLO グラフを完全一致させる:
          - 入力 dtype: float32 (train.py GPULoader と同一)
          - criterion: CrossEntropyLoss(label_smoothing=0.1) (train.py と同一)
          - optimizer: syncfree.AdamW (train.py TPU ブランチと同一)
          - grad clipping: device-side (train.py と同一, .item() なし)
          - zero_grad(set_to_none=True) (train.py と同一)
        """
        tag = f"{arch}/h{hidden}/L{layers}/seq{seq_len}"
        t0 = time.time()
        try:
            pat_key = (arch, hidden, layers, seq_len)
            fut = _model_cache.pop(pat_key, None)
            model = fut.result() if fut is not None else _build_model_cpu(pat_key)
            if model is None:
                raise RuntimeError("モデルビルド失敗")

            model = model.to(device).train()
            # dtype=float32: train.py の GPULoader と同一
            # (autocast が float32→bfloat16 変換を行う → bfloat16入力と HLO が異なる)
            x_dummy = torch.randn(BATCH, seq_len, N_FEATURES, device=device, dtype=torch.float32)
            y_dummy = torch.randint(0, N_CLASSES, (BATCH,), device=device)

            # syncfree.AdamW: train.py TPU ブランチと同一 (optimizer step の HLO が一致)
            try:
                from torch_xla.amp import syncfree as _sf  # type: ignore
                opt = _sf.AdamW(model.parameters(), lr=1e-3)
            except ImportError:
                opt = _optim.AdamW(model.parameters(), lr=1e-3)

            # 大型モデル(h>=1024)は1ステップのみ: 2ステップだとXLAウォッチドッグ(121秒)を超えてkillされる
            n_steps = 1 if hidden >= 1024 else 2
            for _step in range(n_steps):
                opt.zero_grad(set_to_none=True)  # train.py と同一
                with torch.amp.autocast('xla', enabled=True, dtype=torch.bfloat16):
                    logits = model(x_dummy)
                    loss   = criterion(logits, y_dummy)
                loss.backward()
                # デバイスサイド gradient clipping: train.py _train_step と HLO を完全一致
                # (標準 clip_grad_norm_ は内部で .item() → XLA sync → HLO が異なる)
                _grads = [p.grad.detach() for p in model.parameters()
                          if p.grad is not None]
                if _grads:
                    _g_norms = torch.stack([g.norm(2) for g in _grads])
                    _total   = _g_norms.norm(2).clamp(min=1e-6)
                    _coef    = (1.0 / _total).clamp(max=1.0)
                    for _g in _grads:
                        _g.mul_(_coef)
                xm.optimizer_step(opt)  # mark_step() 内包 → ここで XLA グラフを submit

            # eval グラフ (validation ループ用)
            # criterion と argmax も含めて train.py の eval ループに近い HLO を生成
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('xla', enabled=True, dtype=torch.bfloat16):
                    _lo = model(x_dummy)
                _  = criterion(_lo, y_dummy)      # val loss 計算 (train.py と同一)
                __ = (_lo.argmax(1) == y_dummy).sum()  # 正解数計算 (train.py と同一)
                xm.mark_step()

            print(f"[WARMUP rank={rank}] ✓ {tag}  {time.time()-t0:.0f}秒", flush=True)
            return True
        except Exception as e:
            print(f"[WARMUP rank={rank}] ✗ {tag}  エラー: {e}", flush=True)
            return False

    for arch, hidden, layers, seq_len in todo:
        if (arch, hidden, layers, seq_len) in done_set:
            continue

        pkey = _pattern_key(arch, hidden, layers, seq_len)
        tag  = f"{arch}/h{hidden}/L{layers}/seq{seq_len}"

        # S3クレームを試みる (クレームモード時のみ)
        if use_claims:
            if not _try_claim(s3c, bucket, cprefix, pkey):
                skipped_claims.append((arch, hidden, layers, seq_len))
                continue  # 他VMがクレーム中 → スキップ

        print(f"[WARMUP rank={rank}] ({len(done_set)+1}/{len(all_pats)}) {tag} コンパイル中...", flush=True)
        _save_rank(rank, world_size, all_pats, done_set,
                   current=(arch, hidden, layers, seq_len), claim_mode=use_claims)

        if dry_run:
            done_set.add((arch, hidden, layers, seq_len))
            if use_claims:
                _release_claim(s3c, bucket, cprefix, pkey)
            continue

        try:
            _compile_pattern(arch, hidden, layers, seq_len)
        finally:
            # エラーでもクレームは必ず解放 (他VMが再クレームできるように)
            if use_claims:
                _release_claim(s3c, bucket, cprefix, pkey)

        done_set.add((arch, hidden, layers, seq_len))
        _save_rank(rank, world_size, all_pats, done_set, claim_mode=use_claims)
        _flush_pending_s3()
        _pending_s3.append(_s3_executor.submit(_async_upload, known_files.copy()))

    # ── 2パス目: スキップしたパターンのスタールクレームを確認して再試行 ─────────
    if skipped_claims and use_claims:
        print(f"[WARMUP rank={rank}] 2パス目: {len(skipped_claims)}パターンのスタールクレーム確認...",
              flush=True)
        time.sleep(30)  # VMクラッシュ等でスタールになるまで少し待つ

        for arch, hidden, layers, seq_len in skipped_claims:
            if (arch, hidden, layers, seq_len) in done_set:
                continue

            pkey = _pattern_key(arch, hidden, layers, seq_len)
            tag  = f"{arch}/h{hidden}/L{layers}/seq{seq_len}"

            if not _claim_is_stale(s3c, bucket, cprefix, pkey):
                continue  # まだ活動中 → 本当に他VMが担当中

            # スタールクレームを削除して再クレーム
            _release_claim(s3c, bucket, cprefix, pkey)
            if not _try_claim(s3c, bucket, cprefix, pkey):
                continue  # 別のVMが先に再クレーム

            print(f"[WARMUP rank={rank}] (再クレーム) {tag} コンパイル中...", flush=True)
            _save_rank(rank, world_size, all_pats, done_set,
                       current=(arch, hidden, layers, seq_len), claim_mode=True)

            if not dry_run:
                try:
                    _compile_pattern(arch, hidden, layers, seq_len)
                finally:
                    _release_claim(s3c, bucket, cprefix, pkey)

            done_set.add((arch, hidden, layers, seq_len))
            _save_rank(rank, world_size, all_pats, done_set, claim_mode=True)
            _flush_pending_s3()
            _pending_s3.append(_s3_executor.submit(_async_upload, known_files.copy()))

    # 全パターン完了後: 残りのS3アップロードを待機、ビルドexecutorも終了
    _build_executor.shutdown(wait=False)
    concurrent.futures.wait(_pending_s3)
    for f in _pending_s3:
        try:
            known_files = f.result()
        except Exception as e:
            print(f"[WARMUP rank={rank}] S3最終アップロードエラー: {e}", flush=True)
    _s3_executor.shutdown(wait=False)

    _save_rank(rank, world_size, all_pats, done_set, claim_mode=use_claims)
    print(f"[WARMUP rank={rank}] 完了! {len(done_set)}/{len(all_pats)} パターンをキャッシュ", flush=True)


def _worker_env(rank: int, n_dev: int) -> dict:
    """ワーカーサブプロセス用の環境変数を生成"""
    env = os.environ.copy()
    env['PJRT_LOCAL_PROCESS_RANK'] = str(rank)
    env['LOCAL_RANK']              = str(rank)
    env['TPU_VISIBLE_DEVICES']     = str(rank)
    env['TPU_NUM_DEVICES']         = '1'
    # torch_xla が LIBTPU_INIT_ARGS に非対応フラグを追加してlibtpuがクラッシュするのを防ぐ
    # 空文字列をセットしておくと torch_xla の setdefault が上書きしない
    env['LIBTPU_INIT_ARGS'] = ''
    return env


def _run_rank_until_done(rank: int, n_dev: int, dry_run: bool, max_retries: int = 20):
    """
    1チップのwarmupワーカーを実行。クラッシュ(watchdog timeout等)したら
    進捗JSONから自動再開し、全パターン完了まで繰り返す。
    クレームモード時は全パターンが対象、ローカルモード時はrank担当分のみ。
    """
    all_pats = _all_patterns()
    s3c, _, _ = _s3_client()
    # クレームモード(S3あり)では全パターンを対象に; ローカルモードは静的分割
    my_pats = all_pats if s3c else [p for i, p in enumerate(all_pats) if i % n_dev == rank]
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
