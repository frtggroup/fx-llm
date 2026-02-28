#!/usr/bin/env python3
"""
TOP100 モデルのバックフィル処理
- S3 から ONNX / norm_params が欠落しているモデルをダウンロード
- feature_importance (順列重要度) を計算して保存
- all_results.json も更新してダッシュボードに即時反映
"""
import json, os, sys, time
from pathlib import Path

# ─── S3 設定 ─────────────────────────────────────────────────────────────────
S3_ENDPOINT   = os.environ.get('S3_ENDPOINT',   '')
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY',  '')
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY',  '')
S3_BUCKET     = os.environ.get('S3_BUCKET',      'fxea')
S3_PREFIX     = os.environ.get('S3_PREFIX',      'mix')
S3_ENABLED    = bool(S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY)

_S3_CLIENT = None

def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None and S3_ENABLED:
        import boto3
        from botocore.config import Config
        _S3_CLIENT = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name='jp-north-1',
            config=Config(connect_timeout=10, read_timeout=30),
        )
    return _S3_CLIENT

def _s3_list_keys(prefix: str) -> list:
    """S3 の prefix 以下のキーをすべて返す"""
    cl = _s3()
    if cl is None:
        return []
    try:
        keys = []
        paginator = cl.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys
    except Exception as e:
        print(f'  [S3] list error: {e}')
        return []

def _s3_download(s3_key: str, local_path: Path) -> bool:
    cl = _s3()
    if cl is None:
        return False
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cl.download_file(S3_BUCKET, s3_key, str(local_path))
        return True
    except Exception as e:
        print(f'  [S3] download error {s3_key}: {e}')
        return False

def _try_download_onnx(rank_dir: Path, trial_no: int) -> bool:
    """S3 の複数プレフィクスから ONNX をダウンロード試行"""
    if not S3_ENABLED:
        return False
    rank_name = rank_dir.name
    # 候補となる S3 パス (rank-specific のみ。best_* は rank によらず同一ファイルなので除外)
    node_candidates = ['gtx', 'h100', 'cpu']
    s3_patterns = []
    for node in node_candidates:
        s3_patterns.append(f'{S3_PREFIX}/top100_{node}/{rank_name}/fx_model.onnx')
    s3_patterns.append(f'checkpoint/top100/{rank_name}/fx_model.onnx')
    s3_patterns.append(f'{S3_PREFIX}/top100/{rank_name}/fx_model.onnx')

    for s3_key in s3_patterns:
        onnx_dst = rank_dir / 'fx_model.onnx'
        if _s3_download(s3_key, onnx_dst):
            print(f'  [S3] ONNX ダウンロード成功: {s3_key}')
            # norm_params.json も同じ場所から取得 (常に上書き: ONNX と一致させる)
            norm_key = s3_key.replace('fx_model.onnx', 'norm_params.json')
            norm_dst = rank_dir / 'norm_params.json'
            norm_bak = rank_dir / 'norm_params.json.bak'
            if norm_dst.exists():
                norm_dst.rename(norm_bak)
            ok = _s3_download(norm_key, norm_dst)
            if not ok and norm_bak.exists():
                norm_bak.rename(norm_dst)  # 取得失敗ならバックアップを戻す
            elif norm_bak.exists():
                norm_bak.unlink(missing_ok=True)
            return True
    return False

sys.path.insert(0, str(Path(__file__).parent))

WORKSPACE  = Path('/workspace') if Path('/workspace').exists() else Path(__file__).parent.parent
AI_EA_DIR  = Path(__file__).parent
TOP_DIR    = AI_EA_DIR / 'top100'
TRIALS_DIR = AI_EA_DIR / 'trials'
ALL_RESULTS= AI_EA_DIR / 'all_results.json'
DATA_PATH  = Path(os.environ.get('DATA_PATH', '/workspace/data/USDJPY_H1.csv'))

# ─── データ読込 ──────────────────────────────────────────────────────────────
def _load_df():
    from features import load_data, add_indicators
    import numpy as np
    print(f'[BF] データ読込: {DATA_PATH}')
    df = load_data(str(DATA_PATH), timeframe='H1')
    df = add_indicators(df)
    df.replace([np.inf, -np.inf], float('nan'), inplace=True)
    df.dropna(inplace=True)
    return df

# ─── ONNX 推論で特徴量重要度を計算 ─────────────────────────────────────────
def calc_importance_onnx(onnx_path: Path, norm_path: Path,
                         df, feat_indices, seq_len=20, n_samples=300):
    """ONNX モデルを使って permutation importance を計算する"""
    import numpy as np
    import onnxruntime as ort
    from features import FEATURE_COLS, N_FEATURES
    from features import build_dataset

    # テストデータ準備
    from datetime import timedelta
    test_start = df.index[-1] - timedelta(days=365)
    df_te = df[df.index >= test_start].copy()
    if len(df_te) < seq_len + 10:
        df_te = df.copy()

    X_te, _, _ = build_dataset(df_te, seq_len,
                                tp_atr=1.5, sl_atr=1.0, forward_bars=20,
                                feat_indices=feat_indices)
    if len(X_te) == 0:
        return []

    # norm_params ロード
    if norm_path.exists():
        np_data = json.loads(norm_path.read_text(encoding='utf-8'))
        mean = np.array(np_data.get('mean', [0]*X_te.shape[2]), dtype=np.float32)
        std  = np.array(np_data.get('std',  [1]*X_te.shape[2]), dtype=np.float32)
        std  = np.where(std < 1e-8, 1.0, std)
        X_te = ((X_te - mean) / std).astype(np.float32)
    else:
        X_te = X_te.astype(np.float32)

    n = min(n_samples, len(X_te))
    X_s = X_te[:n]

    sess = ort.InferenceSession(str(onnx_path),
                                providers=['CPUExecutionProvider'])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape   # e.g. [batch, seq_len, n_feat]

    # ONNX モデルが期待する seq_len / n_feat で X_s を再構築
    if len(inp_shape) == 3:
        onnx_seq   = inp_shape[1] if isinstance(inp_shape[1], int) and inp_shape[1] > 0 else seq_len
        onnx_nfeat = inp_shape[2] if isinstance(inp_shape[2], int) and inp_shape[2] > 0 else X_s.shape[2]
        if onnx_seq != X_s.shape[1] or onnx_nfeat != X_s.shape[2]:
            # ONNX の期待値に合わせてデータを再生成
            if feat_indices and len(feat_indices) >= onnx_nfeat:
                fi2 = feat_indices[:onnx_nfeat]
            elif feat_indices:
                # feat_indices が足りない場合は全特徴量 (None) を試みる
                fi2 = None if onnx_nfeat == 70 else feat_indices
            else:
                fi2 = feat_indices
            X2, _, _ = build_dataset(df_te, onnx_seq,
                                     tp_atr=1.5, sl_atr=1.0, forward_bars=20,
                                     feat_indices=fi2)
            if len(X2) == 0:
                return []
            if norm_path.exists():
                try:
                    np_data = json.loads(norm_path.read_text(encoding='utf-8'))
                    mean2 = np.array(np_data.get('mean', [0]*X2.shape[2]), dtype=np.float32)
                    std2  = np.array(np_data.get('std',  [1]*X2.shape[2]), dtype=np.float32)
                    std2  = np.where(std2 < 1e-8, 1.0, std2)
                    if len(mean2) == onnx_nfeat:
                        X2 = ((X2 - mean2) / std2).astype(np.float32)
                except Exception:
                    pass
            n2   = min(n_samples, len(X2))
            X_s  = X2[:n2].astype(np.float32)
            n_feat = onnx_nfeat
            # feat_indices を onnx_nfeat に絞る
            if feat_indices and len(feat_indices) >= onnx_nfeat:
                feat_indices = feat_indices[:onnx_nfeat]

    # 最終次元チェック
    if X_s.shape[1:] != (onnx_seq, n_feat):
        print(f'  [BF] 次元ミスマッチ解消できず skipping (got {X_s.shape}, expected (N,{onnx_seq},{n_feat}))')
        return []

    # ベースライン
    try:
        base_out = sess.run(None, {inp_name: X_s})[0]   # (n, 3)
    except Exception as e:
        print(f'  [BF] ONNX 推論エラー (skipping): {e}')
        return []
    base_ent = float(-np.mean(base_out * np.log(base_out + 1e-9)))

    rng = np.random.default_rng(42)
    importances = []
    n_feat = X_s.shape[2]  # noqa: F811 (possibly overwritten above)
    for i in range(n_feat):
        X_p = X_s.copy()
        X_p[:, :, i] = X_p[rng.permutation(n), :, i]
        perm_out = sess.run(None, {inp_name: X_p})[0]
        perm_ent = float(-np.mean(perm_out * np.log(perm_out + 1e-9)))
        score = abs(perm_ent - base_ent)
        gidx  = feat_indices[i] if feat_indices else i
        fname = FEATURE_COLS[gidx] if gidx < len(FEATURE_COLS) else f'feat_{gidx}'
        importances.append((fname, round(score, 6)))

    importances.sort(key=lambda x: -x[1])
    return importances


# ─── all_results.json 更新 ───────────────────────────────────────────────────
def _update_all_results(trial_no: int, feature_importance: list):
    if not ALL_RESULTS.exists():
        return
    try:
        data = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
        updated = False
        for r in data:
            if r.get('trial') == trial_no:
                if not r.get('feature_importance'):
                    r['feature_importance'] = feature_importance
                    updated = True
        if updated:
            tmp = ALL_RESULTS.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding='utf-8')
            tmp.replace(ALL_RESULTS)
    except Exception as e:
        print(f'  [WARN] all_results更新失敗: {e}')


# ─── メイン処理 ──────────────────────────────────────────────────────────────
def main():
    if not TOP_DIR.exists():
        print('[BF] top100 ディレクトリが見つかりません')
        return

    rank_dirs = sorted(TOP_DIR.glob('rank_*'))
    total = len(rank_dirs)
    print(f'[BF] top100 モデル数: {total}')

    # 処理対象: feature_importance なし (ONNX なしは S3 から取得試行)
    to_process = []
    for rank_dir in rank_dirs:
        onnx   = rank_dir / 'fx_model.onnx'
        norm   = rank_dir / 'norm_params.json'
        result = rank_dir / 'result.json'
        if not result.exists():
            continue
        try:
            r = json.loads(result.read_text(encoding='utf-8'))
        except Exception:
            continue
        imp = r.get('feature_importance', [])
        need_process = False
        if not imp:
            need_process = True
        elif isinstance(imp[0][0], int):
            # 数値インデックスのみなら再計算
            need_process = True
        if not need_process:
            continue
        # ONNX なければ S3 から取得試行
        if not onnx.exists():
            trial_no = r.get('trial', 0)
            print(f'[BF] {rank_dir.name} trial#{trial_no}: ONNX なし → S3 ダウンロード試行')
            if not _try_download_onnx(rank_dir, trial_no):
                print(f'  → S3 にも見つからず スキップ')
                continue
        to_process.append((rank_dir, onnx, norm, result, r))

    print(f'[BF] 特徴量重要度バックフィル対象: {len(to_process)} モデル')
    if not to_process:
        print('[BF] 全モデル処理済み')
        return

    # データ読込 (一回だけ)
    try:
        df = _load_df()
    except Exception as e:
        print(f'[BF] データ読込失敗: {e}')
        return

    for idx, (rank_dir, onnx, norm, result_f, r) in enumerate(to_process, 1):
        trial_no   = r.get('trial', 0)
        feat_indices = r.get('feat_indices')   # list or None
        seq_len    = r.get('seq_len', 20)

        # norm_params.json から補完
        if norm.exists():
            try:
                nd = json.loads(norm.read_text(encoding='utf-8'))
                if feat_indices is None:
                    feat_indices = nd.get('feat_indices')
                if seq_len == 20:
                    seq_len = nd.get('seq_len', 20)
            except Exception:
                pass
        rank_name  = rank_dir.name

        print(f'[BF] [{idx}/{len(to_process)}] {rank_name} trial#{trial_no} '
              f'feat_indices={feat_indices is not None} seq_len={seq_len}')
        t0 = time.time()
        try:
            imp = calc_importance_onnx(onnx, norm, df, feat_indices, seq_len)
            if not imp:
                print(f'  → スキップ (データ不足)')
                continue

            top5 = ', '.join(f'{n}({s:.4f})' for n, s in imp[:5])
            print(f'  → TOP5: {top5}  ({time.time()-t0:.1f}s)')

            # result.json 更新
            r['feature_importance'] = imp
            result_f.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                encoding='utf-8')

            # all_results.json も更新
            _update_all_results(trial_no, imp)

        except Exception as e:
            print(f'  → エラー: {e}')
            continue

    print('[BF] バックフィル完了')


if __name__ == '__main__':
    main()
