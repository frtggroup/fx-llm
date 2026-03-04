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
S3_SECRET_KEY = 'Yakrty1484!#'  # hardcoded: env var has escaped # issue
S3_BUCKET     = os.environ.get('S3_BUCKET',      'mix3')
S3_PREFIX     = os.environ.get('S3_PREFIX',      '')
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
            verify=False,
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

_S3_TRIAL_MAP: dict | None = None  # trial_no -> {'onnx': s3_key, 'norm': s3_key}

def _build_s3_trial_map() -> dict:
    """S3 の top100_* 以下の result.json を全件読んで trial_no → S3パス マップを構築"""
    global _S3_TRIAL_MAP
    if _S3_TRIAL_MAP is not None:
        return _S3_TRIAL_MAP
    _S3_TRIAL_MAP = {}
    if not S3_ENABLED:
        return _S3_TRIAL_MAP
    cl = _s3()
    if cl is None:
        return _S3_TRIAL_MAP
    node_prefixes = ['top100_gtx1080ti', 'top100_h200', 'top100_h100', 'top100_gtx', 'top100_cpu']
    if S3_PREFIX:
        node_prefixes = [f'{S3_PREFIX}/{p}' for p in node_prefixes] + node_prefixes
    for prefix in node_prefixes:
        try:
            paginator = cl.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if not key.endswith('/result.json'):
                        continue
                    try:
                        resp = cl.get_object(Bucket=S3_BUCKET, Key=key)
                        r = json.loads(resp['Body'].read())
                        tno = r.get('trial')
                        if tno is None:
                            continue
                        base = key.rsplit('/', 1)[0]
                        onnx_key = base + '/fx_model.onnx'
                        norm_key = base + '/norm_params.json'
                        if tno not in _S3_TRIAL_MAP:
                            _S3_TRIAL_MAP[tno] = {'onnx': onnx_key, 'norm': norm_key, 'result': r}
                    except Exception:
                        pass
        except Exception as e:
            print(f'  [S3] map build error ({prefix}): {e}')
    print(f'  [S3] trial_map built: {len(_S3_TRIAL_MAP)} entries')
    return _S3_TRIAL_MAP


def _try_download_onnx(rank_dir: Path, trial_no: int) -> bool:
    """S3 から ONNX をダウンロード: まず trial_no マップで検索、次に rank_name で検索"""
    if not S3_ENABLED:
        return False

    def _dl_onnx_norm(s3_onnx_key: str) -> bool:
        onnx_dst = rank_dir / 'fx_model.onnx'
        if not _s3_download(s3_onnx_key, onnx_dst):
            return False
        print(f'  [S3] ONNX ダウンロード成功: {s3_onnx_key}')
        norm_key = s3_onnx_key.replace('fx_model.onnx', 'norm_params.json')
        norm_dst = rank_dir / 'norm_params.json'
        norm_bak = rank_dir / 'norm_params.json.bak'
        if norm_dst.exists():
            norm_dst.rename(norm_bak)
        ok = _s3_download(norm_key, norm_dst)
        if not ok and norm_bak.exists():
            norm_bak.rename(norm_dst)
        elif norm_bak.exists():
            norm_bak.unlink(missing_ok=True)
        return True

    # 1) trial_no マップで検索 (最も確実)
    tmap = _build_s3_trial_map()
    if trial_no in tmap:
        if _dl_onnx_norm(tmap[trial_no]['onnx']):
            return True

    # 2) rank_name パターンでフォールバック
    rank_name = rank_dir.name
    node_candidates = ['gtx1080ti', 'h200', 'h100', 'gtx', 'cpu']
    s3_patterns = []
    for node in node_candidates:
        s3_patterns.append(f'top100_{node}/{rank_name}/fx_model.onnx')
        if S3_PREFIX:
            s3_patterns.append(f'{S3_PREFIX}/top100_{node}/{rank_name}/fx_model.onnx')
    for s3_key in s3_patterns:
        if _dl_onnx_norm(s3_key):
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

    # norm_params ロード (shape ミスマッチ時はスキップ)
    if norm_path.exists():
        try:
            np_data = json.loads(norm_path.read_text(encoding='utf-8'))
            mean = np.array(np_data.get('mean', [0]*X_te.shape[2]), dtype=np.float32)
            std  = np.array(np_data.get('std',  [1]*X_te.shape[2]), dtype=np.float32)
            if len(mean) == X_te.shape[2]:
                std  = np.where(std < 1e-8, 1.0, std)
                X_te = ((X_te - mean) / std).astype(np.float32)
            else:
                print(f'  [BF] norm_params shape mismatch ({len(mean)} vs {X_te.shape[2]}), skip norm')
                X_te = X_te.astype(np.float32)
        except Exception:
            X_te = X_te.astype(np.float32)
    else:
        X_te = X_te.astype(np.float32)

    n = min(n_samples, len(X_te))
    X_s = X_te[:n]

    sess = ort.InferenceSession(str(onnx_path),
                                providers=['CPUExecutionProvider'])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape   # e.g. [batch, seq_len, n_feat]

    # ONNX モデルが期待する seq_len / n_feat で X_s を再構築
    onnx_seq = seq_len
    n_feat   = X_s.shape[2]
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

    # 静的バッチサイズ (=1) 対応: バッチ推論できない場合はサンプル数を減らして1件ずつ実行
    static_batch = isinstance(inp_shape[0], int) and inp_shape[0] == 1

    def _infer(X):
        if static_batch:
            outs = np.array([sess.run(None, {inp_name: X[i:i+1]})[0][0] for i in range(len(X))])
        else:
            outs = sess.run(None, {inp_name: X})[0]
        return outs

    # 静的バッチモデルはサンプル数を減らして高速化
    if static_batch and n > 30:
        n = 30
        X_s = X_s[:n]

    # ベースライン
    try:
        base_out = _infer(X_s)   # (n, 3)
    except Exception as e:
        print(f'  [BF] ONNX 推論エラー (skipping): {e}')
        return []
    base_ent = float(-np.mean(base_out * np.log(base_out + 1e-9)))

    rng = np.random.default_rng(42)
    importances = []
    n_feat = X_s.shape[2]
    for i in range(n_feat):
        X_p = X_s.copy()
        X_p[:, :, i] = X_p[rng.permutation(n), :, i]
        perm_out = _infer(X_p)
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

    # ─ all_results.json の上位試行も補完 (top100 ディレクトリ外の試行) ─
    if ALL_RESULTS.exists():
        all_data = json.loads(ALL_RESULTS.read_text(encoding='utf-8'))
        top_by_pf = sorted(all_data, key=lambda x: x.get('pf', 0), reverse=True)[:200]
        already_tno = {r[4].get('trial') for r in to_process}
        tmap = _build_s3_trial_map()
        tmp_dir = AI_EA_DIR / '_tmp_backfill'
        tmp_dir.mkdir(exist_ok=True)
        for ar in top_by_pf:
            tno = ar.get('trial')
            if not tno or tno in already_tno:
                continue
            if ar.get('feature_importance'):
                continue
            if tno not in tmap:
                continue
            # 一時ディレクトリに ONNX/norm をダウンロードして処理
            td = tmp_dir / f'trial_{tno:06d}'
            td.mkdir(exist_ok=True)
            onnx_tmp = td / 'fx_model.onnx'
            norm_tmp = td / 'norm_params.json'
            if not onnx_tmp.exists():
                entry = tmap[tno]
                _s3_download(entry['onnx'], onnx_tmp)
                _s3_download(entry['norm'], norm_tmp)
            if not onnx_tmp.exists():
                continue
            # result から feat_indices/seq_len を取得 (S3 result → all_results のどちらかから)
            sr = tmap[tno].get('result', {})
            feat_indices = ar.get('feat_indices') or sr.get('feat_indices')
            seq_len = ar.get('seq_len') or sr.get('seq_len', 20)
            if norm_tmp.exists() and feat_indices is None:
                try:
                    nd = json.loads(norm_tmp.read_text(encoding='utf-8'))
                    feat_indices = nd.get('feat_indices')
                    if seq_len == 20:
                        seq_len = nd.get('seq_len', 20)
                except Exception:
                    pass
            to_process.append((td, onnx_tmp, norm_tmp, None, ar))
            already_tno.add(tno)

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

        # norm_params.json から補完 (norm_params は ONNX と一致して保存されているのでより信頼性が高い)
        if norm.exists():
            try:
                nd = json.loads(norm.read_text(encoding='utf-8'))
                norm_fi = nd.get('feat_indices')
                # norm_params の feat_indices が all_results より多い場合はそちらを優先
                if norm_fi and (feat_indices is None or len(norm_fi) > len(feat_indices)):
                    feat_indices = norm_fi
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

            # result.json 更新 (top100 ディレクトリ経由の場合のみ)
            if result_f is not None:
                r['feature_importance'] = imp
                result_f.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                    encoding='utf-8')

            # all_results.json も更新
            _update_all_results(trial_no, imp)

        except Exception as e:
            print(f'  → エラー: {e}')
            continue

    # バックフィル結果を S3 に保存
    if S3_ENABLED and ALL_RESULTS.exists():
        try:
            cl = _s3()
            if cl:
                s3_key = (S3_PREFIX + '/' if S3_PREFIX else '') + 'all_results.json'
                cl.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Body=ALL_RESULTS.read_bytes(),
                    ContentType='application/json',
                )
                print(f'[BF] all_results.json → S3 保存完了 ({s3_key})')
        except Exception as e:
            print(f'[BF] S3 保存失敗: {e}')

    print('[BF] バックフィル完了')


if __name__ == '__main__':
    main()
