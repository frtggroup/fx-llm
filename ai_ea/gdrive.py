"""
Google Drive 共有ストレージモジュール
複数ノード（H100/GTX等）が同一フォルダを競合なしで共有する。

設計:
  - 各ノードは自分の NODE_ID プレフィックスを持つファイルのみ書き込む
  - 読み込み時は全ノードのファイルをダウンロードしてマージ
  - フォルダ構造:
      <GDRIVE_FOLDER_ID>/
        results_h100.json
        results_gtx1080ti.json
        meta_h100.json
        meta_gtx1080ti.json
        best_h100/
          fx_model_best.onnx
          norm_params_best.json
          best_result.json
        top100_h100/
          rank_001/
            fx_model.onnx
            norm_params.json
        top100_gtx1080ti/
          ...

環境変数:
  GDRIVE_FOLDER_ID           共有フォルダの ID
  GDRIVE_CREDENTIALS_BASE64  サービスアカウント JSON を base64 エンコードした文字列
"""

from __future__ import annotations
import base64, io, json, os, threading
from pathlib import Path
from typing import Optional

GDRIVE_FOLDER_ID   = os.environ.get('GDRIVE_FOLDER_ID', '')
GDRIVE_CREDS_B64   = os.environ.get('GDRIVE_CREDENTIALS_BASE64', '')
# OAuth2 方式 (個人 My Drive に書き込む場合。サービスアカウントより優先)
GDRIVE_OAUTH_CLIENT_ID     = os.environ.get('GDRIVE_OAUTH_CLIENT_ID', '')
GDRIVE_OAUTH_CLIENT_SECRET = os.environ.get('GDRIVE_OAUTH_CLIENT_SECRET', '')
GDRIVE_OAUTH_REFRESH_TOKEN = os.environ.get('GDRIVE_OAUTH_REFRESH_TOKEN', '')
_USE_OAUTH = bool(GDRIVE_OAUTH_CLIENT_ID and GDRIVE_OAUTH_CLIENT_SECRET and GDRIVE_OAUTH_REFRESH_TOKEN)

GDRIVE_ENABLED     = bool(GDRIVE_FOLDER_ID and (GDRIVE_CREDS_B64 or _USE_OAUTH))

_SCOPES = ['https://www.googleapis.com/auth/drive']

# フォルダIDキャッシュ (パス文字列 → GDrive folder ID)
_folder_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


_HTTP_TIMEOUT = 30  # 全 Drive API コールのソケットタイムアウト (秒)


def _build_service():
    """Google Drive API サービスオブジェクトを生成 (OAuth2 優先 → サービスアカウント)
    socket.setdefaulttimeout でソケットレベルのタイムアウトを設定することで、
    list/download/upload 含む全 API コールが無限待ちにならないようにする。
    """
    import socket
    socket.setdefaulttimeout(_HTTP_TIMEOUT)   # 根本解決: 全ソケット操作にタイムアウト

    from googleapiclient.discovery import build

    if _USE_OAUTH:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials(
            token=None,
            refresh_token=GDRIVE_OAUTH_REFRESH_TOKEN,
            client_id=GDRIVE_OAUTH_CLIENT_ID,
            client_secret=GDRIVE_OAUTH_CLIENT_SECRET,
            token_uri='https://oauth2.googleapis.com/token',
        )
        creds.refresh(Request())
    else:
        from google.oauth2 import service_account
        creds_json = base64.b64decode(GDRIVE_CREDS_B64).decode('utf-8')
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)

    return build('drive', 'v3', credentials=creds, cache_discovery=False)


_svc_lock = threading.Lock()
_svc_instance = None

def _svc():
    global _svc_instance
    if _svc_instance is None:
        with _svc_lock:
            if _svc_instance is None:
                _svc_instance = _build_service()
    return _svc_instance


def _get_or_create_folder(parent_id: str, name: str) -> str:
    """parent_id 配下の name フォルダを取得 (なければ作成)"""
    cache_key = f'{parent_id}/{name}'
    with _cache_lock:
        if cache_key in _folder_cache:
            return _folder_cache[cache_key]

    svc = _svc()
    q = (f"'{parent_id}' in parents and name='{name}' "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    res = svc.files().list(q=q, fields='files(id,name)', pageSize=1).execute()
    files = res.get('files', [])
    if files:
        fid = files[0]['id']
    else:
        meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]}
        fid = svc.files().create(body=meta, fields='id').execute()['id']

    with _cache_lock:
        _folder_cache[cache_key] = fid
    return fid


def _resolve_folder(rel_path: str) -> tuple[str, str]:
    """
    rel_path = 'best_h100/fx_model_best.onnx' のような相対パスを解決。
    Returns (parent_folder_id, filename)
    """
    parts = rel_path.replace('\\', '/').split('/')
    filename = parts[-1]
    folder_id = GDRIVE_FOLDER_ID
    for part in parts[:-1]:
        folder_id = _get_or_create_folder(folder_id, part)
    return folder_id, filename


def _find_file(folder_id: str, name: str) -> Optional[str]:
    """フォルダ内のファイル ID を返す (なければ None)"""
    svc = _svc()
    q = f"'{folder_id}' in parents and name='{name}' and trashed=false"
    res = svc.files().list(q=q, fields='files(id)', pageSize=1).execute()
    files = res.get('files', [])
    return files[0]['id'] if files else None


def make_public_link(rel_key: str, timeout: float = 60.0) -> str:
    """
    rel_key のファイルを「リンクを知っている全員がダウンロード可能」に設定し、
    直接ダウンロードURL を返す。
    失敗時は空文字列を返す。
    """
    if not GDRIVE_ENABLED:
        return ''

    result = ['']
    error  = [None]

    def _do():
        try:
            svc = _svc()
            folder_id, fname = _resolve_folder(rel_key)
            file_id = _find_file(folder_id, fname)
            if not file_id:
                return
            # 誰でも読める権限を付与 (既に付いていれば上書きされず無害)
            try:
                svc.permissions().create(
                    fileId=file_id,
                    body={'type': 'anyone', 'role': 'reader'},
                    fields='id',
                ).execute()
            except Exception:
                pass  # 権限が既に存在する場合も続行
            # webContentLink = 直接ダウンロードURL (ONNX/JSON 用)
            meta = svc.files().get(
                fileId=file_id,
                fields='webContentLink,webViewLink',
            ).execute()
            link = meta.get('webContentLink') or meta.get('webViewLink', '')
            result[0] = link
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f'  [GDrive] make_public_link タイムアウト ({timeout:.0f}s): {rel_key}')
        return ''
    if error[0]:
        print(f'  [GDrive] make_public_link 失敗 {rel_key}: {error[0]}')
        return ''
    return result[0]


def upload_and_share(local_path: Path, rel_key: str, timeout: float = 180.0) -> str:
    """
    ファイルをアップロードして公開リンクを返す (upload + make_public_link の合体版)。
    タイムアウトまたは失敗時は空文字列を返す。
    """
    if not GDRIVE_ENABLED:
        return ''
    if not upload(local_path, rel_key, timeout=timeout):
        return ''
    return make_public_link(rel_key, timeout=60.0)


def upload(local_path: Path, rel_key: str, timeout: float = 120.0) -> bool:
    """
    local_path のファイルを GDrive の rel_key にアップロード。
    既存ファイルがあれば上書き (update)、なければ新規作成。
    timeout 秒以内に完了しない場合は False を返す (メインループのブロック防止)。
    Returns True on success.
    """
    if not GDRIVE_ENABLED:
        return False

    result = [False]
    error  = [None]

    def _do_upload():
        try:
            from googleapiclient.http import MediaFileUpload
            svc = _svc()
            folder_id, fname = _resolve_folder(rel_key)
            mime = _guess_mime(fname)
            media = MediaFileUpload(str(local_path), mimetype=mime, resumable=False)
            existing_id = _find_file(folder_id, fname)
            if existing_id:
                svc.files().update(fileId=existing_id, media_body=media).execute()
            else:
                meta = {'name': fname, 'parents': [folder_id]}
                svc.files().create(body=meta, media_body=media, fields='id').execute()
            result[0] = True
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_do_upload, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f'  [GDrive] upload タイムアウト ({timeout:.0f}s): {rel_key}')
        return False
    if error[0]:
        print(f'  [GDrive] upload失敗 {rel_key}: {error[0]}')
        return False
    return result[0]


def download(rel_key: str, local_path: Path, timeout: float = 60.0) -> bool:
    """
    GDrive の rel_key を local_path にダウンロード。
    timeout 秒でハングした場合は諦めて False を返す。
    Returns True on success.
    """
    if not GDRIVE_ENABLED:
        return False

    import threading as _threading

    result = [False]
    error  = [None]

    def _do_download():
        try:
            from googleapiclient.http import MediaIoBaseDownload
            svc = _svc()
            folder_id, fname = _resolve_folder(rel_key)
            fid = _find_file(folder_id, fname)
            if not fid:
                return
            local_path.parent.mkdir(parents=True, exist_ok=True)
            request = svc.files().get_media(fileId=fid)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, request, chunksize=4 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk()
            local_path.write_bytes(buf.getvalue())
            result[0] = True
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_do_download, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f'  [GDrive] download タイムアウト ({timeout:.0f}s): {rel_key} → スキップ')
        return False
    if error[0]:
        print(f'  [GDrive] download失敗 {rel_key}: {error[0]}')
        return False
    return result[0]


def list_keys(prefix: str = '') -> list[str]:
    """
    GDrive ルートフォルダ直下の「ファイル」一覧 (prefix 一致) を返す。
    フォルダは除外。サブフォルダは再帰しない (ルートレベルのみ)。
    """
    if not GDRIVE_ENABLED:
        return []
    try:
        svc = _svc()
        q = (f"'{GDRIVE_FOLDER_ID}' in parents and trashed=false "
             f"and mimeType!='application/vnd.google-apps.folder'")
        res = svc.files().list(q=q, fields='files(id,name)', pageSize=200).execute()
        names = [f['name'] for f in res.get('files', [])]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return names
    except Exception as e:
        print(f'  [GDrive] list失敗: {e}')
        return []


def list_keys_recursive(folder_prefix: str, timeout: float = 30.0) -> list[str]:
    """
    GDrive ルート配下の folder_prefix フォルダを再帰的に辿り、
    全ファイルの相対パスを返す。timeout 秒でハングした場合は空リストを返す。
    """
    if not GDRIVE_ENABLED:
        return []

    result: list[list] = [[]]
    error:  list[Exception] = [None]

    def _do():
        try:
            svc = _svc()
            found: list[str] = []

            def _recurse(parent_id: str, rel_base: str):
                q = f"'{parent_id}' in parents and trashed=false"
                page_token = None
                while True:
                    kwargs = dict(q=q, fields='files(id,name,mimeType)', pageSize=200)
                    if page_token:
                        kwargs['pageToken'] = page_token
                    res = svc.files().list(**kwargs).execute()
                    for item in res.get('files', []):
                        rel = f'{rel_base}/{item["name"]}' if rel_base else item['name']
                        if item['mimeType'] == 'application/vnd.google-apps.folder':
                            _recurse(item['id'], rel)
                        else:
                            found.append(rel)
                    page_token = res.get('nextPageToken')
                    if not page_token:
                        break

            q = (f"'{GDRIVE_FOLDER_ID}' in parents and trashed=false and "
                 f"mimeType='application/vnd.google-apps.folder'")
            res = svc.files().list(q=q, fields='files(id,name)', pageSize=100).execute()
            for f in res.get('files', []):
                if f['name'].startswith(folder_prefix):
                    _recurse(f['id'], f['name'])
            result[0] = found
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f'  [GDrive] list_recursive タイムアウト ({timeout:.0f}s): {folder_prefix}* → スキップ')
        return []
    if error[0]:
        print(f'  [GDrive] list_recursive失敗: {error[0]}')
        return []
    return result[0]


def list_node_keys(glob_prefix: str, timeout: float = 20.0) -> list[str]:
    """
    全ノードの同種ファイル一覧 (例: 'results_' → ['results_h100.json', ...])
    ルートレベルのみ。timeout 秒でハングした場合は空リストを返す。
    """
    if not GDRIVE_ENABLED:
        return []

    result: list[list] = [[]]

    def _do():
        result[0] = list_keys(glob_prefix)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f'  [GDrive] list_node_keys タイムアウト ({timeout:.0f}s): {glob_prefix}* → スキップ')
        return []
    return result[0]


def upload_bytes(data: bytes, rel_key: str) -> bool:
    """bytes データをファイルとして GDrive にアップロード"""
    if not GDRIVE_ENABLED:
        return False
    try:
        from googleapiclient.http import MediaIoBaseUpload
        svc = _svc()
        folder_id, fname = _resolve_folder(rel_key)
        mime = _guess_mime(fname)
        buf = io.BytesIO(data)
        media = MediaIoBaseUpload(buf, mimetype=mime, resumable=False)
        existing_id = _find_file(folder_id, fname)
        if existing_id:
            svc.files().update(fileId=existing_id, media_body=media).execute()
        else:
            meta = {'name': fname, 'parents': [folder_id]}
            svc.files().create(body=meta, media_body=media, fields='id').execute()
        return True
    except Exception as e:
        print(f'  [GDrive] upload_bytes失敗 {rel_key}: {e}')
        return False


def download_bytes(rel_key: str, timeout: float = 60.0) -> Optional[bytes]:
    """GDrive から bytes としてダウンロード。失敗時 None"""
    if not GDRIVE_ENABLED:
        return None

    import threading as _threading

    result = [None]

    def _do():
        try:
            from googleapiclient.http import MediaIoBaseDownload
            svc = _svc()
            folder_id, fname = _resolve_folder(rel_key)
            fid = _find_file(folder_id, fname)
            if not fid:
                return
            request = svc.files().get_media(fileId=fid)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, request, chunksize=4 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk()
            result[0] = buf.getvalue()
        except Exception as e:
            print(f'  [GDrive] download_bytes失敗 {rel_key}: {e}')

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f'  [GDrive] download_bytes タイムアウト ({timeout:.0f}s): {rel_key}')
        return None
    return result[0]


def test_connection() -> bool:
    """接続テスト。書き込み・読み込み・削除を確認"""
    if not GDRIVE_ENABLED:
        print('  [GDrive] 無効 (GDRIVE_FOLDER_ID または GDRIVE_CREDENTIALS_BASE64 未設定)')
        return False
    try:
        svc = _svc()
        # フォルダ情報取得でテスト
        info = svc.files().get(fileId=GDRIVE_FOLDER_ID, fields='name,id').execute()
        print(f'  [GDrive] 接続OK ✅: フォルダ "{info.get("name")}" (ID:{GDRIVE_FOLDER_ID})')
        # 書き込みテスト
        test_data = b'gdrive_test_' + str(os.getpid()).encode()
        if upload_bytes(test_data, '_test_write.tmp'):
            dl = download_bytes('_test_write.tmp')
            if dl == test_data:
                print('  [GDrive] 読み書きテスト OK ✅')
                # 削除
                try:
                    folder_id, fname = _resolve_folder('_test_write.tmp')
                    fid = _find_file(folder_id, fname)
                    if fid:
                        svc.files().delete(fileId=fid).execute()
                except Exception:
                    pass
                return True
        print('  [GDrive] 書き込みテスト 失敗 ❌')
        return False
    except Exception as e:
        print(f'  [GDrive] 接続テスト 失敗 ❌: {e}')
        return False


def _guess_mime(fname: str) -> str:
    ext = Path(fname).suffix.lower()
    return {'.json': 'application/json', '.onnx': 'application/octet-stream',
            '.html': 'text/html', '.csv': 'text/csv',
            '.pkl': 'application/octet-stream'}.get(ext, 'application/octet-stream')
