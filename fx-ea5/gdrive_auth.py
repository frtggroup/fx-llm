"""
Google Drive OAuth2 リフレッシュトークン取得ツール
一度だけ実行してリフレッシュトークンを取得する。
取得したトークンは GDRIVE_OAUTH_REFRESH_TOKEN 環境変数として設定する。

使い方:
  1. Google Cloud Console で OAuth2 クライアントID (デスクトップアプリ) を作成
  2. client_id と client_secret を手元に用意
  3. python gdrive_auth.py
  4. 表示されたURLをブラウザで開き、認証後コードをペースト
  5. 表示されたリフレッシュトークンを GDRIVE_OAUTH_REFRESH_TOKEN に設定
"""
import json, sys, webbrowser
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("pip install google-auth-oauthlib が必要です")
    sys.exit(1)

SCOPES = ['https://www.googleapis.com/auth/drive']

client_id     = input("OAuth2 client_id を入力: ").strip()
client_secret = input("OAuth2 client_secret を入力: ").strip()

client_config = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
# ブラウザ不要のコンソールフロー
flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')

print("\n=== 以下のURLをブラウザで開いて認証してください ===")
print(auth_url)
print()

try:
    webbrowser.open(auth_url)
except Exception:
    pass

code = input("認証後に表示されたコードをペーストしてください: ").strip()
flow.fetch_token(code=code)
creds = flow.credentials

print("\n=== 認証成功 ===")
print(f"GDRIVE_OAUTH_CLIENT_ID     = {client_id}")
print(f"GDRIVE_OAUTH_CLIENT_SECRET = {client_secret}")
print(f"GDRIVE_OAUTH_REFRESH_TOKEN = {creds.refresh_token}")
print()
print("上記3つを docker run -e または docker-compose の environment に追加してください。")
