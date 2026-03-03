"""
CSV ファイルを Google Drive の共有フォルダにアップロードするユーティリティ。

使い方:
    python ai_ea/gdrive_upload_csv.py path/to/USDJPY_H1.csv

環境変数 (Dockerfile と同じ値を設定してください):
    GDRIVE_FOLDER_ID
    GDRIVE_OAUTH_CLIENT_ID
    GDRIVE_OAUTH_CLIENT_SECRET
    GDRIVE_OAUTH_REFRESH_TOKEN
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gdrive

def main():
    if len(sys.argv) < 2:
        print("使い方: python gdrive_upload_csv.py <CSVファイルパス>")
        print("例   : python gdrive_upload_csv.py data/USDJPY_H1.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {csv_path}")
        sys.exit(1)

    if not gdrive.GDRIVE_ENABLED:
        print("[ERROR] GDrive が設定されていません。")
        print("  GDRIVE_FOLDER_ID / GDRIVE_OAUTH_* 環境変数を設定してください。")
        sys.exit(1)

    print(f"[*] アップロード中: {csv_path.name}  ({csv_path.stat().st_size/1e6:.1f} MB)")
    ok = gdrive.upload(csv_path, csv_path.name)
    if ok:
        print(f"[OK] アップロード完了: {csv_path.name}")
        print(f"     GDrive フォルダ: https://drive.google.com/drive/folders/{gdrive.GDRIVE_FOLDER_ID}")
        print()
        print("コンテナ起動時に自動でダウンロードされます。DATA_URL の指定は不要です。")
    else:
        print("[ERROR] アップロード失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()
