# FX LLM ファインチューニング — さくらインターネット DOK / H100 80GB

## 概要

| 項目 | 内容 |
|------|------|
| モデル | Qwen/Qwen3-8B (QLoRA 4-bit) |
| GPU | H100 80GB (さくら DOK H100プラン) |
| 訓練精度 | BF16 + Flash Attention 2 |
| LoRA rank | 64 (alpha=128) |
| 実効バッチ | 8 × 8 = 64 |
| データ制限 | なし (全件使用) |
| テストデータ | 直近1年を自動確保 |
| ダッシュボード | http://\<DOK_IP\>:7860 (外部公開) |
| SSH | port 22 (鍵認証のみ) |

---

## セットアップ手順

### 1. ローカルで SSH 鍵生成 + Docker ビルド

```powershell
cd f:\FX
.\DOK\setup_local.ps1
```

- SSH 鍵ペアが `DOK\ssh\` に生成されます
- Docker イメージがビルドされます (`fx-llm:latest`)

### 2. Docker Hub に push (さくら DOK で使用するため)

```powershell
.\DOK\setup_local.ps1 -PushImage -DockerHubUser <あなたのDockerHubユーザー名>
```

### 3. さくら DOK でコンテナを作成

DOK 管理画面 (https://doc.sakura.ad.jp) で以下を設定:

| 項目 | 値 |
|------|----|
| イメージ | `<DockerHubユーザー名>/fx-llm:latest` |
| プラン | **H100 (80GB): ¥0.280/秒** |
| HTTP ポート | 7860 |
| SSH | **有効** |

**環境変数** (オプション):
```
LLM_MODEL_ID=Qwen/Qwen3-8B
LLM_EPOCHS=10
LLM_BATCH=8
LLM_GRAD_ACCUM=8
```

### 4. データをアップロード

コンテナ起動後、SSH で CSV データをアップロード:

```powershell
# SSH 接続
ssh -i DOK\ssh\id_ed25519_dok root@<DOK_IP>

# データアップロード
scp -i DOK\ssh\id_ed25519_dok `
    USDJPY_M1_202301012206_202602250650.csv `
    root@<DOK_IP>:/workspace/data/USDJPY_M1.csv
```

> **注意**: コンテナ起動直後にパイプラインが実行されます。  
> データが見つからない場合はエラーで停止し、アップロード後に再起動してください。

### 5. ダッシュボードでリアルタイム確認

ブラウザで開く:
```
http://<DOK_IP>:7860
```

- 訓練の進捗、Loss/Accuracy チャートがリアルタイム表示
- 訓練完了後はバックテスト結果も表示
- HTML レポートとモデルをダウンロード可能

---

## ファイル構成

```
DOK/
├── Dockerfile              # H100 対応 Docker イメージ
├── requirements.txt        # Python パッケージ
├── entrypoint.sh           # 起動スクリプト (UFW/SSH/ダッシュボード/パイプライン)
├── setup_local.ps1         # ローカルセットアップ (SSH鍵生成, Docker ビルド)
├── ssh/
│   ├── authorized_keys     # SSH 公開鍵 (setup_local.ps1 が生成)
│   └── id_ed25519_dok      # SSH 秘密鍵 (ローカル保管, .gitignore)
└── src/
    ├── pipeline.py         # 全工程統合スクリプト
    ├── dataset_prep.py     # データセット生成 (3000件制限なし)
    ├── train_h100.py       # H100 最適化ファインチューニング
    ├── dashboard_server.py # Flask ダッシュボード (port 7860)
    └── backtest_report.py  # バックテスト + HTML レポート生成
```

---

## H100 最適化パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `--batch` | 8 | ミニバッチサイズ |
| `--grad_accum` | 8 | 勾配累積 → 実効バッチ 64 |
| `--lora_r` | 64 | LoRA rank (高精度) |
| `--lora_alpha` | 128 | LoRA alpha = 2×rank |
| `--max_length` | 1024 | 最大トークン長 |
| `--lr` | 5e-5 | 学習率 (cosine decay) |
| dtype | BF16 | H100 ネイティブ精度 |
| Flash Attention | 有効 | Unsloth 組み込み |

---

## チェックポイントから再開

コンテナを再起動して再開する場合:

```
環境変数: LLM_RESUME=1
```

または SSH で直接:
```bash
python /workspace/src/pipeline.py --resume --skip_dataset
```

---

## ダウンロード

| URL | 内容 |
|-----|------|
| `http://<IP>:7860/download/report` | バックテスト HTML レポート |
| `http://<IP>:7860/download/adapter` | 学習済みアダプター (tar.gz) |

---

## 費用目安 (H100 80GB)

- ¥0.280/秒 = ¥1,008/時間
- Qwen3-8B 10エポック ≈ 2〜4時間 → **¥2,000〜¥4,000**
