============================================================
 FX_AI_EA.mq5  使用手順
============================================================

【必要ファイル】
  ダッシュボード (http://<DOK_IP>:7860) の TOP100 から
  好きなモデルをダウンロードすると以下が含まれています:
  
    fx_model.onnx         ← ONNXモデル本体
    norm_params.json      ← 正規化パラメータ (特徴量・seq_len情報)
    result.json           ← バックテスト成績

  ※ベストモデルは /download/best からも取得できます。

【セットアップ手順】
  1. ファイルをコピー
     - fx_model.onnx     → MT5の「MQL5/Files/」フォルダにコピー
     - norm_params.json  → MT5の「MQL5/Files/」フォルダにコピー
     ※名前を変えた場合はEAのパラメータで指定してください

  2. EAをコンパイル
     - MetaEditor で FX_AI_EA.mq5 を開く
     - F7 でコンパイル (エラーなし確認)

  3. チャートにアタッチ
     - USDJPY, H1 チャートにアタッチ
     - 「アルゴリズム取引を許可する」にチェック

  4. パラメータ設定
     ┌──────────────────────────────────────────┐
     │ ModelFile   : fx_model_best.onnx          │  ← ONNXファイル名
     │ NormFile    : norm_params_best.json        │  ← JSONファイル名
     │ Threshold   : 0.40                         │  ← 確率閾値 (0.35-0.50)
     │ TpAtr       : 2.0                          │  ← TP = ATR × 2.0
     │ SlAtr       : 1.0                          │  ← SL = ATR × 1.0
     │ MaxHoldBars : 48                           │  ← 最大保有48バー(2日)
     │ RiskPct     : 1.0                          │  ← 残高の1%リスク
     └──────────────────────────────────────────┘

【入力の仕組み】
  モデルへの入力: [1, seq_len, n_features]
  
  - seq_len     : norm_params.json の "seq_len" から自動取得
  - n_features  : norm_params.json の "feat_indices" から自動取得
    (モデルごとに使用特徴量が異なります)
  
  処理フロー:
    H1バーの確定 → 70特徴量を計算 → feat_indicesで選択
    → mean/stdで正規化 → ONNXモデルで推論
    → [HOLD, BUY, SELL] 確率 → threshold超えでエントリー

【ロット計算 (MQL5 LotSize相当)】
  magnification = 10000 (JPY口座)
  lot = ceil(余剰証拠金 × RiskPct / 10000) / 100 - 0.01
  
  例) 余剰証拠金150,000円, RiskPct=1.0 の場合:
    ceil(150000 × 1.0 / 10000) / 100 - 0.01 = 0.14 lot

【注意事項】
  - 対応: USDJPY H1 のみ
  - MQL5 Build 3370以上 (ONNX対応)
  - バックテスト期間: 直近1年のデータでテストしてください
  - 実運用前に必ずストラテジーテスターで検証すること

【ファイル配置例】
  C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\<id>\
    └── MQL5\
        ├── Experts\
        │   └── FX_AI_EA.mq5        ← EAファイル
        └── Files\
            ├── fx_model_best.onnx  ← モデル
            └── norm_params_best.json ← 正規化パラメータ

============================================================
