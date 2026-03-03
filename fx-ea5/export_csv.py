import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# パスを追加
work_dir = Path(os.path.abspath(r"f:\FX\fx-ea5"))
sys.path.insert(0, str(work_dir))

from features import load_data, add_indicators, BASE_FEATURE_COLS, make_labels

def main():
    DATA_PATH = work_dir.parent / 'USDJPY_M1_202301020700_202602262003.csv'
    
    print("データをロード中...")
    df = load_data(str(DATA_PATH), timeframe='H1')
    
    print("特徴量(indicators)を追加中...")
    df = add_indicators(df)
    
    print("データをクリーンアップ中...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print("ラベル(target)を生成中...")
    labels = make_labels(df, tp_atr=1.5, sl_atr=1.0, forward_bars=20)
    df['target'] = labels
    
    # 有効な行のみ抽出（forward_barsによる末尾の無効データをカット）
    valid_end = len(df) - 20 - 1
    
    # label列を追加（hold/buy/sell）
    df['label'] = df['target'].map({0: 'hold', 1: 'buy', 2: 'sell'})

    # 230次元の特徴量 + 基本価格データ + ターゲット
    cols_to_keep = ['open', 'high', 'low', 'close', 'target', 'label'] + BASE_FEATURE_COLS
    df_export = df[cols_to_keep].iloc[:valid_end]
    
    # 訓練用・テスト用の分割 (このプログラム（train.py）と同じ方法: 2025-01-01で分割)
    TEST_SPLIT_DATE = pd.Timestamp('2025-01-01', tz=df_export.index.tz)
    df_train = df_export[df_export.index < TEST_SPLIT_DATE]
    df_test = df_export[df_export.index >= TEST_SPLIT_DATE]
    
    out_dir = Path(r"f:\FX\FX-DATA")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    
    print(f"訓練データを保存中 ({len(df_train)} 行): {train_path}")
    df_train.to_csv(train_path)
    
    print(f"テストデータを保存中 ({len(df_test)} 行): {test_path}")
    df_test.to_csv(test_path)
    
    print("完了しました。")

if __name__ == '__main__':
    main()
