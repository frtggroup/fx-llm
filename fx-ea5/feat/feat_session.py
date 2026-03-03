"""
feat/feat_session.py — 時間・セッション特徴量
対応 BASE_FEATURE_COLS index: 119-126
"""
import numpy as np
import pandas as pd


def add_session(df_index: pd.DatetimeIndex) -> dict:
    hr  = df_index.hour
    dow = df_index.dayofweek
    return {
        'hour_sin':  pd.Series(np.sin(2 * np.pi * hr  / 24), index=df_index),
        'hour_cos':  pd.Series(np.cos(2 * np.pi * hr  / 24), index=df_index),
        'dow_sin':   pd.Series(np.sin(2 * np.pi * dow / 5),  index=df_index),
        'dow_cos':   pd.Series(np.cos(2 * np.pi * dow / 5),  index=df_index),
        'is_tokyo':  pd.Series(((hr >= 0) & (hr < 9)).astype(float),  index=df_index),
        'is_london': pd.Series(((hr >= 7) & (hr < 16)).astype(float), index=df_index),
        'is_ny':     pd.Series(((hr >= 13) & (hr < 22)).astype(float),index=df_index),
        'is_overlap':pd.Series(((hr >= 13) & (hr < 16)).astype(float),index=df_index),
    }
