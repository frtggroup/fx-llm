"""
FX AI EA - 特徴量セット 100種 (700個対応版)
仕様する特徴量をグループ分けされたリストから出す
"""

import random

# 再現性のため
random.seed(42)

import features as fx
N_GROUPS = len(fx.BASE_FEATURE_COLS)

# 意味的なグループのインデックス範囲 (大まかに分類)
TREND       = list(range(0, 16))
MACD_GRP    = list(range(16, 25))
RSI_GRP     = list(range(25, 29))
STOCH_GRP   = list(range(29, 35))
BB_GRP      = list(range(35, 39))
ATR_ETC_GRP = list(range(39, 57))
OTHER_IND   = list(range(57, 67))
DIFF_GRP    = list(range(67, 73))
ICHI_GRP    = list(range(73, 80))
STATS_GRP   = list(range(80, 95))
CANDLE_GRP  = list(range(95, 104))
BOS_GRP     = list(range(104, 119))
SESS_GRP    = list(range(119, 127))
EXTRA_GRP   = list(range(127, N_GROUPS))

def _s(*groups):
    out = set()
    for g in groups:
        out.update(g)
    return sorted(out)

FEATURE_SETS = []

# トレンド系中心
FEATURE_SETS.append(_s(TREND))
FEATURE_SETS.append(_s(TREND, MACD_GRP))
FEATURE_SETS.append(_s(TREND, RSI_GRP, BB_GRP))

# モメンタム中心
FEATURE_SETS.append(_s(MACD_GRP, RSI_GRP, STOCH_GRP))
FEATURE_SETS.append(_s(RSI_GRP, ICHI_GRP))

# ボラティリティ中心
FEATURE_SETS.append(_s(BB_GRP, ATR_ETC_GRP))

# BOS/プライスアクション中心
FEATURE_SETS.append(_s(BOS_GRP))
FEATURE_SETS.append(_s(BOS_GRP, CANDLE_GRP))
FEATURE_SETS.append(_s(BOS_GRP, TREND))
FEATURE_SETS.append(_s(BOS_GRP, SESS_GRP))

# 統計・変則
FEATURE_SETS.append(_s(STATS_GRP, DIFF_GRP))
FEATURE_SETS.append(_s(STATS_GRP, OTHER_IND))

# カテゴリの組み合わせ (12個できたので、残り88個はサイズ別のランダムな組み合わせやカテゴリ複合で生成)

categories = [TREND, MACD_GRP, RSI_GRP, STOCH_GRP, BB_GRP, ATR_ETC_GRP, 
              OTHER_IND, DIFF_GRP, ICHI_GRP, STATS_GRP, CANDLE_GRP, BOS_GRP, SESS_GRP]

# サイズ別のバランスセット (5, 10, 20, 30, 50, 100, 200, 300, 500特徴量)
size_targets = [5, 10, 20, 30, 50, 100, 200, 300, 400, 500, N_GROUPS]

while len(FEATURE_SETS) < 100:
    target_size = random.choice(size_targets)
    if target_size >= N_GROUPS:
        FEATURE_SETS.append(list(range(N_GROUPS)))
        continue
        
    mode = random.choice(['category_mix', 'pure_random', 'hybrid'])
    
    if mode == 'category_mix':
        # いくつかのカテゴリを選んで追加
        num_cats = random.randint(1, len(categories))
        chosen_cats = random.sample(categories, num_cats)
        combined = _s(*chosen_cats)
        if len(combined) > target_size:
            combined = sorted(random.sample(combined, target_size))
        FEATURE_SETS.append(combined)
        
    elif mode == 'pure_random':
        # 全体からランダムに target_size 個選ぶ
        FEATURE_SETS.append(sorted(random.sample(range(N_GROUPS), target_size)))
        
    elif mode == 'hybrid':
        # 特定カテゴリ + アルファ
        base_cat = random.choice(categories)
        base = list(base_cat)
        rem = target_size - len(base)
        if rem > 0:
            pool = list(set(range(N_GROUPS)) - set(base))
            base.extend(random.sample(pool, min(rem, len(pool))))
        base = sorted(base)
        if len(base) > target_size:
            base = sorted(random.sample(base, target_size))
        FEATURE_SETS.append(base)

# 重複除去とソート
unique_sets = []
seen = set()
for s in FEATURE_SETS:
    t = tuple(s)
    if t not in seen:
        seen.add(t)
        unique_sets.append(s)

# 万が一重複で100個未満になったら補充
while len(unique_sets) < 100:
    target_size = random.choice(size_targets)
    s = sorted(random.sample(range(N_GROUPS), target_size))
    t = tuple(s)
    if t not in seen:
        seen.add(t)
        unique_sets.append(s)
        
FEATURE_SETS = unique_sets[:100]

assert len(FEATURE_SETS) == 100, f"セット数={len(FEATURE_SETS)}"
