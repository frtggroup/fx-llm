"""
LLM v2 バックテスト + HTML レポート生成
v2データセット (llm_test_v2.jsonl) に対応。
CoT ラベルにも対応: "Trend bullish, RSI rising. → BUY" から BUY を抽出。

通常ラベルとCoTラベルを自動判別して推論する。
"""
import sys, json, time, re, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/ai_ea')
sys.path.insert(0, '/workspace/src')

# v2用の定数
WORKSPACE      = Path('/workspace')
ADAPTER_DIR_V2 = WORKSPACE / 'output' / 'llm_adapter_best_v2'  # v2訓練済みアダプター
ADAPTER_DIR_V1 = WORKSPACE / 'output' / 'llm_adapter_best'     # v1にフォールバック
TEST_JSONL_V2  = WORKSPACE / 'output' / 'llm_test_v2.jsonl'
TEST_JSONL_V2C = WORKSPACE / 'output' / 'llm_test_v2_cot.jsonl'
REPORT_DIR     = WORKSPACE / 'reports'
PROGRESS_JSON  = WORKSPACE / 'progress.json'
DATA_PATH      = WORKSPACE / 'data' / 'USDJPY_M1.csv'

LABEL_NAMES    = ['HOLD', 'BUY', 'SELL']
SPREAD         = 0.003
HOLD_BARS      = 48


def update_progress(patch: dict) -> None:
    try:
        cur = {}
        if PROGRESS_JSON.exists():
            cur = json.loads(PROGRESS_JSON.read_text())
        cur.update(patch)
        PROGRESS_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# モデル読み込み
# ──────────────────────────────────────────────────────────────────────────────
def load_model(adapter_dir: Path):
    import torch
    from unsloth import FastLanguageModel
    print(f'  アダプター: {adapter_dir}')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = str(adapter_dir),
        max_seq_length    = 1536,   # v2は少し長い
        load_in_4bit      = True,
        dtype             = None,
        trust_remote_code = True,
    )
    FastLanguageModel.for_inference(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer, device


def make_chat_prompt(prompt_text: str, tokenizer) -> str:
    messages = [
        {'role': 'system',
         'content': ('You are a professional FX trading signal analyst. '
                     'Analyze the market data carefully and respond with '
                     'exactly one word: BUY, SELL, or HOLD.')},
        {'role': 'user', 'content': prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def make_chat_prompt_cot(prompt_text: str, tokenizer) -> str:
    """CoTモード: 推論してから → BUY/SELL/HOLDと答えさせる"""
    messages = [
        {'role': 'system',
         'content': ('You are a professional FX trading signal analyst. '
                     'Analyze the market data step by step, then end your '
                     'response with exactly: → BUY, → SELL, or → HOLD.')},
        {'role': 'user', 'content': prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


# ──────────────────────────────────────────────────────────────────────────────
# 推論 (通常モード: 次トークンlogit)
# ──────────────────────────────────────────────────────────────────────────────
def predict_logit(model, tokenizer, samples: list, device,
                  batch_size: int = 32) -> tuple[list, list]:
    import torch
    label_tids = [tokenizer.encode(n, add_special_tokens=False)[0]
                  for n in LABEL_NAMES]
    lbl_tensor = torch.tensor(label_tids, device=device)
    preds, confs = [], []
    n  = len(samples)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch  = samples[start: start + batch_size]
        texts  = [make_chat_prompt(s['prompt'], tokenizer) for s in batch]
        enc    = tokenizer(texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=1536)
        ids    = enc['input_ids'].to(device)
        masks  = enc['attention_mask'].to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                                  enabled=(device.type == 'cuda')):
            out = model(input_ids=ids, attention_mask=masks)

        last_pos = masks.sum(dim=1) - 1
        for b in range(len(batch)):
            lgt   = out.logits[b, last_pos[b], lbl_tensor]
            probs = torch.softmax(lgt.float(), dim=0).cpu().numpy()
            idx   = int(np.argmax(probs))
            preds.append(LABEL_NAMES[idx])
            confs.append(float(probs[idx]))

        done = start + len(batch)
        print(f'  推論 {done:,}/{n:,}  {time.time()-t0:.0f}s', end='\r', flush=True)
        update_progress({'message': f'推論中... {done:,}/{n:,}'})

    print(f'\n  推論完了: {n:,} サンプル  {time.time()-t0:.1f}s')
    return preds, confs


# ──────────────────────────────────────────────────────────────────────────────
# 推論 (CoTモード: generate → 末尾から抽出)
# ──────────────────────────────────────────────────────────────────────────────
def predict_cot(model, tokenizer, samples: list, device,
                batch_size: int = 8, max_new_tokens: int = 80) -> tuple[list, list]:
    import torch
    preds, confs = [], []
    n  = len(samples)
    t0 = time.time()
    _cot_re = re.compile(r'→\s*(BUY|SELL|HOLD)', re.IGNORECASE)

    for start in range(0, n, batch_size):
        batch  = samples[start: start + batch_size]
        texts  = [make_chat_prompt_cot(s['prompt'], tokenizer) for s in batch]
        enc    = tokenizer(texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=1536)
        ids    = enc['input_ids'].to(device)
        masks  = enc['attention_mask'].to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids      = ids,
                attention_mask = masks,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
                pad_token_id   = tokenizer.eos_token_id,
            )

        for b in range(len(batch)):
            new_ids   = out_ids[b, ids.shape[1]:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True)
            m = _cot_re.search(generated)
            if m:
                preds.append(m.group(1).upper())
                confs.append(0.9)  # CoTは信頼度固定
            else:
                # 末尾の単語で判定
                last_word = generated.strip().split()[-1].upper() if generated.strip() else 'HOLD'
                preds.append(last_word if last_word in LABEL_NAMES else 'HOLD')
                confs.append(0.5)

        done = start + len(batch)
        print(f'  CoT推論 {done:,}/{n:,}  {time.time()-t0:.0f}s', end='\r', flush=True)
        update_progress({'message': f'CoT推論中... {done:,}/{n:,}'})

    print(f'\n  CoT推論完了: {n:,} サンプル  {time.time()-t0:.1f}s')
    return preds, confs


# ──────────────────────────────────────────────────────────────────────────────
# バックテスト（backtest_report.py と共通ロジック）
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(preds: list, df_te,
                 seq_len: int  = 20,
                 tp_mult: float = 1.5,
                 sl_mult: float = 1.0) -> dict:
    from features import add_indicators
    if 'atr14' not in df_te.columns:
        df_te = add_indicators(df_te.copy())

    close  = df_te['close'].values
    atr    = df_te['atr14'].values
    high   = df_te['high'].values
    low    = df_te['low'].values
    dates  = df_te.index
    n      = len(preds)
    trades = []
    pos    = None

    for i in range(n):
        bi = seq_len - 1 + i
        if bi >= len(close):
            break
        c = close[bi]; a = atr[bi]
        pred_str = preds[i]

        if pos:
            # TP/SL チェック
            tp_hit = (pos['type'] == 'BUY'  and high[bi] >= pos['tp']) or \
                     (pos['type'] == 'SELL' and low[bi]  <= pos['tp'])
            sl_hit = (pos['type'] == 'BUY'  and low[bi]  <= pos['sl']) or \
                     (pos['type'] == 'SELL' and high[bi] >= pos['sl'])
            timeout= (i - pos['entry_i']) >= HOLD_BARS

            if tp_hit or sl_hit or timeout:
                exit_price = (pos['tp'] if tp_hit else
                              pos['sl'] if sl_hit else c)
                pnl = (exit_price - pos['entry']) if pos['type'] == 'BUY' \
                      else (pos['entry'] - exit_price)
                pnl -= SPREAD
                reason = 'TP' if tp_hit else 'SL' if sl_hit else 'TIMEOUT'
                trades.append({
                    'entry_date': str(dates[pos['entry_bi']]),
                    'exit_date':  str(dates[bi]),
                    'type':       pos['type'],
                    'entry':      pos['entry'],
                    'exit':       exit_price,
                    'pnl':        pnl,
                    'reason':     reason,
                })
                pos = None

        if pos is None and pred_str in ('BUY', 'SELL'):
            entry = c + SPREAD if pred_str == 'BUY' else c - SPREAD
            tp    = (entry + a * tp_mult) if pred_str == 'BUY' else (entry - a * tp_mult)
            sl    = (entry - a * sl_mult) if pred_str == 'BUY' else (entry + a * sl_mult)
            pos = {'type': pred_str, 'entry': entry, 'tp': tp, 'sl': sl,
                   'entry_i': i, 'entry_bi': bi}

    if not trades:
        return {'trades': 0, 'win_rate': 0, 'pf': 0, 'net_pnl': 0, 'accuracy': 0}

    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gp   = sum(p for p in pnls if p > 0)
    gl   = sum(abs(p) for p in pnls if p < 0)
    return {
        'trades':    len(trades),
        'win_rate':  wins / len(trades),
        'pf':        gp / gl if gl > 0 else 0.0,
        'net_pnl':   sum(pnls),
        'gross_profit': gp,
        'gross_loss':   -gl,
        'trade_list':   trades,
    }


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tp',     type=float, default=1.5)
    p.add_argument('--sl',     type=float, default=1.0)
    p.add_argument('--cot',    action='store_true', help='CoTモードで推論')
    p.add_argument('--adapter', type=str, default='',
                   help='アダプターディレクトリを明示指定（省略時は自動選択）')
    args = p.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    update_progress({'phase': 'backtest', 'message': 'v2バックテスト開始'})

    # アダプター選択
    if args.adapter:
        adapter_dir = Path(args.adapter)
    elif ADAPTER_DIR_V2.exists():
        adapter_dir = ADAPTER_DIR_V2
    elif ADAPTER_DIR_V1.exists():
        adapter_dir = ADAPTER_DIR_V1
        print('  [INFO] v2アダプターなし → v1を使用')
    else:
        print('[ERROR] アダプターが見つかりません')
        sys.exit(1)

    # テストデータ選択
    if args.cot and TEST_JSONL_V2C.exists():
        test_path = TEST_JSONL_V2C
        use_cot   = True
    elif TEST_JSONL_V2.exists():
        test_path = TEST_JSONL_V2
        use_cot   = False
    else:
        print('[ERROR] v2テストデータが見つかりません。先にdataset_prep_v2.pyを実行してください')
        sys.exit(1)

    print(f'  テストデータ: {test_path}', flush=True)
    print(f'  CoTモード   : {use_cot}', flush=True)

    samples = []
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f'  サンプル数: {len(samples):,}', flush=True)

    # モデル読み込み
    model, tokenizer, device = load_model(adapter_dir)

    # 推論
    if use_cot:
        preds, confs = predict_cot(model, tokenizer, samples, device)
    else:
        preds, confs = predict_logit(model, tokenizer, samples, device)

    # 正答率
    true_labels = [s['label'] if s['label'] in LABEL_NAMES
                   else re.search(r'→\s*(BUY|SELL|HOLD)', s['label'], re.I).group(1).upper()
                   if re.search(r'→\s*(BUY|SELL|HOLD)', s['label'], re.I) else 'HOLD'
                   for s in samples]
    accuracy = sum(p == t for p, t in zip(preds, true_labels)) / len(preds)
    print(f'  分類精度: {accuracy:.4f} ({accuracy*100:.2f}%)', flush=True)

    # バックテスト用価格データ
    from features import load_data, add_indicators
    csv_files = sorted((WORKSPACE / 'data').glob('USDJPY_M1*.csv'))
    csv_path  = str(csv_files[-1]) if csv_files else str(DATA_PATH)
    df_te = load_data(csv_path, timeframe='H1')
    df_te = add_indicators(df_te)
    df_te.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df_te.dropna(inplace=True)
    test_start = df_te.index[-1] - __import__('datetime').timedelta(days=365)
    df_te = df_te[df_te.index >= test_start].copy()

    # バックテスト実行
    bt = run_backtest(preds, df_te, tp_mult=args.tp, sl_mult=args.sl)
    print(f'\n  バックテスト結果:', flush=True)
    print(f'    取引数  : {bt["trades"]}', flush=True)
    print(f'    勝率    : {bt["win_rate"]*100:.1f}%', flush=True)
    print(f'    PF      : {bt["pf"]:.3f}', flush=True)
    print(f'    純損益  : {bt["net_pnl"]:.4f}', flush=True)

    # progress.json に記録
    update_progress({
        'backtest_result_v2': {
            'trades':    bt['trades'],
            'win_rate':  bt['win_rate'],
            'pf':        bt['pf'],
            'net_pnl':   bt['net_pnl'],
            'accuracy':  accuracy,
            'cot_mode':  use_cot,
        }
    })

    print('\n  v2バックテスト完了', flush=True)


if __name__ == '__main__':
    main()
