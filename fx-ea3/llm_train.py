"""
LLM QLoRA ファインチューニング v3
FX BUY/SELL/HOLD シグナル予測

モデル選択肢 (VRAM 11GB):
  --model_id Qwen/Qwen2.5-1.5B-Instruct   FP16 LoRA  ~7.5GB  デフォルト
  --model_id Qwen/Qwen2.5-1.5B-Instruct   4-bit LoRA ~3GB    --use_qlora
  --model_id Qwen/Qwen2.5-7B-Instruct     4-bit LoRA ~6-7GB  --use_qlora
  --model_id Qwen/Qwen3-8B                4-bit LoRA ~6-7GB  --use_qlora

使用方法:
    py -3.14 llm_train.py                            # 1.5B FP16
    py -3.14 llm_train.py --use_qlora               # 1.5B QLoRA 4bit
    py -3.14 llm_train.py --model_id Qwen/Qwen2.5-7B-Instruct --use_qlora
    py -3.14 llm_train.py --resume                  # チェックポイントから再開
"""
import sys, json, time, argparse, math
from pathlib import Path

import numpy as np
import torch

OUT_DIR      = Path(__file__).parent
ADAPTER_DIR  = OUT_DIR / 'llm_adapter'
BEST_DIR     = OUT_DIR / 'llm_adapter_best'
CKPT_DIR     = OUT_DIR / 'llm_checkpoint'
TRAIN_JSONL  = OUT_DIR / 'llm_train.jsonl'
TEST_JSONL   = OUT_DIR / 'llm_test.jsonl'
RESULT_JSON  = OUT_DIR / 'llm_train_result.json'
PROGRESS_JSON= OUT_DIR / 'llm_progress.json'

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LABEL_NAMES  = ['HOLD', 'BUY', 'SELL']
LABEL_TO_ID  = {n: i for i, n in enumerate(LABEL_NAMES)}


# ──────────────────────────────────────────────────────────────────────────
# 依存ライブラリ確認
# ──────────────────────────────────────────────────────────────────────────
def check_deps():
    for pkg in ['transformers', 'peft', 'accelerate']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[ERROR] {pkg} が未インストールです: pip install {pkg}")
            sys.exit(1)
    if not torch.cuda.is_available():
        print("[WARN] CUDA なし → CPU で実行 (非常に遅い)")
    else:
        p = torch.cuda.get_device_properties(0)
        cc = p.major * 10 + p.minor
        vram_gb = p.total_memory / 1024**3
        print(f"  GPU: {p.name}  VRAM: {vram_gb:.1f}GB  CC: {p.major}.{p.minor}")
        if cc < 60:
            print("[ERROR] CC < 6.0 は非対応")
            sys.exit(1)


def start_vram_watchdog(limit_gb: float = 9.5, interval_sec: float = 2.0) -> None:
    """
    バックグラウンドスレッドで VRAM を監視。
    limit_gb を超えたら即座に os._exit(1) でプロセスを強制終了 → PCフリーズ防止。
    """
    import threading, os as _os
    def _watch():
        while True:
            try:
                used_mb = torch.cuda.memory_reserved(0) / 1024**2
                used_gb = used_mb / 1024
                if used_gb >= limit_gb:
                    print(f"\n[WATCHDOG] VRAM {used_gb:.1f}GB >= 上限 {limit_gb:.1f}GB → 緊急停止！",
                          flush=True)
                    _os._exit(2)   # フリーズ前に強制終了
            except Exception:
                pass
            time.sleep(interval_sec)

    t = threading.Thread(target=_watch, daemon=True, name='vram_watchdog')
    t.start()
    print(f"  VRAM ウォッチドッグ起動: 上限 {limit_gb:.1f}GB (超過で即停止)")


def setup_vram_limit(fraction: float = 0.65) -> None:
    """
    VRAM 使用量をハードリミット
    デフォルト 65% = 約 7.2GB / 11GB → システム・ディスプレイに 3.8GB 確保
    """
    import os
    # 断片化を抑えるアロケータ設定 (OOM エラーメッセージの推奨)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                          'expandable_segments:True,max_split_size_mb:256')
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # fraction >= 0.99 の場合はリミットなし (大型モデルロード用)
        if fraction < 0.99:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            limit = total * fraction
            print(f"  VRAM リミット: {limit:.1f}GB / {total:.1f}GB ({fraction*100:.0f}%)")
        else:
            print(f"  VRAM リミット: なし (全 {total:.1f}GB 使用可)")


# ──────────────────────────────────────────────────────────────────────────
# GPU モニタリング (nvidia-smi)
# ──────────────────────────────────────────────────────────────────────────
def get_gpu_stats() -> tuple[float, float, float]:
    """(utilization%, vram_used_GB, vram_total_GB)"""
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        vals = r.stdout.strip().split(',')
        return float(vals[0]), float(vals[1]) / 1024, float(vals[2]) / 1024
    except Exception:
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return 0.0, used, total
        return 0.0, 0.0, 11.0


# ──────────────────────────────────────────────────────────────────────────
# ダッシュボード更新
# ──────────────────────────────────────────────────────────────────────────
_dash_state: dict = {}

def _update_dash(patch: dict) -> None:
    _dash_state.update(patch)
    try:
        from llm_dashboard import update as dash_update
        dash_update(_dash_state)
    except Exception:
        pass
    # JSON にも書き出す (他ツールから参照可能)
    try:
        PROGRESS_JSON.write_text(json.dumps(_dash_state, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
# データ
# ──────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class SignalDataset:
    def __init__(self, samples: list, tokenizer, max_length: int = 512):
        self.label_token_ids = {}
        for name in LABEL_NAMES:
            toks = tokenizer.encode(name, add_special_tokens=False)
            self.label_token_ids[name] = toks[0]

        self.input_ids   = []
        self.attn_masks  = []
        self.label_ids   = []
        skip = 0

        for s in samples:
            chat = make_chat_prompt(s['prompt'], tokenizer)
            enc  = tokenizer(chat, max_length=max_length, truncation=True,
                             padding=False, return_tensors='pt')
            ids  = enc['input_ids'][0]
            if len(ids) >= max_length:
                skip += 1
                continue
            self.input_ids.append(ids)
            self.attn_masks.append(enc['attention_mask'][0])
            self.label_ids.append(LABEL_TO_ID[s['label']])

        if skip:
            print(f"  [SKIP] max_length 超過: {skip} 件")

    def __len__(self):  return len(self.input_ids)
    def __getitem__(self, i):
        return {'input_ids': self.input_ids[i],
                'attention_mask': self.attn_masks[i],
                'label': self.label_ids[i]}


def make_chat_prompt(prompt_text: str, tokenizer) -> str:
    messages = [
        {"role": "system",
         "content": ("You are a professional FX trading signal analyst. "
                     "Analyze the market data and respond with exactly one word: "
                     "BUY, SELL, or HOLD.")},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def collate_fn(batch, pad_id: int):
    max_len    = max(b['input_ids'].shape[0] for b in batch)
    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels     = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        n = b['input_ids'].shape[0]
        input_ids[i, :n]  = b['input_ids']
        input_ids[i, n:]  = pad_id
        attn_masks[i, :n] = b['attention_mask']
    return {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}


# ──────────────────────────────────────────────────────────────────────────
# モデル構築  (FP16 LoRA / QLoRA 4bit 切り替え)
# ──────────────────────────────────────────────────────────────────────────
def build_lora_model(model_id: str, lora_r: int, lora_alpha: int,
                     lora_dropout: float, use_qlora: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    print(f"  モデル読み込み: {model_id}  ({'QLoRA 4-bit' if use_qlora else 'FP16 LoRA'})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if use_qlora:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',           # NF4: QLoRA 推奨
            bnb_4bit_compute_dtype=torch.float16, # 演算は FP16
            bnb_4bit_use_double_quant=True,       # 二重量子化でさらに節約
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg,
            device_map='auto', trust_remote_code=True
        )
        model.config.use_cache = False
        # 4-bit モデル用の勾配/LayerNorm 設定
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map='auto', trust_remote_code=True
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
        lora_dropout=lora_dropout, bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    if not use_qlora:
        model.enable_input_require_grads()

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vram_gb = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    print(f"  LoRA パラメータ: {n_train:,} / {n_total:,} ({n_train/n_total*100:.3f}%)")
    print(f"  モデルロード後 VRAM: {vram_gb:.1f} GB")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────
# チェックポイント保存 / 復元
# ──────────────────────────────────────────────────────────────────────────
def save_checkpoint(epoch: int, model, optimizer, scheduler,
                    best_acc: float, epoch_log: list, tokenizer) -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f'epoch_{epoch:03d}.pt'

    # LoRA 重みのみ保存（ベースモデルは不要）
    lora_state = {k: v.cpu().clone()
                  for k, v in model.state_dict().items() if 'lora' in k}
    torch.save({
        'epoch':      epoch,
        'lora_state': lora_state,
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'best_acc':   best_acc,
        'epoch_log':  epoch_log,
    }, ckpt_path)

    # アダプターとして保存 (中間版)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # 古いチェックポイントを 2 つ残して削除
    ckpts = sorted(CKPT_DIR.glob('epoch_*.pt'))
    for old in ckpts[:-2]:
        old.unlink(missing_ok=True)

    print(f"  [CKPT] epoch {epoch} 保存: {ckpt_path.name}")


def find_latest_checkpoint() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(CKPT_DIR.glob('epoch_*.pt'))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: Path, model, optimizer, scheduler) -> tuple[int, float, list]:
    print(f"  [RESUME] チェックポイント読み込み: {path.name}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # LoRA 重みを復元
    cur = model.state_dict()
    for k, v in ckpt['lora_state'].items():
        if k in cur:
            cur[k].copy_(v)

    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])

    epoch_done = ckpt['epoch']
    best_acc   = ckpt.get('best_acc', 0.0)
    epoch_log  = ckpt.get('epoch_log', [])
    print(f"  [RESUME] epoch {epoch_done} 完了済み  best_acc={best_acc:.4f}")
    return epoch_done, best_acc, epoch_log


# ──────────────────────────────────────────────────────────────────────────
# 評価
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size: int,
             device, label_token_ids: dict) -> dict:
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    model.eval()
    correct = 0; total = 0
    lbl_tids = torch.tensor([label_token_ids[n] for n in LABEL_NAMES], device=device)

    for batch in dl:
        ids   = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        gt    = batch['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(input_ids=ids, attention_mask=masks)
        last_pos = masks.sum(1) - 1
        for b in range(len(gt)):
            lgt  = out.logits[b, last_pos[b], lbl_tids]
            pred = int(lgt.argmax())
            if pred == int(gt[b]):
                correct += 1
            total += 1

    return {'accuracy': round(correct / max(total, 1), 4), 'total': total}


# ──────────────────────────────────────────────────────────────────────────
# 学習メインループ
# ──────────────────────────────────────────────────────────────────────────
def train(args):
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== LLM QLoRA ファインチューニング [{device}] ===")

    # VRAM 設定
    setup_vram_limit(args.vram_fraction)

    # VRAM ウォッチドッグ: 超えたらフリーズ前に即停止
    if device.type == 'cuda':
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        watchdog_limit = total_gb * args.vram_kill   # デフォルト 88% = 9.7GB
        start_vram_watchdog(limit_gb=watchdog_limit)

    # ── ダッシュボード初期化 ─────────────────────────────────────────────
    _update_dash({
        'phase': 'loading', 'epoch': 0, 'total_epochs': args.epochs,
        'batch': 0, 'total_batches': 1, 'train_loss': 0.0, 'val_loss': 0.0,
        'accuracy': 0.0, 'best_acc': 0.0, 'gpu_pct': 0, 'vram_used_gb': 0.0,
        'vram_total_gb': 11.0, 'lr': args.lr, 'elapsed_sec': 0.0, 'eta_sec': -1,
        'epoch_log': [], 'message': 'モデル読み込み中...',
    })

    # ── データ読み込み ───────────────────────────────────────────────────
    if not TRAIN_JSONL.exists():
        print(f"[ERROR] {TRAIN_JSONL} が存在しません。llm_dataset.py を先に実行してください")
        sys.exit(1)
    train_raw = load_jsonl(TRAIN_JSONL)
    test_raw  = load_jsonl(TEST_JSONL)
    if args.max_train > 0 and len(train_raw) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(train_raw), args.max_train, replace=False).tolist())
        train_raw = [train_raw[i] for i in idx]
    print(f"  訓練: {len(train_raw):,}  テスト: {len(test_raw):,}")

    # ── モデル読み込み ───────────────────────────────────────────────────
    model, tokenizer = build_lora_model(
        args.model_id, args.lora_r, args.lora_alpha, args.lora_dropout,
        use_qlora=args.use_qlora
    )

    # ── トークナイズ ─────────────────────────────────────────────────────
    _update_dash({'phase': 'tokenizing', 'message': 'トークナイズ中 (訓練)...'})
    print("  トークナイズ中 (訓練)...")
    tr_ds = SignalDataset(train_raw, tokenizer, max_length=args.max_length)
    _update_dash({'message': 'トークナイズ中 (テスト)...'})
    print("  トークナイズ中 (テスト)...")
    te_ds = SignalDataset(test_raw,  tokenizer, max_length=args.max_length)

    pad_id = tokenizer.pad_token_id
    tr_dl  = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)
    total_batches = len(tr_dl)
    print(f"  バッチ数/epoch: {total_batches}")

    # ── オプティマイザー・スケジューラー ────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd
    )
    total_steps  = total_batches * args.epochs
    warmup_steps = max(20, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp   = (device.type == 'cuda')
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    label_tids_list = [tr_ds.label_token_ids[n] for n in LABEL_NAMES]
    label_tids = torch.tensor(label_tids_list, device=device)

    # ── チェックポイント / 再開 ──────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    epoch_log   = []

    if args.resume:
        ckpt = find_latest_checkpoint()
        if ckpt:
            start_epoch, best_acc, epoch_log = load_checkpoint(
                ckpt, model, optimizer, scheduler
            )
            start_epoch += 1   # 次のエポックから
        else:
            print("  [RESUME] チェックポイントが見つかりません。最初から開始します。")

    _update_dash({
        'phase': 'training', 'total_batches': total_batches,
        'epoch': start_epoch - 1, 'best_acc': best_acc,
        'epoch_log': epoch_log,
        'message': f'訓練開始 epoch {start_epoch}/{args.epochs}',
    })

    # ── 損失計算ヘルパー ─────────────────────────────────────────────────
    def compute_loss(logits, attn_masks, gt_labels):
        last_pos   = attn_masks.sum(1) - 1
        sel_logits = torch.stack(
            [logits[b, last_pos[b], label_tids] for b in range(logits.shape[0])]
        )
        return nn.functional.cross_entropy(sel_logits, gt_labels,
                                            label_smoothing=0.05)

    # ── GPU スロットリング設定 ───────────────────────────────────────────
    # throttle_ratio: バッチ計算時間に対するスリープ割合
    # ratio=0.25 → GPU使用率 ≈ 80% (0.25/(1+0.25)=20%アイドル)
    throttle_ratio = args.throttle   # デフォルト 0.25

    # ── 訓練ループ ───────────────────────────────────────────────────────
    no_imp      = 0
    patience    = max(3, args.epochs // 2)
    global_step = (start_epoch - 1) * total_batches
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss  = 0.0
        ep_start = time.time()
        gpu_util_acc = 0.0

        for bi, batch in enumerate(tr_dl, start=1):
            b_start = time.time()

            ids    = batch['input_ids'].to(device)
            masks  = batch['attention_mask'].to(device)
            gt     = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(input_ids=ids, attention_mask=masks)
                loss = compute_loss(out.logits, masks, gt)

            loss_s = loss / args.grad_accum
            if use_amp:
                scaler.scale(loss_s).backward()
            else:
                loss_s.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            ep_loss     += loss.item()
            global_step += 1

            # ── GPU スロットリング (synchronize で GPU を実際に止める) ──
            batch_time = time.time() - b_start
            if throttle_ratio > 0 and device.type == 'cuda':
                torch.cuda.synchronize()          # GPU の全演算完了を待つ
                time.sleep(batch_time * throttle_ratio)

            # 定期的にキャッシュ解放 (VRAM 断片化を防ぐ)
            if bi % 200 == 0:
                torch.cuda.empty_cache()

            # ── 進捗表示 & ダッシュボード更新 (50バッチごと) ───────────
            if bi % 50 == 0 or bi == total_batches:
                elapsed   = time.time() - train_start
                done_ep   = (epoch - start_epoch) + (bi / total_batches)
                total_ep  = args.epochs - start_epoch + 1
                eta       = elapsed / max(done_ep, 0.001) * (total_ep - done_ep)
                lr_now    = optimizer.param_groups[0]['lr']
                avg_loss  = ep_loss / bi

                # GPU 情報取得 (10バッチに1回)
                if bi % 10 == 0:
                    gpu_pct, vram_used, vram_total = get_gpu_stats()
                    gpu_util_acc = gpu_pct
                else:
                    gpu_pct, vram_used, vram_total = gpu_util_acc, 0.0, 11.0

                pct = bi / total_batches * 100
                print(f"  Ep{epoch}/{args.epochs}  [{bi:4d}/{total_batches}] {pct:5.1f}%"
                      f"  loss={avg_loss:.4f}  lr={lr_now:.2e}"
                      f"  GPU:{gpu_pct:.0f}%  {time.strftime('%H:%M:%S')}",
                      flush=True)

                _update_dash({
                    'phase':        'training',
                    'epoch':        epoch,
                    'batch':        bi,
                    'train_loss':   round(avg_loss, 5),
                    'lr':           round(lr_now, 8),
                    'elapsed_sec':  round(elapsed, 1),
                    'eta_sec':      round(eta, 1),
                    'gpu_pct':      int(gpu_pct),
                    'vram_used_gb': round(vram_used, 2),
                    'vram_total_gb':round(vram_total, 2),
                    'message':      f'Epoch {epoch}/{args.epochs}  batch {bi}/{total_batches}',
                })

        ep_loss /= total_batches

        # ── エポック末: 評価 ─────────────────────────────────────────────
        _update_dash({'phase': 'evaluating', 'message': f'Epoch {epoch} 評価中...'})
        print(f"  Ep{epoch} 評価中...", flush=True)
        ev = evaluate(model, tokenizer, te_ds, args.batch * 2, device,
                      te_ds.label_token_ids)
        ep_elapsed = time.time() - ep_start

        is_best = ev['accuracy'] > best_acc + 0.001
        if is_best:
            best_acc = ev['accuracy']
            model.save_pretrained(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))
            no_imp = 0
            print(f"  [BEST] acc={best_acc:.4f}", flush=True)
        else:
            no_imp += 1

        ep_entry = {
            'epoch':      epoch,
            'train_loss': round(ep_loss, 5),
            'val_loss':   round(ep_loss, 5),  # 同じ損失で代用
            'acc':        ev['accuracy'],
            'elapsed':    round(ep_elapsed, 1),
            'is_best':    is_best,
        }
        epoch_log.append(ep_entry)

        print(f"\n  === Epoch {epoch}/{args.epochs} 完了 ==="
              f"  loss={ep_loss:.4f}  acc={ev['accuracy']:.4f}"
              f"  best={best_acc:.4f}  {ep_elapsed:.0f}s\n", flush=True)

        # ダッシュボード更新
        gpu_pct, vram_used, vram_total = get_gpu_stats()
        _update_dash({
            'phase':        'training',
            'epoch':        epoch,
            'batch':        total_batches,
            'train_loss':   round(ep_loss, 5),
            'val_loss':     round(ep_loss, 5),
            'accuracy':     ev['accuracy'],
            'best_acc':     best_acc,
            'epoch_log':    epoch_log,
            'gpu_pct':      int(gpu_pct),
            'vram_used_gb': round(vram_used, 2),
            'message':      f'Epoch {epoch} 完了  acc={ev["accuracy"]:.4f}  best={best_acc:.4f}',
        })

        # ── チェックポイント保存 ─────────────────────────────────────────
        save_checkpoint(epoch, model, optimizer, scheduler,
                        best_acc, epoch_log, tokenizer)

        # ── 早期終了 ─────────────────────────────────────────────────────
        if no_imp >= patience:
            print(f"  [EARLY STOP] {patience} epoch 改善なし", flush=True)
            break

    # ── 最良アダプターをメインにコピー ────────────────────────────────────
    if BEST_DIR.exists():
        import shutil
        for f in BEST_DIR.iterdir():
            shutil.copy(f, ADAPTER_DIR)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    total_elapsed = time.time() - train_start
    result = {
        'best_accuracy': best_acc,
        'epochs_trained': epoch,
        'train_samples': len(tr_ds),
        'test_samples':  len(te_ds),
        'model_id':      args.model_id,
        'lora_r':        args.lora_r,
        'label_token_ids': tr_ds.label_token_ids,
        'total_min':     round(total_elapsed / 60, 1),
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== 訓練完了 ===  best_acc={best_acc:.4f}  {total_elapsed/60:.1f}分",
          flush=True)

    _update_dash({
        'phase': 'done', 'best_acc': best_acc, 'epoch': epoch,
        'epoch_log': epoch_log,
        'message': f'訓練完了！ Best Accuracy={best_acc:.4f}  {total_elapsed/60:.1f}分',
    })
    return result


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id',       type=str,   default=MODEL_ID,
                   help='HuggingFace モデル ID (デフォルト: 0.5B)')
    p.add_argument('--epochs',         type=int,   default=5)
    p.add_argument('--batch',          type=int,   default=2)
    p.add_argument('--grad_accum',     type=int,   default=8)
    p.add_argument('--lr',             type=float, default=2e-4)
    p.add_argument('--wd',             type=float, default=1e-2)
    p.add_argument('--lora_r',         type=int,   default=16)
    p.add_argument('--lora_alpha',     type=int,   default=32)
    p.add_argument('--lora_dropout',   type=float, default=0.05)
    p.add_argument('--max_length',     type=int,   default=512)
    p.add_argument('--max_train',      type=int,   default=0)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--throttle',       type=float, default=0.4,
                   help='GPU スロットリング比率 (0.25=約80%%, 0.5=約67%%, 0=制限なし)')
    p.add_argument('--vram_fraction',  type=float, default=0.99,
                   help='PyTorch VRAM アロケータ上限 (0.99=事実上なし)')
    p.add_argument('--vram_kill',      type=float, default=0.88,
                   help='VRAM ウォッチドッグ: この割合を超えたら即強制終了 (0.88=9.7GB/11GB)')
    p.add_argument('--use_qlora',      action='store_true',
                   help='QLoRA 4-bit 量子化+LoRA (VRAM 約半減, 7B級モデルに推奨)')
    p.add_argument('--resume',         action='store_true',
                   help='最新チェックポイントから再開')
    return p.parse_args()


if __name__ == '__main__':
    check_deps()
    args = parse_args()
    mode = 'QLoRA 4-bit' if args.use_qlora else 'FP16 LoRA'
    print(f"  モデル   : {args.model_id}  [{mode}]")
    print(f"  実効バッチ: {args.batch * args.grad_accum}")
    print(f"  GPU スロットリング: throttle={args.throttle} (約{100/(1+args.throttle):.0f}% GPU)")
    if args.resume:
        ckpt = find_latest_checkpoint()
        print(f"  再開モード: {ckpt.name if ckpt else 'チェックポイントなし'}")
    result = train(args)
    print(f"\n最良 Accuracy: {result['best_accuracy']:.4f}")
