"""
LLM Unsloth QLoRA ファインチューニング — H100 80GB 最適化版
FX BUY/SELL/HOLD シグナル予測

H100 最適化ポイント:
  - batch=8, grad_accum=8  → 実効バッチ=64
  - lora_r=64, lora_alpha=128 (高精度 LoRA)
  - BF16 ネイティブ (H100 Tensor Core 最大活用)
  - Flash Attention 2 (Unsloth 組み込み)
  - max_length=1024 (80GB VRAM で余裕)
  - eval_batch=16 (高速評価)
  - DataLoader num_workers=4
  - VRAM ウォッチドッグ上限 76GB (全体の95%)
  - torch.compile (reduce-overhead)
  - 3000件制限なし (max_train=0)

使用方法:
    python /workspace/src/train_h100.py
    python /workspace/src/train_h100.py --model_id Qwen/Qwen3-14B
    python /workspace/src/train_h100.py --resume
"""
import sys, os, json, time, math, argparse, threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/workspace/ai_ea')
sys.path.insert(0, '/workspace/src')

WORKSPACE    = Path('/workspace')
AI_EA_DIR    = WORKSPACE / 'ai_ea'
ADAPTER_DIR  = WORKSPACE / 'output' / 'llm_adapter'
BEST_DIR     = WORKSPACE / 'output' / 'llm_adapter_best'
CKPT_DIR     = WORKSPACE / 'output' / 'llm_checkpoint'
TRAIN_JSONL  = WORKSPACE / 'output' / 'llm_train.jsonl'
TEST_JSONL   = WORKSPACE / 'output' / 'llm_test.jsonl'
RESULT_JSON  = WORKSPACE / 'output' / 'llm_train_result.json'
PROGRESS_JSON= WORKSPACE / 'progress.json'

DEFAULT_MODEL = "Qwen/Qwen3-8B"
LABEL_NAMES   = ['HOLD', 'BUY', 'SELL']
LABEL_TO_ID   = {n: i for i, n in enumerate(LABEL_NAMES)}


# ──────────────────────────────────────────────────────────────────────────────
# 進捗 JSON 更新 (ダッシュボードが読む)
# ──────────────────────────────────────────────────────────────────────────────
_dash_state: dict = {}

def update_progress(patch: dict) -> None:
    _dash_state.update(patch)
    try:
        PROGRESS_JSON.write_text(
            json.dumps(_dash_state, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# VRAM ウォッチドッグ (H100: 76GB 上限)
# ──────────────────────────────────────────────────────────────────────────────
_watchdog_paused = False

def start_vram_watchdog(limit_gb: float, interval: float = 2.0) -> None:
    def _watch():
        while True:
            try:
                if not _watchdog_paused:
                    used = torch.cuda.memory_allocated(0) / 1024**3
                    if used >= limit_gb:
                        msg = f"[WATCHDOG] VRAM {used:.1f}GB >= {limit_gb:.1f}GB → 緊急停止"
                        print(f"\n{msg}", flush=True)
                        update_progress({'phase': 'error', 'error': msg})
                        os._exit(2)
            except Exception:
                pass
            time.sleep(interval)
    t = threading.Thread(target=_watch, daemon=True, name='vram_watchdog')
    t.start()
    print(f"  VRAM ウォッチドッグ: 上限 {limit_gb:.1f}GB")


# ──────────────────────────────────────────────────────────────────────────────
# GPU 統計
# ──────────────────────────────────────────────────────────────────────────────
def get_gpu_stats() -> tuple:
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3)
        vals = r.stdout.strip().split(',')
        return float(vals[0]), float(vals[1]) / 1024, float(vals[2]) / 1024
    except Exception:
        used  = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0, used, total


# ──────────────────────────────────────────────────────────────────────────────
# データ
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def make_chat_prompt(prompt_text: str, tokenizer) -> str:
    messages = [
        {"role": "system",
         "content": ("You are a professional FX trading signal analyst. "
                     "Analyze the market data and respond with exactly one word: "
                     "BUY, SELL, or HOLD.")},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


class SignalDataset:
    def __init__(self, samples: list, tokenizer, max_length: int = 1024):
        self.label_token_ids = {}
        for name in LABEL_NAMES:
            toks = tokenizer.encode(name, add_special_tokens=False)
            self.label_token_ids[name] = toks[0]

        self.input_ids  = []
        self.attn_masks = []
        self.label_ids  = []
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

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, i):
        return {'input_ids': self.input_ids[i],
                'attention_mask': self.attn_masks[i],
                'label': self.label_ids[i]}


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


# ──────────────────────────────────────────────────────────────────────────────
# チェックポイント
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler,
                    best_acc, epoch_log, tokenizer, step=None):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f'epoch_{epoch:03d}' if step is None
           else f'epoch_{epoch:03d}_step_{step:06d}')
    ckpt_path = CKPT_DIR / f'{tag}.pt'
    lora_state = {k: v.cpu().clone()
                  for k, v in model.state_dict().items() if 'lora' in k}
    torch.save({
        'epoch': epoch, 'step': step, 'lora_state': lora_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc, 'epoch_log': epoch_log,
    }, ckpt_path)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    ckpts = sorted(CKPT_DIR.glob('*.pt'))
    for old in ckpts[:-3]:
        old.unlink(missing_ok=True)
    label = f"epoch {epoch}" if step is None else f"epoch {epoch} step {step}"
    print(f"  [CKPT] {label} 保存: {ckpt_path.name}", flush=True)


def find_latest_checkpoint():
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(CKPT_DIR.glob('*.pt'))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path, model, optimizer, scheduler):
    print(f"  [RESUME] {path.name}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cur = model.state_dict()
    for k, v in ckpt['lora_state'].items():
        if k in cur:
            cur[k].copy_(v)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt.get('best_acc', 0.0), ckpt.get('epoch_log', [])


# ──────────────────────────────────────────────────────────────────────────────
# 評価 (H100: eval_batch=16 で高速化)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size, device, label_token_ids):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                    num_workers=0, pin_memory=True)
    model.eval()
    correct = total = 0
    total_loss = 0.0
    lbl_tids = torch.tensor([label_token_ids[n] for n in LABEL_NAMES],
                             device=device)

    for batch in dl:
        ids   = batch['input_ids'].to(device, non_blocking=True)
        masks = batch['attention_mask'].to(device, non_blocking=True)
        gt    = batch['labels'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=masks)
        last_pos = masks.sum(1) - 1
        sel = torch.stack([out.logits[b, last_pos[b], lbl_tids]
                           for b in range(len(gt))])
        total_loss += nn.functional.cross_entropy(
            sel, gt, label_smoothing=0.05).item()
        correct += int((sel.argmax(dim=1) == gt).sum())
        total   += len(gt)

    model.train()
    return {
        'accuracy': round(correct / max(total, 1), 4),
        'val_loss': round(total_loss / max(len(dl), 1), 5),
        'total': total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# メイン訓練
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from unsloth import FastLanguageModel

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                          'expandable_segments:True,max_split_size_mb:512')

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p          = torch.cuda.get_device_properties(0)
    total_vram = p.total_memory / 1024**3
    print(f"\n=== H100 Unsloth QLoRA ファインチューニング ===")
    print(f"  GPU: {p.name}  VRAM: {total_vram:.1f}GB  CC: {p.major}.{p.minor}")

    watchdog_gb = total_vram * args.vram_kill
    start_vram_watchdog(limit_gb=watchdog_gb)

    update_progress({
        'phase': 'loading', 'epoch': 0, 'total_epochs': args.epochs,
        'batch': 0, 'total_batches': 1, 'train_loss': 0.0,
        'accuracy': 0.0, 'best_acc': 0.0, 'gpu_pct': 0,
        'vram_used_gb': 0.0, 'vram_total_gb': total_vram,
        'lr': args.lr, 'elapsed_sec': 0.0, 'eta_sec': -1,
        'epoch_log': [], 'batch_log': [],
        'message': f'H100 Unsloth モデル読み込み中... ({args.model_id})',
    })

    # ── データ読み込み ────────────────────────────────────────────────────────
    if not TRAIN_JSONL.exists():
        print(f"[ERROR] {TRAIN_JSONL} がありません。pipeline.py を実行してください")
        sys.exit(1)
    train_raw = load_jsonl(TRAIN_JSONL)
    test_raw  = load_jsonl(TEST_JSONL)
    # H100: 件数制限なし (max_train=0 がデフォルト)
    if args.max_train > 0 and len(train_raw) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(train_raw), args.max_train, replace=False).tolist())
        train_raw = [train_raw[i] for i in idx]
    print(f"  訓練: {len(train_raw):,}  テスト: {len(test_raw):,}")

    update_progress({'message': f'データ読み込み完了。訓練:{len(train_raw):,} テスト:{len(test_raw):,}'})

    # ── Unsloth モデルロード (BF16, Flash Attention 2) ────────────────────────
    print(f"  Unsloth モデル: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = args.model_id,
        max_seq_length    = args.max_length,
        load_in_4bit      = True,
        load_in_8bit      = False,
        dtype             = torch.bfloat16,   # H100 ネイティブ BF16
        trust_remote_code = True,
    )
    vram_load = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  ロード後 VRAM: {vram_load:.1f}GB")

    # ── LoRA 設定 (H100: rank=64 で精度向上) ─────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj'],
        bias           = 'none',
        use_gradient_checkpointing = 'unsloth',
        random_state   = args.seed,
        max_seq_length = args.max_length,
    )
    n_total = sum(pp.numel() for pp in model.parameters())
    n_train = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  LoRA パラメータ: {n_train:,} / {n_total:,} ({n_train/n_total*100:.3f}%)")

    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            print("  torch.compile: 有効 (max-autotune)")
        except Exception as e:
            print(f"  torch.compile: スキップ ({e})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ── トークナイズ ──────────────────────────────────────────────────────────
    update_progress({'phase': 'tokenizing', 'message': 'トークナイズ中...'})
    print("  トークナイズ中 (訓練)...")
    tr_ds = SignalDataset(train_raw, tokenizer, args.max_length)
    print("  トークナイズ中 (テスト)...")
    te_ds = SignalDataset(test_raw,  tokenizer, args.max_length)
    print(f"  トークナイズ完了: 訓練 {len(tr_ds):,}  テスト {len(te_ds):,}")

    pad_id   = tokenizer.pad_token_id
    tr_dl    = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_id),
                          num_workers=4, pin_memory=True, prefetch_factor=2)
    total_batches = len(tr_dl)
    print(f"  バッチ数/epoch: {total_batches}")

    # ── オプティマイザ ────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [pp for pp in model.parameters() if pp.requires_grad],
            lr=args.lr, weight_decay=args.wd)
        print("  オプティマイザ: AdamW 8-bit")
    except Exception:
        optimizer = torch.optim.AdamW(
            [pp for pp in model.parameters() if pp.requires_grad],
            lr=args.lr, weight_decay=args.wd, fused=True)
        print("  オプティマイザ: AdamW (fused)")

    total_steps  = total_batches * args.epochs
    warmup_steps = max(50, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # H100 BF16 → GradScaler は不要 (BF16 は動的範囲が広い)
    use_scaler = False

    label_tids = torch.tensor(
        [tr_ds.label_token_ids[n] for n in LABEL_NAMES], device=device)

    # ── チェックポイント再開 ──────────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    epoch_log   = []
    if args.resume:
        ckpt = find_latest_checkpoint()
        if ckpt:
            start_epoch, best_acc, epoch_log = load_checkpoint(
                ckpt, model, optimizer, scheduler)
            start_epoch += 1
        else:
            print("  [RESUME] チェックポイントなし → 最初から開始")

    # eval サブセット
    rng_eval = np.random.default_rng(args.seed + 99)
    eval_idx = sorted(rng_eval.choice(
        len(te_ds), min(args.eval_samples, len(te_ds)), replace=False).tolist())
    eval_sub = [te_ds[i] for i in eval_idx]
    eval_dl  = DataLoader(eval_sub, batch_size=args.eval_batch, shuffle=False,
                          collate_fn=lambda b: collate_fn(b, pad_id),
                          num_workers=0, pin_memory=True)

    update_progress({
        'phase': 'training', 'total_batches': total_batches,
        'total_epochs': args.epochs,
        'epoch': start_epoch - 1, 'best_acc': best_acc,
        'epoch_log': epoch_log, 'batch_log': [],
        'message': f'H100 訓練開始 epoch {start_epoch}/{args.epochs}'
             f'  実効バッチ: {args.batch}×{args.grad_accum}={args.batch*args.grad_accum}',
    })

    def compute_loss(logits, attn_masks, gt_labels):
        last_pos   = attn_masks.sum(1) - 1
        sel_logits = torch.stack(
            [logits[b, last_pos[b], label_tids] for b in range(logits.shape[0])])
        return nn.functional.cross_entropy(sel_logits, gt_labels, label_smoothing=0.05)

    no_imp_acc   = 0
    no_imp_val   = 0
    patience_acc = max(3, args.epochs // 2)
    patience_ov  = 3
    global_step  = (start_epoch - 1) * total_batches
    train_start  = time.time()
    batch_log    = []

    # ── 訓練ループ ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss  = 0.0
        ep_start = time.time()

        for bi, batch in enumerate(tr_dl, start=1):
            ids   = batch['input_ids'].to(device, non_blocking=True)
            masks = batch['attention_mask'].to(device, non_blocking=True)
            gt    = batch['labels'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out  = model(input_ids=ids, attention_mask=masks)
                loss = compute_loss(out.logits, masks, gt)

            (loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [pp for pp in model.parameters() if pp.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            ep_loss     += loss.item()
            global_step += 1

            if bi % 200 == 0:
                torch.cuda.empty_cache()

            if args.save_steps > 0 and bi % args.save_steps == 0:
                save_checkpoint(epoch, model, optimizer, scheduler,
                                best_acc, epoch_log, tokenizer, step=global_step)

            # ── 定期 eval ─────────────────────────────────────────────────────
            cur_val_loss = None
            cur_acc      = None
            if bi % args.eval_steps == 0 or bi == total_batches:
                global _watchdog_paused
                _watchdog_paused = True
                torch.cuda.empty_cache()
                model.eval()
                v_loss = 0.0; v_correct = 0; v_total = 0
                with torch.no_grad():
                    for vb in eval_dl:
                        vids   = vb['input_ids'].to(device, non_blocking=True)
                        vmasks = vb['attention_mask'].to(device, non_blocking=True)
                        vgt    = vb['labels'].to(device, non_blocking=True)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            vout = model(input_ids=vids, attention_mask=vmasks)
                        lp   = vmasks.sum(1) - 1
                        vsel = torch.stack([vout.logits[b, lp[b], label_tids].float()
                                            for b in range(len(vgt))])
                        del vout
                        v_loss    += nn.functional.cross_entropy(
                            vsel, vgt, label_smoothing=0.05).item()
                        v_correct += (vsel.argmax(dim=1) == vgt).sum().item()
                        v_total   += len(vgt)
                        del vids, vmasks, vgt, vsel
                cur_val_loss = round(v_loss   / max(len(eval_dl), 1), 4)
                cur_acc      = round(v_correct / max(v_total, 1) * 100, 2)
                model.train()
                torch.cuda.empty_cache()
                _watchdog_paused = False

            entry = {'step': global_step, 'train_loss': round(loss.item(), 4)}
            if cur_val_loss is not None:
                entry['val_loss'] = cur_val_loss
                entry['acc']      = cur_acc
            batch_log.append(entry)
            if len(batch_log) > 4000:
                batch_log = batch_log[-4000:]

            if bi % 50 == 0 or bi == total_batches:
                elapsed  = time.time() - train_start
                done_ep  = (epoch - start_epoch) + (bi / total_batches)
                total_ep = args.epochs - start_epoch + 1
                eta      = elapsed / max(done_ep, 0.001) * (total_ep - done_ep)
                lr_now   = optimizer.param_groups[0]['lr']
                avg_loss = ep_loss / bi
                gpu_pct, vram_used, vram_total = get_gpu_stats()
                pct = bi / total_batches * 100
                print(f"  Ep{epoch}/{args.epochs} [{bi:5d}/{total_batches}] "
                      f"{pct:5.1f}%  loss={avg_loss:.4f}  lr={lr_now:.2e}"
                      f"  GPU:{gpu_pct:.0f}%  VRAM:{vram_used:.1f}GB"
                      f"  {time.strftime('%H:%M:%S')}", flush=True)
                update_progress({
                    'phase': 'training', 'epoch': epoch, 'batch': bi,
                    'train_loss': round(avg_loss, 5), 'lr': round(lr_now, 8),
                    'elapsed_sec': round(elapsed, 1), 'eta_sec': round(eta, 1),
                    'gpu_pct': int(gpu_pct),
                    'vram_used_gb': round(vram_used, 2),
                    'vram_total_gb': round(vram_total, 2),
                    'batch_log': batch_log,
                    'message': (f'Epoch {epoch}/{args.epochs}  '
                                f'batch {bi}/{total_batches}  '
                                f'loss={avg_loss:.4f}'),
                })

        ep_loss /= total_batches
        save_checkpoint(epoch, model, optimizer, scheduler,
                        best_acc, epoch_log, tokenizer)

        print(f"  Ep{epoch} 評価中...", flush=True)
        torch.cuda.empty_cache()
        ev = evaluate(model, tokenizer, te_ds, args.eval_batch, device,
                      te_ds.label_token_ids)
        ep_elapsed = time.time() - ep_start
        val_loss   = ev['val_loss']
        is_best    = ev['accuracy'] > best_acc + 0.001

        if is_best:
            best_acc = ev['accuracy']
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))
            no_imp_acc = 0
            print(f"  [BEST] acc={best_acc:.4f}  val_loss={val_loss:.4f}", flush=True)
        else:
            no_imp_acc += 1

        if len(epoch_log) >= 1:
            prev_val = epoch_log[-1].get('val_loss', val_loss)
            if val_loss > prev_val * 1.02:
                no_imp_val += 1
                print(f"  [OVERFIT] val_loss 上昇 ({prev_val:.4f} → {val_loss:.4f}) "
                      f"連続 {no_imp_val} 回", flush=True)
            else:
                no_imp_val = 0

        ep_entry = {
            'epoch': epoch, 'train_loss': round(ep_loss, 5),
            'val_loss': round(val_loss, 5), 'acc': ev['accuracy'],
            'elapsed': round(ep_elapsed, 1), 'is_best': is_best,
            'step_end': global_step,
        }
        epoch_log.append(ep_entry)

        print(f"\n  === Epoch {epoch}/{args.epochs} 完了 ==="
              f"  train={ep_loss:.4f}  val={val_loss:.4f}"
              f"  acc={ev['accuracy']:.4f}  best={best_acc:.4f}"
              f"  {ep_elapsed:.0f}s\n", flush=True)

        gpu_pct, vram_used, vram_total = get_gpu_stats()
        update_progress({
            'phase': 'training', 'epoch': epoch, 'batch': total_batches,
            'train_loss': round(ep_loss, 5), 'val_loss': round(val_loss, 5),
            'accuracy': ev['accuracy'], 'best_acc': best_acc,
            'epoch_log': epoch_log, 'gpu_pct': int(gpu_pct),
            'vram_used_gb': round(vram_used, 2),
            'message': (f'Epoch {epoch} 完了  '
                        f'train={ep_loss:.4f}  val={val_loss:.4f}  '
                        f'acc={ev["accuracy"]:.4f}'),
        })

        stop_reason = None
        if no_imp_val >= patience_ov:
            stop_reason = f'過学習検知: val_loss が {patience_ov} epoch 連続悪化'
        elif no_imp_acc >= patience_acc:
            stop_reason = f'精度改善なし: {patience_acc} epoch 連続'
        if stop_reason:
            print(f"  [EARLY STOP] {stop_reason}", flush=True)
            break

    # 最良アダプターを確定保存
    if BEST_DIR.exists():
        import shutil
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        for f in BEST_DIR.iterdir():
            shutil.copy(f, ADAPTER_DIR)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    total_elapsed = time.time() - train_start
    result = {
        'best_accuracy': best_acc,
        'epochs_trained': epoch,
        'train_samples': len(tr_ds),
        'test_samples': len(te_ds),
        'model_id': args.model_id,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'batch': args.batch,
        'grad_accum': args.grad_accum,
        'effective_batch': args.batch * args.grad_accum,
        'max_length': args.max_length,
        'label_token_ids': tr_ds.label_token_ids,
        'total_min': round(total_elapsed / 60, 1),
        'mode': 'h100_unsloth_qlora_bf16',
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== 訓練完了 ===  best_acc={best_acc:.4f}  "
          f"{total_elapsed/60:.1f}分", flush=True)

    update_progress({
        'phase': 'done', 'best_acc': best_acc,
        'train_result': result,
        'message': f'訓練完了！ Best Acc={best_acc:.4f}  {total_elapsed/60:.1f}分',
    })
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='H100 Unsloth QLoRA Fine-tuning')
    p.add_argument('--model_id',     type=str,   default=DEFAULT_MODEL)
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--batch',        type=int,   default=8,
                   help='バッチサイズ (H100 80GB: 8~16 推奨)')
    p.add_argument('--grad_accum',   type=int,   default=8,
                   help='勾配累積 → 実効バッチ = batch × grad_accum')
    p.add_argument('--lr',           type=float, default=5e-5)
    p.add_argument('--wd',           type=float, default=1e-2)
    p.add_argument('--lora_r',       type=int,   default=64)
    p.add_argument('--lora_alpha',   type=int,   default=128)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--max_length',   type=int,   default=1024)
    p.add_argument('--max_train',    type=int,   default=0,
                   help='訓練データ上限 (0=全件使用, H100デフォルト)')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--eval_steps',   type=int,   default=100)
    p.add_argument('--eval_samples', type=int,   default=200)
    p.add_argument('--eval_batch',   type=int,   default=16)
    p.add_argument('--save_steps',   type=int,   default=200)
    p.add_argument('--vram_kill',    type=float, default=0.95,
                   help='VRAM ウォッチドッグ上限割合 (0.95=76GB/80GB)')
    p.add_argument('--compile',      action='store_true', default=True,
                   help='torch.compile 有効化 (H100 推奨)')
    p.add_argument('--resume',       action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"  モデル       : {args.model_id}")
    print(f"  実効バッチ   : {args.batch} x {args.grad_accum} = {args.batch * args.grad_accum}")
    print(f"  LoRA rank    : {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  max_length   : {args.max_length}")
    print(f"  訓練上限     : {'全件' if args.max_train == 0 else f'{args.max_train:,}件'}")
    if args.resume:
        ckpt = find_latest_checkpoint()
        print(f"  再開         : {ckpt.name if ckpt else 'チェックポイントなし'}")
    result = train(args)
    print(f"\n最良 Accuracy: {result['best_accuracy']:.4f}")
