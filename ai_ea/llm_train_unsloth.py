"""
LLM Unsloth QLoRA ファインチューニング
FX BUY/SELL/HOLD シグナル予測

- Unsloth FastLanguageModel (2x高速, 60%VRAM削減)
- QLoRA 4-bit (NF4 + double quant)
- adamw_8bit オプティマイザ
- VRAM ウォッチドッグ (超過で即停止 → PCフリーズ防止)
- チェックポイント保存・再開

使用方法:
    # Qwen3-8B (推奨)
    python llm_train_unsloth.py --model_id Qwen/Qwen3-8B --batch 1 --grad_accum 32

    # Qwen3-4B (軽量)
    python llm_train_unsloth.py --model_id Qwen/Qwen3-4B --batch 2 --grad_accum 16

    # チェックポイントから再開
    python llm_train_unsloth.py --resume
"""
import sys, os, json, time, math, argparse, threading
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

MODEL_ID    = "Qwen/Qwen3-8B"
LABEL_NAMES = ['HOLD', 'BUY', 'SELL']
LABEL_TO_ID = {n: i for i, n in enumerate(LABEL_NAMES)}


# ──────────────────────────────────────────────────────────────────────────
# VRAM ウォッチドッグ
# ──────────────────────────────────────────────────────────────────────────
_watchdog_paused = False   # eval 中は一時停止

def start_vram_watchdog(limit_gb: float = 9.5, interval_sec: float = 1.5) -> None:
    """VRAM が limit_gb を超えたら即 os._exit(2) → PCフリーズ防止
    memory_allocated() (実使用量のみ) を監視し、eval 中は一時停止する"""
    def _watch():
        while True:
            try:
                if not _watchdog_paused:
                    # reserved ではなく allocated (キャッシュ除く実使用量) で判定
                    used = torch.cuda.memory_allocated(0) / 1024**3
                    if used >= limit_gb:
                        print(f"\n[WATCHDOG] VRAM {used:.1f}GB >= {limit_gb:.1f}GB → 緊急停止！",
                              flush=True)
                        os._exit(2)
            except Exception:
                pass
            time.sleep(interval_sec)
    t = threading.Thread(target=_watch, daemon=True, name='vram_watchdog')
    t.start()
    print(f"  VRAM ウォッチドッグ: 上限 {limit_gb:.1f}GB")


# ──────────────────────────────────────────────────────────────────────────
# GPU モニタリング
# ──────────────────────────────────────────────────────────────────────────
def get_gpu_stats() -> tuple:
    """(utilization%, vram_used_GB, vram_total_GB)"""
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2)
        vals = r.stdout.strip().split(',')
        return float(vals[0]), float(vals[1])/1024, float(vals[2])/1024
    except Exception:
        used  = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0, used, total


# ──────────────────────────────────────────────────────────────────────────
# ダッシュボード更新
# ──────────────────────────────────────────────────────────────────────────
_dash_state: dict = {}

def _update_dash(patch: dict) -> None:
    _dash_state.update(patch)
    try:
        sys.path.insert(0, str(OUT_DIR))
        from llm_dashboard import update as dash_update
        dash_update(_dash_state)
    except Exception:
        pass
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


class SignalDataset:
    def __init__(self, samples: list, tokenizer, max_length: int = 512):
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


# ──────────────────────────────────────────────────────────────────────────
# チェックポイント
# ──────────────────────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler,
                    best_acc, epoch_log, tokenizer, step=None):
    CKPT_DIR.mkdir(exist_ok=True)
    tag = f'epoch_{epoch:03d}' if step is None else f'epoch_{epoch:03d}_step_{step:06d}'
    ckpt_path = CKPT_DIR / f'{tag}.pt'
    lora_state = {k: v.cpu().clone()
                  for k, v in model.state_dict().items() if 'lora' in k}
    torch.save({
        'epoch': epoch, 'step': step, 'lora_state': lora_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc, 'epoch_log': epoch_log,
    }, ckpt_path)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    # 古いものを 3 つ残して削除
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


# ──────────────────────────────────────────────────────────────────────────
# 評価
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size, device, label_token_ids):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    model.eval()
    correct = total = 0
    total_loss = 0.0
    lbl_tids = torch.tensor([label_token_ids[n] for n in LABEL_NAMES], device=device)

    for batch in dl:
        ids   = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        gt    = batch['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=True):
            out = model(input_ids=ids, attention_mask=masks)
        last_pos = masks.sum(1) - 1
        # val_loss
        sel = torch.stack([out.logits[b, last_pos[b], lbl_tids] for b in range(len(gt))])
        total_loss += nn.functional.cross_entropy(sel, gt, label_smoothing=0.05).item()
        # accuracy
        for b in range(len(gt)):
            pred = int(sel[b].argmax())
            correct += int(pred == int(gt[b]))
            total   += 1

    model.train()
    return {
        'accuracy': round(correct / max(total, 1), 4),
        'val_loss': round(total_loss / max(len(dl), 1), 5),
        'total': total,
    }


# ──────────────────────────────────────────────────────────────────────────
# メイン訓練
# ──────────────────────────────────────────────────────────────────────────
def train(args):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from unsloth import FastLanguageModel

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                          'expandable_segments:True,max_split_size_mb:256')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p      = torch.cuda.get_device_properties(0)
    total_vram = p.total_memory / 1024**3
    print(f"\n=== Unsloth QLoRA ファインチューニング ===")
    print(f"  GPU: {p.name}  VRAM: {total_vram:.1f}GB  CC: {p.major}.{p.minor}")

    # VRAM ウォッチドッグ起動
    watchdog_gb = total_vram * args.vram_kill
    start_vram_watchdog(limit_gb=watchdog_gb)

    _update_dash({'phase': 'loading', 'epoch': 0, 'total_epochs': args.epochs,
                  'batch': 0, 'total_batches': 1, 'train_loss': 0.0,
                  'accuracy': 0.0, 'best_acc': 0.0, 'gpu_pct': 0,
                  'vram_used_gb': 0.0, 'vram_total_gb': total_vram,
                  'lr': args.lr, 'elapsed_sec': 0.0, 'eta_sec': -1,
                  'epoch_log': [], 'message': 'Unsloth モデル読み込み中...'})

    # ── データ読み込み ───────────────────────────────────────────────────
    if not TRAIN_JSONL.exists():
        print(f"[ERROR] {TRAIN_JSONL} がありません。先に llm_dataset.py を実行してください")
        sys.exit(1)
    train_raw = load_jsonl(TRAIN_JSONL)
    test_raw  = load_jsonl(TEST_JSONL)
    if args.max_train > 0 and len(train_raw) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(train_raw), args.max_train, replace=False).tolist())
        train_raw = [train_raw[i] for i in idx]
    print(f"  訓練: {len(train_raw):,}  テスト: {len(test_raw):,}")

    # ── Unsloth モデルロード ─────────────────────────────────────────────
    print(f"  Unsloth モデル: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model_id,
        max_seq_length = args.max_length,   # RoPE / KV cache をここで制限
        load_in_4bit   = True,
        load_in_8bit   = False,
        dtype          = None,
        trust_remote_code = True,
        # Unsloth 独自の VRAM 節約オプション
        token          = None,
    )

    vram_load = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  ロード後 VRAM: {vram_load:.1f}GB")

    # ── Unsloth LoRA 設定 ────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj'],
        bias           = 'none',
        use_gradient_checkpointing = 'unsloth',  # Unsloth 独自の高効率実装
        random_state   = args.seed,
        max_seq_length = args.max_length,
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA パラメータ: {n_train:,} / {n_total:,} ({n_train/n_total*100:.3f}%)")

    # torch.compile で推論グラフを最適化 (triton-windows が使える場合)
    if args.compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  torch.compile: 有効 (初回バッチは遅いが以降高速化)")
        except Exception as e:
            print(f"  torch.compile: スキップ ({e})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ── トークナイズ ─────────────────────────────────────────────────────
    print("  トークナイズ中 (訓練)...")
    tr_ds = SignalDataset(train_raw, tokenizer, args.max_length)
    print("  トークナイズ中 (テスト)...")
    te_ds = SignalDataset(test_raw,  tokenizer, args.max_length)

    pad_id = tokenizer.pad_token_id
    tr_dl  = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)
    total_batches = len(tr_dl)
    print(f"  バッチ数/epoch: {total_batches}")

    # ── オプティマイザ (adamw_8bit で VRAM 節約) ────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.wd)
        print("  オプティマイザ: AdamW 8-bit (VRAM 節約)")
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.wd)
        print("  オプティマイザ: AdamW FP32")

    total_steps  = total_batches * args.epochs
    warmup_steps = max(20, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    label_tids = torch.tensor(
        [tr_ds.label_token_ids[n] for n in LABEL_NAMES], device=device)

    # ── チェックポイント / 再開 ──────────────────────────────────────────
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

    _update_dash({'phase': 'training', 'total_batches': total_batches,
                  'epoch': start_epoch - 1, 'best_acc': best_acc,
                  'epoch_log': epoch_log,
                  'message': f'Unsloth 訓練開始 epoch {start_epoch}/{args.epochs}'})

    def compute_loss(logits, attn_masks, gt_labels):
        last_pos   = attn_masks.sum(1) - 1
        sel_logits = torch.stack(
            [logits[b, last_pos[b], label_tids] for b in range(logits.shape[0])])
        return nn.functional.cross_entropy(sel_logits, gt_labels, label_smoothing=0.05)

    throttle_ratio   = args.throttle
    no_imp_acc       = 0
    no_imp_val       = 0
    patience_acc     = max(3, args.epochs // 2)
    patience_overfit = 2
    global_step      = (start_epoch - 1) * total_batches
    train_start      = time.time()
    batch_log        = []   # [{step, train_loss, val_loss}] バッチごと履歴

    # val_loss 高速評価用の固定サブセット (eval_samples 件)
    rng_eval  = np.random.default_rng(args.seed + 99)
    eval_idx  = sorted(rng_eval.choice(len(te_ds), min(args.eval_samples, len(te_ds)),
                                       replace=False).tolist())
    eval_sub  = [te_ds[i] for i in eval_idx]
    from torch.utils.data import DataLoader as _DL
    eval_dl   = _DL(eval_sub, batch_size=1, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)

    # ── 訓練ループ ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss  = 0.0
        ep_start = time.time()
        gpu_pct_last = 0.0

        for bi, batch in enumerate(tr_dl, start=1):
            b_start = time.time()
            ids   = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            gt    = batch['labels'].to(device)

            with torch.amp.autocast('cuda'):
                out  = model(input_ids=ids, attention_mask=masks)
                loss = compute_loss(out.logits, masks, gt)

            scaler.scale(loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            ep_loss     += loss.item()
            global_step += 1

            # GPU スロットリング (throttle=0 なら無効)
            if throttle_ratio > 0:
                batch_time = time.time() - b_start
                torch.cuda.synchronize()
                time.sleep(batch_time * throttle_ratio)

            # 定期キャッシュ解放
            if bi % 100 == 0:
                torch.cuda.empty_cache()

            # 定期チェックポイント保存 (save_steps ごと)
            if args.save_steps > 0 and bi % args.save_steps == 0:
                save_checkpoint(epoch, model, optimizer, scheduler,
                                best_acc, epoch_log, tokenizer, step=global_step)

            # ── 定期 val_loss + accuracy 計算 (eval_steps ごと) ──────────
            cur_val_loss = None
            cur_acc      = None
            if bi % args.eval_steps == 0 or bi == total_batches:
                global _watchdog_paused
                _watchdog_paused = True        # eval 中はウォッチドッグ停止
                torch.cuda.empty_cache()
                model.eval()
                v_loss = 0.0; v_correct = 0; v_total = 0
                with torch.no_grad():
                    for vb in eval_dl:
                        vids   = vb['input_ids'].to(device)
                        vmasks = vb['attention_mask'].to(device)
                        vgt    = vb['labels'].to(device)
                        with torch.amp.autocast('cuda'):
                            vout = model(input_ids=vids, attention_mask=vmasks)
                        # logits は vocab×seq で巨大 → 必要箇所だけ抜いて即解放
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
                _watchdog_paused = False       # ウォッチドッグ再開

            # バッチ loss / val_loss / accuracy を蓄積
            entry = {'step': global_step, 'train_loss': round(loss.item(), 4)}
            if cur_val_loss is not None:
                entry['train_loss'] = round(ep_loss / bi, 4)  # 評価時はエポック平均に上書き
                entry['val_loss'] = cur_val_loss
                entry['acc']      = cur_acc
            batch_log.append(entry)
            if len(batch_log) > 2000:
                batch_log = batch_log[-2000:]

            # 進捗表示 & ダッシュボード更新 (50バッチごと)
            if bi % 50 == 0 or bi == total_batches:
                elapsed  = time.time() - train_start
                done_ep  = (epoch - start_epoch) + (bi / total_batches)
                total_ep = args.epochs - start_epoch + 1
                eta      = elapsed / max(done_ep, 0.001) * (total_ep - done_ep)
                lr_now   = optimizer.param_groups[0]['lr']
                avg_loss = ep_loss / bi

                gpu_pct, vram_used, vram_total = get_gpu_stats()
                gpu_pct_last = gpu_pct
                pct = bi / total_batches * 100
                print(f"  Ep{epoch}/{args.epochs}  [{bi:5d}/{total_batches}] {pct:5.1f}%"
                      f"  loss={avg_loss:.4f}  lr={lr_now:.2e}"
                      f"  GPU:{gpu_pct:.0f}%  VRAM:{vram_used:.1f}GB"
                      f"  {time.strftime('%H:%M:%S')}", flush=True)

                _update_dash({
                    'phase': 'training', 'epoch': epoch, 'batch': bi,
                    'train_loss': round(avg_loss, 5), 'lr': round(lr_now, 8),
                    'elapsed_sec': round(elapsed, 1), 'eta_sec': round(eta, 1),
                    'gpu_pct': int(gpu_pct), 'vram_used_gb': round(vram_used, 2),
                    'vram_total_gb': round(vram_total, 2),
                    'batch_log': batch_log,
                    'message': f'Epoch {epoch}/{args.epochs}  batch {bi}/{total_batches}',
                })

        ep_loss /= total_batches

        # エポック末評価
        # チェックポイントを評価の前に保存 (評価中に死んでもロスしない)
        save_checkpoint(epoch, model, optimizer, scheduler,
                        best_acc, epoch_log, tokenizer)

        print(f"  Ep{epoch} 評価中...", flush=True)
        torch.cuda.empty_cache()   # 評価前にキャッシュ解放
        ev = evaluate(model, tokenizer, te_ds, 1, device,   # batch=1 で VRAM スパイク防止
                      te_ds.label_token_ids)
        ep_elapsed = time.time() - ep_start

        val_loss = ev['val_loss']
        is_best  = ev['accuracy'] > best_acc + 0.001

        if is_best:
            best_acc = ev['accuracy']
            model.save_pretrained(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))
            no_imp_acc = 0
            print(f"  [BEST] acc={best_acc:.4f}  val_loss={val_loss:.4f}", flush=True)
        else:
            no_imp_acc += 1

        # 過学習検知: val_loss が前 epoch より悪化
        if len(epoch_log) >= 1:
            prev_val = epoch_log[-1].get('val_loss', val_loss)
            if val_loss > prev_val * 1.02:   # 2% 以上悪化
                no_imp_val += 1
                print(f"  [OVERFIT] val_loss 上昇 ({prev_val:.4f} → {val_loss:.4f}) "
                      f"連続 {no_imp_val} 回", flush=True)
            else:
                no_imp_val = 0

        ep_entry = {
            'epoch':      epoch,
            'train_loss': round(ep_loss, 5),
            'val_loss':   round(val_loss, 5),
            'acc':        ev['accuracy'],
            'elapsed':    round(ep_elapsed, 1),
            'is_best':    is_best,
            'step_end':   global_step,   # グラフ上の x 位置
        }
        epoch_log.append(ep_entry)

        print(f"\n  === Epoch {epoch}/{args.epochs} 完了 ==="
              f"  train={ep_loss:.4f}  val={val_loss:.4f}"
              f"  acc={ev['accuracy']:.4f}  best={best_acc:.4f}"
              f"  {ep_elapsed:.0f}s\n", flush=True)

        gpu_pct, vram_used, vram_total = get_gpu_stats()
        _update_dash({
            'phase': 'training', 'epoch': epoch, 'batch': total_batches,
            'train_loss': round(ep_loss, 5), 'val_loss': round(val_loss, 5),
            'accuracy': ev['accuracy'], 'best_acc': best_acc,
            'epoch_log': epoch_log, 'gpu_pct': int(gpu_pct),
            'vram_used_gb': round(vram_used, 2),
            'message': f'Epoch {epoch} 完了  train={ep_loss:.4f}  val={val_loss:.4f}  acc={ev["accuracy"]:.4f}',
        })

        # 早期終了判定
        stop_reason = None
        if no_imp_val >= patience_overfit:
            stop_reason = f'過学習検知: val_loss が {patience_overfit} epoch 連続悪化'
        elif no_imp_acc >= patience_acc:
            stop_reason = f'精度改善なし: {patience_acc} epoch 連続'
        if stop_reason:
            print(f"  [EARLY STOP] {stop_reason}", flush=True)
            break

    # 最良を保存
    if BEST_DIR.exists():
        import shutil
        for f in BEST_DIR.iterdir():
            shutil.copy(f, ADAPTER_DIR)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    total_elapsed = time.time() - train_start
    result = {
        'best_accuracy': best_acc, 'epochs_trained': epoch,
        'train_samples': len(tr_ds), 'test_samples': len(te_ds),
        'model_id': args.model_id, 'lora_r': args.lora_r,
        'label_token_ids': tr_ds.label_token_ids,
        'total_min': round(total_elapsed / 60, 1),
        'mode': 'unsloth_qlora',
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== 訓練完了 ===  best_acc={best_acc:.4f}  {total_elapsed/60:.1f}分",
          flush=True)
    _update_dash({'phase': 'done', 'best_acc': best_acc,
                  'message': f'完了！ Best Acc={best_acc:.4f}  {total_elapsed/60:.1f}分'})
    return result


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id',     type=str,   default=MODEL_ID)
    p.add_argument('--epochs',       type=int,   default=5)
    p.add_argument('--batch',        type=int,   default=1,
                   help='バッチサイズ (8B: 1, 4B: 2 推奨)')
    p.add_argument('--grad_accum',   type=int,   default=32,
                   help='勾配累積 (実効バッチ = batch × grad_accum)')
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--wd',           type=float, default=1e-2)
    p.add_argument('--lora_r',       type=int,   default=8,
                   help='LoRA rank (8=VRAM節約, 16=高精度)')
    p.add_argument('--lora_alpha',   type=int,   default=16)
    p.add_argument('--lora_dropout', type=float, default=0.0,
                   help='Unsloth 推奨: 0 (最適化済み)')
    p.add_argument('--max_length',   type=int,   default=448,
                   help='最大トークン長 (実データ平均437, 448=VRAM節約)')
    p.add_argument('--max_train',    type=int,   default=0)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--throttle',     type=float, default=0.0,
                   help='GPU スロットリング (0=制限なし, 0.4=約71%)')
    p.add_argument('--compile',      action='store_true',
                   help='torch.compile で高速化 (初回コンパイルに数分かかる)')
    p.add_argument('--eval_steps',   type=int,   default=50,
                   help='何バッチごとに val_loss を計算するか')
    p.add_argument('--eval_samples', type=int,   default=50,
                   help='val_loss 計算に使うテストサンプル数 (小さいほど高速)')
    p.add_argument('--save_steps',   type=int,   default=50,
                   help='何バッチごとにチェックポイント保存するか (0=epoch末のみ)')
    p.add_argument('--vram_kill',    type=float, default=0.88,
                   help='VRAM ウォッチドッグ上限割合 (0.88=9.7GB/11GB)')
    p.add_argument('--resume',       action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"  モデル    : {args.model_id}")
    print(f"  実効バッチ: {args.batch} x {args.grad_accum} = {args.batch * args.grad_accum}")
    print(f"  スロットリング: {args.throttle} (GPU 約{100/(1+args.throttle):.0f}%)")
    if args.resume:
        ckpt = find_latest_checkpoint()
        print(f"  再開: {ckpt.name if ckpt else 'チェックポイントなし'}")
    result = train(args)
    print(f"\n最良 Accuracy: {result['best_accuracy']:.4f}")
