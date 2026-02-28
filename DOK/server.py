"""
FX LLM Training Server  ─  Sakura Cloud H100 用
FastAPI で学習管理・進捗配信・モデルDL・成果物DLを提供
"""
import asyncio, base64, json, os, signal, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── パス設定 ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
TRAIN_DIR   = BASE_DIR / "ai_ea"           # 学習スクリプト置き場
STATE_FILE  = BASE_DIR / "train_state.json"
LOG_FILE    = BASE_DIR / "train_run.log"
CKPT_DIR    = BASE_DIR / "checkpoints"
REPORT_DIR  = BASE_DIR / "reports"
MODEL_CACHE = Path("/root/.cache/huggingface/hub")

CKPT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── 推奨モデル 10 選 (H100 80GB で動くもの) ──────────────────────────────────
MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct",      "name": "Qwen2.5-7B",       "vram_4bit_gb": 5,
     "desc": "汎用・高速。FXシグナル向けベースライン", "tags": ["推奨", "高速"]},
    {"id": "Qwen/Qwen2.5-14B-Instruct",     "name": "Qwen2.5-14B",      "vram_4bit_gb": 9,
     "desc": "7Bより精度向上。H100で余裕",           "tags": ["推奨"]},
    {"id": "Qwen/Qwen2.5-32B-Instruct",     "name": "Qwen2.5-32B",      "vram_4bit_gb": 20,
     "desc": "高精度・大規模推論",                    "tags": ["大型"]},
    {"id": "Qwen/Qwen2.5-72B-Instruct",     "name": "Qwen2.5-72B",      "vram_4bit_gb": 42,
     "desc": "最高精度。H100 80GBギリギリ可",        "tags": ["最大", "超高精度"]},
    {"id": "Qwen/Qwen3-8B",                  "name": "Qwen3-8B",          "vram_4bit_gb": 6,
     "desc": "最新Qwen3系。ローカル実績あり",         "tags": ["最新"]},
    {"id": "Qwen/Qwen3-14B",                 "name": "Qwen3-14B",         "vram_4bit_gb": 10,
     "desc": "Qwen3最新14B。高い推論力",             "tags": ["最新", "推奨"]},
    {"id": "meta-llama/Llama-3.1-8B-Instruct",  "name": "Llama3.1-8B",  "vram_4bit_gb": 5,
     "desc": "Meta製。汎用ファインチューニング向け",  "tags": ["Meta"]},
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "name": "Llama3.1-70B", "vram_4bit_gb": 40,
     "desc": "Meta最大級。H100 80GBで運用可",        "tags": ["Meta", "大型"]},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3","name": "Mistral-7B",   "vram_4bit_gb": 5,
     "desc": "軽量・高速。FXシグナルの定番",         "tags": ["軽量"]},
    {"id": "google/gemma-2-9b-it",           "name": "Gemma2-9B",         "vram_4bit_gb": 6,
     "desc": "Google製。指示追従性高い",             "tags": ["Google"]},
]
MODEL_IDS = {m["id"] for m in MODELS}

# ── グローバル状態 ────────────────────────────────────────────────────────────
_train_proc: Optional[subprocess.Popen] = None
_dl_status: dict = {}       # model_id -> {status, progress, error}

app = FastAPI(title="FX LLM Training Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── 状態ファイル読み込み ──────────────────────────────────────────────────────
def _read_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"phase": "waiting", "message": "待機中"}


def _is_training() -> bool:
    global _train_proc
    if _train_proc is None:
        return False
    return _train_proc.poll() is None


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    st = _read_state()
    st["server_time"] = datetime.now().isoformat()
    st["training_active"] = _is_training()
    # ダウンロード状況を付加
    st["dl_status"] = _dl_status
    return JSONResponse(st)


@app.get("/api/models")
def get_models():
    result = []
    for m in MODELS:
        m2 = dict(m)
        m2["dl_status"] = _dl_status.get(m["id"], {"status": "not_downloaded"})
        # キャッシュに存在するか確認
        cache_name = m["id"].replace("/", "--")
        m2["cached"] = (MODEL_CACHE / f"models--{cache_name}").exists()
        result.append(m2)
    return result


class StartConfig(BaseModel):
    model_id: str = "Qwen/Qwen3-8B"
    epochs: int = 10
    batch: int = 16
    grad_accum: int = 4
    lr: float = 2e-4
    max_length: int = 1024
    lora_r: int = 64
    max_train: int = 0          # 0 = 全件
    eval_steps: int = 100
    compile: bool = True
    resume: bool = False


@app.post("/api/start")
def start_training(cfg: StartConfig):
    global _train_proc
    if _is_training():
        raise HTTPException(400, "既に学習中です")
    if cfg.model_id not in MODEL_IDS:
        raise HTTPException(400, f"未知のモデル: {cfg.model_id}")

    script = TRAIN_DIR / "train_h100.py"
    if not script.exists():
        script = BASE_DIR / "train_h100.py"

    cmd = [
        sys.executable, str(script),
        "--model_id",    cfg.model_id,
        "--epochs",      str(cfg.epochs),
        "--batch",       str(cfg.batch),
        "--grad_accum",  str(cfg.grad_accum),
        "--lr",          str(cfg.lr),
        "--max_length",  str(cfg.max_length),
        "--lora_r",      str(cfg.lora_r),
        "--eval_steps",  str(cfg.eval_steps),
        "--state_file",  str(STATE_FILE),
        "--ckpt_dir",    str(CKPT_DIR),
    ]
    if cfg.max_train > 0:
        cmd += ["--max_train", str(cfg.max_train)]
    if cfg.compile:
        cmd.append("--compile")
    if cfg.resume:
        cmd.append("--resume")

    log_fh = open(LOG_FILE, "w", encoding="utf-8")
    _train_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                   cwd=str(BASE_DIR))
    return {"status": "started", "pid": _train_proc.pid}


@app.post("/api/stop")
def stop_training():
    global _train_proc
    if not _is_training():
        raise HTTPException(400, "学習中ではありません")
    _train_proc.send_signal(signal.SIGTERM)
    try:
        _train_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _train_proc.kill()
    return {"status": "stopped"}


# ── モデルダウンロード ─────────────────────────────────────────────────────────

def _dl_worker(model_id: str):
    _dl_status[model_id] = {"status": "downloading", "progress": 0, "error": ""}
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.gguf", "*.bin", "original/*"],
            local_dir_use_symlinks=False,
        )
        _dl_status[model_id] = {"status": "done", "progress": 100, "error": ""}
    except Exception as e:
        _dl_status[model_id] = {"status": "error", "progress": 0, "error": str(e)}


@app.post("/api/download_model/{model_id_b64}")
def download_model(model_id_b64: str):
    model_id = base64.urlsafe_b64decode(model_id_b64 + "==").decode()
    if model_id not in MODEL_IDS:
        raise HTTPException(400, f"未知のモデル: {model_id}")
    if _dl_status.get(model_id, {}).get("status") == "downloading":
        raise HTTPException(400, "ダウンロード中です")
    t = threading.Thread(target=_dl_worker, args=(model_id,), daemon=True)
    t.start()
    return {"status": "started", "model_id": model_id}


# ── 成果物ダウンロード ─────────────────────────────────────────────────────────

@app.get("/download/adapter")
def dl_adapter():
    """最新の LoRA アダプタを zip で返す"""
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in CKPT_DIR.rglob("*.pt"):
            zf.write(f, f.relative_to(CKPT_DIR))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=lora_adapter.zip"})


@app.get("/download/log")
def dl_log():
    if not LOG_FILE.exists():
        raise HTTPException(404, "ログなし")
    return FileResponse(LOG_FILE, filename="train_run.log")


@app.get("/download/state")
def dl_state():
    if not STATE_FILE.exists():
        raise HTTPException(404, "状態ファイルなし")
    return FileResponse(STATE_FILE, filename="train_state.json")


@app.get("/download/report")
def dl_report():
    files = sorted(REPORT_DIR.glob("*.html"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise HTTPException(404, "レポートなし")
    return FileResponse(files[-1], filename=files[-1].name)


# ── ヘルスチェック ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}


# ── 静的ファイル (ダッシュボード HTML) ────────────────────────────────────────
if (BASE_DIR / "local_monitor.html").exists():
    @app.get("/")
    def root():
        return FileResponse(BASE_DIR / "local_monitor.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
