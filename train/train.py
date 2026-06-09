import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager

# Set up HF_HOME before any huggingface imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_CACHE_DIR = os.path.join(_ROOT, ".hf_cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)

# trl loads UTF-8 .jinja templates via Path.read_text(); Windows defaults to cp1252.
if sys.platform == "win32":
    import pathlib

    _pathlib_read_text = pathlib.Path.read_text

    def _path_read_text_utf8(self, *args, **kwargs):
        if "encoding" not in kwargs and not args:
            kwargs["encoding"] = "utf-8"
        return _pathlib_read_text(self, *args, **kwargs)

    pathlib.Path.read_text = _path_read_text_utf8

import torch
from flask import Flask, jsonify, render_template
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.view_data import load_training_dataset

# Load configuration from JSON
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.json")
with open(_CONFIG_FILE, "r") as f:
    _config = json.load(f)

_DATA_DIR = os.path.join(_ROOT, "data")

# Model configuration
MODEL_NAME = _config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
_model_dir_name = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
OUTPUT_DIR = os.path.join(_DATA_DIR, f"finetuned_lora_{_model_dir_name}")

# Training config
REQUIRE_CUDA = _config.get("require_cuda", False)
EPOCHS = _config.get("epochs", 3)
BATCH_SIZE = _config.get("batch_size", 2)
CPU_BATCH_SIZE = _config.get("cpu_batch_size", 1)
GRADIENT_ACCUMULATION_STEPS = _config.get("gradient_accumulation_steps", 4)
LEARNING_RATE = _config.get("learning_rate", 2e-4)
MAX_SEQ_LENGTH = _config.get("max_seq_length", 512)

# LoRA config
LORA_R = _config.get("lora_r", 8)
LORA_ALPHA = _config.get("lora_alpha", 16)
LORA_DROPOUT = _config.get("lora_dropout", 0.05)
TARGET_MODULES = _config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

# ChromaDB
CHROMA_PATH = os.path.join(_DATA_DIR, "chroma_db")
CHROMA_COLLECTION = _config.get("chroma_collection", "qa_samples")

app = Flask(__name__, template_folder=".")
_state_lock = threading.Lock()
STEP_TOTAL = 5
_STEP_LABELS = {
    1: "Đọc dữ liệu (ChromaDB)",
    2: "Tải tokenizer",
    3: "Tải model (checkpoint shards)",
    4: "Cấu hình LoRA",
    5: "Huấn luyện",
}

_train_state = {
    "status": "IDLE",
    "stage": "Waiting to start",
    "step": 0,
    "step_total": STEP_TOTAL,
    "step_label": "",
    "elapsed_sec": 0,
    "progress_percent": 0.0,
    "current_epoch": 0.0,
    "total_epochs": 0.0,
    "global_step": 0,
    "dataset_rows": 0,
    "output_dir": "",
    "message": "",
    "device": "",
}


def update_state(**kwargs) -> None:
    with _state_lock:
        _train_state.update(kwargs)


def set_phase(step: int, stage: str, message: str, progress_percent: float) -> None:
    update_state(
        status="RUNNING",
        step=step,
        step_total=STEP_TOTAL,
        step_label=_STEP_LABELS.get(step, ""),
        stage=stage,
        message=message,
        progress_percent=progress_percent,
        elapsed_sec=0,
    )


@contextmanager
def stage_heartbeat(step: int, stage: str, message: str, interval: float = 2.0):
    """Cập nhật dashboard khi bước dài (load model) — tránh tưởng bị treo."""
    stop = threading.Event()
    start = time.monotonic()

    def tick() -> None:
        while not stop.wait(interval):
            sec = int(time.monotonic() - start)
            update_state(
                status="RUNNING",
                step=step,
                step_total=STEP_TOTAL,
                step_label=_STEP_LABELS.get(step, ""),
                stage=stage,
                elapsed_sec=sec,
                message=(
                    f"[Bước {step}/{STEP_TOTAL}] {message} — {sec}s " 
                ),
            )

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 1)


def _use_cuda() -> bool:
    return torch.cuda.is_available()


def ensure_train_device() -> str:
    if _use_cuda():
        name = torch.cuda.get_device_name(0)
        update_state(device=name, message=f"GPU: {name}")
        print(f"[DEVICE] CUDA OK — {name}")
        return name

    hint = ( 
        "Để dùng GPU: cài driver NVIDIA + PyTorch CUDA:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
    if REQUIRE_CUDA:
        print(f"[DEVICE] ERROR — {hint}")
        update_state(
            status="ERROR",
            stage="No GPU",
            device="CPU only",
            message=hint.split("\n")[0],
        )
        raise RuntimeError(hint)

    print(f"[DEVICE] WARNING — {hint.split(chr(10))[0]}")
    update_state(device="CPU", message="Train trên CPU (full precision)")
    return "CPU"


def load_base_model():
    """4-bit + bitsandbytes trên CUDA; full precision trên CPU."""
    if _use_cuda():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return prepare_model_for_kbit_training(model)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model


def build_training_args(output_dir: str) -> SFTConfig:
    use_cuda = _use_cuda()
    batch_size = (
        BATCH_SIZE
        if use_cuda
        else CPU_BATCH_SIZE
    )
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=use_cuda,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        use_cpu=not use_cuda,
    )


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        progress = 55.0
        if state.max_steps and state.max_steps > 0:
            train_pct = state.global_step / state.max_steps
            progress = min(100.0, 55.0 + train_pct * 45.0)
        update_state(
            status="RUNNING",
            step=5,
            step_total=STEP_TOTAL,
            step_label=_STEP_LABELS[5],
            stage="Training",
            progress_percent=progress,
            current_epoch=float(state.epoch or 0.0),
            total_epochs=float(args.num_train_epochs),
            global_step=int(state.global_step),
            message=f"loss={logs.get('loss', 'n/a')}",
        )


@app.route("/api/stats")
def api_stats():
    with _state_lock:
        return jsonify(dict(_train_state))


@app.route("/")
def index():
    return render_template("index.html")


def start_dashboard(host: str, port: int) -> None:
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main():
    full_output_dir = os.path.normpath(OUTPUT_DIR)
    update_state(
        total_epochs=float(EPOCHS),
        output_dir=full_output_dir,
    )

    ensure_train_device()

    print("[1/5] Loading dataset from ChromaDB (data/view_data.py)...")
    set_phase(1, "Preparing data", "Đọc mẫu selected từ ChromaDB", 2.0)
    dataset = load_training_dataset()
    update_state(dataset_rows=len(dataset), progress_percent=8.0)

    print("[2/5] Initializing tokenizer...")
    set_phase(2, "Loading tokenizer", "Tải tokenizer từ Hugging Face", 10.0)
    with stage_heartbeat(2, "Loading tokenizer", "Tải tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        """Chuyển đổi dữ liệu sang cấu trúc ChatML chuẩn của dòng Qwen"""
        im_end = "<|" + "im_end" + "|>"
        texts = []
        for user_text, assistant_text in zip(examples["user"], examples["assistant"]):
            text = (
                f"<|im_start|>user\n{user_text}{im_end}\n"
                f"<|im_start|>assistant\n{assistant_text}{im_end}"
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    update_state(progress_percent=12.0, message="Đã format dataset ChatML")

    quant_label = "4-bit" if _use_cuda() else "full precision (CPU)"
    print(f"[3/5] Loading base model {MODEL_NAME} ({quant_label})...")
    set_phase(
        3,
        "Loading model",
        f"{MODEL_NAME} — shard 1/2 rồi 2/2 (terminal: Loading checkpoint shards)",
        15.0,
    )

    with stage_heartbeat(3, "Loading model", f"Tải {MODEL_NAME} ({quant_label})"):
        model = load_base_model()
    set_phase(3, "Loading model", "Model đã load xong", 40.0)
    print("[3/5] Model loaded.")

    print("[4/5] Applying LoRA configuration...")
    set_phase(4, "Configuring LoRA", "Gắn adapter LoRA", 45.0)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("[5/5] Building TrainingArguments and SFTTrainer...")
    set_phase(4, "Preparing trainer", "Khởi tạo SFTTrainer", 50.0)
    update_state(output_dir=full_output_dir)

    training_args = build_training_args(full_output_dir)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[ProgressCallback()],
    )

    print("\n>>> START FINE-TUNING <<<")
    set_phase(5, "Training", "Bắt đầu fine-tune — % sẽ tăng theo step", 55.0)
    trainer.train()

    print(f"\n[OK] Training completed. Saving model to: {full_output_dir}")
    update_state(
        status="SAVING",
        step=5,
        stage="Saving model",
        progress_percent=98.0,
        message="Đang lưu adapter + tokenizer",
    )
    trainer.model.save_pretrained(full_output_dir)
    tokenizer.save_pretrained(full_output_dir)
    update_state(
        status="DONE",
        step=5,
        stage="Completed",
        progress_percent=100.0,
        message="Huấn luyện xong",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Cho phép train khi không có CUDA (bỏ qua REQUIRE_CUDA trong config)",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Bắt buộc GPU CUDA (ghi đè REQUIRE_CUDA=False trong config)",
    )
    args = parser.parse_args()

    if args.allow_cpu:
        globals()["REQUIRE_CUDA"] = False
    if args.require_cuda:
        globals()["REQUIRE_CUDA"] = True

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        args=("0.0.0.0", args.port),
        daemon=True,
    )
    dashboard_thread.start()
    print(f"[DASHBOARD] http://0.0.0.0:{args.port}")
    try:
        main()
    except Exception as exc:
        update_state(status="ERROR", stage="Failed", message=str(exc)[:300])
        raise
