import argparse
import json
import os
import threading
import urllib.error
import urllib.request

import numpy as np
from flask import Flask, jsonify, render_template
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from generate.chroma_db import QAStore, pair_document
from generate.checks import check_rules

# Load configuration from JSON
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CONFIG_DIR)
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "generate_config.json")
_HF_CACHE_DIR = os.path.join(_ROOT_DIR, ".hf_cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)

with open(_CONFIG_FILE, "r") as f:
    _config = json.load(f)

INFERENCE_MODE = _config.get("inference_mode", "api")
_api_config = _config.get("api", {})
_local_config = _config.get("local", {})
_common_config = _config.get("common", {})

# Get model name based on inference mode
if INFERENCE_MODE == "api":
    MODEL_NAME = _api_config.get("model", "qwen/qwen3-vl-30b-a3b-instruct")
    API_URL = _api_config.get("url", "https://openrouter.ai/api/v1/chat/completions")
    API_KEY = _api_config.get("key", "")
    API_MODEL = _api_config.get("model", MODEL_NAME)
    API_TIMEOUT_SECONDS = _api_config.get("timeout_seconds", 120)
    API_REASONING_EFFORT = _api_config.get("reasoning_effort", "none")
    API_REASONING_EXCLUDE = _api_config.get("reasoning_exclude", True)
else:
    MODEL_NAME = _local_config.get("model", "qwen/qwen3-vl-30b-a3b-instruct")
    API_URL = None
    API_KEY = None
    API_MODEL = None
    API_TIMEOUT_SECONDS = None
    API_REASONING_EFFORT = None
    API_REASONING_EXCLUDE = None

# Embedding model
EMBED_MODEL = _common_config.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Storage paths
DATA_DIR = os.path.join(_ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
CHROMA_COLLECTION = _common_config.get("chroma_collection", "qa_samples")
PROMPT_PATH = os.path.join(_CONFIG_DIR, "prompt.txt")

# Pipeline limits
TARGET_COUNT = _common_config.get("target_count", 100)
SAMPLES_PER_REQUEST = max(1, int(_common_config.get("samples_per_request", 1)))
MAX_ATTEMPTS = _common_config.get("max_attempts", TARGET_COUNT * 50)
MAX_TOKENS = _common_config.get("max_tokens", 2048)
TEMPERATURE = _common_config.get("temperature", 0.8)
TOP_P = _common_config.get("top_p", 0.95)

# Duplicate filtering
VECTOR_SIM_THRESHOLD = _common_config.get("vector_sim_threshold", 0.999999)

app = Flask(__name__, template_folder=".")
_pipeline_active = False
_pipeline_lock = threading.Lock()


class ChatData(BaseModel):
    user: str
    assistant: str


def _chat_item_json_schema() -> dict:
    schema = ChatData.model_json_schema()
    schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                prop["additionalProperties"] = False
    return schema


def chat_json_schema() -> dict:
    """JSON Schema for API structured output (same shape as local JsonSchemaParser)."""
    item_schema = _chat_item_json_schema()
    if SAMPLES_PER_REQUEST <= 1:
        return item_schema
    return {
        "type": "array",
        "items": item_schema,
        "minItems": SAMPLES_PER_REQUEST,
        "maxItems": SAMPLES_PER_REQUEST,
    }


def api_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chat_data",
            "strict": True,
            "schema": chat_json_schema(),
        },
    }


def check_json(raw: str):
    try:
        return True, ChatData.model_validate_json(raw), None
    except Exception as e:
        return False, None, str(e)


def validate_record(raw: str):
    ok_json, data, err = check_json(raw)
    if not ok_json:
        return False, None, None, "invalid_json", err

    ok_rule, reason = check_rules(data.user, data.assistant)
    if not ok_rule:
        return False, data.user, data.assistant, "bad_rule", reason

    return True, data.user, data.assistant, None, None


def build_embedder():
    return SentenceTransformer(EMBED_MODEL)


def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(1, -1)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def pair_embedding(embedder, user_text, assistant_text):
    text = pair_document(user_text, assistant_text)
    emb = embedder.encode([text])[0]
    return normalize(emb)


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n\r \t")


def build_instruction() -> str:
    instruction = load_prompt()
    if SAMPLES_PER_REQUEST > 1:
        instruction += (
            f"\n\nYêu cầu bổ sung: trả về đúng {SAMPLES_PER_REQUEST} cặp hội thoại "
            'trong một mảng JSON (array), mỗi phần tử là object có "user" và "assistant". '
            "Các cặp phải khác chủ đề/tình huống nhau."
        )
    return instruction


def split_raw_records(raw: str) -> list[str]:
    """Turn one model response into per-sample JSON strings."""
    if SAMPLES_PER_REQUEST <= 1:
        return [raw]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if not isinstance(data, list):
        return [raw]
    return [json.dumps(item, ensure_ascii=False) for item in data]


def format_chat_prompt(tokenizer, instruction: str) -> str:
    """Wrap the full instruction in Qwen ChatML so the model sees all of it."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_generator():
    if INFERENCE_MODE == "api":
        return None, None, None

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tok)

    parser = JsonSchemaParser(chat_json_schema())
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tok, parser)
    return pipe, prefix_fn, tok


def generate_with_api(prompt: str) -> str:
    if not API_KEY:
        raise ValueError(
            "API_KEY is required when INFERENCE_MODE='api'. "
            "Set generate_config.API_KEY."
        )

    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "response_format": api_response_format(),
        "reasoning": {
            "effort": API_REASONING_EFFORT,
            "exclude": API_REASONING_EXCLUDE,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        if e.code == 400 and "response_format" in payload:
            print(f"[API_ERROR] {e.code}: json_schema not supported, retrying without response_format")
            payload.pop("response_format", None)
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                API_URL,
                data=body,
                headers=req.headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
        else:
            print(f"[API_ERROR] {e.code}: {detail[:200]}")
            raise RuntimeError(f"API request failed: {e.code} {detail}") from e
    except urllib.error.URLError as e:
        print(f"[API_ERROR] Connection error: {e}")
        raise RuntimeError(f"API request error: {e}") from e

    if not raw_body.strip():
        print(f"[API_ERROR] Empty response body from {API_URL}")
        raise RuntimeError(f"API returned empty body (API_URL={API_URL})")
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as e:
        preview = raw_body[:200].replace("\n", " ")
        print(f"[API_ERROR] Invalid JSON response: {preview}")
        raise RuntimeError(f"API response is not JSON: {preview}") from e

    choices = data.get("choices") or []
    if not choices:
        print(f"[API_ERROR] No choices in response: {data.keys()}")
        raise RuntimeError(f"API response missing choices: {data}")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    model = data.get("model", API_MODEL)
    print(
        f"[API_WARN] Empty content (model={model}, finish_reason={finish!r}); retrying"
    )
    return ""


def generate_record(gen, prefix_fn, tokenizer) -> str:
    instruction = build_instruction()
    if INFERENCE_MODE == "api":
        return generate_with_api(instruction)

    chat_prompt = format_chat_prompt(tokenizer, instruction)
    return gen(
        chat_prompt,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        prefix_allowed_tokens_fn=prefix_fn,
        return_full_text=False,
    )[0]["generated_text"].strip()


def get_stats():
    with _pipeline_lock:
        pipeline_active = _pipeline_active

    chroma_exists = os.path.isdir(CHROMA_PATH)

    if pipeline_active and not chroma_exists:
        system_status = "LOADING MODEL (Please wait a few minutes...)"
    elif pipeline_active:
        system_status = "RUNNING - PIPELINE IS GENERATING DATA"
    else:
        system_status = "STOPPED - PIPELINE IS NOT ACTIVE"

    if not chroma_exists:
        return {
            "system_status": system_status,
            "pipeline_active": pipeline_active,
            "selected": 0,
            "duplicate": 0,
            "invalid_json": 0,
            "bad_rule": 0,
            "total": 0,
            "recent": [],
            "recent_duplicates": [],
        }

    store = QAStore()
    stats = store.get_stats()
    status_counts = stats["status_counts"]

    return {
        "system_status": system_status,
        "pipeline_active": pipeline_active,
        "selected": status_counts.get("selected", 0),
        "duplicate": status_counts.get("duplicate_context", 0),
        "invalid_json": status_counts.get("invalid_json", 0),
        "bad_rule": status_counts.get("bad_rule", 0),
        "total": stats["total"],
        "recent": stats["recent"],
        "recent_duplicates": stats["recent_duplicates"],
    }


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/")
def index():
    return render_template("index.html")


def start_dashboard(host: str, port: int) -> None:
    app.run(host=host, port=port, debug=False, use_reloader=False)


def run_pipeline(target: int) -> None:
    global _pipeline_active

    store = QAStore()

    with _pipeline_lock:
        _pipeline_active = True

    try:
        emb_model = build_embedder()
        gen, prefix_fn, tokenizer = build_generator()
        instruction = build_instruction()

        ok = store.count_selected()
        tries = 0
        print(f"[START] target={target}, selected={ok}")
        print(f"[BATCH] samples_per_request={SAMPLES_PER_REQUEST}")
        print(f"[MODE] Inference mode: {INFERENCE_MODE}")
        if INFERENCE_MODE == "api":
            print(f"[API] Model: {API_MODEL}, URL: {API_URL}")
        else:
            print(f"[LOCAL] Model: {MODEL_NAME}")
        print(f"[PROMPT] loaded {len(instruction)} chars from {PROMPT_PATH}")
        print("=" * 60)

        while ok < target and tries < MAX_ATTEMPTS:
            tries += 1

            raw = generate_record(gen, prefix_fn, tokenizer)
            record_raws = split_raw_records(raw)

            if SAMPLES_PER_REQUEST > 1 and len(record_raws) != SAMPLES_PER_REQUEST:
                store.insert_sample(
                    None,
                    None,
                    "invalid_json",
                    raw,
                    f"expected {SAMPLES_PER_REQUEST} items, got {len(record_raws)}",
                )
                print(
                    f"[INVALID_JSON] Attempt {tries}: "
                    f"expected {SAMPLES_PER_REQUEST} items, got {len(record_raws)}"
                )
                continue

            for item_raw in record_raws:
                if ok >= target:
                    break

                valid, user_text, assistant_text, status, reason = validate_record(item_raw)
                if not valid:
                    if status == "invalid_json":
                        store.insert_sample(None, None, "invalid_json", item_raw, reason)
                        print(f"[INVALID_JSON] Attempt {tries}: {reason}")
                        if item_raw and len(item_raw) < 500:
                            print(f"  Raw: {item_raw}")
                    else:
                        store.insert_sample(user_text, assistant_text, "bad_rule", item_raw, reason)
                        print(f"[BAD_RULE] Attempt {tries}: {reason}")
                        if user_text:
                            print(f"  User: {user_text[:100]}...")
                        if assistant_text:
                            print(f"  Assistant: {assistant_text[:100]}...")
                    continue

                vec = pair_embedding(emb_model, user_text, assistant_text)

                dup, score, nearest_id = store.is_duplicate(vec)
                if dup:
                    store.insert_sample(
                        user_text,
                        assistant_text,
                        "duplicate_context",
                        item_raw,
                        f"sim={score:.4f}",
                        nearest_selected_id=nearest_id,
                    )
                    print(f"[DUPLICATE] Attempt {tries}: Similarity={score:.6f}, Nearest ID={nearest_id}")
                    print(f"  User: {user_text[:100]}...")
                    continue

                store.insert_sample(
                    user_text,
                    assistant_text,
                    "selected",
                    item_raw,
                    None,
                    embedding=vec[0],
                )
                ok += 1
                print(f"[OK] {ok}/{target} | User: {user_text[:80]}...")
    finally:
        with _pipeline_lock:
            _pipeline_active = False
        stats = store.get_stats() if os.path.isdir(CHROMA_PATH) else {}
        store.close()
        print("\n" + "=" * 60)
        print("[DONE] Pipeline completed")
        print("=" * 60)
        if stats:
            status_counts = stats.get("status_counts", {})
            print(f"✓ Selected:         {status_counts.get('selected', 0)}")
            print(f"✗ Invalid JSON:     {status_counts.get('invalid_json', 0)}")
            print(f"✗ Bad Rules:        {status_counts.get('bad_rule', 0)}")
            print(f"✗ Duplicates:       {status_counts.get('duplicate_context', 0)}")
            print(f"Total Attempts:     {tries}")
            print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_COUNT)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        args=("0.0.0.0", args.port),
        daemon=True,
    )
    dashboard_thread.start()
    print(f"[DASHBOARD] http://0.0.0.0:{args.port}")

    run_pipeline(args.target)
