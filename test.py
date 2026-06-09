"""Interactive chat — base model + LoRA adapter from train_config."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.join(_ROOT, "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

import train_config  # noqa: F401 — HF_HOME before huggingface imports

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_IM_END = "<|" + "im_end" + "|>"


def _adapter_dir() -> str:
    return os.path.normpath(train_config.OUTPUT_DIR)


def _has_adapter(adapter_dir: str) -> bool:
    return os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))


def _format_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    parts: list[str] = []
    for user_msg, assistant_msg in history:
        parts.append(
            f"<|im_start|>user\n{user_msg}{_IM_END}\n"
            f"<|im_start|>assistant\n{assistant_msg}{_IM_END}\n"
        )
    parts.append(f"<|im_start|>user\n{user_text}{_IM_END}\n<|im_start|>assistant\n")
    return "".join(parts)


def load_model_and_tokenizer():
    adapter_dir = _adapter_dir()
    has_lora = _has_adapter(adapter_dir)

    print(f"[TEST] Base model: {train_config.MODEL_NAME}")
    if has_lora:
        print(f"[TEST] LoRA adapter: {adapter_dir}")
    else:
        print(f"[TEST] No LoRA at {adapter_dir} — using base model only.")

    tokenizer = AutoTokenizer.from_pretrained(train_config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if has_lora:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_reply(
    model,
    tokenizer,
    history: list[tuple[str, str]],
    user_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = _format_prompt(history, user_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if _IM_END in text:
        text = text.split(_IM_END)[0]
    return text.strip()


def run_chat(max_new_tokens: int, temperature: float) -> None:
    model, tokenizer = load_model_and_tokenizer()
    history: list[tuple[str, str]] = []

    print("\n--- Test chat (train_config model) ---")
    print("Lệnh: /exit hoặc /quit để thoát, /clear xóa lịch sử hội thoại.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[TEST] Bye.")
            break

        if not user_text:
            continue
        lower = user_text.lower()
        if lower in ("/exit", "/quit", "exit", "quit"):
            print("[TEST] Bye.")
            break
        if lower == "/clear":
            history.clear()
            print("[TEST] History cleared.")
            continue

        reply = generate_reply(
            model,
            tokenizer,
            history,
            user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        history.append((user_text, reply))
        print(f"Assistant: {reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test chat with fine-tuned LoRA model")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    run_chat(max_new_tokens=args.max_new_tokens, temperature=args.temperature)


if __name__ == "__main__":
    main()
