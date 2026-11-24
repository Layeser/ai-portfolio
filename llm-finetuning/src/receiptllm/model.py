"""Model loading and batched greedy generation (base model, or base + LoRA adapter)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from receiptllm.data import messages

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def load(base: str = BASE, adapter: str | None = None, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model


def prompt(tok, text: str) -> str:
    return tok.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, texts: list[str], batch_size: int = 8, max_new_tokens: int = 768) -> list[str]:
    # longest first: batches of similar length waste less padding
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    results: dict[int, str] = {}
    for k in range(0, len(order), batch_size):
        idx = order[k : k + batch_size]
        enc = tok([prompt(tok, texts[i]) for i in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, out[:, enc["input_ids"].shape[1] :], strict=True):
            results[i] = tok.decode(seq, skip_special_tokens=True)
    outputs = [results[i] for i in range(len(texts))]
    return outputs
