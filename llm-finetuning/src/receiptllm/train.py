"""LoRA fine-tuning (PEFT) of a small instruct LLM on receipt -> JSON extraction.

Loss on the answer tokens only (the system + user prompt is masked). The checkpoint kept is the epoch
with the best field F1 on the validation split; the test split is scored once, by evaluate.py.

    python -m receiptllm.train --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from receiptllm.data import ROOT, load, messages, target_json
from receiptllm.metrics import parse_json, score
from receiptllm.model import BASE, generate

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, record: dict, max_len: int) -> dict:
    """input_ids and labels, with -100 on the prompt so that only the JSON answer is learned."""
    prompt = tok.apply_chat_template(messages(record["text"]), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target_json(record["target"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
        labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
        mask[i, : len(b["input_ids"])] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="artifacts/lora")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    out = ROOT / args.out

    if not torch.cuda.is_available():
        raise SystemExit("LoRA training needs a CUDA GPU; inference and the API run on CPU (make download-weights)")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)", flush=True)

    train = [encode(tok, r, args.max_len) for r in load("train")]
    val = load("validation")
    steps = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    history, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # length-bucketed shuffling: similar lengths in a batch, random batch order
        order = sorted(range(len(train)), key=lambda i: len(train[i]["input_ids"]) + random.random() * 64)
        batches = [order[k : k + args.batch] for k in range(0, len(order), args.batch)]
        random.shuffle(batches)
        running = 0.0
        for step, idx in enumerate(batches, 1):
            b = {k: v.cuda() for k, v in collate([train[i] for i in idx], tok.pad_token_id).items()}
            loss = model(**b).loss / args.accum
            loss.backward()
            running += loss.item()
            if step % args.accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(batches)} loss {running * args.accum / step:.4f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        model.eval()
        model.config.use_cache = True  # checkpointing turns the KV cache off; generation needs it
        tok.padding_side = "left"
        preds = [parse_json(o) for o in generate(tok, model, [r["text"] for r in val], batch_size=16)]
        tok.padding_side = "right"
        s = score(preds, [r["target"] for r in val])
        record = {"epoch": epoch, "train_loss": round(running * args.accum / len(batches), 4),
                  **{k: s[k] for k in ("field_f1", "doc_accuracy", "valid_json")},
                  "minutes": round((time.time() - t0) / 60, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if s["field_f1"] > best:
            best = s["field_f1"]
            model.save_pretrained(out)
            tok.save_pretrained(out)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "train_history.json").write_text(json.dumps(
        {"args": vars(args), "trainable_params": trainable, "total_params": total, "history": history}, indent=2))


if __name__ == "__main__":
    main()
