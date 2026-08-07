"""
Train a tiny EN→FR Transformer on our handcrafted parallel corpus.

Usage:
    python -m nlp.en_fr_translation.train
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.en_fr_translation.data import (
    BOS,
    EOS,
    PAD,
    PAIRS,
    build_vocab,
    decode,
    encode,
)
from nlp.en_fr_translation.model import Seq2SeqTransformer
from utils.common import set_seed


def collate(batch, pad_id: int):
    srcs, tgts = zip(*batch)
    src = pad_sequence([torch.tensor(s) for s in srcs], batch_first=True, padding_value=pad_id)
    tgt = pad_sequence([torch.tensor(t) for t in tgts], batch_first=True, padding_value=pad_id)
    # teacher forcing: input = tgt[:-1], target = tgt[1:]
    return src, tgt[:, :-1], tgt[:, 1:]


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en_stoi, en_itos = build_vocab([e for e, _ in PAIRS])
    fr_stoi, fr_itos = build_vocab([f for _, f in PAIRS])
    pad_id = en_stoi[PAD]
    assert fr_stoi[PAD] == pad_id

    # Tiny handcrafted corpus → train on all pairs (teaching demo, not a benchmark).
    train = [
        (encode(e, en_stoi, add_bos_eos=False), encode(f, fr_stoi, add_bos_eos=True))
        for e, f in PAIRS
    ]

    model = Seq2SeqTransformer(
        src_vocab=len(en_stoi),
        tgt_vocab=len(fr_stoi),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        pad_id=pad_id,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    epochs, batch_size = 100, 16
    print(f"device={device} | pairs={len(train)} | V_en={len(en_stoi)} V_fr={len(fr_stoi)}")

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        total, n = 0.0, 0
        for i in range(0, len(train), batch_size):
            batch = train[i : i + batch_size]
            src, tgt_in, tgt_out = collate(batch, pad_id)
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * src.size(0)
            n += src.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss={total / n:.4f}")

    print("\n=== Autoregressive decode (greedy) ===")
    model.eval()
    demos = [
        "hello",
        "thank you",
        "i love machine learning",
        "where is the station",
        "the cat is black",
        "i speak french",
        "neural networks are powerful",
        "transformers changed nlp",
    ]
    for en in demos:
        src = torch.tensor([encode(en, en_stoi, add_bos_eos=False)], device=device)
        ids = model.generate(src, bos_id=fr_stoi[BOS], eos_id=fr_stoi[EOS])
        print(f"EN: {en}")
        print(f"FR: {decode(ids, fr_itos)}\n")


if __name__ == "__main__":
    main()
