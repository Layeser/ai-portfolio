"""
Train a tiny EN→FR Transformer on the handcrafted corpus and measure it on held-out sentences.

Usage:
    python -m mlscratch.nlp.en_fr_translation.train
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from mlscratch.nlp.en_fr_translation.data import BOS, EOS, PAD, build_vocab, decode, encode, train_test_split
from mlscratch.nlp.en_fr_translation.model import Seq2SeqTransformer
from mlscratch.utils.common import set_seed


def collate(batch, pad_id: int):
    srcs, tgts = zip(*batch, strict=True)
    src = pad_sequence([torch.tensor(s) for s in srcs], batch_first=True, padding_value=pad_id)
    tgt = pad_sequence([torch.tensor(t) for t in tgts], batch_first=True, padding_value=pad_id)
    # teacher forcing: input = tgt[:-1], target = tgt[1:]
    return src, tgt[:, :-1], tgt[:, 1:]


@dataclass
class Translator:
    model: Seq2SeqTransformer
    en_stoi: dict[str, int]
    fr_stoi: dict[str, int]
    fr_itos: dict[int, str]
    history: dict[str, list[float]] = field(default_factory=dict)

    def translate(self, sentence: str) -> str:
        device = next(self.model.parameters()).device
        src = torch.tensor([encode(sentence, self.en_stoi, add_bos_eos=False)], device=device)
        ids = self.model.generate(src, bos_id=self.fr_stoi[BOS], eos_id=self.fr_stoi[EOS])
        return decode(ids, self.fr_itos)


def exact_match(translator: Translator, pairs: list[tuple[str, str]]) -> float:
    """Share of sentences translated word for word as the reference (greedy decoding)."""
    return sum(translator.translate(e) == f for e, f in pairs) / max(len(pairs), 1)


def train(
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]] | None = None,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 3e-4,
    label_smoothing: float = 0.1,
    eval_every: int = 10,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> Translator:
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # The vocabulary comes from the training pairs only: a held-out word would be <unk>, which train_test_split avoids.
    en_stoi, _ = build_vocab([e for e, _ in train_pairs])
    fr_stoi, fr_itos = build_vocab([f for _, f in train_pairs])
    pad_id = en_stoi[PAD]
    assert fr_stoi[PAD] == pad_id

    data = [(encode(e, en_stoi, add_bos_eos=False), encode(f, fr_stoi)) for e, f in train_pairs]
    model = Seq2SeqTransformer(len(en_stoi), len(fr_stoi), pad_id=pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Label smoothing: the target keeps 90 % of the mass, the rest is spread over the vocabulary, which stops the
    # model from pushing logits to infinity on a corpus small enough to memorise.
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    translator = Translator(model, en_stoi, fr_stoi, fr_itos, {"epoch": [], "loss": [], "train_em": [], "test_em": []})
    rng = random.Random(seed)

    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(data)
        total, n = 0.0, 0
        for i in range(0, len(data), batch_size):
            src, tgt_in, tgt_out = (t.to(device) for t in collate(data[i : i + batch_size], pad_id))
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * src.size(0)
            n += src.size(0)
        if ep % eval_every == 0 or ep == 1:
            h = translator.history
            h["epoch"].append(ep)
            h["loss"].append(total / n)
            h["train_em"].append(exact_match(translator, train_pairs))
            h["test_em"].append(exact_match(translator, test_pairs) if test_pairs else float("nan"))
            if verbose:
                print(f"epoch {ep:3d}  loss={h['loss'][-1]:.3f}  train EM={h['train_em'][-1]:.2f}"
                      f"  held-out EM={h['test_em'][-1]:.2f}")
    return translator


def main():
    train_pairs, test_pairs = train_test_split()
    print(f"train={len(train_pairs)} pairs | held-out={len(test_pairs)} pairs")
    translator = train(train_pairs, test_pairs)

    print("\n=== Held-out sentences (never seen in training), greedy decoding ===")
    for en, fr in test_pairs[:10]:
        out = translator.translate(en)
        print(f"{'ok ' if out == fr else 'err'}  {en:28s} → {out:28s} (ref: {fr})")


if __name__ == "__main__":
    main()
