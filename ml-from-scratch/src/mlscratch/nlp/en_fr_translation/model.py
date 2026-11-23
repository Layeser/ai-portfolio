"""
Tiny Encoder–Decoder Transformer for EN→FR.

Autoregressive decoding: at each step the decoder predicts the next French token
conditioned on encoded English + previously generated French tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.proj = nn.Linear(d_model, tgt_vocab)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == self.pad_id

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """
        src:    (B, S)
        tgt_in: (B, T)  teacher-forced input (BOS + tokens[:-1])
        returns logits (B, T, V)
        """
        src_emb = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=self._pad_mask(src))
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), diagonal=1)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=self._pad_mask(tgt_in),
            memory_key_padding_mask=self._pad_mask(src),
        )
        return self.proj(out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 32,
    ) -> list[int]:
        """Autoregressive greedy decode for a single source sequence (1, S)."""
        self.eval()
        device = src.device
        src_emb = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=self._pad_mask(src))
        ys = torch.tensor([[bos_id]], device=device)
        for _ in range(max_len):
            tgt_emb = self.pos(self.tgt_emb(ys) * math.sqrt(self.d_model))
            T = ys.size(1)
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=self._pad_mask(src),
            )
            next_id = self.proj(out[:, -1:]).argmax(-1)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return ys[0].tolist()
