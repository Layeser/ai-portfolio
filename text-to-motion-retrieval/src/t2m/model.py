"""Dual encoder (CLIP-style): a pretrained sentence encoder for text, a Transformer for motion,
both projected to a shared, L2-normalised embedding space and trained with symmetric InfoNCE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(self, in_dim: int = 384, d_model: int = 256, layers: int = 4, heads: int = 4,
                 dropout: float = 0.1, max_frames: int = 512):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """motion: (B, T, in_dim), mask: (B, T) True for real frames -> (B, d_model) masked mean."""
        x = self.input(motion) + self.pos[:, : motion.shape[1]]
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)


class TextEncoder(nn.Module):
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 64):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_length = max_length
        self.out_dim = self.model.config.hidden_size

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(device)
        hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class DualEncoder(nn.Module):
    def __init__(self, text_model: str = "sentence-transformers/all-MiniLM-L6-v2", embed_dim: int = 256,
                 motion_dim: int = 384, d_model: int = 256, layers: int = 4):
        super().__init__()
        self.text = TextEncoder(text_model)
        self.motion = MotionEncoder(motion_dim, d_model, layers)
        self.text_proj = nn.Linear(self.text.out_dim, embed_dim)
        self.motion_proj = nn.Linear(d_model, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self.text_proj(self.text(texts)), dim=-1)

    def encode_motion(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.motion_proj(self.motion(motion, mask)), dim=-1)

    def forward(self, motion, mask, texts) -> torch.Tensor:
        return clip_loss(self.encode_text(texts), self.encode_motion(motion, mask), self.logit_scale)


def clip_loss(text_emb: torch.Tensor, motion_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: each text must pick its motion among the batch, and vice versa."""
    logits = logit_scale.clamp(max=math.log(100)).exp() * text_emb @ motion_emb.T
    target = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
