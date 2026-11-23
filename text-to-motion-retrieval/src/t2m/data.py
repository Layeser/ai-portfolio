"""Motion / text data: a held-out split that the model never sees, and challenge-style query groups.

The original data-challenge "val" folders reuse motions that are also in the training list
(checked by hashing: 640/640 sampled candidates were training motions). Scores on them measure
memorisation, not retrieval. Here a seeded 10 % of motion ids is held out *before* training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("T2M_DATA_DIR", ROOT / "data"))
MAX_FRAMES = 196
SEED = 42


def motion_ids(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(p.stem for p in (data_dir / "motions").glob("*.npy"))


def load_texts(motion_id: str, data_dir: Path = DATA_DIR) -> list[str]:
    lines = (data_dir / "texts" / f"{motion_id}.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def split_ids(ids: list[str], holdout: float = 0.1, seed: int = SEED) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    n_hold = int(len(ids) * holdout)
    return sorted(shuffled[n_hold:]), sorted(shuffled[:n_hold])


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, ids: list[str], data_dir: Path = DATA_DIR) -> Normalizer:
        """Per-feature mean / std on training motions only (no statistics from held-out data)."""
        total, total_sq, count = 0.0, 0.0, 0
        for mid in ids:
            m = np.load(data_dir / "motions" / f"{mid}.npy").astype(np.float64)
            total = total + m.sum(0)
            total_sq = total_sq + (m**2).sum(0)
            count += len(m)
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-8))
        return cls(mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32))

    def __call__(self, motion: np.ndarray) -> np.ndarray:
        return (motion - self.mean) / self.std

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> Normalizer:
        z = np.load(path)
        return cls(z["mean"], z["std"])


def crop(motion: np.ndarray, max_frames: int, rng: np.random.Generator | None) -> np.ndarray:
    if len(motion) <= max_frames:
        return motion
    start = rng.integers(0, len(motion) - max_frames + 1) if rng is not None else (len(motion) - max_frames) // 2
    return motion[start:start + max_frames]


class MotionTextDataset(Dataset):
    """One item = one motion with one of its descriptions (random in training, the first at eval)."""

    def __init__(self, ids: list[str], normalizer: Normalizer, train: bool, data_dir: Path = DATA_DIR,
                 max_frames: int = MAX_FRAMES, noise_std: float = 0.02):
        self.ids, self.norm, self.train, self.data_dir = ids, normalizer, train, data_dir
        self.max_frames, self.noise_std = max_frames, noise_std
        self.texts = {mid: load_texts(mid, data_dir) for mid in ids}
        self.rng = np.random.default_rng(SEED)
        # Motions are small (~1 GB total in float32): cache them as float16 to keep RAM low.
        self.motions = {mid: self.norm(np.load(data_dir / "motions" / f"{mid}.npy")).astype(np.float16) for mid in ids}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        mid = self.ids[i]
        motion = crop(self.motions[mid].astype(np.float32), self.max_frames, self.rng if self.train else None)
        if self.train and self.noise_std:
            motion = motion + self.rng.normal(0, self.noise_std, motion.shape).astype(np.float32)
        texts = self.texts[mid]
        text = texts[self.rng.integers(len(texts))] if self.train else texts[0]
        return torch.from_numpy(motion), text, mid


def collate(batch):
    motions, texts, ids = zip(*batch, strict=True)
    lengths = torch.tensor([len(m) for m in motions])
    padded = torch.zeros(len(motions), int(lengths.max()), motions[0].shape[1])
    for i, m in enumerate(motions):
        padded[i, : len(m)] = m
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]  # True = real frame
    return padded, mask, list(texts), list(ids)


def query_groups(ids: list[str], n_candidates: int = 32, seed: int = SEED) -> list[tuple[str, list[str]]]:
    """Challenge-style protocol on held-out motions: each motion is the target of one query,
    hidden among `n_candidates - 1` random held-out distractors."""
    rng = np.random.default_rng(seed)
    n_candidates = min(n_candidates, len(ids))  # tiny datasets (tests) have fewer motions
    groups = []
    for target in ids:
        distractors = rng.choice([m for m in ids if m != target], size=n_candidates - 1, replace=False)
        candidates = list(distractors) + [target]
        rng.shuffle(candidates)
        groups.append((target, candidates))
    return groups
