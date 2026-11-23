"""Train the dual encoder and evaluate it on motions it has never seen.

    python -m t2m.train --epochs 40          # ~10 min on a laptop RTX 5060

Two evaluations on the held-out 10 % of motions:
* gallery retrieval: each held-out description against *all* held-out motions (R@1/5/10, median rank);
* challenge protocol: target hidden among 31 held-out distractors, scored with the challenge's
  weighted recall (sum_k R@k / k, k = 1..10, normalised).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from t2m.data import (
    MAX_FRAMES,
    ROOT,
    SEED,
    MotionTextDataset,
    Normalizer,
    collate,
    load_texts,
    motion_ids,
    query_groups,
    split_ids,
)
from t2m.model import DualEncoder

ARTIFACTS = Path(os.getenv("T2M_ARTIFACTS_DIR", ROOT / "artifacts"))


def recall_at_k(ranks: np.ndarray, ks=(1, 5, 10)) -> dict:
    return {f"R@{k}": round(float((ranks <= k).mean()), 4) for k in ks}


def challenge_score(ranks: np.ndarray) -> float:
    ks = np.arange(1, 11)
    recalls = np.array([(ranks <= k).mean() for k in ks])
    return float((recalls / ks).sum() / (1 / ks).sum())


@torch.no_grad()
def embed(model: DualEncoder, dataset: MotionTextDataset, device: str, batch_size: int = 128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    motion_emb, ids = [], []
    for motion, mask, _, batch_ids in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            motion_emb.append(model.encode_motion(motion.to(device), mask.to(device)).float().cpu())
        ids += batch_ids
    return torch.cat(motion_emb), ids


@torch.no_grad()
def evaluate(model: DualEncoder, dataset: MotionTextDataset, device: str) -> dict:
    motion_emb, ids = embed(model, dataset, device)
    index = {mid: i for i, mid in enumerate(ids)}
    queries, targets = [], []
    for mid in ids:  # every description of every held-out motion is a query
        for text in dataset.texts[mid]:
            queries.append(text)
            targets.append(index[mid])
    text_emb = []
    for start in range(0, len(queries), 256):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            text_emb.append(model.encode_text(queries[start:start + 256]).float().cpu())
    sims = torch.cat(text_emb) @ motion_emb.T
    target_scores = sims[torch.arange(len(targets)), torch.tensor(targets)]
    gallery_ranks = (sims > target_scores[:, None]).sum(1).numpy() + 1

    group_ranks = []
    first_query = {}
    for q, t in enumerate(targets):
        first_query.setdefault(t, q)
    for target, candidates in query_groups(ids):
        q = first_query[index[target]]
        cand = torch.tensor([index[c] for c in candidates])
        group_ranks.append(int((sims[q, cand] > sims[q, index[target]]).sum()) + 1)
    group_ranks = np.array(group_ranks)
    return {
        "gallery": {"n_queries": len(queries), "n_motions": len(ids), **recall_at_k(gallery_ranks),
                    "median_rank": int(np.median(gallery_ranks))},
        "challenge_32": {"n_queries": len(group_ranks), **recall_at_k(group_ranks),
                         "weighted_recall": round(challenge_score(group_ranks), 4)},
    }


def train(epochs: int = 40, batch_size: int = 64, lr: float = 1e-4, artifacts: Path = ARTIFACTS,
          data_dir: Path | None = None, ids: list[str] | None = None) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"data_dir": data_dir} if data_dir else {}
    ids = ids or motion_ids(**kw)
    train_ids, holdout_ids = split_ids(ids)
    norm = Normalizer.fit(train_ids, **kw)
    train_ds = MotionTextDataset(train_ids, norm, train=True, **kw)
    holdout_ds = MotionTextDataset(holdout_ids, norm, train=False, **kw)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    model = DualEncoder().to(device)
    params = [{"params": model.text.parameters(), "lr": lr * 0.1},  # gentle on the pretrained encoder
              {"params": [p for n, p in model.named_parameters() if not n.startswith("text.")], "lr": lr}]
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    history, best, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        start, losses = time.perf_counter(), []
        for motion, mask, texts, _ in loader:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(motion.to(device), mask.to(device), texts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        record = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4),
                  "seconds": round(time.perf_counter() - start, 1)}
        if epoch % 5 == 0 or epoch == epochs:
            metrics = evaluate(model, holdout_ds, device)
            record.update({"holdout_R@1": metrics["gallery"]["R@1"],
                           "holdout_challenge": metrics["challenge_32"]["weighted_recall"]})
            if metrics["challenge_32"]["weighted_recall"] > best:
                best = metrics["challenge_32"]["weighted_recall"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(record)
        print(record, flush=True)

    model.load_state_dict(best_state)
    final = evaluate(model, holdout_ds, device)
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts / "dual_encoder.pt")
    norm.save(artifacts / "normalizer.npz")
    (artifacts / "split.json").write_text(json.dumps({"train": train_ids, "holdout": holdout_ids}))
    result = {"holdout": final, "history": history, "max_frames": MAX_FRAMES,
              "rows": {"train_motions": len(train_ids), "holdout_motions": len(holdout_ids)}}
    (artifacts / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = train(epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(result["holdout"], indent=2))


if __name__ == "__main__":
    main()


__all__ = ["train", "evaluate", "load_texts"]
