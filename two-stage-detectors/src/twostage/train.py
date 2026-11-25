"""Train one detector on VOC 2007 trainval (minus a 300-image holdout), monitoring holdout mAP.

    python -m twostage.train --model faster_rcnn --epochs 12

SGD with momentum, linear warmup, step decay (x0.1) at 75 % of the epochs, bf16 autocast.
After every epoch: checkpoints/<model>/last.pt, holdout mAP, and best.pt when it improves.
History in reports/history_<model>.json (the test set is never used here).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from twostage.evaluate import _to, evaluate_model
from twostage.models import build
from twostage.voc import ROOT, VOCDetection, collate, split_ids

DEFAULTS = {  # (lr, images per batch, epochs) per model
    "faster_rcnn": (0.005, 2, 12),
    "fast_rcnn": (0.005, 2, 8),
    "rcnn": (0.005, 2, 4),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(DEFAULTS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=0, help="stop early (smoke test)")
    parser.add_argument("--holdout-images", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    lr, batch, epochs = DEFAULTS[args.model]
    lr, batch, epochs = args.lr or lr, args.batch or batch, args.epochs or epochs

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU; to use the models on CPU, run make download-weights and evaluate")
    device = "cuda"
    model = build(args.model).to(device)
    dataset = VOCDetection(split_ids("train"), train=True, proposals=model.needs_proposals)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=args.workers,
                        drop_last=True, persistent_workers=True)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0}],
                          lr=lr, momentum=0.9)
    iters_per_epoch, warmup = len(loader), 500
    step_at = int(0.75 * epochs) * iters_per_epoch

    def lr_at(it: int) -> float:
        factor = min(1.0, (it + 1) / warmup) * (0.1 if it >= step_at else 1.0)
        return lr * factor

    out_dir = ROOT / "checkpoints" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / "reports" / f"history_{args.model}.json"
    history, start_epoch, best = [], 0, -1.0
    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch, best = ckpt["epoch"], ckpt["best"]
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        print(f"resumed at epoch {start_epoch}, best holdout mAP {best:.4f}", flush=True)

    it = start_epoch * iters_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        sums, count, t0 = {}, 0, time.time()
        for images, targets in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [_to(t, device) for t in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at iter {it}: { {k: v.item() for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            count += 1
            it += 1
            if it % 200 == 0:
                avg = {k: round(v / count, 4) for k, v in sums.items()}
                print(f"epoch {epoch + 1} iter {it} lr {lr_at(it):.5f} {avg} "
                      f"{(time.time() - t0) / count:.3f}s/it", flush=True)
            if args.max_iters and it >= args.max_iters:
                break
        train_minutes = (time.time() - t0) / 60
        result = evaluate_model(model, "holdout", args.holdout_images, device, workers=args.workers)
        record = {"epoch": epoch + 1, "train_loss": {k: round(v / max(count, 1), 4) for k, v in sums.items()},
                  "holdout_mAP07": result["mAP07"], "holdout_mAP": result["mAP"],
                  "train_minutes": round(train_minutes, 1), "eval_s_per_image": result["seconds_per_image"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                 "best": max(best, result["mAP07"])}
        torch.save(state, out_dir / "last.pt")
        if result["mAP07"] > best:
            best = result["mAP07"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "holdout_mAP07": best}, out_dir / "best.pt")
            print(f"  new best holdout mAP07 {best:.4f}", flush=True)
        history_path.write_text(json.dumps(history, indent=2))
        if args.max_iters and it >= args.max_iters:
            break


if __name__ == "__main__":
    main()
