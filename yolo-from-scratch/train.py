#!/usr/bin/env python3
"""Train YOLOv1 on a detection dataset.

Usage:
    python train.py --config config/train.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path so src/ and utils/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loss import YOLOv1Loss
from src.model import YOLOv1
from utils.dataset import YOLODataset
from utils.helpers import get_device, load_checkpoint, load_config, save_checkpoint, set_seed
from utils.voc import prepare_voc_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv1")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Force download and prepare PASCAL VOC before training",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (defaults to checkpoints/<save_dir>/last.pt)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint and start training from scratch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from the config",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["custom", "pretrained_resnet18", "pretrained_resnet50"],
        help="Backbone variant to use for the detector",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Use pretrained torchvision weights for the selected backbone",
    )
    return parser.parse_args()


def ensure_dataset(cfg: dict, project_root: Path, force_prepare: bool = False) -> None:
    """Download and convert PASCAL VOC if train/val folders are missing."""
    train_root = project_root / cfg["data"]["train_root"]
    voc_cfg = cfg.get("data", {}).get("voc", {})

    needs_prepare = force_prepare or not train_root.exists() or not any(train_root.glob("images/*"))
    if not needs_prepare:
        return

    if not force_prepare and not voc_cfg.get("auto_prepare", True):
        raise FileNotFoundError(
            f"Training data not found at {train_root}. "
            "Run: python prepare_voc.py"
        )

    print("Preparing PASCAL VOC dataset...")
    output_dir = project_root / voc_cfg.get("output_dir", "data/voc")
    max_train = voc_cfg.get("max_train_samples", 2000)
    max_val = voc_cfg.get("max_val_samples", 500)
    metadata = prepare_voc_dataset(
        output_dir=output_dir,
        years=[str(y) for y in voc_cfg.get("years", ["2007"])],
        max_train_samples=None if max_train == 0 else max_train,
        max_val_samples=None if max_val == 0 else max_val,
        selected_classes=voc_cfg.get("classes"),
        force_download=voc_cfg.get("force_download", False),
    )

    # Sync num_classes when using a class subset
    if metadata["num_classes"] != cfg["model"]["num_classes"]:
        print(
            f"  Updating num_classes: {cfg['model']['num_classes']} -> {metadata['num_classes']}"
        )
        cfg["model"]["num_classes"] = metadata["num_classes"]


def compute_lr(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr: float,
) -> float:
    """Learning rate for a given (1-indexed) epoch: linear warmup then cosine decay."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def resolve_start_epoch(checkpoint_epoch: int | None, total_epochs: int, requested_start_epoch: int | None = None) -> int:
    """Resolve the first epoch to train from scratch or after a checkpoint."""
    if requested_start_epoch is not None:
        return requested_start_epoch
    if checkpoint_epoch is None:
        return 1
    return checkpoint_epoch + 1


def train_one_epoch(
    model: YOLOv1,
    loader: DataLoader,
    criterion: YOLOv1Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    grad_clip: float | None = None,
) -> float:
    """Run a single training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for _, (images, targets) in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss, components = criterion(predictions, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{components['loss']:.4f}",
            coord=f"{components['coord']:.4f}",
            cls=f"{components['class']:.4f}",
        )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: YOLOv1,
    loader: DataLoader,
    criterion: YOLOv1Loss,
    device: torch.device,
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images)
        loss, _ = criterion(predictions, targets)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    ensure_dataset(cfg, PROJECT_ROOT, force_prepare=args.prepare_data)

    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    # --- Model ---
    backbone_type = args.backbone or cfg["model"].get("backbone", "custom")
    pretrained_backbone = args.pretrained_backbone or bool(cfg["model"].get("pretrained_backbone", False))

    model = YOLOv1(
        num_classes=cfg["model"]["num_classes"],
        num_boxes=cfg["model"]["num_boxes"],
        grid_size=cfg["model"]["grid_size"],
        backbone_type=backbone_type,
        pretrained_backbone=pretrained_backbone,
    ).to(device)

    # --- Loss ---
    criterion = YOLOv1Loss(
        grid_size=cfg["model"]["grid_size"],
        num_boxes=cfg["model"]["num_boxes"],
        num_classes=cfg["model"]["num_classes"],
        lambda_coord=cfg["train"]["lambda_coord"],
        lambda_noobj=cfg["train"]["lambda_noobj"],
    )

    # --- Optimizer (SGD as in original paper) ---
    # AdamW by default: on 16 real VOC images, SGD (lr 1e-3) reached mAP 0.06 after 200 steps
    # while AdamW reached 0.96 -- the 50M randomly initialised neck/head parameters need it.
    if cfg["train"].get("optimizer", "adamw") == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"]
        )

    # --- Data ---
    train_dataset = YOLODataset(
        root=PROJECT_ROOT / cfg["data"]["train_root"],
        image_size=cfg["data"]["image_size"],
        grid_size=cfg["model"]["grid_size"],
        num_classes=cfg["model"]["num_classes"],
        augment=cfg["data"]["augment"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = None
    val_root = PROJECT_ROOT / cfg["data"]["val_root"]
    if val_root.exists():
        val_dataset = YOLODataset(
            root=val_root,
            image_size=cfg["data"]["image_size"],
            grid_size=cfg["model"]["grid_size"],
            num_classes=cfg["model"]["num_classes"],
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training loop ---
    save_dir = PROJECT_ROOT / cfg["train"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    total_epochs = cfg["train"]["epochs"] if args.epochs is None else args.epochs
    start_epoch = 1
    resume_path: Path | None = None

    if not args.fresh:
        if args.resume_from is not None:
            resume_path = PROJECT_ROOT / args.resume_from
        else:
            resume_path = save_dir / "last.pt"
        if resume_path.exists():
            checkpoint = load_checkpoint(resume_path, model, optimizer=optimizer, device=device)
            start_epoch = resolve_start_epoch(checkpoint.get("epoch"), total_epochs)
            print(f"Resuming from {resume_path} at epoch {checkpoint.get('epoch', 0)}")
        else:
            print(f"No checkpoint found at {resume_path}; starting from scratch.")

    if args.fresh:
        print("Starting training from scratch (--fresh).")

    base_lr = cfg["train"]["learning_rate"]
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    min_lr = float(cfg["train"].get("min_learning_rate", base_lr))

    for epoch in range(start_epoch, total_epochs + 1):
        lr = compute_lr(epoch, total_epochs, base_lr, warmup_epochs, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print(f"\n=== Epoch {epoch}/{total_epochs} (lr={lr:.6f}) ===")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            cfg["train"]["log_interval"],
            grad_clip=cfg["train"].get("grad_clip"),
        )
        print(f"Train loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, save_dir, "best.pt")
                print("  Saved best checkpoint.")

        if epoch % cfg["train"]["save_interval"] == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, save_dir, f"epoch_{epoch}.pt")

    save_checkpoint(model, optimizer, total_epochs, train_loss, save_dir, "last.pt")
    print(f"\nTraining complete. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
