"""Training utilities for the lightweight U-Net baseline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep.dataset import CamVidRoadDataset
from src.deep.losses import bce_dice_loss
from src.deep.unet import UNet
from src.metrics import confusion_values, metrics_from_confusion


MODEL_PATH = PROJECT_ROOT / "results" / "models" / "unet_best.pt"
HISTORY_PATH = PROJECT_ROOT / "results" / "metrics" / "deep" / "unet_history.json"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = CamVidRoadDataset(split="train", image_size=image_size, augment=True)
    val_dataset = CamVidRoadDataset(split="val", image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Validate for one epoch and return loss plus segmentation metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            logits = model(images)
            loss = bce_dice_loss(logits, masks)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).cpu().numpy().astype(np.uint8)
            targets = (masks >= 0.5).cpu().numpy().astype(np.uint8)

            for pred_mask, target_mask in zip(predictions, targets):
                confusion = confusion_values(
                    pred_mask[0],
                    target_mask[0],
                    threshold=0.5,
                )
                for key in total_confusion:
                    total_confusion[key] += confusion[key]

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    metrics = metrics_from_confusion(total_confusion)
    metrics["val_loss"] = total_loss / max(total_samples, 1)

    return metrics


def save_history(history: list[dict], output_path: str | Path = HISTORY_PATH) -> None:
    """Save training history to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    output_path: str | Path = MODEL_PATH,
) -> None:
    """Save the best model checkpoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, output_path)


def train_unet(
    epochs: int = 20,
    batch_size: int = 8,
    image_size: int = 256,
    learning_rate: float = 1e-3,
    base_channels: int = 32,
    patience: int = 5,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[torch.nn.Module, list[dict]]:
    """Train U-Net and save the best model based on validation Dice."""
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = UNet(base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "base_channels": base_channels,
        "patience": patience,
        "seed": seed,
    }

    history = []
    best_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(epoch_record)
        save_history(history)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
            print(f"Saved new best model to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without validation Dice improvement.")
            break

    save_history(history)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight U-Net baseline.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    train_unet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        patience=args.patience,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
