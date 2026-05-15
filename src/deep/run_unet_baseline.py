"""Train, predict, and evaluate the lightweight U-Net baseline."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep.predict import PREDICTIONS_ROOT, predict_split, save_prediction_figures
from src.deep.train import HISTORY_PATH, MODEL_PATH, train_unet
from src.metrics import evaluate_folder, save_metrics


GT_ROOT = PROJECT_ROOT / "data" / "processed" / "binary_masks"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics" / "deep"


def print_metrics_table(metrics: dict) -> None:
    """Print final validation metrics."""
    print()
    print(f"{'Method':<10} {'Pixel Acc':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 58)
    print(
        f"{'U-Net':<10} "
        f"{metrics['pixel_accuracy']:>10.4f} "
        f"{metrics['iou']:>8.4f} "
        f"{metrics['dice']:>8.4f} "
        f"{metrics['precision']:>10.4f} "
        f"{metrics['recall']:>8.4f}"
    )


def run_unet_baseline(
    epochs: int = 20,
    batch_size: int = 8,
    image_size: int = 256,
    learning_rate: float = 1e-3,
    base_channels: int = 32,
    patience: int = 5,
    num_workers: int = 0,
    threshold: float = 0.5,
    num_figures: int = 10,
) -> dict:
    """Train U-Net and evaluate validation predictions."""
    train_unet(
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        base_channels=base_channels,
        patience=patience,
        num_workers=num_workers,
    )

    saved_predictions = predict_split(
        split="val",
        model_path=MODEL_PATH,
        image_size=image_size,
        batch_size=batch_size,
        threshold=threshold,
        num_workers=num_workers,
    )

    metrics = evaluate_folder(
        pred_dir=PREDICTIONS_ROOT / "val",
        gt_dir=GT_ROOT / "val",
        include_per_image=True,
    )
    metrics["method"] = "U-Net"
    metrics["split"] = "val"
    metrics["model_path"] = MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()
    metrics["prediction_dir"] = (PREDICTIONS_ROOT / "val").relative_to(PROJECT_ROOT).as_posix()
    metrics["ground_truth_dir"] = (GT_ROOT / "val").relative_to(PROJECT_ROOT).as_posix()

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_ROOT / "unet_val.json"
    save_metrics(metrics, metrics_path)

    saved_figures = save_prediction_figures(split="val", num_figures=num_figures)

    print_metrics_table(metrics)
    print()
    print(f"Saved best model to {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved training history to {HISTORY_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved {len(saved_predictions)} validation predictions to {(PREDICTIONS_ROOT / 'val').relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Saved validation metrics to {metrics_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved {len(saved_figures)} validation figures to {Path('results/figures/deep/val').as_posix()}/")
    print("U-Net baseline complete.")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight U-Net baseline.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-figures", type=int, default=10)
    args = parser.parse_args()

    run_unet_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        patience=args.patience,
        num_workers=args.num_workers,
        threshold=args.threshold,
        num_figures=args.num_figures,
    )


if __name__ == "__main__":
    main()
