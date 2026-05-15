"""Evaluate a trained U-Net checkpoint without retraining."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep.predict import PREDICTIONS_ROOT, predict_split, save_prediction_figures
from src.metrics import evaluate_folder, save_metrics


GT_ROOT = PROJECT_ROOT / "data" / "processed" / "binary_masks"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "models" / "unet_best.pt"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics" / "deep"


def print_metrics_table(split: str, metrics: dict) -> None:
    """Print a compact U-Net metrics table."""
    print()
    print(f"U-NET {split.upper()} RESULTS")
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


def evaluate_unet(
    split: str = "test",
    image_size: int = 256,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    batch_size: int = 8,
    threshold: float = 0.5,
    num_workers: int = 0,
    num_figures: int = 10,
) -> dict:
    """Generate predictions, evaluate metrics, and save figures for one split."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(f"U-Net checkpoint not found: {checkpoint}")

    saved_predictions = predict_split(
        split=split,
        model_path=checkpoint,
        image_size=image_size,
        batch_size=batch_size,
        threshold=threshold,
        num_workers=num_workers,
    )

    pred_dir = PREDICTIONS_ROOT / split
    gt_dir = GT_ROOT / split
    metrics = evaluate_folder(pred_dir=pred_dir, gt_dir=gt_dir, include_per_image=True)
    metrics["method"] = "U-Net"
    metrics["split"] = split
    metrics["checkpoint"] = checkpoint.relative_to(PROJECT_ROOT).as_posix()
    metrics["prediction_dir"] = pred_dir.relative_to(PROJECT_ROOT).as_posix()
    metrics["ground_truth_dir"] = gt_dir.relative_to(PROJECT_ROOT).as_posix()

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_ROOT / f"unet_{split}.json"
    save_metrics(metrics, metrics_path)

    saved_figures = save_prediction_figures(split=split, num_figures=num_figures)

    print_metrics_table(split, metrics)
    print()
    print(f"Loaded checkpoint: {checkpoint.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved {len(saved_predictions)} predictions to {pred_dir.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Saved metrics to {metrics_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved {len(saved_figures)} figures to {Path('results/figures/deep') / split}/")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-figures", type=int, default=10)
    args = parser.parse_args()

    evaluate_unet(
        split=args.split,
        image_size=args.image_size,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
        num_figures=args.num_figures,
    )


if __name__ == "__main__":
    main()
