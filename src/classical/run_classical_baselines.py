"""Run and evaluate all classical road-segmentation baselines."""

from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classical import hsv_baseline, kmeans_baseline, otsu_baseline
from src.metrics import evaluate_folder, save_metrics


RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
GT_ROOT = PROJECT_ROOT / "data" / "processed" / "binary_masks"
PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "predictions"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics" / "classical"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "classical"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
RANDOM_SEED = 42


def list_images(image_dir: Path) -> list[Path]:
    """List input image files."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_predictions(split: str, kmeans_k: int = 5, kmeans_max_side: int = 320) -> dict[str, Path]:
    """Run all classical baselines and return their prediction directories."""
    print(f"Running HSV baseline on {split}...")
    hsv_baseline.process_split(
        split=split,
        image_root=RAW_ROOT,
        output_root=PREDICTIONS_ROOT / "hsv",
    )

    print(f"Running Otsu baseline on {split}...")
    otsu_baseline.process_split(
        split=split,
        image_root=RAW_ROOT,
        output_root=PREDICTIONS_ROOT / "otsu",
    )

    print(f"Running K-means baseline on {split}...")
    kmeans_baseline.process_split(
        split=split,
        image_root=RAW_ROOT,
        raw_output_root=PREDICTIONS_ROOT / "kmeans_raw",
        morph_output_root=PREDICTIONS_ROOT / "kmeans_morph",
        canonical_output_root=PREDICTIONS_ROOT / "kmeans",
        k=kmeans_k,
        max_side=kmeans_max_side,
    )

    return {
        "HSV": PREDICTIONS_ROOT / "hsv" / split,
        "Otsu": PREDICTIONS_ROOT / "otsu" / split,
        "KMeans Raw": PREDICTIONS_ROOT / "kmeans_raw" / split,
        "KMeans Morph": PREDICTIONS_ROOT / "kmeans_morph" / split,
    }


def evaluate_methods(split: str, prediction_dirs: dict[str, Path]) -> dict[str, dict]:
    """Evaluate every method and save one metrics JSON file per method."""
    gt_dir = GT_ROOT / split
    method_to_file_stem = {
        "HSV": "hsv",
        "Otsu": "otsu",
        "KMeans Raw": "kmeans_raw",
        "KMeans Morph": "kmeans_morph",
    }

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for method_name, pred_dir in prediction_dirs.items():
        metrics = evaluate_folder(pred_dir=pred_dir, gt_dir=gt_dir, include_per_image=True)
        metrics["method"] = method_name
        metrics["split"] = split
        metrics["prediction_dir"] = pred_dir.relative_to(PROJECT_ROOT).as_posix()
        metrics["ground_truth_dir"] = gt_dir.relative_to(PROJECT_ROOT).as_posix()

        output_path = METRICS_ROOT / f"{method_to_file_stem[method_name]}_{split}.json"
        save_metrics(metrics, output_path)
        all_metrics[method_name] = metrics

    return all_metrics


def print_metrics_table(all_metrics: dict[str, dict]) -> None:
    """Print a compact terminal table for report-friendly comparison."""
    print()
    print(f"{'Method':<15} {'Pixel Acc':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 64)

    for method_name, metrics in all_metrics.items():
        print(
            f"{method_name:<15} "
            f"{metrics['pixel_accuracy']:>10.4f} "
            f"{metrics['iou']:>8.4f} "
            f"{metrics['dice']:>8.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>8.4f}"
        )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary mask for display."""
    return np.array(Image.open(mask_path).convert("L"))


def save_visual_comparisons(
    split: str,
    prediction_dirs: dict[str, Path],
    num_figures: int = 10,
) -> list[Path]:
    """Save side-by-side comparison figures for the selected split."""
    image_dir = RAW_ROOT / split
    gt_dir = GT_ROOT / split
    output_dir = FIGURES_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise ValueError(f"No images found for split: {split}")

    rng = random.Random(RANDOM_SEED)
    selected_images = rng.sample(images, min(num_figures, len(images)))

    saved_paths = []
    for index, image_path in enumerate(selected_images, start=1):
        filename = image_path.with_suffix(".png").name
        input_image = Image.open(image_path).convert("RGB")
        gt_mask = load_mask(gt_dir / filename)
        hsv_mask = load_mask(prediction_dirs["HSV"] / filename)
        otsu_mask = load_mask(prediction_dirs["Otsu"] / filename)
        kmeans_raw_mask = load_mask(prediction_dirs["KMeans Raw"] / filename)
        kmeans_morph_mask = load_mask(prediction_dirs["KMeans Morph"] / filename)

        panels = [
            (input_image, "Input image", None),
            (gt_mask, "Ground truth", "gray"),
            (hsv_mask, "HSV", "gray"),
            (otsu_mask, "Otsu", "gray"),
            (kmeans_raw_mask, "K-means raw", "gray"),
            (kmeans_morph_mask, "K-means morph", "gray"),
        ]

        fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
        for ax, (panel, title, cmap) in zip(axes, panels):
            if cmap == "gray":
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(filename)
        fig.tight_layout()

        output_path = output_dir / f"classical_comparison_{index:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and evaluate classical segmentation baselines.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-figures", type=int, default=10)
    parser.add_argument("--kmeans-k", type=int, default=5)
    parser.add_argument("--kmeans-max-side", type=int, default=320)
    args = parser.parse_args()

    prediction_dirs = run_predictions(
        split=args.split,
        kmeans_k=args.kmeans_k,
        kmeans_max_side=args.kmeans_max_side,
    )
    all_metrics = evaluate_methods(args.split, prediction_dirs)
    print_metrics_table(all_metrics)

    saved_figures = save_visual_comparisons(
        split=args.split,
        prediction_dirs=prediction_dirs,
        num_figures=args.num_figures,
    )

    print()
    print(f"Saved metrics JSON files to {METRICS_ROOT.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Saved {len(saved_figures)} comparison figures to {(FIGURES_ROOT / args.split).relative_to(PROJECT_ROOT).as_posix()}/")
    print("Classical baseline evaluation complete.")


if __name__ == "__main__":
    main()
