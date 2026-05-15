"""Generate best/worst U-Net and classical-vs-U-Net failure-case figures."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from metrics import evaluate_mask_pair
from deep.predict import make_error_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
GT_ROOT = PROJECT_ROOT / "data" / "processed" / "binary_masks"
UNET_PRED_ROOT = PROJECT_ROOT / "results" / "predictions" / "unet"
CLASSICAL_PRED_ROOT = PROJECT_ROOT / "results" / "predictions" / "kmeans_morph"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "figures" / "failure_cases"


def load_metrics_json(path: Path) -> dict | None:
    """Load metrics JSON if it exists."""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_per_image_metrics(split: str) -> list[dict]:
    """Read or recompute U-Net per-image metrics for a split."""
    metrics_path = METRICS_ROOT / "deep" / f"unet_{split}.json"
    metrics = load_metrics_json(metrics_path)

    if metrics and "per_image" in metrics:
        return metrics["per_image"]

    pred_dir = UNET_PRED_ROOT / split
    gt_dir = GT_ROOT / split
    prediction_paths = sorted(pred_dir.glob("*.png"))

    if not prediction_paths:
        raise FileNotFoundError(
            f"No U-Net predictions found in {pred_dir}. "
            f"Run src/deep/evaluate_unet.py --split {split} first."
        )

    per_image = []
    for pred_path in prediction_paths:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground-truth mask: {gt_path}")

        image_metrics = evaluate_mask_pair(pred_path, gt_path)
        per_image.append({"filename": pred_path.name, **image_metrics})

    return per_image


def load_rgb(path: Path) -> Image.Image:
    """Load an RGB image."""
    return Image.open(path).convert("RGB")


def load_gray(path: Path) -> np.ndarray:
    """Load a grayscale mask."""
    return np.array(Image.open(path).convert("L"))


def save_unet_case_figure(split: str, filename: str, output_path: Path, title: str) -> None:
    """Save a four-panel U-Net case figure."""
    image = load_rgb(RAW_ROOT / split / filename)
    gt = load_gray(GT_ROOT / split / filename)
    unet = load_gray(UNET_PRED_ROOT / split / filename)
    error = make_error_map(unet, gt)

    panels = [
        (image, "Input image", None),
        (gt, "Ground truth", "gray"),
        (unet, "U-Net prediction", "gray"),
        (error, "U-Net error map", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (panel, panel_title, cmap) in zip(axes, panels):
        if cmap == "gray":
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_classical_vs_unet_figure(split: str, filename: str, output_path: Path, title: str) -> None:
    """Save a five-panel comparison figure using K-means morphology as classical."""
    image = load_rgb(RAW_ROOT / split / filename)
    gt = load_gray(GT_ROOT / split / filename)
    classical = load_gray(CLASSICAL_PRED_ROOT / split / filename)
    unet = load_gray(UNET_PRED_ROOT / split / filename)
    error = make_error_map(unet, gt)

    panels = [
        (image, "Input image", None),
        (gt, "Ground truth", "gray"),
        (classical, "K-means morph", "gray"),
        (unet, "U-Net prediction", "gray"),
        (error, "U-Net error map", None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    for ax, (panel, panel_title, cmap) in zip(axes, panels):
        if cmap == "gray":
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure-case figures.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-cases", type=int, default=5)
    args = parser.parse_args()

    per_image = get_per_image_metrics(args.split)
    sorted_by_iou = sorted(per_image, key=lambda item: item["iou"])

    worst_cases = sorted_by_iou[: args.num_cases]
    best_cases = list(reversed(sorted_by_iou[-args.num_cases :]))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(worst_cases, start=1):
        filename = case["filename"]
        save_unet_case_figure(
            split=args.split,
            filename=filename,
            output_path=OUTPUT_ROOT / f"worst_unet_{index:02d}.png",
            title=f"Worst U-Net case {index}: {filename} (IoU={case['iou']:.3f})",
        )
        save_classical_vs_unet_figure(
            split=args.split,
            filename=filename,
            output_path=OUTPUT_ROOT / f"classical_vs_unet_failure_{index:02d}.png",
            title=f"Classical vs U-Net failure {index}: {filename} (U-Net IoU={case['iou']:.3f})",
        )

    for index, case in enumerate(best_cases, start=1):
        filename = case["filename"]
        save_unet_case_figure(
            split=args.split,
            filename=filename,
            output_path=OUTPUT_ROOT / f"best_unet_{index:02d}.png",
            title=f"Best U-Net case {index}: {filename} (IoU={case['iou']:.3f})",
        )

    print(f"Generated {len(worst_cases)} worst-case U-Net figures.")
    print(f"Generated {len(best_cases)} best-case U-Net figures.")
    print(f"Generated {len(worst_cases)} classical-vs-U-Net failure figures.")
    print(f"Saved failure-case figures to {OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
