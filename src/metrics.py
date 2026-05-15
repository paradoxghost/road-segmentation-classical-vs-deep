"""Reusable binary segmentation metrics for road segmentation.

The project uses binary masks where:

- road = 255
- non-road = 0

These helpers also accept masks stored as 0/1 arrays or probability maps.
"""

from pathlib import Path
import json

import numpy as np
from PIL import Image


EPSILON = 1e-7
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_mask(mask: str | Path | np.ndarray) -> np.ndarray:
    """Load a mask from a path or return an existing NumPy array."""
    if isinstance(mask, np.ndarray):
        return mask

    mask_path = Path(mask)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    return np.array(Image.open(mask_path))


def mask_to_bool(mask: str | Path | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a mask to a boolean road/non-road mask.

    Supports:
    - binary masks with values 0 and 255
    - binary masks with values 0 and 1
    - probability maps with values between 0 and 1

    If values are larger than 1, the threshold is interpreted on a 0-255 scale.
    For example, threshold=0.5 becomes 127.5 for 0/255 masks.
    """
    mask_array = load_mask(mask)

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    mask_array = mask_array.astype(np.float32)
    threshold_value = threshold * 255.0 if mask_array.max(initial=0) > 1.0 else threshold

    return mask_array > threshold_value


def confusion_values(
    pred_mask: str | Path | np.ndarray,
    gt_mask: str | Path | np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute TP, FP, TN, and FN for one prediction/ground-truth pair."""
    pred = mask_to_bool(pred_mask, threshold=threshold)
    gt = mask_to_bool(gt_mask, threshold=threshold)

    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: prediction={pred.shape}, ground_truth={gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def pixel_accuracy(tp: int, fp: int, tn: int, fn: int, eps: float = EPSILON) -> float:
    """Compute pixel accuracy."""
    return float((tp + tn) / (tp + fp + tn + fn + eps))


def precision_score(tp: int, fp: int, eps: float = EPSILON) -> float:
    """Compute precision: among predicted road pixels, how many are road."""
    return float(tp / (tp + fp + eps))


def recall_score(tp: int, fn: int, eps: float = EPSILON) -> float:
    """Compute recall: among true road pixels, how many were found."""
    return float(tp / (tp + fn + eps))


def iou_score(tp: int, fp: int, fn: int, eps: float = EPSILON) -> float:
    """Compute IoU, also called the Jaccard score."""
    return float(tp / (tp + fp + fn + eps))


def dice_score(tp: int, fp: int, fn: int, eps: float = EPSILON) -> float:
    """Compute Dice score, also called the segmentation F1 score."""
    return float((2 * tp) / (2 * tp + fp + fn + eps))


def metrics_from_confusion(confusion: dict, eps: float = EPSILON) -> dict:
    """Compute all metrics from TP, FP, TN, and FN values."""
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    tn = int(confusion["tn"])
    fn = int(confusion["fn"])

    return {
        "pixel_accuracy": pixel_accuracy(tp, fp, tn, fn, eps=eps),
        "iou": iou_score(tp, fp, fn, eps=eps),
        "dice": dice_score(tp, fp, fn, eps=eps),
        "precision": precision_score(tp, fp, eps=eps),
        "recall": recall_score(tp, fn, eps=eps),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_mask_pair(
    pred_mask: str | Path | np.ndarray,
    gt_mask: str | Path | np.ndarray,
    threshold: float = 0.5,
    eps: float = EPSILON,
) -> dict:
    """Evaluate one predicted mask against one ground-truth mask."""
    confusion = confusion_values(pred_mask, gt_mask, threshold=threshold)
    return metrics_from_confusion(confusion, eps=eps)


def list_mask_files(directory: str | Path) -> list[Path]:
    """List mask image files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_folder(
    pred_dir: str | Path,
    gt_dir: str | Path,
    threshold: float = 0.5,
    eps: float = EPSILON,
    include_per_image: bool = True,
) -> dict:
    """Evaluate a folder of predicted masks against ground-truth masks.

    Prediction filenames are expected to match the ground-truth filenames.
    Metrics are computed globally by summing TP, FP, TN, and FN over all images.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    gt_paths = list_mask_files(gt_dir)
    if not gt_paths:
        raise ValueError(f"No ground-truth masks found in: {gt_dir}")

    total_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    per_image = []

    for gt_path in gt_paths:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predicted mask for {gt_path.name}: {pred_path}")

        image_metrics = evaluate_mask_pair(pred_path, gt_path, threshold=threshold, eps=eps)

        for key in total_confusion:
            total_confusion[key] += image_metrics[key]

        if include_per_image:
            per_image.append({"filename": gt_path.name, **image_metrics})

    results = metrics_from_confusion(total_confusion, eps=eps)
    results["num_images"] = len(gt_paths)

    if include_per_image:
        results["per_image"] = per_image

    return results


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    """Save metrics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = EPSILON) -> dict:
    """Backward-compatible wrapper for earlier code."""
    return evaluate_mask_pair(pred_mask, gt_mask, eps=eps)


def _run_dummy_test() -> None:
    """Run a small self-test when this file is executed directly."""
    gt_mask = np.array(
        [
            [255, 255, 0, 0],
            [255, 0, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 0],
        ],
        dtype=np.uint8,
    )

    pred_mask = np.array(
        [
            [255, 0, 0, 0],
            [255, 255, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    metrics = evaluate_mask_pair(pred_mask, gt_mask)

    expected_confusion = {"tp": 4, "fp": 2, "tn": 8, "fn": 2}
    for key, expected_value in expected_confusion.items():
        if metrics[key] != expected_value:
            raise AssertionError(f"Expected {key}={expected_value}, got {metrics[key]}")

    probability_mask = pred_mask.astype(np.float32) / 255.0
    probability_metrics = evaluate_mask_pair(probability_mask, gt_mask)
    for key in expected_confusion:
        if probability_metrics[key] != expected_confusion[key]:
            raise AssertionError(f"Probability-mask test failed for {key}")

    print("Dummy segmentation metrics test")
    print("-------------------------------")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\nAll metric functions passed the dummy test.")


if __name__ == "__main__":
    _run_dummy_test()
