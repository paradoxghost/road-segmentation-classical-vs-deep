"""Inference utilities for the lightweight U-Net baseline."""

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
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep.dataset import CamVidRoadDataset, RAW_ROOT, MASK_ROOT
from src.deep.unet import UNet


MODEL_PATH = PROJECT_ROOT / "results" / "models" / "unet_best.pt"
PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "predictions" / "unet"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "deep"
RANDOM_SEED = 42


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str | Path = MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[UNet, dict]:
    """Load the best U-Net checkpoint."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if device is None:
        device = get_device()

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    base_channels = int(config.get("base_channels", 32))

    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def predict_split(
    split: str = "val",
    model_path: str | Path = MODEL_PATH,
    output_root: str | Path = PREDICTIONS_ROOT,
    image_size: int | None = None,
    batch_size: int = 8,
    threshold: float = 0.5,
    num_workers: int = 0,
) -> list[Path]:
    """Generate binary U-Net predictions for one split."""
    device = get_device()
    model, config = load_model(model_path=model_path, device=device)

    if image_size is None:
        image_size = int(config.get("image_size", 256))

    dataset = CamVidRoadDataset(split=split, image_size=image_size, augment=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = Path(output_root) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            filenames = batch["filename"]

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).cpu().numpy().astype(np.uint8) * 255

            for prediction, filename in zip(predictions, filenames):
                prediction_mask = prediction[0]
                gt_path = MASK_ROOT / split / filename

                with Image.open(gt_path) as gt_mask:
                    original_size = gt_mask.size

                prediction_image = Image.fromarray(prediction_mask).resize(
                    original_size,
                    resample=Image.NEAREST,
                )

                output_path = output_dir / filename
                prediction_image.save(output_path)
                saved_paths.append(output_path)

    return saved_paths


def make_error_map(prediction_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create an RGB error map: green TP, red FP, blue FN, black TN."""
    pred = prediction_mask > 127
    gt = gt_mask > 127

    error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
    error_map[pred & gt] = (0, 180, 0)
    error_map[pred & ~gt] = (255, 0, 0)
    error_map[~pred & gt] = (0, 80, 255)

    return error_map


def save_prediction_figures(
    split: str = "val",
    predictions_root: str | Path = PREDICTIONS_ROOT,
    figures_root: str | Path = FIGURES_ROOT,
    num_figures: int = 10,
) -> list[Path]:
    """Save input, ground truth, prediction, and error-map figures."""
    image_dir = RAW_ROOT / split
    gt_dir = MASK_ROOT / split
    pred_dir = Path(predictions_root) / split
    output_dir = Path(figures_root) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = sorted(pred_dir.glob("*.png"))
    if not prediction_paths:
        raise ValueError(f"No U-Net prediction masks found in: {pred_dir}")

    rng = random.Random(RANDOM_SEED)
    selected_predictions = rng.sample(prediction_paths, min(num_figures, len(prediction_paths)))

    saved_paths = []
    for index, pred_path in enumerate(selected_predictions, start=1):
        filename = pred_path.name
        image_path = image_dir / filename
        gt_path = gt_dir / filename

        input_image = Image.open(image_path).convert("RGB")
        gt_mask = np.array(Image.open(gt_path).convert("L"))
        prediction_mask = np.array(Image.open(pred_path).convert("L"))
        error_map = make_error_map(prediction_mask, gt_mask)

        panels = [
            (input_image, "Input image", None),
            (gt_mask, "Ground truth", "gray"),
            (prediction_mask, "U-Net prediction", "gray"),
            (error_map, "Error map", None),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, (panel, title, cmap) in zip(axes, panels):
            if cmap == "gray":
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(filename)
        fig.tight_layout()

        output_path = output_dir / f"unet_comparison_{index:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate U-Net predictions for one split.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    saved_predictions = predict_split(
        split=args.split,
        model_path=args.model_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
    )
    saved_figures = save_prediction_figures(split=args.split)

    print(f"Saved {len(saved_predictions)} U-Net predictions to {PREDICTIONS_ROOT / args.split}")
    print(f"Saved {len(saved_figures)} U-Net figures to {FIGURES_ROOT / args.split}")


if __name__ == "__main__":
    main()
