"""Otsu-thresholding baseline for binary road segmentation."""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image

try:
    from .postprocess import postprocess_mask, to_uint8_mask
except ImportError:
    from postprocess import postprocess_mask, to_uint8_mask


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "predictions" / "otsu"


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image as RGB."""
    return np.array(Image.open(image_path).convert("RGB"))


def lower_image_prior(height: int, width: int, start_fraction: float = 0.30) -> np.ndarray:
    """Create a mask favoring the lower image area where roads usually appear."""
    y_coordinates = np.arange(height, dtype=np.float32)[:, None]
    y_normalized = y_coordinates / max(height - 1, 1)
    return np.repeat(y_normalized >= start_fraction, width, axis=1)


def score_road_like_mask(mask: np.ndarray) -> float:
    """Score whether a binary mask looks like a road in a driving image."""
    binary = mask > 0
    height, width = binary.shape

    lower_half = binary[height // 2 :, :]
    upper_half = binary[: height // 2, :]
    bottom_center = binary[int(height * 0.70) :, int(width * 0.30) : int(width * 0.70)]

    lower_fraction = float(lower_half.mean())
    upper_fraction = float(upper_half.mean()) if upper_half.size else 0.0
    bottom_center_fraction = float(bottom_center.mean()) if bottom_center.size else 0.0

    # A road-like Otsu result should cover the bottom center more than the sky/top.
    return (2.0 * bottom_center_fraction) + lower_fraction - (1.5 * upper_fraction)


def choose_otsu_direction(binary: np.ndarray) -> np.ndarray:
    """Choose normal or inverted Otsu output using a simple lower-image heuristic."""
    inverted = 255 - binary
    return binary if score_road_like_mask(binary) >= score_road_like_mask(inverted) else inverted


def predict_otsu_mask(
    image_rgb: np.ndarray,
    use_lower_prior: bool = True,
    apply_postprocess: bool = True,
) -> np.ndarray:
    """Predict a road mask using Otsu thresholding on the saturation channel.

    Road surfaces are often less saturated than many scene objects. Otsu gives
    an automatic threshold, then a heuristic chooses whether low or high values
    should be treated as road.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]

    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = choose_otsu_direction(binary)

    if use_lower_prior:
        height, width = mask.shape
        mask = np.where(lower_image_prior(height, width, start_fraction=0.30), mask, 0)

    if apply_postprocess:
        mask = postprocess_mask(mask, min_area=700)

    return to_uint8_mask(mask)


def list_images(image_dir: Path) -> list[Path]:
    """List input image files."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(image_path: str | Path, output_path: str | Path) -> Path:
    """Generate and save one Otsu baseline prediction."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = load_rgb_image(image_path)
    prediction = predict_otsu_mask(image_rgb)

    Image.fromarray(prediction).save(output_path)
    return output_path


def process_split(
    split: str,
    image_root: str | Path = RAW_ROOT,
    output_root: str | Path = PREDICTIONS_ROOT,
) -> list[Path]:
    """Run Otsu thresholding on every image in one split."""
    image_dir = Path(image_root) / split
    output_dir = Path(output_root) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for image_path in list_images(image_dir):
        output_path = output_dir / image_path.with_suffix(".png").name
        saved_paths.append(process_image(image_path, output_path))

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Otsu road-segmentation baseline.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    saved_paths = process_split(args.split)
    print(f"Saved {len(saved_paths)} Otsu predictions to {PREDICTIONS_ROOT / args.split}")


if __name__ == "__main__":
    main()
